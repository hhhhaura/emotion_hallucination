# TODO: greedy stop with input eos_token_id and for HF and LocalVLLM
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from functools import lru_cache
import httpx
import openai
import requests
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
from typing import Any, get_args, Literal, overload, Union, TypedDict, TypeAlias
from vllm import LLM, SamplingParams, PromptType, TokensPrompt
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from .cache_lib import hashable_cache

# vLLM (and OpenAI-style) cap on completion `logprobs`; tune to match `--max-logprobs` on your server.
BASE_VLLM_MAX_LOGPROBS = 20

QED = []
# QED = ["\\qed", " \\qed", "$\\qed$", " $\\qed$"]

FINISH = ["[FINISH]", " [FINISH]"]

THINK = ["</think>"]

EOS = ["<|im_end|>", "<|endoftext|>"]

ROLE = [
    f"{prefix}{role}{suffix}"
    for prefix in [" "]
    for role in ["user", "User", "assistant", "Assistant", "human", "Human"]
    for suffix in [":"]
]


def parse_output_string(outputs: list[str]) -> list[dict]:
    parsed_outputs = [{"role": "assistant"} for _ in range(len(outputs))]
    for i, out in enumerate(outputs):
        if "<think>" in out and "</think>" in out:
            out2 = out.replace("<think>", "").strip()
            reasoning, content = map(lambda c: c.strip(), out2.split("</think>", 1))
            parsed_outputs[i]["reasoning_content"] = reasoning
            parsed_outputs[i]["content"] = content
        else:
            parsed_outputs[i]["content"] = out
    return parsed_outputs


def compose_parsed_output(parsed_outputs: list[dict]) -> list[str]:
    outputs = ["" for _ in range(len(parsed_outputs))]
    for i, msg in enumerate(parsed_outputs):
        if "reasoning_content" in msg:
            outputs[i] += "<think>\n" + msg["reasoning_content"]
            if "content" in msg:
                outputs[i] += "\n</think>\n\n" + msg["content"]
        else:
            outputs[i] = msg["content"]
    return outputs


class BaseHFLM:
    encode_cache: dict[str, list[int]] = {}

    @overload
    def __init__(self, *, local_model_path: str, device: str) -> None: ...

    @overload
    def __init__(self, *, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None: ...

    def __init__(
        self,
        *,
        local_model_path: str = None,
        device: str = None,
        model: PreTrainedModel = None,
        tokenizer: PreTrainedTokenizerBase = None,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        stop_at_thinking: bool = False,
        **kwargs: Any,
    ):
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.device = model.device
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                dtype=torch.float16,
            ).to(device)
            self.model.eval()
            self.device = device

        self.generate_kwargs = {
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 50,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "stopping_criteria": [transformers.StopStringCriteria(self.tokenizer, QED + FINISH + EOS + ROLE + (THINK if stop_at_thinking else []))],
            # "repetition_penalty": 1.005,
            # "renormalize_logits": True
            **kwargs,
        }

    def encode(self, txt: str) -> list[int]:
        if txt in self.encode_cache:
            return self.encode_cache[txt]
        else:
            return self.tokenizer.encode(txt, add_special_tokens=False)

    def encode_string(
        self,
        txt: str | list[str],
        *,
        padding_side: Literal["left", "right"] = "left",
    ) -> dict[str, torch.Tensor]:
        txt = [txt] if isinstance(txt, str) else txt

        # encoded = self.tokenizer(
        #     txt,
        #     padding=True,
        #     padding_side=padding_side,
        #     return_tensors="pt",
        # )
        # encoded["input_ids"] = encoded["input_ids"].to(self.device)
        # encoded["attention_mask"] = encoded["attention_mask"].to(self.device)

        input_ids = [torch.tensor(self.encode(t)) for t in txt]
        encoded = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
                padding_side=padding_side
            ).to(self.device)
        }
        if padding_side == "left":
            encoded["attention_mask"] = torch.tensor(
                [
                    [0] * (encoded["input_ids"].shape[1] - len(input_ids[i])) + [1] * len(input_ids[i])
                    for i in range(len(input_ids))
                ], dtype=torch.long, device=self.device
            )
        else:
            encoded["attention_mask"] = torch.tensor(
                [
                    [1] * len(input_ids[i]) + [0] * (encoded["input_ids"].shape[1] - len(input_ids[i]))
                    for i in range(len(input_ids))
                ], dtype=torch.long, device=self.device
            )

        return encoded

    def decode(
        self,
        token_ids: int | list[int] | torch.Tensor,
        *,
        skip_special_tokens: bool = False
    ) -> str:
        decoded_string = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        if not skip_special_tokens:
            # For skip_special_tokens = False, we need to remove the right pad_token but keep the endding eos_token.
            # In case of pad_token == eos_token, we strip both and manually append the eos_token afterwards.
            if decoded_string.endswith(self.tokenizer.pad_token):
                decoded_string = decoded_string.rstrip(self.tokenizer.pad_token).rstrip(self.tokenizer.eos_token) + self.tokenizer.eos_token
        return decoded_string

    def batch_decode(
        self,
        sequences: list[int] | list[list[int]] | torch.Tensor,
        *,
        skip_special_tokens: bool = False
    ) -> list[str]:
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
            )
            for seq in sequences
        ]

    def generate(self, inputs: dict[str, torch.Tensor], **kwargs: Any):
        final_kwargs = {**self.generate_kwargs, **kwargs}
        with torch.no_grad():
            model_outputs = self.model.generate(
                **inputs,
                **final_kwargs,
                return_dict_in_generate=True,
            )
        return model_outputs

    def sample_from_encoded_inputs(self, encoded_inputs: dict[str, torch.Tensor], **kwargs: Any) -> list[str]:
        model_outputs = self.generate(encoded_inputs, **kwargs)
        output_string = self.batch_decode(model_outputs.sequences[:, encoded_inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        for i, out in enumerate(output_string):
            self.encode_cache[out] = model_outputs.sequences[i, encoded_inputs["input_ids"].shape[1]:].tolist()
        return output_string

    def sample(
        self,
        txt: str | list[str],
        **kwargs: Any
    ) -> list[str]:
        txt = [txt] if isinstance(txt, str) else txt
        encoded_inputs = self.encode_string(txt)
        return self.sample_from_encoded_inputs(encoded_inputs, **kwargs)

    @hashable_cache(lru_cache(maxsize=2))
    def log_prob(
        self,
        txt: str | list[str],
        outputs: list[str],
        *,
        reduction: bool = True,
    ) -> torch.Tensor:
        txt = [txt] if isinstance(txt, str) else txt
        encoded_inputs = self.encode_string(txt, padding_side="left") if txt is not None else None
        encoded_outputs = self.encode_string(outputs, padding_side="right")

        if encoded_inputs is not None:
            encoded_inputs_outputs = {
                "input_ids": torch.cat((encoded_inputs["input_ids"], encoded_outputs["input_ids"]), dim=-1).to(self.device),
                "attention_mask": torch.cat((encoded_inputs["attention_mask"], encoded_outputs["attention_mask"]), dim=-1).to(self.device),
            }
        else:
            encoded_inputs_outputs = {
                "input_ids": encoded_outputs["input_ids"].to(self.device),
                "attention_mask": encoded_outputs["attention_mask"].to(self.device),
            }
        out_len = encoded_outputs["input_ids"].shape[1]

        with torch.no_grad():
            try:
                model_outputs = self.model(
                    **encoded_inputs_outputs,
                    logits_to_keep=out_len + 1,
                )
            except TypeError:
                model_outputs = self.model(**encoded_inputs_outputs)

        # Align logits to output tokens.
        # If logits_to_keep is honored, logits should already correspond to (output_len + 1) tokens.
        # Otherwise, slice the final (output_len + 1) window.
        output_logits = model_outputs.logits.clone()
        if output_logits.shape[1] != out_len + 1:
            output_logits = output_logits[:, -(out_len + 1) :, :]

        # NOTE: Infer log_prob offline
        #   Constrast to obtain log_prob during generation, inferring prompt log prob should take logit
        #   processor into account. However, it's inapplicable for vllm infered log_prob since we cannot
        #   obtain full logit.
        # from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
        # in_len = encoded_inputs["input_ids"].shape[1]
        # gen_config, _ = self.model._prepare_generation_config(None, None, **{**self.generate_kwargs, **kwargs})
        # logit_processors = LogitsProcessorList([
        #     TemperatureLogitsWarper(gen_config.temperature),
        #     TopPLogitsWarper(gen_config.top_p, min_tokens_to_keep=1),
        #     TopKLogitsWarper(gen_config.top_k, min_tokens_to_keep=1)
        # ])
        # for t in range(out_len):
        #     output_logits[:, t, :] = logit_processors(encoded_inputs_outputs['input_ids'][:, :in_len + t], output_logits[:, t, :])
        # logp = F.log_softmax(output_logits, dim=-1)

        logp = F.log_softmax(output_logits / self.generate_kwargs["temperature"], dim=-1)
        token_logp = torch.gather(logp[:, :-1, :], -1, encoded_outputs["input_ids"][:, int(txt is None):, None]).squeeze(-1).float()
        masked = torch.where(encoded_outputs["attention_mask"][:, int(txt is None):] == 1, token_logp, 0)
        return masked.sum(dim=-1) if reduction else masked


class ChatHFLM(BaseHFLM):
    def __init__(
        self,
        *,
        local_model_path: str = None,
        device: str = None,
        model: PreTrainedModel = None,
        tokenizer: PreTrainedTokenizerBase = None,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        enable_thinking: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            local_model_path=local_model_path,
            device=device, model=model, tokenizer=tokenizer,
            temperature=temperature, max_new_tokens=max_new_tokens, **kwargs
        )

        if enable_thinking:
            self.enable_think()
        else:
            self.disable_think()

    def enable_think(self):
        self.enable_thinking = True

    def disable_think(self):
        self.enable_thinking = False

    def apply_chat_template(
        self,
        messages: list[dict] | list[list[dict]],
        *,
        continue_final_message: bool = False,
        add_generation_prompt: bool = True,
    ) -> list[str]:
        """
        Apply the chat template to one or more conversations.

        Parameters
        ----------
        messages : list[dict] | list[list[dict]]
            Either a single conversation (list of message dictionaries) or a batch of
            conversations where each element is a list of message dictionaries.
        continue_final_message : bool, optional
            When True, the template allows continuation of the last message without
            inserting a generation prompt after it. Default is False.
        add_generation_prompt : bool, optional
            When True, a generation prompt is appended to each formatted conversation to
            indicate where the model should start generating. Set to False for already
            finished conversations. Default is True.

        Returns
        -------
        list[str]
            A list of formatted conversation strings, one for each input conversation.
        """
        messages = [messages] if isinstance(messages[0], dict) else messages

        add_generation_prompt = add_generation_prompt and not continue_final_message
        enable_thinking = self.enable_thinking and add_generation_prompt

        out: list[str] = []
        for msg_list in messages:
            try:
                txt = self.tokenizer.apply_chat_template(
                    msg_list,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                txt = self.tokenizer.apply_chat_template(
                    msg_list,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                )
            # Workaround removing duplicated thinking part when the last message is unfinished reasoning from assistant
            #   In qwen3, incomplete assistant thinking will result in duplicated <think> -- "<think>\n\n</think>\n\n<think>..."
            txt = txt.replace("<think>\n\n</think>\n\n<think>", "<think>")
            out.append(txt)
        return out

    def encode_conversation(
        self,
        messages: list[dict] | list[list[dict]],
        *,
        continue_final_message: bool = False,
        add_generation_prompt: bool = True,
        padding_side: Literal["left", "right"] = "left",
    ) -> dict[str, torch.Tensor]:
        txt = self.apply_chat_template(
            messages,
            continue_final_message=continue_final_message,
            add_generation_prompt=add_generation_prompt,
        )
        return self.encode_string(txt, padding_side=padding_side)

    def parse_output_string(self, outputs: list[str]) -> list[dict]:
        return parse_output_string(outputs)

    def compose_parsed_output(self, parsed_outputs: list[dict]) -> list[str]:
        return compose_parsed_output(parsed_outputs)

    def sample(
        self,
        messages: list[dict] | list[list[dict]],
        *,
        continue_final_message: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        messages = [messages] if isinstance(messages[0], dict) else messages
        encoded_inputs = self.encode_conversation(messages, continue_final_message=continue_final_message)
        return self.sample_from_encoded_inputs(encoded_inputs, **kwargs)

    @hashable_cache(lru_cache(maxsize=2))
    def log_prob(
        self,
        messages: list[dict] | list[list[dict]],
        outputs: list[str],
        *,
        continue_final_message: bool = False,
        reduction: bool = True,
    ) -> torch.Tensor:
        # Issue: Discrepancy between model.generate and model.forward:
        #   https://github.com/huggingface/transformers/issues/24801
        messages = [messages] if isinstance(messages[0], dict) else messages
        encoded_inputs = self.encode_conversation(messages, continue_final_message=continue_final_message)
        encoded_outputs = self.encode_string(outputs, padding_side="right")

        prompt = {
            "input_ids": torch.cat((encoded_inputs["input_ids"], encoded_outputs["input_ids"]), dim=-1).to(self.device),
            "attention_mask": torch.cat((encoded_inputs["attention_mask"], encoded_outputs["attention_mask"]), dim=-1).to(self.device),
        }

        out_len = encoded_outputs["input_ids"].shape[1]
        with torch.no_grad():
            try:
                model_outputs = self.model(
                    **prompt,
                    logits_to_keep=out_len + 1,
                )
            except TypeError:
                model_outputs = self.model(**prompt)
        logp = F.log_softmax(model_outputs.logits / self.generate_kwargs["temperature"], dim=-1)

        # Align logits to output tokens.
        # If logits_to_keep is honored, logits should already correspond to (output_len + 1) tokens.
        # Otherwise, slice the final (output_len + 1) window.
        if logp.shape[1] != out_len + 1:
            logp = logp[:, -(out_len + 1) :, :]

        token_logp = torch.gather(logp[:, :-1, :], -1, encoded_outputs["input_ids"][:, :, None]).squeeze(-1).float()
        masked = torch.where(encoded_outputs["attention_mask"] == 1, token_logp, 0)
        return masked.sum(dim=-1) if reduction else masked


class BaseLocalVLLM:
    encode_cache: dict[str, list[int]] = {}

    @overload
    def __init__(self, *, local_model_path: str, max_model_len: int, gpu_memory_utilization: float, max_num_seqs: int) -> None: ...

    @overload
    def __init__(self, *, model: LLM) -> None: ...

    def __init__(
        self,
        *,
        local_model_path: str = None,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.25,
        max_num_seqs: int = 10,
        model: LLM = None,
        temperature: float = 1.0,
        max_tokens: int = 512,
        seed: int = None,
        stop_at_thinking: bool = False,
        **kwargs: Any,
    ):
        if model is not None:
            self.model = model
        else:
            self.model = LLM(
                model=local_model_path,
                tokenizer=local_model_path,
                trust_remote_code=True,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                tensor_parallel_size=1,  # Force single GPU usage
                enforce_eager=True
            )
        self.tokenizer = self.model.get_tokenizer()

        self.generate_kwargs = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": max_tokens,
            "seed": seed,
            "skip_special_tokens": False,
            "include_stop_str_in_output": True,
            "stop": QED + FINISH + EOS + ROLE + (THINK if stop_at_thinking else []),
            "logprobs": 1,
            # "repetition_penalty": 1.005,
            # "presence_penalty": 1,
            # "frequency_penalty": 0.1,
            **kwargs
        }

    def encode(self, txt: str) -> list[int]:
        if txt in self.encode_cache:
            return self.encode_cache[txt]
        else:
            return self.tokenizer.encode(txt, add_special_tokens=False)

    def encode_string(
        self,
        txt: str | list[str],
        *,
        padding_side: Literal["left", "right"] = "left"  # dummy argument
    ) -> list[PromptType]:
        txt = [txt] if isinstance(txt, str) else txt
        return [TokensPrompt(prompt_token_ids=self.encode(t)) for t in txt]

    def decode(
        self,
        token_ids: int | list[int] | torch.Tensor,
        *,
        skip_special_tokens: bool = False
    ) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self,
        sequences: list[int] | list[list[int]] | torch.Tensor,
        *,
        skip_special_tokens: bool = False
    ) -> list[str]:
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
            )
            for seq in sequences
        ]

    def generate(self, inputs: PromptType | Sequence[PromptType], **kwargs: Any):
        if "max_new_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
        final_kwargs = {**self.generate_kwargs, **kwargs}
        model_outputs = self.model.generate(
            inputs,
            sampling_params=SamplingParams(**final_kwargs),
            use_tqdm=False
        )
        return model_outputs

    def sample_from_encoded_inputs(self, encoded_inputs: list[PromptType], **kwargs: Any) -> list[str]:
        model_outputs = self.generate(encoded_inputs, **kwargs)
        for out in model_outputs:
            # for j, dct in enumerate(out.outputs[0].logprobs[:-1]):
            #     if self.tokenizer.eos_token_id in dct:
            #         print("\033[34m", {f"{self.decode([int(k[9:])])}": torch.exp(torch.tensor(dct[k])) for k in dct}, "\033[0m")
            #         out.outputs[0].token_ids = out.outputs[0].token_ids[:j] + [self.tokenizer.eos_token_id]
            #         out.outputs[0].text = self.decode(out.outputs[0].token_ids)
            #         break
            self.encode_cache[out.outputs[0].text] = out.outputs[0].token_ids
        output_string = [out.outputs[0].text for out in model_outputs]
        return output_string

    def sample(
        self,
        txt: str | list[str],
        **kwargs: Any
    ) -> list[str]:
        txt = [txt] if isinstance(txt, str) else txt
        encoded_inputs = self.encode_string(txt)
        return self.sample_from_encoded_inputs(encoded_inputs, **kwargs)

    @hashable_cache(lru_cache(maxsize=2))
    def log_prob(
        self,
        txt: str | list[str],
        outputs: list[str],
        *,
        reduction: bool = True
    ) -> torch.Tensor:
        txt = [txt] if isinstance(txt, str) else txt
        encoded_outputs = self.encode_string(outputs)
        if txt is not None:
            encoded_inputs = self.encode_string(txt)
            encoded_inputs_outputs = [PromptType(prompt_token_ids=inp["prompt_token_ids"] + out["prompt_token_ids"]) for inp, out in zip(encoded_inputs, encoded_outputs)]
        else:
            encoded_inputs_outputs = encoded_outputs

        model_outputs = self.model.generate(
            encoded_inputs_outputs,
            sampling_params=SamplingParams(
                max_tokens=1,
                prompt_logprobs=1
            ),
            use_tqdm=False
        )

        # Retrieve prompt log probs
        output_log_prob = [
            [
                m_out.prompt_logprobs[-j][encoded_outputs[i]["prompt_token_ids"][-j]].logprob
                for j in range(1, len(encoded_outputs[i]["prompt_token_ids"]) + int(txt is not None))
            ][::-1]
            for i, m_out in enumerate(model_outputs)
        ]

        # Pad to equal length
        output_log_prob = torch.tensor(
            [
                output_log_prob[i] + [0] * (max(map(len, output_log_prob)) - len(output_log_prob[i]))
                for i in range(len(model_outputs))
            ],
            dtype=torch.float16
        )

        return output_log_prob.sum(dim=-1) if reduction else output_log_prob


class ChatLocalVLLM(BaseLocalVLLM):
    def __init__(
        self,
        *,
        local_model_path: str = None,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.25,
        max_num_seqs: int = 10,
        model: LLM = None,
        temperature: float = 1.0,
        max_tokens: int = 512,
        seed: int = None,
        enable_thinking: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            local_model_path=local_model_path, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=max_num_seqs,
            model=model, temperature=temperature, max_tokens=max_tokens, seed=seed, **kwargs
        )

        if enable_thinking:
            self.enable_think()
        else:
            self.disable_think()

    def enable_think(self):
        self.enable_thinking = True

    def disable_think(self):
        self.enable_thinking = False

    def apply_chat_template(
        self,
        messages: list[dict] | list[list[dict]],
        *,
        continue_final_message: bool = False,
        add_generation_prompt: bool = True,
    ) -> list[str]:
        """
        Apply the chat template to one or more conversations.

        Parameters
        ----------
        messages : list[dict] | list[list[dict]]
            Either a single conversation (list of message dictionaries) or a batch of
            conversations where each element is a list of message dictionaries.
        continue_final_message : bool, optional
            When True, the template allows continuation of the last message without
            inserting a generation prompt after it. Default is False.
        add_generation_prompt : bool, optional
            When True, a generation prompt is appended to each formatted conversation to
            indicate where the model should start generating. Set to False for already
            finished conversations. Default is True.

        Returns
        -------
        list[str]
            A list of formatted conversation strings, one for each input conversation.
        """
        messages = [messages] if isinstance(messages[0], dict) else messages

        add_generation_prompt = add_generation_prompt and not continue_final_message
        enable_thinking = self.enable_thinking and add_generation_prompt

        out: list[str] = []
        for msg_list in messages:
            try:
                txt = self.tokenizer.apply_chat_template(
                    msg_list,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                txt = self.tokenizer.apply_chat_template(
                    msg_list,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                )
            # Workaround removing duplicated thinking part when the last message is unfinished reasoning from assistant
            #   In qwen3, incomplete assistant thinking will result in duplicated <think> -- "<think>\n\n</think>\n\n<think>..."
            txt = txt.replace('<think>\n\n</think>\n\n<think>', '<think>')
            out.append(txt)
        return out

    def encode_conversation(
        self,
        messages: list[dict] | list[list[dict]],
        *,
        continue_final_message: bool = False,
        add_generation_prompt: bool = True,
        padding_side: Literal["left", "right"] = "left",
    ) -> list[PromptType]:
        txt = self.apply_chat_template(
            messages,
            continue_final_message=continue_final_message,
            add_generation_prompt=add_generation_prompt,
        )
        return self.encode_string(txt, padding_side=padding_side)

    def parse_output_string(self, outputs: list[str]) -> list[dict]:
        return parse_output_string(outputs)

    def compose_parsed_output(self, parsed_outputs: list[dict]) -> list[str]:
        return compose_parsed_output(parsed_outputs)

    def sample(
        self,
        messages: list[ChatCompletionMessageParam]| list[list[ChatCompletionMessageParam]],
        *,
        continue_final_message: bool = False,
        **kwargs: Any
    ) -> list[str]:
        messages = [messages] if isinstance(messages[0], dict) else messages
        encoded_inputs = self.encode_conversation(messages, continue_final_message=continue_final_message)
        return self.sample_from_encoded_inputs(encoded_inputs, **kwargs)

    def log_prob(
        self,
        messages: list[dict] | list[list[dict]],
        outputs: list[str],
        *,
        continue_final_message: bool = False,
        reduction: bool = True
    ) -> torch.Tensor:
        messages = [messages] if isinstance(messages[0], dict) else messages
        encoded_inputs = self.encode_conversation(messages, continue_final_message=continue_final_message)
        encoded_outputs = self.encode_string(outputs)
        encoded_inputs_outputs = [
            TokensPrompt(prompt_token_ids=inp["prompt_token_ids"] + out["prompt_token_ids"])
            for inp, out in zip(encoded_inputs, encoded_outputs)
        ]
        model_outputs = self.model.generate(
            encoded_inputs_outputs,
            sampling_params=SamplingParams(
                max_tokens=1,
                prompt_logprobs=1
            ),
            use_tqdm=False
        )

        # Retrieve prompt log probs
        output_log_prob = [
            [
                m_out.prompt_logprobs[-j][encoded_outputs[i]["prompt_token_ids"][-j]].logprob
                for j in range(1, len(encoded_outputs[i]["prompt_token_ids"]) + 1)
            ][::-1]
            for i, m_out in enumerate(model_outputs)
        ]

        # Pad to equal length
        output_log_prob = torch.tensor(
            [
                output_log_prob[i] + [0] * (max(map(len, output_log_prob)) - len(output_log_prob[i]))
                for i in range(len(model_outputs))
            ],
            dtype=torch.float16
        )

        return output_log_prob.sum(dim=-1) if reduction else output_log_prob


class ServerConfig(TypedDict):
    model: str
    url: str
    api_key: str


class BaseServerVLLM:
    # NOTE: LM should be served with `--logprobs-mode "processed_logprobs"`
    #       to get correctly normalized log probabilities.
    # vLLM caps completion logprobs (often 20). Override via `max_logprobs=` or a subclass
    # class attribute if your server allows more.
    max_allowed_logprobs: int | None = None
    encode_cache: dict[str, list[int]] = {}

    def __init__(
        self,
        *,
        config: ServerConfig,
        temperature: float = 1.0,
        max_tokens: int = 512,
        stop_at_thinking: bool = False,
        seed: int = None,
        max_logprobs: int | None = None,
        **kwargs: Any,
    ):
        self.config = config
        self.client = openai.AsyncClient(
            base_url=self.config["url"],
            api_key=self.config["api_key"],
            http_client=httpx.AsyncClient(verify=False)
        )

        self._extra_body = {
            "skip_special_tokens": False,
            "include_stop_str_in_output": True,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "top_k": 50,
            # "repetition_penalty": 1.005,
            # "length_penalty": 1.0,
            **kwargs.pop("extra_body", {})
        }

        self.generate_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "stop": QED + FINISH + EOS + ROLE + (THINK if stop_at_thinking else []),
            "logprobs": 1,
            "top_p": 0.95,
            # "presence_penalty": 0,
            **kwargs
        }

        if max_logprobs is not None:
            self.max_allowed_logprobs = max_logprobs
        elif type(self).max_allowed_logprobs is not None:
            self.max_allowed_logprobs = type(self).max_allowed_logprobs
        else:
            self.max_allowed_logprobs = BASE_VLLM_MAX_LOGPROBS

    @property
    def extra_body(self):
        return self._extra_body

    def tokenize(self, txt: str) -> list[int]:
        response = requests.post(
            self.config["url"].rsplit("/v1", 1)[0] + "/tokenize",
            json={"prompt": txt},
            headers={"accept": "application/json", "Content-Type": "application/json"}
        )
        return response.json()["tokens"]

    def detokenize(self, tokens: list[int]):
        response = requests.post(
            self.config["url"].rsplit("/v1", 1)[0] + "/detokenize",
            json={'tokens': tokens},
            headers={"accept": "application/json", "Content-Type": "application/json"}
        )
        return response.json()["prompt"]

    def encode(self, txt: str) -> list[int]:
        if txt in self.encode_cache:
            return self.encode_cache[txt]
        else:
            return self.tokenize(txt)

    def encode_string(
        self,
        txt: str | list[str],
        *,
        padding_side: Literal["left", "right"] = "left",
    ) -> list[list[int]]:
        txt = [txt] if isinstance(txt, str) else txt
        return [self.encode(t) for t in txt]

    def decode(
        self,
        token_ids: int | list[int],
        *,
        skip_special_tokens: bool = False
    ) -> str:
        token_ids = [token_ids] if isinstance(token_ids, int) else token_ids
        return self.detokenize(token_ids)

    def batch_decode(
        self,
        sequences: list[int] | list[list[int]],
        *,
        skip_special_tokens: bool = False
    ) -> list[str]:
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
            )
            for seq in sequences
        ]

    def generate(self, inputs: str | list[str] | int | list[int], **kwargs: Any):
        if "max_new_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
        extra_body = {
            **self.extra_body,
            **kwargs.pop("extra_body", {})
        }
        final_kwargs = {
            **self.generate_kwargs,
            **kwargs,
            "extra_body": extra_body
        }
        model_outputs = asyncio.run(
            self.client.completions.create(
                model=self.config["model"],
                prompt=inputs,
                **final_kwargs
            )
        )
        return model_outputs

    def sample_from_encoded_inputs(self, encoded_inputs: list[list[int]], **kwargs: Any) -> list[str]:
        model_outputs = self.generate(encoded_inputs, **kwargs)
        for i in range(len(model_outputs.choices)):
            # for j, dct in enumerate(model_outputs.choices[i].logprobs.top_logprobs[:-1]):
            #     if "token_id:151643" in dct:
            #         print("\033[34m", {f"{self.decode([int(k[9:])])}": torch.exp(torch.tensor(dct[k])) for k in dct}, "\033[0m")
            #         model_outputs.choices[i].token_ids = model_outputs.choices[i].token_ids[:j] + [151643]
            #         model_outputs.choices[i].text = self.decode(model_outputs.choices[i].token_ids)
            #         break
            self.encode_cache[self.decode(encoded_inputs[i] + model_outputs.choices[i].token_ids)] = encoded_inputs[i] + model_outputs.choices[i].token_ids
            self.encode_cache[model_outputs.choices[i].text] = model_outputs.choices[i].token_ids
        output_string = [model_outputs.choices[i].text for i in range(len(model_outputs.choices))]
        return output_string

    def sample(
        self,
        txt: str | list[str],
        **kwargs: Any
    ) -> list[str]:
        txt = [txt] if isinstance(txt, str) else txt
        encoded_inputs = self.encode_string(txt)
        return self.sample_from_encoded_inputs(encoded_inputs, **kwargs)

    def log_prob(
        self,
        txt: str | list[str],
        outputs: list[str],
        *,
        reduction: bool = True
    ) -> torch.Tensor:
        txt = [txt] if isinstance(txt, str) else txt
        encoded_outputs = self.encode_string(outputs)
        if txt is not None:
            encoded_inputs = self.encode_string(txt)
            encoded_inputs_outputs = [inp + out for inp, out in zip(encoded_inputs, encoded_outputs)]
        else:
            encoded_inputs_outputs = encoded_outputs

        response = asyncio.run(
            self.client.completions.create(
                model=self.config["model"],
                prompt=encoded_inputs_outputs,
                max_tokens=1,
                extra_body={**self.extra_body, "prompt_logprobs": 1},
            )
        )
        # Retrieve prompt log probs
        output_log_prob = [
            [
                r_out.prompt_logprobs[-j][str(encoded_outputs[i][-j])]["logprob"]
                for j in range(1, len(encoded_outputs[i]) + int(txt is not None))
            ][::-1]
            for i, r_out in enumerate(response.choices)
        ]

        # Pad to equal length
        output_log_prob = torch.tensor(
            [
                output_log_prob[i] + [0] * (max(map(len, output_log_prob)) - len(output_log_prob[i]))
                for i in range(len(outputs))
            ],
            dtype=torch.float16
        )

        return output_log_prob.sum(dim=-1) if reduction else output_log_prob

    def next_token_log_prob(
        self,
        txt: str | list[str],
        outputs: list[str],
        *,
        top_k: int | None = None,
        **kwargs: Any
    ):
        txt = [txt] if isinstance(txt, str) else txt
        encoded_outputs = self.encode_string(outputs)
        if txt is not None:
            encoded_inputs = self.encode_string(txt)
            encoded_inputs_outputs = [inp + out for inp, out in zip(encoded_inputs, encoded_outputs)]
        else:
            encoded_inputs_outputs = encoded_outputs

        n_logprobs = (
            self.max_allowed_logprobs
            if top_k is None
            else min(top_k, BASE_VLLM_MAX_LOGPROBS)
        )
        response = self.generate(
            encoded_inputs_outputs,
            max_tokens=1,
            top_p=1,
            logprobs=n_logprobs,
            extra_body={'top_k': top_k},
            **kwargs
        )
        return [{int(k[9:]): v for k, v in response.choices[i].logprobs.top_logprobs[0].items()} for i in range(len(response.choices))]


class ChatServerVLLM(BaseServerVLLM):
    chat_template_kwargs: dict[str, Any] = {}

    def __init__(
        self,
        *,
        config: ServerConfig,
        temperature: float = 1.0,
        max_tokens: int = 512,
        seed: int = None,
        enable_thinking: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, temperature=temperature,
            max_tokens=max_tokens, seed=seed, **kwargs
        )

        if enable_thinking:
            self.enable_think()
        else:
            self.disable_think()

    @property
    def extra_body(self):
        # If sample() is implemented with client.chat.completions.create, then
        #   `chat_template_kwargs` should be passed through extra_body.
        # If sample() is implemented with client.completions.create, then
        #   `chat_template_kwargs` should be passed through encode_conversation.
        return {**self._extra_body, "chat_template_kwargs": self.chat_template_kwargs}

    def enable_think(self):
        self.enable_thinking = True
        self.chat_template_kwargs["enable_thinking"] = True

    def disable_think(self):
        self.enable_thinking = False
        self.chat_template_kwargs["enable_thinking"] = False

    def apply_chat_template(
        self,
        messages: list[dict] | list[list[dict]],
        *,
        continue_final_message: bool = False,
        add_generation_prompt: bool = True,
    ) -> list[str]:
        return self.batch_decode(
            self.encode_conversation(
                messages,
                continue_final_message=continue_final_message,
                add_generation_prompt=add_generation_prompt
            )
        )

    def encode_conversation(
        self,
        messages: list[dict] | list[list[dict]],
        *,
        continue_final_message: bool = False,
        add_generation_prompt: bool = True,
        padding_side: Literal["left", "right"] = "left",
    ) -> list[list[int]]:
        messages = [messages] if isinstance(messages[0], dict) else messages
        add_generation_prompt = add_generation_prompt and not continue_final_message
        token_ids = [
            requests.post(
                self.config["url"].rsplit("/v1", 1)[0] + "/tokenize",
                json={
                    "messages": msg,
                    "chat_template_kwargs": self.chat_template_kwargs,
                    "add_generation_prompt": add_generation_prompt,
                    "continue_final_message": continue_final_message
                },
                headers={"accept": "application/json", "Content-Type": "application/json"}
            ).json()["tokens"]
            for msg in messages
        ]
        target = " ".join(map(str, self.tokenize("<think>\n\n</think>\n\n<think>")))
        replacement = " ".join(map(str, self.tokenize("<think>")))
        return [list(map(int, " ".join(map(str, token_ids[i])).replace(target, replacement).split(" "))) for i in range(len(token_ids))]

    def parse_output_string(self, outputs: list[str]) -> list[dict]:
        return parse_output_string(outputs)

    def compose_parsed_output(self, parsed_outputs: list[dict]) -> list[str]:
        return compose_parsed_output(parsed_outputs)

    def sample(
        self,
        messages: list[dict] | list[list[dict]],
        *,
        continue_final_message: bool = False,
        **kwargs: Any
    ) -> list[str]:
        # Single string prompt or batched string prompts -> completions API (BaseServerVLLM), not chat encode.
        if isinstance(messages, str):
            messages = [messages]
        elif isinstance(messages, list) and messages and all(isinstance(m, str) for m in messages):
            return super().sample(messages, **kwargs)
        messages = [messages] if isinstance(messages[0], dict) or isinstance(messages[0], str) else messages
        # yunchenc: add sampling from string directly (one prompt string after normalization above)
        if isinstance(messages[0], str):
            return super().sample(messages, **kwargs)
        encoded_inputs = self.encode_conversation(messages, continue_final_message=continue_final_message)
        return self.sample_from_encoded_inputs(encoded_inputs, **kwargs)

    def log_prob(
        self,
        messages: list[dict] | list[list[dict]],
        outputs: list[str],
        *,
        continue_final_message: bool = False,
        reduction: bool = True
    ) -> torch.Tensor:
        if isinstance(messages, str):
            messages = [messages]
        elif isinstance(messages, list) and messages and all(isinstance(m, str) for m in messages):
            return super().log_prob(messages, outputs, reduction=reduction)
        messages = [messages] if isinstance(messages[0], dict) or isinstance(messages[0], str) else messages
        # yunchenc: add log prob from string directly
        if isinstance(messages[0], str):
            return super().log_prob(messages, outputs, reduction=reduction)
        encoded_inputs = self.encode_conversation(messages, continue_final_message=continue_final_message)
        encoded_outputs = self.encode_string(outputs)
        encoded_inputs_outputs = [inp + out for inp, out in zip(encoded_inputs, encoded_outputs)]

        response = asyncio.run(
            self.client.completions.create(
                model=self.config["model"],
                prompt=encoded_inputs_outputs,
                max_tokens=1,
                extra_body={**self.extra_body, "prompt_logprobs": 1}
            )
        )
        # Retrieve prompt log probs
        output_log_prob = [
            [
                r_out.prompt_logprobs[-j][str(encoded_outputs[i][-j])]["logprob"]
                for j in range(1, len(encoded_outputs[i]) + 1)
            ][::-1]
            for i, r_out in enumerate(response.choices)
        ]

        # Pad to equal length
        output_log_prob = torch.tensor(
            [
                output_log_prob[i] + [0] * (max(map(len, output_log_prob)) - len(output_log_prob[i]))
                for i in range(len(outputs))
            ],
            dtype=torch.float16
        )

        return output_log_prob.sum(dim=-1) if reduction else output_log_prob

    def next_token_log_prob(
        self,
        messages: list[dict] | list[list[dict]],
        outputs: list[str],
        *,
        continue_final_message: bool = False,
        top_k: int | None = None,
        **kwargs: Any
    ):
        messages = [messages] if isinstance(messages[0], dict) else messages
        # yunchenc: add next token log prob from string directly
        if isinstance(messages[0], str):
            return super().next_token_log_prob(messages, outputs, top_k=top_k, **kwargs)
        encoded_inputs = self.encode_conversation(messages, continue_final_message=continue_final_message)
        encoded_outputs = self.encode_string(outputs)
        encoded_inputs_outputs = [inp + out for inp, out in zip(encoded_inputs, encoded_outputs)]

        n_logprobs = (
            self.max_allowed_logprobs
            if top_k is None
            else min(top_k, BASE_VLLM_MAX_LOGPROBS)
        )
        response = self.generate(
            encoded_inputs_outputs,
            max_tokens=1,
            top_p=1,
            logprobs=n_logprobs,
            extra_body={'top_k': top_k},
            **kwargs
        )
        return [{int(k[9:]): v for k, v in response.choices[i].logprobs.top_logprobs[0].items()} for i in range(len(response.choices))]


BASE_LM: TypeAlias = Union[BaseHFLM, BaseLocalVLLM, BaseServerVLLM]

CHAT_LM: TypeAlias = Union[ChatHFLM, ChatLocalVLLM, ChatServerVLLM]

LM: TypeAlias = Union[BaseHFLM, ChatHFLM, BaseLocalVLLM, ChatLocalVLLM, BaseServerVLLM, ChatServerVLLM]


def is_base_lm(lm: Any) -> bool:
    return type(lm) in get_args(BASE_LM)


def is_chat_lm(lm: Any) -> bool:
    return type(lm) in get_args(CHAT_LM)


def is_hf_lm(lm: Any) -> bool:
    return isinstance(lm, BaseHFLM)


def is_local_vllm(lm: Any) -> bool:
    return isinstance(lm, BaseLocalVLLM)


def is_server_vllm(lm: Any) -> bool:
    return isinstance(lm, BaseServerVLLM)