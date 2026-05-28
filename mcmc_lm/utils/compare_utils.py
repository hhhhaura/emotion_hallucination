from collections import defaultdict
from pydantic import Field
from sklearn.isotonic import IsotonicRegression
import torch
from typing import Annotated, Any, Literal, Sequence, TypeAlias


BINARY: TypeAlias = Literal[0, 1]
NONNEGATIVE: TypeAlias = Annotated[float, Field(ge=0.)]
UNIT_INTERVAL: TypeAlias = Annotated[float, Field(ge=0., le=1.)]


def argmax_prediction(
    predictions: Sequence[Any],
    confidence: Sequence[NONNEGATIVE],
    normalize: bool = True
) -> tuple[int, Any, UNIT_INTERVAL]:
    """Select the prediction with the maximum individual confidence, then
    compute its normalized confidence over identical predictions.

    Returns:
        tuple[int, Any, UNIT_INTERVAL]:
            The index of the best prediction, the best prediction itself,
            and its normalized aggregated confidence.
    """
    confidence: torch.Tensor = torch.tensor(confidence)
    best_idx = confidence.argmax().item()
    best_prediction = predictions[best_idx]
    if normalize:
        confidence = confidence[best_idx] / confidence.sum()
    else:
        confidence = confidence[best_idx]
    return best_idx, best_prediction, confidence.item()


def argmax_prediction_with_aggregated_confidence(
    predictions: Sequence[Any],
    confidence: Sequence[NONNEGATIVE],
    normalize: bool = True
) -> tuple[int, Any, UNIT_INTERVAL]:
    """Select the prediction with the maximum individual confidence, then
    compute its normalized aggregated confidence over identical predictions.

    Returns:
        tuple[int, Any, UNIT_INTERVAL]:
            The index of the best prediction, the best prediction itself,
            and its normalized aggregated confidence.
    """
    confidence: torch.Tensor = torch.tensor(confidence)
    best_idx = confidence.argmax().item()
    best_prediction = predictions[best_idx]
    best_prediction_matched_indices = [i for i in range(len(predictions)) if predictions[i] == best_prediction]
    if normalize:
        confidence = confidence[best_prediction_matched_indices].sum() / confidence.sum()
    else:
        confidence = confidence[best_prediction_matched_indices].sum()
    return best_idx, best_prediction, confidence.item()


def consensus_prediction(
    predictions: Sequence[Any],
    weights: Sequence[NONNEGATIVE],
    normalize: bool = True
) -> tuple[int, Any, UNIT_INTERVAL]:
    """Return the consensus prediction and its normalized aggregated confidence.
    Args:
        predictions: A sequence of predictions.
        weight: A sequence of weights corresponding to the predictions, which should
                be $\\frac{\\pi(y)}{p(y)}}$, the importance weight of the prediction
                sampled from the proposal distribution $p(y)$.
    Returns:
        tuple[int, Any, UNIT_INTERVAL]:
            One index of the selected prediction group, the selected prediction,
            and its normalized aggregated confidence.
    """
    total_weight = sum(weights)
    grouped_indices = defaultdict(list)
    grouped_weights = defaultdict(int)
    for i, pred in enumerate(predictions):
        grouped_indices[pred].append(i)
        grouped_weights[pred] += weights[i]
    if normalize:
        for pred, w in grouped_weights.items():
            grouped_weights[pred] = w / total_weight
    best_prediction = max(grouped_weights, key=lambda k: grouped_weights[k])
    best_idx = grouped_indices[best_prediction][0]
    return best_idx, best_prediction, grouped_weights[best_prediction]


def binary_roc_auc(
    confidence: Sequence[float],
    correctness: Sequence[BINARY]
) -> UNIT_INTERVAL | None:
    """Instance-level ROC AUC with the following formula:
    $$
        \\frac{1}{|Y^+||Y^-|}\\sum_{y_+\\in Y^+}\\sum_{y_-\\in Y^-} score(y_+) > score(y_-)
    $$
    """
    count_0 = 0
    correct_pairs = 0
    rank = torch.tensor(confidence).argsort(descending=False)
    sorted_correctness = torch.tensor(correctness, dtype=float)[rank]
    for x in sorted_correctness:
        if x == 0:
            count_0 += 1
        else:  # x == 1
            correct_pairs += count_0
    total_pairs = count_0 * (len(sorted_correctness) - count_0)
    return correct_pairs / total_pairs if total_pairs > 0 else None


def binary_brier_score(
    confidence: Sequence[float],
    correctness: Sequence[BINARY]
) -> UNIT_INTERVAL | None:
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    confidence = torch.tensor(ir.fit_transform(confidence, correctness))
    return (torch.tensor(correctness).int() - confidence).pow(2).sum() / len(correctness)


def brier_reliability(
    confidence: Sequence[float],
    correctness: Sequence[BINARY]
) -> UNIT_INTERVAL | None:
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    confidence: torch.Tensor = torch.tensor(ir.fit_transform(confidence, correctness))
    correctness: torch.Tensor = torch.tensor(correctness).int()
    uniq_conf, uniq_indices, uniq_counts = confidence.unique(return_inverse=True, return_counts=True)
    N = len(correctness)
    K = uniq_conf.shape[0]
    o_k_bar = (correctness[None, :] * (torch.arange(K)[:, None] == uniq_indices[None, :])).sum(dim=1) / uniq_counts
    return ((uniq_conf - o_k_bar).pow(2) * uniq_counts).sum() / N


def brier_resolution(
    confidence: Sequence[float],
    correctness: Sequence[BINARY]
) -> UNIT_INTERVAL | None:
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    confidence: torch.Tensor = torch.tensor(ir.fit_transform(confidence, correctness))
    correctness: torch.Tensor = torch.tensor(correctness).int()
    uniq_conf, uniq_indices, uniq_counts = confidence.unique(return_inverse=True, return_counts=True)
    N = len(correctness)
    K = uniq_conf.shape[0]
    o_k_bar = (correctness[None, :] * (torch.arange(K)[:, None] == uniq_indices[None, :])).sum(dim=1) / uniq_counts
    o_bar = correctness.sum() / N
    return (uniq_counts * (o_k_bar - o_bar).pow(2)).sum() / N


def brier_uncertainty(
    confidence: Sequence[float],
    correctness: Sequence[BINARY]
) -> UNIT_INTERVAL | None:
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    confidence: torch.Tensor = torch.tensor(ir.fit_transform(confidence, correctness))
    correctness: torch.Tensor = torch.tensor(correctness).int()
    N = len(correctness)
    o_bar = correctness.sum() / N
    return o_bar * (1 - o_bar)


def brier_refinement(
    confidence: Sequence[float],
    correctness: Sequence[BINARY]
) -> UNIT_INTERVAL | None:
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    confidence: torch.Tensor = torch.tensor(ir.fit_transform(confidence, correctness))
    correctness: torch.Tensor = torch.tensor(correctness).int()
    uniq_conf, uniq_indices, uniq_counts = confidence.unique(return_inverse=True, return_counts=True)
    N = len(correctness)
    K = uniq_conf.shape[0]
    o_k_bar = (correctness[None, :] * (torch.arange(K)[:, None] == uniq_indices[None, :])).sum(dim=1) / uniq_counts
    return (uniq_counts * (o_k_bar * (1 - o_k_bar))).sum() / N


def area_under_receiver_operating_characterisitc_curve(
    confidence: Sequence[float],
    correctness: Sequence[BINARY],
    return_plot_data: bool = False
) -> UNIT_INTERVAL | tuple[UNIT_INTERVAL, torch.Tensor, torch.Tensor]:
    count_0 = 0
    correct_pairs = 0
    rank = torch.tensor(confidence).argsort(descending=False)
    sorted_correctness = torch.tensor(correctness, dtype=float)[rank]
    for x in sorted_correctness:
        if x == 0:
            count_0 += 1
        else:  # x == 1
            correct_pairs += count_0
    total_pairs = count_0 * (len(sorted_correctness) - count_0)
    auroc = correct_pairs / total_pairs if total_pairs > 0 else 0
    if return_plot_data:
        fn = sorted_correctness.cumsum(0)
        tp = sorted_correctness.sum() - fn
        fp = (1 - sorted_correctness).cumsum(0)
        tn = (1 - sorted_correctness).sum() - fp
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        return auroc, fpr, tpr
    else:
        return auroc


def area_under_accuracy_coverage_curve(
    confidence: Sequence[float],
    correctness: Sequence[BINARY],
    return_plot_data: bool = False
) -> UNIT_INTERVAL | tuple[UNIT_INTERVAL, torch.Tensor, torch.Tensor]:
    rank = torch.tensor(confidence).argsort(descending=True)
    sorted_correctness = torch.tensor(correctness, dtype=float)[rank]
    cum_score = sorted_correctness.cumsum(0)
    effective_size = torch.arange(1, len(confidence) + 1).float()
    selective_accuracy = cum_score / effective_size
    if return_plot_data:
        abstention_rate = 1 - effective_size / len(confidence)
        return selective_accuracy.mean(), abstention_rate, selective_accuracy
    else:
        return selective_accuracy.mean()


def expected_calibration_error(
    confidence: Sequence[float],
    correctness: Sequence[BINARY],
    bins: int = 10,
    return_plot_data: bool = False,
    isotonic_regression_transform: bool = True,
    expected_prediction_with_confidence: bool = True
) -> UNIT_INTERVAL | tuple[UNIT_INTERVAL, torch.Tensor, torch.Tensor]:
    """Implementation of ECE in `Selectively Answering Ambiguous Questions`

    > Predictions are grouped into ten equally sized bins, ranked by the evaluated
    system's assigned confidence scores. We compute the mean absolute distance
    between the average confidence score and the accuracy of predictions in each bin,
    averaging across all bins. If we interpret a confidence score to represent a
    probability, this corresponds to the difference in the predicted probability of
    correctness from the actual probability of correctness.
    """
    if isotonic_regression_transform:
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        confidence = torch.tensor(ir.fit_transform(confidence, correctness))
    else:
        confidence: torch.Tensor = torch.tensor(confidence)
    bins = min(bins, len(confidence))
    rank = confidence.argsort(descending=False)
    sorted_correctness = torch.tensor(correctness, dtype=float)[rank]
    bin_size = len(confidence) // bins
    calibrated_prediction = torch.tensor([sorted_correctness[i * bin_size:(i + 1) * bin_size].mean() for i in range(bins)])
    if expected_prediction_with_confidence:
        expected_prediction = torch.tensor([confidence[rank][i * bin_size:(i + 1) * bin_size].mean() for i in range(bins)])
    else:
        expected_prediction = torch.linspace(0, 1, bins + 1)[1:] - 0.5 / bins
    if return_plot_data:
        return (calibrated_prediction - expected_prediction).abs().mean(), expected_prediction, calibrated_prediction
    else:
        return (calibrated_prediction - expected_prediction).abs().mean()


def coverage_at_acc(
    confidence: Sequence[float],
    correctness: Sequence[BINARY],
    accuracy: UNIT_INTERVAL,
    normalize: bool = True
) -> float | UNIT_INTERVAL:
    """Implementation of Coverage@Acc in `Selectively Answering Ambiguous Questions`

    > we measure the fraction of questions the system can answer correctly if it
    needs to maintain a certain accuracy. Specifically, C@Acc is the maximum
    coverage such that the accuracy on the C% of most-confident predictions is at
    least Acc%.
    """
    rank = torch.tensor(confidence).argsort(descending=True)
    sorted_correctness = torch.tensor(correctness, dtype=float)[rank]
    cum_score = sorted_correctness.cumsum(0)
    effective_size = torch.arange(1, len(confidence) + 1).float()
    selective_accuracy = cum_score / effective_size
    feasible_coverages = torch.where(selective_accuracy >= accuracy)[0]
    if not feasible_coverages.size()[0]:
        return 0
    if normalize:
        return (feasible_coverages[-1].item() + 1) / len(confidence)
    else:
        return feasible_coverages[-1].item() + 1


def coverage_at_acc_curve(
    confidence: Sequence[float],
    correctness: Sequence[BINARY],
    normalize: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Implementation of Coverage@Acc in `Selectively Answering Ambiguous Questions`

    The upper envelope curve of reversed `area_under_accuracy_coverage_curve`.
    (abstention rate = N_{data} - coverage)
    """
    rank = torch.tensor(confidence).argsort(descending=True)
    sorted_correctness = torch.tensor(correctness, dtype=float)[rank]
    cum_score = sorted_correctness.cumsum(0)
    effective_size = torch.arange(1, len(confidence) + 1).float()
    selective_accuracy = cum_score / effective_size
    coverage = (len(confidence) - 1 - selective_accuracy.flip(0).cummax(dim=0).indices).unique()
    accuracy = selective_accuracy[coverage]
    if normalize:
        return coverage / len(confidence), accuracy
    else:
        return coverage, accuracy


def top_k_acc(
    confidence: Sequence[float],
    correctness: Sequence[BINARY],
) -> torch.Tensor:
    rank = torch.tensor(confidence).argsort(descending=True)
    sorted_correctness = torch.tensor(correctness, dtype=float)[rank]
    return (sorted_correctness.cumsum(0) > 0).int()


def conformal_predictive_distribution(
    confidence: Sequence[float],
    correctness: Sequence[BINARY],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conformal p-value based predictive distribution.

    Assume larger confidence, the more likely it is to be correct.
    For score S, correctness C \\in \\{+, -\\}, we have
    $$
        \\pi_+(s) = P(S \\leq s | C = +) = \\frac{|\\{(S_i, C_i) | S_i \\leq s, C_i = +\\}| + 1}{|\\{(S_i, C_i) | C_i = +\\}| + 1}
        \\pi_-(s) = P(S \\geq s | C = -) = \\frac{|\\{(S_i, C_i) | S_i \\geq s, C_i = - \\}| + 1}{|\\{(S_i, C_i) | C_i = -\\}| + 1}
        P(C = + | S = s) = \\frac{\\pi_+(s)}{\\pi_+(s) + \\pi_-(s)}
    $$
    """
    confidence: torch.Tensor = torch.tensor(confidence)
    rank = confidence.argsort(descending=False)
    sorted_correctness = torch.tensor(correctness, dtype=float)[rank]
    positive_cnt = sorted_correctness.sum()
    negative_cnt = (1 - sorted_correctness).sum()
    positive_cum_score = sorted_correctness.cumsum(0)
    negative_cum_score = negative_cnt - (1 - sorted_correctness).cumsum(0)
    positive_prob = (positive_cum_score + 1) / (positive_cnt + 1)
    negative_prob = (negative_cum_score + 1) / (negative_cnt + 1)
    return confidence[rank], positive_prob / (positive_prob + negative_prob)

