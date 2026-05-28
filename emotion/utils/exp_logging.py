"""Logging for Math500 experiment runners: tqdm-safe lines on stderr."""

from __future__ import annotations

import logging
import sys

LOG_NAME = "experiments.math500"


class _TqdmCompatibleHandler(logging.Handler):
    """Write log records via tqdm.write when available so nested bars stay intact."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            try:
                from tqdm.auto import tqdm  # type: ignore

                tqdm.write(msg, file=sys.stderr)
            except Exception:
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
        except Exception:
            self.handleError(record)


def configure_experiment_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Attach a single handler to the experiments logger (idempotent).
    Call once at the start of each runner's main().
    """
    log = logging.getLogger(LOG_NAME)
    log.setLevel(level)
    if not log.handlers:
        h = _TqdmCompatibleHandler()
        h.setLevel(level)
        h.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        log.addHandler(h)
        log.propagate = False
    return log


def get_exp_logger(name_suffix: str | None = None) -> logging.Logger:
    if name_suffix:
        return logging.getLogger(f"{LOG_NAME}.{name_suffix}")
    return logging.getLogger(LOG_NAME)
