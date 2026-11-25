"""Part 1 - an honest leaderboard: scores with uncertainty, a paired test, rank stability.

Fill in every function marked TODO. Keep the function names and return shapes:
tests/test_public.py and report.py rely on them. Add helpers freely.

`correct` is always a boolean array of shape (n_questions, n_models);
correct[q, m] is True when model m answered question q correctly.
"""
from __future__ import annotations

import math

import numpy as np

_CHUNK = 200  # bootstrap replicates processed at once, to bound memory


def _ranking(acc: np.ndarray) -> np.ndarray:
    """Model indices best first; ties keep file column order (stable sort)."""
    return np.argsort(-np.asarray(acc), kind="stable")


def accuracy(correct: np.ndarray) -> np.ndarray:
    """Per-model accuracy, shape (n_models,)."""
    return np.asarray(correct, dtype=float).mean(axis=0)


def bootstrap_accuracy_ci(
    correct: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap interval for each model's accuracy.

    Resample QUESTIONS with replacement n_boot times. Return (low, high), each of
    shape (n_models,), the 2.5th and 97.5th percentiles when level = 0.95.
    """
    correct = np.asarray(correct, dtype=bool)
    n = correct.shape[0]
    rng = np.random.default_rng(seed)
    stats = np.empty((n_boot, correct.shape[1]))
    for start in range(0, n_boot, _CHUNK):
        stop = min(start + _CHUNK, n_boot)
        idx = rng.integers(0, n, size=(stop - start, n))
        stats[start:stop] = correct[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(stats, [alpha, 1.0 - alpha], axis=0)
    return low, high


def mcnemar_exact_p(a_correct: np.ndarray, b_correct: np.ndarray) -> float:
    """Two-sided exact McNemar p-value for two models answering the same questions.

    Only questions where exactly one of the two models is right carry information.
    If there are none, return 1.0.
    """
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    # P(X <= k) for X ~ Binomial(n, 1/2), doubled for a two-sided test
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return float(min(1.0, 2.0 * tail))


def compare_top_two(
    correct: np.ndarray, models: list[str], n_boot: int = 2000, seed: int = 0
) -> dict:
    """Paired comparison of the #1 and #2 model.

    Return a dict with exactly these keys:
        leader, runner_up      model names
        accuracy_gap           leader accuracy minus runner-up accuracy
        gap_ci_low, gap_ci_high  95% PAIRED bootstrap interval for the gap
        p_value                from mcnemar_exact_p
        significant            True when p_value < 0.05
    """
    correct = np.asarray(correct, dtype=bool)
    acc = accuracy(correct)
    order = _ranking(acc)
    lead, runner = int(order[0]), int(order[1])
    gap = float(acc[lead] - acc[runner])
    # paired bootstrap: the SAME resampled questions score both models
    n = correct.shape[0]
    diff = correct[:, lead].astype(float) - correct[:, runner].astype(float)
    rng = np.random.default_rng(seed)
    gaps = np.empty(n_boot)
    for start in range(0, n_boot, _CHUNK):
        stop = min(start + _CHUNK, n_boot)
        idx = rng.integers(0, n, size=(stop - start, n))
        gaps[start:stop] = diff[idx].mean(axis=1)
    lo, hi = np.quantile(gaps, [0.025, 0.975])
    p = mcnemar_exact_p(correct[:, lead], correct[:, runner])
    return {
        "leader": models[lead],
        "runner_up": models[runner],
        "accuracy_gap": gap,
        "gap_ci_low": float(lo),
        "gap_ci_high": float(hi),
        "p_value": float(p),
        "significant": bool(p < 0.05),
    }


def rank_stability(
    correct: np.ndarray, sizes: list[int], n_draws: int = 2000, seed: int = 0
) -> list[dict]:
    """How often the full-benchmark leader is still #1 on a re-drawn benchmark.

    For each size in `sizes`: draw `size` questions with replacement, n_draws times.
    The leader "holds" on a draw when its accuracy is >= every other model's
    (ties count as holding). Return [{"n": size, "p_leader_holds": fraction}, ...].
    """
    correct = np.asarray(correct, dtype=bool)
    n, m = correct.shape
    leader = int(_ranking(accuracy(correct))[0])
    others = [j for j in range(m) if j != leader]
    rng = np.random.default_rng(seed)
    out = []
    for size in sizes:
        holds = 0
        for start in range(0, n_draws, _CHUNK):
            stop = min(start + _CHUNK, n_draws)
            idx = rng.integers(0, n, size=(stop - start, size))
            counts = correct[idx].sum(axis=1)          # (draws, models), integer scores
            holds += int((counts[:, leader] >= counts[:, others].max(axis=1)).sum())
        out.append({"n": int(size), "p_leader_holds": holds / n_draws})
    return out


def default_sizes(n_questions: int) -> list[int]:
    """Benchmark sizes to test: 50, 200, 500 (when smaller than the file) and the full size."""
    return [s for s in (50, 200, 500) if s < n_questions] + [n_questions]
