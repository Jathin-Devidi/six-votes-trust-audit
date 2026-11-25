"""Part 2 - how many independent opinions is this panel really worth?

Fill in every function marked TODO. Keep the function names and return shapes.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def phi_matrix(correct: np.ndarray) -> np.ndarray:
    """Pairwise phi coefficient between the models' ERROR indicators, shape (n_models, n_models).

    Phi is the Pearson correlation of two 0/1 vectors (1 = wrong on that question).
    Diagonal is 1. If a model is never wrong or always wrong, its correlation with
    the others is undefined: use 0 for those entries.
    """
    err = (~np.asarray(correct, dtype=bool)).astype(float)
    centred = err - err.mean(axis=0)
    cov = centred.T @ centred
    sd = np.sqrt(np.diag(cov))
    denom = np.outer(sd, sd)
    phi = np.zeros_like(cov)
    ok = denom > 1e-12                      # constant models (never / always wrong) -> 0
    phi[ok] = cov[ok] / denom[ok]
    phi = np.clip(phi, -1.0, 1.0)
    np.fill_diagonal(phi, 1.0)
    return phi


def mean_offdiag(phi: np.ndarray) -> float:
    """Mean of the entries above the diagonal (each pair counted once)."""
    phi = np.asarray(phi, dtype=float)
    iu = np.triu_indices(phi.shape[0], k=1)
    return float(phi[iu].mean()) if len(iu[0]) else 0.0


def n_effective(n_models: int, mean_phi: float) -> float:
    """n_eff = n / (1 + (n - 1) * mean_phi)."""
    return float(n_models / (1.0 + (n_models - 1) * mean_phi))


def plurality_vote(answers: np.ndarray) -> np.ndarray:
    """The panel's answer to each question, shape (n_questions,).

    `answers` has shape (n_questions, n_models) and holds labels such as "A".
    The most common answer wins. On a tie, take the answer given by the earliest
    column among the tied answers.
    """
    answers = np.asarray(answers)
    # support[q, j] = how many models gave the same answer as column j did on question q
    support = (answers[:, :, None] == answers[:, None, :]).sum(axis=2)
    pick = support.argmax(axis=1)           # argmax returns the FIRST maximum -> earliest column wins ties
    return answers[np.arange(answers.shape[0]), pick]


def vote_accuracy(answers: np.ndarray, gold: np.ndarray) -> float:
    """Accuracy of plurality_vote against the gold labels."""
    return float((plurality_vote(answers) == np.asarray(gold)).mean())


def majority_correct_rate(correct: np.ndarray) -> float:
    """Share of questions where STRICTLY more than half of the models are right."""
    correct = np.asarray(correct, dtype=bool)
    return float((correct.sum(axis=1) * 2 > correct.shape[1]).mean())


def majority_correct_if_independent(accs: np.ndarray) -> float:
    """The same rate for imaginary models with accuracies `accs` whose errors are independent.

    Compute it exactly (the number of correct models follows a Poisson-binomial
    distribution) or by a large simulation.
    """
    accs = np.asarray(accs, dtype=float)
    n = len(accs)
    dist = np.zeros(n + 1)                  # exact Poisson-binomial by dynamic programming
    dist[0] = 1.0
    for a in accs:
        dist[1:] = dist[1:] * (1.0 - a) + dist[:-1] * a
        dist[0] *= 1.0 - a
    need = n // 2 + 1                       # strictly more than half
    return float(dist[need:].sum())


def select_panel(panel, k: int = 3) -> dict:
    """Choose a k-model panel on the dev split, then score it once on the test split.

    `panel` is the object returned by io.load_predictions.

    1. Rank models by accuracy on the dev rows.
    2. Score every k-model subset by vote_accuracy on dev. Inside a subset, order the
       columns by dev accuracy, best first, so that ties go to the strongest member.
    3. Keep the subset with the highest dev score.
    4. Score that subset, and the naive "top k by dev accuracy" subset, on the test rows.

    Return a dict with exactly these keys:
        k, models, dev_accuracy, test_accuracy,
        baseline_top_k_models, baseline_top_k_dev_accuracy, baseline_top_k_test_accuracy
    """
    models = list(panel.models)
    m = len(models)
    k = max(1, min(k, m))
    dev = np.asarray(panel.split) == "dev"
    test = np.asarray(panel.split) == "test"
    if not dev.any() or not test.any():     # no usable split: fall back to all rows
        dev = test = np.ones(len(panel.gold), dtype=bool)
    correct = panel.correct
    dev_acc = correct[dev].mean(axis=0)
    order = list(np.argsort(-dev_acc, kind="stable"))   # best first, ties by column order
    ans_dev, gold_dev = panel.answers[dev], panel.gold[dev]
    ans_test, gold_test = panel.answers[test], panel.gold[test]

    def score(cols, ans, gold):
        return vote_accuracy(ans[:, list(cols)], gold)

    best, best_score = None, -1.0
    for cols in combinations(order, k):     # each tuple is already in dev-accuracy order
        sc = score(cols, ans_dev, gold_dev)
        if sc > best_score + 1e-12:
            best, best_score = cols, sc
    base = tuple(order[:k])
    return {
        "k": k,
        "models": [models[j] for j in best],
        "dev_accuracy": float(best_score),
        "test_accuracy": float(score(best, ans_test, gold_test)),
        "baseline_top_k_models": [models[j] for j in base],
        "baseline_top_k_dev_accuracy": float(score(base, ans_dev, gold_dev)),
        "baseline_top_k_test_accuracy": float(score(base, ans_test, gold_test)),
    }
