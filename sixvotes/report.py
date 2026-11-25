"""Part 4 - run the whole audit and write results.json and REPORT.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import leaderboard as lb
from . import panel as pn
from .io import Panel


def _category_gaps(panel: Panel, min_rows: int = 30) -> list[dict]:
    """For each model, its weakest category compared with the rest of the benchmark."""
    correct = panel.correct
    cats = np.asarray(panel.category)
    rows = []
    for j, name in enumerate(panel.models):
        worst = None
        for cat in sorted(set(cats)):
            m = cats == cat
            if m.sum() < min_rows or (~m).sum() < min_rows:
                continue
            inside, outside = float(correct[m, j].mean()), float(correct[~m, j].mean())
            if worst is None or inside - outside < worst["gap"]:
                worst = {"model": name, "category": str(cat), "n": int(m.sum()),
                         "in_category": inside, "elsewhere": outside, "gap": inside - outside}
        if worst:
            rows.append(worst)
    return sorted(rows, key=lambda r: r["gap"])


def _closest_pair(panel: Panel, phi: np.ndarray) -> dict | None:
    """The two models whose errors correlate most, and how often their wrong answers match."""
    m = len(panel.models)
    best, pair = -2.0, None
    for i in range(m):
        for j in range(i + 1, m):
            if phi[i, j] > best:
                best, pair = phi[i, j], (i, j)
    i, j = pair
    both = ~panel.correct[:, i] & ~panel.correct[:, j]
    same = float((panel.answers[both, i] == panel.answers[both, j]).mean()) if both.any() else 0.0
    return {"a": panel.models[i], "b": panel.models[j], "phi": float(best),
            "both_wrong": int(both.sum()), "same_wrong_answer": same}


def run_audit(panel: Panel, out_dir: str | Path, seed: int = 0, n_boot: int = 2000, k: int = 3) -> dict:
    """Compute everything from Parts 1 and 2 and write two files into out_dir:

        results.json   exact layout in docs/OUTPUT_CONTRACT.md (this file is auto-graded)
        REPORT.md      a short report a hiring-product manager could read (Part 4)

    Must work for ANY predictions file with the same four meta columns: any number
    of models (2 or more), any model names, any answer labels, any number of rows.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    correct = panel.correct
    models = list(panel.models)
    n_q, n_m = correct.shape

    # ---- Part 1
    acc = lb.accuracy(correct)
    low, high = lb.bootstrap_accuracy_ci(correct, n_boot=n_boot, seed=seed)
    order = np.argsort(-acc, kind="stable")
    leaderboard = [
        {"model": models[j], "accuracy": float(acc[j]), "ci_low": float(low[j]),
         "ci_high": float(high[j]), "rank": r + 1}
        for r, j in enumerate(order)
    ]
    top2 = lb.compare_top_two(correct, models, n_boot=n_boot, seed=seed)
    stability = lb.rank_stability(correct, lb.default_sizes(n_q), n_draws=n_boot, seed=seed)

    # ---- Part 2
    phi = pn.phi_matrix(correct)
    mean_phi = pn.mean_offdiag(phi)
    panel_block = {
        "phi_matrix": [[float(x) for x in row] for row in phi],
        "mean_phi": float(mean_phi),
        "n_eff": float(pn.n_effective(n_m, mean_phi)),
        "vote_accuracy": float(pn.vote_accuracy(panel.answers, panel.gold)),
        "majority_correct_rate": float(pn.majority_correct_rate(correct)),
        "majority_correct_rate_if_independent": float(pn.majority_correct_if_independent(acc)),
    }
    recommended = pn.select_panel(panel, k=k)

    results = {
        "n_questions": int(n_q),
        "models": models,
        "leaderboard": leaderboard,
        "top2": top2,
        "rank_stability": stability,
        "panel": panel_block,
        "recommended_panel": recommended,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "REPORT.md").write_text(_markdown(panel, results, _closest_pair(panel, phi), _category_gaps(panel)))
    return results


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _markdown(panel: Panel, r: dict, pair: dict, gaps: list[dict]) -> str:
    t2, pb, rp = r["top2"], r["panel"], r["recommended_panel"]
    n_m = len(r["models"])
    best_single = r["leaderboard"][0]
    L = []
    L.append("# Six Votes audit report\n")
    L.append(f"{r['n_questions']} questions, {n_m} models. All numbers come from `results.json`.\n")

    L.append("## 1. Is the top model really #1?\n")
    L.append("| Rank | Model | Accuracy | 95% interval |\n|---|---|---|---|")
    for e in r["leaderboard"]:
        L.append(f"| {e['rank']} | {e['model']} | {_pct(e['accuracy'])} | {_pct(e['ci_low'])} to {_pct(e['ci_high'])} |")
    verdict = ("a real lead" if t2["significant"] else "NOT a statistically reliable lead")
    L.append(f"\n**{t2['leader']} vs {t2['runner_up']}:** gap {100 * t2['accuracy_gap']:.1f} points, "
             f"95% paired interval {100 * t2['gap_ci_low']:.1f} to {100 * t2['gap_ci_high']:.1f}, "
             f"exact McNemar p = {t2['p_value']:.3f}. That is {verdict}.\n")
    L.append("If the benchmark were drawn again, how often would the same model still come first?\n")
    L.append("| Questions | Leader holds #1 |\n|---|---|")
    for s in r["rank_stability"]:
        L.append(f"| {s['n']} | {_pct(s['p_leader_holds'])} |")

    L.append("\n## 2. How many independent opinions are the models worth?\n")
    L.append(f"- Mean error correlation (phi): **{pb['mean_phi']:.2f}**")
    L.append(f"- Effective number of models: **{pb['n_eff']:.2f} of {n_m}**")
    L.append(f"- Plurality vote accuracy: **{_pct(pb['vote_accuracy'])}** "
             f"(best single model {_pct(best_single['accuracy'])})")
    L.append(f"- A majority is right on {_pct(pb['majority_correct_rate'])} of questions; "
             f"independent models with the same accuracies would manage "
             f"{_pct(pb['majority_correct_rate_if_independent'])}.")
    L.append(f"- The average hides structure: **{pair['a']} and {pair['b']}** have phi {pair['phi']:.2f}; when both "
             f"are wrong ({pair['both_wrong']} questions) they give the same wrong answer "
             f"{_pct(pair['same_wrong_answer'])} of the time. Two votes from them are close to one vote counted twice.\n")

    L.append("## 3. Best three-model panel (chosen on dev, scored once on test)\n")
    L.append(f"- Chosen panel: {', '.join(rp['models'])} - dev {_pct(rp['dev_accuracy'])}, "
             f"test {_pct(rp['test_accuracy'])}")
    L.append(f"- Top-{rp['k']} by accuracy: {', '.join(rp['baseline_top_k_models'])} - dev "
             f"{_pct(rp['baseline_top_k_dev_accuracy'])}, test {_pct(rp['baseline_top_k_test_accuracy'])}")
    L.append("- The dev score of the chosen panel is optimistic because it was the best of many subsets "
             "on that same data; the test score is the honest one.\n")

    if gaps:
        L.append("## 4. Where each model is weakest\n")
        L.append("| Model | Weakest category | There | Elsewhere |\n|---|---|---|---|")
        for g in gaps:
            L.append(f"| {g['model']} | {g['category']} (n={g['n']}) | {_pct(g['in_category'])} | {_pct(g['elsewhere'])} |")
        L.append("")

    L.append("## Caveats\n")
    L.append("- Question-level bootstrap assumes questions are independent draws from the same population.")
    L.append("- Models that share a vendor or training data can share mistakes; correlation here is measured, not explained.")
    L.append("- Accuracy is measured against the supplied gold answers; wrong gold labels would hit every model alike.")
    return "\n".join(L) + "\n"
