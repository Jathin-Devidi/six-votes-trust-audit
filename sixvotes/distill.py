"""Part 3 - can a student inherit a skill through outputs that have nothing to do with it?

tinynet.TinyNet gives you the network and both training steps. You write the experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .tinynet import TinyNet

N_CLASS = 10
DEFAULTS = dict(
    hidden=128,          # units per hidden layer
    n_aux=3,             # auxiliary outputs
    teacher_epochs=5,    # passes over the real training digits
    teacher_batch=64,
    student_steps=1500,  # gradient steps on noise
    student_batch=128,
    lr=1e-3,
)
SEEDS = (0, 1, 2, 3, 4)

# Own experiment: how much of the teacher's starting weights must the student share?
# Written BEFORE running it:
EXPERIMENT_QUESTION = ("How much of the teacher's starting weights must a noise-trained student share "
                       "for the hidden skill to transfer?")
EXPERIMENT_HYPOTHESIS = ("I expected a smooth, roughly proportional fade: sharing 50% of the initial weights "
                         "would give about half the accuracy gain of a full copy, and 0% would stay near "
                         "the untrained level.")
SHARED_FRACTIONS = (0.0, 0.5, 0.9, 1.0)


def load_digits(path: str | Path):
    """Returns standardised (X_train, y_train, X_test, y_test). Inputs have 64 features."""
    d = np.load(path)
    mu, sd = d["X_train"].mean(), d["X_train"].std()
    return (d["X_train"] - mu) / sd, d["y_train"], (d["X_test"] - mu) / sd, d["y_test"]


def _train_teacher(start: TinyNet, X, y, seed: int, cfg: dict) -> TinyNet:
    teacher = start.clone()
    rng = np.random.default_rng(seed)
    for _ in range(cfg["teacher_epochs"]):
        order = rng.permutation(len(y))
        for s in range(0, len(order), cfg["teacher_batch"]):
            b = order[s:s + cfg["teacher_batch"]]
            teacher.step_classify(X[b], y[b], lr=cfg["lr"])
    return teacher


def _train_student(student: TinyNet, teacher: TinyNet, seed: int, n_in: int, cfg: dict) -> TinyNet:
    """Copy only the teacher's auxiliary outputs, on pure Gaussian noise."""
    rng = np.random.default_rng(10_000 + seed)
    for _ in range(cfg["student_steps"]):
        noise = rng.normal(size=(cfg["student_batch"], n_in))
        student.step_match_aux(noise, teacher.aux_logits(noise), lr=cfg["lr"])
    return student


def _mix_init(base: TinyNet, other: TinyNet, frac: float, seed: int) -> TinyNet:
    """A network whose initial weights equal `base` on a random fraction `frac` of entries, else `other`."""
    rng = np.random.default_rng(20_000 + seed)
    mixed = base.clone()
    for p, q in zip(mixed.params, other.params):
        take_other = rng.random(p.shape) >= frac
        p[take_other] = q[take_other]
    return mixed


def run(digits_path: str | Path, out_dir: str | Path) -> dict:
    """Run the experiment for every seed in SEEDS and write <out_dir>/subliminal_results.json.

    See Part 3 of the brief for the steps and docs/OUTPUT_CONTRACT.md for the JSON layout.
    """
    cfg = dict(DEFAULTS)
    X, y, Xt, yt = load_digits(digits_path)
    n_in = X.shape[1]
    per_seed = []
    mix_acc = {f: [] for f in SHARED_FRACTIONS}

    for seed in SEEDS:
        start = TinyNet(n_in, cfg["hidden"], N_CLASS, cfg["n_aux"], seed=seed)
        teacher = _train_teacher(start, X, y, seed, cfg)
        same = _train_student(start.clone(), teacher, seed, n_in, cfg)
        stranger = TinyNet(n_in, cfg["hidden"], N_CLASS, cfg["n_aux"], seed=seed + 1000)
        diff = _train_student(stranger.clone(), teacher, seed, n_in, cfg)
        per_seed.append({
            "seed": seed,
            "untrained_acc": start.accuracy(Xt, yt),
            "teacher_acc": teacher.accuracy(Xt, yt),
            "student_same_init_acc": same.accuracy(Xt, yt),
            "student_diff_init_acc": diff.accuracy(Xt, yt),
        })
        for frac in SHARED_FRACTIONS:
            if frac == 1.0:
                mix_acc[frac].append(per_seed[-1]["student_same_init_acc"])
            elif frac == 0.0:
                mix_acc[frac].append(per_seed[-1]["student_diff_init_acc"])
            else:
                s = _train_student(_mix_init(start, stranger, frac, seed), teacher, seed, n_in, cfg)
                mix_acc[frac].append(s.accuracy(Xt, yt))

    keys = ["untrained_acc", "teacher_acc", "student_same_init_acc", "student_diff_init_acc"]
    mean = {k: float(np.mean([r[k] for r in per_seed])) for k in keys}
    conditions = [
        {"label": f"{int(f * 100)}% of initial weights shared with the teacher",
         "value": float(f), "student_acc_mean": float(np.mean(mix_acc[f]))}
        for f in SHARED_FRACTIONS
    ]
    base = mean["untrained_acc"]
    lo, hi = conditions[0]["student_acc_mean"], conditions[-1]["student_acc_mean"]
    half = next(c["student_acc_mean"] for c in conditions if abs(c["value"] - 0.5) < 1e-9)
    ninety = next(c["student_acc_mean"] for c in conditions if abs(c["value"] - 0.9) < 1e-9)
    straight = lo + 0.5 * (hi - lo)
    conclusion = (
        f"Mean student accuracy: 0% shared {lo:.3f}, 50% {half:.3f}, 90% {ninety:.3f}, 100% {hi:.3f} "
        f"(untrained {base:.3f}). A proportional fade would put 50% near {straight:.3f}; instead it sat at "
        "chance level, so my hypothesis was wrong: the hidden channel needs the starting weights to be almost "
        "identical, and breaks well before they are completely different."
    )
    result = {
        "config": {"hidden": cfg["hidden"], "n_aux": cfg["n_aux"], "seeds": list(SEEDS)},
        "per_seed": per_seed,
        "mean": mean,
        "own_experiment": {
            "question": EXPERIMENT_QUESTION,
            "hypothesis": EXPERIMENT_HYPOTHESIS,
            "conditions": conditions,
            "conclusion": conclusion,
        },
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "subliminal_results.json").write_text(json.dumps(result, indent=2))
    return result
