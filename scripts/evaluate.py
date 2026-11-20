"""Choose the decision threshold on VALIDATION, then evaluate ONCE on TEST.

Outputs (results/):
  decision.json          threshold chosen on val (also copied into the model dir)
  metrics.json           test precision/recall/F1 for the profanity class, confusion
                         matrix, per-domain numbers, word-level span metrics,
                         TF-IDF baseline for comparison
  confusion_matrix.png   test confusion matrix
  pr_curve.png           precision-recall curve on val (threshold marked) and test
  test_predictions.csv   every test comment with score / prediction / found words
  errors.csv             all test false positives and false negatives

Usage:  python scripts/evaluate.py [--model models/kazakh-profanity-gliner]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kzprof.inference import ProfanityDetector  # noqa: E402

RESULTS = ROOT / "results"
SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def best_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Threshold in [0.05, 0.95] with the highest F1; ties -> the higher threshold
    (fewer false positives at the same F1)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.round(np.arange(0.05, 0.951, 0.01), 2):
        p, r, f1, _ = precision_recall_fscore_support(y, scores >= t, average="binary", zero_division=0)
        if f1 >= best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def binary_report(y: np.ndarray, pred: np.ndarray) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "positives": int(y.sum()), "precision": round(float(p), 4),
            "recall": round(float(r), 4), "f1": round(float(f1), 4),
            "accuracy": round(float((tp + tn) / len(y)), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def span_report(rows: list[dict], preds: list[dict]) -> dict:
    """Word-level: a predicted span counts as correct if it overlaps a gold span."""
    tp = fp = fn = 0
    for r, p in zip(rows, preds):
        gold = [tuple(s) for s in r["spans"]]
        found = [(s["start"], s["end"]) for s in p["spans"]]
        hit_gold = set()
        for fs, fe in found:
            match = [g for g in gold if fs < g[1] and fe > g[0]]
            if match:
                tp += 1
                hit_gold.update(match)
            else:
                fp += 1
        fn += len([g for g in gold if g not in hit_gold])
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(rc, 4), "f1": round(f1, 4)}


def tfidf_baseline(train: list[dict], val: list[dict], test: list[dict]) -> dict:
    """Character n-gram TF-IDF + logistic regression, threshold also tuned on val.
    Vocabulary is fitted on TRAIN only."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True)
    xtr = vec.fit_transform([r["text"] for r in train])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    clf.fit(xtr, [r["label"] for r in train])
    sv = clf.predict_proba(vec.transform([r["text"] for r in val]))[:, 1]
    t, _ = best_threshold(np.array([r["label"] for r in val]), sv)
    st = clf.predict_proba(vec.transform([r["text"] for r in test]))[:, 1]
    yt = np.array([r["label"] for r in test])
    return {"threshold": t, **binary_report(yt, st >= t), "pr_auc": round(float(average_precision_score(yt, st)), 4)}


def plot_confusion(cm: dict, path: Path) -> None:
    m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=150)
    ax.imshow(m, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred 0\n(clean)", "pred 1\n(profane)"])
    ax.set_yticks([0, 1], ["true 0\n(clean)", "true 1\n(profane)"])
    ax.set_title("Test confusion matrix")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pr(curves: dict, threshold: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    for name, (y, s) in curves.items():
        p, r, t = precision_recall_curve(y, s)
        ax.plot(r, p, label=f"{name} (AP={average_precision_score(y, s):.3f})")
        if name == "val":
            k = int(np.argmin(np.abs(t - threshold)))
            ax.scatter([r[k]], [p[k]], color="red", zorder=5, label=f"threshold={threshold}")
    ax.set_xlabel("recall (profanity)")
    ax.set_ylabel("precision (profanity)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "kazakh-profanity-gliner"))
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    data = ROOT / "data" / "processed"
    train, val, test = (load_jsonl(data / f"{s}.jsonl") for s in ("train", "val", "test"))

    det = ProfanityDetector(args.model, threshold=0.5)

    # 1) threshold on validation only
    val_pred = det.predict_batch([r["text"] for r in val])
    yv = np.array([r["label"] for r in val])
    sv = np.array([p["score"] for p in val_pred])
    threshold, val_f1 = best_threshold(yv, sv)
    decision = {"threshold": threshold, "val_f1_at_threshold": round(val_f1, 4), "chosen_on": "validation"}
    (RESULTS / "decision.json").write_text(json.dumps(decision, indent=2))
    shutil.copy(RESULTS / "decision.json", Path(args.model) / "decision.json")
    print("threshold chosen on val:", decision)

    # 2) single evaluation on test
    det.threshold = threshold
    test_pred = det.predict_batch([r["text"] for r in test])
    yt = np.array([r["label"] for r in test])
    st = np.array([p["score"] for p in test_pred])
    pt = (st >= threshold).astype(int)

    metrics = {"model": str(Path(args.model).name), "threshold": threshold,
               "test": binary_report(yt, pt), "test_pr_auc": round(float(average_precision_score(yt, st)), 4),
               "val_at_threshold": binary_report(yv, (sv >= threshold).astype(int)),
               "test_by_domain": {}, "test_word_level": span_report(test, test_pred)}
    for domain in sorted({r["domain"] for r in test}):
        idx = np.array([r["domain"] == domain for r in test])
        metrics["test_by_domain"][domain] = binary_report(yt[idx], pt[idx])
    hard = np.array([bool(r["hard_negative"]) for r in test])
    metrics["test_hard_negatives"] = {"n": int(hard.sum()), "false_positives": int(pt[hard].sum()),
                                      "false_positive_rate": round(float(pt[hard].mean()), 4) if hard.any() else None}
    metrics["baseline_tfidf_char_lr"] = tfidf_baseline(train, val, test)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    plot_confusion(metrics["test"]["confusion_matrix"], RESULTS / "confusion_matrix.png")
    plot_pr({"val": (yv, sv), "test": (yt, st)}, threshold, RESULTS / "pr_curve.png")

    fields = ["id", "domain", "stars", "label", "pred", "score", "found_words", "gold_words", "text"]
    with (RESULTS / "test_predictions.csv").open("w", newline="", encoding="utf-8") as fh, \
         (RESULTS / "errors.csv").open("w", newline="", encoding="utf-8") as eh:
        w, ew = csv.DictWriter(fh, fields), csv.DictWriter(eh, ["error"] + fields)
        w.writeheader()
        ew.writeheader()
        for r, p in sorted(zip(test, test_pred), key=lambda t: -t[1]["score"]):
            row = {"id": r["id"], "domain": r["domain"], "stars": r["stars"], "label": r["label"],
                   "pred": p["label"], "score": p["score"],
                   "found_words": " | ".join(s["text"] for s in p["spans"]),
                   "gold_words": " | ".join(r["text"][s:e] for s, e in r["spans"]), "text": r["text"]}
            w.writerow(row)
            if row["label"] != row["pred"]:
                ew.writerow({"error": "FP" if row["pred"] else "FN", **row})
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
