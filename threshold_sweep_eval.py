# threshold_sweep_eval.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# sklearn for metrics
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score
)

def parse_thresholds(args) -> List[float]:
    if args.thresholds:
        return [float(x) for x in args.thresholds.split(",")]
    # fallback to range definition
    ths = []
    t = args.start
    while t <= args.stop + 1e-12:
        ths.append(round(t, 10))
        t += args.step
    return ths

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def compute_metrics(y_true: Optional[np.ndarray], y_pred: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    out = {}
    if y_true is None:
        return out
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out.update(dict(accuracy=acc, precision=pr, recall=rc, f1=f1))
    # probabilistic metrics (might fail if only one class in y_true)
    try:
        out["auroc"] = roc_auc_score(y_true, prob)
    except Exception:
        pass
    try:
        out["auprc"] = average_precision_score(y_true, prob)
    except Exception:
        pass
    return out

def plot_prob_gt_pred(
    start_s: np.ndarray,
    prob: np.ndarray,
    label: Optional[np.ndarray],
    pred: np.ndarray,
    thr: float,
    out_png: Path,
    title: Optional[str] = None
):
    t = np.asarray(start_s, float)
    p = np.asarray(prob, float) * 100.0
    pred_pct = np.asarray(pred, int) * 100.0
    lab_pct = (np.asarray(label, int) * 100.0) if label is not None else None

    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()

    # probability curve
    ax.plot(t, p, linewidth=1.5, label="FoG probability (%)")
    # threshold
    ax.axhline(thr * 100.0, linestyle="--", linewidth=1.2, label=f"Threshold = {thr*100:.0f}%")
    # ground truth
    if lab_pct is not None:
        ax.step(t, lab_pct, where="post", linewidth=1.2, label="Ground truth (×100)")
    # predicted
    ax.step(t, pred_pct, where="post", linewidth=1.2, label="Predicted (×100)", alpha=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Percentage / 0–100")
    if len(t):
        ax.set_xlim(t.min(), t.max() + 2.0)
    ax.set_ylim(0, 100)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

    fig.tight_layout()
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def load_pred_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]:
    """Returns (start_s, prob, label_or_None, title_stem)"""
    df = pd.read_csv(csv_path)
    required = {"fog_prob", "start_s"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} missing required columns {required}")
    start = df["start_s"].values
    prob = df["fog_prob"].values.astype(float)
    label = df["label"].values.astype(int) if "label" in df.columns else None
    title = csv_path.stem
    return start, prob, label, title

def main():
    ap = argparse.ArgumentParser(description="Threshold sweep over prediction CSVs; metrics & GT-inclusive plots.")
    ap.add_argument("--results_dir", required=True, help="Folder with *.pred.csv from inference")
    ap.add_argument("--out_dir", required=True, help="Folder to write sweep outputs (plots + csv summaries)")
    ap.add_argument("--thresholds", default="", help='Comma list, e.g. "0.20,0.30,0.40,0.50,0.60"')
    ap.add_argument("--start", type=float, default=0.20)
    ap.add_argument("--stop", type=float, default=0.60)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--file_glob", default="*.pred.csv", help="Pattern of prediction files to include")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    thresholds = parse_thresholds(args)
    pred_files = sorted(results_dir.glob(args.file_glob))
    if not pred_files:
        raise SystemExit(f"No files match {results_dir / args.file_glob}")

    # tables to accumulate metrics
    rows = []
    per_file_best: Dict[str, Dict] = {}
    overall_counts_by_thr: Dict[float, int] = {}
    overall_scores_by_thr: Dict[float, Dict[str, float]] = {}

    for csv_path in pred_files:
        start, prob, label, title = load_pred_csv(csv_path)
        n = len(prob)
        if n == 0:
            continue

        # per-file sweep
        best_for_file = {"f1": -1.0, "threshold": None, "row": None}
        for thr in thresholds:
            pred = (prob >= thr).astype(int)

            metrics = compute_metrics(label, pred, prob)
            row = {
                "file": csv_path.name,
                "threshold": thr,
                "n_windows": n,
                **metrics
            }
            rows.append(row)

            # save plot with GT + Pred + Prob + Threshold
            plot_name = out_dir / "plots" / f"{csv_path.stem}.thr{thr:.2f}.png"
            plot_prob_gt_pred(start, prob, label, pred, thr, plot_name, title=f"{title} (thr={thr:.2f})")

            # track best per file (by F1 of positive class)
            if metrics and metrics.get("f1", -1.0) > best_for_file["f1"]:
                best_for_file = {"f1": metrics["f1"], "threshold": thr, "row": row}

            # accumulate for overall averages (weighted by windows)
            if metrics:
                overall_counts_by_thr[thr] = overall_counts_by_thr.get(thr, 0) + n
                agg = overall_scores_by_thr.get(thr, {})
                # sum weighted by count; normalize later
                for k in ["accuracy", "precision", "recall", "f1"]:
                    if k in metrics:
                        agg[k] = agg.get(k, 0.0) + metrics[k] * n
                overall_scores_by_thr[thr] = agg

        if best_for_file["row"]:
            per_file_best[csv_path.name] = best_for_file["row"]

    # write per-file sweep table
    df_all = pd.DataFrame(rows)
    df_all.to_csv(out_dir / "per_file_by_threshold.csv", index=False)

    # write per-file best (by F1)
    df_best_files = pd.DataFrame(list(per_file_best.values()))
    if not df_best_files.empty:
        df_best_files.to_csv(out_dir / "per_file_best_threshold_by_f1.csv", index=False)

    # compute overall weighted averages per threshold and pick best
    overall_rows = []
    best_overall = {"f1": -1.0, "threshold": None, "row": None}
    for thr in thresholds:
        n_total = overall_counts_by_thr.get(thr, 0)
        if n_total == 0:
            continue
        sums = overall_scores_by_thr[thr]
        row = {
            "threshold": thr,
            "n_windows_total": n_total,
            "accuracy": sums.get("accuracy", 0.0) / n_total,
            "precision": sums.get("precision", 0.0) / n_total,
            "recall": sums.get("recall", 0.0) / n_total,
            "f1": sums.get("f1", 0.0) / n_total
        }
        overall_rows.append(row)
        if row["f1"] > best_overall["f1"]:
            best_overall = {"f1": row["f1"], "threshold": thr, "row": row}

    df_overall = pd.DataFrame(overall_rows)
    if not df_overall.empty:
        df_overall.sort_values("threshold").to_csv(out_dir / "overall_weighted_by_threshold.csv", index=False)

    # write a tiny text summary
    summary_txt = out_dir / "BEST_THRESHOLD.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        if best_overall["row"]:
            r = best_overall["row"]
            f.write(
                f"Best overall threshold by weighted F1:\n"
                f"  threshold = {r['threshold']:.2f}\n"
                f"  F1        = {r['f1']:.4f}\n"
                f"  Precision = {r['precision']:.4f}\n"
                f"  Recall    = {r['recall']:.4f}\n"
                f"  Accuracy  = {r['accuracy']:.4f}\n"
                f"  n_windows = {int(r['n_windows_total'])}\n"
            )
        else:
            f.write("Could not compute an overall best threshold (no labels found?).\n")

    print("Wrote:")
    print(" -", out_dir / "per_file_by_threshold.csv")
    print(" -", out_dir / "per_file_best_threshold_by_f1.csv")
    print(" -", out_dir / "overall_weighted_by_threshold.csv")
    print(" -", out_dir / "BEST_THRESHOLD.txt")
    print(" - plots in", out_dir / "plots")
