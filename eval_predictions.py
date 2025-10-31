# eval_predictions.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_confusion_plot(cm, labels, out_png):
    import itertools
    fig = plt.figure(figsize=(4.5,4))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def save_prob_vs_label_plot(start_s, prob, label, thr, out_png, title=None):
    t = np.asarray(start_s, float)
    p = np.asarray(prob, float) * 100.0
    fig = plt.figure(figsize=(12,4))
    ax = plt.gca()
    ax.plot(t, p, linewidth=1.5, label="FoG probability (%)")
    ax.axhline(thr*100.0, linestyle="--", linewidth=1.2, label=f"Threshold = {thr*100:.0f}%")
    if label is not None:
        lab_pct = np.asarray(label, float) * 100.0
        ax.step(t, lab_pct, where="post", linewidth=1.2, label="Ground truth (x100)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Percentage / 0–100")
    if len(t): ax.set_xlim(t.min(), t.max()+2.0)
    ax.set_ylim(0, 100)
    if title: ax.set_title(title)
    ax.legend(loc="upper right"); ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_csv", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--threshold", type=float, default=0.85)
    args = ap.parse_args()

    df = pd.read_csv(args.predictions_csv)
    if not {"fog_prob","pred","start_s"}.issubset(df.columns):
        raise ValueError("predictions_csv must have fog_prob, pred, start_s columns")
    y_true = df["label"].values if "label" in df.columns else None
    y_pred = df["pred"].values
    prob   = df["fog_prob"].values
    start  = df["start_s"].values

    # metrics
    metrics = {}
    if y_true is not None:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, average_precision_score
        acc = accuracy_score(y_true, y_pred)
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        metrics.update(dict(accuracy=acc, precision=pr, recall=rc, f1=f1))
        # curves (probability needed)
        try:
            metrics["auroc"] = roc_auc_score(y_true, prob)
        except Exception:
            pass
        try:
            metrics["auprc"] = average_precision_score(y_true, prob)
        except Exception:
            pass
        # confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        save_confusion_plot(cm, ["No-FOG","FOG"], args.out_prefix + ".confusion.png")
        pd.DataFrame(cm, index=["True_0","True_1"], columns=["Pred_0","Pred_1"]).to_csv(args.out_prefix + ".confusion.csv")

    # save prob vs label overlay
    save_prob_vs_label_plot(start, prob, (y_true if y_true is not None else None), args.threshold,
                            args.out_prefix + ".prob_vs_label.png",
                            title=Path(args.predictions_csv).stem)

    # save metrics table
    if metrics:
        pd.DataFrame([metrics]).to_csv(args.out_prefix + ".metrics.csv", index=False)
        print("Saved metrics:", metrics)
    else:
        print("No ground truth labels found; saved probability overlay only.")

if __name__ == "__main__":
    main()
