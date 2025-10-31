#!/usr/bin/env python3
"""
Inference for FoG CNN on 2s@40Hz windows, with GT+Pred overlay plotting.

Robust model loader:
1) Try load_model() for full SavedModel / full H5 / .keras.
2) If that fails and H5 is weights-only, try:
   a) If --model_json works on your Keras, use it.
   b) Otherwise, FALL BACK to rebuilding the exact triple-branch architecture
      (as in your model.json) in code, then load weights by name.

Also handles models with 1 input (X) or 3 inputs ([X, X, X]).

Outputs:
- predictions_csv (per-window proba + labels if present)
- intervals_csv   (merged FoG intervals)
- plot_png        (probability timeline with threshold + GT + Pred + shaded intervals)
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import load_model, model_from_json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- plotting & intervals ----------

def merge_intervals(times, preds, window_len_s=2.0, stride_s=0.5, threshold=0.5,
                    merge_gap_s=0.0, min_duration_s=0.0):
    calls = (preds >= threshold).astype(int)
    intervals, curr_start = [], None
    probs = preds

    def push_interval(start_idx, end_idx):
        if start_idx is None or end_idx is None:
            return
        start_t = times[start_idx]
        end_t   = times[end_idx] + window_len_s
        dur     = end_t - start_t
        if dur >= min_duration_s:
            mean_prob = float(np.mean(probs[start_idx:end_idx+1]))
            intervals.append((start_t, end_t, dur, mean_prob))

    for i, c in enumerate(calls):
        if c == 1 and curr_start is None:
            curr_start = i
        elif c == 0 and curr_start is not None:
            push_interval(curr_start, i-1)
            curr_start = None
    if curr_start is not None:
        push_interval(curr_start, len(calls)-1)

    if merge_gap_s > 0 and intervals:
        merged = []
        s0,e0,d0,p0 = intervals[0]
        for (s1,e1,d1,p1) in intervals[1:]:
            if (s1 - e0) <= merge_gap_s:
                e0 = e1
                d0 = e0 - s0
                p0 = (p0 + p1) / 2.0
            else:
                merged.append((s0,e0,d0,p0))
                s0,e0,d0,p0 = s1,e1,d1,p1
        merged.append((s0,e0,d0,p0))
        intervals = merged

    return pd.DataFrame(intervals, columns=["start_s","end_s","duration_s","mean_prob"])


def save_probability_plot(times_start_s,
                          fog_prob,
                          threshold,
                          intervals_df,
                          png_path,
                          title=None,
                          label=None,
                          pred=None):
    times = np.asarray(times_start_s, float)
    probs_pct = np.asarray(fog_prob, float) * 100.0
    thr_pct = threshold * 100.0

    fig = plt.figure(figsize=(12,4.8))
    ax = plt.gca()

    # probability curve
    ax.plot(times, probs_pct, linewidth=1.5, label="FoG probability (%)")

    # threshold line
    ax.axhline(thr_pct, linestyle="--", linewidth=1.2, label=f"Threshold = {thr_pct:.0f}%")

    if pred is not None:
        pred_pct = (np.asarray(pred, int) * 100.0)
        ax.step(times, pred_pct, where="post", linewidth=1.2, alpha=0.85, label="Predicted (×100)")

    # shaded merged intervals (pred-based)
    if intervals_df is not None and len(intervals_df) > 0:
        for _, row in intervals_df.iterrows():
            ax.axvspan(float(row["start_s"]), float(row["end_s"]), alpha=0.18)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Percentage / 0–100")
    if len(times):
        ax.set_xlim(times.min(), times.max()+2.0)
    ax.set_ylim(0,100)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


# ---------- fallback: rebuild your triple-branch CNN ----------

def build_triple_branch_model(input_shape=(80, 3)):
    """
    Exact architecture from your JSON:
      Inputs: small, medium, large (each (80,3))
      small : Conv1D(18) -> Dropout -> MaxPool(3) -> Conv1D(9)  -> Dropout -> MaxPool(3) -> Flatten
      medium: Conv1D(12) -> Dropout -> MaxPool(3) -> Conv1D(6)  -> Dropout -> MaxPool(3) -> Flatten
      large : Conv1D(6)  -> Dropout -> MaxPool(3) -> Conv1D(3)  -> Dropout -> MaxPool(3) -> Flatten
      concat -> Dense(16, relu) -> Dropout -> Dense(1, sigmoid, L2=0.001)
    Layer names match the JSON exactly so weights load by name.
    """
    l2 = regularizers.l2(0.001)

    # Inputs
    inp_small  = layers.Input(shape=input_shape, name="small")
    inp_medium = layers.Input(shape=input_shape, name="medium")
    inp_large  = layers.Input(shape=input_shape, name="large")

    # ----- small branch -----
    xS = layers.Conv1D(16, 18, activation="relu", padding="valid", name="conv1d")(inp_small)
    xS = layers.Dropout(0.5, name="dropout")(xS)
    xS = layers.MaxPooling1D(pool_size=3, strides=3, padding="valid", name="max_pooling1d")(xS)

    xS = layers.Conv1D(16, 9, activation="relu", padding="valid", name="conv1d_1")(xS)
    xS = layers.Dropout(0.5, name="dropout_1")(xS)
    xS = layers.MaxPooling1D(pool_size=3, strides=3, padding="valid", name="max_pooling1d_1")(xS)

    xS = layers.Flatten(name="flatten")(xS)

    # ----- medium branch -----
    xM = layers.Conv1D(16, 12, activation="relu", padding="valid", name="conv1d_2")(inp_medium)
    xM = layers.Dropout(0.5, name="dropout_2")(xM)
    xM = layers.MaxPooling1D(pool_size=3, strides=3, padding="valid", name="max_pooling1d_2")(xM)

    xM = layers.Conv1D(16, 6, activation="relu", padding="valid", name="conv1d_3")(xM)
    xM = layers.Dropout(0.5, name="dropout_3")(xM)
    xM = layers.MaxPooling1D(pool_size=3, strides=3, padding="valid", name="max_pooling1d_3")(xM)

    xM = layers.Flatten(name="flatten_1")(xM)

    # ----- large branch -----
    xL = layers.Conv1D(16, 6, activation="relu", padding="valid", name="conv1d_4")(inp_large)
    xL = layers.Dropout(0.5, name="dropout_4")(xL)
    xL = layers.MaxPooling1D(pool_size=3, strides=3, padding="valid", name="max_pooling1d_4")(xL)

    xL = layers.Conv1D(16, 3, activation="relu", padding="valid", name="conv1d_5")(xL)
    xL = layers.Dropout(0.5, name="dropout_5")(xL)
    xL = layers.MaxPooling1D(pool_size=3, strides=3, padding="valid", name="max_pooling1d_5")(xL)

    xL = layers.Flatten(name="flatten_2")(xL)

    # concat + head
    x = layers.Concatenate(name="concatenate")([xS, xM, xL])  # 272 features total
    x = layers.Dense(16, activation="relu", name="dense")(x)
    x = layers.Dropout(0.5, name="dropout_6")(x)
    out = layers.Dense(1, activation="sigmoid", kernel_regularizer=l2, name="dense_1")(x)

    return models.Model(inputs=[inp_small, inp_medium, inp_large], outputs=out, name="model")


def try_load_weights_only(h5_path: Path, json_path: Path | None):
    """
    Strategy:
    - If legacy JSON can be deserialized, use that.
    - If not, rebuild the known triple-branch architecture and load weights by name.
    """
    arch = None
    if json_path and json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                arch = f.read()
            return model_from_json(arch)
        except Exception:
            arch = None  # fall through to manual rebuild

    # Manual rebuild fallback
    model = build_triple_branch_model(input_shape=(80,3))
    model.load_weights(str(h5_path), by_name=True, skip_mismatch=True)
    return model


def load_any_model(model_path: Path, model_json: Path | None):
    mp = str(model_path)
    try:
        return load_model(mp)
    except Exception as e:
        msg = str(e)
        if "No model config found" in msg or "does not appear to be a model" in msg:
            return try_load_weights_only(model_path, model_json)
        raise


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(description="Run pretrained CNN on Daphnet-style windows and plot confidence.")
    p.add_argument("--windows_npz", required=True)
    p.add_argument("--manifest_csv", required=True)
    p.add_argument("--model_path", required=True, help="H5/.keras/SavedModel OR weights-only H5")
    p.add_argument("--model_json", default=None, help="(optional) JSON architecture; ignored if not needed")

    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--window_s", type=float, default=2.0)
    p.add_argument("--stride_s", type=float, default=0.5)
    p.add_argument("--merge_gap_s", type=float, default=0.0)
    p.add_argument("--min_duration_s", type=float, default=0.0)

    p.add_argument("--predictions_csv", required=True)
    p.add_argument("--intervals_csv", required=True)

    p.add_argument("--plot_png", default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    data = np.load(args.windows_npz)
    if "X" not in data:
        raise ValueError("NPZ missing 'X'")
    X = data["X"]  # (N,80,3)
    y = data["y"] if "y" in data else None

    model = load_any_model(Path(args.model_path), Path(args.model_json) if args.model_json else None)

    # Handle multi-input models (3 inputs). If single input, pass X.
    if (isinstance(model.input_shape, list) or isinstance(model.inputs, list)) and len(model.inputs) == 3:
        X_in = [X, X, X]
    else:
        X_in = X

    proba = model.predict(X_in, verbose=0)
    proba = np.asarray(proba).squeeze()
    if proba.ndim == 1:
        fog_prob = proba
    elif proba.ndim == 2 and proba.shape[1] == 1:
        fog_prob = proba[:,0]
    elif proba.ndim == 2 and proba.shape[1] == 2:
        fog_prob = proba[:,1]
    else:
        raise ValueError(f"Unexpected model output shape {proba.shape}")

    pred_label = (fog_prob >= args.threshold).astype(int)

    manifest = pd.read_csv(args.manifest_csv)
    for need in ["window_index","start_s","end_s"]:
        if need not in manifest.columns:
            raise ValueError("Manifest CSV missing required columns: window_index, start_s, end_s")
    if len(manifest) != len(fog_prob):
        raise ValueError(f"Manifest length ({len(manifest)}) != predictions ({len(fog_prob)})")

    out = manifest.copy()
    out["fog_prob"] = fog_prob
    out["pred"]     = pred_label
    if y is not None and len(y) == len(fog_prob):
        out["label"] = y.astype(int)

    Path(args.predictions_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.predictions_csv, index=False)

    intervals = merge_intervals(
        times=manifest["start_s"].values.astype(float),
        preds=fog_prob,
        window_len_s=args.window_s,
        stride_s=args.stride_s,
        threshold=args.threshold,
        merge_gap_s=args.merge_gap_s,
        min_duration_s=args.min_duration_s,
    )
    Path(args.intervals_csv).parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(args.intervals_csv, index=False)

    print(f"Wrote per-window predictions to {args.predictions_csv}")
    print(f"Wrote merged intervals to {args.intervals_csv}")

    if args.plot_png:
        title = args.title if args.title else Path(args.predictions_csv).stem
        # pass label & pred to overlay on the plot
        save_probability_plot(
            times_start_s=manifest["start_s"].values.astype(float),
            fog_prob=fog_prob,
            threshold=args.threshold,
            intervals_df=intervals,
            png_path=args.plot_png,
            title=title,
            label=out["label"].values if "label" in out.columns else None,
            pred=out["pred"].values
        )
        print(f"Wrote probability plot to {args.plot_png}")

    # Optional quick metrics
    if "label" in out.columns:
        try:
            from sklearn.metrics import classification_report
            print("\nQuick metrics vs. labels (windowed):")
            print(classification_report(out["label"].values, out["pred"].values, digits=4))
        except Exception:
            pass


if __name__ == "__main__":
    main()
