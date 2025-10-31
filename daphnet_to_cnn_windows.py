#!/usr/bin/env python3
"""
Convert Daphnet-style accelerometer streams (tri-axial) into
2-second CNN input windows at 40 Hz (80x3 per window).

Outputs:
- .npz with arrays:
    X: (num_windows, 80, 3) float32
    y: (num_windows,) int8   [optional, if labels available]
- manifest.csv mapping each window to source file and time interval.

Handles:
- CSV with columns (case-insensitive): time_ms/time_s, accx, accy, accz, [fog]
- Unit handling: --units g | ms2
- Resampling from fs_in (default 64) to 40 Hz using polyphase
- Sliding window: default 2.0 s window, 0.5 s stride
- Normalization (default: per-window z-score)
- Python 3.13-safe resampling ratio
"""

import argparse
import math
from pathlib import Path
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------- utils ----------------

def _find_col(df, names):
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None

def _safe_fraction_ratio(num, den, max_den=1000):
    """
    Build a rational approximation of num/den that is safe on Python 3.13.
    Returns (up, down) as integers.
    """
    f_num = Fraction(str(float(num))).limit_denominator(max_den)
    f_den = Fraction(str(float(den))).limit_denominator(max_den)
    frac = (f_num / f_den).limit_denominator(max_den)
    return frac.numerator, frac.denominator

def _infer_fs_from_time(t_sec):
    """Infer sampling rate from time vector in seconds (robust median dt)."""
    if t_sec is None or len(t_sec) < 3:
        return None
    dt = np.diff(t_sec)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return None
    med = float(np.median(dt))
    if med <= 0:
        return None
    return 1.0 / med

# --------------- I/O -------------------

def load_csv(path):
    df = pd.read_csv(path)
    # Flexible column detection
    t_col = _find_col(df, ["time_ms", "time", "timestamp_ms", "time_s", "timestamp_s"])
    x_col = _find_col(df, ["accx", "ax", "x"])
    y_col = _find_col(df, ["accy", "ay", "y"])
    z_col = _find_col(df, ["accz", "az", "z"])
    fog_col = _find_col(df, ["fog", "label", "is_fog", "y"])

    # If this doesn't look like a sensor CSV, return sentinel to let caller skip it
    if x_col is None or y_col is None or z_col is None:
        return None, None, None

    # Time in seconds (float)
    if t_col is None:
        t = None
    else:
        t_raw = df[t_col].astype(float).values
        if "ms" in t_col.lower():
            t = t_raw / 1000.0
        elif t_col.lower().endswith("_s") or "time_s" in t_col.lower():
            t = t_raw
        else:
            t = t_raw / 1000.0 if np.nanmedian(t_raw) > 1000 else t_raw

    X = df[[x_col, y_col, z_col]].astype(float).values
    y = df[fog_col].astype(int).values if fog_col else None
    return t, X, y

def zscore_per_window(w):
    mu = np.nanmean(w, axis=0, keepdims=True)
    sigma = np.nanstd(w, axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (w - mu) / sigma

# ------------- core logic --------------

def convert_stream(
    t, X,
    fs_in=64.0,
    fs_out=40.0,
    window_s=2.0,
    stride_s=0.5,
    units="ms2",
    g_value=9.80665,
    normalize="per_window",
):
    """
    t: np.ndarray | None   (seconds). If None, assume uniform fs_in.
    X: (N,3)
    Returns:
        Xw: (num_windows, Wout, 3)
        idx_ranges: list of (start_index_out, end_index_out)
        t_res: resampled time (seconds)
    """
    X = np.asarray(X, dtype=np.float64)

    # Units conversion
    if units.lower() == "g":
        X = X * g_value
    elif units.lower() == "ms2":
        pass
    else:
        raise ValueError("--units must be 'g' or 'ms2'")

    # If no time, construct
    if t is None:
        n = X.shape[0]
        t = np.arange(n) / float(fs_in)
    else:
        t = np.asarray(t, dtype=np.float64)

    # Resample to fs_out
    up, down = _safe_fraction_ratio(fs_out, fs_in, max_den=1000)
    X_res = resample_poly(X, up=up, down=down, axis=0)
    t_res = np.linspace(t[0], t[-1], num=X_res.shape[0], endpoint=True)

    W = int(round(window_s * fs_out))   # 2.0 * 40 = 80
    S = max(1, int(round(stride_s * fs_out)))

    windows = []
    idx_ranges = []
    for start in range(0, len(X_res) - W + 1, S):
        seg = X_res[start:start+W, :]
        if normalize == "per_window":
            seg = zscore_per_window(seg)
        elif normalize == "none":
            seg = seg.astype(np.float32)
        else:
            raise ValueError("--normalize must be 'per_window' or 'none'")
        windows.append(seg.astype(np.float32))
        idx_ranges.append((start, start+W))

    if len(windows) == 0:
        raise ValueError("No windows produced; check input length and parameters.")

    Xw = np.stack(windows, axis=0)
    return Xw, idx_ranges, t_res

def attach_labels_to_windows(t_res, idx_ranges, y_samples, majority=True):
    if y_samples is None:
        return None
    y_window = []
    for a, b in idx_ranges:
        seg = y_samples[a:b]
        if seg.size == 0:
            y_window.append(0)
        else:
            if majority:
                y_window.append(1 if (np.sum(seg >= 1) / seg.size) >= 0.5 else 0)
            else:
                y_window.append(1 if np.any(seg >= 1) else 0)
    return np.asarray(y_window, dtype=np.int8)

# --------------- CLI -------------------

def main():
    p = argparse.ArgumentParser(description="Convert Daphnet-style tri-ax accelerometer to 2s 40Hz CNN windows.")
    p.add_argument("--input", required=True, help="Input CSV with columns: time_ms|time_s, accX, accY, accZ, [fog]")
    p.add_argument("--output_npz", required=True, help="Output .npz path")
    p.add_argument("--manifest_csv", required=True, help="Output manifest CSV path")
    p.add_argument("--fs_in", type=float, default=64.0, help="Input sampling rate (Hz). Default: 64")
    p.add_argument("--fs_out", type=float, default=40.0, help="Output sampling rate (Hz). Default: 40")
    p.add_argument("--window_s", type=float, default=2.0, help="Window size seconds. Default: 2.0")
    p.add_argument("--stride_s", type=float, default=0.5, help="Stride seconds. Default: 0.5")
    p.add_argument("--units", choices=["g", "ms2"], default="ms2", help="Input units of accel. Default: ms2")
    p.add_argument("--normalize", choices=["per_window", "none"], default="per_window",
                   help="Per-window z-score (default) or none.")
    p.add_argument("--label_majority", action="store_true",
                   help="If set and labels exist, label a window as 1 if >=50% samples are 1 (else any-positive).")
    args = p.parse_args()

    # Load
    t, X, y = load_csv(args.input)
    if X is None:
        print(f"[SKIP] '{args.input}' doesn't have accX/accY/accZ — likely a manifest or non-sensor CSV.")
        return

    # If time present, consider inferring fs_in unless user forced it
    fs_in_eff = args.fs_in
    if t is not None and args.fs_in is not None:
        inferred = _infer_fs_from_time(np.asarray(t, dtype=float))
        if inferred and 10.0 <= inferred <= 1000.0 and abs(inferred - args.fs_in) > 1e-3:
            # prefer explicit user value; but warn for awareness
            print(f"[INFO] Detected fs_in≈{inferred:.3f} Hz (using --fs_in {args.fs_in} Hz).")

    # Build windows
    Xw, idx_ranges, t_res = convert_stream(
        t, X,
        fs_in=fs_in_eff,
        fs_out=args.fs_out,
        window_s=args.window_s,
        stride_s=args.stride_s,
        units=args.units,
        normalize=args.normalize,
    )

    # Labels: align by nearest time sample after resampling
    yw = None
    if y is not None:
        idx_nearest = np.searchsorted(
            np.linspace(t_res[0], t_res[-1], num=len(y), endpoint=True),
            t_res, side="left"
        )
        idx_nearest = np.clip(idx_nearest, 0, len(y)-1)
        y_res = y[idx_nearest]
        yw = attach_labels_to_windows(t_res, idx_ranges, y_res, majority=args.label_majority)

    # Save
    np.savez_compressed(args.output_npz, X=Xw.astype(np.float32), **({"y": yw} if yw is not None else {}))

    starts = [t_res[a] for (a, b) in idx_ranges]
    ends   = [t_res[b-1] for (a, b) in idx_ranges]
    manifest = pd.DataFrame({
        "window_index": np.arange(len(idx_ranges), dtype=int),
        "start_s": starts,
        "end_s": ends,
        "num_samples": [int(round(args.window_s * args.fs_out))]*len(idx_ranges),
    })
    if yw is not None:
        manifest["label"] = yw.astype(int)
    Path(args.manifest_csv).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest_csv, index=False)

    print(f"Saved {Xw.shape[0]} windows to {args.output_npz}")
    print(f"Manifest written to {args.manifest_csv}")
    
    

if __name__ == "__main__":
    main()
