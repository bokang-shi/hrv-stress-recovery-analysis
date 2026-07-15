"""
Summarise RR interval artefacts in WESAD ECG recordings.

The artefact rules follow common HRV preprocessing guidance: remove implausible
heart rates and sudden RR interval jumps.
"""

import argparse
import pickle
import warnings
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd


DEFAULT_SUBJECTS = ("S2", "S3", "S4", "S11", "S13", "S14", "S16", "S17")
FS = 700

ECG_CLEAN_METHOD = "neurokit"
RPEAK_METHOD = "nabian2018"

HR_MIN_BPM = 40
HR_MAX_BPM = 200
RR_CHANGE_FRAC = 0.20

MIN_SEGMENT_SECONDS = 30

OUT_BY_SEGMENT = "wesad_rr_artefacts_by_segment.csv"
OUT_SUMMARY = "wesad_rr_artefacts_summary.csv"

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_wesad_pickle(pkl_path):
    with pkl_path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def get_ecg_and_labels(data):
    """Extract ECG and labels from the standard WESAD structure."""
    ecg = np.asarray(data["signal"]["chest"]["ECG"]).squeeze()
    labels = np.asarray(data["label"]).squeeze()

    if ecg.ndim != 1:
        raise ValueError(f"Unexpected ECG shape: {ecg.shape}")
    if labels.ndim != 1:
        raise ValueError(f"Unexpected label shape: {labels.shape}")
    if len(ecg) != len(labels):
        raise ValueError(f"ECG and labels length mismatch: {len(ecg)} vs {len(labels)}")

    return ecg, labels


def split_into_continuous_label_segments(labels):
    """Split labels into continuous segments."""
    segments = []
    n = len(labels)
    if n == 0:
        return segments

    start = 0
    cur = labels[0]
    for i in range(1, n):
        if labels[i] != cur:
            segments.append((int(cur), start, i))
            start = i
            cur = labels[i]
    segments.append((int(cur), start, n))
    return segments


def artefact_mask_from_rr(rr_s, hr_min=HR_MIN_BPM, hr_max=HR_MAX_BPM, rr_change_frac=RR_CHANGE_FRAC):
    """Flag artefactual RR intervals."""
    rr_s = np.asarray(rr_s, dtype=float)
    bad = np.zeros(len(rr_s), dtype=bool)

    rr_min = 60.0 / hr_max
    rr_max = 60.0 / hr_min
    bad |= (rr_s < rr_min) | (rr_s > rr_max)

    if len(rr_s) >= 2:
        prev = rr_s[:-1]
        cur = rr_s[1:]
        prev_safe = np.where(prev == 0, np.nan, prev)
        rel_change = np.abs(cur - prev) / prev_safe
        jump_bad = rel_change > rr_change_frac
        bad[1:] |= np.nan_to_num(jump_bad, nan=True)

    return bad


def process_ecg_segment(ecg_seg, fs):
    """Clean ECG, detect R-peaks, and compute RR artefacts."""
    ecg_clean = nk.ecg_clean(ecg_seg, sampling_rate=fs, method=ECG_CLEAN_METHOD)
    _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=fs, method=RPEAK_METHOD)
    peak_idx = np.asarray(rpeaks.get("ECG_R_Peaks", []), dtype=int)

    if len(peak_idx) < 3:
        return np.array([]), np.array([], dtype=bool)

    rr_s = np.diff(peak_idx) / float(fs)
    bad = artefact_mask_from_rr(rr_s)
    return rr_s, bad


def summarize_group(g):
    total_rr = g["n_rr"].sum()
    total_bad = g["n_bad"].sum()
    pct_weighted = (total_bad / total_rr * 100.0) if total_rr > 0 else np.nan
    return pd.Series({
        "segments": int(len(g)),
        "duration_s_total": float(g["duration_s"].sum()),
        "total_rr": int(total_rr),
        "total_bad": int(total_bad),
        "pct_bad_weighted": float(pct_weighted) if np.isfinite(pct_weighted) else np.nan,
    })


def analyse_wesad_rr_artefacts(data_dir, output_dir, subjects):
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    rows = []

    for subject in subjects:
        pkl_path = data_dir / f"{subject}.pkl"
        if not pkl_path.is_file():
            print(f"[WARN] File not found: {pkl_path}")
            continue

        print(f"[INFO] Loading {pkl_path}")
        data = load_wesad_pickle(pkl_path)

        ecg, labels = get_ecg_and_labels(data)
        segments = split_into_continuous_label_segments(labels)

        seg_id = 0
        for label_val, start, end in segments:
            duration_s = (end - start) / float(FS)
            if duration_s < MIN_SEGMENT_SECONDS:
                continue

            rr_s, bad = process_ecg_segment(ecg[start:end], FS)

            n_rr = int(len(rr_s))
            n_bad = int(np.sum(bad)) if n_rr > 0 else 0
            pct_bad = (n_bad / n_rr * 100.0) if n_rr > 0 else np.nan

            rows.append({
                "subject": subject,
                "label": int(label_val),
                "segment_id": int(seg_id),
                "start_sample": int(start),
                "end_sample": int(end),
                "duration_s": float(duration_s),
                "n_rr": n_rr,
                "n_bad": n_bad,
                "pct_bad": float(pct_bad) if np.isfinite(pct_bad) else np.nan,
            })
            seg_id += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    out_by_segment = output_dir / OUT_BY_SEGMENT
    df.to_csv(out_by_segment, index=False)
    print(f"[SAVED] {out_by_segment}")

    if len(df) == 0:
        print("[INFO] No valid segments found.")
        return

    summary = (
        df.groupby(["subject", "label"], as_index=False)
        .apply(summarize_group)
        .reset_index(drop=True)
    )

    out_summary = output_dir / OUT_SUMMARY
    summary.to_csv(out_summary, index=False)
    print(f"[SAVED] {out_summary}")

    overall_rr = df["n_rr"].sum()
    overall_bad = df["n_bad"].sum()
    overall_pct = (overall_bad / overall_rr * 100.0) if overall_rr > 0 else np.nan
    print(f"[INFO] Overall artefacts: {overall_bad}/{overall_rr} = {overall_pct:.3f}%")

    subj_sum = (
        df.groupby("subject", as_index=False)
        .apply(lambda g: pd.Series({
            "total_rr": int(g["n_rr"].sum()),
            "total_bad": int(g["n_bad"].sum()),
            "pct_bad_weighted": (g["n_bad"].sum() / g["n_rr"].sum() * 100.0)
            if g["n_rr"].sum() > 0 else np.nan,
        }))
        .reset_index(drop=True)
    )

    print("\n[INFO] Per-subject overall artefact %:")
    print(subj_sum.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Summarise WESAD RR interval artefacts.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/wesad"),
                        help="Folder containing WESAD subject .pkl files.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/validation"),
                        help="Folder for output CSV files.")
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS),
                        help="Comma-separated WESAD subject IDs to process.")
    return parser.parse_args()


def main():
    args = parse_args()
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    analyse_wesad_rr_artefacts(args.data_dir, args.out_dir, subjects)


if __name__ == "__main__":
    main()
