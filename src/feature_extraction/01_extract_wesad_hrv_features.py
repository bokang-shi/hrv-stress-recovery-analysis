"""
Extract HRV features from WESAD ECG recordings.

For each subject, baseline and stress ECG segments are split into 5-minute
windows with a 1-minute step. HRV features are calculated per window and then
summarised using the median for each phase.
"""

import argparse
import pickle
import warnings
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd


# NumPy 2.0 compatibility for older NeuroKit2 versions.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

warnings.filterwarnings("ignore")

DEFAULT_SUBJECTS = ("S2", "S3", "S4", "S11", "S13", "S14", "S16", "S17")
FS = 700

WINDOW_SEC = 300
STEP_SEC = 60
MIN_PEAKS = 200

FEATURE_KEYS = {
    "RMSSD": "HRV_RMSSD",
    "SDNN": "HRV_SDNN",
    "pNN50": "HRV_pNN50",
    "LF/HF": "HRV_LFHF",
}


def get_window_hrv(segment, fs):
    """Compute HRV features for one ECG window."""
    seg = np.nan_to_num(np.asarray(segment, dtype=np.float64))

    if np.max(seg) - np.min(seg) < 0.05:
        return None

    try:
        clean = nk.ecg_clean(seg, sampling_rate=fs, method="neurokit")
        _, info = nk.ecg_peaks(clean, sampling_rate=fs)
        r_peaks = info["ECG_R_Peaks"]

        if len(r_peaks) < MIN_PEAKS:
            return None

        hrv_t = nk.hrv_time(info, sampling_rate=fs)
        hrv_f = nk.hrv_frequency(
            info,
            sampling_rate=fs,
            psd_method="lomb",
            normalize=False,
        )

        return pd.concat([hrv_t, hrv_f], axis=1).iloc[0]

    except Exception:
        return None


def process_phase_sliding(signal, fs, phase_name=""):
    window_samples = WINDOW_SEC * fs
    step_samples = STEP_SEC * fs
    total_len = len(signal)

    if total_len < window_samples:
        print(f"[{phase_name}] shorter than 5 minutes; skipped")
        return None

    valid_hrv_list = []
    num_windows = (total_len - window_samples) // step_samples + 1
    print(f"[{phase_name}] windows={num_windows}", end=" | ", flush=True)

    for i in range(0, total_len - window_samples + 1, step_samples):
        segment = signal[i:i + window_samples]
        hrv_res = get_window_hrv(segment, fs)

        if hrv_res is not None:
            valid_hrv_list.append(hrv_res)

    print(f"valid={len(valid_hrv_list)}")

    if not valid_hrv_list:
        return None

    return pd.DataFrame(valid_hrv_list).median()


def add_phase_features(row, feat_series, prefix):
    for key in ("RMSSD", "SDNN", "pNN50", "LF/HF"):
        nk_key = FEATURE_KEYS[key]
        row[f"{prefix}_{key}"] = feat_series.get(nk_key, np.nan)

    lf_ms2 = feat_series.get("HRV_LF", np.nan)
    hf_ms2 = feat_series.get("HRV_HF", np.nan)

    row[f"{prefix}_LF_ms2"] = lf_ms2
    row[f"{prefix}_HF_ms2"] = hf_ms2

    if not np.isnan(lf_ms2) and not np.isnan(hf_ms2) and (lf_ms2 + hf_ms2) > 0:
        total_power = lf_ms2 + hf_ms2
        row[f"{prefix}_LF_nu"] = lf_ms2 / total_power
        row[f"{prefix}_HF_nu"] = hf_ms2 / total_power
    else:
        row[f"{prefix}_LF_nu"] = np.nan
        row[f"{prefix}_HF_nu"] = np.nan


def extract_wesad_hrv(data_dir, output_file, subjects):
    results = []
    print("=== WESAD HRV extraction ===")

    for sid in subjects:
        pkl_path = data_dir / f"{sid}.pkl"
        if not pkl_path.exists():
            print(f"[WARN] Missing {pkl_path}")
            continue

        print(f"\nProcessing {sid}...")
        try:
            with pkl_path.open("rb") as f:
                data = pickle.load(f, encoding="latin1")
        except Exception as exc:
            print(f"[WARN] Could not load {sid}: {exc}")
            continue

        ecg = data["signal"]["chest"]["ECG"].flatten()
        labels = data["label"]

        idx_base = np.where(labels == 1)[0]
        idx_stress = np.where(labels == 2)[0]

        if len(idx_base) == 0 or len(idx_stress) == 0:
            print("  Missing baseline or stress labels; skipped")
            continue

        feat_base = process_phase_sliding(ecg[idx_base[0]:idx_base[-1] + 1], FS, "Base")
        feat_stress = process_phase_sliding(ecg[idx_stress[0]:idx_stress[-1] + 1], FS, "Stress")

        if feat_base is None or feat_stress is None:
            continue

        row = {"Subject": sid}
        add_phase_features(row, feat_base, "Base")
        add_phase_features(row, feat_stress, "Stress")
        results.append(row)

    if not results:
        print("\nNo valid data extracted")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_file, index=False)
    print(f"\nDone. Saved to: {output_file}")
    print(df_out[["Subject", "Base_LF_ms2", "Base_LF_nu", "Base_HF_ms2", "Base_HF_nu"]].head())


def parse_args():
    parser = argparse.ArgumentParser(description="Extract WESAD HRV features from ECG pickle files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/wesad"),
                        help="Folder containing WESAD subject .pkl files.")
    parser.add_argument("--out", type=Path, default=Path("outputs/wesad_hrv_sliding_median_ms2.csv"),
                        help="Output CSV path.")
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS),
                        help="Comma-separated WESAD subject IDs to process.")
    return parser.parse_args()


def main():
    args = parse_args()
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    extract_wesad_hrv(args.data_dir, args.out, subjects)


if __name__ == "__main__":
    main()
