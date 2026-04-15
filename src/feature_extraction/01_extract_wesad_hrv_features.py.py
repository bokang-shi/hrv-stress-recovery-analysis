
"""
Step 1: WESAD HRV extraction
5-min window, 1-min step
Median aggregation
Lomb-Scargle with absolute power (ms^2)
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
import neurokit2 as nk
import warnings

# NumPy 2.0 compatibility
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

warnings.filterwarnings("ignore")

# Settings
DATA_PATH = r"C:\Users\Windows\OneDrive - Imperial College London\Desktop\HRV_local"
SUBJECTS = ['S2', 'S3', 'S4', 'S11', 'S13', 'S14', 'S16', 'S17']
FS = 700
SAVE_FILENAME = "wesad_hrv_sliding_median_ms2.csv"

WINDOW_SEC = 300
STEP_SEC = 60
MIN_PEAKS = 200

FEATURE_KEYS = {
    "RMSSD": "HRV_RMSSD",
    "SDNN": "HRV_SDNN",
    "pNN50": "HRV_pNN50",
    "LF_ms2": "HRV_LF",
    "HF_ms2": "HRV_HF",
    "LF_nu": "HRV_LFnu",
    "HF_nu": "HRV_HFnu",
    "LF/HF": "HRV_LFHF",
}

def get_window_hrv(segment, fs):
    """Compute HRV for one window."""
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
            normalize=False
        )

        return pd.concat([hrv_t, hrv_f], axis=1).iloc[0]

    except Exception:
        return None


def process_phase_sliding(signal, fs, phase_name=""):
    window_samples = WINDOW_SEC * fs
    step_samples = STEP_SEC * fs
    total_len = len(signal)

    if total_len < window_samples:
        print(f"[{phase_name}] shorter than 5 min -> skip")
        return None

    valid_hrv_list = []
    num_windows = (total_len - window_samples) // step_samples + 1
    print(f"[{phase_name}] Windows: {num_windows} | Valid: ", end="", flush=True)

    for i in range(0, total_len - window_samples + 1, step_samples):
        segment = signal[i:i + window_samples]
        hrv_res = get_window_hrv(segment, fs)

        if hrv_res is not None:
            valid_hrv_list.append(hrv_res)

    print(len(valid_hrv_list))

    if len(valid_hrv_list) == 0:
        return None

    return pd.DataFrame(valid_hrv_list).median()


results = []
print("=== Start HRV extraction ===")

for sid in SUBJECTS:
    pkl_path = os.path.join(DATA_PATH, f"{sid}.pkl")
    if not os.path.exists(pkl_path):
        continue

    print(f"\nProcessing {sid}...")
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    except:
        continue

    ecg = data["signal"]["chest"]["ECG"].flatten()
    labels = data["label"]

    idx_base = np.where(labels == 1)[0]
    idx_stress = np.where(labels == 2)[0]

    if len(idx_base) == 0 or len(idx_stress) == 0:
        print("  Missing labels -> skip")
        continue

    feat_base = process_phase_sliding(ecg[idx_base[0]:idx_base[-1] + 1], FS, "Base")
    feat_stress = process_phase_sliding(ecg[idx_stress[0]:idx_stress[-1] + 1], FS, "Stress")

    if feat_base is None or feat_stress is None:
        continue

    row = {'Subject': sid}

    def process_row_data(feat_series, prefix):
        for key in ["RMSSD", "SDNN", "pNN50", "LF/HF"]:
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

    process_row_data(feat_base, "Base")
    process_row_data(feat_stress, "Stress")

    results.append(row)

if len(results) > 0:
    df_out = pd.DataFrame(results)
    df_out.to_csv(SAVE_FILENAME, index=False)
    print(f"\nDone. Saved to: {SAVE_FILENAME}")
    print(df_out[['Subject', 'Base_LF_ms2', 'Base_LF_nu', 'Base_HF_ms2', 'Base_HF_nu']].head())
else:
    print("\nNo valid data")