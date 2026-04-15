
"""
Local H5 HRV feature extraction (A4 ECG)
5-min window, 1-min step
Median aggregation
Lomb-Scargle, absolute power (ms^2)
"""

import os
import re
import h5py
import numpy as np
import pandas as pd
import neurokit2 as nk
import warnings
from datetime import datetime

# ---------- NumPy 2.0 patch ----------
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
# -------------------------------------

warnings.filterwarnings("ignore")

# ========== Settings ==========
DATA_PATH = r"C:\Users\Windows\OneDrive - Imperial College London\Desktop\HRV_local"
DEFAULT_FS = 1000

WINDOW_SEC = 300
STEP_SEC = 60
MIN_PEAKS = 200

DATE_TAG = datetime.today().strftime("%Y%m%d")
SAVE_FILENAME = os.path.join(
    DATA_PATH, f"local_hrv_sliding_median_ms2_{DATE_TAG}.csv"
)

PHASE_MAP = {
    "baseline": "Base",
    "stress": "Stress",
    "recovery": "Recovery",
}

FEATURE_KEYS = {
    "RMSSD": "HRV_RMSSD",
    "SDNN": "HRV_SDNN",
    "pNN50": "HRV_pNN50",
    "LF_ms2": "HRV_LF",
    "HF_ms2": "HRV_HF",
    "LF/HF": "HRV_LFHF",
}

# ========== HRV for one window ==========
def get_window_hrv(segment, fs):
    seg = np.nan_to_num(np.asarray(segment, dtype=np.float64))

    # Skip nearly flat signals
    if np.std(seg) < 1e-7 or np.ptp(seg) < 1e-6:
        return None

    try:
        clean = nk.ecg_clean(seg, sampling_rate=fs, method="neurokit")
        _, info = nk.ecg_peaks(clean, sampling_rate=fs)

        if len(info["ECG_R_Peaks"]) < MIN_PEAKS:
            return None

        hrv_t = nk.hrv_time(info, sampling_rate=fs)
        hrv_f = nk.hrv_frequency(
            info,
            sampling_rate=fs,
            psd_method="lomb",
            normalize=False,   # absolute power in ms^2
        )

        return pd.concat([hrv_t, hrv_f], axis=1).iloc[0]

    except Exception:
        return None


def process_phase_sliding(signal, fs, phase=""):
    win = int(WINDOW_SEC * fs)
    step = int(STEP_SEC * fs)

    if len(signal) < win:
        print(f"[{phase}] shorter than 5 min -> skip")
        return None

    feats = []
    n_win = (len(signal) - win) // step + 1
    print(f"[{phase}] windows={n_win}", end=" | ")

    for i in range(0, len(signal) - win + 1, step):
        h = get_window_hrv(signal[i:i+win], fs)
        if h is not None:
            feats.append(h)

    print(f"valid={len(feats)}")

    if len(feats) == 0:
        return None

    return pd.DataFrame(feats).median()


# ========== Parse filename ==========
def parse_filename(fname):
    m = re.match(r"^(?P<subj>[^_]+)_(?P<phase>[^.]+)\.h5$", fname, re.I)
    if not m:
        return None, None
    return m.group("subj"), m.group("phase").lower()


# ========== Main ==========
results = {}
print("=== Start HRV extraction (A4 ECG) ===")

for fname in os.listdir(DATA_PATH):
    if not fname.lower().endswith(".h5"):
        continue

    subj, phase = parse_filename(fname)
    if subj is None or phase not in PHASE_MAP:
        continue

    print(f"\nProcessing {fname}")
    with h5py.File(os.path.join(DATA_PATH, fname), "r") as f:
        if "A4" not in f:
            print("  A4 not found -> skip")
            continue

        ecg = np.asarray(f["A4"]).squeeze().astype(float)
        fs = DEFAULT_FS

    feat = process_phase_sliding(ecg, fs, PHASE_MAP[phase])
    if feat is None:
        continue

    if subj not in results:
        results[subj] = {"Subject": subj}

    prefix = PHASE_MAP[phase]

    # Basic features
    for k in ["RMSSD", "SDNN", "pNN50", "LF/HF"]:
        results[subj][f"{prefix}_{k}"] = feat.get(FEATURE_KEYS[k], np.nan)

    # Absolute power
    lf = feat.get("HRV_LF", np.nan)
    hf = feat.get("HRV_HF", np.nan)

    results[subj][f"{prefix}_LF_ms2"] = lf
    results[subj][f"{prefix}_HF_ms2"] = hf

    # Compute normalized units manually
    if not np.isnan(lf) and not np.isnan(hf) and (lf + hf) > 0:
        results[subj][f"{prefix}_LF_nu"] = lf / (lf + hf)
        results[subj][f"{prefix}_HF_nu"] = hf / (lf + hf)
    else:
        results[subj][f"{prefix}_LF_nu"] = np.nan
        results[subj][f"{prefix}_HF_nu"] = np.nan


# ========== Save ==========
if results:
    df = pd.DataFrame(results.values())
    df.to_csv(SAVE_FILENAME, index=False)
    print(f"\nDone. Saved to:\n{SAVE_FILENAME}")
else:
    print("\nNo valid data extracted")