"""
Extract HRV features from primary BITalino/OpenSignals HDF5 recordings.

Expected filenames use the pattern <subject>_<phase>.h5, for example:
    p2_baseline.h5
    p2_stress.h5
    p2_recovery.h5
"""

import argparse
import re
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import neurokit2 as nk
import numpy as np
import pandas as pd


# NumPy 2.0 compatibility for older NeuroKit2 versions.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

warnings.filterwarnings("ignore")

DEFAULT_FS = 1000
DEFAULT_CHANNEL = "A4"

WINDOW_SEC = 300
STEP_SEC = 60
MIN_PEAKS = 200

PHASE_MAP = {
    "baseline": "Base",
    "stress": "Stress",
    "recovery": "Recovery",
}

FEATURE_KEYS = {
    "RMSSD": "HRV_RMSSD",
    "SDNN": "HRV_SDNN",
    "pNN50": "HRV_pNN50",
    "LF/HF": "HRV_LFHF",
}


def get_window_hrv(segment, fs):
    """Compute HRV features for one ECG window."""
    seg = np.nan_to_num(np.asarray(segment, dtype=np.float64))

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
            normalize=False,
        )

        return pd.concat([hrv_t, hrv_f], axis=1).iloc[0]

    except Exception:
        return None


def process_phase_sliding(signal, fs, phase=""):
    win = int(WINDOW_SEC * fs)
    step = int(STEP_SEC * fs)

    if len(signal) < win:
        print(f"[{phase}] shorter than 5 minutes; skipped")
        return None

    feats = []
    n_win = (len(signal) - win) // step + 1
    print(f"[{phase}] windows={n_win}", end=" | ")

    for i in range(0, len(signal) - win + 1, step):
        hrv_features = get_window_hrv(signal[i:i + win], fs)
        if hrv_features is not None:
            feats.append(hrv_features)

    print(f"valid={len(feats)}")

    if not feats:
        return None

    return pd.DataFrame(feats).median()


def parse_filename(fname):
    match = re.match(r"^(?P<subj>[^_]+)_(?P<phase>[^.]+)\.h5$", fname, re.I)
    if not match:
        return None, None
    return match.group("subj"), match.group("phase").lower()


def add_phase_features(results, subject, prefix, feat):
    for key in ("RMSSD", "SDNN", "pNN50", "LF/HF"):
        results[subject][f"{prefix}_{key}"] = feat.get(FEATURE_KEYS[key], np.nan)

    lf = feat.get("HRV_LF", np.nan)
    hf = feat.get("HRV_HF", np.nan)

    results[subject][f"{prefix}_LF_ms2"] = lf
    results[subject][f"{prefix}_HF_ms2"] = hf

    if not np.isnan(lf) and not np.isnan(hf) and (lf + hf) > 0:
        results[subject][f"{prefix}_LF_nu"] = lf / (lf + hf)
        results[subject][f"{prefix}_HF_nu"] = hf / (lf + hf)
    else:
        results[subject][f"{prefix}_LF_nu"] = np.nan
        results[subject][f"{prefix}_HF_nu"] = np.nan


def extract_primary_hrv(data_dir, output_file, sampling_rate, channel):
    results = {}
    print("=== Primary ECG HRV extraction ===")

    for h5_path in sorted(data_dir.glob("*.h5")):
        subj, phase = parse_filename(h5_path.name)
        if subj is None or phase not in PHASE_MAP:
            print(f"[WARN] Unexpected filename, skipped: {h5_path.name}")
            continue

        print(f"\nProcessing {h5_path.name}")
        with h5py.File(h5_path, "r") as f:
            if channel not in f:
                print(f"  Channel {channel} not found; skipped")
                continue

            ecg = np.asarray(f[channel]).squeeze().astype(float)

        feat = process_phase_sliding(ecg, sampling_rate, PHASE_MAP[phase])
        if feat is None:
            continue

        if subj not in results:
            results[subj] = {"Subject": subj}

        add_phase_features(results, subj, PHASE_MAP[phase], feat)

    if not results:
        print("\nNo valid data extracted")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results.values())
    df.to_csv(output_file, index=False)
    print(f"\nDone. Saved to: {output_file}")


def parse_args():
    date_tag = datetime.today().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description="Extract HRV features from primary HDF5 ECG files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/primary"),
                        help="Folder containing <subject>_<phase>.h5 files.")
    parser.add_argument("--out", type=Path,
                        default=Path(f"outputs/local_hrv_sliding_median_ms2_{date_tag}.csv"),
                        help="Output CSV path.")
    parser.add_argument("--sampling-rate", type=int, default=DEFAULT_FS,
                        help="ECG sampling rate in Hz.")
    parser.add_argument("--channel", default=DEFAULT_CHANNEL,
                        help="HDF5 channel containing ECG data.")
    return parser.parse_args()


def main():
    args = parse_args()
    extract_primary_hrv(args.data_dir, args.out, args.sampling_rate, args.channel)


if __name__ == "__main__":
    main()
