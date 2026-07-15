"""
Benchmark NeuroKit2 R-peak detection methods on MIT-BIH ECG records.

Gaussian noise is added at several SNR levels, then detected R-peaks are
compared with MIT-BIH annotations using an F1 score.
"""

import argparse
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import seaborn as sns
import wfdb


warnings.filterwarnings("ignore")

RECORD_IDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

DATABASE = "mitdb"

METHODS_TO_TEST = [
    "neurokit",
    "pantompkins1985",
    "hamilton2002",
    "nabian2018",
    "elgendi2010",
    "kalidas2017",
    "engzeemod2012",
    "christov2004",
    "martinez2004",
    "rodrigues2021",
]

SNR_LEVELS = [0, 5, 10, 15, 20, "Clean"]


def add_gaussian_noise(signal, snr, rng):
    """Add white Gaussian noise at a given SNR."""
    if snr == "Clean":
        return signal

    signal_power = np.mean(signal ** 2)
    if signal_power == 0:
        return signal

    snr_linear = 10 ** (snr / 10)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def calculate_metrics(true_peaks, detected_peaks, fs, tolerance_sec=0.05):
    """Compute F1 score for detected R-peaks."""
    tolerance = int(tolerance_sec * fs)
    tp = 0
    fp = 0

    true_arr = np.array(true_peaks)
    det_arr = np.array(detected_peaks)

    if len(det_arr) == 0:
        return 0

    for det in det_arr:
        if len(true_arr) > 0 and np.min(np.abs(true_arr - det)) <= tolerance:
            tp += 1
        else:
            fp += 1

    fn = len(true_arr) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1


def run_benchmark(duration_minutes, output_file, heatmap_file, seed, show_plot):
    rng = np.random.default_rng(seed)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Start robustness test ({duration_minutes}-min segment per record)")
    print(f"Records: {len(RECORD_IDS)} | Methods: {len(METHODS_TO_TEST)} | Noise levels: {len(SNR_LEVELS)}")
    print(f"Saving results to: {output_file}")
    print("=" * 60)

    all_results = []
    start_total = time.time()

    for i, rid in enumerate(RECORD_IDS):
        t0 = time.time()

        try:
            record = wfdb.rdrecord(rid, pn_dir=DATABASE)
            annotation = wfdb.rdann(rid, "atr", pn_dir=DATABASE)

            fs = record.fs
            full_signal = record.p_signal[:, 0]
            full_peaks = annotation.sample

            limit = duration_minutes * 60 * fs
            if len(full_signal) > limit:
                signal_slice = full_signal[:limit]
                peaks_slice = full_peaks[full_peaks < limit]
            else:
                signal_slice = full_signal
                peaks_slice = full_peaks

            for snr in SNR_LEVELS:
                noisy_signal = add_gaussian_noise(signal_slice, snr, rng)

                try:
                    ecg_cleaned = nk.ecg_clean(noisy_signal, sampling_rate=fs, method="neurokit")
                except Exception:
                    ecg_cleaned = noisy_signal

                for method in METHODS_TO_TEST:
                    print(
                        f"[{i + 1}/{len(RECORD_IDS)}] Rec:{rid} | SNR:{str(snr).rjust(5)} "
                        f"| Alg:{method.ljust(15)}",
                        end="\r",
                    )

                    try:
                        _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method=method)
                        detected = info["ECG_R_Peaks"]
                        f1 = calculate_metrics(peaks_slice, detected, fs)
                    except Exception:
                        f1 = 0.0

                    all_results.append({
                        "Record": rid,
                        "Method": method,
                        "SNR": snr,
                        "F1": f1,
                    })

            print(f"[{i + 1}/{len(RECORD_IDS)}] Record {rid} done ({time.time() - t0:.1f}s)")

        except Exception as exc:
            print(f"\nError processing {rid}: {exc}")

    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print(f"Finished. Total time: {(time.time() - start_total) / 60:.1f} min")
    print("Generating heatmap...")

    heatmap_df = df.groupby(["Method", "SNR"])["F1"].mean().reset_index()
    heatmap_matrix = heatmap_df.pivot(index="Method", columns="SNR", values="F1")

    cols_order = ["0", "5", "10", "15", "20", "Clean"]
    heatmap_matrix.columns = heatmap_matrix.columns.astype(str)
    heatmap_matrix = heatmap_matrix.reindex(columns=cols_order)

    plt.figure(figsize=(12, 7))
    sns.set(font_scale=1.1)
    sns.heatmap(
        heatmap_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdBu",
        vmin=0,
        vmax=1,
        linewidths=1,
        linecolor="white",
    )

    plt.title(f"Mean F1 Score across MIT-BIH ({duration_minutes} min segments)")
    plt.xlabel("Signal-to-Noise Ratio (dB)")
    plt.ylabel("Algorithm")

    plt.tight_layout()
    plt.savefig(heatmap_file, dpi=300)
    print(f"Saved figure: {heatmap_file}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark R-peak detection methods on MIT-BIH.")
    parser.add_argument("--duration-minutes", type=int, default=5,
                        help="Length of ECG segment used from each record.")
    parser.add_argument("--out", type=Path, default=Path("outputs/validation/robustness_results_5min.csv"),
                        help="Output CSV path.")
    parser.add_argument("--heatmap", type=Path, default=Path("outputs/validation/final_heatmap_5min.png"),
                        help="Output heatmap path.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for added Gaussian noise.")
    parser.add_argument("--show", action="store_true",
                        help="Display the heatmap interactively after saving.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_benchmark(args.duration_minutes, args.out, args.heatmap, args.seed, args.show)


if __name__ == "__main__":
    main()
