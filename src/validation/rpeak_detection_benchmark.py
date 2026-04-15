import wfdb
import neurokit2 as nk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import time
import os

# Settings
warnings.filterwarnings("ignore")

record_ids = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

database = "mitdb"

methods_to_test = [
    "neurokit",
    "pantompkins1985",
    "hamilton2002",
    "nabian2018",
    "elgendi2010",
    "kalidas2017",
    "engzeemod2012",
    "christov2004",
    "martinez2004",
    "rodrigues2021"
]

snr_levels = [0, 5, 10, 15, 20, "Clean"]

DURATION_MINUTES = 5
output_file = "robustness_results_5min.csv"


def add_gaussian_noise(signal, snr):
    """Add white Gaussian noise at a given SNR."""
    if snr == "Clean":
        return signal

    signal_power = np.mean(signal ** 2)
    if signal_power == 0:
        return signal

    snr_linear = 10 ** (snr / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
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


print("=" * 60)
print(f"Start robustness test ({DURATION_MINUTES}-min segment per record)")
print(f"Records: {len(record_ids)} | Methods: {len(methods_to_test)} | Noise levels: {len(snr_levels)}")
print(f"Saving results to: {output_file}")
print("=" * 60)

# Create output file if needed
if not os.path.exists(output_file):
    pd.DataFrame(columns=["Record", "Method", "SNR", "F1"]).to_csv(output_file, index=False)

start_total = time.time()

for i, rid in enumerate(record_ids):
    t0 = time.time()

    try:
        # Load record and annotations
        record = wfdb.rdrecord(rid, pn_dir=database)
        annotation = wfdb.rdann(rid, "atr", pn_dir=database)

        fs = record.fs
        full_signal = record.p_signal[:, 0]
        full_peaks = annotation.sample

        # Use only the first 5 minutes
        limit = DURATION_MINUTES * 60 * fs
        if len(full_signal) > limit:
            signal_slice = full_signal[:limit]
            peaks_slice = full_peaks[full_peaks < limit]
        else:
            signal_slice = full_signal
            peaks_slice = full_peaks

        batch_results = []

        for snr in snr_levels:
            noisy_signal = add_gaussian_noise(signal_slice, snr)

            try:
                ecg_cleaned = nk.ecg_clean(noisy_signal, sampling_rate=fs, method="neurokit")
            except Exception:
                ecg_cleaned = noisy_signal

            for method in methods_to_test:
                print(
                    f"[{i+1}/{len(record_ids)}] Rec:{rid} | SNR:{str(snr).rjust(5)} | Alg:{method.ljust(15)}",
                    end="\r"
                )

                try:
                    _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method=method)
                    detected = info["ECG_R_Peaks"]
                    f1 = calculate_metrics(peaks_slice, detected, fs)
                except Exception:
                    f1 = 0.0

                batch_results.append({
                    "Record": rid,
                    "Method": method,
                    "SNR": snr,
                    "F1": f1
                })

        # Save after each record
        pd.DataFrame(batch_results).to_csv(output_file, mode="a", header=False, index=False)
        print(f"[{i+1}/{len(record_ids)}] Record {rid} done ({time.time() - t0:.1f}s)                            ")

    except Exception as e:
        print(f"\nError processing {rid}: {e}")

print("\n" + "=" * 60)
print(f"Finished. Total time: {(time.time() - start_total) / 60:.1f} min")

print("Generating heatmap...")

try:
    df = pd.read_csv(output_file)

    # Mean F1 across records
    heatmap_df = df.groupby(["Method", "SNR"])["F1"].mean().reset_index()

    heatmap_matrix = heatmap_df.pivot(index="Method", columns="SNR", values="F1")

    # Put Clean at the end
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
        linecolor="white"
    )

    plt.title(f"Mean F1 Score across MIT-BIH ({DURATION_MINUTES} min segments)")
    plt.xlabel("Signal-to-Noise Ratio (dB)")
    plt.ylabel("Algorithm")

    plt.tight_layout()
    plt.savefig("final_heatmap_5min.png", dpi=300)
    print("Saved figure: final_heatmap_5min.png")
    plt.show()

except Exception as e:
    print(f"Plotting failed: {e}")
