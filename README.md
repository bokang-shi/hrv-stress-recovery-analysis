# HRV Stress Recovery Analysis

This repository contains the code used for an undergraduate research project on
personalised stress recovery assessment using heart rate variability (HRV).

The aim is not to build a general stress detection product. Instead, the code
supports a controlled workflow for:

1. validating an ECG-to-HRV processing pipeline on WESAD,
2. applying the same pipeline to primary ECG recordings collected during a MIST
   stress task, and
3. comparing post-stress HRV recovery within the same individual.

This framing matters because recovery responses vary between people. The
pipeline is intended to help compare recovery conditions, such as aromatherapy
or control recovery, using objective HRV features.

## Repository Structure

```text
src/
  feature_extraction/
    01_extract_wesad_hrv_features.py      # HRV extraction from WESAD ECG
    02_extract_primary_hrv_features.py    # HRV extraction from primary HDF5 ECG

  validation/
    wesad_rr_artefact_analysis.py         # RR artefact summary for WESAD
    rpeak_detection_benchmark.py          # R-peak detector benchmark on MIT-BIH

  statistics/
    03_run_wilcoxon_hrv_stats.m           # paired Wilcoxon tests with FDR correction
    04_bootstrap_hrv_confidence_intervals.m

  stress_induction_task/
    mist_stress_task.py                   # Tkinter implementation of the MIST task

data/
  README.md                               # expected local data layout
```

Raw ECG datasets and generated results are not included in the repository.

## Python Setup

The Python scripts were developed with Python 3 and common scientific Python
libraries.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, replace the activation command with:

```bash
source .venv/bin/activate
```

MATLAB is required for the statistical analysis scripts in `src/statistics/`.

## Data

See [data/README.md](data/README.md) for the expected local data layout.

In brief:

- WESAD subject files should be available as `.pkl` files.
- Primary ECG files should be HDF5 `.h5` files named by subject and phase, for
  example `p2_baseline.h5`, `p2_stress.h5`, and `p2_recovery.h5`.
- Generated CSV files and figures are written to `outputs/` by default.

## Typical Workflow

### 1. Validate WESAD RR interval quality

```bash
python src/validation/wesad_rr_artefact_analysis.py --data-dir data/wesad
```

### 2. Benchmark R-peak detection methods

This script downloads records from PhysioNet through `wfdb`, so it needs an
internet connection.

```bash
python src/validation/rpeak_detection_benchmark.py
```

### 3. Extract WESAD HRV features

```bash
python src/feature_extraction/01_extract_wesad_hrv_features.py --data-dir data/wesad
```

### 4. Extract primary experiment HRV features

```bash
python src/feature_extraction/02_extract_primary_hrv_features.py --data-dir data/primary
```

### 5. Run statistical analysis

Use MATLAB to run the scripts in `src/statistics/` after placing the processed
CSV file path in the script settings section.

## Main HRV Features

The analysis focuses on common time-domain and frequency-domain HRV features:

- RMSSD
- SDNN
- pNN50
- LF power
- HF power
- LF/HF ratio

Subject-level phase values are calculated using 5-minute sliding windows with a
1-minute step, followed by median aggregation. This keeps the workflow close to
standard HRV guidance while reducing the effect of noisy windows.

## Notes and Limitations

This is a research codebase for a student project, not a clinical device or
validated diagnostic tool. Results should be interpreted as a proof-of-concept
for HRV-guided, within-subject comparison of recovery responses.

Important limitations include:

- small primary sample size,
- ECG signal quality variation,
- individual differences in baseline autonomic function,
- limited recovery conditions tested so far, and
- dependence on controlled stress induction.

## Code Availability

The repository is organised to support reproduction of the key computational
steps described in the project report: ECG preprocessing, HRV feature
extraction, validation, statistical analysis, and the MIST stress task.
