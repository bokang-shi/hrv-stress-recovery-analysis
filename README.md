# HRV Stress Recovery Analysis

This repository contains the implementation of an ECG-based heart rate variability (HRV) analysis pipeline developed to evaluate acute psychological stress and compare post-stress recovery responses.

The project integrates:

* ECG preprocessing and R-peak detection
* HRV feature extraction (time and frequency domain)
* Statistical analysis of stress-induced changes
* Implementation of the MIST stress task for experimental data collection

## Repository Structure

```text
src/
  feature_extraction/       # HRV extraction from WESAD and primary data
  statistics/               # Wilcoxon test and bootstrap confidence intervals
  validation/               # Signal quality and R-peak detection benchmarking
  stress_induction_task/    # MIST stress task implementation
```

## Dependencies

This project was developed in Python and MATLAB.
Key Python libraries include:
- numpy
- pandas
- neurokit2
- scipy
- matplotlib

MATLAB is required for statistical analysis scripts.

## Main Components

### Feature Extraction

* `01_extract_wesad_hrv_features.py`
* `02_extract_primary_hrv_features.py`

### Statistical Analysis

* `03_run_wilcoxon_hrv_stats.m`
* `04_bootstrap_hrv_confidence_intervals.m`

### Validation

* `wesad_rr_artefact_analysis.py`
* `rpeak_detection_benchmark.py`

### Stress Induction Task

* `mist_stress_task.py`

## Data Availability

Datasets are not included in this repository due to size and access restrictions.
Scripts are provided for processing these datasets, and file paths may need to be adapted locally.

## Notes

Experimental protocols and results are described in the accompanying report and are not duplicated in this repository.
The repository is structured to reflect the methodological workflow and supports reproducibility of the computational analysis.
