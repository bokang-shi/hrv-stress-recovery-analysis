# Data Layout

Raw ECG data is not included in this repository.

Suggested local layout:

```text
data/
  wesad/
    S2.pkl
    S3.pkl
    ...

  primary/
    p2_baseline.h5
    p2_stress.h5
    p2_recovery.h5
    ...
```

The primary extraction script expects HDF5 filenames in the form:

```text
<subject>_<phase>.h5
```

where `<phase>` is one of:

- `baseline`
- `stress`
- `recovery`

The ECG channel is expected to be stored as `A4`, matching the BITalino
OpenSignals export used in the project.

Generated files are written to `outputs/` by default.
