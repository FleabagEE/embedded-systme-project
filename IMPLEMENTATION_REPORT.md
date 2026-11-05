# Implementation notes

The current implementation is an in-place TensorFlow 2.15 / Keras 2.15 refactor
on `colab-training`. Use `config.yaml`, the README commands, and
`notebooks/colab_training.ipynb`; generated experiments use `report.json` plus an
optional history.csv. No per-experiment Markdown reports are generated.

Retained: SC/DW, standard stem, nested shared widths, mixed-duration batching,
GAP, official folds, FP32/INT8 TFLite validation, legacy registry export, and the
CREST profiling identity/schema. There were no physical CREST adapters in the
repository; no board integration or measurements are claimed.

Added: 1.5-second inputs (148 frames), CPU/CUDA runtime selection, independent
online waveform/spectrogram Compose transforms, accuracy-only checkpoint control,
durable experiment sequence allocation, report-derived summary, subprocess sweeps
with completed-run skipping, and a lightweight Colab controller.

Verification commands:

```sh
PY=/opt/miniconda3/envs/tinyml-tf/bin/python
"$PY" -m unittest discover -s tests -v
"$PY" -m training.train --dry-run --limit 6 --set runtime.device=cpu \
  --set training.batch_size=2 --set 'training.duration_secs=[1,1.5,2]'
```

The local environment supports CPU TensorFlow 2.15.1. Tests exercise actual small
training/INT8 conversion and invocation, concurrent allocation, shared subnets,
checkpoint selection and equal-duration metrics, online sampling and normalization,
legacy report/config access, notebook JSON/Python syntax, and absence of embedded
credentials. Exact results and commit are reported in the delivery message.

Limitations: live Colab/CUDA and physical MCU deployment are not tested merely by
notebook validation or mocked runtime tests. CUDA kernel determinism, driver
compatibility, Tensor Arena/SRAM and CREST device performance require validation on
the target system. Optimizer-state resume is not implemented: completed runs can
be skipped, and failed/interrupted attempts retry under new IDs. Reports preserve
nulls for unmeasured quantities and clearly label synthetic/subset smoke metrics.
