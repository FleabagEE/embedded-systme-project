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

The architecture extension adds MobileNetV2-Tiny (inverted residuals, ReLU6,
width-specific BatchNorm parameters/statistics), avg/max/mix global pooling for
both model families, optional sandwich sampling and training-only population BN
recalibration. A common model dispatcher retains one training/export pipeline.
Validation aggregates every selected width-duration configuration equally. The
progress callback reports epoch timing, best checkpoint, LR, patience and recent
mean ETA. New shared runs nest subnet records in one primary report.json; legacy
per-subnet reports and CREST registry access remain supported.

Verification for this extension: 50 unit/integration tests passed in the existing
tinyml-tf environment. Tests include actual shared MobileNet training, all nine
width-duration combinations, FP32/INT8 conversion and CPU invocation, BN isolation
and recalibration, frozen-subnet equality, gradient summation, callback ETA and
checkpoint restoration. A separate real UrbanSound8K diagnostic run (four files
per split, two epochs, width 0.25, 1.5 seconds, max pooling) completed and exported
FP32/INT8 models. These small runs do not measure full-dataset accuracy gains.

Export now records static Micro resolver/operator checks, including int8
concat/reduce-max scale constraints. Exported input batch and temporal dimensions
are fixed. Firmware/operator-version validation and ESP-NN acceleration, SRAM,
latency and physical board execution remain unverified.

Verification commands:

Sliding-window validation extension: the canonical config and Colab now default
to deterministic full-clip sliding windows, stride 0.5 seconds and mean probability
aggregation. Fixed mode keeps the existing crop positions, padding and aggregation;
historical configs without the new section retain fixed validation. Maximum
probability aggregation is also available for sliding windows. All width-duration
clip accuracies receive equal checkpoint weight.

The new host validation module batches windows and caches resampled audio/exact
features in private temporary directories, with no prediction caching. Reports
include protocol, clip/window counts, per-configuration metrics and separate
training/validation timing. Different evaluation protocols use distinct result
keys and configuration fingerprints. TFLite input shapes, checkpoint formats,
training/calibration augmentation policy and CREST profiling remain unchanged.

Verification: 61 tests passed in tinyml-tf, including fixed regression, sliding
positions/tail/padding, cache and batch invariance, fresh predictions, aggregation,
all model/width/duration combinations, checkpoint selection and Colab overrides.
A two-epoch CPU diagnostic run on six local UrbanSound8K clips per split evaluated
105 windows (315 width-window predictions) per epoch and exported/invoked all 18
FP32/INT8 variants. Its single report was about 145 KB. Validation took about 1.40s
then 0.37s; initialization/cache effects and this tiny subset prevent treating
those figures as a representative performance benchmark. No test-fold selection,
CUDA execution or physical board measurement was performed.

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
