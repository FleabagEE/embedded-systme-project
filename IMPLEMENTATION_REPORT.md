# Implementation and verification report

Implemented on 2026-10-23 using only
`/opt/miniconda3/envs/tinyml-tf/bin/python` for Python execution.

## Delivered

- Local UrbanSound8K metadata/WAV loading with official fold separation and native
  labels; the previous download-on-import script is now a local inspection CLI.
- Float PCM mono conversion, high-quality resampling, pitch-preserving speed
  perturbation, random training crop, centered padding, gain, shared Log-Mel
  frontend, and safe deterministic waveform caching.
- Comparable BN-free SC and DW architectures with width tokens 1/5/25 and global
  pooling; one shared nested-width model for `all`, joint gradient updates, and
  verified static subnet export for both convolution families.
- YAML configuration/CLI overrides, deterministic runtime controls, training
  history, best-weight checkpoints, early stopping, source/environment/data
  provenance, collision-safe run paths and an atomic process-locked model registry.
- Fixed-duration FP32 and fully INT8 TFLite export, training-only calibration,
  scale/zero-point/operator/byte-size metadata, CPU interpreter validation, and
  original-file accuracy/macro-F1 for Keras, TFLite and shared/frozen subnets.
- Baseline/search configurations, dataset-independent unittest suite, README,
  future deployment/selection boundaries and profiling schema.

`requirements-lock.txt` SHA-256 before and after implementation:
`91c6982b6763258442807c560fadd9a46a7fa89060d0e53f065260b4ba5b8317`.
No new environment, installation, dataset download, or held-out test evaluation
was performed. The workspace was not a Git repository; code source hashes are
recorded instead of fabricating a commit ID.

## Commands and observed outcomes

All commands below use this shell variable:

```sh
PY=/opt/miniconda3/envs/tinyml-tf/bin/python
```

Interpreter inspection found Python 3.11.17, macOS arm64, TensorFlow 2.15.1,
NumPy 1.26.4 and CPU availability. No supported GPU was reported.

```sh
"$PY" -m data.loader
"$PY" -m unittest discover -s tests -v
```

Dataset inspection found 10 native classes and 8,732 metadata rows, partitioned
7,079/816/837 across train/validation/test. The completed unittest suite passes
16 tests. It covers feature shapes and a TensorFlow STFT reference; audio scaling,
resampling, augmentation order/probability injection; official splits and
calibration isolation; exact names/overrides/collisions; original-file metrics;
SC/DW temporal shape flexibility; shared-kernel identity and gradient prefixes;
joint loss and one optimizer update; frozen/supernet matching; full synthetic
training/export; INT8 IO/invocation; and identical weights in 1s/2s exports.

```sh
"$PY" -m training.train --dry-run --set training.batch_size=4 \
  --set augmentation.enabled=false --run-id baseline_synthetic_smoke

"$PY" -m training.train --set training.epochs=1 --set training.batch_size=4 \
  --set quantization.representative_samples=12 \
  --set quantization.output_duration_sec=2 --limit 20 \
  --run-id baseline_local_verified
```

Both completed. The real-data check trained on 20 fold-1–8 files, validated on 20
fold-9 files, calibrated on 12 training files, and wrote FP32/INT8 exports for both
durations. It is a one-epoch **diagnostic subset**, not a performance benchmark.
The 1s validation accuracy was 0.05 and macro-F1 was 0.00952381 for Keras FP32,
FP32 TFLite and INT8 TFLite. Each FP32 file is 39,796 bytes; each INT8 file is
22,616 bytes. Matching frozen-weight SHA-256 values confirm that 1s/2s exports use
identical learned weights.

```sh
"$PY" -m training.train --dry-run --set model.width=all \
  --set training.batch_size=4 --set augmentation.enabled=false \
  --set quantization.representative_samples=8 --limit 8 \
  --run-id multiwidth_final_verified

"$PY" -m training.train --dry-run --set model.width=all \
  --set model.conv_type=SC --set audio.n_mels=64 --set audio.duration_sec=2 \
  --set quantization.output_duration_sec=2 --set 'model.base_channels=[8,12]' \
  --set model.num_blocks=2 --set training.batch_size=2 \
  --set augmentation.enabled=false --set quantization.representative_samples=4 \
  --limit 4 --run-id sc64_2s_shared_smoke
```

The DW 40-bin/1s and SC 64-bin/2s shared smoke runs complete one synthetic epoch,
save shared checkpoints, and export all three static widths to FP32 and INT8.
Every exported subnet passes the CPU invocation and shared-model agreement
checks. Their synthetic metrics are explicitly ineligible as empirical dataset
results. An initial multiwidth config serialization issue was fixed and is covered
by the integration test; its failed run directory is retained with diagnostics.

```sh
"$PY" -m quantization.convert_int8 \
  --model-dir models/1s/DSCNN_40_DW_5/baseline_local_data_check --duration 2
"$PY" -m evaluation.evaluate \
  --model-dir models/1s/DSCNN_40_DW_5/baseline_local_verified \
  --split val --duration 2 --limit 20
"$PY" -m evaluation.evaluate \
  --model-dir models/1s/DSCNN_40_DW_all/multiwidth_synthetic_verified --split val
"$PY" -m quantization.validate_int8 \
  models/1s/DSCNN_40_DW_5/baseline_local_verified/exports/1s/int8.tflite
```

The standalone conversion, evaluation and validation CLIs succeeded. The 1s INT8
model uses input `[1,98,40,1]`, output `[1,10]`, both INT8, and invokes on CPU.
The separate 2s model uses input `[1,198,40,1]`. Cross-duration validation uses
1s-trained weights without retraining and is recorded as such; this tiny subset
scored 0 accuracy/F1 at 2s. CLI evaluation of shared runs also reloads the original
shared checkpoint and reports shared/frozen/Keras/TFLite results for each width.

## Limits and work not executed

- Full 50-epoch/full-training-fold experiments and held-out fold-10 evaluation
  were not run. No useful model accuracy is inferred from these integration checks.
- Optimizer-state training resume is not implemented; checkpoint behavior is
  documented. Best weights can be restored for inference and frozen exports.
- The 48-config search space is supplied as configuration; a batch sweep runner
  is optional and not implemented.
- No firmware, device benchmarks, Tensor Arena sizing, latency/energy prediction,
  or hardware selection policy is implemented or claimed.
- TensorFlow 2.15 prints generic converter calibration-statistics and Apple
  optimizer performance warnings. Actual INT8 tensor validation and CPU invocation
  pass; these warnings did not block the pipeline. GPU training was unavailable
  and was not required.
- Linux/platform-specific installation instructions are documented but were not
  executed; the existing Conda environment needed no changes.
