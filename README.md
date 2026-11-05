# Reproducible UrbanSound8K CNN / DS-CNN training

TensorFlow 2.15 / Keras 2.15, standard or depthwise-separable convolution, a standard
convolution stem, nested shared widths, mixed durations, and verified FP32 / INT8
TFLite exports. The existing architecture and native UrbanSound8K folds are retained.

## Local setup

Use Python 3.11. Existing local environment:

```sh
PY=/opt/miniconda3/envs/tinyml-tf/bin/python
"$PY" -m unittest discover -s tests -v
"$PY" -m training.train --config config.yaml
```

For a new CPU/Linux environment, install `requirements-training.txt`. CUDA/Linux
training uses `requirements-colab.txt`, which adds the CUDA libraries for the same
TensorFlow version. `requirements-lock.txt` retains the historical package versions
apart from the removed platform-specific GPU dependency; it is not the portable
Colab installer. No environment changes or dataset downloads occur on import.

`config.yaml` is the single canonical configuration. Change existing keys with
repeatable `--set key=value` overrides. Runtime `auto` prefers CUDA when TensorFlow
has a CUDA build and sees a GPU, otherwise CPU; `cpu` explicitly hides GPUs.
`cuda` requires CUDA (`gpu` remains an alias). Non-CUDA builds run on CPU. Device
selection happens before initialization; restart Python when changing devices or
determinism. No alternate GPU backend is used. A tiny throwaway optimizer step
checks actual output/gradient placement and resets the experiment seed afterwards.

## Data, durations, and checkpoint selection

Dataset root is independent of the output root:

```text
<dataset.root>/metadata/UrbanSound8K.csv
<dataset.root>/audio/fold1/*.wav ... fold10/*.wav
```

Native class IDs are read from metadata. Default train folds are 1–8, validation is
fold 9, and fold 10 is held out. Training, augmentation noise sources and PTQ cannot
use folds 9/10. Crops of an original file always remain in its original fold.

```sh
# 1.5 seconds, 40 mel bins, DW, width multiplier 0.5
"$PY" -m training.train --set 'training.duration_secs=[1.5]' \
  --set audio.n_mels=40 --set model.conv_type=DW --set model.width=5

# Any subset is supported, including all three durations.
"$PY" -m training.train --set 'training.duration_secs=[1,1.5,2]' --set runtime.device=auto

# Shared-parameter widths 1.0 / 0.5 / 0.25; same batch, one optimizer update.
"$PY" -m training.train --set model.width=all

# CPU synthetic smoke, with enough batches to exercise all three durations.
"$PY" -m training.train --dry-run --limit 6 --set runtime.device=cpu \
  --set training.batch_size=2 --set 'training.duration_secs=[1,1.5,2]'
```

A batch has one fixed duration. Batches alternate selected durations and update the
same model; the epoch start rotates and files are shuffled. Each original file is
retrieved once per epoch, not once per duration. Very small smoke datasets need
enough batches to visit every selected duration. `history.csv` records these counts.
Empty `training.duration_secs` falls back to `audio.duration_sec`. Empty
`quantization.output_duration_secs` follows the training durations; an explicit
list exports only those requested durations. Formats are configurable as
`quantization.formats: [fp32, int8]`.

With the retained 400-sample Hann window, 160-sample hop and 512 FFT, input shapes
excluding batch are `[98,n_mels,1]`, `[148,n_mels,1]`, and `[198,n_mels,1]` for
1s / 1.5s / 2s. Global average pooling lets the same host weights accept these
lengths. TFLite exports use separate fixed shapes; accuracy across durations is
measured separately, not assumed equal.

Every validation epoch evaluates every selected duration. `val_accuracy` is the
**equal-weight mean of per-duration original-file accuracies**. Shared-width runs
first average the three widths within each duration, then average durations.
Counts of crops or batches never weight this checkpoint metric. Per-duration
accuracy and macro-F1 are recorded. ModelCheckpoint, EarlyStopping and LR plateau
control monitor only this exact `val_accuracy`, mode `max`. Best weights are
restored before export, and their precise epoch is reported. Terminal/CSV output
includes accuracy, duration metrics, used/remaining patience, best epoch and LR.

## Composable online augmentation

The ordered pipeline is: normalized float WAV read → channel mean → soxr HQ
resampling → valid crop/window → waveform Compose → Log-Mel → spectrogram Compose
→ feature normalization → model. PCM scaling occurs once. Each transform has its
own enabled flag, probability and parameters in `config.yaml`; decisions are
independent and parameters are resampled at each training retrieval.

| Transform | Probability | Default parameters |
| --- | ---: | --- |
| RandomTimeShift | 0.5 | circular shift up to ±0.2s |
| RandomGain | 0.5 | −6 to +6 dB, no clipping |
| AddBackgroundNoise | 0.4 | 15–30 dB SNR |
| PitchShift | 0.2 | −1 to +1 semitone |
| TimeStretch | 0.2 | pitch-preserving rate 0.9–1.1 |
| FrequencyMask | 0.5 | up to 4 mel bins |
| TimeMask | 0.5 | up to 10 frames per 100 actual frames |

Noise defaults to Gaussian background noise, requiring no external download.
`source: training_audio` samples another training-fold waveform only. Circular
shifts preserve all samples; slowed signals retain their highest-energy window;
masks replace limited bands with the clip mean and are capped at 20% per axis.
The strongest band/frame is retained if it falls inside a mask, protecting a
dominant narrow-band event or brief transient.
Silence/degenerate inputs remain finite. These safeguards avoid total erasure,
but energy is not proof of semantic event retention; audit demanding ablations.

Random streams use seed, epoch, retrieval index and worker ID. The loader is
currently sequential; no unseeded DataLoader workers are started. Rebuilding a
dataset with the same seed reproduces its sequence; repeated training retrievals
resample it. Validation/test and calibration bypass both Compose stages entirely.
Deterministic validation crops are evenly spaced (default 3); probabilities are
averaged by original file ID before classification. Calibration uses one center
crop from training folds, including the identical feature-normalization step.

Normalization defaults to `none` (identity), preserving the original frontend.
`audio.normalization: fixed` applies `(features-mean)/std` **after masks**, using
`normalization_mean` / `normalization_std` supplied from training data only. No
statistics are fitted on validation/test. Mel uses Slaney area normalization and
`ln(max(power,1e-6))`; no per-duration scaling changes. Cache only deterministic
mono/resampled waveforms, never augmented audio or spectrograms.

```sh
# Ablations need no code changes.
"$PY" -m training.train --set augmentation.enabled=false
"$PY" -m training.train --set augmentation.waveform.PitchShift.enabled=false
"$PY" -m training.train --set augmentation.spectrogram.TimeMask.probability=0.2
```

## Models and experiment reports

Canonical model identity remains `DSCNN_{40|64}_{SC|DW}_{1|5|25|all}`. Width tokens
1/5/25 mean multipliers 1.0/0.5/0.25. Channels use `max(1,floor(base*multiplier))`.
The standard Conv2D stem, stride-2 blocks, bias/ReLU, BN-free policy, GAP and dense
classifier are retained. Shared models slice the same convolution/classifier
Variables and export ordinary static subnets; outputs are checked against the
supernet at every trained duration. Old configurations without the stem flag keep
their original architecture when loading historical artifacts.

```text
models/
  .allocations/                 # durable sequence reservations; do not delete
  dscnn_001/
    report.json                 # primary record, including resolved config
    history.csv                 # the only full history
    best.weights.h5             # finalized best checkpoint
    best.keras                  # existing standalone inference model
    exports/1s/{fp32,int8}.tflite
    exports/1.5s/{fp32,int8}.tflite
    exports/2s/{fp32,int8}.tflite
  summary.csv                   # one row per experiment, rebuilt from reports
```

Only requested durations/formats are exported. Shared runs keep the parent
checkpoint and static models under `subnets/DSCNN_.../`, each with a compact
report and exports; child configs are inherited from the parent report. No new
YAML snapshots, per-format JSON files or Markdown reports are generated.
`--run-id` is an optional human label, not a directory override.

Sequence reservations use atomic directory creation, survive failed/deleted runs,
and prevent concurrent processes from overwriting/reusing allocated IDs.
Checkpoints and temporary Keras files live in `runtime.work_dir` or the system
temporary directory. Final models are published only after validation. JSON and
export writes use atomic replacement. As with any FUSE/network filesystem, this
requires the filesystem to honor atomic mkdir/rename; reservations must be kept.

Reports include execution timestamps/status, config/seed/Git/source version,
dependency/device information, folds, exact best epoch, per-duration metrics,
parameter/MAC methodology, export paths/sizes/hashes, calibration settings,
FP32-vs-INT8 accuracy differences, failure details, and unmeasured values as null.
MACs analytically count the existing conv/depthwise/dense operations for each
input shape; they exclude bias/ReLU/GAP/frontend and are not SRAM/latency measures.
Device/arena measurements remain null. Existing CREST identity fields
`run_id`, `effective_width`, `duration_sec` and `profiling/schema.json` are retained.
Confusion matrices are optional (`evaluation.include_confusion_matrix=true`);
the default report stores the required accuracy/F1 metrics without large matrices.

Legacy split JSON/run.json reports and existing TFLite paths remain readable.
Existing `model_registry.csv` files continue to be updated; fresh experiments use
only summary.csv unless the legacy registry is explicitly requested:

```sh
"$PY" -m training.registry --model-dir models
"$PY" -m training.registry --model-dir models --legacy
```

Legacy-registry mode alone writes `legacy_config.yaml` for consumers
that read the original CSV `config_path` as YAML; default reports do not need it.
Completed configurations can be skipped with `--skip-completed`, verifying their
model hashes. Matching also checks the preprocessing/training source hash and data
metadata, so changed code is not silently treated as a completed configuration.
Interrupted/failed configurations allocate a fresh ID on retry and
preserve their earlier reports. Optimizer-state resume is not implemented; the
skip/retry behavior is explicit, never an overwrite disguised as resume.

## Export, evaluation and batch runs

```sh
"$PY" -m training.sweep --config config.yaml --space configs/search_space.yaml --skip-completed
"$PY" -m quantization.convert_int8 --model-dir models/dscnn_001 --duration 1.5
"$PY" -m evaluation.evaluate --model-dir models/dscnn_001 --split val --duration 1.5
# On another machine, override the dataset location independently:
"$PY" -m evaluation.evaluate --model-dir models/dscnn_001 --split val \
  --set dataset.root=/path/to/urbansound8k
# Explicit final held-out evaluation only, after configuration selection:
"$PY" -m evaluation.evaluate --model-dir models/dscnn_001 --split test --duration 1.5
```

Sweeps launch the existing CLI in a new process per configuration, save each
completed result immediately, skip matching finalized runs when requested, and
continue after experiment failures. No TensorFlow model/GPU state is retained in
the controller. FP32 and INT8 use the same deterministic examples, crop positions,
class IDs and aggregation. PTQ requires training-only representative data,
`TFLITE_BUILTINS_INT8`, INT8 IO, no float tensors and successful CPU invocation.
Physical MCU/operator/arena compatibility still needs board-specific validation;
no firmware, timing/energy or accuracy results are invented.

## Google Colab

Open [`notebooks/colab_training.ipynb`](notebooks/colab_training.ipynb) and run cells
in order: select GPU runtime → mount Drive/check and clone `colab-training` →
create the isolated Python 3.11 environment/install pinned dependencies → set
independent dataset/output paths → edit form controls → train → inspect summary.
The notebook refuses a missing branch and never falls back to main or embeds
credentials. It stages data/work/cache under `/content` and saves final artifacts
to `/content/drive/MyDrive/EmbeddedSystem/models/`. Prior finalized runs survive
runtime loss. The notebook kernel is not rewritten; training subprocesses use
the pinned environment and pip-provided CUDA libraries. Live Colab/physical CUDA
execution must be verified in that runtime; local mocked selection is not a GPU
benchmark. The optional final-test cell is disabled by default.
