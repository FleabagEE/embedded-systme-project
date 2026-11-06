# Reproducible UrbanSound8K DS-CNN / MobileNetV2-Tiny training

TensorFlow 2.15 / Keras 2.15, standard or depthwise-separable convolution, a standard
convolution stem, nested shared widths, mixed durations, and verified FP32 / INT8
TFLite exports. MobileNetV2-Tiny adds inverted residuals and switchable BatchNorm.
Both architectures use the same trainer, dataset, validation and export paths;
native UrbanSound8K folds and DS-CNN defaults are retained.

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

## Log-Mel configuration and rerunning one experiment

The ordinary single-training and Cartesian batch paths support **40, 64 and 128**
Mel bins. The notebook dropdown, batch choices, `configs/search_space.yaml` and
backend validation include 128. Numeric form values such as `128`, `128.0` and
`"128"` resolve to the same integer count before feature/model construction.
Unsupported or fractional counts fail with an explicit `audio.n_mels` error.

The regular notebook launch first calls the selected CLI with `--check-config`.
It validates inputs without allocating an experiment, loading data, starting
TensorFlow or exporting a model, then prints the actual backend source path,
resolved Mel bins/input shapes (single mode), or validated batch configuration
count and Mel choices. Actual training also logs its resolved Log-Mel count.
The single-run check prints augmentation and FrequencyMask settings, making a
128-Mel / mask-8 setup visible before training.

```sh
# Inspect the actual backend and resolved 128-Mel shapes before training.
"$PY" -m training.train --check-config --set audio.n_mels=128 \
  --set model.architecture=mobilenetv2 --set model.mobile_spatial_conv=DW \
  --set 'training.duration_secs=[1,2]'
"$PY" -m training.sweep --check-config --space configs/search_space.yaml
```

If an open Colab form still has only 40/64, open the latest branch notebook and
rerun its branch setup cell. Pulling source code updates the checkout but does not
replace cells already open in a separate Colab notebook. Adding a dropdown entry
alone cannot update an older backend. The launch check now fails before training
with the backend error and a setup hint when this mismatch occurs. No automatic
reset of local edits is performed.

To rerun just the last unfinished configuration using ordinary controls, choose:

```text
experiment_mode: single
architecture: MobileNetV2-Tiny
mobile_spatial_conv: DW
mel_bins: 128
pooling: max
width_multiplier: all
training_duration: mixed
mixed_durations: [1.0, 2.0]
validation_method: fixed
seed: 42
epochs: 50
batch_size: 64
```

In **Data augmentation config**, choose `custom`, keep augmentation enabled and
set `frequency_mask_max_bins=8` (with FrequencyMask enabled). The custom fields
apply only in custom mode; changing them while `augmentation_config=config.yaml`
is selected continues to use YAML values. Rerun both the augmentation and
Experiment controls cells before the regular Train cell. Each new run gets a
fresh per-architecture number; existing reports, reservations and weights remain
intact. This reruns the configuration from initialization, not a partial optimizer
checkpoint. Do not launch a whole batch if only one configuration remains.

For exact saved settings, the CLI can also read the resolved configuration from
an existing experiment report, including an interrupted report:

```sh
"$PY" -m training.train --config /path/to/models/mobilenetv2_013/report.json \
  --set dataset.root=/path/to/urbansound8k --set runtime.output_dir=/path/to/models
```

The dedicated multi-baseline controller and its preset file have been removed;
ordinary single/batch training, MobileNet SC/DW ablations, mixed durations, model
exports and legacy report reading remain available.

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
1s / 1.5s / 2s. Global pooling (`avg`, `max`, or `mix`) lets the same host weights accept these
lengths. TFLite exports use separate fixed shapes; accuracy across durations is
measured separately, not assumed equal.

Every validation epoch evaluates every selected duration. `val_accuracy` is the
**equal-weight mean of per-duration original-file accuracies**. Shared-width runs
first average the selected widths within each duration, then average durations.
Thus all selected width-duration configurations have equal checkpoint weight.
Counts of crops or batches never weight this checkpoint metric. Per-duration
accuracy and macro-F1 are recorded. ModelCheckpoint, EarlyStopping and LR plateau
control monitor only this exact `val_accuracy`, mode `max`. Best weights are
restored before export, and their precise epoch is reported. Terminal/CSV output
includes accuracy, duration metrics, used/remaining patience, best epoch and LR.
The reusable progress callback prints an epoch summary and a width-duration accuracy
table. After `training.eta_min_epochs` (default 2), ETA uses the moving mean of
the most recent `training.eta_window` epochs (default 5). It excludes export and
can overstate time if early stopping ends training sooner. Timings are also stored
in the primary report; full epoch metrics remain only in history.csv.

## Fixed-crop and sliding-window validation

New configurations and Colab default to `validation.method: sliding_window`.
Historical experiment configs without a `validation` section continue to use
`fixed`; loading or evaluating them does not silently adopt the new default.

```yaml
validation:
  method: sliding_window       # fixed | sliding_window
  stride_seconds: 0.5
  aggregation: mean_probability # mean_probability | max_probability
  batch_size: 32
  include_last_window: true
  short_clip_padding: zero
  cache_features: true
```

`fixed` preserves the existing `evaluation.num_crops` protocol (default 3): evenly
spaced, rounded and deduplicated crop positions, a center crop when count=1,
centered zero padding for short audio, and mean probabilities per original file.
It retains the existing training batch size for inference batching. All
sliding-only settings above are ignored in this mode, including aggregation.

`sliding_window` loads the complete mono/resampled clip, converts duration and
stride to integer samples with `round(seconds * sample_rate)`, and starts windows
at `0, stride_samples, ...` while the window fits. With `include_last_window=true`,
it appends `clip_samples-window_samples` if needed, so the last window reaches the
end exactly and never duplicates an existing start. Setting it false explicitly
omits an unaligned tail. Short or empty clips yield one window, zero-padded on the
right. For a 4-second clip, stride 0.5 produces 7 / 6 / 5 windows for 1 / 1.5 / 2
second models. Stride is configurable; choosing a stride larger than the input
window leaves gaps between windows.

Every window uses the same deterministic Log-Mel frontend and feature
normalization as the fixed protocol, with no random crops, waveform augmentation
or spectrogram masks. DS-CNN logits are softmaxed per window; MobileNet's softmax
outputs are used directly. `mean_probability` averages class probabilities across
windows; `max_probability` takes their per-class maximum. Argmax produces one
prediction per clip, then accuracy and macro-F1 use the original class mapping
(including absent classes as zero F1). `mean_logits` is not implemented.

Every epoch evaluates all selected width-duration configurations. The checkpoint
metric is their **equal-weight mean of clip accuracies**, never pooled window
accuracy. Checkpoint, early stopping and LR reduction still monitor this same
`val_accuracy`. `val_loss` retains mean per-window cross-entropy; it does not select
checkpoints. Changing validation does not alter training crops, augmentations,
shared-width sampling, gradient updates or training-only INT8 calibration.

Windows of the same duration are batched with `validation.batch_size`. The trainer
preprocesses a batch once and runs every selected width on it. A private temporary
cache saves unaugmented resampled audio and exact per-window features across
widths, durations, epochs and post-export evaluation. This avoids decoding or
resampling the same validation clip repeatedly. Caches are isolated per run/config
and removed with the training workspace; no predictions are cached. Disable with
`validation.cache_features=false` to trade extra preprocessing for less temporary
disk usage. RAM is bounded by batches and one clip's windows; the default cache
can occupy hundreds of MB of temporary disk for the full validation fold.
Arbitrary final offsets may not align with the STFT hop, so the implementation
extracts each window exactly instead of slicing a whole-clip spectrogram.

Epoch logs and history.csv separate training and validation seconds and count
unique clips, windows summed across durations, and window predictions across all
widths. The primary report records the protocol, best-epoch counts/runtime,
per-configuration clip accuracy/F1/counts, optional preprocessing/inference timing,
and per-epoch training/validation timing arrays. Inference timings include output
synchronization/host transfer and are host evaluation timings, not MCU latency.
Batching/caching preserve the examples and aggregation; floating-point backend
roundoff is still possible. Protocol settings participate in the existing
configuration fingerprint, so skip-completed never confuses fixed and sliding runs.

```sh
# New default: full-clip sliding window, stride 0.5 seconds, mean probabilities.
"$PY" -m training.train --set validation.method=sliding_window \
  --set validation.stride_seconds=0.5 --set validation.aggregation=mean_probability

# Retained three-crop protocol.
"$PY" -m training.train --set validation.method=fixed --set evaluation.num_crops=3

# Compare a saved model using another host protocol. Original evaluation metrics
# remain under their key; a different protocol gets a distinct report key.
"$PY" -m evaluation.evaluate --model-dir models/dscnn_001 --split val \
  --set validation.method=sliding_window --set validation.stride_seconds=0.25
```

In Colab's existing **Experiment controls** cell, set `validation_method` to
`fixed` or `sliding_window`, `validation_stride` to the step in seconds, and
`validation_aggregation` to `mean_probability` or `max_probability`. These controls
reach both Single Training and Batch Sweep through the existing CLI. Fixed mode
uses YAML `evaluation.num_crops=3`; stride and aggregation controls then have no
effect. The evaluation CLI's existing `--num-crops N` explicitly selects fixed mode.

Sliding validation is a host evaluation policy. TFLite input shapes and model
weight formats remain unchanged. It adds no loops to CREST or hardware profiling:
one-window MCU latency and multi-window full-clip inference are different modes.
Full-clip inference requires one forward pass per window and therefore more total
work; deployment window scheduling and aggregation must be explicitly implemented
and measured separately if desired.

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
Validation uses the selected deterministic fixed-crop or sliding-window protocol
described below; every original file has one aggregated prediction. Calibration
continues to use one center
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

## Architecture and pooling controls

`model.architecture` selects `dscnn` (default) or `mobilenetv2` (MobileNetV2-Tiny).
`model.pooling` applies to both: `avg` averages spatial axes, `max` takes their
maximum, and `mix` concatenates average then max vectors before the classifier.
Mix doubles classifier input channels. Shared classifiers slice the avg and max
channel prefixes separately, preserving exact static-subnet extraction.

MobileNetV2-Tiny uses a standard 3×3 stride-2 stem, then stages with configurable
channels/strides/repeats. Defaults are stem 16, stages `[16,24,32,48]`, strides
`[1,2,2,1]`, repeats `[1,2,2,1]` and expansion ratio 3. Each bottleneck expands with
1×1 + BN + ReLU6, applies 3×3 depthwise + BN + ReLU6, then linear 1×1 projection
+ BN. Residual addition requires stride 1 and equal channels. Global pooling and
a dense 10-class softmax finish the model. DS-CNN retains its original logits
output and BN-free layers; loss/evaluation handle the two output conventions.
`model.conv_type` chooses SC/DW DS-CNN blocks. MobileNet uses the independent
`model.mobile_spatial_conv`: `DW` (default) retains the original inverted residual;
`SC` is an explicit ablation replacing only the 3×3 depthwise layer with a full
3×3 standard convolution from expanded channels to expanded channels. The stem,
expansion/projection, BN/ReLU6, residual rule and width slicing remain unchanged.
SC has more parameters/MACs and is not the standard MobileNetV2 bottleneck. Reports
record its actual `spatial_conv_type`; analytical MACs account for the full kernel.
Missing `mobile_spatial_conv` in historical MobileNet configs always means DW,
even if their old DS-CNN-only `conv_type` field says SC. Original DW model names
remain `MobileNetV2Tiny_{mels}_{width}`; the ablation adds `_SC` before width.

For `model.width=all`, the same convolution Variables receive the summed loss
gradient from every sampled width, with one optimizer update per minibatch.
MobileNet's switchable BN has separate gamma/beta/mean/variance per width; frozen
subnets copy their own BN state. `model.width_options` selects tokens from
`['1','5','25']`. `all_width_training.strategy=all` preserves the previous behavior.
`strategy=sandwich` always samples the largest/smallest selected widths plus
`sandwich_random_widths` intermediate widths without replacement (default 1).
With three widths and default 1, this visits all three; 0 visits endpoints only.
Validation still evaluates every selected width at every selected duration.

Optional `quantization.bn_recalibration_samples` (default 0, off) recomputes each
frozen subnet's BN population statistics before export. Inputs are deterministic,
unaugmented training-fold examples interleaved across selected training durations;
the number is a total sample budget, and the report records actual duration counts.
Use at least one sample per selected duration. Calibration proceeds layer by layer,
using already calibrated upstream BN statistics; gamma/beta and convolution/dense
weights remain unchanged. It can be slow and can change validation accuracy.
Checkpoint selection and the shared-vs-frozen equivalence check use the original
best checkpoint; exported Keras/TFLite evaluation records the recalibrated result.
All duration exports of a subnet share those same calibrated weights. INT8 PTQ
then independently uses training-only representative inputs for each duration.

```sh
# Existing DS-CNN, configurable mix pooling and all three durations.
"$PY" -m training.train --set model.architecture=dscnn --set model.conv_type=DW \
  --set model.pooling=mix --set model.width=all --set 'training.duration_secs=[1,1.5,2]'

# Shared MobileNetV2-Tiny; exports 3 widths × 3 fixed durations × FP32/INT8.
"$PY" -m training.train --set model.architecture=mobilenetv2 --set model.width=all \
  --set model.pooling=avg --set model.expansion_ratio=3 \
  --set 'training.duration_secs=[1,1.5,2]' --set all_width_training.strategy=sandwich

# Optional training-only BN recalibration; no package changes needed.
"$PY" -m training.train --set model.architecture=mobilenetv2 \
  --set quantization.bn_recalibration_samples=96
```

## Models and experiment reports

Canonical model identity remains `DSCNN_{40|64|128}_{SC|DW}_{1|5|25|all}`. Width tokens
1/5/25 mean multipliers 1.0/0.5/0.25. Channels use `max(1,floor(base*multiplier))`.
The standard Conv2D stem, stride-2 blocks, bias/ReLU, BN-free policy, default average pooling and dense
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
checkpoint and static models under `subnets/{canonical_name}/` with exports.
Their compact records are nested under `subnets` in the **one primary report.json**,
with no child JSON/CSV files. Child configs are inherited from that parent report.
The existing report accessors, evaluation CLI and optional CREST registry handle
both this layout and older per-subnet reports. No new
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
Exports audit the actual operator list against reviewed ESP TFLite Micro resolver
registrations, check INT8 concat/max-reduction scale/zero-point restrictions, and
record resolver calls and unreviewed operators. `static_checks_passed` means only
these static checks passed; it is not a firmware compatibility certification.
Global `MEAN`, `REDUCE_MAX` and `CONCATENATION` should be budgeted as reference
kernels, without assuming ESP-NN acceleration. Physical MCU/operator/arena
compatibility still needs board-specific validation;
no firmware, timing/energy or accuracy results are invented.

## Google Colab

Open [`notebooks/colab_training.ipynb`](notebooks/colab_training.ipynb) and run cells
in order: select GPU runtime → mount Drive/check and clone `colab-training` →
create the isolated Python 3.11 environment/install pinned dependencies → set
independent dataset/output paths → edit augmentation and Experiment controls → train → inspect summary.
Experiment controls include DS-CNN/MobileNetV2-Tiny, avg/max/mix pooling, width,
durations/mixed subsets, Mel bins, SC/DW, expansion ratio, shared-width strategy,
optional BN recalibration, seed, epochs and batch size. Batch `architectures` and
`poolings` lists default to the selected values; edit them to extend the Cartesian
sweep. Avoid sweeping both SC/DW for MobileNet alone because it ignores that setting.
The notebook refuses a missing branch and never falls back to main or embeds
credentials. It stages data/work/cache under `/content` and saves final artifacts
to `/content/drive/MyDrive/EmbeddedSystem/models/`. Prior finalized runs survive
runtime loss. The notebook kernel is not rewritten; training subprocesses use
the pinned environment and pip-provided CUDA libraries. Live Colab/physical CUDA
execution must be verified in that runtime; local mocked selection is not a GPU
benchmark. The optional final-test cell is disabled by default.

ESP operator references reviewed for the static audit (firmware must pin and
validate its own version): [resolver registrations](https://github.com/espressif/esp-tflite-micro/blob/master/tensorflow/lite/micro/micro_mutable_op_resolver.h),
[reduction kernels](https://github.com/espressif/esp-tflite-micro/blob/master/tensorflow/lite/micro/kernels/reduce_common.cc),
[concatenation kernel](https://github.com/espressif/esp-tflite-micro/blob/master/tensorflow/lite/micro/kernels/concatenation.cc).
The Raspberry Pi can use desktop TFLite, but Pi timing and the XIAO ESP32-S3 Sense
firmware/arena, preprocessing parity, latency and energy require target testing.
