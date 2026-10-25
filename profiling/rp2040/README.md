# Arduino Nano RP2040 Connect profiling

On-device latency and memory for the `DSCNN_64_SC_all` int8 subnets
(run `20261023T213435Z_774e54f8_ffb532`), plus a tiny DS-CNN baseline,
measured on an **Arduino Nano RP2040 Connect** (RP2040, Cortex-M0+).
Latency and memory only; no energy measurement.

## Files

| File | What it is |
|---|---|
| `rp2040_summary.csv` | One row per (model, board). The main table; columns below. |
| `DSCNN_64_SC_{1,5,25}_{1s,2s}_int8_full_a<KiB>_<timestamp>.txt` | Raw results for each subnet's full run: header, memory, summary, then one line per timed run (`board=... model=... run=<k> invoke_us=<int> preprocess_us=-1 arena_used_bytes=<int> status=ok`) and the full serial log. `a<KiB>` is the compiled tensor-arena size. |
| `dscnn_us8k_min_int8_20261024_170252.txt` | Same format, for the baseline: the smallest DS-CNN from CREST's UrbanSound8K search space (2 depthwise-separable blocks, 128 parameters, untrained, input 201x64). |
| `tflite_analysis_20261025.json` | Static analysis of each `.tflite`: per-layer op, shapes, MACs and live activation bytes, total MACs, and peak activation memory. Paths inside are from the measurement host. |

## CSV columns

| Column | Meaning |
|---|---|
| `model` | Candidate name: `<subnet>_<duration>_int8`, or `dscnn_us8k_min_int8` for the baseline. |
| `board` | `nano_rp2040_connect`. |
| `width` | Channel-width multiplier (`training/utils.py` `WIDTHS`: `_1` = 1.0, `_5` = 0.5, `_25` = 0.25). Empty for the baseline. |
| `window` | Input frames: 98 (1 s export), 198 (2 s export), 201 (baseline). Every model has 64 mel bins. |
| `macs` | Multiply-accumulates per inference, conv and dense layers only. Matches `estimated_macs` in each subnet's `run.json`. |
| `params` | Parameter count (`run.json` `num_parameters`; 128 for the baseline). |
| `flash_bytes` | Program flash from the `arduino-cli` compile summary (sketch + TFLM + model). The board has 16 MiB. |
| `static_ram_bytes` | Static RAM from the compile summary. **Includes the tensor arena**, which is a static array, plus ~50 KB of mbed/TFLM/sketch. |
| `arena_used_bytes` | Tensor-arena bytes actually used, from `interpreter->arena_used_bytes()` after `AllocateTensors()`. This is the real arena requirement. |
| `median_us`, `p95_us` | Median and 95th-percentile (nearest-rank) `Invoke()` time in microseconds, timed with `micros()` around each call. |
| `n_runs` | Timed runs in the full run (warm-up excluded). |
| `mac_per_s` | `macs / (median_us / 1e6)`. |
| `stride_s` | Required decision period: 1 s for 1 s exports, 2 s for 2 s exports and the baseline. |
| `fits_stride` | `True` if `median_us` is below `stride_s`. |
| `val_acc`, `val_macro_f1` | int8 TFLite accuracy / macro-F1 on the **validation fold (fold 9)**, from `run.json`. Empty for the untrained baseline. |
| `core_version` | Arduino core used to build: `arduino:mbed_nano@4.6.0`. |
| `clock_hz` | CPU clock: 125,000,000 Hz (see below). |
| `arena_kb` | Compiled tensor-arena size in KiB (must be at least `arena_used_bytes`). |
| `status` | `ok` for a successful run. |
| `results_file` | Path of the raw file in the CREST repo; the same file name is in this folder. |

Mapping to `profiling/schema.json`:
- `effective_width` <- the subnet suffix
- `duration_sec` <- `stride_s`
- `latency_ms` <- `median_us / 1000`
- `flash_bytes` <- `flash_bytes`
- `peak_sram_bytes` <- `static_ram_bytes`
- `tensor_arena_bytes` <- `arena_used_bytes`
- `energy_mj` stays null: no energy was measured.

## How the numbers were produced

- **Tooling:** CREST fork [`FleabagEE/crest`](https://github.com/FleabagEE/crest), branch `feat/rp2040-backend`, commit `089baab`. The backend is `src/crest/microcontrollers/arduino_nano_rp2040.py`, the sketch is `sketches/crest_inference_latency.ino`, and the scripts are in `analysis_scripts/rp2040_latency_bench/`.
- **Toolchain:** `arduino-cli` 1.5.1 with `arduino:mbed_nano@4.6.0` (FQBN `arduino:mbed_nano:nanorp2040connect`), TFLM from `Chirale_TensorFlowLite` (CREST's pinned submodule), with all ops resolved.
- **Clock: 125 MHz.** Read from the core's compiled `clocks_init` in `libmbed.a` (PLL VCO 1500 MHz / 6 / 2). Not measured at runtime.
- **Models:** the `int8.tflite` export of each subnet, used unchanged (int8 input and output). fp32 exports were not tested.
- **Inputs are dummy data:** synthetic uniform values in [0, 5), quantized with the input scale and zero point. No audio and no on-device log-mel, so `preprocess_us` is `-1` and latency is `Invoke()` only. Latency does not depend on input values for these ops.
- **Timing:** 5 untimed warm-up `Invoke()` calls, then N timed calls with nothing printed during the timed window. N was set from a 10-run probe so each full run takes about 2 minutes: 100 runs (at most) for the fast models, 88 for SC_5 2s, 37 for SC_1 1s, 30 (minimum) for SC_1 2s. Run-to-run spread was under 0.03%.
- **Arena:** set per model from a static estimate (peak live activations + conv scratch + per-channel quantization + 2 KiB, plus 10%). Every model allocated at the first size tried.
- **Accuracy:** validation fold only. The held-out test fold has not been evaluated (`held_out_test_evaluated: False` in `run.json`).

## Regenerating the CSV

In a CREST checkout at `feat/rp2040-backend` with the `crest` conda env, after staging the models and running the bench (see `analysis_scripts/rp2040_latency_bench/README.md`):

```bash
python analysis_scripts/rp2040_latency_bench/summarize_rp2040_results.py \
  --models dscnn_us8k_min_int8 \
    DSCNN_64_SC_25_1s_int8 DSCNN_64_SC_25_2s_int8 \
    DSCNN_64_SC_5_1s_int8 DSCNN_64_SC_5_2s_int8 \
    DSCNN_64_SC_1_1s_int8 DSCNN_64_SC_1_2s_int8
```

It reads the candidates in `models/rp2040_bench/` and the raw files in `analysis_scripts/rp2040_latency_bench/results/`, and writes `analysis_scripts/rp2040_latency_bench/results/rp2040_summary.csv`.
