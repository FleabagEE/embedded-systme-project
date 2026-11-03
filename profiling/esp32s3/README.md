# Seeed XIAO ESP32-S3 Sense profiling

On-device latency and memory for the `DSCNN_64_SC_all` int8 subnets
(run `20261023T213435Z_774e54f8_ffb532`), plus the same tiny DS-CNN baseline
as `../rp2040/` and `../rpi4/`, measured on a **Seeed XIAO ESP32-S3 Sense**
(ESP32-S3 rev v0.2, dual Xtensa LX7, 8 MB PSRAM). Latency and memory only; no
energy was measured.

## Files

| File | What it is |
|---|---|
| `esp32s3_summary.csv` | One row per model (7 rows). Same columns as `../rp2040/rp2040_summary.csv`, plus `memory_location`. |
| `xiao_esp32s3_<model>_full_a<KiB>_<timestamp>.txt` | Raw results of each model's full run: header, summary (including the board's `env` line with clock, radio state and arena placement), one line per timed run (`board=xiao_esp32s3 model=... run=<k> invoke_us=<int> preprocess_us=-1 arena_used_bytes=<int> status=ok`), and the full serial log. `a<KiB>` is the compiled tensor-arena size. |

## CSV columns

All columns mean the same as in `../rp2040/README.md`. ESP32-S3 values:

| Column | On this board |
|---|---|
| `board` | `xiao_esp32s3`. |
| `flash_bytes` | Program flash from the `arduino-cli` compile summary (sketch + core + TFLM + model), of a 3 MB app partition. |
| `static_ram_bytes` | Static DRAM from the compile summary. It **includes the tensor arena**, which is a static array in internal SRAM, plus 45,780 B for the rest of the sketch. The board limit is 327,680 B. |
| `arena_used_bytes` | Tensor-arena bytes actually used, from `interpreter->arena_used_bytes()` after `AllocateTensors()`. |
| `core_version` | `esp32:esp32@3.3.12`. |
| `clock_hz` | 240,000,000. The CPU clock is set in `setup()` and confirmed by the board (`cpu_mhz=240` in every run). |
| `memory_location` | Where the tensor arena lived: `internal_sram` for every model. The board checks this at run time with `esp_ptr_external_ram()`; every arena was at `0x3FC9BAE0`. The PSRAM fallback was never needed. |

## How the numbers were produced

- **Tooling:** CREST fork [`FleabagEE/crest`](https://github.com/FleabagEE/crest), branch `feat/esp32s3-backend`, commit `c750c4c`. The backend is `src/crest/microcontrollers/xiao_esp32s3.py`, and the sketch is the same `sketches/crest_inference_latency.ino` as on the RP2040 (ESP32 branch).
- **Toolchain:** esp32 Arduino core **3.3.12**, FQBN `esp32:esp32:XIAO_ESP32S3`. Board options: `USBMode=hwcdc`, `CPUFreq=240`, `PartitionScheme=default_8MB`, `PSRAM=disabled`, `DebugLevel=none`.
- **TFLM: esp-tflite-micro with ESP-NN active.** The core ships a prebuilt `esp-tflite-micro` linked with Espressif's ESP-NN. Its conv, depthwise-conv and fully-connected kernels call the ESP32-S3-optimized `esp_nn_*_esp32s3` routines, and these are present in the linked firmware. The Chirale TFLM used on the RP2040 is not compiled on this board. All 8 ops the models use are registered explicitly.
- **Clock and radios:** 240 MHz, **Wi-Fi off** (`WiFi.mode(WIFI_OFF)`, `wifi_mode=0`) and **Bluetooth off** (`btStop()`, `bt_started=0`), confirmed in every run's `env` line.
- **Inputs are dummy data:** synthetic uniform values in [0, 5), quantized with the input scale and zero point, as on the other boards. No audio and no log-mel, so `preprocess_us` is `-1` and latency is `Invoke()` only.
- **Timing:** 5 untimed warm-up `Invoke()` calls, then 100 timed calls with `micros()`, with nothing printed during the timed window. Run-to-run spread is under 0.04%.
- **Accuracy:** validation fold only (fold 9), copied from each subnet's `run.json`. The held-out test fold has not been evaluated.

## Observation: ESP-NN needs more tensor arena

ESP-NN's kernels allocate scratch buffers inside the tensor arena, so the
arena these models need on the ESP32-S3 is larger than on the RP2040
(CMSIS-NN). Ratio of the ESP32-S3 `arena_used_bytes` to a static estimate
calibrated on the RP2040 (peak live activations + conv scratch + per-channel
quantization + 2 KiB):

| Model | Ratio |
|---|---|
| baseline | 0.99 |
| SC_25 1s / 2s | 1.64 / 1.66 |
| SC_5 1s / 2s | 1.17 / 1.04 |
| SC_1 1s / 2s | 2.08 / 1.19 |

The ratio is **not** constant: it ranges from 0.99 to 2.08, and a
least-squares fit over the first five models (1.18) underpredicts SC_1 1s.

**Unverified hypothesis:** the full-width models' numbers would be explained
if ESP-NN's convolution scratch holds a copy of the layer's filter weights.
The 128→128 layer has 147,456 B of weights. Adding that to the layer's live
activations and about 5 KB of fixed overhead gives about 170 KB (1 s) and
about 189 KB (2 s), within about 1% of the measured 170,044 / 188,828 B.
This has **not** been checked against the ESP-NN source and should not be
quoted as the mechanism.

## Regenerating the CSV

In a CREST checkout at `feat/esp32s3-backend` with the `crest` conda env,
after staging the models and running
`analysis_scripts/rp2040_latency_bench/run_rp2040_bench_plan.py --board xiao_esp32s3`:

```bash
python analysis_scripts/rp2040_latency_bench/summarize_rp2040_results.py --board xiao_esp32s3 \
  --models dscnn_us8k_min_int8 \
    DSCNN_64_SC_25_1s_int8 DSCNN_64_SC_25_2s_int8 \
    DSCNN_64_SC_5_1s_int8 DSCNN_64_SC_5_2s_int8 \
    DSCNN_64_SC_1_1s_int8 DSCNN_64_SC_1_2s_int8
```

It writes `analysis_scripts/rp2040_latency_bench/results/esp32s3_summary.csv`.
The copy here has `results_file` reduced to file names.
