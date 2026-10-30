# Raspberry Pi 4 profiling

On-device latency and memory for the `DSCNN_64_SC_all` int8 subnets
(run `20261023T213435Z_774e54f8_ffb532`), plus the same tiny DS-CNN baseline
as `../rp2040/`, measured on a **Raspberry Pi 4 Model B** (4x Cortex-A72) with
**1 and 4 interpreter threads**. Latency and memory only; no energy was
measured.

## Files

| File | What it is |
|---|---|
| `rpi4_summary.csv` | One row per (model, threads): 7 models x {1, 4} threads = 14 rows. Columns below. |
| `<model>_rpi4_t<threads>_<timestamp>.txt` | Raw results for one run: header, summary (including threads, runtime, governor, clock, temperature and throttle flags), one line per timed run (`board=rpi4 model=... run=<k> invoke_us=<int> preprocess_us=-1 arena_used_bytes=-1 status=ok`), and the full runner output. `t1`/`t4` is the thread count. |
| `dscnn_us8k_min_int8_rpi4_t1_20261026_161854.txt` | The baseline's failed attempt **with** XNNPACK, kept as evidence for the note below. Not used in the CSV. |

Host names in the raw files are replaced by `<pi-host>`.

## CSV columns

The columns are those of `../rp2040/rp2040_summary.csv` (see its README),
followed by the Pi-specific columns. Where a shared column means something
different on the Pi:

| Column | Meaning on the Pi |
|---|---|
| `board` | `rpi4`. |
| `flash_bytes` | Size of the `.tflite` file. |
| `static_ram_bytes` | Peak **process-memory (RSS) increase** of the runner, measured from just before the model is loaded to the last timed run. **Not comparable to the RP2040's static RAM**; see Memory below. |
| `arena_used_bytes`, `arena_kb` | Empty: LiteRT on Linux has no fixed tensor arena. |
| `median_us`, `p95_us` | Median and 95th-percentile (nearest-rank) `invoke()` time in microseconds, timed with `time.perf_counter_ns()` around each call. |
| `mac_per_s` | `macs / (median_us / 1e6)`. |
| `fits_stride` | `True` if `median_us` is below `stride_s` (1 s for 1 s exports, 2 s for 2 s exports and the baseline). |
| `core_version` | Runtime: `ai_edge_litert@2.3.0`. |
| `clock_hz` | cpu0 `scaling_cur_freq` at the end of the run: 1,800,000,000 for every row. |
| `results_file` | File name of the row's raw results file, in this folder. |
| `threads` | **Interpreter threads (1 or 4).** Energy will be estimated later with a different active power per thread count, so always group by this column. |
| `xnnpack` | `default` = XNNPACK delegate on; `off` = reference kernels (baseline only, see below). |
| `governor` | cpufreq governor of cpu0 during the run (`performance` everywhere). |
| `temp_after_c` | SoC temperature right after the timed runs. |
| `throttled_before`, `throttled_after` | `vcgencmd get_throttled` before and after the timed runs (`0x0` = no under-voltage, capping or throttling, now or since boot). |
| `flags` | `none` for a clean run. Otherwise `+`-joined: `<condition>_now` (active after the run), `<condition>_during_run` (newly set during it), `throttle_unknown`, or `hot` (>= 80 C). **Every row is `none`.** |

`model`, `width`, `window`, `macs`, `params`, `n_runs`, `stride_s`,
`val_acc`, `val_macro_f1` and `status` mean the same as in the RP2040 CSV,
so the two files can be joined on `model`.

## How the numbers were produced

- **Tooling:** CREST fork [`FleabagEE/crest`](https://github.com/FleabagEE/crest), branch `feat/rpi4-backend`, commit `9415cdd`. The backend is `src/crest/microcontrollers/raspberry_pi4.py`, the runner is `sketches/rpi4/crest_litert_runner.py`, and the scripts are in `analysis_scripts/rpi4_latency_bench/`. The host copies each `.tflite` and the runner to the Pi with `scp` and runs the runner over `ssh`.
- **Board and software:** Raspberry Pi 4 Model B Rev 1.5, 1.8 GiB RAM, Debian 13 (trixie, 64-bit, kernel 6.18), Python 3.13.5, `ai-edge-litert` 2.3.0 in a venv, numpy 2.2.4.
- **Clock and governor:** cpufreq governor `performance` on all 4 cores, at **1.8 GHz**. Before each of the three benchmark sessions the bench checked every core's governor over SSH and would have stopped if any was not `performance`; all were. The governor was set by hand and resets on reboot.
- **Throttle checks:** `vcgencmd get_throttled` and the SoC temperature were read before and after every run. All runs read `0x0` both times; the maximum temperature was 51.1 C. No rows are flagged.
- **Models:** the same files as `../rp2040/`: the `int8.tflite` export of each subnet, unchanged (int8 input and output), plus the CREST baseline.
- **Inputs are dummy data:** synthetic uniform values in [0, 5), quantized with the input scale and zero point, as on the RP2040. No audio and no log-mel, so `preprocess_us` is `-1` and latency is `invoke()` only.
- **Timing:** 5 untimed warm-up `invoke()` calls, then 100 timed calls, with nothing printed during the timed loop. p95 is 0.6-31% above the median; the largest relative spread is on the shortest 4-thread run (SC_25 1s, 174 vs. 228 us).
- **XNNPACK note (baseline):** the six course models ran with the default XNNPACK delegate. **The baseline (`dscnn_us8k_min_int8`) ran with XNNPACK off** (`xnnpack=off`). Its graph has a dynamic batch dimension (input signature `[-1, 201, 64]`, plus SHAPE/STRIDED_SLICE/PACK ops before the reshape), and LiteRT 2.3.0's XNNPACK delegate fails to prepare it (`failed to create XNNPACK runtime`; see the `..._161854.txt` file). Its numbers are therefore reference-kernel numbers and are not directly comparable to the course models' XNNPACK numbers. Re-exporting it with a fixed batch of 1 would allow XNNPACK, but would change the graph compared with the one measured on the RP2040.
- **Accuracy:** validation fold only, copied from each subnet's `run.json`; the held-out test fold has not been evaluated.

## Memory: why it is not comparable to the RP2040

On the RP2040, `static_ram_bytes` is the linker's static RAM (mbed + TFLM +
sketch + the tensor arena) and `arena_used_bytes` is the exact TFLM arena.
On the Pi, `static_ram_bytes` is how much the Linux process's resident memory
grew between loading the model and the last timed run (the `VmHWM` peak after
resetting it through `/proc/self/clear_refs`). That growth is 10.1-11.5 MB
for **every** model, and is dominated by interpreter and XNNPACK setup; only
about 1.5 MB varies with the model. Use it to check that a model fits
comfortably on the Pi, not to compare memory across boards.

## Regenerating the CSV

In a CREST checkout at `feat/rpi4-backend` with the `crest` conda env, after
staging the models (see `../rp2040/README.md`) and running
`analysis_scripts/rpi4_latency_bench/run_rpi4_latency_bench.py` with
`--threads 1` and `--threads 4` (add `--no-xnnpack` for the baseline):

```bash
python analysis_scripts/rpi4_latency_bench/summarize_rpi4_results.py --threads 1 4 \
  --models dscnn_us8k_min_int8 \
    DSCNN_64_SC_25_1s_int8 DSCNN_64_SC_25_2s_int8 \
    DSCNN_64_SC_5_1s_int8 DSCNN_64_SC_5_2s_int8 \
    DSCNN_64_SC_1_1s_int8 DSCNN_64_SC_1_2s_int8
```

It writes `analysis_scripts/rpi4_latency_bench/results/rpi4_summary.csv`.
The copy here has `results_file` reduced to file names.
