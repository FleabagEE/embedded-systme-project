---
title: "Right Model, Right Board"
---

# Right Model, Right Board: Joint Model–Platform Selection for Acoustic Event Detection

**Group 5 (P05)** · Ching-Yu Cheng and Ping-Hung Chen
Mentor: Joseph Zales · ECE M202A, Fall 2026, Prof. Mani Srivastava

## Midterm checkpoint (Nov 10, 2026)
- [Checkpoint slides (PDF)](P05_Checkpoint.pdf)

## Research question
Can we jointly select a neural network configuration and hardware platform that meets
application-specific accuracy and latency constraints at minimum energy, while minimizing
the amount of hardware profiling required?

## Approach
We extend CREST (Zales, Sharma, Srivastava, arXiv:2606.15004) so that the board becomes a
search variable. Given an application spec (window L, stride S, minimum accuracy,
response limit R), the selector picks the model–board pair with the lowest energy per
inference. Energy is computed from measured time per mode × per-board power.

- Dataset: UrbanSound8K (Salamon et al., ACM MM 2014)
- Boards: Arduino Nano RP2040 Connect, XIAO ESP32-S3 Sense, Raspberry Pi 4
- Models: int8 DS-CNN and MobileNetV2 variants (24 configurations, 1 s / 2 s windows)

## Status (preliminary)
- Three new CREST backends; the same int8 model runs and is timed on all three boards
- On-board latency and memory for 6 of 48 models
- Integer log-mel front end on both MCUs, bit-exact against the host implementation
- Energy is currently a datasheet-based estimate; per-board power calibration with an INA228 is next

## Code
- Project repo: <https://github.com/FleabagEE/embedded-systme-project>
- CREST fork: <https://github.com/FleabagEE/crest>
