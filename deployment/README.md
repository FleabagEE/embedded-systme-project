Device adapters consume fixed-batch, fixed-duration TFLite artifacts and the
primary experiment report. Export performs CPU invocation and a static Micro
operator audit (including int8 concat/reduce-max quantization constraints).
`static_checks_passed` does not certify an ESP32-S3 firmware build.

The report lists required MicroMutableOpResolver calls and unreviewed operators.
BN is folded into convolution at conversion; shared channel slicing and
switchable BN stay in the trainer, while deployed subnets are ordinary static
models. Budget global MEAN/REDUCE_MAX/CONCATENATION as reference kernels unless
your firmware demonstrates an optimized path. Verify operator versions, resolver
registration, actual ESP-NN dispatch, frontend parity, tensor arena and invocation
with the target firmware. Raspberry Pi host execution also needs device testing.
No board firmware, Tensor Arena measurements, latency or energy results are
claimed by this project. Missing device measurements remain null.

Fixed-crop and sliding-window validation are host evaluation protocols. Exported
model inputs remain a single fixed-duration window. Full-clip sliding inference
requires multiple model invocations plus host/application aggregation. This does
not change the definition of single-window CREST/MCU latency, the profiling schema,
or introduce deployment window loops. Reported validation seconds include host
preprocessing/aggregation and must not be interpreted as target-device latency.
