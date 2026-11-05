"""Static export checks against reviewed ESP TFLite Micro registrations.

This is deliberately not a firmware/arena validation or an ESP-NN speed claim.
Sources reviewed 2026-11-04: Espressif's MicroMutableOpResolver, reduce_common.cc
and concatenation.cc. No network access is required during export.
"""
import numpy as np

SOURCE = 'https://github.com/espressif/esp-tflite-micro/tree/master/tensorflow/lite/micro'
REGISTRATIONS = {
    'CONV_2D': 'AddConv2D', 'DEPTHWISE_CONV_2D': 'AddDepthwiseConv2D',
    'FULLY_CONNECTED': 'AddFullyConnected', 'ADD': 'AddAdd',
    'MEAN': 'AddMean', 'REDUCE_MAX': 'AddReduceMax',
    'CONCATENATION': 'AddConcatenation', 'SOFTMAX': 'AddSoftmax',
    'RESHAPE': 'AddReshape', 'RELU': 'AddRelu', 'RELU6': 'AddRelu6',
    'QUANTIZE': 'AddQuantize', 'AVERAGE_POOL_2D': 'AddAveragePool2D',
    'MAX_POOL_2D': 'AddMaxPool2D',
}


def audit_operators(runner):
    interpreter = runner.interpreter
    ops = [op for op in interpreter._get_ops_details() if op['op_name'] != 'DELEGATE']
    names = sorted({op['op_name'] for op in ops})
    tensors = {t['index']: t for t in interpreter.get_tensor_details()}
    unknown = sorted(set(names) - REGISTRATIONS.keys())
    issues, checks = [], []
    for op in ops:
        name = op['op_name']
        if name not in ('CONCATENATION', 'REDUCE_MAX'):
            continue
        # Micro's int8 concat/max kernels require equal input/output quantization.
        # REDUCE_MAX's second input is an int32 axis, not an activation.
        inputs = op['inputs'] if name == 'CONCATENATION' else op['inputs'][:1]
        related = [tensors[int(i)] for i in [*inputs, *op['outputs']] if int(i) >= 0]
        if not related or related[0]['dtype'] != np.int8:
            continue
        same = all(t['dtype'] == np.int8 and t['quantization'] == related[0]['quantization'] for t in related)
        checks.append({'operator': name, 'matching_int8_quantization': same})
        if not same:
            issues.append(f'{name}: Micro requires identical int8 scales and zero points')
    signature = runner.input.get('shape_signature', runner.input['shape'])
    concrete = all(int(v) > 0 for v in signature) and np.array_equal(signature, runner.input['shape'])
    if not concrete:
        issues.append('Input allocation must have a concrete fixed shape')
    return {
        'status': 'static_checks_passed' if not unknown and not issues else 'requires_review',
        'operators': names,
        'required_resolver_calls': [REGISTRATIONS[n] for n in names if n in REGISTRATIONS],
        'unreviewed_operators': unknown, 'issues': issues, 'quantization_checks': checks,
        'fixed_input_shape': concrete, 'runtime_resize_required': False,
        'reference_kernels_to_budget': sorted(set(names) & {'MEAN', 'REDUCE_MAX', 'CONCATENATION'}),
        'source': SOURCE, 'source_reviewed_date': '2026-11-04',
        'firmware_invocation_verified': False, 'esp_nn_optimization_verified': False,
        'tensor_arena_bytes': None,
        'limitation': 'Registration/static checks only. Verify operator versions, firmware kernels, arena, frontend parity and on-board invocation on the target; desktop TFLite success is insufficient.',
    }
