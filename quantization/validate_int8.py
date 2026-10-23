"""CPU TFLite inference with explicit quantization and dequantization."""
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf


class TFLiteRunner:
    def __init__(self, model_path=None, model_content=None, require_int8=False):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path) if model_path else None,
                                               model_content=model_content, num_threads=1)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.cpu_invocation_success = False
        if require_int8:
            if self.input['dtype'] != np.int8 or self.output['dtype'] != np.int8:
                raise ValueError('Full INT8 requires int8 input AND output')
            floating = [t['name'] for t in self.interpreter.get_tensor_details() if np.issubdtype(t['dtype'], np.floating)]
            if floating:
                raise ValueError(f'Unexpected float tensors in fully integer model: {floating}')
        self.require_int8 = require_int8

    def __call__(self, batch):
        results = []
        for sample in np.asarray(batch, np.float32):
            value = sample[None]
            if tuple(value.shape) != tuple(self.input['shape']):
                raise ValueError(f"Fixed TFLite shape {self.input['shape'].tolist()} does not match {value.shape}; export the requested duration")
            if self.input['dtype'] == np.int8:
                scale, zero = self.input['quantization']
                if scale <= 0:
                    raise ValueError('Invalid input quantization scale')
                value = np.clip(np.rint(value / scale) + zero, -128, 127).astype(np.int8)
            self.interpreter.set_tensor(self.input['index'], value.astype(self.input['dtype']))
            self.interpreter.invoke()
            self.cpu_invocation_success = True
            output = self.interpreter.get_tensor(self.output['index'])
            if self.output['dtype'] == np.int8:
                scale, zero = self.output['quantization']
                output = (output.astype(np.float32) - zero) * scale
            results.append(output[0])
        return np.stack(results)

    def metadata(self):
        def detail(t):
            scale, zero = t['quantization']
            return {'shape': t['shape'].tolist(), 'dtype': np.dtype(t['dtype']).name,
                    'scale': float(scale), 'zero_point': int(zero)}
        # Private introspection is isolated here; inference uses public Interpreter APIs.
        operators = [op['op_name'] for op in self.interpreter._get_ops_details()]
        return {'input': detail(self.input), 'output': detail(self.output), 'operators': operators,
                'cpu_invocation_success': self.cpu_invocation_success, 'tensor_arena_bytes': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    args = parser.parse_args()
    runner = TFLiteRunner(args.model, require_int8=True)
    output = runner(np.zeros(runner.input['shape'], np.float32))
    print({'output_shape': output.shape, 'metadata': runner.metadata(), 'cpu_invocation_success': True})


if __name__ == '__main__':
    main()
