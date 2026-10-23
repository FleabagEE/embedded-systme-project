import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from tests.common import config
from training.runtime import runtime_info, select_device, verify_training_device


class RuntimeSelectionTests(unittest.TestCase):
    def fake_tf(self, gpus):
        tf = Mock()
        tf.__version__ = '2.15.1'
        tf.config.list_physical_devices.return_value = gpus
        return tf

    def test_auto_prefers_gpu(self):
        gpu = Mock(name='gpu')
        gpu.name = '/physical_device:GPU:0'
        tf = self.fake_tf([gpu])
        with patch('training.runtime._selection', {}), patch('training.runtime.platform.system', return_value='Darwin'):
            self.assertEqual(select_device(tf, {'device': 'auto', 'deterministic': True}), '/GPU:0')
            tf.config.list_physical_devices.assert_called_once_with('GPU')
            tf.config.set_visible_devices.assert_called_once_with([gpu], 'GPU')
            self.assertEqual(runtime_info()['backend'], 'Metal')
            # Detection is deliberately not claimed as execution verification.
            self.assertFalse(runtime_info()['gpu_execution_verified'])

    def test_auto_cpu_fallback_and_explicit_cpu(self):
        for requested, gpus in [('auto', []), ('cpu', [Mock(name='gpu')])]:
            tf = self.fake_tf(gpus)
            with patch('training.runtime._selection', {}):
                self.assertEqual(select_device(tf, {'device': requested, 'deterministic': True}), '/CPU:0')
                tf.config.set_visible_devices.assert_called_once_with([], 'GPU')

    def test_explicit_gpu_missing_and_late_visibility_change_fail(self):
        with self.assertRaisesRegex(RuntimeError, 'GPU requested'):
            select_device(self.fake_tf([]), {'device': 'gpu', 'deterministic': True})
        tf = self.fake_tf([])
        tf.config.set_visible_devices.side_effect = RuntimeError('initialized')
        with self.assertRaisesRegex(RuntimeError, 'Restart Python'):
            select_device(tf, {'device': 'cpu', 'deterministic': True})

    def test_real_cpu_training_probe_and_seed_restoration(self):
        import tensorflow as tf
        for conv, width in [('SC', '5'), ('DW', '5'), ('DW', 'all')]:
            c = config()
            c['model'].update(conv_type=conv, width=width)
            report = verify_training_device(c)
            self.assertTrue(report['weights_changed'])
            self.assertEqual(report['optimizer_iterations'], 1)
            self.assertFalse(report['gpu_execution_verified'])
            self.assertTrue(all('CPU:0' in d for d in report['logits_devices'].values()))
            self.assertTrue(report['repeated_forward_and_gradients_bitwise_equal'])
            np.testing.assert_equal(np.random.random(), np.random.RandomState(c['training']['seed']).random_sample())

    def test_module_cli_writes_execution_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'device.json'
            result = subprocess.run([sys.executable, '-m', 'training.runtime', '--set', 'runtime.device=cpu',
                                     '--set', 'model.base_channels=[8,12]', '--set', 'model.num_blocks=2',
                                     '--output', str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report['selected_device'], '/CPU:0')
            self.assertTrue(report['training_probe']['weights_changed'])
            self.assertFalse(report['gpu_execution_verified'])
