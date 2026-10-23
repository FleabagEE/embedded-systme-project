import unittest
import numpy as np
from evaluation.metrics import classification_metrics


class MetricsTests(unittest.TestCase):
    def test_original_file_aggregation_and_absent_classes(self):
        result = classification_metrics([0, 0, 1], np.array([[5., 0], [4., 0], [0., 5]]), ['a', 'a', 'b'], {0: 'a', 1: 'b'}, 'val')
        self.assertEqual(result['num_original_files'], 2)
        self.assertEqual(result['accuracy'], 1)
        self.assertEqual(result['macro_f1'], 1)
        with self.assertRaises(ValueError):
            classification_metrics([0, 1], np.ones((2, 2)), ['same', 'same'], {0: 'a', 1: 'b'}, 'val')
