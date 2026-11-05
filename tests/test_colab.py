import json
from pathlib import Path
import re
import unittest


class ColabTests(unittest.TestCase):
    def test_notebook_valid_python_cli_and_no_credentials(self):
        path = Path('notebooks/colab_training.ipynb')
        nb = json.loads(path.read_text())
        self.assertEqual(nb['nbformat'], 4)
        source = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type']=='code':
                compile(''.join(cell['source']),f'notebook-cell-{i}','exec')
        self.assertIn('training.train',source)
        self.assertIn('training.sweep',source)
        self.assertIn('colab-training',source)
        self.assertIn("'runtime.device': 'auto'",source)
        self.assertIn('run_final_test = False',source)
        self.assertIn('/content/drive/MyDrive/EmbeddedSystem/models',source)
        self.assertNotRegex(source,r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|AKIA[A-Z0-9]{16}')
        self.assertFalse(Path('configs/baseline.yaml').exists())
