"""Collision-safe run directories and process-locked atomic CSV registry updates."""
import csv
import datetime as dt
import fcntl
import io
import re
import uuid
from pathlib import Path
import yaml

from training.utils import atomic_write, canonical_name, digest, duration_token, run_duration_token
from training.artifacts import read_report, config_path

FIELDS = ['run_id', 'canonical_name', 'duration_sec', 'n_mels', 'conv_type', 'requested_width',
          'effective_width', 'trained_parent', 'train_folds', 'val_folds', 'seed',
          'num_parameters', 'estimated_macs', 'fp32_bytes', 'int8_bytes',
          'validation_accuracy_fp32', 'validation_macro_f1_fp32',
          'validation_accuracy_int8', 'validation_macro_f1_int8', 'status',
          'config_path', 'weight_path', 'tflite_path', 'data_source', 'artifact_id',
          'diagnostic_subset', 'validation_num_original_files', 'training_duration_secs']


def create_run(config, run_id=None):
    run_id = run_id or f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}_{digest(config)[:8]}_{uuid.uuid4().hex[:6]}"
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise ValueError('run_id must be a safe single directory component')
    run = Path(config['runtime']['output_dir']).resolve() / run_duration_token(config) / canonical_name(config) / run_id
    run.mkdir(parents=True, exist_ok=False)
    atomic_write(run / 'config.yaml', yaml.safe_dump(config, sort_keys=False))
    return run


def upsert_registry(output_root, row):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'model_registry.csv'
    with (root / '.registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        if path.exists():
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
        key = str(row['artifact_id'])
        existing = next((r for r in rows if r['artifact_id'] == key), None)
        if existing is None:
            rows.append({k: row.get(k, '') for k in FIELDS})
        else:
            existing.update(row)
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        atomic_write(path, stream.getvalue())


def register_artifact(artifact):
    artifact = Path(artifact).resolve()
    report = read_report(artifact)
    config = yaml.safe_load(config_path(artifact).read_text())
    durations = [float(tag[:-1]) for tag in report.get('exports', {})] or [config['audio']['duration_sec']]
    for duration in durations:
        _register_duration(artifact, report, config, duration)


def _register_duration(artifact, report, config, duration):
    meta = report['metadata']
    tag = duration_token(duration)
    arch = report.get('exports', {}).get(tag, {}).get('fp32', {}).get('architecture', report['architecture'])
    metrics = report.get('evaluation', {}).get(tag, {}).get('val', {})
    fp = metrics.get('keras_fp32', {})
    it = metrics.get('int8_tflite', {})
    exp = artifact / 'exports' / duration_token(duration)
    row = {'run_id': meta['run_id'], 'canonical_name': meta['canonical_name'],
           'duration_sec': duration, 'n_mels': config['audio']['n_mels'],
           'conv_type': config['model']['conv_type'], 'requested_width': meta['requested_width'],
           'effective_width': meta['effective_width'], 'trained_parent': meta.get('trained_parent'),
           'train_folds': config['dataset']['train_folds'], 'val_folds': config['dataset']['val_folds'],
           'seed': config['training']['seed'], 'num_parameters': arch['num_parameters'],
           'estimated_macs': arch['estimated_macs'],
           'fp32_bytes': (exp / 'fp32.tflite').stat().st_size if (exp / 'fp32.tflite').exists() else None,
           'int8_bytes': (exp / 'int8.tflite').stat().st_size if (exp / 'int8.tflite').exists() else None,
           'validation_accuracy_fp32': fp.get('accuracy') if fp.get('split') == 'val' else None,
           'validation_macro_f1_fp32': fp.get('macro_f1') if fp.get('split') == 'val' else None,
           'validation_accuracy_int8': it.get('accuracy') if it.get('split') == 'val' else None,
           'validation_macro_f1_int8': it.get('macro_f1') if it.get('split') == 'val' else None,
           'status': report.get('state', {}).get('status', meta.get('status', 'trained')), 'config_path': str(config_path(artifact)),
           'weight_path': str(artifact / 'best.keras'), 'tflite_path': str(exp / 'int8.tflite'),
           'data_source': meta['data_source'],
           'artifact_id': str(artifact) if duration == config['audio']['duration_sec'] else f'{artifact}::{tag}',
           'diagnostic_subset': fp.get('diagnostic_subset', meta.get('diagnostic_subset')),
           'validation_num_original_files': fp.get('num_original_files') if fp.get('split') == 'val' else None,
           'training_duration_secs': meta.get('training_duration_secs', [duration])}
    upsert_registry(meta['registry_root'], row)
