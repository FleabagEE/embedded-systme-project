"""Collision-safe run directories and process-locked atomic CSV registry updates."""
import csv
import json
import fcntl
import io
import re
from pathlib import Path
import yaml

from training.utils import atomic_write, duration_token, load_config, software_versions
from training.artifacts import read_report, config_path


def experiment_key(config, synthetic=False, limit=None):
    import copy
    from training.utils import digest, training_durations
    stable = copy.deepcopy(config)
    stable['audio']['duration_sec'] = training_durations(stable)[0]
    stable['runtime'] = {'deterministic': stable['runtime']['deterministic']}
    stable['dataset'].pop('cache_dir', None)
    path = Path(stable['dataset'].pop('root')).expanduser() / 'metadata' / 'UrbanSound8K.csv'
    import hashlib
    stable['dataset']['metadata_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() and not synthetic else None
    return digest({'config': stable, 'synthetic': synthetic, 'limit': limit,
                   'source_sha256': software_versions()['source_sha256']})


def find_completed(config, synthetic=False, limit=None):
    import hashlib
    key = experiment_key(config, synthetic, limit)
    for path in sorted(Path(config['runtime']['output_dir']).glob('*/report.json')):
        report = read_report(path.parent)
        if report.get('state', {}).get('status') != 'complete' or report.get('configuration_hash') != key:
            continue
        artifacts = [path.parent] + [path.parent / r for r in report.get('artifacts', []) if r != '.']
        valid = True
        for artifact in artifacts:
            hashes = read_report(artifact).get('file_hashes', {})
            if not hashes or any(not (artifact / name).is_file() or hashlib.sha256((artifact / name).read_bytes()).hexdigest() != value for name, value in hashes.items()):
                valid = False
                break
        if valid:
            return path.parent.resolve()
    return None

FIELDS = ['run_id', 'canonical_name', 'duration_sec', 'n_mels', 'conv_type', 'requested_width',
          'effective_width', 'trained_parent', 'train_folds', 'val_folds', 'seed',
          'num_parameters', 'estimated_macs', 'fp32_bytes', 'int8_bytes',
          'validation_accuracy_fp32', 'validation_macro_f1_fp32',
          'validation_accuracy_int8', 'validation_macro_f1_int8', 'status',
          'config_path', 'weight_path', 'tflite_path', 'data_source', 'artifact_id',
          'diagnostic_subset', 'validation_num_original_files', 'training_duration_secs']


def create_run(config, run_id=None, *, sequence_start=None):
    """Durable reservations + atomic mkdir prevent reuse, even after a failed/deleted run."""
    from training.artifacts import utc_now, update_report, report_lock
    if sequence_start is not None and (type(sequence_start) is not int or sequence_start < 1):
        raise ValueError('sequence_start must be a positive integer')
    architecture = config['model'].get('architecture', 'dscnn').lower()
    if not re.fullmatch(r'[a-z][a-z0-9_-]*', architecture):
        raise ValueError('architecture must be a safe lowercase name')
    root = Path(config['runtime']['output_dir']).expanduser().resolve()
    reservations = root / '.allocations'
    reservations.mkdir(parents=True, exist_ok=True)
    # Serialize scanning/reservation across architectures. Durable mkdir
    # reservations still protect IDs after failed or deleted experiments.
    with report_lock(reservations).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pattern = re.compile((re.escape(architecture) if sequence_start is None else r'[a-z][a-z0-9_-]*') + r'_(\d+)$')
        used = [int(match.group(1)) for folder in (root, reservations) for p in folder.iterdir()
                if (match := pattern.fullmatch(p.name))]
        sequence = max(max(used, default=0) + 1, sequence_start or 1)
        while True:
            name = f'{architecture}_{sequence:03d}'
            try:
                (reservations / name).mkdir()
            except FileExistsError:
                sequence += 1
                continue
            run = root / name
            try:
                run.mkdir()
            except FileExistsError:
                sequence += 1
                continue
            update_report(run, schema_version=4, centralized_artifacts=True, experiment={'id': name, 'sequence': sequence,
                          'architecture': architecture, 'label': run_id,
                          'sequence_policy': 'global' if sequence_start is not None else 'architecture'}, config=config,
                          seed=config['training']['seed'], software_versions=software_versions(),
                          state={'status': 'allocated', 'created_at': utc_now()},
                          validation=None, test=None, failure=None, file_hashes={})
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
    config = load_config(config_path(artifact), training=False)
    durations = [float(tag[:-1]) for tag in report.get('exports', {})] or [config['audio']['duration_sec']]
    if (Path(meta_root(report)) / 'model_registry.csv').exists():
        for duration in durations:
            _register_duration(artifact, report, config, duration)
    write_summary(meta_root(report))


def _register_duration(artifact, report, config, duration):
    meta = report['metadata']
    tag = duration_token(duration)
    variants = report.get('exports', {}).get(tag, {})
    measured = variants.get('fp32') or variants.get('int8') or {}
    arch = measured.get('architecture', report['architecture'])
    metrics = report.get('evaluation', {}).get(tag, {}).get('val', {})
    fp = metrics.get('keras_fp32', {})
    it = metrics.get('int8_tflite', {})
    exp = artifact / 'exports' / duration_token(duration)
    # Only legacy consumers require a YAML snapshot; normal experiments keep config in report.json.
    legacy_config = artifact / 'legacy_config.yaml'
    atomic_write(legacy_config, yaml.safe_dump(config, sort_keys=False))
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
           'status': report.get('state', {}).get('status', meta.get('status', 'trained')), 'config_path': str(legacy_config),
           'weight_path': str(artifact / 'best.keras'), 'tflite_path': str(exp / 'int8.tflite'),
           'data_source': meta['data_source'],
           'artifact_id': str(artifact) if duration == config['audio']['duration_sec'] else f'{artifact}::{tag}',
           'diagnostic_subset': fp.get('diagnostic_subset', meta.get('diagnostic_subset')),
           'validation_num_original_files': fp.get('num_original_files') if fp.get('split') == 'val' else None,
           'training_duration_secs': meta.get('training_duration_secs', [duration])}
    upsert_registry(meta['registry_root'], row)


def meta_root(report):
    return report.get('metadata', {}).get('registry_root', report.get('config', {}).get('runtime', {}).get('output_dir', 'models'))


def write_summary(output_root):
    """Rebuild one row per experiment from reports; reports remain the source of truth."""
    from training.artifacts import report_lock
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    with report_lock(root / 'summary.csv').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        for path in sorted(root.glob('*/report.json')):
            report = read_report(path.parent)
            state = report.get('state', {})
            if state.get('status') not in ('complete', 'failed', 'interrupted'):
                continue
            config = report.get('config', {})
            validation = report.get('validation') or {}
            exports = report.get('exports', {})
            if not exports and report.get('artifacts'):
                exports = {relative: read_report(path.parent / relative).get('exports', {})
                           for relative in report['artifacts']}
            def compact_exports(value):
                if isinstance(value, dict) and 'byte_size' in value:
                    return {k: value.get(k) for k in ('filename', 'byte_size', 'sha256')}
                return {k: compact_exports(v) for k, v in value.items() if isinstance(v, dict)}
            exports = compact_exports(exports)
            rows.append({'experiment_id': report['experiment']['id'], 'sequence': report['experiment']['sequence'],
                         'architecture': report['experiment']['architecture'], 'status': state['status'],
                         'seed': config['training']['seed'], 'conv_type': config['model']['conv_type'],
                         'pooling': config['model'].get('pooling', 'avg'),
                         'spatial_conv_type': config['model'].get('mobile_spatial_conv', 'DW') if report['experiment']['architecture'] == 'mobilenetv2' else config['model']['conv_type'],
                         'label': report['experiment'].get('label'),
                         'expansion_ratio': config['model'].get('expansion_ratio', 3) if report['experiment']['architecture'] == 'mobilenetv2' else None,
                         'width': config['model']['width'], 'n_mels': config['audio']['n_mels'],
                         'validation_method': (validation.get('protocol') or {}).get('method', config.get('validation', {}).get('method', 'fixed')),
                         'durations': json.dumps(config['training'].get('duration_secs') or [config['audio']['duration_sec']]),
                         'best_epoch': (report.get('checkpoint') or {}).get('best_epoch'),
                         'val_accuracy': validation.get('accuracy'), 'macro_f1': validation.get('macro_f1'),
                         'validation_by_duration': json.dumps(validation.get('per_duration')),
                         'num_parameters': (report.get('architecture') or {}).get('num_parameters'),
                         'report_path': str(path.resolve()), 'exports': json.dumps(exports, separators=(',', ':')),
                         'failure': (report.get('failure') or {}).get('message')})
        fields = list(rows[0]) if rows else ['experiment_id', 'status', 'val_accuracy']
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
        atomic_write(root / 'summary.csv', stream.getvalue())
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Rebuild summary.csv; optionally export the legacy CREST registry.')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--legacy', action='store_true')
    args = parser.parse_args()
    if args.legacy:
        # Explicit opt-in: retain the established per-width/per-duration CSV contract.
        root = Path(args.model_dir)
        if not (root / 'model_registry.csv').exists():
            atomic_write(root / 'model_registry.csv', ','.join(FIELDS) + '\n')
        for path in root.glob('*/report.json'):
            report = read_report(path.parent)
            if report.get('state', {}).get('status') == 'complete':
                for relative in report.get('artifacts', ['.']):
                    register_artifact(path.parent / relative)
    rows = write_summary(args.model_dir)
    print(f'{len(rows)} experiments -> {Path(args.model_dir) / "summary.csv"}')


if __name__ == '__main__':
    main()
