"""One run.json per artifact, with lossless support for older split JSON records."""
from __future__ import annotations
import argparse
import fcntl
from pathlib import Path

from training.utils import duration_token, load_config, read_json, save_json

LEGACY = {'metadata.json': 'metadata', 'architecture.json': 'architecture',
          'software_versions.json': 'software_versions', 'dataset_manifest.json': 'dataset_manifest',
          'first_training_batch.json': 'first_training_batch', 'status.json': 'state',
          'checkpoint.json': 'checkpoint'}


def config_path(artifact):
    artifact = Path(artifact).resolve()
    for directory in (artifact, *artifact.parents):
        if (directory / 'config.yaml').is_file():
            return directory / 'config.yaml'
    raise FileNotFoundError(f'No experiment config.yaml above {artifact}')


def legacy_files(artifact):
    root = Path(artifact)
    return ([root / name for name in (*LEGACY, 'metrics.json', 'artifacts.json') if (root / name).is_file()]
            + sorted((root / 'exports').glob('*/*.json'))
            + sorted((root / 'evaluation').glob('*/*.json')))


def read_report(artifact):
    root = Path(artifact)
    if (root / 'run.json').is_file():
        return read_json(root / 'run.json')
    report = {'schema_version': 2}
    for name, section in LEGACY.items():
        if (root / name).is_file():
            report[section] = read_json(root / name)
    if (root / 'artifacts.json').is_file():
        report['artifacts'] = read_json(root / 'artifacts.json')['artifacts']
    for path in sorted((root / 'exports').glob('*/*.json')):
        report.setdefault('exports', {}).setdefault(path.parent.name, {})[path.stem] = read_json(path)
    for path in sorted((root / 'evaluation').glob('*/*.json')):
        report.setdefault('evaluation', {}).setdefault(path.parent.name, {})[path.stem] = read_json(path)
    if (root / 'metrics.json').is_file():
        metrics = read_json(root / 'metrics.json')
        duration = report.get('metadata', {}).get('trained_duration_sec', 1)
        saved = report.setdefault('evaluation', {}).setdefault(duration_token(duration), {})
        if 'val' not in saved:
            saved['val'] = metrics
        elif saved['val'] != metrics:
            # Preserve conflicting historical summaries rather than discarding results.
            report['legacy_metrics_summary'] = metrics
    return report


def update_report(artifact, **sections):
    root = Path(artifact)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report = read_report(root)
        report.update(sections)
        save_json(root / 'run.json', report)
    return report


def store_result(artifact, section, duration, key, value):
    root = Path(artifact)
    with (root / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report = read_report(root)
        report.setdefault(section, {}).setdefault(duration_token(duration), {})[key] = value
        save_json(root / 'run.json', report)


def validation_metrics(artifact, duration=None):
    if duration is None:
        duration = load_config(config_path(artifact))['audio']['duration_sec']
    return read_report(artifact).get('evaluation', {}).get(duration_token(duration), {}).get('val', {})


def compact(artifact):
    """Merge records, reread and verify, then remove only the enumerated source JSONs."""
    root = Path(artifact)
    old = legacy_files(root)
    if not old:
        return 0
    report = read_report(root)
    if report.get('state', {}).get('status') == 'training':
        raise RuntimeError(f'Refusing to compact an active training run: {root}')
    snapshots = {p: p.read_bytes() for p in old}
    save_json(root / 'run.json', report)
    if read_json(root / 'run.json') != report:
        raise RuntimeError(f'Merged report verification failed: {root}')
    if any(p.read_bytes() != original for p, original in snapshots.items()):
        raise RuntimeError(f'Legacy files changed during consolidation: {root}')
    for path in old:
        path.unlink()
    return len(old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path('models'))
    args = parser.parse_args()
    roots = {p.parent for name in ('config.yaml', 'run.json') for p in args.model_dir.rglob(name)}
    for root in sorted(roots):
        if legacy_files(root):
            print(f'{root}: consolidated {compact(root)} JSON files into run.json')


if __name__ == '__main__':
    main()
