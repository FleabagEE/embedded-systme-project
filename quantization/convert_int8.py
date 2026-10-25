"""Training-fold-only PTQ, fixed-duration exports and exact conversion diagnostics."""
from __future__ import annotations
import argparse
import copy
import hashlib
from pathlib import Path
import numpy as np

from training.utils import atomic_write, configure_runtime, duration_token, load_config, export_durations
from training.artifacts import config_path, read_report, store_result


def convert_model(model, representative=None):
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if representative is not None:
        samples = list(representative)
        if not samples:
            raise ValueError('INT8 calibration requires nonempty representative training features')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: ([sample.astype(np.float32)] for sample in samples)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return converter.convert()


def export_artifact(artifact, config, train_dataset, *, duration=None, skip_int8=False):
    import tensorflow as tf
    from training.dscnn import fix_input_shape, architecture_metadata
    from quantization.validate_int8 import TFLiteRunner
    artifact = Path(artifact)
    duration = config['audio']['duration_sec'] if duration is None else duration
    if train_dataset.split != 'train':
        raise ValueError('PTQ requires a training-fold dataset')
    model = tf.keras.models.load_model(artifact / 'best.keras', compile=False)
    fixed = fix_input_shape(model, config, duration)
    target = artifact / 'exports' / duration_token(duration)
    target.mkdir(parents=True, exist_ok=True)
    run_meta = read_report(artifact)['metadata']
    cfg = copy.deepcopy(config)
    cfg['audio']['duration_sec'] = duration
    # Retain the caller's diagnostic subset; use the requested-duration frontend.
    from training.dataset import AudioDataset, SyntheticDataset
    synthetic = run_meta['data_source'] == 'synthetic_smoke_only'
    if synthetic:
        source = SyntheticDataset(cfg, 'train', train_dataset.count)
    else:
        source = AudioDataset(train_dataset.records, cfg, 'train')
    calibration = [] if skip_int8 else list(source.representative(cfg['quantization']['representative_samples']))
    provenance = {'run_id': run_meta['run_id'], 'trained_parent': run_meta.get('trained_parent'),
                  'effective_width': run_meta['effective_width'], 'trained_duration_sec': run_meta['trained_duration_sec'],
                  'training_duration_secs': run_meta.get('training_duration_secs', [run_meta['trained_duration_sec']]),
                  'export_duration_sec': duration, 'identical_frozen_weights': True,
                  'data_source': run_meta['data_source'], 'calibration_split': 'train',
                  'calibration_folds': config['dataset']['train_folds'] if not synthetic else [],
                  'calibration_count': len(calibration),
                  'calibration_file_ids': [r.file_id for r in source.records[:len(calibration)]] if not synthetic else [],
                  'calibration_augmentation': False, 'frontend': cfg['audio'],
                  'frozen_weight_sha256': hashlib.sha256(b''.join(w.tobytes() for w in fixed.get_weights())).hexdigest()}
    for kind in ('fp32', 'int8'):
        if kind == 'int8' and skip_int8:
            continue
        try:
            content = convert_model(fixed, calibration if kind == 'int8' else None)
            runner = TFLiteRunner(model_content=content, require_int8=kind == 'int8')
            details = runner.metadata()
            sample = calibration[0] if calibration else np.zeros(runner.input['shape'], np.float32)
            output = runner(sample)
            if output.shape != (1, config['model']['num_classes']) or not np.all(np.isfinite(output)):
                raise ValueError(f'Invalid interpreter output: {output.shape}')
            details.update(provenance)
            details.update({'cpu_invocation_success': True, 'byte_size': len(content),
                            'architecture': architecture_metadata(fixed, config, run_meta['effective_width'], duration)})
            atomic_write(target / f'{kind}.tflite', content)
            store_result(artifact, 'exports', duration, kind, details)
        except Exception as exc:
            store_result(artifact, 'exports', duration, f'{kind}_conversion_error', {'type': type(exc).__name__, 'message': str(exc), **provenance})
            raise RuntimeError(f'{kind} conversion failed for {artifact} at {duration}s: {exc}') from exc
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--duration', type=int, choices=[1, 2])
    args = parser.parse_args()
    from evaluation.evaluate import artifact_directories
    from training.dataset import AudioDataset, SyntheticDataset, load_metadata
    from training.registry import register_artifact
    for artifact in artifact_directories(args.model_dir):
        config = load_config(config_path(artifact))
        configure_runtime(config)
        meta = read_report(artifact)['metadata']
        if meta['data_source'] == 'synthetic_smoke_only':
            dataset = SyntheticDataset(config, 'train', meta['synthetic_count'])
        else:
            records, classes = load_metadata(config['dataset']['root'])
            if {str(k): v for k, v in classes.items()} != meta['class_mapping']:
                raise ValueError('Class mapping differs from training')
            dataset = AudioDataset(records, config, 'train', limit=meta.get('dataset_limit'))
        for duration in [args.duration] if args.duration else export_durations(config):
            print(export_artifact(artifact, config, dataset, duration=duration))
        register_artifact(artifact)


if __name__ == '__main__':
    main()
