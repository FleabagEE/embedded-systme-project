"""Evaluate Keras, FP32 TFLite and INT8 TFLite by original recording."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import numpy as np

from evaluation.metrics import classification_metrics
from training.utils import configure_runtime, duration_token, load_config
from training.artifacts import config_path, read_report, store_result


def evaluate_predictor(predict, dataset, classes, synthetic=False):
    logits, labels, ids = [], [], []
    for x, y, file_ids in dataset.batches(training=False):
        logits.extend(np.asarray(predict(x)))
        labels.extend(y.tolist())
        ids.extend(file_ids)
    return classification_metrics(labels, np.asarray(logits), ids, classes, dataset.split, synthetic)


def artifact_directories(path):
    path = Path(path)
    if (path / 'best.keras').is_file():
        return [path]
    report = read_report(path)
    if report.get('artifacts'):
        return [path / item for item in report['artifacts']]
    raise FileNotFoundError(f'No best.keras or artifact list in {path}')


def evaluate_artifact(artifact, dataset, config, classes, *, duration=None, synthetic=False, diagnostic_subset=False):
    import tensorflow as tf
    from quantization.validate_int8 import TFLiteRunner
    artifact = Path(artifact)
    duration = config['audio']['duration_sec'] if duration is None else duration
    model = tf.keras.models.load_model(artifact / 'best.keras', compile=False)
    results = {'keras_fp32': evaluate_predictor(lambda x: model(x, training=False).numpy(), dataset, classes, synthetic)}
    for fmt, filename in [('fp32_tflite', 'fp32.tflite'), ('int8_tflite', 'int8.tflite')]:
        path = artifact / 'exports' / duration_token(duration) / filename
        if path.is_file():
            results[fmt] = evaluate_predictor(TFLiteRunner(path, require_int8=fmt == 'int8_tflite'), dataset, classes, synthetic)
    meta = read_report(artifact)['metadata']
    if meta.get('trained_parent'):
        from training.dscnn import SharedCNN
        # Prefer a colocated checkpoint if the entire run has been moved.
        local_parent = artifact.parent.parent
        parent = local_parent if (local_parent / 'shared_best.weights.h5').is_file() else Path(meta['parent_run_path'])
        checkpoint = parent / 'shared_best.weights.h5'
        if checkpoint.is_file():
            supernet = SharedCNN(config)
            from training.features import feature_shape
            supernet(np.zeros([1, *feature_shape(config['audio'])], np.float32))
            supernet.load_weights(checkpoint)
            results['shared_supernet_fp32'] = evaluate_predictor(
                lambda x: supernet(x, width=meta['effective_width'], training=False).numpy(), dataset, classes, synthetic)
    for record in results.values():
        record.update({'trained_parent': meta.get('trained_parent'), 'effective_width': meta['effective_width'],
                       'run_id': meta['run_id'], 'trained_duration_sec': meta['trained_duration_sec'],
                       'evaluation_duration_sec': duration, 'folds': config['dataset'][f'{dataset.split}_folds'],
                       'training_duration_secs': meta.get('training_duration_secs', [meta['trained_duration_sec']]),
                       'duration_seen_during_training': duration in meta.get('training_duration_secs', [meta['trained_duration_sec']]),
                       'evaluation_num_crops': config.get('evaluation', {}).get('num_crops', 1),
                       'diagnostic_subset': meta.get('diagnostic_subset', False) or diagnostic_subset})
    if meta.get('trained_parent'):
        results['frozen_subnet_fp32'] = copy.deepcopy(results['keras_fp32'])
    saved_crops = load_config(config_path(artifact)).get('evaluation', {}).get('num_crops', 1)
    crops = config.get('evaluation', {}).get('num_crops', 1)
    key = dataset.split if crops == saved_crops else f'{dataset.split}_{crops}crops'
    if diagnostic_subset:
        key += f"_subset_{results['keras_fp32']['num_original_files']}"
    store_result(artifact, 'evaluation', duration, key, results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--duration', type=int, choices=[1, 2])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--num-crops', type=int, help='Deterministic evenly spaced crops; probabilities averaged per original file')
    args = parser.parse_args()
    from training.dataset import AudioDataset, SyntheticDataset, load_metadata
    from training.registry import register_artifact
    for artifact in artifact_directories(args.model_dir):
        config = load_config(config_path(artifact))
        if args.duration:
            config['audio']['duration_sec'] = args.duration
        if args.num_crops is not None:
            if args.num_crops < 1:
                parser.error('--num-crops must be positive')
            config.setdefault('evaluation', {})['num_crops'] = args.num_crops
        configure_runtime(config)
        meta = read_report(artifact)['metadata']
        synthetic = meta['data_source'] == 'synthetic_smoke_only'
        if synthetic:
            dataset = SyntheticDataset(config, args.split, args.limit or meta['synthetic_count'])
        else:
            records, classes = load_metadata(config['dataset']['root'])
            if {str(k): v for k, v in classes.items()} != meta['class_mapping']:
                raise ValueError('Dataset class mapping differs from trained mapping')
            dataset = AudioDataset(records, config, args.split, limit=args.limit)
        result = evaluate_artifact(artifact, dataset, config, meta['class_mapping'], duration=args.duration,
                                   synthetic=synthetic, diagnostic_subset=args.limit is not None)
        register_artifact(artifact)
        print(artifact, {k: {'accuracy': v['accuracy'], 'macro_f1': v['macro_f1'], 'data_source': v['data_source']} for k, v in result.items()})


if __name__ == '__main__':
    main()
