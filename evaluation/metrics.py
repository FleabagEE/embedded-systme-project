"""Metrics after averaging crop probabilities by original audio ID."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


AGGREGATION = 'Average softmax probabilities across deterministic crops of each original file, then take argmax; one center crop when num_crops=1.'


def softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = np.exp(z - z.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def classification_metrics(labels, logits, file_ids, class_mapping, split, synthetic=False):
    if len(labels) != len(logits) or len(labels) != len(file_ids) or not len(labels):
        raise ValueError('Nonempty labels, logits and original file IDs must have equal length')
    groups = {}
    for label, probability, file_id in zip(labels, softmax(logits), file_ids):
        if file_id not in groups:
            groups[file_id] = [int(label), []]
        if groups[file_id][0] != int(label):
            raise ValueError(f'Conflicting labels for original file {file_id}')
        groups[file_id][1].append(probability)
    truth = [g[0] for g in groups.values()]
    pred = [int(np.argmax(np.mean(g[1], axis=0))) for g in groups.values()]
    all_labels = list(range(len(class_mapping)))
    return {'accuracy': float(accuracy_score(truth, pred)),
            'macro_f1': float(f1_score(truth, pred, labels=all_labels, average='macro', zero_division=0)),
            'confusion_matrix': confusion_matrix(truth, pred, labels=all_labels).tolist(),
            'split': split, 'num_original_files': len(groups), 'num_crops': len(labels),
            'aggregation': AGGREGATION, 'class_mapping': class_mapping,
            'data_source': 'synthetic_smoke_only' if synthetic else 'UrbanSound8K',
            'empirical_dataset_result': not synthetic}
