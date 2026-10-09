"""Classification metrics and multi-run aggregation."""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def classification_metrics(labels, predictions, num_classes, class_names=None):
    """Frame/image-level metrics.

    ``macro_f1`` averages only classes present in ``labels`` (a class with no
    ground-truth samples has undefined recall); ``macro_f1_all`` averages all
    ``num_classes`` classes with absent classes scored as 0, as in the original
    notebook.
    """
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    class_ids = np.arange(num_classes)
    names = class_names or [str(i) for i in class_ids]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=class_ids, zero_division=0)
    present = support > 0
    return {
        'accuracy': float(np.mean(labels == predictions)) if len(labels) else 0.0,
        'balanced_accuracy': float(recall[present].mean()) if present.any() else 0.0,
        'macro_f1': float(f1[present].mean()) if present.any() else 0.0,
        'macro_f1_all': float(f1.mean()),
        'weighted_f1': float(np.average(f1, weights=support)) if support.sum() else 0.0,
        'per_class': {
            names[i]: {'precision': float(precision[i]), 'recall': float(recall[i]),
                       'f1': float(f1[i]), 'support': int(support[i])}
            for i in class_ids
        },
        'confusion_matrix': confusion_matrix(labels, predictions,
                                             labels=class_ids).tolist(),
    }


SCALAR_KEYS = ('loss', 'accuracy', 'balanced_accuracy', 'macro_f1',
               'macro_f1_all', 'weighted_f1')


def aggregate_runs(results):
    """Mean / std / values of scalar metrics across seeds or folds."""
    if not results:
        return {}
    summary = {}
    for key in results[0]:
        values = [float(r[key]) for r in results if key in r]
        if values:
            summary[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                'values': values,
            }
    return summary


def format_summary_table(rows, keys):
    """Markdown table: rows = [(name, aggregate_dict)]."""
    header = '| run | ' + ' | '.join(keys) + ' |'
    lines = [header, '|' + '---|' * (len(keys) + 1)]
    for name, agg in rows:
        cells = []
        for key in keys:
            if key in agg:
                mean, std = agg[key]['mean'], agg[key]['std']
                n = len(agg[key]['values'])
                cells.append(f'{mean:.4f} ± {std:.4f}' if n > 1 else f'{mean:.4f}')
            else:
                cells.append('-')
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)
