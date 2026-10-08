"""Temporal-segmentation metrics (Lea et al. 2017; Farha & Gall, MS-TCN 2019)
and probability smoothing."""

import numpy as np


def get_segments(labels):
    """Run-length encode a label sequence -> list of (label, start, end_exclusive)."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(labels)]])
    return [(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def edit_score(prediction, target):
    """Normalised segmental Levenshtein similarity in [0, 100]."""
    p = [s[0] for s in get_segments(prediction)]
    y = [s[0] for s in get_segments(target)]
    m, n = len(p), len(y)
    table = np.zeros((m + 1, n + 1))
    table[:, 0] = np.arange(m + 1)
    table[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if p[i - 1] == y[j - 1] else 1
            table[i, j] = min(table[i - 1, j] + 1, table[i, j - 1] + 1,
                              table[i - 1, j - 1] + cost)
    return (1 - table[m, n] / max(m, n, 1)) * 100


def segment_f1_counts(prediction, target, overlap):
    """True/false positives and false negatives of segments at IoU >= overlap."""
    pred_segments = get_segments(prediction)
    true_segments = get_segments(target)
    used = np.zeros(len(true_segments), dtype=bool)
    tp = fp = 0
    for label, start, end in pred_segments:
        best_iou, best_index = 0.0, -1
        for index, (t_label, t_start, t_end) in enumerate(true_segments):
            if t_label != label:
                continue
            intersection = max(0, min(end, t_end) - max(start, t_start))
            union = max(end, t_end) - min(start, t_start)
            iou = intersection / union
            if iou > best_iou:
                best_iou, best_index = iou, index
        if best_iou >= overlap and not used[best_index]:
            tp += 1
            used[best_index] = True
        else:
            fp += 1
    fn = int((~used).sum())
    return tp, fp, fn


def segmental_metrics(predictions, targets, overlaps=(0.1, 0.25, 0.5)):
    """Edit score (mean over sequences) and F1@k (counts pooled over sequences)."""
    edits = [edit_score(p, t) for p, t in zip(predictions, targets)]
    result = {'edit': float(np.mean(edits)) if edits else 0.0}
    for overlap in overlaps:
        tp = fp = fn = 0
        for p, t in zip(predictions, targets):
            a, b, c = segment_f1_counts(p, t, overlap)
            tp, fp, fn = tp + a, fp + b, fn + c
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        result[f'f1@{int(overlap * 100)}'] = f1 * 100
    return result


def smooth_probabilities(probabilities, window):
    """Centred moving average over time (window odd, edges use partial windows)."""
    if window <= 1:
        return probabilities
    half = window // 2
    padded = np.pad(probabilities, ((half, half), (0, 0)), mode='edge')
    kernel = np.ones(window) / window
    return np.stack([np.convolve(padded[:, c], kernel, mode='valid')
                     for c in range(probabilities.shape[1])], axis=1)
