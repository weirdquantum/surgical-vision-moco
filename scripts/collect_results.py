"""Collect every experiment summary under runs/ into one Markdown report.

Besides per-experiment tables it computes paired comparisons: runs that share
seeds (or cross-validation folds) are compared seed by seed, which removes the
seed-to-seed variance shared by both arms.

    python scripts/collect_results.py --runs runs --output runs/ALL_RESULTS.md
"""

import argparse
import json
import math
from pathlib import Path

from scipy import stats


def load(runs, rel):
    path = runs / rel / 'summary.json'
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as file:
        return json.load(file)


def test_node(summary, stage=None):
    return summary[stage]['test'] if stage else summary['test']


def fmt(entry, scale=100.0, digits=1):
    if entry is None:
        return '-'
    mean, std = entry['mean'] * scale, entry['std'] * scale
    return f'{mean:.{digits}f} ± {std:.{digits}f}' if len(entry['values']) > 1 else f'{mean:.{digits}f}'


def paired(a, b):
    n = min(len(a), len(b))
    diffs = [b[i] - a[i] for i in range(n)]
    mean = sum(diffs) / n
    std = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0
    p = None
    if n > 1 and std > 0:
        p = float(2 * stats.t.sf(abs(mean / (std / math.sqrt(n))), n - 1))
    wins = sum(d > 0 for d in diffs)
    return n, mean, std, p, wins


CLASSIFICATION_COLUMNS = ('accuracy', 'macro_f1', 'balanced_accuracy')
SEGMENT_COLUMNS = ('accuracy', 'macro_f1', 'macro_f1_all', 'edit', 'f1@10', 'f1@50')

PAIRS = [
    # (title, (run A, stage A), (run B, stage B))
    ('P1 C0→C1 architecture (CNN → ResNet-18)', ('instrument_classification/legacy_c0', None), ('instrument_classification/legacy_c1', None)),
    ('P1 C1→C2 optimiser (SGD → Adam)', ('instrument_classification/legacy_c1', None), ('instrument_classification/legacy_c2', None)),
    ('P1 C2→C3 batch size (8 → 16)', ('instrument_classification/legacy_c2', None), ('instrument_classification/legacy_c3', None)),
    ('P1 C4→C5 initialisation (random → ImageNet)', ('instrument_classification/legacy_c4', None), ('instrument_classification/legacy_c5', None)),
    ('P1 original protocol → new protocol (ResNet-18)', ('instrument_classification/legacy_c5', None), ('instrument_classification/resnet18_imagenet', None)),
    ('P1 ResNet-18 → ConvNeXt-Tiny', ('instrument_classification/resnet18_imagenet', None), ('instrument_classification/convnext_tiny', None)),
    ('P1 ConvNeXt-Tiny → ensemble', ('instrument_classification/convnext_tiny', None), ('instrument_classification/ensemble', None)),
    ('P1 ConvNet 150 → 300 epochs', ('instrument_classification/convnet_scratch', None), ('instrument_classification/convnet_scratch_300ep', None)),
    ('P2 legacy ResNet-18 → ConvNeXt (frame)', ('action_recognition/legacy_resnet18', 'frame'), ('action_recognition/convnext_tiny_mstcn', 'frame')),
    ('P2 frame → smoothed', ('action_recognition/convnext_tiny_mstcn', 'frame'), ('action_recognition/convnext_tiny_mstcn', 'smoothed')),
    ('P2 frame → MS-TCN', ('action_recognition/convnext_tiny_mstcn', 'frame'), ('action_recognition/convnext_tiny_mstcn', 'temporal')),
    ('P2 MS-TCN without → with cross-fitting', ('action_recognition/convnext_tiny_mstcn_no_crossfit', 'temporal'), ('action_recognition/convnext_tiny_mstcn', 'temporal')),
    ('P2 surgery CV: frame → MS-TCN', ('action_recognition/convnext_tiny_mstcn_surgery_cv', 'frame'), ('action_recognition/convnext_tiny_mstcn_surgery_cv', 'temporal')),
    ('P3 ConvNet scratch → MoCo (150 ep)', ('contrastive_learning/finetune_convnet_scratch', None), ('contrastive_learning/finetune_convnet_moco', None)),
    ('P3 ConvNet scratch → legacy MoCo (150 ep)', ('contrastive_learning/finetune_convnet_scratch', None), ('contrastive_learning/finetune_convnet_legacy_moco', None)),
    ('P3 ConvNet scratch → MoCo (300 ep)', ('instrument_classification/convnet_scratch_300ep', None), ('contrastive_learning/finetune_convnet_moco_300ep', None)),
    ('P3 ResNet-18 ImageNet → ImageNet+MoCo', ('contrastive_learning/finetune_resnet18_imagenet', None), ('contrastive_learning/finetune_resnet18_imagenet_moco', None)),
]


def experiment_tables(runs, lines):
    classification, segmentation = [], []
    for path in sorted(runs.rglob('summary.json')):
        rel = path.parent.relative_to(runs).as_posix()
        summary = json.loads(path.read_text(encoding='utf-8'))
        if 'frame' in summary:
            for stage in ('frame', 'smoothed', 'temporal'):
                if stage in summary:
                    node = test_node(summary, stage)
                    segmentation.append((f'{rel} [{stage}]', node))
        elif 'test' in summary:
            classification.append((rel, summary))
    lines += ['## Classification experiments (test set)', '',
              '| run | n | accuracy % | macro-F1 % | balanced acc % | pooled CV acc % |',
              '|---|---|---|---|---|---|']
    for rel, summary in classification:
        node = summary['test']
        n = len(node['accuracy']['values'])
        pooled = summary.get('test_pooled_over_folds', {}).get('accuracy')
        lines.append(f'| {rel} | {n} | ' + ' | '.join(fmt(node.get(k)) for k in CLASSIFICATION_COLUMNS)
                     + f' | {fmt(pooled)} |')
    lines += ['', '## Action-recognition experiments (test set)', '',
              '| run [stage] | n | ' + ' | '.join(SEGMENT_COLUMNS) + ' |',
              '|---|---|' + '---|' * len(SEGMENT_COLUMNS)]
    for name, node in segmentation:
        n = len(node['accuracy']['values'])
        cells = [fmt(node.get(k), 100.0 if k in ('accuracy', 'macro_f1', 'macro_f1_all') else 1.0)
                 for k in SEGMENT_COLUMNS]
        lines.append(f'| {name} | {n} | ' + ' | '.join(cells) + ' |')


def pretraining_tables(runs, lines):
    rows = []
    for path in sorted((runs / 'contrastive_learning').glob('*/seed*/summary.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        if 'best_knn_val' in data:
            rows.append(f'| {path.parent.parent.name} | {data["best_knn_val"] * 100:.1f} | '
                        f'{data["final_loss"]:.3f} |')
    if rows:
        lines += ['', '## MoCo pretraining', '', '| run | best kNN val % | final loss |',
                  '|---|---|---|', *rows]
    probe = runs / 'contrastive_learning' / 'probe' / 'probe.json'
    if probe.exists():
        data = json.loads(probe.read_text(encoding='utf-8'))
        lines += ['', '## Frozen-feature probe (accuracy %)', '',
                  '| encoder | kNN val | kNN test | linear val | linear test |', '|---|---|---|---|---|']
        for name, result in data.items():
            short = name.split('contrastive_learning/')[-1]
            lines.append(f'| {short} | ' + ' | '.join(
                f'{result[k]["accuracy"] * 100:.1f}' for k in ('knn_val', 'knn_test', 'linear_val', 'linear_test')) + ' |')


def pair_table(runs, lines):
    lines += ['', '## Paired comparisons (same seeds / folds; B − A, percentage points)', '',
              '| comparison | n | Δ accuracy | Δ macro-F1 | B better in | p (paired t) |',
              '|---|---|---|---|---|---|']
    for title, (rel_a, stage_a), (rel_b, stage_b) in PAIRS:
        a, b = load(runs, rel_a), load(runs, rel_b)
        if a is None or b is None:
            continue
        node_a, node_b = test_node(a, stage_a), test_node(b, stage_b)
        n, mean, std, p, wins = paired(node_a['accuracy']['values'], node_b['accuracy']['values'])
        _, mean_f1, std_f1, _, _ = paired(node_a['macro_f1']['values'], node_b['macro_f1']['values'])
        p_text = f'{p:.3f}' if p is not None else '-'
        lines.append(f'| {title} | {n} | {mean * 100:+.1f} ± {std * 100:.1f} | '
                     f'{mean_f1 * 100:+.1f} ± {std_f1 * 100:.1f} | {wins}/{n} | {p_text} |')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', default='runs')
    parser.add_argument('--output', default='runs/ALL_RESULTS.md')
    args = parser.parse_args()
    runs = Path(args.runs)
    lines = ['# All results', '', 'Mean ± std over seeds (or folds). Generated by scripts/collect_results.py.', '']
    experiment_tables(runs, lines)
    pretraining_tables(runs, lines)
    pair_table(runs, lines)
    text = '\n'.join(lines) + '\n'
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
