"""Average the predicted probabilities of several finished experiments.

The member list must be fixed in advance (not chosen by test accuracy).
Seeds are matched across experiments, so seed k of the ensemble averages
seed k of every member.

Example:
    python -m instrument_classification.ensemble \
        runs/instrument_classification/convnext_tiny \
        runs/instrument_classification/efficientnet_v2_s \
        --output runs/instrument_classification/ensemble
"""

import argparse
from pathlib import Path

import numpy as np

from common.metrics import aggregate_runs, classification_metrics, format_summary_table
from common.utils import save_json

from .data import CLASS_NAMES, NUM_CLASSES
from .train import scalar_metrics


def load_predictions(run_dir):
    """{seed_dir_name: npz} for an official-split experiment."""
    return {p.parent.name: np.load(p) for p in sorted(Path(run_dir).glob('seed*/predictions.npz'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    members = [load_predictions(run) for run in args.runs]
    seeds = sorted(set.intersection(*[set(m) for m in members]))
    if not seeds:
        raise SystemExit('No common seeds with predictions.npz found.')
    results = {'val': [], 'test': []}
    for seed in seeds:
        for split in ('val', 'test'):
            paths = members[0][seed][f'{split}_paths']
            for member in members[1:]:
                if not np.array_equal(member[seed][f'{split}_paths'], paths):
                    raise SystemExit(f'Image order differs between runs ({seed}, {split}).')
            probs = np.mean([m[seed][f'{split}_probabilities'] for m in members], axis=0)
            labels = members[0][seed][f'{split}_labels']
            results[split].append(scalar_metrics(classification_metrics(
                labels, probs.argmax(1), NUM_CLASSES, CLASS_NAMES)))
    summary = {'members': args.runs, 'seeds': seeds,
               'val': aggregate_runs(results['val']),
               'test': aggregate_runs(results['test'])}
    save_json(summary, Path(args.output) / 'summary.json')
    print(format_summary_table([('val', summary['val']), ('test', summary['test'])],
                               ['accuracy', 'macro_f1', 'balanced_accuracy']))


if __name__ == '__main__':
    main()
