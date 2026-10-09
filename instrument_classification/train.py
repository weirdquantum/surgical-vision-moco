"""Train / select / test an instrument classifier for every seed (and fold).

Example:
    python -m instrument_classification.train \
        --config instrument_classification/configs/convnext_tiny.yaml
"""

import math
from pathlib import Path

import numpy as np
import torch

from common.data import build_transform, make_loader
from common.engine import evaluate, fit
from common.metrics import SCALAR_KEYS, aggregate_runs, classification_metrics, \
    format_summary_table
from common.networks import build_classifier, count_parameters
from common.plotting import plot_confusion, plot_history
from common.utils import (Logger, base_arg_parser, describe_environment, get_device,
                          load_json, resolve_config, save_json, save_yaml, set_seed)

from .data import CLASS_NAMES, NUM_CLASSES, InstrumentDataset, build_splits, \
    subsample_stratified


def scalar_metrics(metrics):
    return {k: metrics[k] for k in SCALAR_KEYS if k in metrics}


def scale_schedule(train_cfg, fraction):
    """Keep the number of optimisation steps (and validations) roughly constant
    when training on a fraction of the labelled images."""
    if fraction >= 1.0 or not train_cfg.get('scale_epochs_with_fraction', True):
        return train_cfg
    return {**train_cfg,
            'epochs': math.ceil(train_cfg['epochs'] / fraction),
            'warmup_epochs': train_cfg.get('warmup_epochs', 0) / fraction,
            'eval_every': train_cfg.get('eval_every', 1) * math.ceil(1 / fraction)}


def run_single(cfg, seed, fold, out_dir, device, log):
    out_dir = Path(out_dir)
    if cfg.get('resume', True) and (out_dir / 'metrics.json').exists():
        log(f'skip {out_dir} (metrics.json exists)')
        return load_json(out_dir / 'metrics.json')
    set_seed(seed, cfg.get('deterministic', False))

    data_cfg, train_cfg = cfg['data'], cfg['train']
    splits = build_splits(data_cfg, seed=seed, fold=fold)
    fraction = data_cfg.get('train_fraction', 1.0)
    if fraction < 1.0:
        splits['train'] = subsample_stratified(splits['train'], fraction, seed)
        train_cfg = scale_schedule(train_cfg, fraction)
        log(f'label fraction {fraction}: {len(splits["train"])} training images, '
            f'{train_cfg["epochs"]} epochs, validation every {train_cfg["eval_every"]}')
    preload = data_cfg.get('preload_size')
    train_set = InstrumentDataset(splits['train'], build_transform(cfg['augment'], True), preload)
    eval_tf = build_transform(cfg['augment'], False)
    val_set = InstrumentDataset(splits['val'], eval_tf, preload)
    test_set = InstrumentDataset(splits['test'], eval_tf, preload)
    log(f'seed {seed} fold {fold}: train {len(train_set)} / val {len(val_set)} / '
        f'test {len(test_set)} images')

    workers = data_cfg.get('num_workers', 2)
    train_loader = make_loader(train_set, train_cfg['batch_size'], True, workers, seed,
                               drop_last=(train_cfg.get('drop_last', False)
                                          and len(train_set) >= train_cfg['batch_size']),
                               device=device)
    val_loader = make_loader(val_set, cfg['eval'].get('batch_size', 32), False, workers,
                             seed, device=device)
    test_loader = make_loader(test_set, cfg['eval'].get('batch_size', 32), False, workers,
                              seed, device=device)

    model = build_classifier(cfg['model'], NUM_CLASSES).to(device)
    log(f'model {cfg["model"]["backbone"]} | trainable parameters {count_parameters(model):,}')
    history, best = fit(model, train_loader, val_loader, train_cfg, device, out_dir,
                        NUM_CLASSES, train_set.labels, CLASS_NAMES, log)

    # Final evaluation: the selected checkpoint only, test set touched once.
    model.load_state_dict(torch.load(out_dir / 'best.pt', map_location=device,
                                     weights_only=True))
    tta = cfg['eval'].get('tta_hflip', False)
    val_metrics, val_out = evaluate(model, val_loader, device, NUM_CLASSES, CLASS_NAMES,
                                    tta_hflip=tta)
    test_metrics, test_out = evaluate(model, test_loader, device, NUM_CLASSES, CLASS_NAMES,
                                      tta_hflip=tta)
    np.savez_compressed(out_dir / 'predictions.npz',
                        **{f'val_{k}': v for k, v in val_out.items() if k != 'loss'},
                        **{f'test_{k}': v for k, v in test_out.items() if k != 'loss'})
    plot_history(history, out_dir / 'learning_curves.png', best['epoch'],
                 f'{cfg["name"]} seed {seed} fold {fold}')
    plot_confusion(test_metrics['confusion_matrix'], CLASS_NAMES,
                   out_dir / 'test_confusion.png', 'test')
    result = {'seed': seed, 'fold': fold, 'best': best,
              'val': val_metrics, 'test': test_metrics}
    save_json(result, out_dir / 'metrics.json')
    log(f'selected epoch {best["epoch"]} | val acc {val_metrics["accuracy"]:.4f} | '
        f'TEST acc {test_metrics["accuracy"]:.4f} macro-F1 {test_metrics["macro_f1"]:.4f}')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def folds_for(cfg):
    if cfg['data'].get('split', 'official') == 'video_cv':
        return list(range(cfg['data'].get('n_folds', 5)))
    return [0]


def run_experiment(cfg):
    out_root = Path(cfg['output_dir'])
    out_root.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg.get('device', 'auto'))
    log = Logger(out_root / 'train.log')
    save_yaml(cfg, out_root / 'config.yaml')
    log(f'{cfg["name"]} | {describe_environment(device)}')

    results = []
    for seed in cfg['seeds']:
        for fold in folds_for(cfg):
            run_dir = out_root / f'seed{seed}' / (f'fold{fold}' if len(folds_for(cfg)) > 1 else '')
            results.append(run_single(cfg, seed, fold, run_dir, device, log))

    summary = {
        'name': cfg['name'],
        'split': cfg['data'].get('split', 'official'),
        'n_runs': len(results),
        'val': aggregate_runs([scalar_metrics(r['val']) for r in results]),
        'test': aggregate_runs([scalar_metrics(r['test']) for r in results]),
        'best_epochs': [r['best']['epoch'] for r in results],
    }
    if len(folds_for(cfg)) > 1:
        # Cross-validation: pool the test predictions of all folds per seed.
        pooled = []
        for seed in cfg['seeds']:
            labels, preds = [], []
            for fold in folds_for(cfg):
                data = np.load(out_root / f'seed{seed}' / f'fold{fold}' / 'predictions.npz')
                labels.append(data['test_labels'])
                preds.append(data['test_predictions'])
            pooled.append(scalar_metrics(classification_metrics(
                np.concatenate(labels), np.concatenate(preds), NUM_CLASSES, CLASS_NAMES)))
        summary['test_pooled_over_folds'] = aggregate_runs(pooled)
    save_json(summary, out_root / 'summary.json')
    table = format_summary_table(
        [('val', summary['val']), ('test', summary['test'])],
        ['accuracy', 'macro_f1', 'balanced_accuracy', 'loss'])
    (out_root / 'summary.md').write_text(table + '\n', encoding='utf-8')
    log('\n' + table)
    return summary


def main():
    args = base_arg_parser(__doc__).parse_args()
    run_experiment(resolve_config(args, 'instrument_classification'))


if __name__ == '__main__':
    main()
