"""Surgical action recognition: frame model -> smoothing -> MS-TCN.

Per seed (and fold):
  1. fine-tune a frame classifier, select the epoch with best val macro-F1;
  2. export per-frame logits for every sequence;
  3. tune a moving-average window on val and apply it to test;
  4. (optional) cross-fit frame models on train surgeries so the temporal
     model is trained on realistic, out-of-sample train predictions;
  5. (optional) train an MS-TCN on the sequences, select on val, test once.
Finished stages are skipped on re-run, so an interrupted run can resume.

    python -m action_recognition.train --config action_recognition/configs/convnext_tiny_mstcn.yaml
"""

import random
import shutil
from pathlib import Path

import numpy as np
import torch

from common.data import build_transform, make_loader
from common.engine import class_weights, fit, predict
from common.metrics import SCALAR_KEYS, aggregate_runs, classification_metrics, \
    format_summary_table
from common.networks import build_classifier, count_parameters
from common.plotting import plot_confusion, plot_history
from common.utils import (Logger, base_arg_parser, describe_environment, get_device,
                          load_json, resolve_config, save_json, save_yaml, set_seed)

from .data import NUM_ACTIONS, SHORT_NAMES, FrameDataset, build_sequences, class_counts
from .metrics import segmental_metrics, smooth_probabilities
from .temporal import predict_sequences, prepare_inputs, train_temporal

REPORT_KEYS = list(SCALAR_KEYS) + ['edit', 'f1@10', 'f1@25', 'f1@50']


def sequence_metrics(probabilities, sequences):
    labels = [s.labels for s in sequences]
    predictions = [p.argmax(1) for p in probabilities]
    metrics = classification_metrics(np.concatenate(labels), np.concatenate(predictions),
                                      NUM_ACTIONS, SHORT_NAMES)
    metrics.update(segmental_metrics(predictions, labels))
    return metrics


def scalars(metrics):
    return {k: metrics[k] for k in REPORT_KEYS if k in metrics}


# ----------------------------------------------------------------------------
# Stage 1: frame model
# ----------------------------------------------------------------------------

def train_frame_model(cfg, train_seqs, val_seqs, out_dir, device, seed, log):
    set_seed(seed)
    data_cfg, train_cfg = cfg['data'], cfg['train']
    train_set = FrameDataset(train_seqs, build_transform(cfg['augment'], True))
    val_set = FrameDataset(val_seqs, build_transform(cfg['augment'], False))
    workers = data_cfg.get('num_workers', 4)
    train_loader = make_loader(train_set, train_cfg['batch_size'], True, workers, seed,
                               drop_last=train_cfg.get('drop_last', True), device=device)
    val_loader = make_loader(val_set, cfg['eval'].get('batch_size', 64), False, workers,
                             seed, device=device)
    model = build_classifier(cfg['model'], NUM_ACTIONS).to(device)
    log(f'frame model {cfg["model"]["backbone"]}: {count_parameters(model):,} params | '
        f'{len(train_set)} train / {len(val_set)} val frames')
    history, best = fit(model, train_loader, val_loader, train_cfg, device, out_dir,
                        NUM_ACTIONS, train_set.labels, SHORT_NAMES, log)
    plot_history(history, out_dir / 'learning_curves.png', best['epoch'])
    model.load_state_dict(torch.load(out_dir / 'best.pt', map_location=device,
                                     weights_only=True))
    return model


def export_outputs(model, sequences, cfg, device, out_dir, seed):
    """Save per-sequence logits to out_dir/<name>.npz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = FrameDataset(sequences, build_transform(cfg['augment'], False))
    loader = make_loader(dataset, cfg['eval'].get('batch_size', 64), False,
                         cfg['data'].get('num_workers', 4), seed, device=device)
    outputs = predict(model, loader, device)
    offset = 0
    for sequence in sequences:
        end = offset + len(sequence)
        np.savez_compressed(out_dir / f'{sequence.name}.npz',
                            logits=outputs['logits'][offset:end],
                            labels=sequence.labels, frame_ids=sequence.frame_ids)
        offset = end


def load_outputs(directory, sequences):
    loaded = []
    for sequence in sequences:
        path = Path(directory) / f'{sequence.name}.npz'
        if path.exists():
            with np.load(path) as data:
                loaded.append({key: data[key] for key in data.files})
    return loaded


def have_outputs(directory, sequences):
    return all((Path(directory) / f'{s.name}.npz').exists() for s in sequences)


# ----------------------------------------------------------------------------
# One seed / fold
# ----------------------------------------------------------------------------

def crossfit_folds(train_seqs, n_folds, seed):
    surgeries = sorted({s.surgery for s in train_seqs})
    random.Random(seed).shuffle(surgeries)
    n_folds = min(n_folds, len(surgeries))
    return [surgeries[i::n_folds] for i in range(n_folds)]


def reuse_frame_outputs(source_dir, out_dir, log):
    """Copy another run's exported frame outputs so that only the later stages
    differ (e.g. the cross-fitting ablation uses the identical frame model)."""
    source_dir = Path(source_dir)
    if not (source_dir / 'outputs').is_dir():
        raise FileNotFoundError(f'No frame outputs to reuse in {source_dir}')
    if not (out_dir / 'outputs').exists():
        shutil.copytree(source_dir / 'outputs', out_dir / 'outputs')
    (out_dir / 'frame').mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / 'frame' / 'best_info.json', out_dir / 'frame' / 'best_info.json')
    log(f'reusing frame-model outputs from {source_dir}')


def run_single(cfg, seed, fold, out_dir, device, log, reuse_dir=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get('resume', True) and (out_dir / 'metrics.json').exists():
        log(f'skip {out_dir} (metrics.json exists)')
        return load_json(out_dir / 'metrics.json')
    if reuse_dir is not None:
        reuse_frame_outputs(reuse_dir, out_dir, log)
    splits = build_sequences(cfg['data'], fold)
    if cfg.get('smoke'):
        splits = {k: v[:2] for k, v in splits.items()}
    counts = {k: class_counts(v).tolist() for k, v in splits.items()}
    log(f'seed {seed} fold {fold} | videos ' + ' | '.join(
        f'{k}: {[s.name for s in v]}' for k, v in splits.items()))
    log(f'class counts {counts}')
    result = {'seed': seed, 'fold': fold, 'class_counts': counts}

    # 1-2. Frame model and exported outputs.
    outputs_dir = out_dir / 'outputs'
    if not all(have_outputs(outputs_dir / k, v) for k, v in splits.items()):
        model = train_frame_model(cfg, splits['train'], splits['val'], out_dir / 'frame',
                                  device, seed, log)
        for split, sequences in splits.items():
            export_outputs(model, sequences, cfg, device, outputs_dir / split, seed)
        del model
    outputs = {k: load_outputs(outputs_dir / k, v) for k, v in splits.items()}
    probs = {k: [torch.softmax(torch.from_numpy(o['logits']), 1).numpy() for o in v]
             for k, v in outputs.items()}
    result['frame'] = {k: sequence_metrics(probs[k], splits[k]) for k in ('val', 'test')}
    result['frame']['best'] = load_json(out_dir / 'frame' / 'best_info.json')

    # 3. Moving-average smoothing, window chosen on val only.
    windows = cfg.get('smoothing', {}).get('windows', [1, 3, 5, 7, 9, 11, 15, 21])
    val_scores = {w: sequence_metrics([smooth_probabilities(p, w) for p in probs['val']],
                                      splits['val'])['macro_f1'] for w in windows}
    window = max(windows, key=lambda w: (val_scores[w], -w))
    result['smoothed'] = {
        'window': window, 'val_scores': val_scores,
        **{k: sequence_metrics([smooth_probabilities(p, window) for p in probs[k]], splits[k])
           for k in ('val', 'test')}}
    log(f'smoothing window {window} (val macro-F1 {val_scores[window]:.4f})')

    # 4-5. Temporal model.
    tcfg = cfg.get('temporal', {})
    if tcfg.get('enabled', False):
        train_inputs = outputs['train']
        if tcfg.get('crossfit_folds', 0) > 1:
            crossfit_dir = out_dir / 'crossfit_outputs' / 'train'
            for k, held_out in enumerate(crossfit_folds(splits['train'], tcfg['crossfit_folds'], seed)):
                held = [s for s in splits['train'] if s.surgery in held_out]
                if have_outputs(crossfit_dir, held):
                    continue
                rest = [s for s in splits['train'] if s.surgery not in held_out]
                log(f'cross-fit fold {k}: train on {[s.name for s in rest]}, '
                    f'predict {[s.name for s in held]}')
                fold_cfg = {**cfg, 'train': {**cfg['train'], 'epochs': tcfg.get(
                    'crossfit_epochs', cfg['train']['epochs'])}}
                model = train_frame_model(fold_cfg, rest, splits['val'],
                                          out_dir / 'crossfit' / f'fold{k}', device,
                                          seed * 100 + k + 1, log)
                export_outputs(model, held, cfg, device, crossfit_dir, seed)
                del model
            train_inputs = load_outputs(crossfit_dir, splits['train'])
        else:
            log('[warning] temporal model trained on in-sample frame predictions; '
                'set temporal.crossfit_folds >= 2 for realistic train inputs')
        weight = class_weights(np.concatenate([o['labels'] for o in train_inputs]),
                               NUM_ACTIONS, tcfg.get('class_weights'))
        model, stats, best_epoch = train_temporal(
            train_inputs, outputs['val'], tcfg, NUM_ACTIONS, SHORT_NAMES, device,
            out_dir, log, weight, seed)
        result['temporal'] = {'best_epoch': best_epoch}
        for split in ('val', 'test'):
            data, _ = prepare_inputs(outputs[split], stats)
            result['temporal'][split] = sequence_metrics(
                predict_sequences(model, data, device), splits[split])

    final = result.get('temporal', result['smoothed'])['test']
    plot_confusion(final['confusion_matrix'], SHORT_NAMES, out_dir / 'test_confusion.png')
    save_json(result, out_dir / 'metrics.json')
    for stage in ('frame', 'smoothed', 'temporal'):
        if stage in result:
            m = result[stage]['test']
            log(f'TEST {stage:8s} acc {m["accuracy"]:.4f} macro-F1 {m["macro_f1"]:.4f} '
                f'edit {m["edit"]:.1f} F1@50 {m["f1@50"]:.1f}')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def run_experiment(cfg):
    out_root = Path(cfg['output_dir'])
    out_root.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg.get('device', 'auto'))
    log = Logger(out_root / 'train.log')
    save_yaml(cfg, out_root / 'config.yaml')
    log(f'{cfg["name"]} | {describe_environment(device)}')
    folds = (list(range(cfg['data'].get('n_folds', 4)))
             if cfg['data'].get('split', 'official') == 'surgery_cv' else [0])
    results = []
    for seed in cfg['seeds']:
        for fold in folds:
            relative = Path(f'seed{seed}') / (f'fold{fold}' if len(folds) > 1 else '')
            reuse = cfg.get('reuse_frame_outputs_from')
            results.append(run_single(cfg, seed, fold, out_root / relative, device, log,
                                      reuse_dir=Path(reuse) / relative if reuse else None))

    summary, rows = {'name': cfg['name'], 'n_runs': len(results)}, []
    for stage in ('frame', 'smoothed', 'temporal'):
        if stage in results[0]:
            summary[stage] = {split: aggregate_runs([scalars(r[stage][split]) for r in results])
                              for split in ('val', 'test')}
            rows.append((f'{stage} (test)', summary[stage]['test']))
    save_json(summary, out_root / 'summary.json')
    table = format_summary_table(rows, ['accuracy', 'macro_f1', 'macro_f1_all', 'edit',
                                        'f1@10', 'f1@50'])
    (out_root / 'summary.md').write_text(table + '\n', encoding='utf-8')
    log('\n' + table)
    return summary


def main():
    args = base_arg_parser(__doc__).parse_args()
    run_experiment(resolve_config(args, 'action_recognition'))


if __name__ == '__main__':
    main()
