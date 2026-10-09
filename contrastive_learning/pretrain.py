"""Self-supervised MoCo pretraining on the unlabelled instrument images.

    python -m contrastive_learning.pretrain \
        --config contrastive_learning/configs/moco_resnet18_imagenet.yaml

Writes ``seed<k>/encoder_last.pt`` and ``seed<k>/encoder_best.pt`` (best
kNN accuracy on the labelled validation split, monitored every
``eval_every`` epochs). Fine-tune them with the ``finetune_*`` configs.
"""

import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

from common.data import IMAGE_EXTENSIONS, build_transform, load_rgb, make_loader
from common.engine import build_optimizer, build_scheduler
from common.plotting import plot_curve
from common.utils import (Logger, base_arg_parser, describe_environment, get_device,
                          resolve_config, save_json, save_yaml, set_seed)
from instrument_classification.data import build_splits

from .moco import MoCo
from .probe import extract_features, knn_predict


class TwoViewDataset(Dataset):
    """Two independently augmented views of each unlabelled image."""

    def __init__(self, paths, transform, preload_size=None):
        self.paths = list(paths)
        self.transform = transform
        self.cache = [load_rgb(p, preload_size) for p in self.paths] if preload_size else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = self.cache[index] if self.cache else load_rgb(self.paths[index])
        return self.transform(image), self.transform(image)


def unlabelled_paths(data_cfg):
    directory = Path(data_cfg['root']) / data_cfg.get('unlabelled_dir', 'dataset-full')
    paths = sorted(str(p) for p in directory.rglob('*')
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f'No images in {directory}')
    return paths


def knn_monitor(backbone, cfg, device, labelled):
    train_x, train_y = extract_features(backbone, labelled['train'],
                                        cfg['monitor']['image_size'], device)
    val_x, val_y = extract_features(backbone, labelled['val'],
                                    cfg['monitor']['image_size'], device)
    prediction = knn_predict(train_x, train_y, val_x, cfg['monitor'].get('k', 20))
    return float((prediction == val_y).mean())


def save_encoder(model, cfg, path, **extra):
    torch.save({'backbone': model.encoder_q.backbone.state_dict(),
                'head': model.encoder_q.head.state_dict(),
                'backbone_name': cfg['moco']['backbone'], **extra}, path)


def run_pretrain(cfg, seed, out_dir, device, log):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get('resume', True) and (out_dir / 'encoder_last.pt').exists():
        log(f'skip {out_dir} (encoder_last.pt exists)')
        return
    set_seed(seed)
    pcfg = cfg['pretrain']
    paths = unlabelled_paths(cfg['data'])
    dataset = TwoViewDataset(paths, build_transform(cfg['augment'], True),
                             cfg['data'].get('preload_size'))
    loader = make_loader(dataset, pcfg['batch_size'], True, cfg['data'].get('num_workers', 2),
                         seed, drop_last=True, device=device)
    labelled = build_splits({**cfg['data'], 'split': 'official'})
    model = MoCo(cfg['moco']).to(device)
    optimizer = build_optimizer(model.encoder_q, pcfg)
    max_steps = pcfg.get('max_steps_per_epoch')
    steps = min(len(loader), max_steps) if max_steps else len(loader)
    scheduler = build_scheduler(optimizer, pcfg, steps)
    log(f'{len(dataset)} unlabelled images | {steps} steps/epoch | '
        f'queue {model.K} | T {model.T} | m {model.m} | key BN {model.key_bn}')

    history = {'epoch': [], 'loss': [], 'positive_accuracy': [], 'knn_val': []}
    best_knn = -1.0
    for epoch in range(1, pcfg['epochs'] + 1):
        model.train()
        start, total_loss, total_acc = time.time(), 0.0, 0.0
        for step, (view1, view2) in enumerate(loader):
            if max_steps and step >= max_steps:
                break
            loss, accuracy = model(view1.to(device, non_blocking=True),
                                   view2.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            total_acc += accuracy.item()
        history['epoch'].append(epoch)
        history['loss'].append(total_loss / steps)
        history['positive_accuracy'].append(total_acc / steps)
        knn = None
        if epoch % pcfg.get('eval_every', 20) == 0 or epoch == pcfg['epochs']:
            knn = knn_monitor(model.encoder_q.backbone, cfg, device, labelled)
            if knn > best_knn:
                best_knn = knn
                save_encoder(model, cfg, out_dir / 'encoder_best.pt', epoch=epoch, knn_val=knn)
        history['knn_val'].append(knn)
        save_json(history, out_dir / 'history.json')
        if knn is not None or epoch % 10 == 0 or epoch == 1:
            knn_text = f' | kNN val {knn:.4f}' if knn is not None else ''
            log(f'epoch {epoch:04d}/{pcfg["epochs"]} | loss {history["loss"][-1]:.4f} | '
                f'pos-acc {history["positive_accuracy"][-1]:.3f} | lr '
                f'{optimizer.param_groups[0]["lr"]:.2e} | {time.time() - start:.1f}s{knn_text}')
    save_encoder(model, cfg, out_dir / 'encoder_last.pt', epoch=pcfg['epochs'])
    plot_curve(history['loss'], out_dir / 'loss.png', 'epoch', 'InfoNCE loss', cfg['name'])
    knn_points = [(e, v) for e, v in zip(history['epoch'], history['knn_val']) if v is not None]
    save_json({'best_knn_val': best_knn, 'knn_curve': knn_points,
               'final_loss': history['loss'][-1]}, out_dir / 'summary.json')
    log(f'done | best kNN val {best_knn:.4f} | final loss {history["loss"][-1]:.4f}')


def main():
    args = base_arg_parser(__doc__).parse_args()
    cfg = resolve_config(args, 'contrastive_learning')
    out_root = Path(cfg['output_dir'])
    device = get_device(cfg.get('device', 'auto'))
    log = Logger(out_root / 'pretrain.log')
    save_yaml(cfg, out_root / 'config.yaml')
    log(f'{cfg["name"]} | {describe_environment(device)}')
    for seed in cfg['seeds']:
        run_pretrain(cfg, seed, out_root / f'seed{seed}', device, log)


if __name__ == '__main__':
    main()
