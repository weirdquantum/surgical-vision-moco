"""Supervised training / evaluation loop shared by the classification projects.

Features: separate backbone/head learning rates, no weight decay on norms and
biases, warmup + cosine (or constant / step) schedules, label smoothing,
class-weighted loss, MixUp/CutMix, gradient clipping, AMP on CUDA, EMA weights,
early stopping and configurable checkpoint selection.
"""

import copy
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .metrics import classification_metrics
from .utils import save_json


# ----------------------------------------------------------------------------
# Optimisation helpers
# ----------------------------------------------------------------------------

def build_param_groups(model, lr, weight_decay, backbone_lr_mult=1.0):
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = name.startswith('backbone.')
        no_decay = param.ndim <= 1 or name.endswith('.bias')
        key = (is_backbone, no_decay)
        if key not in groups:
            groups[key] = {
                'params': [],
                'lr': lr * (backbone_lr_mult if is_backbone else 1.0),
                'weight_decay': 0.0 if no_decay else weight_decay,
            }
        groups[key]['params'].append(param)
    for group in groups.values():
        group['initial_lr'] = group['lr']
    return list(groups.values())


def build_optimizer(model, cfg):
    name = cfg.get('optimizer', 'adamw').lower()
    lr = cfg['lr']
    groups = build_param_groups(model, lr, cfg.get('weight_decay', 0.0),
                                cfg.get('backbone_lr_mult', 1.0))
    if name == 'sgd':
        return torch.optim.SGD(groups, lr=lr, momentum=cfg.get('momentum', 0.0),
                               nesterov=cfg.get('nesterov', False))
    if name == 'adam':
        return torch.optim.Adam(groups, lr=lr)
    if name == 'adamw':
        return torch.optim.AdamW(groups, lr=lr, betas=tuple(cfg.get('betas', (0.9, 0.999))))
    raise ValueError(f'Unknown optimizer {name}')


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """Per-step LambdaLR: linear warmup then cosine / constant / step decay."""
    kind = cfg.get('scheduler', 'cosine')
    epochs = cfg['epochs']
    total = max(1, epochs * steps_per_epoch)
    warmup = int(cfg.get('warmup_epochs', 0) * steps_per_epoch)
    min_ratio = cfg.get('min_lr_ratio', 0.01)
    step_size = cfg.get('step_size', 10) * steps_per_epoch
    gamma = cfg.get('gamma', 0.5)

    def factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        if kind == 'constant':
            return 1.0
        if kind == 'step':
            return gamma ** (step // max(1, step_size))
        if kind == 'cosine':
            progress = (step - warmup) / max(1, total - warmup)
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        raise ValueError(f'Unknown scheduler {kind}')

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def class_weights(labels, num_classes, mode):
    """'none' | 'inverse' | 'sqrt_inverse'; normalised to mean 1 over seen classes."""
    if not mode or mode == 'none':
        return None
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(float)
    weights = np.ones(num_classes)
    seen = counts > 0
    power = 1.0 if mode == 'inverse' else 0.5
    weights[seen] = (counts[seen].sum() / counts[seen]) ** power
    weights[seen] /= weights[seen].mean()
    return torch.tensor(weights, dtype=torch.float32)


class ModelEma:
    """Exponential moving average of weights (and BN buffers)."""

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # Short warmup so early epochs are not dominated by the random init.
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        ema_state = self.module.state_dict()
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                ema_state[key].mul_(decay).add_(value.detach(), alpha=1 - decay)
            else:
                ema_state[key].copy_(value)


def mix_batch(images, labels, num_classes, cfg):
    """MixUp / CutMix with probability ``mix_prob``; returns soft targets."""
    targets = F.one_hot(labels, num_classes).float()
    if np.random.rand() >= cfg.get('mix_prob', 0.0):
        return images, targets
    permutation = torch.randperm(images.size(0), device=images.device)
    use_cutmix = cfg.get('cutmix_alpha', 0) > 0 and (
        cfg.get('mixup_alpha', 0) <= 0 or np.random.rand() < 0.5)
    if use_cutmix:
        lam = np.random.beta(cfg['cutmix_alpha'], cfg['cutmix_alpha'])
        _, _, h, w = images.shape
        cut_h, cut_w = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)
        x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
    else:
        lam = np.random.beta(cfg['mixup_alpha'], cfg['mixup_alpha'])
        images = lam * images + (1 - lam) * images[permutation]
    targets = lam * targets + (1 - lam) * targets[permutation]
    return images, targets


def autocast_context(device, enabled):
    if enabled and device.type == 'cuda':
        return torch.autocast('cuda', dtype=torch.float16)
    return nullcontext()


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

@torch.inference_mode()
def predict(model, loader, device, tta_hflip=False, return_features=False,
            max_steps=None):
    """Run a model over a loader yielding (images, labels, paths)."""
    model.eval()
    logits_list, labels_list, paths, features_list = [], [], [], []
    for step, (images, labels, batch_paths) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        images = images.to(device, non_blocking=True)
        if return_features:
            features = model.forward_features(images)
            logits = model.head(features)
            features_list.append(features.float().cpu())
        else:
            logits = model(images)
        if tta_hflip:
            flipped = torch.flip(images, dims=[3])
            logits = 0.5 * (logits.softmax(1) + model(flipped).softmax(1))
            logits = torch.log(logits.clamp_min(1e-8))
        logits_list.append(logits.float().cpu())
        labels_list.append(labels.cpu())
        paths.extend(batch_paths)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    out = {
        'logits': logits.numpy(),
        'probabilities': logits.softmax(1).numpy(),
        'labels': labels.numpy(),
        'predictions': logits.argmax(1).numpy(),
        'paths': np.asarray(paths, dtype=str),
        'loss': float(F.cross_entropy(logits, labels).item()),
    }
    if return_features:
        out['features'] = torch.cat(features_list).numpy()
    return out


def evaluate(model, loader, device, num_classes, class_names=None, **kwargs):
    outputs = predict(model, loader, device, **kwargs)
    metrics = classification_metrics(outputs['labels'], outputs['predictions'],
                                     num_classes, class_names)
    metrics['loss'] = outputs['loss']
    return metrics, outputs


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def _is_better(value, best, mode, min_delta):
    if best is None:
        return True
    return value < best - min_delta if mode == 'min' else value > best + min_delta


def fit(model, train_loader, val_loader, cfg, device, out_dir, num_classes,
        train_labels, class_names=None, log=print):
    """Train ``model`` and keep the checkpoint chosen by ``cfg['selection']``.

    Returns (history, best_info). ``best.pt`` holds the selected weights (EMA
    weights when EMA is enabled) and ``last.pt`` the final weights.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = cfg['epochs']
    max_steps = cfg.get('max_steps_per_epoch')
    steps_per_epoch = len(train_loader) if not max_steps else min(max_steps, len(train_loader))

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    weights = class_weights(train_labels, num_classes, cfg.get('class_weights'))
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None,
        label_smoothing=cfg.get('label_smoothing', 0.0))
    use_amp = cfg.get('amp', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    ema = ModelEma(model, cfg['ema_decay']) if cfg.get('ema_decay') else None
    mixing = cfg.get('mix_prob', 0) > 0

    selection = cfg.get('selection', {'metric': 'loss', 'mode': 'min'})
    sel_metric, sel_mode = selection['metric'], selection.get('mode', 'min')
    min_delta = selection.get('min_delta', 0.0)
    patience = cfg.get('early_stopping_patience')

    history = {key: [] for key in ['epoch', 'lr', 'train_loss', 'train_accuracy',
                                   'val_loss', 'val_accuracy', 'val_macro_f1',
                                   'val_balanced_accuracy', 'time_s']}
    best_value, best_info, epochs_without_improvement = None, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        total_loss, correct, seen = 0.0, 0, 0
        for step, (images, labels, _) in enumerate(train_loader):
            if max_steps and step >= max_steps:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            targets = labels
            if mixing:
                images, targets = mix_batch(images, labels, num_classes, cfg)
            with autocast_context(device, use_amp):
                logits = model(images)
                loss = criterion(logits.float(), targets)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.get('grad_clip'):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema:
                ema.update(model)
            batch = labels.size(0)
            total_loss += loss.item() * batch
            correct += (logits.argmax(1) == labels).sum().item()
            seen += batch

        eval_model = ema.module if ema else model
        if epoch % cfg.get('eval_every', 1) and epoch != epochs:
            continue
        val_metrics, _ = evaluate(eval_model, val_loader, device, num_classes,
                                  class_names, max_steps=max_steps)
        history['epoch'].append(epoch)
        history['lr'].append(optimizer.param_groups[-1]['lr'])
        history['train_loss'].append(total_loss / max(seen, 1))
        history['train_accuracy'].append(correct / max(seen, 1))
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['val_balanced_accuracy'].append(val_metrics['balanced_accuracy'])
        history['time_s'].append(time.time() - start)

        value = val_metrics[sel_metric]
        improved = _is_better(value, best_value, sel_mode, min_delta)
        if improved:
            best_value = value
            epochs_without_improvement = 0
            best_info = {'epoch': epoch, 'selection_metric': sel_metric,
                         'selection_mode': sel_mode,
                         **{f'val_{k}': val_metrics[k] for k in
                            ('loss', 'accuracy', 'macro_f1', 'balanced_accuracy')}}
            torch.save(eval_model.state_dict(), out_dir / 'best.pt')
        else:
            epochs_without_improvement += 1
        save_json(history, out_dir / 'history.json')
        log(f'epoch {epoch:03d}/{epochs} | lr {history["lr"][-1]:.2e} | '
            f'train loss {history["train_loss"][-1]:.4f} acc {history["train_accuracy"][-1]:.4f} | '
            f'val loss {val_metrics["loss"]:.4f} acc {val_metrics["accuracy"]:.4f} '
            f'mF1 {val_metrics["macro_f1"]:.4f}{" *" if improved else ""} '
            f'| {history["time_s"][-1]:.1f}s')
        if patience and epochs_without_improvement >= patience:
            log(f'early stopping after {epoch} epochs (patience {patience})')
            break

    torch.save(eval_model.state_dict(), out_dir / 'last.pt')
    save_json(best_info, out_dir / 'best_info.json')
    return history, best_info
