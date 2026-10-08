"""MS-TCN temporal refinement over per-frame outputs (Farha & Gall, CVPR 2019).

The frame model sees one frame at a time; actions, however, are long
contiguous segments. A multi-stage dilated temporal convolutional network
takes the whole sequence of per-frame log-probabilities (or features) and
predicts all frame labels jointly, which removes isolated flickering errors.
The model is non-causal (offline recognition of a recorded video).
"""

import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common.metrics import classification_metrics
from common.utils import save_json

from .metrics import segmental_metrics


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, channels, dropout):
        super().__init__()
        self.conv_dilated = nn.Conv1d(channels, channels, 3, padding=dilation,
                                      dilation=dilation)
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.conv_1x1(F.relu(self.conv_dilated(x))))


class SingleStage(nn.Module):
    def __init__(self, num_layers, channels, in_dim, num_classes, dropout):
        super().__init__()
        self.conv_in = nn.Conv1d(in_dim, channels, 1)
        self.layers = nn.ModuleList([DilatedResidualLayer(2 ** i, channels, dropout)
                                     for i in range(num_layers)])
        self.conv_out = nn.Conv1d(channels, num_classes, 1)

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.layers:
            x = layer(x)
        return self.conv_out(x)


class MSTCN(nn.Module):
    """Input (B, D, T) -> list of per-stage logits (B, C, T)."""

    def __init__(self, in_dim, num_classes, num_stages=3, num_layers=8,
                 channels=64, dropout=0.5):
        super().__init__()
        self.stage1 = SingleStage(num_layers, channels, in_dim, num_classes, dropout)
        self.stages = nn.ModuleList([
            SingleStage(num_layers, channels, num_classes, num_classes, dropout)
            for _ in range(num_stages - 1)])

    def forward(self, x):
        out = self.stage1(x)
        outputs = [out]
        for stage in self.stages:
            out = stage(F.softmax(out, dim=1))
            outputs.append(out)
        return outputs


def temporal_loss(outputs, target, weight, smooth_weight, tau=4.0):
    """Cross-entropy on every stage + truncated MSE over log-prob differences."""
    loss = 0.0
    for out in outputs:
        logits = out[0].transpose(0, 1)  # (T, C)
        loss = loss + F.cross_entropy(logits, target, weight=weight)
        if smooth_weight and out.shape[2] > 1:
            log_probs = F.log_softmax(out, dim=1)
            diff = log_probs[:, :, 1:] - log_probs.detach()[:, :, :-1]
            loss = loss + smooth_weight * torch.clamp(diff ** 2, 0, tau ** 2).mean()
    return loss


def prepare_inputs(sequences, input_kind, stats=None):
    """sequences: list of dicts with 'logits' / 'features' / 'labels'.

    Returns list of (x (D, T) float32, y (T,)) and the normalisation stats.
    """
    if input_kind == 'logits':
        arrays = [torch.log_softmax(torch.from_numpy(s['logits']).float(), 1).numpy()
                  for s in sequences]
    elif input_kind == 'features':
        arrays = [s['features'].astype(np.float32) for s in sequences]
    else:
        raise ValueError(input_kind)
    if stats is None:
        stacked = np.concatenate(arrays)
        stats = (stacked.mean(0), stacked.std(0) + 1e-6)
    mean, std = stats
    data = [(torch.from_numpy(((a - mean) / std).T.copy()),
             torch.from_numpy(s['labels']).long()) for a, s in zip(arrays, sequences)]
    return data, stats


@torch.no_grad()
def predict_sequences(model, data, device):
    model.eval()
    probabilities = []
    for x, _ in data:
        out = model(x[None].to(device))[-1][0]
        probabilities.append(F.softmax(out, dim=0).T.cpu().numpy())
    return probabilities


def evaluate_sequences(probabilities, labels, num_classes, class_names):
    predictions = [p.argmax(1) for p in probabilities]
    metrics = classification_metrics(np.concatenate(labels), np.concatenate(predictions),
                                     num_classes, class_names)
    metrics.update(segmental_metrics(predictions, labels))
    return metrics


def train_temporal(train_seqs, val_seqs, cfg, num_classes, class_names, device,
                   out_dir, log, class_weight=None, seed=0):
    """Train MS-TCN on train sequences; keep the epoch with best val macro-F1."""
    torch.manual_seed(seed)
    random.seed(seed)
    train_data, stats = prepare_inputs(train_seqs, cfg['input'])
    val_data, _ = prepare_inputs(val_seqs, cfg['input'], stats)
    in_dim = train_data[0][0].shape[0]
    model = MSTCN(in_dim, num_classes, cfg.get('num_stages', 3), cfg.get('num_layers', 8),
                  cfg.get('channels', 64), cfg.get('dropout', 0.5)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 5e-4),
                                 weight_decay=cfg.get('weight_decay', 1e-4))
    weight = class_weight.to(device) if class_weight is not None else None
    noise, input_dropout = cfg.get('input_noise', 0.0), cfg.get('input_dropout', 0.0)

    best_score, best_state, best_epoch, history = -1.0, None, 0, []
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        order = list(range(len(train_data)))
        random.shuffle(order)
        total = 0.0
        for index in order:
            x, y = train_data[index]
            x = x.to(device)
            if noise:
                x = x + noise * torch.randn_like(x)
            if input_dropout:
                # Drop whole input channels so no single dimension is trusted.
                x = F.dropout1d(x[None], input_dropout, training=True)[0]
            outputs = model(x[None])
            loss = temporal_loss(outputs, y.to(device), weight, cfg.get('smooth_weight', 0.15))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
        val_probs = predict_sequences(model, val_data, device)
        val_metrics = evaluate_sequences(val_probs, [y.numpy() for _, y in val_data],
                                         num_classes, class_names)
        history.append({'epoch': epoch, 'train_loss': total / len(order),
                        'val_accuracy': val_metrics['accuracy'],
                        'val_macro_f1': val_metrics['macro_f1'],
                        'val_edit': val_metrics['edit']})
        score = val_metrics['macro_f1'] + 1e-3 * val_metrics['edit'] / 100
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch % max(1, cfg['epochs'] // 10) == 0 or epoch == cfg['epochs']:
            log(f'  temporal epoch {epoch:03d} | loss {history[-1]["train_loss"]:.4f} | '
                f'val acc {val_metrics["accuracy"]:.4f} mF1 {val_metrics["macro_f1"]:.4f} '
                f'edit {val_metrics["edit"]:.1f}')
    model.load_state_dict(best_state)
    torch.save({'model': best_state, 'stats': stats, 'cfg': cfg}, out_dir / 'temporal.pt')
    save_json(history, out_dir / 'temporal_history.json')
    log(f'  temporal model selected at epoch {best_epoch}')
    return model, stats, best_epoch
