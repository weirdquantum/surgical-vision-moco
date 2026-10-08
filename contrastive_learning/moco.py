"""Momentum Contrast (He et al., CVPR 2020) with the MoCo v2 improvements.

Fixes relative to the original notebook implementation:

* Key-encoder BatchNorm. The notebook put ``encoder_k`` in eval mode but only
  momentum-updated its *parameters*; its BN running statistics stayed at the
  initial mean 0 / var 1 forever, so keys were computed with unnormalised
  activations. Here the key encoder runs in train mode with *shuffled*
  BatchNorm (``key_bn: shuffle``); the old behaviour is kept as
  ``key_bn: frozen_eval`` only for reproduction.
* Single-device "Shuffle BN" is simulated with ``SplitBatchNorm`` (statistics
  over sub-batches) so queries and their keys never share BN statistics,
  which otherwise lets the network cheat the pretext task.
* MLP projection head, temperature 0.2, stronger augmentation and an optional
  symmetric loss (MoCo v2 / v3).
"""

import torch
import torch.nn.functional as F
from torch import nn

from common.networks import build_backbone


class SplitBatchNorm(nn.BatchNorm2d):
    """BatchNorm whose training statistics are computed per sub-batch.

    State-dict compatible with ``nn.BatchNorm2d``.
    """

    def __init__(self, num_features, num_splits, **kwargs):
        super().__init__(num_features, **kwargs)
        self.num_splits = num_splits

    def forward(self, x):
        n, c, h, w = x.shape
        if not self.training or self.num_splits == 1:
            return super().forward(x)
        if n % self.num_splits:
            raise ValueError(f'batch {n} not divisible by bn_splits {self.num_splits}')
        mean = self.running_mean.repeat(self.num_splits)
        var = self.running_var.repeat(self.num_splits)
        out = F.batch_norm(
            x.view(-1, c * self.num_splits, h, w), mean, var,
            self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
            True, self.momentum, self.eps).view(n, c, h, w)
        self.running_mean.copy_(mean.view(self.num_splits, c).mean(0))
        self.running_var.copy_(var.view(self.num_splits, c).mean(0))
        return out


def convert_split_bn(module, num_splits):
    if num_splits <= 1:
        return module
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d) and not isinstance(child, SplitBatchNorm):
            split = SplitBatchNorm(child.num_features, num_splits, eps=child.eps,
                                   momentum=child.momentum)
            split.load_state_dict(child.state_dict())
            setattr(module, name, split)
        else:
            convert_split_bn(child, num_splits)
    return module


class Encoder(nn.Module):
    """Backbone (pooled features) + projection head."""

    def __init__(self, backbone, dim, mlp_hidden):
        super().__init__()
        self.backbone = backbone
        in_dim = backbone.out_dim
        if mlp_hidden:
            self.head = nn.Sequential(nn.Linear(in_dim, mlp_hidden), nn.ReLU(inplace=True),
                                      nn.Linear(mlp_hidden, dim))
        else:
            self.head = nn.Linear(in_dim, dim)

    def forward(self, x):
        return self.head(self.backbone(x))


class MoCo(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.K = cfg['queue_size']
        self.m = cfg['momentum']
        self.T = cfg['temperature']
        self.key_bn = cfg.get('key_bn', 'shuffle')
        self.symmetric = cfg.get('symmetric', False)
        dim = cfg.get('dim', 128)
        mlp_hidden = cfg.get('mlp_hidden')

        def make_encoder():
            backbone = build_backbone(cfg['backbone'], pretrained=cfg.get('pretrained', False))
            backbone = convert_split_bn(backbone, cfg.get('bn_splits', 1))
            return Encoder(backbone, dim, mlp_hidden)

        self.encoder_q = make_encoder()
        self.encoder_k = make_encoder()
        self.encoder_k.load_state_dict(self.encoder_q.state_dict())
        for param in self.encoder_k.parameters():
            param.requires_grad_(False)
        self.register_buffer('queue', F.normalize(torch.randn(dim, self.K), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        for q, k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k.mul_(self.m).add_(q.detach(), alpha=1 - self.m)

    @torch.no_grad()
    def _enqueue(self, keys):
        n = keys.shape[0]
        if n > self.K:
            raise ValueError(f'{n} keys per step exceed queue size {self.K}')
        ptr = int(self.queue_ptr)
        index = (torch.arange(n, device=keys.device) + ptr) % self.K
        self.queue[:, index] = keys.T
        self.queue_ptr[0] = (ptr + n) % self.K

    @torch.no_grad()
    def _keys(self, images):
        if self.key_bn == 'frozen_eval':
            self.encoder_k.eval()
            return F.normalize(self.encoder_k(images), dim=1)
        # Shuffle so each key lands in a different BN sub-batch than its query.
        self.encoder_k.train()
        order = torch.randperm(images.shape[0], device=images.device)
        keys = F.normalize(self.encoder_k(images[order]), dim=1)
        return keys[torch.argsort(order)]

    def _contrastive(self, q, k):
        positive = (q * k).sum(1, keepdim=True)
        negative = q @ self.queue.clone().detach()
        logits = torch.cat([positive, negative], 1) / self.T
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
        accuracy = (logits.argmax(1) == 0).float().mean()
        return F.cross_entropy(logits, labels), accuracy

    def forward(self, view1, view2):
        """Returns (loss, top-1 accuracy of picking the positive key)."""
        self._momentum_update()
        q1 = F.normalize(self.encoder_q(view1), dim=1)
        k2 = self._keys(view2)
        loss, accuracy = self._contrastive(q1, k2)
        keys = k2
        if self.symmetric:
            q2 = F.normalize(self.encoder_q(view2), dim=1)
            k1 = self._keys(view1)
            loss2, accuracy2 = self._contrastive(q2, k1)
            loss, accuracy = (loss + loss2) / 2, (accuracy + accuracy2) / 2
            keys = torch.cat([k1, k2])
        self._enqueue(keys)
        return loss, accuracy
