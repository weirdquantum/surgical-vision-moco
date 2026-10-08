"""Backbones shared by the three projects and a generic classifier wrapper.

Every backbone maps (B, 3, H, W) to pooled features (B, D) so that the same
weights can be used for supervised training, MoCo pretraining and fine-tuning.
"""

import torch
from torch import nn
from torchvision import models


# ----------------------------------------------------------------------------
# Custom networks from the assignment
# ----------------------------------------------------------------------------

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class ConvNetBackbone(nn.Module):
    """Five Conv-BN-ReLU-MaxPool blocks (3->32->64->128->256->256) + GAP.

    Same convolutional layout as the assignment's ConvNet / ConvNetNew, but the
    flattened 11,520-d feature map is replaced by global average pooling, which
    removes 23.6M of the original 26.7M parameters and makes the network
    resolution-independent.
    """

    def __init__(self, channels=(32, 64, 128, 256, 256)):
        super().__init__()
        layers, in_channels = [], 3
        for out_channels in channels:
            layers.append(conv_block(in_channels, out_channels))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = in_channels

    def forward(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)


class ConvNetLegacy(nn.Module):
    """The original assignment baseline (flatten -> 2048 -> 1024 -> classes).

    Kept only to reproduce configuration C0; requires 175x300 inputs.
    """

    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                          nn.ReLU(), nn.MaxPool2d(2, 2))
            for i, o in [(3, 32), (32, 64), (64, 128), (128, 256), (256, 256)]
        ])
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 5 * 9, 2048), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    """Basic ResNet block with a projection shortcut when the shape changes."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18Backbone(nn.Module):
    """ResNet-18 written from scratch (He et al., 2016), without the classifier."""

    def __init__(self, zero_init_residual=True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        stages, in_channels = [], 64
        for out_channels, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            stages.append(nn.Sequential(
                ResidualBlock(in_channels, out_channels, stride),
                ResidualBlock(out_channels, out_channels),
            ))
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        if zero_init_residual:
            # Each residual branch starts as identity, which stabilises
            # from-scratch training (Goyal et al., 2017).
            for module in self.modules():
                if isinstance(module, ResidualBlock):
                    nn.init.zeros_(module.bn2.weight)

    def forward(self, x):
        return torch.flatten(self.pool(self.stages(self.stem(x))), 1)


# ----------------------------------------------------------------------------
# torchvision backbones
# ----------------------------------------------------------------------------

TORCHVISION_BACKBONES = {
    'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    'convnext_tiny': (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    'efficientnet_b0': (models.efficientnet_b0,
                        models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    'efficientnet_v2_s': (models.efficientnet_v2_s,
                          models.EfficientNet_V2_S_Weights.IMAGENET1K_V1),
}


def _strip_torchvision_head(name, network):
    if name.startswith('resnet'):
        out_dim = network.fc.in_features
        network.fc = nn.Identity()
    elif name.startswith('convnext'):
        # classifier = [LayerNorm2d, Flatten, Linear]; keep the norm + flatten.
        out_dim = network.classifier[2].in_features
        network.classifier[2] = nn.Identity()
    elif name.startswith('efficientnet'):
        out_dim = network.classifier[1].in_features
        network.classifier = nn.Identity()
    else:
        raise ValueError(name)
    network.out_dim = out_dim
    return network


def build_backbone(name, pretrained=False, drop_path=0.0, zero_init_residual=True):
    """Return a module mapping images to pooled features; ``.out_dim`` is D."""
    if name == 'convnet':
        return ConvNetBackbone()
    if name == 'resnet18_custom':
        return ResNet18Backbone(zero_init_residual=zero_init_residual)
    if name not in TORCHVISION_BACKBONES:
        raise ValueError(f'Unknown backbone {name!r}; choose from '
                         f'convnet, resnet18_custom, {", ".join(TORCHVISION_BACKBONES)}')
    builder, weights = TORCHVISION_BACKBONES[name]
    kwargs = {}
    if drop_path and (name.startswith('convnext') or name.startswith('efficientnet')):
        kwargs['stochastic_depth_prob'] = drop_path
    network = builder(weights=weights if pretrained else None, **kwargs)
    return _strip_torchvision_head(name, network)


class Classifier(nn.Module):
    """Backbone + dropout + linear head. Parameter names start with
    ``backbone.`` / ``head.`` so the optimiser can use separate learning rates."""

    def __init__(self, backbone, num_classes, dropout=0.0):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(backbone.out_dim, num_classes))

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.backbone(x))


def build_classifier(model_cfg, num_classes):
    name = model_cfg['backbone']
    if name == 'convnet_legacy':
        return ConvNetLegacy(num_classes)
    backbone = build_backbone(name, pretrained=model_cfg.get('pretrained', False),
                              drop_path=model_cfg.get('drop_path', 0.0),
                              zero_init_residual=model_cfg.get('zero_init_residual', True))
    model = Classifier(backbone, num_classes, dropout=model_cfg.get('dropout', 0.0))
    init_checkpoint = model_cfg.get('init_checkpoint')
    if init_checkpoint:
        load_backbone_weights(model.backbone, init_checkpoint)
    return model


def load_backbone_weights(backbone, path):
    """Load a ``{'backbone': state_dict}`` checkpoint (e.g. from MoCo)."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint.get('backbone', checkpoint)
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'Backbone mismatch loading {path}: missing={missing}, '
                           f'unexpected={unexpected}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
