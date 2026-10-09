"""Frozen-representation evaluation: weighted kNN and a linear probe.

Compares encoders without any fine-tuning, e.g.

    python -m contrastive_learning.probe --encoders \
        random:convnet imagenet:resnet18 \
        runs/contrastive_learning/moco_resnet18_imagenet/seed0/encoder_best.pt \
        --output runs/contrastive_learning/probe
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common.data import build_transform, make_loader
from common.metrics import classification_metrics
from common.networks import build_backbone
from common.utils import get_device, save_json
from instrument_classification.data import (CLASS_NAMES, NUM_CLASSES, InstrumentDataset,
                                            build_splits)


@torch.inference_mode()
def extract_features(backbone, records, image_size, device, batch_size=64):
    dataset = InstrumentDataset(records, build_transform({'image_size': image_size}, False),
                                preload_size=None)
    loader = make_loader(dataset, batch_size, False, 0, 0, device=device)
    backbone.eval()
    features, labels = [], []
    for images, targets, _ in loader:
        features.append(F.normalize(backbone(images.to(device)).float(), dim=1).cpu())
        labels.append(targets)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def knn_predict(train_x, train_y, query_x, k=20, temperature=0.07, num_classes=NUM_CLASSES):
    """Similarity-weighted kNN on L2-normalised features (Wu et al., 2018)."""
    similarity = torch.from_numpy(query_x) @ torch.from_numpy(train_x).T
    k = min(k, similarity.shape[1])
    top_sim, top_idx = similarity.topk(k, dim=1)
    weights = (top_sim / temperature).exp()
    votes = torch.zeros(query_x.shape[0], num_classes)
    votes.scatter_add_(1, torch.from_numpy(train_y)[top_idx], weights)
    return votes.argmax(1).numpy()


def linear_probe(features, labels, c_grid=(0.01, 0.1, 1.0, 10.0, 100.0)):
    """Logistic regression on frozen features; C chosen on val only."""
    scaler = StandardScaler().fit(features['train'])
    scaled = {k: scaler.transform(v) for k, v in features.items()}
    best = None
    for c in c_grid:
        clf = LogisticRegression(C=c, max_iter=5000).fit(scaled['train'], labels['train'])
        val_acc = (clf.predict(scaled['val']) == labels['val']).mean()
        if best is None or val_acc > best[0]:
            best = (val_acc, c, clf)
    _, c, clf = best
    return c, {split: clf.predict(scaled[split]) for split in ('val', 'test')}


def load_encoder(spec, device):
    """'random:<backbone>', 'imagenet:<backbone>' or a MoCo encoder checkpoint."""
    if ':' in spec and not Path(spec).exists():
        kind, name = spec.split(':', 1)
        torch.manual_seed(0)  # makes the random-initialisation baseline reproducible
        return build_backbone(name, pretrained=(kind == 'imagenet')).to(device)
    checkpoint = torch.load(spec, map_location='cpu', weights_only=True)
    backbone = build_backbone(checkpoint['backbone_name'])
    backbone.load_state_dict(checkpoint['backbone'])
    return backbone.to(device)


def evaluate_encoder(backbone, data_root, image_size, device, k=20):
    splits = build_splits({'root': data_root, 'split': 'official'})
    features, labels = {}, {}
    for split, records in splits.items():
        features[split], labels[split] = extract_features(backbone, records, image_size, device)
    result = {}
    for split in ('val', 'test'):
        prediction = knn_predict(features['train'], labels['train'], features[split], k)
        result[f'knn_{split}'] = classification_metrics(labels[split], prediction,
                                                        NUM_CLASSES, CLASS_NAMES)
    c, predictions = linear_probe(features, labels)
    result['linear_probe_C'] = c
    for split in ('val', 'test'):
        result[f'linear_{split}'] = classification_metrics(labels[split], predictions[split],
                                                           NUM_CLASSES, CLASS_NAMES)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoders', nargs='+', required=True)
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--image-size', type=int, nargs=2, default=[224, 384])
    parser.add_argument('--output', default='runs/contrastive_learning/probe')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    device = get_device(args.device)
    summary = {}
    print('| encoder | kNN val | kNN test | linear val | linear test |')
    print('|---|---|---|---|---|')
    for spec in args.encoders:
        result = evaluate_encoder(load_encoder(spec, device), args.data_root,
                                  args.image_size, device)
        summary[spec] = result
        print(f'| {spec} | ' + ' | '.join(
            f'{result[key]["accuracy"]:.4f}'
            for key in ('knn_val', 'knn_test', 'linear_val', 'linear_test')) + ' |')
    save_json(summary, Path(args.output) / 'probe.json')


if __name__ == '__main__':
    main()
