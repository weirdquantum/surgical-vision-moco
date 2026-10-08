"""Fast unit tests (no dataset or network download needed): `pytest -q`."""

import numpy as np
import pytest
import torch

from action_recognition.data import label_for_frame, load_ranges, surgery_id
from action_recognition.metrics import edit_score, segmental_metrics, smooth_probabilities
from action_recognition.temporal import MSTCN, temporal_loss
from common.data import group_kfold
from common.engine import build_scheduler, class_weights, mix_batch
from common.metrics import classification_metrics
from common.networks import build_backbone, build_classifier
from common.utils import deep_merge, load_config
from contrastive_learning.moco import MoCo, SplitBatchNorm, convert_split_bn
from instrument_classification.data import parse_record


# --- data parsing ----------------------------------------------------------

def test_parse_record():
    record = parse_record('/x/y_z/v08_038625_Ir.jpg')
    assert (record.video, record.frame, record.label) == ('v08', 38625, 4)
    assert parse_record('/x/.DS_Store') is None
    assert parse_record('/x/ReadMe.txt') is None


def test_ranges_and_labels(tmp_path):
    path = tmp_path / 'a.txt'
    path.write_text('00000,00099,0\n00100,00199,3\n\n')
    ranges = load_ranges(path)
    assert label_for_frame(0, ranges) == 0
    assert label_for_frame(150, ranges) == 3
    with pytest.raises(ValueError):
        label_for_frame(500, ranges)
    path.write_text('0,100,0\n50,150,1\n')
    with pytest.raises(ValueError):
        load_ranges(path)


def test_surgery_id():
    assert surgery_id('video_15_1') == surgery_id('video_15_2') == 'video_15'
    assert surgery_id('video_05') == 'video_05'


def test_group_kfold_disjoint_and_complete():
    groups = [f'v{i:02d}' for i in range(1, 11)]
    seen_test = []
    for fold in range(5):
        train, val, test = group_kfold(groups, 5, fold)
        assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)
        assert sorted(train + val + test) == sorted(groups)
        seen_test += test
    assert sorted(seen_test) == sorted(groups)


# --- configs ----------------------------------------------------------------

def test_config_inheritance_and_overrides():
    cfg = load_config('instrument_classification/configs/legacy_c5.yaml',
                      ['train.epochs=3', 'model.dropout=0.5'])
    assert cfg['model']['backbone'] == 'resnet18' and cfg['model']['pretrained'] is True
    assert cfg['train']['optimizer'] == 'adam' and cfg['train']['epochs'] == 3
    assert cfg['augment']['policy'] == 'legacy_task1'
    assert deep_merge({'a': {'b': 1, 'c': 2}}, {'a': {'b': 3}}) == {'a': {'b': 3, 'c': 2}}


# --- networks ---------------------------------------------------------------

@pytest.mark.parametrize('name', ['convnet', 'resnet18_custom', 'resnet18', 'convnext_tiny',
                                  'efficientnet_b0'])
def test_backbone_shapes(name):
    backbone = build_backbone(name, pretrained=False).eval()
    with torch.no_grad():
        features = backbone(torch.zeros(2, 3, 96, 160))
    assert features.shape == (2, backbone.out_dim)


def test_legacy_convnet_matches_assignment():
    model = build_classifier({'backbone': 'convnet_legacy'}, 7).eval()
    assert model(torch.zeros(1, 3, 175, 300)).shape == (1, 7)
    assert sum(p.numel() for p in model.parameters()) == 26_680_327


# --- engine -----------------------------------------------------------------

def test_class_weights_and_scheduler():
    weights = class_weights([0, 0, 0, 1], 3, 'inverse')
    assert weights[1] > weights[0] and weights[2] == 1.0
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = build_scheduler(optimizer, {'epochs': 10, 'warmup_epochs': 1,
                                            'scheduler': 'cosine', 'min_lr_ratio': 0.0}, 10)
    lrs = []
    for _ in range(100):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == pytest.approx(0.1) and max(lrs) == pytest.approx(1.0) and lrs[-1] < 0.01


def test_mix_batch_targets_sum_to_one():
    images, labels = torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 2, 1])
    _, targets = mix_batch(images, labels, 3, {'mix_prob': 1.0, 'mixup_alpha': 0.2})
    assert torch.allclose(targets.sum(1), torch.ones(4))


def test_macro_f1_ignores_absent_classes():
    metrics = classification_metrics([0, 0, 1, 1], [0, 0, 1, 1], 3)
    assert metrics['macro_f1'] == 1.0 and metrics['macro_f1_all'] == pytest.approx(2 / 3)


# --- temporal ---------------------------------------------------------------

def test_segment_metrics():
    target = np.array([0] * 10 + [1] * 10)
    assert edit_score(target, target) == 100
    flicker = target.copy()
    flicker[3] = 1
    assert edit_score(flicker, target) < 100
    assert segmental_metrics([target], [target])['f1@50'] == 100
    smoothed = smooth_probabilities(np.eye(2)[flicker], 5).argmax(1)
    assert (smoothed == target).all()


def test_mstcn_shapes_and_loss():
    model = MSTCN(in_dim=8, num_classes=8, num_stages=2, num_layers=4, channels=16)
    outputs = model(torch.randn(1, 8, 50))
    assert len(outputs) == 2 and outputs[-1].shape == (1, 8, 50)
    loss = temporal_loss(outputs, torch.randint(0, 8, (50,)), None, 0.15)
    loss.backward()


# --- MoCo -------------------------------------------------------------------

def test_split_bn_state_dict_compatible():
    net = convert_split_bn(torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4)), 2)
    assert isinstance(net[1], SplitBatchNorm)
    reference = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4))
    reference.load_state_dict(net.state_dict())
    net.train()
    assert net(torch.randn(4, 3, 8, 8)).shape == (4, 4, 6, 6)


def test_moco_queue_wraps_and_key_bn_updates():
    cfg = {'backbone': 'convnet', 'queue_size': 12, 'momentum': 0.9, 'temperature': 0.2,
           'mlp_hidden': 32, 'bn_splits': 2, 'symmetric': True, 'key_bn': 'shuffle'}
    model = MoCo(cfg).train()
    before = model.encoder_k.backbone.features[0][1].running_mean.clone()
    for _ in range(3):  # 3 steps x 8 keys = 24 -> wraps a 12-slot queue twice
        loss, accuracy = model(torch.randn(4, 3, 64, 96), torch.randn(4, 3, 64, 96))
        loss.backward()
    assert int(model.queue_ptr) == 24 % 12
    assert torch.allclose(model.queue.norm(dim=0), torch.ones(12), atol=1e-5)
    # The original bug: key BN statistics never moved. Now they must.
    assert not torch.allclose(model.encoder_k.backbone.features[0][1].running_mean, before)
    assert 0.0 <= float(accuracy) <= 1.0
