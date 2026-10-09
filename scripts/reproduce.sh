#!/usr/bin/env bash
# Reproduce every experiment reported in the READMEs, in dependency order.
#   bash scripts/reproduce.sh          # all three projects (~8 h on an A100)
#   bash scripts/reproduce.sh p2       # one project: p1 | p2 | p3
# Finished runs are skipped, so the script can simply be re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
ONLY=${1:-all}
PY=${PYTHON:-python}
P1=instrument_classification/configs
P2=action_recognition/configs
P3=contrastive_learning/configs
R=runs

cls() { $PY -m instrument_classification.train --config "$1"; }
act() { $PY -m action_recognition.train --config "$1"; }

if [[ $ONLY == all || $ONLY == p1 ]]; then
  echo "=== Project 1: instrument classification ==="
  for c in legacy_c0 legacy_c1 legacy_c2 legacy_c3 legacy_c4 legacy_c5 \
           convnet_scratch convnet_scratch_300ep resnet18_imagenet convnext_tiny efficientnet_v2_s; do
    cls "$P1/$c.yaml"
  done
  $PY -m instrument_classification.ensemble "$R/instrument_classification/convnext_tiny" \
      "$R/instrument_classification/efficientnet_v2_s" --output "$R/instrument_classification/ensemble"
  cls "$P1/convnext_tiny_video_cv.yaml"
fi

if [[ $ONLY == all || $ONLY == p2 ]]; then
  echo "=== Project 2: action recognition ==="
  [[ -d data/RARP_1FPS_288x512 ]] || $PY -m action_recognition.prepare --data-root data
  act "$P2/legacy_resnet18.yaml"
  act "$P2/convnext_tiny_mstcn.yaml"
  act "$P2/convnext_tiny_mstcn_no_crossfit.yaml"   # reuses the frame model above
  act "$P2/convnext_tiny_surgery_cv.yaml"
  act "$P2/convnext_tiny_mstcn_surgery_cv.yaml"    # reuses the frame models above
fi

if [[ $ONLY == all || $ONLY == p3 ]]; then
  echo "=== Project 3: MoCo ==="
  for c in legacy_moco_convnet moco_convnet moco_resnet18_imagenet; do
    $PY -m contrastive_learning.pretrain --config "$P3/$c.yaml"
  done
  # finetune_convnet_moco_300ep is compared with project 1's convnet_scratch_300ep.
  for c in finetune_convnet_scratch finetune_convnet_legacy_moco finetune_convnet_moco \
           finetune_convnet_moco_300ep finetune_resnet18_imagenet finetune_resnet18_imagenet_moco; do
    cls "$P3/$c.yaml"
  done
  $PY -m contrastive_learning.probe --output "$R/contrastive_learning/probe" --encoders \
      random:convnet "$R/contrastive_learning/legacy_moco_convnet/seed0/encoder_last.pt" \
      "$R/contrastive_learning/moco_convnet/seed0/encoder_best.pt" \
      imagenet:resnet18 "$R/contrastive_learning/moco_resnet18_imagenet/seed0/encoder_best.pt"
fi

$PY scripts/collect_results.py --runs "$R" --output "$R/ALL_RESULTS.md"
