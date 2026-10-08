#!/usr/bin/env bash
# Run every experiment of the three projects in dependency order.
#   bash scripts/run_all.sh core   # main results + key comparisons
#   bash scripts/run_all.sh full   # + remaining ablations
# Finished runs are skipped, so the script can simply be re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
TIER=${1:-core}
PY=${PYTHON:-python}
P1=instrument_classification/configs
P2=action_recognition/configs
P3=contrastive_learning/configs
R=runs

p1() { $PY -m instrument_classification.train --config "$P1/$1.yaml"; }
p2() { $PY -m action_recognition.train --config "$P2/$1.yaml"; }
pre() { $PY -m contrastive_learning.pretrain --config "$P3/$1.yaml"; }
ft() { $PY -m instrument_classification.train --config "$P3/$1.yaml"; }

echo "=== Project 1: instrument classification ==="
p1 legacy_c0
p1 legacy_c5
p1 convnet_scratch
p1 resnet18_imagenet
p1 convnext_tiny
p1 efficientnet_v2_s
$PY -m instrument_classification.ensemble "$R/instrument_classification/convnext_tiny" \
    "$R/instrument_classification/efficientnet_v2_s" --output "$R/instrument_classification/ensemble"
p1 convnext_tiny_video_cv
if [[ $TIER == full ]]; then
  for c in legacy_c1 legacy_c2 legacy_c3 legacy_c4 resnet50 resnet18_custom_scratch; do p1 $c; done
fi

echo "=== Project 2: action recognition ==="
[[ -d data/RARP_1FPS_288x512 ]] || $PY -m action_recognition.prepare --data-root data
p2 legacy_resnet18
p2 convnext_tiny_mstcn          # also reports frame-only and smoothed results
if [[ $TIER == full ]]; then
  p2 efficientnet_v2_s_mstcn
  p2 convnext_tiny_surgery_cv
fi

echo "=== Project 3: MoCo ==="
pre legacy_moco_convnet
pre moco_convnet
pre moco_resnet18_imagenet
for c in finetune_convnet_scratch finetune_convnet_legacy_moco finetune_convnet_moco \
         finetune_resnet18_imagenet finetune_resnet18_imagenet_moco; do ft $c; done
ENCODERS=(random:convnet "$R/contrastive_learning/legacy_moco_convnet/seed0/encoder_last.pt"
          "$R/contrastive_learning/moco_convnet/seed0/encoder_best.pt"
          imagenet:resnet18 "$R/contrastive_learning/moco_resnet18_imagenet/seed0/encoder_best.pt")
if [[ $TIER == full ]]; then
  pre moco_resnet18
  ft finetune_resnet18_scratch
  ft finetune_resnet18_moco
  ENCODERS+=(random:resnet18 "$R/contrastive_learning/moco_resnet18/seed0/encoder_best.pt")
fi
$PY -m contrastive_learning.probe --encoders "${ENCODERS[@]}" --output "$R/contrastive_learning/probe"
echo "All done. Summaries: $R/*/*/summary.md"
