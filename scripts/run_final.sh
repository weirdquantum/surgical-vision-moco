#!/usr/bin/env bash
# Remaining experiments after the core + extra runs, in one go.
#   bash scripts/run_final.sh          # required (~3.5 h on an A100)
#   bash scripts/run_final.sh full     # + optional ablations (~3 h more)
# Finished runs are skipped, so the script can be re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
TIER=${1:-required}
PY=${PYTHON:-python}
P1=instrument_classification/configs
P2=action_recognition/configs
P3=contrastive_learning/configs
R=runs

p1() { $PY -m instrument_classification.train --config "$@"; }
p2() { $PY -m action_recognition.train --config "$@"; }

echo "=== [1/5] Project 1: complete the original C0-C5 comparison (multi-seed) ==="
for c in legacy_c1 legacy_c2 legacy_c3 legacy_c4; do p1 "$P1/$c.yaml"; done

echo "=== [2/5] Project 2: MS-TCN under surgery-level CV, cross-fitting ablation ==="
p2 "$P2/convnext_tiny_mstcn_surgery_cv.yaml"
p2 "$P2/convnext_tiny_mstcn_no_crossfit.yaml"

echo "=== [3/5] Project 3: fair 300-epoch comparison scratch vs MoCo ==="
p1 "$P3/finetune_convnet_moco_300ep.yaml"

echo "=== [4/5] Project 3: label efficiency (10% / 25% / 50% of labelled images) ==="
for f in 0.1 0.25 0.5; do
  p1 "$P1/convnet_scratch_300ep.yaml"            --set data.train_fraction=$f --output-dir "$R/label_efficiency/convnet_scratch/f$f"
  p1 "$P3/finetune_convnet_moco_300ep.yaml"      --set data.train_fraction=$f --output-dir "$R/label_efficiency/convnet_moco/f$f"
  p1 "$P3/finetune_resnet18_imagenet.yaml"       --set data.train_fraction=$f --output-dir "$R/label_efficiency/resnet18_imagenet/f$f"
  p1 "$P3/finetune_resnet18_imagenet_moco.yaml"  --set data.train_fraction=$f --output-dir "$R/label_efficiency/resnet18_imagenet_moco/f$f"
done

if [[ $TIER == full ]]; then
  echo "=== [optional] extra backbones and MoCo for a from-scratch ResNet-18 ==="
  p1 "$P1/resnet50.yaml"
  p1 "$P1/resnet18_custom_scratch.yaml"
  p2 "$P2/efficientnet_v2_s_mstcn.yaml"
  $PY -m contrastive_learning.pretrain --config "$P3/moco_resnet18.yaml"
  p1 "$P3/finetune_resnet18_scratch.yaml"
  p1 "$P3/finetune_resnet18_moco.yaml"
fi

echo "=== [5/5] Collect all results ==="
$PY scripts/collect_results.py --runs "$R" --output "$R/ALL_RESULTS.md"
