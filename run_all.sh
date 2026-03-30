#!/bin/bash
set -e

source venv/bin/activate

echo "=============================="
echo "  EXPERIMENTOS - damageVLMs"
echo "  $(date)"
echo "=============================="

run() {
  echo ""
  echo ">>> $1"
  echo "    $(date)"
  shift
  caffeinate -d python train.py "$@"
  echo "    ✅ Done: $(date)"
}

# Fase 1 — FuseLIP (backbone congelado)
run "FuseLIP multimodal - damage_dataset"    --config configs/disaster.yaml --models fuselip_mlp_multimodal      --datasets damage_dataset
run "FuseLIP text - damage_dataset"          --config configs/disaster.yaml --models fuselip_mlp_text            --datasets damage_dataset
run "FuseLIP image+text+multi - crisisMMD"   --config configs/disaster.yaml --models fuselip_mlp_image fuselip_mlp_multimodal fuselip_mlp_text --datasets crisisMMD

# Fase 2 — SigLIP contrastivo (fine-tuning completo)
run "SigLIP sigmoid+ce - damage_dataset"     --config configs/disaster.yaml --models siglip_sigmoid siglip_crossentropy          --datasets damage_dataset
run "SigLIP2 sigmoid - damage_dataset"       --config configs/disaster.yaml --models siglip2_sigmoid                             --datasets damage_dataset
run "SigLIP+SigLIP2 - crisisMMD"             --config configs/disaster.yaml --models siglip_sigmoid siglip_crossentropy siglip2_sigmoid --datasets crisisMMD

# Fase 3 — Two-Stage
run "Two-Stage - ambos datasets"             --config configs/disaster.yaml --models siglip_twoStage siglip_twoStage_fewshot --datasets damage_dataset crisisMMD

# Fase 4 — Zero-shot baseline
run "Zero-shot baseline"                     --config configs/disaster.yaml --eval-only

echo ""
echo "=============================="
echo "  TODOS LOS EXPERIMENTOS OK"
echo "  $(date)"
echo "=============================="
