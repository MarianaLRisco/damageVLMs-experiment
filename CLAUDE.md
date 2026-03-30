# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchmarking framework for evaluating Vision-Language Models (VLMs) on disaster/crisis image classification. Compares multiple architectures (SigLIP, SigLIP2, FuseLIP) across training strategies (contrastive, two-stage, few-shot, MLP head).

## Commands

### Training

```bash
# Run full training for all models and datasets
python train.py --config configs/disaster.yaml

# Filter by specific models or datasets
python train.py --config configs/disaster.yaml --models siglip_sigmoid siglip_twoStage
python train.py --config configs/disaster.yaml --datasets damage_dataset

# Zero-shot evaluation only (no training)
python train.py --config configs/disaster.yaml --eval-only

# Combine filters
python train.py --config configs/disaster.yaml --models fuselip_mlp_image --datasets crisisMMD --eval-only
```

### Environment

```bash
pip install -r requirements.txt
```

Docker uses Python 3.11 with CUDA 11.8 (see `Dockerfile`).

## Architecture

### Entry Point: `train.py`

Dispatches to one of four pipelines based on model type from YAML config:

| Function | Models | Strategy |
|---|---|---|
| `run_contrastive()` | `siglip_sigmoid`, `siglip_crossentropy`, `siglip2_*` | Full fine-tune with contrastive loss |
| `run_two_stage()` | `siglip_twoStage`, `siglip2_twoStage*` | Stage 1: LayerNorm-only contrastive; Stage 2: linear classifier |
| `run_fuselip_mlp()` | `fuselip_mlp_{image,text,multimodal}` | Frozen FuseLIP backbone + MLP head |
| `run_eval_only()` | any | Zero-shot cosine similarity scoring |

### Config: `configs/disaster.yaml`

Central configuration for all datasets and model hyperparameters. Each model entry specifies: `pretrained`, `loss` type, `lr`, `epochs`, `batch_size`, and `output_dir`. Adding a new model/dataset only requires a YAML entry and an appropriate pipeline function.

### Data Flow

1. `engine/dataset_loader.py` loads CSVs → returns `(train_df, val_df, test_df)` DataFrames with standardized columns: `image_path`, `post_text`, `labels`
2. `engine/dataloaders/` — one dataloader class per model family (matches processor/tokenizer format expected by each backbone)
3. `engine/trainer_*.py` — trainer classes handle optimization, scheduling, checkpointing, and metrics

### Models (`models/`)

- `pretrained.py` — loaders for each backbone family (`load_siglip_pretrained`, `fuselip_model_loader`, etc.)
- `siglip_crossentropy.py` — SigLIP/SigLIP2 wrappers with symmetric cross-entropy loss (temperature=0.07)
- `siglip_twostage.py` — frozen backbone + trainable `nn.Linear` classifier; class embeddings computed from text descriptions at init
- `fuselip_mlp.py` — frozen FuseLIP + MLP head (`Linear→ReLU→Dropout→Linear→ReLU→Linear`); supports image/text/multimodal embedding concatenation

### Trainers (`engine/`)

- `trainer_base.py` — contrastive trainer (AdamW + ReduceLROnPlateau); handles both `sigmoid` and `crossentropy` loss keys
- `trainer_twoStage.py` — Stage 1 (`TrainerFirstStep`, LayerNorm-only params) + Stage 2 (`TrainerSecondStep`, AMP + CosineAnnealingLR on classifier only); `loss_key` is `"attr"` for SigLIP, `"dict"` for SigLIP2
- `trainer_mlp.py` — MLP trainer with best-state checkpointing; calls `evaluate_test()` to save metrics and confusion matrix

### Outputs

- Metrics saved to `outputs/disasternet/{model_name}_metrics.txt`
- Confusion matrices saved to `outputs/disasternet/{model_name}_confusion_matrix.png`
- Checkpoints saved to `checkpoints/`

## Dataset Layout

Expected structure under `dataset/`:
```
dataset/
├── damage_dataset/
│   ├── train_data_clean.csv   # columns: image_path, post_text, labels
│   └── test_data_clean.csv
└── crisisMMD/
    └── csv_splits/
        ├── train.csv          # columns: image, tweet_text, label
        ├── dev.csv
        └── test.csv
```

Validation split for `damage_dataset` is created automatically (10% stratified from train).

## Key Implementation Details

- **Device priority**: CUDA → MPS (Apple Silicon) → CPU; `batch_size` auto-reduces to 2 on MPS/CPU
- **SigLIP2 difference**: requires `spatial_shapes` argument and uses `loss_key="dict"` in two-stage trainers
- **Few-shot**: `generate_fewshot_dataframe()` in `engine/dataloaders/twoStage.py` samples N examples per class (default 16, with replacement)
- **Zero-shot evaluation**: `evaluate_contrastive()` computes cosine similarity between image embeddings and per-class text description embeddings
