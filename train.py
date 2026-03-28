import argparse
import os
import sys
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Make engine/ and models/ importable from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "engine"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

from dataset_loader import load_damage_dataset, load_crisisMMD
from dataloaders.sigmoidCrossentropy import ImageTextDataset
from dataloaders.twoStage import ImageTextTwoStage, generate_fewshot_dataframe
from dataloaders.fuselip import ImageTextFuseLIP
from trainer_base import Trainer
from trainer_twoStage import TrainerFirstStep, TrainerSecondStep, freeze_all_except_layernorm
from trainer_mlp import TrainerMLP
from pretrained import load_siglip_pretrained, fuselip_model_loader
from siglip_crossentropy import SigLIPCrossentropy, SigLIP2Crossentropy
from siglip_twostage import SigLIPLinearClassifier, SigLIP2LinearClassifier
from fuselip_mlp import FuseLIPMLPClassifier


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(requested: str = "cuda") -> str:
    if requested == "cuda":
        if torch.cuda.is_available():
            print(f"[INFO] Using CUDA: {torch.cuda.get_device_name(0)}")
            return "cuda"
        elif torch.backends.mps.is_available():
            print("[INFO] CUDA not available, using MPS (Apple Silicon)")
            return "mps"
        else:
            print("[INFO] CUDA not available, falling back to CPU")
            return "cpu"
    return requested


def get_num_workers(device: str) -> int:
    return 4 if device == "cuda" else 0


def get_batch_size(requested: int, device: str) -> int:
    if device == "mps" or device == "cpu":
        return min(requested, 2)
    return requested


def build_description_map(classes, prompts):
    """Build {class_name: prompt} dict from parallel lists."""
    return dict(zip(classes, prompts))


def save_metrics(metrics, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")


def save_confusion_matrix(cm, label_names, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_confusion_matrix.png"))
    plt.close()


def evaluate_contrastive(model, test_df, classes, description_map, processor, device, output_dir, name):
    """
    Zero-shot classification for contrastive models (SigLIP sigmoid/crossentropy).
    Computes cosine similarity between image embeddings and class text embeddings.
    """
    from PIL import Image

    model.eval()

    # Encode class text prompts
    prompts = [description_map[cls] for cls in classes]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)

    with torch.no_grad():
        # Handle both AutoModel and custom wrappers
        if hasattr(model, "base_model"):
            text_out = model.base_model(**text_inputs)
        else:
            text_out = model(**text_inputs)
        text_embeds = F.normalize(text_out.text_embeds, dim=-1)  # (num_classes, D)

    all_preds, all_targets = [], []
    label2id = {cls: i for i, cls in enumerate(classes)}

    for _, row in test_df.iterrows():
        image_path = row["image_path"]
        label = row["labels"]
        if label not in label2id:
            continue
        target = label2id[label]

        try:
            image = Image.open(image_path.replace("\\", "/")).convert("RGB")
        except Exception:
            continue

        img_inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            if hasattr(model, "base_model"):
                img_out = model.base_model(**img_inputs)
            else:
                img_out = model(**img_inputs)
            img_embed = F.normalize(img_out.image_embeds, dim=-1)  # (1, D)

        sims = (img_embed @ text_embeds.t()).squeeze(0)
        pred = sims.argmax().item()
        all_preds.append(pred)
        all_targets.append(target)

    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    print(f"\n=== Test [{name}] ===")
    print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(classification_report(all_targets, all_preds, target_names=classes, zero_division=0))

    cm = confusion_matrix(all_targets, all_preds)
    save_metrics({"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}, output_dir, name)
    save_confusion_matrix(cm, classes, output_dir, name)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def evaluate_classifier(classifier_model, test_loader, classes, device, output_dir, name, loss_key="attr"):
    """Evaluate a SigLIPLinearClassifier on test_loader."""
    classifier_model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            pixel_values = inputs["pixel_values"].to(device)
            if loss_key == "attr":
                logits = classifier_model(pixel_values, no_grad_backbone=True)
            else:
                logits = classifier_model(
                    pixel_values=pixel_values,
                    spatial_shapes=inputs.get("spatial_shapes", None),
                    no_grad_backbone=True,
                )
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.tolist())

    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    print(f"\n=== Test [{name}] ===")
    print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(classification_report(all_targets, all_preds, target_names=classes, zero_division=0))

    cm = confusion_matrix(all_targets, all_preds)
    save_metrics({"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}, output_dir, name)
    save_confusion_matrix(cm, classes, output_dir, name)


# ─────────────────────────────────────────
# Pipelines
# ─────────────────────────────────────────

def run_contrastive(model_name, model_cfg, train_df, val_df, test_df, dataset_cfg, config):
    """Pipeline for siglip_sigmoid and siglip_crossentropy (and siglip2 variants)."""
    hp = model_cfg["hyperparams"]
    device = config["device"]
    pretrained = model_cfg["pretrained"]
    checkpoint = model_cfg["checkpoint"]
    loss_key = hp["loss_key"]
    batch_size = get_batch_size(hp.get("batch_size", 32), device)
    nw = get_num_workers(device)
    classes = dataset_cfg["classes"]
    prompts_en = dataset_cfg["prompts"]["english"]
    description_map = build_description_map(classes, prompts_en)
    output_dir = os.path.join(config["output"]["dir"], dataset_cfg["name"], model_name)

    print(f"\n{'='*60}")
    print(f"Experiment: {model_name} | Dataset: {dataset_cfg['name']}")
    print(f"{'='*60}")
    print(f"batch_size={batch_size} | num_workers={nw} | device={device}")

    # Load model
    if loss_key == "sigmoid":
        model, processor = load_siglip_pretrained(pretrained, device=device)
    else:
        if "siglip2" in model_name:
            model = SigLIP2Crossentropy(pretrained)
        else:
            model = SigLIPCrossentropy(pretrained)
        if hasattr(model.base_model, 'gradient_checkpointing_enable'):
            model.base_model.gradient_checkpointing_enable()
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(pretrained)

    # Build datasets and loaders
    train_dataset = ImageTextDataset(train_df, processor)
    val_dataset = ImageTextDataset(val_df, processor)
    pin = device == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin)

    # Train
    trainer = Trainer(
        model=model,
        model_type=model_name,
        epochs=hp["epochs"],
        lr=hp["lr"],
        device=device,
        loss_key=loss_key,
        weight_decay=hp.get("weight_decay", 1e-4),
        factor=hp.get("factor", 0.5),
        patience=hp.get("patience", 2),
    )
    trainer.fit(train_loader, val_loader, hp["epochs"])
    trainer.save(model, processor, output_dir=checkpoint, model_type=model_name)

    # Evaluate on test set (zero-shot classification)
    evaluate_contrastive(model, test_df, classes, description_map, processor, device, output_dir, model_name)


def run_two_stage(model_name, model_cfg, train_df, val_df, test_df, dataset_cfg, config):
    """Pipeline for siglip_twoStage and fewshot variants (and siglip2)."""
    hp = model_cfg["hyperparams"]
    device = config["device"]
    pretrained = model_cfg["pretrained"]
    checkpoint = model_cfg["checkpoint"]
    loss_key = hp["loss_key"]
    batch_size = get_batch_size(hp.get("batch_size", 32), device)
    nw = get_num_workers(device)
    classes = dataset_cfg["classes"]
    prompts_en = dataset_cfg["prompts"]["english"]
    description_map = build_description_map(classes, prompts_en)
    output_dir = os.path.join(config["output"]["dir"], dataset_cfg["name"], model_name)
    is_siglip2 = "siglip2" in model_name
    is_fewshot = "fewshot" in model_name
    n_shots = hp.get("fewshot_samples_per_class", 16)

    print(f"\n{'='*60}")
    print(f"Experiment: {model_name} | Dataset: {dataset_cfg['name']}")
    print(f"{'='*60}")
    print(f"batch_size={batch_size} | num_workers={nw} | device={device}")

    # Load base model and processor
    base_model, processor = load_siglip_pretrained(pretrained, device=device)
    base_model = base_model.to(device)

    # ─── Stage 1: freeze all except LayerNorm, train contrastively ───
    print("\n--- Stage 1: Contrastive fine-tuning (LayerNorm only) ---")
    freeze_all_except_layernorm(base_model)

    train_dataset_s1 = ImageTextTwoStage(train_df, processor)
    val_dataset_s1 = ImageTextTwoStage(val_df, processor)
    pin = device == "cuda"
    train_loader_s1 = DataLoader(train_dataset_s1, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pin)
    val_loader_s1 = DataLoader(val_dataset_s1, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin)

    trainer_s1 = TrainerFirstStep(
        model=base_model,
        lr=hp["stage1_lr"],
        device=device,
        loss_key=loss_key,
        weight_decay=hp.get("stage1_weight_decay", 1e-3),
    )
    trainer_s1.fit(train_loader_s1, val_loader_s1, epochs=hp["stage1_epochs"])
    trainer_s1.save(base_model, processor, output_dir=checkpoint, model_type=f"{model_name}_stage1")

    # ─── Stage 2: train linear classifier (frozen backbone) ───
    print("\n--- Stage 2: Linear classifier training ---")

    stage2_train_df = train_df
    if is_fewshot:
        stage2_train_df = generate_fewshot_dataframe(train_df, num_shots=n_shots)

    if is_siglip2:
        classifier = SigLIP2LinearClassifier(
            base_model=base_model,
            processor=processor,
            classnames=classes,
            description_map=description_map,
            device=device,
        ).to(device)
    else:
        classifier = SigLIPLinearClassifier(
            base_model=base_model,
            processor=processor,
            classnames=classes,
            description_map=description_map,
            device=device,
        ).to(device)

    train_dataset_s2 = ImageTextTwoStage(stage2_train_df, processor)
    train_loader_s2 = DataLoader(train_dataset_s2, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pin)

    optimizer_s2 = torch.optim.AdamW(
        [classifier.classifier],
        lr=hp["stage2_lr"],
        weight_decay=hp.get("stage2_weight_decay", 1e-3),
    )
    scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s2, T_max=hp.get("stage2_T_max", 70)
    )
    scaler_s2 = torch.cuda.amp.GradScaler() if device == "cuda" else torch.cuda.amp.GradScaler(enabled=False)

    trainer_s2 = TrainerSecondStep(
        model=classifier,
        train_loader=train_loader_s2,
        optimizer=optimizer_s2,
        scheduler=scheduler_s2,
        scaler=scaler_s2,
        device=device,
        num_epochs=hp["stage2_epochs"],
        loss_key=loss_key,
    )
    trainer_s2.train_epochs(hp["stage2_epochs"])
    os.makedirs(os.path.join(checkpoint, model_name), exist_ok=True)
    torch.save(classifier.state_dict(), os.path.join(checkpoint, model_name, "classifier.pth"))

    # Evaluate
    test_dataset = ImageTextTwoStage(test_df, processor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin)
    evaluate_classifier(classifier, test_loader, classes, device, output_dir, model_name, loss_key)


def run_fuselip_mlp(model_name, model_cfg, train_df, val_df, test_df, dataset_cfg, config):
    """Pipeline for fuselip_mlp_{image|text|multimodal}."""
    hp = model_cfg["hyperparams"]
    device = config["device"]
    checkpoint = model_cfg["checkpoint"]
    mode = hp["mode"]
    embed_dim = hp.get("embed_dim", 512)
    batch_size = get_batch_size(hp.get("batch_size", 64), device)
    nw = get_num_workers(device)
    classes = dataset_cfg["classes"]
    output_dir = os.path.join(config["output"]["dir"], dataset_cfg["name"], model_name)

    print(f"\n{'='*60}")
    print(f"Experiment: {model_name} (mode={mode}) | Dataset: {dataset_cfg['name']}")
    print(f"{'='*60}")

    backbone, image_processor, text_tokenizer = fuselip_model_loader(device=device)
    num_classes = len(classes)

    model = FuseLIPMLPClassifier(
        backbone=backbone,
        num_classes=num_classes,
        mode=mode,
        embed_dim=embed_dim,
    ).to(device)

    train_dataset = ImageTextFuseLIP(train_df, image_processor, text_tokenizer)
    val_dataset = ImageTextFuseLIP(val_df, image_processor, text_tokenizer)
    test_dataset = ImageTextFuseLIP(test_df, image_processor, text_tokenizer)

    pin = device == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin)

    trainer = TrainerMLP(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=hp["epochs"],
        lr=hp["lr"],
        device=device,
        weight_decay=hp.get("weight_decay", 1e-4),
    )
    trainer.fit()
    trainer.save(checkpoint, model_name)
    trainer.evaluate_test(test_loader, classes, output_dir, model_name)


def run_eval_only(model_name, model_cfg, test_df, dataset_cfg, config):
    """Zero-shot evaluation without training. Low memory: no optimizer, no gradients stored."""
    device = config["device"]
    pretrained = model_cfg.get("pretrained", "google/siglip-base-patch16-256-multilingual")
    classes = dataset_cfg["classes"]
    prompts_en = dataset_cfg["prompts"]["english"]
    description_map = build_description_map(classes, prompts_en)
    output_dir = os.path.join(config["output"]["dir"], dataset_cfg["name"], model_name)

    print(f"\n{'='*60}")
    print(f"[eval-only] {model_name} | Dataset: {dataset_cfg['name']}")
    print(f"{'='*60}")

    model, processor = load_siglip_pretrained(pretrained, device=device)
    model = model.to(device)
    model.eval()

    evaluate_contrastive(model, test_df, classes, description_map, processor, device, output_dir, model_name)
    del model


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run VLM disaster classification experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--models", nargs="*", default=None, help="Run only these model names (optional filter)")
    parser.add_argument("--datasets", nargs="*", default=None, help="Run only these dataset names (optional filter)")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, run zero-shot evaluation only")
    args = parser.parse_args()

    config = load_config(args.config)
    config["device"] = get_device(config.get("device", "cuda"))
    config["eval_only"] = args.eval_only
    print(f"[INFO] Using device: {config['device']}")
    if config["eval_only"]:
        print("[INFO] eval-only mode: skipping training, running zero-shot evaluation")

    for dataset_cfg in config["datasets"]:
        if args.datasets and dataset_cfg["name"] not in args.datasets:
            continue

        name = dataset_cfg["name"]
        root = dataset_cfg["root"]

        print(f"\n{'#'*60}")
        print(f"Loading dataset: {name} from {root}")
        print(f"{'#'*60}")

        if name == "damage_dataset":
            train_df, val_df, test_df = load_damage_dataset(root)
        elif name == "crisisMMD":
            train_df, val_df, test_df = load_crisisMMD(root)
        else:
            raise ValueError(f"Unknown dataset: {name}")

        print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

        for model_cfg in config["models"]:
            model_name = model_cfg["name"]

            if args.models and model_name not in args.models:
                continue

            try:
                if config["eval_only"]:
                    if model_name in ("siglip_sigmoid", "siglip_crossentropy",
                                      "siglip2_sigmoid", "siglip2_crossentropy",
                                      "siglip_twoStage", "siglip_twoStage_fewshot",
                                      "siglip2_twoStage", "siglip2_twoStage_fewshot"):
                        run_eval_only(model_name, model_cfg, test_df, dataset_cfg, config)
                    else:
                        print(f"[SKIP] eval-only not supported for: {model_name}")
                elif "twoStage" in model_name:
                    run_two_stage(model_name, model_cfg, train_df, val_df, test_df, dataset_cfg, config)
                elif model_name.startswith("fuselip_mlp"):
                    run_fuselip_mlp(model_name, model_cfg, train_df, val_df, test_df, dataset_cfg, config)
                elif model_name in ("siglip_sigmoid", "siglip_crossentropy",
                                    "siglip2_sigmoid", "siglip2_crossentropy"):
                    run_contrastive(model_name, model_cfg, train_df, val_df, test_df, dataset_cfg, config)
                else:
                    print(f"[SKIP] No pipeline defined for: {model_name}")
            except Exception as e:
                print(f"[ERROR] {model_name} on {name}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
