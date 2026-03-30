"""Contrastive model evaluation functions."""

import os
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


def save_metrics(metrics, output_dir, name, report=None):
    """Save metrics to text file."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        if report:
            f.write("\n")
            f.write(report)


def save_confusion_matrix(cm, label_names, output_dir, name):
    """Save confusion matrix as PNG heatmap."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=label_names,
        yticklabels=label_names,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_confusion_matrix.png"))
    plt.close()


def evaluate_contrastive(
    model, test_df, classes, description_map, processor, device, output_dir, name
):
    """
    Zero-shot classification for contrastive models (SigLIP sigmoid/crossentropy).
    Computes cosine similarity between image embeddings and class text embeddings.

    Args:
        model: Contrastive model (SigLIP or similar)
        test_df: Test DataFrame with image_path and labels columns
        classes: List of class names
        description_map: Dict mapping class names to text prompts
        processor: Model processor for tokenizing inputs
        device: Device to run evaluation on
        output_dir: Directory to save results
        name: Name for this experiment (used in filenames)

    Returns:
        Dict with accuracy, precision, recall, f1 scores
    """
    model.eval()
    base_model = model.base_model if hasattr(model, "base_model") else model

    # Encode class text prompts
    prompts = [description_map[cls] for cls in classes]
    text_inputs = processor(
        text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=64
    ).to(device)

    with torch.no_grad():
        if hasattr(base_model, "get_text_features"):
            text_out = base_model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask", None),
            )
            # Extract tensor from BaseModelOutputWithPooling
            if hasattr(text_out, "text_embeds"):
                text_features = text_out.text_embeds
            elif hasattr(text_out, "pooler_output"):
                text_features = text_out.pooler_output
            elif isinstance(text_out, dict) and "pooler_output" in text_out:
                text_features = text_out["pooler_output"]
            else:
                # Safe access to last_hidden_state for dict-like outputs
                if isinstance(text_out, dict):
                    last_hidden = text_out.get("last_hidden_state")
                    if last_hidden is not None:
                        text_features = last_hidden[:, 0, :]
                    else:
                        raise ValueError("Cannot extract text features from model output")
                else:
                    text_features = text_out.last_hidden_state[:, 0, :]
        else:
            text_out = base_model(**text_inputs)
            # Handle different model output formats
            if hasattr(text_out, "text_embeds"):
                text_features = text_out.text_embeds
            elif hasattr(text_out, "pooler_output"):
                text_features = text_out.pooler_output
            elif isinstance(text_out, dict) and "pooler_output" in text_out:
                text_features = text_out["pooler_output"]
            else:
                # Safe access to last_hidden_state for dict-like outputs
                if isinstance(text_out, dict):
                    last_hidden = text_out.get("last_hidden_state")
                    if last_hidden is not None:
                        text_features = last_hidden[:, 0, :]
                    else:
                        raise ValueError("Cannot extract text features from model output")
                else:
                    text_features = text_out.last_hidden_state[:, 0, :]

        # Ensure text_features is a tensor
        if not isinstance(text_features, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(text_features)}")

        text_embeds = F.normalize(text_features, dim=-1)  # (num_classes, D)

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
            if hasattr(base_model, "get_image_features"):
                image_feature_kwargs = {"pixel_values": img_inputs["pixel_values"]}
                if "pixel_attention_mask" in img_inputs:
                    image_feature_kwargs["pixel_attention_mask"] = img_inputs[
                        "pixel_attention_mask"
                    ]
                if "spatial_shapes" in img_inputs:
                    image_feature_kwargs["spatial_shapes"] = img_inputs["spatial_shapes"]
                img_out = base_model.get_image_features(**image_feature_kwargs)
                # Extract tensor from BaseModelOutputWithPooling
                if hasattr(img_out, "image_embeds"):
                    image_features = img_out.image_embeds
                elif hasattr(img_out, "pooler_output"):
                    image_features = img_out.pooler_output
                elif isinstance(img_out, dict) and "pooler_output" in img_out:
                    image_features = img_out["pooler_output"]
                else:
                    # Safe access to last_hidden_state for dict-like outputs
                    if isinstance(img_out, dict):
                        last_hidden = img_out.get("last_hidden_state")
                        if last_hidden is not None:
                            image_features = last_hidden[:, 0, :]
                        else:
                            raise ValueError("Cannot extract image features from model output")
                    else:
                        image_features = img_out.last_hidden_state[:, 0, :]
            else:
                img_out = base_model(**img_inputs)
                # Handle different model output formats
                if hasattr(img_out, "image_embeds"):
                    image_features = img_out.image_embeds
                elif hasattr(img_out, "pooler_output"):
                    image_features = img_out.pooler_output
                elif isinstance(img_out, dict) and "pooler_output" in img_out:
                    image_features = img_out["pooler_output"]
                else:
                    # Safe access to last_hidden_state for dict-like outputs
                    if isinstance(img_out, dict):
                        last_hidden = img_out.get("last_hidden_state")
                        if last_hidden is not None:
                            image_features = last_hidden[:, 0, :]
                        else:
                            raise ValueError("Cannot extract image features from model output")
                    else:
                        image_features = img_out.last_hidden_state[:, 0, :]

            # Ensure image_features is a tensor
            if not isinstance(image_features, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(image_features)}")

            img_embed = F.normalize(image_features, dim=-1)  # (1, D)

        sims = (img_embed @ text_embeds.t()).squeeze(0)
        pred = sims.argmax().item()
        all_preds.append(pred)
        all_targets.append(target)

    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    print(f"\n=== Test [{name}] ===")
    print(
        f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
    )
    print(classification_report(all_targets, all_preds, target_names=classes, zero_division=0))

    report = classification_report(all_targets, all_preds, target_names=classes, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    save_metrics(
        {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1},
        output_dir,
        name,
        report=report,
    )
    save_confusion_matrix(cm, classes, output_dir, name)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
