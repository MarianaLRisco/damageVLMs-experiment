"""Classifier evaluation functions."""

import os
import torch
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


def save_metrics(metrics, output_dir, name):
    """Save metrics to text file."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")


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


def evaluate_classifier(
    classifier_model, test_loader, classes, device, output_dir, name, loss_key="attr"
):
    """
    Evaluate a SigLIPLinearClassifier on test_loader.

    Args:
        classifier_model: Trained classifier model
        test_loader: DataLoader for test set
        classes: List of class names
        device: Device to run evaluation on
        output_dir: Directory to save results
        name: Name for this experiment (used in filenames)
        loss_key: Type of loss ("attr" for SigLIP, "dict" for SigLIP2)
    """
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
    print(
        f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
    )
    print(classification_report(all_targets, all_preds, target_names=classes, zero_division=0))

    cm = confusion_matrix(all_targets, all_preds)
    save_metrics(
        {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1},
        output_dir,
        name,
    )
    save_confusion_matrix(cm, classes, output_dir, name)
