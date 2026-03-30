import os
import torch
from PIL import Image
from torch.nn.functional import normalize
from sentence_transformers.util import cos_sim
from tqdm.auto import tqdm
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class InferenceClip:
    def __init__(self, model, tokenizer, preprocess, test_data, labels, prompts, output_cfg):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.test_data = test_data
        self.labels = labels
        self.prompts = prompts
        self.output_cfg = output_cfg
        
        # Create output dir
        os.makedirs(self.output_cfg["dir"], exist_ok=True)


    def torchfunction(self, device="cpu"):
        with torch.no_grad():
            tokens = self.tokenizer(self.prompts).to(device)
            text_emb = self.model.encode_text(tokens)
            text_emb = normalize(text_emb, dim=-1)
        return text_emb


    def inference(self, device="cpu"):
        text_emb = self.torchfunction(device)
        preds, probs, errors = [], [], []

        for _, row in tqdm(self.test_data.iterrows(), total=len(self.test_data)):
            img_path = row["image_path"]
            true_label = row["labels"]

            full_path = os.path.join(self.output_cfg["dataset_root"], img_path).replace("\\", "/")

            try:
                image = Image.open(full_path).convert("RGB")
                img_tensor = self.preprocess(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    img_emb = self.model.encode_image(img_tensor)
                    img_emb = normalize(img_emb, dim=-1)

                    sim = cos_sim(img_emb, text_emb)
                    pred_idx = sim.argmax().item()

                pred_label = self.labels[pred_idx]
                pred_prob = sim[0][int(pred_idx)].item()

                preds.append(pred_label)
                probs.append(pred_prob)

                if pred_label != true_label:
                    errors.append({
                        "image_path": img_path,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "predicted_prob": pred_prob
                    })

            except Exception as e:
                print(f"Error procesando {full_path}: {e}")

        return preds, probs, errors


    def confusion_matrix_and_metrics(self):
        true_labels = self.test_data["labels"].tolist()
        preds, probs, errors = self.inference()

        cm = confusion_matrix(true_labels, preds, labels=self.labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.labels, yticklabels=self.labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        cm_path = os.path.join(self.output_cfg["dir"],
                               self.output_cfg["confusion_matrix_filename"])
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix guardada en: {cm_path}")

        # Métricas
        accuracy = accuracy_score(true_labels, preds)
        precision = precision_score(true_labels, preds, average="weighted")
        recall = recall_score(true_labels, preds, average="weighted")
        f1 = f1_score(true_labels, preds, average="weighted")

        metrics_path = os.path.join(self.output_cfg["dir"],
                                    self.output_cfg["metrics_log_filename"])

        with open(metrics_path, "w") as f:
            f.write("Classification Report:\n")
            f.write(str(classification_report(true_labels, preds, target_names=self.labels)))
            f.write("\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")

        print(f"Métricas guardadas en: {metrics_path}")

