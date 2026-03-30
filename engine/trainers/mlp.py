import os
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class TrainerMLP:
    """
    Trains the MLP head of a FuseLIPMLPClassifier with the backbone frozen.
    """

    def __init__(self, model, train_loader, val_loader, epochs, lr, device, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Only optimize the MLP head
        self.optimizer = torch.optim.AdamW(
            self.model.mlp_head.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

    def _forward(self, batch):
        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        return self.model(pixel_values, input_ids, attention_mask)

    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in self.train_loader:
            targets = batch["labels"].to(self.device)
            logits = self._forward(batch)
            loss = F.cross_entropy(logits, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)

        return total_loss / total, correct / total

    def eval_epoch(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                targets = batch["labels"].to(self.device)
                logits = self._forward(batch)
                loss = F.cross_entropy(logits, targets)
                total_loss += loss.item() * targets.size(0)
                correct += (logits.argmax(1) == targets).sum().item()
                total += targets.size(0)
        return total_loss / total, correct / total

    def fit(self):
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.eval_epoch()
            self.scheduler.step(val_loss)

            print(
                f"[Epoch {epoch+1}/{self.epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def save(self, output_dir, model_name):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(output_dir, f"{model_name}.pth"))
        print(f"Model saved to {output_dir}/{model_name}.pth")

    def evaluate_test(self, test_loader, label_names, output_dir, model_name):
        """
        Runs inference on test_loader and saves metrics + confusion matrix.
        """
        self.model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in test_loader:
                targets = batch["labels"].to(self.device)
                logits = self._forward(batch)
                preds = logits.argmax(1)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

        acc = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

        print(f"\n=== Test Results [{model_name}] ===")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print(classification_report(all_targets, all_preds, target_names=label_names, zero_division=0))

        os.makedirs(output_dir, exist_ok=True)

        # Save metrics
        metrics_path = os.path.join(output_dir, f"{model_name}_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Accuracy:  {acc:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1:        {f1:.4f}\n\n")
            f.write(classification_report(all_targets, all_preds, target_names=label_names, zero_division=0))

        # Save confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix — {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
        plt.close()

        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
