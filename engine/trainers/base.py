import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def plot_training_metric(train_losses, val_losses):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend()
    plt.savefig("training_progress.png")
    plt.close()


class Trainer:
    """
    Trainer for contrastive learning (SigLIP sigmoid/crossentropy).
    Supports both loss_key="sigmoid" and loss_key="crossentropy".
    """

    def __init__(
        self,
        model,
        model_type: str,
        epochs: int,
        lr: float,
        device: str,
        loss_key: str = "sigmoid",
        weight_decay: float = 1e-4,
        factor: float = 0.5,
        patience: int = 2,
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.loss_key = loss_key
        self.model_type = model_type

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=factor, patience=patience
        )

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(self.device)
            pixel_values = batch["pixel_values"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            spatial_shapes = batch.get("spatial_shapes")
            if spatial_shapes is not None:
                spatial_shapes = spatial_shapes.to(self.device)

            # Forward pass — only pass spatial_shapes if the model accepts it
            forward_kwargs = dict(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                return_loss=True,
            )
            if spatial_shapes is not None:
                forward_kwargs["spatial_shapes"] = spatial_shapes
            outputs = self.model(**forward_kwargs)

            loss = outputs["loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def eval_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                pixel_values = batch["pixel_values"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                spatial_shapes = batch.get("spatial_shapes")
                if spatial_shapes is not None:
                    spatial_shapes = spatial_shapes.to(self.device)

                # Forward pass — only pass spatial_shapes if the model accepts it
                forward_kwargs = dict(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    return_loss=True,
                )
                if spatial_shapes is not None:
                    forward_kwargs["spatial_shapes"] = spatial_shapes
                outputs = self.model(**forward_kwargs)

                loss = outputs["loss"]
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def fit(self, train_loader, val_loader, epochs):
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.eval_epoch(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")

        plot_training_metric(train_losses, val_losses)

    def save(self, model, processor, output_dir: str = "/output", model_type: str = ""):
        os.makedirs(f"{output_dir}/{model_type}", exist_ok=True)
        if self.loss_key == "sigmoid":
            model.save_pretrained(f"{output_dir}/{model_type}")
        elif self.loss_key == "crossentropy":
            torch.save(self.model.state_dict(), f"{output_dir}/{model_type}/model.pt")
        
        if processor is not None:
            processor.save_pretrained(f"{output_dir}/{model_type}")
        print(f"Model saved to {output_dir}")