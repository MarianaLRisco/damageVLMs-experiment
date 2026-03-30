# engine/trainer_base.py
import torch
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def plot_training_metric(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

#siglip and siglip2 trainer base
class Trainer:
    def __init__(self, model, model_type, epochs, lr, device, loss_key, weight_decay=1e-4, factor=0.5, patience=2):
        self.model = model.to(device)
        self.model_type = model_type
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.loss_key = loss_key
        self.weight_decay = weight_decay
        self.patience = patience
        self.factor = factor
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=factor, patience=patience
        )
    
    def _get_loss(self, outputs):
        if self.loss_key == "sigmoid":
            return outputs.loss
        elif self.loss_key == "crossentropy":
            return outputs["loss"]
        else:
            raise ValueError(f"loss_key inválido: {self.loss_key}")

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch, return_loss=True)
            loss = self._get_loss(outputs)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
        
        return total_loss / len(train_loader)

    def eval_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch, return_loss=True)
                total_loss +=  self._get_loss(outputs).item()
        
        return total_loss / len(val_loader)

    def fit(self, train_loader, val_loader, epochs):
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.eval_epoch(val_loader)

            self.scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")

        plot_training_metric(train_losses, val_losses)

    def save(self, model, processor, output_dir = "/output", model_type =str):
        os.makedirs(output_dir, exist_ok=True)
        if self.loss_key == "sigmoid":
            model.save_pretrained(f"{output_dir}/{model_type}")
        elif self.loss_key == "crossentropy":
            torch.save(self.model.state_dict(), f"{output_dir}/{model_type}/model.pt")
        
        if processor is not None:
            processor.save_pretrained(f"{output_dir}/{model_type}")
        print(f"Model saved to {output_dir}")




