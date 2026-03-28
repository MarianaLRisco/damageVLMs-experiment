import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from torch import Tensor
from contextlib import nullcontext
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
    plt.savefig("training_loss.png")
    plt.close()

def freeze_all_except_layernorm(model):
    """Congela todos los parámetros excepto las capas LayerNorm (Stage 1 de TwoStage)."""
    for name, param in model.named_parameters():
        if "layernorm" in name.lower() or "layer_norm" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False


#siglip two-stage trainer
class TrainerFirstStep:
    def __init__(self, model, lr, device, loss_key = 'attr', weight_decay=1e-4, factor=0.5, patience=2):
        self.model = model.to(device)
        self.device = device
        self.loss_key = loss_key
        self.weight_decay = weight_decay
        self.patience = patience
        self.factor = factor
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=factor, patience=patience
        )
    
    def _get_loss(self, outputs):
        if self.loss_key == "attr":
            return outputs.loss
        elif self.loss_key == "dict":
            return outputs["loss"]
        else:
            raise ValueError(f"loss_key inválido: {self.loss_key}")

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch, _ = batch
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
                batch, _ = batch
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

            print(f"Epoch {epoch+1}/{epochs} | Train Loss {train_loss:.4f} | Va l Loss {val_loss:.4f}")

        plot_training_metric(train_losses, val_losses)
    
    def save(self, model, processor, output_dir = "/output", model_type =str):
        os.makedirs(output_dir, exist_ok=True)
        if self.loss_key == "attr":
            model.save_pretrained(f"{output_dir}/{model_type}")
        elif self.loss_key == "dict":
            torch.save(self.model.state_dict(), f"{output_dir}/{model_type}/model.pt")
        
        if processor is not None:
            processor.save_pretrained(f"{output_dir}/{model_type}")
        print(f"Model saved to {output_dir}")




class TrainerSecondStep:
    def __init__(self, model, train_loader,optimizer, scheduler, scaler, device, num_epochs, loss_key='attr'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.num_epochs = num_epochs
        self.loss_key = loss_key
        self.train_loader = train_loader
    
    def train_epochs(self, num_epochs):
        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for inputs, targets in self.train_loader:
                # Mover datos a GPU
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                targets = targets.to(self.device)

                # Forward con autocast (solo CUDA)
                device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
                with torch.amp.autocast(device_type=device_type, enabled=(self.device == "cuda")):
                    if self.loss_key == "attr":
                        logits = self.model(inputs["pixel_values"], no_grad_backbone=True)
                    elif self.loss_key == "dict":
                        logits = self.model(pixel_values=inputs["pixel_values"], spatial_shapes=inputs["spatial_shapes"], no_grad_backbone=True)
                    else:
                        raise ValueError(f"loss_key inválido: {self.loss_key}")
                    # logits = self.model(inputs["pixel_values"], no_grad_backbone=True)
                    loss = F.cross_entropy(logits, targets)

                # Backward con gradiente escalado
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()  
                self.scheduler.step()

                # Métricas acumuladas
                total_loss += loss.item() * targets.size(0)
                total_correct += (logits.argmax(dim=1) == targets).sum().item()
                total_samples += targets.size(0)

            # Métricas por epoch
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            current_lr = self.scheduler.get_last_lr()[0]

            print(
                f"[Epoch {epoch+1}/{num_epochs}] "
                f"LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}"
            )
    def save (self, model, model_type, output_dir):
        #save classifier model
        torch.save(model.state_dict(), f"{output_dir}/{model_type}/{model_type}.pth")   
