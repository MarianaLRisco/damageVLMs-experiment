from pipelines.base import BasePipeline
from data.utils import get_batch_size, get_num_workers


import torch
import torch.nn as nn
import torch.optim as optim
import src.models.convnets as convnets

class MultiModelPipeline(BasePipeline):

    def run(self):

        model = convnets(
            model_name=self.cfg.model_name,
            num_classes=self.cfg.model_cfg["num_classes"],
            trainable_layers=self.cfg.model_cfg.get("trainable_layers", 2)
        )

        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.cfg.model_cfg.get("lr", 1e-3)
        )

        train_loader = self._build_dataloader(self.cfg.train_df)
        val_loader = self._build_dataloader(self.cfg.val_df)

        for epoch in range(self.cfg.model_cfg.get("epochs", 5)):
            model.train()

            total_loss = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

        return {"loss": total_loss}