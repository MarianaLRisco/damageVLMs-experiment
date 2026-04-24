"""Image-only ConvNet training pipeline."""

from __future__ import annotations

import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from functools import partial
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from data.utils import get_batch_size, get_num_workers
from engine.dataloaders.image_only import ImageOnlyDataset, build_image_transform
from evaluation.classifier import save_confusion_matrix, save_metrics
from pipelines.base import BasePipeline
from models import convnets
from sklearn.metrics import classification_report, confusion_matrix


ModelMetrics = Dict[str, float]
ManifestData = Dict[str, Any]
SummaryData = Dict[str, Any]


REQUIRED_COLUMNS: tuple[str, ...] = ("image_path", "labels")


def _seed_worker(worker_id: int, base_seed: int) -> None:
    """Seed worker-local RNGs for deterministic dataloader behavior."""
    worker_seed = int(base_seed + worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _coerce_label(value: Any, class_to_idx: Mapping[str, int]) -> int:
    """Normalize label values to integer class indexes.

    Supports:
    - class names present in class_to_idx
    - integer-like values
    - numpy/pandas numeric values
    """
    if pd.isna(value):
        raise ValueError("Missing label value")

    if isinstance(value, str):
        if value in class_to_idx:
            return class_to_idx[value]
        value = value.strip()
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)

    if isinstance(value, torch.Tensor):
        value = value.item()

    if isinstance(value, float) and value.is_integer():
        value = int(value)

    try:
        return int(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Unsupported label value: {value!r}") from err


class ConvNetPipeline(BasePipeline):
    """Training pipeline for image-only ConvNet baselines."""

    def _build_output_dir(self) -> str:
        """Build ConvNet output path including experiment seed identity."""
        base_output_dir = super()._build_output_dir()
        seed = self._resolve_seed()
        return os.path.join(base_output_dir, f"seed_{seed}")

    def _resolve_seed(self) -> int:
        """Resolve and validate random seed from global configuration."""
        configured_seed = self.cfg.global_config.get("seed")
        if configured_seed is None:
            return 42

        try:
            return int(configured_seed)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Invalid ConvNet seed value: {configured_seed!r}") from err

    def _seed_everything(self, seed: int) -> None:
        """Seed all deterministic randomness channels used by ConvNet flow."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)

    def _resolve_dataloader_workers(self, requested_num_workers: int, seed: int | None) -> Tuple[int, str]:
        """Resolve loader workers and policy under a safe seed policy."""
        normalized_workers = max(0, int(requested_num_workers))

        if normalized_workers <= 0:
            return 0, "requested_zero_or_negative_workers"

        if seed is not None and platform.system() == "Darwin":
            return 0, "single_process_for_seeded_dataloader_on_darwin"

        return normalized_workers, "picklable_worker_init_fn"

    def _make_loader_generator(self, base_seed: int, split_index: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(base_seed + split_index)
        return generator

    def run(self) -> ModelMetrics:
        run_started_at = datetime.now(timezone.utc)
        seed = self._resolve_seed()
        self._seed_everything(seed)

        self._run_preflight_checks()

        hp = self.cfg.model_cfg.get("hyperparams", self.cfg.model_cfg)
        dataset_classes: Sequence[str] = self.cfg.dataset_cfg["classes"]
        num_classes = self.cfg.model_cfg.get("num_classes", len(dataset_classes))
        if num_classes <= 0:
            num_classes = len(dataset_classes)
        image_size = self.cfg.model_cfg.get("image_size", 224)
        trainable_layers = int(self.cfg.model_cfg.get("trainable_layers", 2))
        use_pretrained = bool(self.cfg.model_cfg.get("use_pretrained", True))

        epochs = int(hp.get("epochs", 1))
        batch_size = get_batch_size(int(hp.get("batch_size", 32)), self.device)
        lr = float(hp.get("lr", 1e-3))
        weight_decay = float(hp.get("weight_decay", 0.0))
        num_workers = get_num_workers(self.device)
        num_workers = int(hp.get("num_workers", num_workers))
        num_workers, dataloader_seed_policy = self._resolve_dataloader_workers(
            num_workers,
            seed,
        )

        if num_classes <= 0:
            raise ValueError("num_classes must be > 0 for ConvNet training")

        print(f"\n{'=' * 60}")
        print(f"Experiment: {self.cfg.model_name} | Dataset: {self.cfg.dataset_cfg['name']}")
        print(
            f"batch_size={batch_size} | lr={lr} | epochs={epochs} | "
            f"device={self.device} | num_workers={num_workers} | seed={seed} | "
            f"dataloader_seed_policy={dataloader_seed_policy}"
        )
        print(f"{'=' * 60}")

        model = convnets.build_model(
            model_name=self.cfg.model_name,
            num_classes=num_classes,
            trainable_layers=trainable_layers,
            use_pretrained=use_pretrained,
        ).to(self.device)

        train_loader, val_loader, test_loader = self._build_loaders(
            dataset_classes,
            image_size,
            batch_size,
            num_workers,
            seed,
        )
        image_preprocess_mode = str(self.cfg.dataset_cfg.get("image_preprocess_mode", "legacy_rgb"))

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for ConvNet model")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

        best_state = None
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self._evaluate(model, val_loader, criterion)

            print(
                f"[Epoch {epoch:02d}/{epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        test_loss, test_acc, report, cm = self._evaluate(
            model,
            test_loader,
            criterion,
            include_report=True,
        )

        metrics = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
        }

        save_path = Path(self.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model_out = save_path / f"{self.cfg.model_name}_best.pth"
        torch.save(model.state_dict(), str(model_out))

        save_metrics(
            metrics,
            self.output_dir,
            self.cfg.model_name,
            report=f"{report}\n",
        )
        save_confusion_matrix(cm, list(dataset_classes), self.output_dir, self.cfg.model_name)

        run_finished_at = datetime.now(timezone.utc)
        manifest_data = self._build_run_manifest(
            dataset_classes=dataset_classes,
            num_classes=num_classes,
            image_preprocess_mode=image_preprocess_mode,
            seed=seed,
            loader_seed_policy=dataloader_seed_policy,
            dataloader_workers=num_workers,
            hp=hp,
            dataset_sizes={
                "train": len(self.cfg.train_df),
                "val": len(self.cfg.val_df),
                "test": len(self.cfg.test_df),
            },
            run_started_at=run_started_at,
            run_finished_at=run_finished_at,
            metrics=metrics,
        )
        run_manifest_path = Path(self.output_dir) / "run_manifest.json"
        with open(run_manifest_path, "w", encoding="utf-8") as fp:
            json.dump(manifest_data, fp, indent=2)

        self._update_benchmark_summary(manifest_data, metrics)

        return metrics

    def _run_preflight_checks(self) -> None:
        """Validate config and dataset before heavy execution."""
        self._validate_expected_config()
        self._validate_device_contract()

        dataset_classes: Sequence[str] = self.cfg.dataset_cfg["classes"]
        num_classes = self.cfg.model_cfg.get("num_classes", len(dataset_classes))
        if num_classes <= 0:
            num_classes = len(dataset_classes)

        class_to_idx = {name: idx for idx, name in enumerate(dataset_classes)}
        for split_name, split_df in (
            ("train", self.cfg.train_df),
            ("val", self.cfg.val_df),
            ("test", self.cfg.test_df),
        ):
            self._validate_dataset_split(split_name, split_df, class_to_idx, num_classes)

    def _validate_expected_config(self) -> None:
        """Validate expected config shape for ConvNet image-only benchmark."""
        if "name" not in self.cfg.dataset_cfg or not isinstance(self.cfg.dataset_cfg["name"], str):
            raise ValueError("dataset_cfg must include a non-empty 'name'")

        classes = self.cfg.dataset_cfg.get("classes")
        if not isinstance(classes, Iterable) or isinstance(classes, (str, bytes)):
            raise ValueError("dataset_cfg['classes'] must be a non-empty sequence of class names")

        class_list = list(classes)
        if len(class_list) < 2:
            raise ValueError("ConvNet benchmark requires at least 2 classes")
        if len(set(class_list)) != len(class_list):
            raise ValueError("dataset_cfg['classes'] contains duplicate class names")

        hyperparams = self.cfg.model_cfg.get("hyperparams", {})
        if not isinstance(hyperparams, dict):
            raise ValueError("model_cfg['hyperparams'] must be a dictionary when provided")

        if "epochs" in hyperparams and int(hyperparams["epochs"]) <= 0:
            raise ValueError("hyperparams['epochs'] must be > 0")

        requested_device = self.cfg.global_config.get("device", "cpu")
        if requested_device not in {"cpu", "cuda", "mps"}:
            raise ValueError("global_config['device'] must be 'cpu', 'cuda' or 'mps'")

        num_classes = self.cfg.model_cfg.get("num_classes", 0)
        if num_classes and num_classes != len(class_list):
            raise ValueError("model_cfg['num_classes'] must match len(dataset_cfg['classes'])")

    def _validate_dataset_split(
        self,
        split_name: str,
        df: pd.DataFrame,
        class_to_idx: Mapping[str, int],
        num_classes: int,
    ) -> None:
        """Validate dataframe schema, label values, and image file existence."""
        if df is None or df.empty:
            raise ValueError(f"Split '{split_name}' dataframe is empty")

        missing_columns = {column for column in REQUIRED_COLUMNS if column not in df.columns}
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Split '{split_name}' missing required columns: {missing}")

        missing_images: list[str] = []
        invalid_labels: set[int] = set()
        for idx, raw_label in enumerate(df["labels"]):
            image_path = Path(df.iloc[idx]["image_path"])
            if not image_path.is_file():
                missing_images.append(str(image_path))

            try:
                label_idx = _coerce_label(raw_label, class_to_idx)
            except ValueError as err:
                raise ValueError(
                    f"Split '{split_name}' contains invalid label at row {idx}: {raw_label!r}"
                ) from err

            if label_idx < 0 or label_idx >= num_classes:
                invalid_labels.add(label_idx)

        if missing_images:
            preview = ", ".join(missing_images[:5])
            raise FileNotFoundError(
                f"Split '{split_name}' has missing image files ({len(missing_images)} total): {preview}"
            )

        if invalid_labels:
            values = ", ".join(str(value) for value in sorted(invalid_labels))
            raise ValueError(
                f"Split '{split_name}' contains labels outside model classes: [{values}]"
            )

    def _validate_device_contract(self) -> None:
        """Fail fast when this benchmark explicitly requests mps but it is unavailable."""
        requested_device = self.cfg.global_config.get("device", self.device)
        if requested_device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "ConvNet image-only benchmark requested device='mps', but PyTorch MPS is not available."
            )

    def _build_run_manifest(
        self,
        dataset_classes: Sequence[str],
        num_classes: int,
        image_preprocess_mode: str,
        seed: int,
        loader_seed_policy: str,
        dataloader_workers: int,
        hp: Dict[str, Any],
        dataset_sizes: Mapping[str, int],
        run_started_at: datetime,
        run_finished_at: datetime,
        metrics: Mapping[str, float],
    ) -> ManifestData:
        """Create run-level metadata payload."""
        return {
            "run_id": run_started_at.strftime("%Y%m%dT%H%M%S%fZ"),
            "run_started_at": run_started_at.isoformat(),
            "run_finished_at": run_finished_at.isoformat(),
            "duration_seconds": (run_finished_at - run_started_at).total_seconds(),
            "pipeline": self.__class__.__name__,
            "dataset": self.cfg.dataset_cfg["name"],
            "model": self.cfg.model_name,
            "config": {
                "requested_device": self.cfg.global_config.get("device"),
                "seed": seed,
                "model_cfg": self.cfg.model_cfg,
                "dataset_cfg": {
                    "name": self.cfg.dataset_cfg["name"],
                    "classes": list(dataset_classes),
                    "num_classes": num_classes,
                    "image_preprocess_mode": image_preprocess_mode,
                },
                "global_output_dir": self.cfg.global_config["output"]["dir"],
                "hyperparams": dict(hp),
            },
            "seed_reproducibility": {
                "effective_seed": seed,
                "requested_seed": self.cfg.global_config.get("seed"),
                "loader_seed_policy": loader_seed_policy,
                "loader_num_workers": dataloader_workers,
            },
            "resolved_device": self.device,
            "seed": seed,
            "dataset_sizes": dict(dataset_sizes),
            "num_classes": num_classes,
            "class_count": len(dataset_classes),
            "metrics": dict(metrics),
            "output_dir": str(Path(self.output_dir)),
        }

    def _update_benchmark_summary(self, manifest_data: ManifestData, metrics: Mapping[str, float]) -> None:
        """Update or create a consolidatable benchmark summary for this dataset."""
        output_root = Path(self.cfg.global_config["output"]["dir"])
        summary_path = output_root / self.cfg.dataset_cfg["name"] / "benchmark_summary.json"

        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as fp:
                    summary_data: SummaryData = json.load(fp)
            except json.JSONDecodeError:
                summary_data = {}
        else:
            summary_data = {}

        if not isinstance(summary_data, dict):
            summary_data = {}

        runs = summary_data.get("runs")
        if not isinstance(runs, list):
            runs = []

        run_record = {
            "run_id": manifest_data["run_id"],
            "dataset": manifest_data["dataset"],
            "model": manifest_data["model"],
            "seed": manifest_data.get("seed"),
            "timestamp": manifest_data["run_finished_at"],
            "resolved_device": manifest_data["resolved_device"],
            "requested_device": manifest_data["config"].get("requested_device"),
            "metrics": dict(metrics),
            "output_dir": manifest_data["output_dir"],
            "hyperparams": manifest_data["config"].get("hyperparams", {}),
        }
        runs.append(run_record)

        benchmark_entries: dict[str, dict[str, float | str]] = {
            key: value
            for key, value in (summary_data.get("benchmark_best", {}) if isinstance(summary_data.get("benchmark_best"), dict) else {}).items()
        }
        test_accuracy = metrics.get("test_accuracy")
        if isinstance(test_accuracy, (int, float)):
            model_name = manifest_data["model"]
            previous_best = benchmark_entries.get(model_name, {})
            if not previous_best:
                benchmark_entries[model_name] = {
                    "test_accuracy": test_accuracy,
                    "run_id": manifest_data["run_id"],
                    "timestamp": manifest_data["run_finished_at"],
                }
            elif test_accuracy > float(previous_best.get("test_accuracy", float("-inf"))):
                benchmark_entries[model_name] = {
                    "test_accuracy": test_accuracy,
                    "run_id": manifest_data["run_id"],
                    "timestamp": manifest_data["run_finished_at"],
                }

        summary_data.update(
            {
                "schema": "convnet-image-only-benchmark/v1",
                "generated_at": manifest_data["run_finished_at"],
                "dataset": manifest_data["dataset"],
                "runs": runs,
                "total_runs": len(runs),
                "benchmark_best": benchmark_entries,
            }
        )

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as fp:
            json.dump(summary_data, fp, indent=2)

    def _build_loaders(
        self,
        dataset_classes: Sequence[str],
        image_size: int,
        batch_size: int,
        num_workers: int,
        seed: int,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Build train/val/test loaders for image-only ConvNets."""
        num_workers, _ = self._resolve_dataloader_workers(num_workers, seed)
        augmentation_cfg = self.cfg.dataset_cfg.get("train_augmentations")
        image_preprocess_mode = str(self.cfg.dataset_cfg.get("image_preprocess_mode", "legacy_rgb"))

        train_transform = build_image_transform(
            image_size=image_size,
            is_train=True,
            augmentation_cfg=augmentation_cfg,
        )
        valid_transform = build_image_transform(image_size, is_train=False)

        pin_memory = self.device == "cuda"
        loader_params = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": False,
        }

        if num_workers > 0:
            loader_params["worker_init_fn"] = partial(_seed_worker, base_seed=seed)

        train_loader = DataLoader(
            ImageOnlyDataset(
                self.cfg.train_df,
                dataset_classes,
                transform=train_transform,
                preprocess_mode=image_preprocess_mode,
            ),
            shuffle=True,
            generator=self._make_loader_generator(seed, split_index=0),
            **loader_params,
        )
        val_loader = DataLoader(
            ImageOnlyDataset(
                self.cfg.val_df,
                dataset_classes,
                transform=valid_transform,
                preprocess_mode=image_preprocess_mode,
            ),
            shuffle=False,
            generator=self._make_loader_generator(seed, split_index=1),
            **loader_params,
        )
        test_loader = DataLoader(
            ImageOnlyDataset(
                self.cfg.test_df,
                dataset_classes,
                transform=valid_transform,
                preprocess_mode=image_preprocess_mode,
            ),
            shuffle=False,
            generator=self._make_loader_generator(seed, split_index=2),
            **loader_params,
        )

        return train_loader, val_loader, test_loader

    def _train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float]:
        model.train()
        running_loss = 0.0
        total_samples = 0
        correct = 0

        for images, targets in loader:
            images, targets = images.to(self.device), targets.to(self.device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_len = targets.size(0)
            running_loss += float(loss.item()) * batch_len
            total_samples += batch_len
            correct += int((outputs.argmax(dim=1) == targets).sum().item())

        train_loss = running_loss / max(total_samples, 1)
        train_acc = correct / max(total_samples, 1)
        return train_loss, train_acc

    def _evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        include_report: bool = False,
    ) -> Tuple[float, float, str, torch.Tensor] | Tuple[float, float]:
        model.eval()
        running_loss = 0.0
        total_samples = 0
        correct = 0
        all_preds: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            for images, targets in loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                batch_len = targets.size(0)
                running_loss += float(loss.item()) * batch_len
                total_samples += batch_len
                preds = outputs.argmax(dim=1)
                correct += int((preds == targets).sum().item())
                all_preds.extend(int(v) for v in preds.cpu().tolist())
                all_targets.extend(int(v) for v in targets.cpu().tolist())

        loss = running_loss / max(total_samples, 1)
        acc = correct / max(total_samples, 1)

        if not include_report:
            return loss, acc

        report = classification_report(
            all_targets,
            all_preds,
            zero_division=0,
        )
        cm = confusion_matrix(all_targets, all_preds)
        print(f"\n=== Test [{self.cfg.model_name}] ===")
        print(f"Accuracy: {acc:.4f}")
        print(report)
        return loss, acc, report, torch.as_tensor(cm, dtype=torch.long)
