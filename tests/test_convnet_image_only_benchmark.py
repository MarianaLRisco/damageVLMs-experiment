"""Minimal regression tests for ConvNet image-only benchmark invariants."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from config_loaders.loader import load_config
import train
from engine.dataloaders.image_only import ImageOnlyDataset, build_image_transform
from pipelines import registry
from pipelines.base import PipelineConfig
from pipelines.convnets import ConvNetPipeline


class ConvNetImageOnlyBenchmarkTests(unittest.TestCase):
    """Cover minimal behavior for benchmark dispatch and image-only data loading."""

    def _mk_pipeline_config(
        self,
        model_name: str,
        output_dir: str,
        seed: int = 42,
        image_path: str = "/tmp/sample.jpg",
        sample_labels: list[int] | list[str] = None,
    ) -> PipelineConfig:
        if sample_labels is None:
            sample_labels = [0]

        sample = pd.DataFrame(
            {
                "image_path": [image_path],
                "labels": sample_labels,
                "post_text": [""],
            }
        )
        return PipelineConfig(
            model_name=model_name,
            model_cfg={"name": model_name, "hyperparams": {}},
            dataset_cfg={"name": "damage_dataset", "classes": ["non_damage", "fires"]},
            train_df=sample,
            val_df=sample,
            test_df=sample,
            global_config={"output": {"dir": output_dir}, "seed": seed},
        )

    @staticmethod
    def _mk_image_file(tmp_dir: str, name: str = "sample.jpg") -> str:
        image_path = Path(tmp_dir) / name
        Image.new("RGB", (32, 16), color=(10, 120, 240)).save(image_path)
        return str(image_path)

    class _EmptyLoader:
        def __iter__(self):
            return iter(())

    def test_output_dir_includes_seed(self) -> None:
        """ConvNet output directories must be seed-specific to avoid run collisions."""
        with TemporaryDirectory() as output_dir:
            config = self._mk_pipeline_config("resnet50", output_dir, seed=123)
            pipeline = ConvNetPipeline(config)
            self.assertIn("seed_123", pipeline.output_dir)

    def test_pipeline_registry_dispatches_convnet_models(self) -> None:
        """Ensure configured convnet model names resolve to ConvNetPipeline."""
        with TemporaryDirectory() as output_dir:
            for model_name in ("resnet50", "efficientnet_b0", "vgg16"):
                with self.subTest(model_name=model_name):
                    config = self._mk_pipeline_config(model_name, output_dir)
                    pipeline = registry.get_pipeline(model_name, config)
                    self.assertIsInstance(pipeline, ConvNetPipeline)

    def test_pipeline_registry_rejects_unknown_model(self) -> None:
        """Unknown models must fail fast with a clear ValueError."""
        with TemporaryDirectory() as output_dir:
            config = self._mk_pipeline_config("missing_model", output_dir)
            with self.assertRaisesRegex(ValueError, "No pipeline found for model: missing_model"):
                registry.get_pipeline("missing_model", config)

    def test_resolve_seed_helper_uses_cli_and_defaults(self) -> None:
        """Seed helper should honor CLI override and fallback to config/default."""
        config = {"seed": 7}
        self.assertEqual(train._resolve_seed(123, config), 123)

        config_without_seed = {}
        self.assertEqual(train._resolve_seed(None, config_without_seed), 42)

    def test_build_loaders_sets_seeded_generators(self) -> None:
        """Loaders for ConvNet should use seeded generators and a worker init function on non-Darwin."""
        call_args = []

        def _capture_loader(*args, **kwargs):
            call_args.append(kwargs)
            return self._EmptyLoader()

        config = self._mk_pipeline_config(
            "resnet50",
            output_dir="/tmp",
            seed=555,
            sample_labels=[0],
        )
        pipeline = ConvNetPipeline(config)

        with patch("pipelines.convnets.platform.system", return_value="Linux"), patch(
            "pipelines.convnets.DataLoader",
            side_effect=_capture_loader,
        ):
            train_loader, val_loader, test_loader = pipeline._build_loaders(
                ["non_damage", "fires"],
                image_size=224,
                batch_size=2,
                num_workers=2,
                seed=555,
            )

        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        self.assertEqual(len(call_args), 3)

        for idx, kwargs in enumerate(call_args):
            self.assertIn("generator", kwargs)
            self.assertIn("worker_init_fn", kwargs)
            generator = kwargs["generator"]
            self.assertEqual(generator.initial_seed(), 555 + idx)

    def test_build_loaders_forces_single_process_on_darwin_for_seed_reproducibility(self) -> None:
        """Mac/Apple Silicon should use serial loaders under seeded ConvNet runs."""
        call_args = []

        def _capture_loader(*args, **kwargs):
            call_args.append(kwargs)
            return self._EmptyLoader()

        config = self._mk_pipeline_config(
            "resnet50",
            output_dir="/tmp",
            seed=555,
            sample_labels=[0],
        )
        pipeline = ConvNetPipeline(config)

        with patch("pipelines.convnets.platform.system", return_value="Darwin"), patch(
            "pipelines.convnets.DataLoader",
            side_effect=_capture_loader,
        ):
            train_loader, val_loader, test_loader = pipeline._build_loaders(
                ["non_damage", "fires"],
                image_size=224,
                batch_size=2,
                num_workers=2,
                seed=555,
            )

        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        self.assertEqual(len(call_args), 3)

        for kwargs in call_args:
            self.assertEqual(kwargs["num_workers"], 0)
            self.assertNotIn("worker_init_fn", kwargs)

    def test_convnets_image_only_yaml_loads_expected_shape(self) -> None:
        """Validate essential structure for the image-only benchmark configuration."""
        config_path = (
            Path(__file__).resolve().parent.parent / "configs" / "convnets_image_only.yaml"
        )
        config = load_config(str(config_path))

        self.assertEqual(config.get("device"), "mps")
        self.assertIn("datasets", config)
        self.assertIn("models", config)
        self.assertIsInstance(config["datasets"], list)
        self.assertIsInstance(config["models"], list)
        self.assertGreaterEqual(len(config["datasets"]), 2)
        self.assertGreaterEqual(len(config["models"]), 3)

        dataset_names = [dataset["name"] for dataset in config["datasets"]]
        self.assertEqual(dataset_names[0], "damage_dataset")
        self.assertEqual(dataset_names[1], "crisisMMD")

        model_names = [model_cfg["name"] for model_cfg in config["models"]]
        self.assertEqual(model_names, ["resnet50", "efficientnet_b0", "vgg16"])

        first_dataset = config["datasets"][0]
        self.assertIn("classes", first_dataset)
        self.assertIn("root", first_dataset)
        self.assertGreater(len(first_dataset["classes"]), 0)

        for model_cfg in config["models"]:
            self.assertIn("hyperparams", model_cfg)
            self.assertIn("batch_size", model_cfg["hyperparams"])

    def test_crisismmd_rgbsafe_experiment_config_is_explicit_and_surgical(self) -> None:
        """Experimental rgbsafe config should be explicit and limited to the target setup."""
        config_path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "convnets_image_only_crisismmd_resnet50_rgbsafe.yaml"
        )
        config = load_config(str(config_path))

        self.assertEqual(len(config.get("datasets", [])), 1)
        self.assertEqual(len(config.get("models", [])), 1)

        dataset_cfg = config["datasets"][0]
        model_cfg = config["models"][0]

        self.assertEqual(dataset_cfg["name"], "crisisMMD")
        self.assertEqual(dataset_cfg.get("image_preprocess_mode"), "rgbsafe")
        self.assertEqual(model_cfg["name"], "resnet50")
        self.assertEqual(config["output"]["dir"], "outputs/convnets_crisismmd_resnet50_rgbsafe/")
        self.assertEqual(config.get("seed"), 42)

    def test_crisismmd_resnet50_aug_config_is_explicit(self) -> None:
        """Augmented crisisMMD config must stay explicit and isolated by scope."""
        config_path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "convnets_image_only_crisismmd_resnet50_aug.yaml"
        )
        config = load_config(str(config_path))

        self.assertEqual(len(config.get("datasets", [])), 1)
        self.assertEqual(len(config.get("models", [])), 1)

        dataset_cfg = config["datasets"][0]
        model_cfg = config["models"][0]

        self.assertEqual(dataset_cfg["name"], "crisisMMD")
        self.assertEqual(model_cfg["name"], "resnet50")
        self.assertIn("train_augmentations", dataset_cfg)
        self.assertIn("random_rotation", dataset_cfg["train_augmentations"])
        self.assertIn("color_jitter", dataset_cfg["train_augmentations"])
        self.assertEqual(config["output"]["dir"], "outputs/convnets_crisismmd_resnet50_aug")
        self.assertEqual(config.get("seed"), 42)

    def test_build_image_transform_uses_configured_train_augmentations(self) -> None:
        """Configured image augmentations must be appended in train pipeline."""
        transform = build_image_transform(
            image_size=224,
            is_train=True,
            augmentation_cfg={
                "random_rotation": 8,
                "color_jitter": {
                    "brightness": 0.1,
                    "contrast": 0.1,
                    "saturation": 0.1,
                },
            },
        )

        transform_types = [type(op).__name__ for op in transform.transforms]
        self.assertIn("RandomRotation", transform_types)
        self.assertIn("ColorJitter", transform_types)

    def test_image_only_dataset_shape_and_label_mapping(self) -> None:
        """Dataset rows map labels via class names and read temporary image files correctly."""
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "sample.jpg"
            Image.new("RGB", (32, 16), color=(10, 120, 240)).save(image_path)

            data = pd.DataFrame(
                {
                    "image_path": [str(image_path)],
                    "labels": ["fires"],
                    "post_text": [""],
                }
            )

            dataset = ImageOnlyDataset(
                data,
                classes=["non_damage", "fires", "flood"],
                transform=transforms.Compose(
                    [
                        transforms.Resize((8, 8)),
                        transforms.ToTensor(),
                    ]
                ),
            )

            image, target = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.class_to_idx, {
                "non_damage": 0,
                "fires": 1,
                "flood": 2,
            })
            self.assertEqual(image.shape, (3, 8, 8))
            self.assertEqual(image.dtype, torch.float32)
            self.assertIsInstance(target.item(), int)
            self.assertEqual(int(target.item()), 1)

    def test_image_only_dataset_rgbsafe_handles_palette_transparency(self) -> None:
        """RGB-safe preprocessing should load palette+transparency images without errors."""
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "palette_transparent.png"
            palette_image = Image.new("P", (4, 4), color=0)
            palette_image.putpalette([
                255,
                255,
                255,
                255,
                0,
                0,
            ] + [0, 0, 0] * 254)
            palette_image.paste(1, (1, 1, 3, 3))
            palette_image.info["transparency"] = 0
            palette_image.save(image_path)

            data = pd.DataFrame(
                {
                    "image_path": [str(image_path)],
                    "labels": ["fires"],
                    "post_text": [""],
                }
            )

            dataset = ImageOnlyDataset(
                data,
                classes=["non_damage", "fires"],
                transform=transforms.Compose([transforms.ToTensor()]),
                preprocess_mode="rgbsafe",
            )

            image, target = dataset[0]
            self.assertEqual(image.shape, (3, 4, 4))
            self.assertEqual(int(target.item()), 1)

    def test_image_only_dataset_rgbsafe_handles_palette_without_transparency(self) -> None:
        """RGB-safe preprocessing should also handle opaque palette images deterministically."""
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "palette_opaque.png"
            palette_image = Image.new("P", (4, 4), color=1)
            palette_image.putpalette([
                0,
                0,
                0,
                10,
                120,
                240,
            ] + [0, 0, 0] * 254)
            palette_image.save(image_path)

            data = pd.DataFrame(
                {
                    "image_path": [str(image_path)],
                    "labels": ["fires"],
                    "post_text": [""],
                }
            )

            dataset = ImageOnlyDataset(
                data,
                classes=["non_damage", "fires"],
                transform=transforms.Compose([transforms.ToTensor()]),
                preprocess_mode="rgbsafe",
            )

            image, target = dataset[0]
            self.assertEqual(image.shape, (3, 4, 4))
            self.assertEqual(int(target.item()), 1)

    def test_image_only_dataset_rejects_unknown_preprocess_mode(self) -> None:
        """Unknown preprocess modes must fail fast."""
        sample = pd.DataFrame(
            {
                "image_path": ["/tmp/sample.jpg"],
                "labels": [0],
                "post_text": [""],
            }
        )

        with self.assertRaisesRegex(ValueError, "Unsupported preprocess_mode"):
            ImageOnlyDataset(
                sample,
                classes=["non_damage", "fires"],
                preprocess_mode="unknown_mode",
            )

    def test_preflight_rejects_missing_required_columns(self) -> None:
        """Preflight should fail when required dataset columns are not present."""
        with TemporaryDirectory() as tmp_dir:
            image_path = self._mk_image_file(tmp_dir)
            broken_frame = pd.DataFrame(
                {
                    "image_path": [image_path],
                    "post_text": [""],
                }
            )

            config = PipelineConfig(
                model_name="resnet50",
                model_cfg={"name": "resnet50", "hyperparams": {}},
                dataset_cfg={"name": "damage_dataset", "classes": ["non_damage", "fires"]},
                train_df=broken_frame,
                val_df=broken_frame,
                test_df=broken_frame,
                global_config={"output": {"dir": tmp_dir}},
            )

            pipeline = ConvNetPipeline(config)
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                pipeline._run_preflight_checks()

    def test_preflight_rejects_missing_image_files(self) -> None:
        """Preflight should fail if image files are not found."""
        with TemporaryDirectory() as tmp_dir:
            config = self._mk_pipeline_config(
                model_name="resnet50",
                output_dir=tmp_dir,
                image_path="/definitely/missing/image.jpg",
            )
            pipeline = ConvNetPipeline(config)

            with self.assertRaisesRegex(FileNotFoundError, "missing image files"):
                pipeline._run_preflight_checks()

    def test_preflight_rejects_mps_unavailable(self) -> None:
        """Requested MPS must fail fast when torch reports it unavailable."""
        with TemporaryDirectory() as tmp_dir:
            image_path = self._mk_image_file(tmp_dir)
            sample = pd.DataFrame(
                {
                    "image_path": [image_path],
                    "labels": [0],
                    "post_text": [""],
                }
            )
            config = PipelineConfig(
                model_name="resnet50",
                model_cfg={"name": "resnet50", "hyperparams": {}},
                dataset_cfg={"name": "damage_dataset", "classes": ["non_damage", "fires"]},
                train_df=sample,
                val_df=sample,
                test_df=sample,
                global_config={"output": {"dir": tmp_dir}, "device": "mps"},
            )

            pipeline = ConvNetPipeline(config)
            with patch("torch.backends.mps.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "requested device='mps'"):
                    pipeline._run_preflight_checks()

    def test_run_emits_manifest_and_updates_summary(self) -> None:
        """A full pipeline run must leave per-run manifest and dataset summary artifacts."""
        with TemporaryDirectory() as tmp_dir:
            image_path = self._mk_image_file(tmp_dir)
            config = PipelineConfig(
                model_name="resnet50",
                model_cfg={"name": "resnet50", "hyperparams": {}},
                dataset_cfg={"name": "damage_dataset", "classes": ["non_damage", "fires"]},
                train_df=pd.DataFrame({
                    "image_path": [image_path],
                    "labels": [0],
                    "post_text": [""],
                }),
                val_df=pd.DataFrame({
                    "image_path": [image_path],
                    "labels": [0],
                    "post_text": [""],
                }),
                test_df=pd.DataFrame({
                    "image_path": [image_path],
                    "labels": [0],
                    "post_text": [""],
                }),
                global_config={"output": {"dir": tmp_dir}, "device": "cpu", "seed": 2024},
            )

            pipeline = ConvNetPipeline(config)
            dataset_loader = self._EmptyLoader()

            with patch.object(
                pipeline,
                "_build_loaders",
                return_value=(dataset_loader, dataset_loader, dataset_loader),
            ), patch.object(
                pipeline,
                "_train_one_epoch",
                return_value=(0.4, 0.5),
            ), patch.object(
                pipeline,
                "_evaluate",
                side_effect=[
                    (0.6, 0.3),
                    (0.2, 0.8, "precision\n", torch.tensor([[1, 0], [0, 1]], dtype=torch.long)),
                ],
            ), patch("pipelines.convnets.convnets.build_model", return_value=nn.Linear(1, 2)), patch(
                "torch.save"
            ), patch("pipelines.convnets.save_metrics"), patch(
                "pipelines.convnets.save_confusion_matrix"
            ):
                metrics = pipeline.run()

            self.assertIn("test_accuracy", metrics)
            self.assertIn("test_loss", metrics)

            manifest_path = Path(tmp_dir) / "damage_dataset" / "resnet50" / "seed_2024" / "run_manifest.json"
            summary_path = Path(tmp_dir) / "damage_dataset" / "benchmark_summary.json"

            with open(manifest_path, "r", encoding="utf-8") as fp:
                manifest = json.load(fp)
            with open(summary_path, "r", encoding="utf-8") as fp:
                summary = json.load(fp)

            self.assertEqual(manifest["dataset"], "damage_dataset")
            self.assertEqual(manifest["model"], "resnet50")
            self.assertEqual(manifest["config"]["requested_device"], "cpu")
            self.assertEqual(manifest["resolved_device"], "cpu")
            self.assertEqual(manifest["seed"], 2024)
            self.assertEqual(manifest["config"]["seed"], 2024)
            self.assertIn("seed_reproducibility", manifest)
            self.assertEqual(manifest["seed_reproducibility"]["effective_seed"], 2024)
            self.assertEqual(manifest["seed_reproducibility"]["loader_seed_policy"], "requested_zero_or_negative_workers")

            self.assertGreaterEqual(summary.get("total_runs", 0), 1)
            self.assertEqual(summary["runs"][0]["model"], "resnet50")
            self.assertEqual(summary["runs"][0]["seed"], 2024)
