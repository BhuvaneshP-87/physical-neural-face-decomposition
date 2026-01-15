"""Training utilities for the neural residual appearance model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..config import TrainingConfig
from ..evaluation.metrics import compute_evaluation_metrics
from ..evaluation.visualization import save_image_grid
from ..experiments.checkpointing import CheckpointManager
from ..config import ExperimentConfig
from ..models.residual_net import ResidualAppearanceNet
from ..optimization.losses import PerceptualLoss, compute_inverse_rendering_losses
from ..renderer.geometry import depth_to_normals
from ..renderer.torch_renderer import TorchFaceRenderer


@dataclass(slots=True)
class TrainingLog:
    """Aggregated training statistics."""

    epochs: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    psnr: list[float] = field(default_factory=list)
    ssim: list[float] = field(default_factory=list)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


class ResidualTrainer:
    """Train a residual appearance network using ground-truth or synthetic supervision."""

    def __init__(
        self,
        model: ResidualAppearanceNet | None = None,
        renderer: TorchFaceRenderer | None = None,
        config: TrainingConfig | None = None,
        experiment_config: ExperimentConfig | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = (model or ResidualAppearanceNet()).to(self.device)
        self.renderer = renderer or TorchFaceRenderer()
        self.config = config or TrainingConfig()
        self.experiment_config = experiment_config or ExperimentConfig()
        self.perceptual_loss = PerceptualLoss(use_vgg=True)
        self.optimizer = Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            self.experiment_config.metadata.run_dir,
            checkpoint_dir=self.config.checkpoint_dir,
            log_dir=self.config.log_dir,
            keep_last=self.config.keep_last_checkpoints,
        )
        self.checkpoint_manager.save_config(self.experiment_config)
        self.checkpoint_dir = self.checkpoint_manager.checkpoint_dir

    def train_epoch(self, loader: DataLoader) -> tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0

        for batch in loader:
            batch = _move_batch_to_device(batch, self.device)
            image = batch["image"]
            if image.ndim == 3:
                image = image.unsqueeze(0)
            depth = batch["depth"]
            if depth.ndim == 3:
                depth = depth.unsqueeze(0)
            albedo = batch["albedo"]
            if albedo.ndim == 3:
                albedo = albedo.unsqueeze(0)
            lighting = batch["lighting"]
            if lighting.ndim == 2:
                lighting = lighting.unsqueeze(0)
            mask = batch.get("mask")
            if mask is not None and mask.ndim == 3:
                mask = mask.unsqueeze(0)

            self.optimizer.zero_grad(set_to_none=True)
            physical = self.renderer(depth, albedo, lighting, mask=mask)
            normals = depth_to_normals(depth)
            residual_output = self.model(
                physical.image if physical.image.ndim == 4 else physical.image.unsqueeze(0),
                albedo=albedo,
                normals=normals,
                depth=depth,
            )
            losses = compute_inverse_rendering_losses(
                residual_output.refined,
                image,
                mask=mask,
                lighting=lighting,
                albedo=albedo,
                depth=depth,
                residual=residual_output.residual,
                perceptual_loss=self.perceptual_loss,
            )
            total = (
                losses.reconstruction
                + 0.5 * losses.perceptual
                + 0.1 * losses.residual_penalty
                + 0.01 * losses.albedo_smoothness
                + 0.01 * losses.depth_smoothness
            )
            total.backward()
            self.optimizer.step()

            metrics = compute_evaluation_metrics(residual_output.refined.detach(), image.detach(), use_lpips=False)
            total_loss += float(total.detach().cpu())
            total_psnr += metrics.psnr
            total_ssim += metrics.ssim
            num_batches += 1

        if num_batches == 0:
            return 0.0, 0.0, 0.0
        return total_loss / num_batches, total_psnr / num_batches, total_ssim / num_batches

    def fit(
        self,
        dataset,
        *,
        epochs: int | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        log_every: int | None = None,
        preview_path: str | Path | None = None,
        resume_from: str | Path | None = None,
    ) -> TrainingLog:
        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size
        num_workers = num_workers or self.config.num_workers
        log_every = log_every or self.config.log_every
        save_every = self.config.save_every

        resume_path = Path(resume_from) if resume_from is not None else self.experiment_config.metadata.resume_from
        start_epoch = 0
        if resume_path is not None:
            checkpoint = self.checkpoint_manager.load_checkpoint(resume_path)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint.get("epoch", 0))

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        log = TrainingLog()

        for epoch in range(start_epoch, start_epoch + epochs):
            loss, psnr_value, ssim_value = self.train_epoch(loader)
            epoch_number = epoch + 1
            log.epochs.append(epoch_number)
            log.losses.append(loss)
            log.psnr.append(psnr_value)
            log.ssim.append(ssim_value)

            metrics = {"loss": loss, "psnr": psnr_value, "ssim": ssim_value}
            self.checkpoint_manager.log_metrics(epoch_number, metrics, split="train")

            if epoch_number % save_every == 0 or epoch == start_epoch or epoch == start_epoch + epochs - 1:
                self.checkpoint_manager.save_training_state(
                    epoch=epoch_number,
                    model=self.model,
                    optimizer=self.optimizer,
                    metrics=metrics,
                )

            if preview_path is not None and len(dataset) > 0 and (epoch_number % log_every == 0 or epoch == start_epoch or epoch == start_epoch + epochs - 1):
                sample = dataset[0]
                sample = _move_batch_to_device(sample, self.device)
                image = sample["image"].unsqueeze(0) if sample["image"].ndim == 3 else sample["image"]
                depth = sample["depth"].unsqueeze(0) if sample["depth"].ndim == 3 else sample["depth"]
                albedo = sample["albedo"].unsqueeze(0) if sample["albedo"].ndim == 3 else sample["albedo"]
                lighting = sample["lighting"].unsqueeze(0) if sample["lighting"].ndim == 2 else sample["lighting"]
                mask = sample.get("mask")
                if mask is not None and mask.ndim == 3:
                    mask = mask.unsqueeze(0)
                physical = self.renderer(depth, albedo, lighting, mask=mask)
                normals = depth_to_normals(depth)
                refined = self.model(physical.image, albedo=albedo, normals=normals, depth=depth).refined
                save_image_grid(
                    Path(preview_path) / f"epoch_{epoch_number:04d}.png",
                    [image.squeeze(0), physical.image.squeeze(0), refined.squeeze(0)],
                    nrow=3,
                )

        return log
