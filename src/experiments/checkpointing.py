"""Checkpoint and metric logging utilities for experiment runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


@dataclass(slots=True)
class CheckpointRecord:
    """Metadata associated with a saved checkpoint."""

    epoch: int
    path: Path
    created_at: str
    metrics: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "path": str(self.path),
            "created_at": self.created_at,
            "metrics": self.metrics,
            "extra": self.extra,
        }


class CheckpointManager:
    """Persist checkpoints, metrics, and run metadata for training jobs."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        checkpoint_dir: str | Path | None = None,
        log_dir: str | Path | None = None,
        keep_last: int = 3,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.checkpoint_dir = Path(checkpoint_dir or self.root_dir / "checkpoints")
        self.log_dir = Path(log_dir or self.root_dir / "logs")
        self.keep_last = keep_last
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.jsonl"
        self.manifest_path = self.root_dir / "manifest.json"

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: CheckpointManager._to_serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [CheckpointManager._to_serializable(item) for item in value]
        if hasattr(value, "item") and callable(value.item):
            try:
                return value.item()
            except Exception:
                pass
        return value

    def save_config(self, config: Any) -> Path:
        """Persist the experiment configuration alongside the run."""

        path = self.root_dir / "config.json"
        if hasattr(config, "to_dict"):
            payload = config.to_dict()
        elif isinstance(config, dict):
            payload = config
        else:
            raise TypeError("Config object must provide to_dict() or be a dictionary.")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path

    def log_metrics(self, step: int, metrics: dict[str, float], *, split: str = "train", extra: dict[str, Any] | None = None) -> Path:
        """Append a metrics record in JSONL format."""

        record = {
            "timestamp": self._timestamp(),
            "step": step,
            "split": split,
            "metrics": {key: float(value) for key, value in metrics.items()},
        }
        if extra:
            record["extra"] = self._to_serializable(extra)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return self.metrics_path

    def _update_manifest(self, record: CheckpointRecord, *, latest: bool = True, best: bool = False) -> None:
        manifest: dict[str, Any] = {}
        if self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text())
            except Exception:
                manifest = {}

        checkpoints = manifest.setdefault("checkpoints", [])
        checkpoints.append(record.to_dict())
        manifest["checkpoints"] = checkpoints[-100:]
        if latest:
            manifest["latest_checkpoint"] = str(record.path)
        if best:
            manifest["best_checkpoint"] = str(record.path)
            manifest["best_metrics"] = record.metrics
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def save_checkpoint(
        self,
        name: str,
        payload: dict[str, Any],
        *,
        epoch: int,
        metrics: dict[str, float] | None = None,
        extra: dict[str, Any] | None = None,
        best: bool = False,
    ) -> Path:
        """Save a checkpoint payload and update the run manifest."""

        path = self.checkpoint_dir / name
        torch.save(payload, path)
        record = CheckpointRecord(
            epoch=epoch,
            path=path.resolve(),
            created_at=self._timestamp(),
            metrics=metrics or {},
            extra=extra or {},
        )
        self._update_manifest(record, latest=True, best=best)
        self.prune_checkpoints()
        return path

    def save_training_state(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        metrics: dict[str, float] | None = None,
        extra: dict[str, Any] | None = None,
        best: bool = False,
    ) -> Path:
        """Convenience wrapper for standard training checkpoints."""

        payload: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        if metrics is not None:
            payload["metrics"] = metrics
        if extra is not None:
            payload["extra"] = extra
        checkpoint_name = f"epoch_{epoch:04d}.pt"
        return self.save_checkpoint(
            checkpoint_name,
            payload,
            epoch=epoch,
            metrics=metrics,
            extra=extra,
            best=best,
        )

    def prune_checkpoints(self) -> None:
        """Keep only the most recent checkpoints on disk."""

        checkpoint_files = sorted(self.checkpoint_dir.glob("epoch_*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
        for stale_checkpoint in checkpoint_files[self.keep_last :]:
            try:
                stale_checkpoint.unlink()
            except FileNotFoundError:
                continue

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load a saved checkpoint payload."""

        return torch.load(Path(path), map_location="cpu")

    def load_latest(self) -> dict[str, Any] | None:
        """Load the most recently recorded checkpoint, if present."""

        if self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text())
                latest = manifest.get("latest_checkpoint")
                if latest:
                    try:
                        return self.load_checkpoint(latest)
                    except Exception:
                        pass
            except Exception:
                pass
        latest_checkpoint = sorted(self.checkpoint_dir.glob("epoch_*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not latest_checkpoint:
            return None
        return self.load_checkpoint(latest_checkpoint[0])
