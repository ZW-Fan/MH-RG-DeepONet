from __future__ import annotations

from pathlib import Path
import torch

from mhrg.utils.io import ensure_dir


def get_ckpt_path(root: str | Path, run_seed: int, model_name: str) -> Path:
    seed_dir = ensure_dir(Path(root) / f"seed_{run_seed}")
    return seed_dir / f"{model_name}.pt"


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, train_time_sec: float, log_tr: list[float], log_te: list[float], normalization: dict) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_time_sec": train_time_sec,
        "log_tr": log_tr,
        "log_te": log_te,
        "normalization": normalization,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model, optimizer=None, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
