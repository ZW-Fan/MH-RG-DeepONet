from __future__ import annotations

from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader

from mhrg.data.dataset import MultiTimeDataset
from mhrg.data.features import features6
from mhrg.data.normalization import compute_output_stats, compute_sensor_stats


def build_nlse_grid(nx: int, x_min: float, x_max: float, dt: float, t_max: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(x_min, x_max, nx, dtype=np.float32)
    t = np.arange(0.0, t_max + 1e-9, dt, dtype=np.float32)
    return x, t


def load_nlse_arrays(data_root: str | Path, cfg: dict) -> dict[str, np.ndarray]:
    root = Path(data_root)
    return {
        "u0_train": np.load(root / cfg["u0_train"]),
        "sol_train": np.load(root / cfg["sol_train"]),
        "u0_test": np.load(root / cfg["u0_test"]),
        "sol_test": np.load(root / cfg["sol_test"]),
    }


def build_dataloaders(config: dict) -> tuple[dict, DataLoader, DataLoader]:
    x, t = build_nlse_grid(
        nx=config["nlse"]["nx"],
        x_min=config["nlse"]["x_min"],
        x_max=config["nlse"]["x_max"],
        dt=config["nlse"]["dt"],
        t_max=config["nlse"]["t_max"],
    )
    arrays = load_nlse_arrays(config["data"]["root"], config["data"])

    sensor_mu_ch, sensor_std_ch = compute_sensor_stats(arrays["u0_train"])
    y_mu_ch, y_std_ch = compute_output_stats(arrays["sol_train"])
    feats_train = np.stack([features6(x, u) for u in arrays["u0_train"]]).astype(np.float32)
    feat_mu = feats_train.mean(axis=0).astype(np.float32)
    feat_std = (feats_train.std(axis=0) + 1e-8).astype(np.float32)

    tr_ds = MultiTimeDataset(
        x=x, t=t,
        u0=arrays["u0_train"], sol=arrays["sol_train"],
        points=config["training"]["points_per_sample"],
        feat_mu=feat_mu, feat_std=feat_std,
        sensor_mu_ch=sensor_mu_ch, sensor_std_ch=sensor_std_ch,
        y_mu_ch=y_mu_ch, y_std_ch=y_std_ch,
    )
    te_ds = MultiTimeDataset(
        x=x, t=t,
        u0=arrays["u0_test"], sol=arrays["sol_test"],
        points=config["training"]["points_per_sample"],
        feat_mu=feat_mu, feat_std=feat_std,
        sensor_mu_ch=sensor_mu_ch, sensor_std_ch=sensor_std_ch,
        y_mu_ch=y_mu_ch, y_std_ch=y_std_ch,
    )

    tl_tr = DataLoader(tr_ds, batch_size=config["training"]["batch_size"], shuffle=True, pin_memory=True, num_workers=0)
    tl_te = DataLoader(te_ds, batch_size=config["training"]["batch_size"], shuffle=False, pin_memory=True, num_workers=0)

    stats = {
        "x": x,
        "t": t,
        "sensor_mu_ch": sensor_mu_ch,
        "sensor_std_ch": sensor_std_ch,
        "y_mu_ch": y_mu_ch,
        "y_std_ch": y_std_ch,
        "feat_mu": feat_mu,
        "feat_std": feat_std,
        **arrays,
    }
    return stats, tl_tr, tl_te
