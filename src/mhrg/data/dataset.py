from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from mhrg.data.features import features6
from mhrg.data.normalization import normalize_output_pair, normalize_sensor_global


class MultiTimeDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        t: np.ndarray,
        u0: np.ndarray,
        sol: np.ndarray,
        points: int,
        feat_mu: np.ndarray,
        feat_std: np.ndarray,
        sensor_mu_ch: np.ndarray,
        sensor_std_ch: np.ndarray,
        y_mu_ch: np.ndarray,
        y_std_ch: np.ndarray,
    ) -> None:
        self.x = x.astype(np.float32)
        self.t = t.astype(np.float32)
        self.u0 = u0
        self.sol = sol
        self.nx = len(x)
        self.nt = len(t)
        self.points = points

        raw_feats = np.stack([features6(x, u) for u in u0]).astype(np.float32)
        self.phys = (raw_feats - feat_mu[None, :]) / (feat_std[None, :] + 1e-8)

        self.sensor_mu_ch = sensor_mu_ch.astype(np.float32)
        self.sensor_std_ch = sensor_std_ch.astype(np.float32)
        self.y_mu_ch = y_mu_ch.astype(np.float32)
        self.y_std_ch = y_std_ch.astype(np.float32)

    def __len__(self) -> int:
        return len(self.u0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        psi0 = self.u0[idx]
        sensor = normalize_sensor_global(psi0, self.sensor_mu_ch, self.sensor_std_ch)

        ti = np.random.randint(0, self.nt, self.points)
        xi = np.random.randint(0, self.nx, self.points)
        trunk = np.stack([self.x[xi], self.t[ti]], axis=1).astype(np.float32)

        y_complex = self.sol[idx][ti, xi]
        y_norm = normalize_output_pair(y_complex, self.y_mu_ch, self.y_std_ch)

        return {
            "sensor": torch.tensor(sensor, dtype=torch.float32),
            "phys": torch.tensor(self.phys[idx], dtype=torch.float32),
            "trunk": torch.tensor(trunk, dtype=torch.float32),
            "sol": torch.tensor(y_norm, dtype=torch.float32),
        }
