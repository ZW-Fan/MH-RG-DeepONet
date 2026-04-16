from __future__ import annotations

import numpy as np
import torch

from mhrg.data.features import features6
from mhrg.data.normalization import denormalize_output_pair, normalize_sensor_global
from mhrg.utils.metrics import fullfield_mse


@torch.no_grad()
def predict_full_field(model, u0, x, t, feat_mu, feat_std, sensor_mu_ch, sensor_std_ch, y_mu_ch, y_std_ch):
    dev = next(model.parameters()).device
    nx = len(x)
    nt = len(t)
    sensor = normalize_sensor_global(u0, sensor_mu_ch, sensor_std_ch)
    sensor = torch.tensor(sensor, dtype=torch.float32, device=dev).unsqueeze(0)
    phi = features6(x, u0).astype(np.float32)
    phi = (phi - feat_mu) / (feat_std + 1e-8)
    phi = torch.tensor(phi, dtype=torch.float32, device=dev).unsqueeze(0)
    X, T = np.meshgrid(x, t)
    trunk = np.stack([X, T], axis=-1).astype(np.float32).reshape(1, -1, 2)
    trunk = torch.tensor(trunk, dtype=torch.float32, device=dev)
    out_norm = model(sensor, trunk, phi).view(nt, nx, 2).detach().cpu().numpy()
    return denormalize_output_pair(out_norm, y_mu_ch, y_std_ch)


@torch.no_grad()
def full_mse_for(model, u0_test, sol_test, x, t, feat_mu, feat_std, sensor_mu_ch, sensor_std_ch, y_mu_ch, y_std_ch):
    errs = []
    for u0, gt in zip(u0_test, sol_test):
        pred = predict_full_field(model, u0, x, t, feat_mu, feat_std, sensor_mu_ch, sensor_std_ch, y_mu_ch, y_std_ch)
        errs.append(fullfield_mse(pred, gt))
    return np.array(errs, dtype=np.float32)
