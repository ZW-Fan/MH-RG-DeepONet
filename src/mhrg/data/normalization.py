from __future__ import annotations

import numpy as np


def compute_sensor_stats(u0_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    re = u0_arr.real.astype(np.float32)
    im = u0_arr.imag.astype(np.float32)
    mu = np.array([float(re.mean()), float(im.mean())], dtype=np.float32)
    std = np.array([float(re.std() + 1e-8), float(im.std() + 1e-8)], dtype=np.float32)
    return mu, std


def compute_output_stats(sol_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    re = sol_arr.real.astype(np.float32)
    im = sol_arr.imag.astype(np.float32)
    mu = np.array([float(re.mean()), float(im.mean())], dtype=np.float32)
    std = np.array([float(re.std() + 1e-8), float(im.std() + 1e-8)], dtype=np.float32)
    return mu, std


def normalize_sensor_global(psi0: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    re = (psi0.real.astype(np.float32) - mu[0]) / std[0]
    im = (psi0.imag.astype(np.float32) - mu[1]) / std[1]
    return np.concatenate([re, im], axis=0).astype(np.float32)


def normalize_output_pair(y_complex: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    y_re = (y_complex.real.astype(np.float32) - mu[0]) / std[0]
    y_im = (y_complex.imag.astype(np.float32) - mu[1]) / std[1]
    return np.stack([y_re, y_im], axis=-1).astype(np.float32)


def denormalize_output_pair(y_pred_2ch: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    re = y_pred_2ch[..., 0] * std[0] + mu[0]
    im = y_pred_2ch[..., 1] * std[1] + mu[1]
    return re + 1j * im
