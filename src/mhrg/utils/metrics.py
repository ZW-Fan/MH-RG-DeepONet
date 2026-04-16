from __future__ import annotations

import numpy as np


def fullfield_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    d = pred - gt
    return float(np.mean(d.real ** 2 + d.imag ** 2))
