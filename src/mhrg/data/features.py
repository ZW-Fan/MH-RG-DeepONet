from __future__ import annotations

import numpy as np


def features6(x: np.ndarray, psi0: np.ndarray) -> np.ndarray:
    """Return [E, Amax, xc, var, P, bw] for a 1D complex field."""
    nx = len(x)
    dx = float(x[1] - x[0])
    abs2 = np.abs(psi0) ** 2

    energy = float(np.sum(abs2) * dx)
    amplitude = float(np.max(np.abs(psi0)))
    xc = float(np.sum(x * abs2) * dx / (energy + 1e-12))
    var = float(np.sum(((x - xc) ** 2) * abs2) * dx / (energy + 1e-12))

    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    dpsi = np.fft.ifft(1j * kx * np.fft.fft(psi0))
    momentum = float(np.sum(np.imag(np.conj(psi0) * dpsi)) * dx)

    psi_k = np.fft.fft(psi0)
    w = np.abs(psi_k) ** 2
    ws = float(np.sum(w) + 1e-12)
    k_mean = float(np.sum(kx * w) / ws)
    bandwidth = float(np.sqrt(max(np.sum((kx - k_mean) ** 2 * w) / ws, 0.0)))

    return np.array([energy, amplitude, xc, var, momentum, bandwidth], dtype=np.float32)
