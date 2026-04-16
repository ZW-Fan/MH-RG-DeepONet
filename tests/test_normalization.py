import numpy as np

from mhrg.data.normalization import denormalize_output_pair, normalize_output_pair


def test_output_normalize_denormalize_roundtrip():
    z = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex64)
    mu = np.array([0.1, -0.2], dtype=np.float32)
    std = np.array([1.5, 2.0], dtype=np.float32)
    zn = normalize_output_pair(z, mu, std)
    zrec = denormalize_output_pair(zn, mu, std)
    assert np.allclose(z, zrec, atol=1e-6)
