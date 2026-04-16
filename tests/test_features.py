import numpy as np

from mhrg.data.features import features6


def test_features6_shape_and_finite():
    x = np.linspace(-10, 10, 128, dtype=np.float32)
    psi0 = np.exp(-x**2).astype(np.complex64)
    feat = features6(x, psi0)
    assert feat.shape == (6,)
    assert np.all(np.isfinite(feat))
