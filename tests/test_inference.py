import numpy as np
import torch

from mhrg.training.eval import predict_full_field
from mhrg.models.deeponet import VanillaDeepONet


def test_predict_full_field_shape():
    model = VanillaDeepONet(sensor_in=16, trunk_in=2, hid=32, p=8, out_dim=2)
    x = np.linspace(-1, 1, 8, dtype=np.float32)
    t = np.linspace(0, 1, 4, dtype=np.float32)
    u0 = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex64)
    feat_mu = np.zeros(6, dtype=np.float32)
    feat_std = np.ones(6, dtype=np.float32)
    sensor_mu = np.zeros(2, dtype=np.float32)
    sensor_std = np.ones(2, dtype=np.float32)
    y_mu = np.zeros(2, dtype=np.float32)
    y_std = np.ones(2, dtype=np.float32)
    pred = predict_full_field(model, u0, x, t, feat_mu, feat_std, sensor_mu, sensor_std, y_mu, y_std)
    assert pred.shape == (4, 8)
