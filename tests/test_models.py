import torch

from mhrg.models.deeponet import VanillaDeepONet, RGDeepONet
from mhrg.models.mh_rg import MHRGDeepONet


def test_model_forward_shapes():
    b, m = 2, 5
    sensor = torch.randn(b, 256)
    trunk = torch.randn(b, m, 2)
    phys = torch.randn(b, 6)

    vanilla = VanillaDeepONet(256, 2, 64, 32, 2)
    rg = RGDeepONet(256, 2, 64, 32, 2)
    mhrg = MHRGDeepONet(256, 2, 64, 32, 2, R=4)

    assert vanilla(sensor, trunk, phys).shape == (b, m, 2)
    assert rg(sensor, trunk, phys).shape == (b, m, 2)
    assert mhrg(sensor, trunk, phys).shape == (b, m, 2)
