from __future__ import annotations

import torch
import torch.nn as nn

from mhrg.models.blocks import BranchMLPFiLM, MLP, MulGate1pAlphaTanh, PreBranchGate


class VanillaDeepONet(nn.Module):
    def __init__(self, sensor_in: int, trunk_in: int, hid: int, p: int, out_dim: int):
        super().__init__()
        self.p = p
        self.branch = MLP(sensor_in, hid, p)
        self.trunk = MLP(trunk_in, hid, p)
        self.head = nn.Linear(p, out_dim)

    def forward(self, sensor: torch.Tensor, trunk: torch.Tensor, phys: torch.Tensor | None = None) -> torch.Tensor:
        b, m, _ = trunk.shape
        branch_out = self.branch(sensor)
        trunk_out = self.trunk(trunk.reshape(b * m, -1)).view(b, m, self.p)
        return self.head(branch_out.unsqueeze(1) * trunk_out)


class ConcatDeepONet(nn.Module):
    def __init__(self, sensor_in: int, trunk_in: int, hid: int, p: int, out_dim: int, feat_dim: int):
        super().__init__()
        self.p = p
        self.branch = MLP(sensor_in + feat_dim, hid, p)
        self.trunk = MLP(trunk_in, hid, p)
        self.head = nn.Linear(p, out_dim)

    def forward(self, sensor: torch.Tensor, trunk: torch.Tensor, phys: torch.Tensor) -> torch.Tensor:
        b, m, _ = trunk.shape
        branch_in = torch.cat([sensor, phys], dim=-1)
        branch_out = self.branch(branch_in)
        trunk_out = self.trunk(trunk.reshape(b * m, -1)).view(b, m, self.p)
        return self.head(branch_out.unsqueeze(1) * trunk_out)


class FiLMDeepONet(nn.Module):
    def __init__(self, sensor_in: int, trunk_in: int, hid: int, p: int, out_dim: int, feat_dim: int):
        super().__init__()
        self.p = p
        self.branch = BranchMLPFiLM(sensor_in, hid, p, cond_dim=feat_dim)
        self.trunk = MLP(trunk_in, hid, p)
        self.head = nn.Linear(p, out_dim)

    def forward(self, sensor: torch.Tensor, trunk: torch.Tensor, phys: torch.Tensor) -> torch.Tensor:
        b, m, _ = trunk.shape
        branch_out = self.branch(sensor, phys)
        trunk_out = self.trunk(trunk.reshape(b * m, -1)).view(b, m, self.p)
        return self.head(branch_out.unsqueeze(1) * trunk_out)


class RGDeepONet(nn.Module):
    def __init__(self, sensor_in: int, trunk_in: int, hid: int, p: int, out_dim: int, feat_dim: int = 6, alpha_gate: float = 0.5, alpha_pre: float = 1.0, preB_hid: int = 16):
        super().__init__()
        self.p = p
        self.branch = MLP(sensor_in, hid, p)
        self.trunk = MLP(trunk_in, hid, p)
        self.head = nn.Linear(p, out_dim)
        self.gB = MulGate1pAlphaTanh(feat_dim, p, alpha_gate)
        self.gT = MulGate1pAlphaTanh(feat_dim, p, alpha_gate)
        self.preB = PreBranchGate(feat_dim, sensor_in, alpha=alpha_pre, hid=preB_hid)

    def forward(self, sensor: torch.Tensor, trunk: torch.Tensor, phys: torch.Tensor) -> torch.Tensor:
        b, m, _ = trunk.shape
        sensor_g = sensor * self.preB(phys)
        branch_out = self.branch(sensor_g) * self.gB(phys)
        trunk_out = self.trunk(trunk.reshape(b * m, -1)).view(b, m, self.p)
        trunk_out = trunk_out * self.gT(phys).unsqueeze(1)
        return self.head(branch_out.unsqueeze(1) * trunk_out)
