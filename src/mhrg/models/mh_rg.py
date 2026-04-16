from __future__ import annotations

import torch
import torch.nn as nn

from mhrg.models.blocks import LowRankGate, MLP, PreBranchGate


class MHRGDeepONet(nn.Module):
    def __init__(self, sensor_in: int, trunk_in: int, hid: int, p: int, out_dim: int, feat_dim: int = 6, R: int = 6, r_gate: int = 8, hid_gate: int = 32, preB_hid: int = 16, alpha_pre: float = 1.0, alpha_b: float = 0.5, alpha_t: float = 0.5):
        super().__init__()
        self.p = p
        self.R = R
        self.branch = MLP(sensor_in, hid, p)
        self.trunk = MLP(trunk_in, hid, p)
        self.preB = PreBranchGate(feat_dim, sensor_in, alpha=alpha_pre, hid=preB_hid)
        shared_U_B = nn.Linear(r_gate, p, bias=False)
        shared_U_T = nn.Linear(r_gate, p, bias=False)
        self.gb = nn.ModuleList([LowRankGate(feat_dim, p, r_g=r_gate, hid=hid_gate, alpha=alpha_b, shared_U=shared_U_B) for _ in range(R)])
        self.gt = nn.ModuleList([LowRankGate(feat_dim, p, r_g=r_gate, hid=hid_gate, alpha=alpha_t, shared_U=shared_U_T) for _ in range(R)])
        self.W = nn.ModuleList([nn.Linear(p, out_dim) for _ in range(R)])

    def forward(self, sensor: torch.Tensor, trunk: torch.Tensor, phys: torch.Tensor, return_aux: bool = False, detach_aux: bool = True):
        b, m, _ = trunk.shape
        preB_gate = self.preB(phys)
        sensor_g = sensor * preB_gate
        branch_out = self.branch(sensor_g)
        trunk_out = self.trunk(trunk.reshape(b * m, -1)).view(b, m, self.p)
        y = 0.0
        aux = None
        if return_aux:
            aux = {"preB_gate": preB_gate, "gB_list": [], "gT_list": [], "head_outputs": []}
        for r in range(self.R):
            gB = self.gb[r](phys)
            gT = self.gt[r](phys)
            yr = self.W[r]((branch_out * gB).unsqueeze(1) * (trunk_out * gT.unsqueeze(1)))
            y = y + yr
            if aux is not None:
                aux["gB_list"].append(gB)
                aux["gT_list"].append(gT)
                aux["head_outputs"].append(yr)
        if aux is not None and detach_aux:
            detached = {}
            for k, v in aux.items():
                detached[k] = [vv.detach().cpu() for vv in v] if isinstance(v, list) else v.detach().cpu()
            return y, detached
        return (y, aux) if aux is not None else y
