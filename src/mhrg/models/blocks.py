from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hid: int, d_out: int, layers: int = 3, act=nn.GELU):
        super().__init__()
        mods = []
        d = d_in
        for _ in range(layers):
            mods += [nn.Linear(d, d_hid), act()]
            d = d_hid
        mods += [nn.Linear(d_hid, d_out)]
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MulGate1pAlphaTanh(nn.Module):
    def __init__(self, feat_dim: int, p: int, alpha: float = 0.5, hid: int | None = None):
        super().__init__()
        if hid is None:
            hid = max(64, p)
        self.alpha = float(alpha)
        self.net = nn.Sequential(nn.Linear(feat_dim, hid), nn.GELU(), nn.Linear(hid, p))

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return 1.0 + self.alpha * torch.tanh(self.net(phi))


class PreBranchGate(nn.Module):
    def __init__(self, feat_dim: int, sensor_dim: int, alpha: float = 1.0, hid: int = 16):
        super().__init__()
        self.alpha = float(alpha)
        self.net = nn.Sequential(nn.Linear(feat_dim, hid), nn.GELU(), nn.Linear(hid, sensor_dim))

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return 1.0 + self.alpha * torch.tanh(self.net(phi))


class BranchMLPFiLM(nn.Module):
    def __init__(self, d_in: int, d_hid: int, d_out: int, cond_dim: int, layers: int = 3, act=nn.GELU):
        super().__init__()
        self.act = act()
        self.linears = nn.ModuleList()
        self.film_generators = nn.ModuleList()
        d = d_in
        for _ in range(layers):
            self.linears.append(nn.Linear(d, d_hid))
            self.film_generators.append(nn.Sequential(nn.Linear(cond_dim, 128), nn.GELU(), nn.Linear(128, 2 * d_hid)))
            d = d_hid
        self.out_linear = nn.Linear(d_hid, d_out)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = x
        for linear, film_gen in zip(self.linears, self.film_generators):
            h = linear(h)
            gamma_beta = film_gen(cond)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            h = self.act(gamma * h + beta)
        return self.out_linear(h)


class LowRankGate(nn.Module):
    def __init__(self, feat_dim: int, p: int, r_g: int = 8, hid: int = 32, alpha: float = 0.5, shared_U: nn.Linear | None = None):
        super().__init__()
        self.alpha = float(alpha)
        self.mlp = nn.Sequential(nn.Linear(feat_dim, hid), nn.GELU(), nn.Linear(hid, r_g))
        self.U = nn.Linear(r_g, p, bias=False) if shared_U is None else shared_U

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        g = self.U(self.mlp(phi))
        return 1.0 + self.alpha * torch.tanh(g)
