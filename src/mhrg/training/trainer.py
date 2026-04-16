from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mhrg.models.deeponet import ConcatDeepONet, FiLMDeepONet, RGDeepONet, VanillaDeepONet
from mhrg.models.mh_rg import MHRGDeepONet
from mhrg.training.checkpoint import get_ckpt_path, load_checkpoint, save_checkpoint
from mhrg.training.eval import full_mse_for
from mhrg.utils.seed import seed_all


def build_models(config: dict, sensor_in: int, trunk_in: int):
    mcfg = config["model"]
    h_base = mcfg["hidden_width"]
    p_base = mcfg["latent_width"]
    p_mhrg = mcfg["mh_rg_latent_width"]
    out_dim = mcfg["out_dim"]
    feat_dim = mcfg["feat_dim"]

    models = {
        "vanilla": VanillaDeepONet(sensor_in, trunk_in, h_base, p_base, out_dim),
        "concat": ConcatDeepONet(sensor_in, trunk_in, h_base, p_base, out_dim, feat_dim),
        "film": FiLMDeepONet(sensor_in, trunk_in, h_base, p_base, out_dim, feat_dim),
        "rg": RGDeepONet(sensor_in, trunk_in, h_base, p_base, out_dim, feat_dim=feat_dim, alpha_gate=mcfg["alpha_b"], alpha_pre=mcfg["alpha_pre"], preB_hid=mcfg["preb_mh"]),
    }
    for R in mcfg["mh_rg_heads"]:
        models[f"mhrg_R{R}"] = MHRGDeepONet(sensor_in, trunk_in, h_base, p_mhrg, out_dim, feat_dim=feat_dim, R=R, r_gate=mcfg["rgate_mh"], hid_gate=mcfg["hidg_mh"], preB_hid=mcfg["preb_mh"], alpha_pre=mcfg["alpha_pre"], alpha_b=mcfg["alpha_b"], alpha_t=mcfg["alpha_t"])
    return models


def run_epoch(model, loader, opt=None, grad_clip: float = 1.0):
    train = opt is not None
    model.train() if train else model.eval()
    mse = nn.MSELoss()
    total = 0.0
    n = 0
    dev = next(model.parameters()).device
    for batch in loader:
        s = batch["sensor"].to(dev, non_blocking=True)
        tr = batch["trunk"].to(dev, non_blocking=True)
        p = batch["phys"].to(dev, non_blocking=True)
        y = batch["sol"].to(dev, non_blocking=True)
        with torch.set_grad_enabled(train):
            out = model(s, tr, p)
            loss = mse(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
        total += float(loss.item()) * s.size(0)
        n += s.size(0)
    return total / max(1, n)


def train_all_seeds(config: dict, stats: dict, train_loader, test_loader, keep_models: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_seeds = config["runtime"]["run_seeds"]
    sensor_in = 2 * len(stats["x"])
    trunk_in = 2
    grad_clip = float(config["training"].get("grad_clip", 1.0))

    order = ["vanilla", "concat", "film", "rg"] + [f"mhrg_R{R}" for R in config["model"]["mh_rg_heads"]]
    all_runs = []
    for run_seed in run_seeds:
        seed_all(run_seed)
        models = build_models(config, sensor_in=sensor_in, trunk_in=trunk_in)
        for m in models.values():
            m.to(device)
        opts = {n: optim.Adam(models[n].parameters(), lr=config["training"]["lr"]) for n in order}
        log = {f"{n}_{k}": [] for n in order for k in ("tr", "te")}
        train_time = {n: 0.0 for n in order}
        checkpoint_root = config["experiment"]["checkpoint_root"]
        all_exist = all(get_ckpt_path(checkpoint_root, run_seed, n).exists() for n in order)
        if all_exist and not config["training"].get("force_retrain", False):
            for n in order:
                payload = load_checkpoint(get_ckpt_path(checkpoint_root, run_seed, n), models[n], optimizer=opts[n], map_location=device)
                train_time[n] = float(payload.get("train_time_sec", 0.0))
                log[f"{n}_tr"] = payload.get("log_tr", [])
                log[f"{n}_te"] = payload.get("log_te", [])
        else:
            for ep in range(1, config["training"]["epochs"] + 1):
                for n in order:
                    seed_all(run_seed + 1000 * ep)
                    t0 = time.time()
                    tr_loss = run_epoch(models[n], train_loader, opt=opts[n], grad_clip=grad_clip)
                    train_time[n] += time.time() - t0
                    log[f"{n}_tr"].append(tr_loss)
                for n in order:
                    seed_all(run_seed + 500000 + 1000 * ep)
                    te_loss = run_epoch(models[n], test_loader, opt=None, grad_clip=grad_clip)
                    log[f"{n}_te"].append(te_loss)
            normalization = {k: stats[k] for k in ["sensor_mu_ch", "sensor_std_ch", "y_mu_ch", "y_std_ch", "feat_mu", "feat_std"]}
            for n in order:
                save_checkpoint(get_ckpt_path(checkpoint_root, run_seed, n), models[n], opts[n], config["training"]["epochs"], train_time[n], log[f"{n}_tr"], log[f"{n}_te"], normalization)

        summary = {}
        for n in order:
            errs = full_mse_for(models[n], stats["u0_test"], stats["sol_test"], stats["x"], stats["t"], stats["feat_mu"], stats["feat_std"], stats["sensor_mu_ch"], stats["sensor_std_ch"], stats["y_mu_ch"], stats["y_std_ch"])
            summary[n] = {
                "mean": float(errs.mean()),
                "std_within_testset": float(errs.std(ddof=1)),
                "all_errs": errs.copy(),
                "train_time_sec": train_time[n],
                "last_te_loss": float(log[f"{n}_te"][-1]) if log[f"{n}_te"] else None,
            }
        out = {"run_seed": run_seed, "summary": summary, "log": log}
        if keep_models:
            out["models"] = {n: models[n].to("cpu") for n in order}
        else:
            for m in models.values():
                m.to("cpu")
        all_runs.append(out)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return order, all_runs


def aggregate_runs(order: list[str], all_runs: list[dict]) -> dict:
    agg = {}
    for n in order:
        per_seed_means = [run["summary"][n]["mean"] for run in all_runs]
        per_seed_stds = [run["summary"][n]["std_within_testset"] for run in all_runs]
        pooled_errs = np.concatenate([run["summary"][n]["all_errs"] for run in all_runs], axis=0)
        agg[n] = {
            "mean_of_seed_means": float(np.mean(per_seed_means)),
            "std_of_seed_means": float(np.std(per_seed_means, ddof=1)) if len(per_seed_means) > 1 else 0.0,
            "mean_within_testset_std": float(np.mean(per_seed_stds)),
            "pooled_mean": float(np.mean(pooled_errs)),
            "pooled_std": float(np.std(pooled_errs, ddof=1)),
            "per_seed_means": per_seed_means,
        }
    return agg
