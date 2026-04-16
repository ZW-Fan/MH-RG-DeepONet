# MH-RG DeepONet

Reference implementation for the **Multi-Head Residual-Gated DeepONet for Coherent Nonlinear Dynamics**. 

The manuscript is available: [arXiv:2604.11972](https://arxiv.org/abs/2604.11972)

The dataset can be downloaded from [Dataset](https://doi.org/10.5281/zenodo.19287715)

This repository contains a clean, modular PyTorch codebase for:

- training DeepONet-family baselines on the 1D focusing NLSE collision dataset
- evaluating full-field test error across random seeds
- reproducing noise-robustness experiments
- saving checkpoints, logs, and aggregated metrics in a reproducible layout

## Implemented models

- Vanilla DeepONet
- Concat DeepONet
- FiLM DeepONet
- RG DeepONet
- MH-RG DeepONet with configurable head counts

## Repository layout

```text
mh-rg-deeponet/
├── configs/                # YAML experiment configs
├── data/                   # place dataset files here
├── scripts/                # CLI entry points
├── src/mhrg/               # package source code
├── tests/                  # lightweight unit tests
└── outputs/                # metrics, logs, and figures
```

## Dataset

Place the processed NLSE dataset files under `data/`:

- `u0_train.npy`
- `sol_train.npy`
- `u0_test.npy`
- `sol_test.npy`

Expected shapes:

- `u0_train`: `(N_train, Nx)` complex array
- `sol_train`: `(N_train, Nt, Nx)` complex array
- `u0_test`: `(N_test, Nx)` complex array
- `sol_test`: `(N_test, Nt, Nx)` complex array

## Installation

### Option 1: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### Option 2: conda

```bash
conda create -n mhrg python=3.11 -y
conda activate mhrg
pip install -r requirements.txt
pip install -e .
```

## Quick start

### Train

```bash
python scripts/train_nlse.py --config configs/nlse_train.yaml
```

### Evaluate saved checkpoints

```bash
python scripts/evaluate_nlse.py --config configs/nlse_eval.yaml
```

### Run noise robustness study

```bash
python scripts/noise_study.py --config configs/nlse_noise.yaml
```

## Reproducibility

- train-set statistics only are used for global normalization
- all experiment seeds are read from config files
- checkpoints are saved per seed and per model
- aggregated results are written to `outputs/metrics/`

## Notes

This first public release is intentionally focused on the NLSE collision benchmark only. It is designed as a paper companion repository.
