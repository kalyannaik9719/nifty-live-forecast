from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import ClippedAdam


DEFAULT_FEATURES = [
    "ret1",
    "ret5",
    "sma5",
    "sma20",
    "ema12",
    "volatility",
    "close_sma5_gap",
    "close_sma20_gap",
    "range",
    "hour",
    "minute",
    "regime_code",
]


class BayesianMLP(PyroModule):
    def __init__(self, in_features: int, hidden: int = 32):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](in_features, hidden)
        self.fc1.weight = PyroSample(dist.Normal(0.0, 1.0).expand([hidden, in_features]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0.0, 1.0).expand([hidden]).to_event(1))

        self.out = PyroModule[nn.Linear](hidden, 1)
        self.out.weight = PyroSample(dist.Normal(0.0, 1.0).expand([1, hidden]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0.0, 1.0).expand([1]).to_event(1))

    def forward(self, x, y=None):
        x = torch.relu(self.fc1(x))
        mean = self.out(x).squeeze(-1)
        sigma = pyro.sample("sigma", dist.Uniform(0.0001, 0.05))
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean


def ensure_regime_code(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "regime_code" not in out.columns:
        mapping = {"bull": 0.5, "bear": -0.5, "sideways": 0.0, "high_vol": 1.0}
        out["regime_code"] = out.get("regime", pd.Series(["sideways"] * len(out))).map(mapping).fillna(0.0)
    return out


def train_bnn(
    df: pd.DataFrame,
    artifact_dir: Path,
    features: list[str] | None = None,
    target_col: str = "target_return",
    steps: int = 1500,
):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    features = features or DEFAULT_FEATURES

    df = ensure_regime_code(df).replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target_col]).reset_index(drop=True)

    if len(df) < 200:
        raise ValueError("Not enough rows to train BNN.")

    X = df[features].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-6
    Xs = (X - x_mean) / x_std

    X_t = torch.tensor(Xs)
    y_t = torch.tensor(y)

    pyro.clear_param_store()
    model = BayesianMLP(in_features=len(features))
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, ClippedAdam({"lr": 0.01}), loss=Trace_ELBO())

    for _ in range(steps):
        svi.step(X_t, y_t)

    pyro.get_param_store().save(str(artifact_dir / "bnn_param_store.pt"))
    (artifact_dir / "bnn_meta.json").write_text(
        json.dumps(
            {"features": features, "x_mean": x_mean.tolist(), "x_std": x_std.tolist()},
            indent=2,
        )
    )
    print("Saved BNN artifacts in", artifact_dir)


def infer_bnn(latest_df: pd.DataFrame, artifact_dir: Path, samples: int = 200) -> dict:
    meta = json.loads((artifact_dir / "bnn_meta.json").read_text())
    features = meta["features"]
    x_mean = np.array(meta["x_mean"], dtype=np.float32)
    x_std = np.array(meta["x_std"], dtype=np.float32)

    df = ensure_regime_code(latest_df).replace([np.inf, -np.inf], np.nan).dropna(subset=features).reset_index(drop=True)
    if len(df) < 1:
        raise ValueError("No usable rows for BNN inference.")

    X = df[features].tail(1).to_numpy(dtype=np.float32)
    Xs = (X - x_mean) / x_std
    X_t = torch.tensor(Xs)

    pyro.clear_param_store()

    # PyTorch 2.6+ safe-load compatibility for trusted local checkpoints
    torch.serialization.add_safe_globals([constraints._Real])
    pyro.get_param_store().load(str(artifact_dir / "bnn_param_store.pt"))

    model = BayesianMLP(in_features=len(features))
    guide = AutoDiagonalNormal(model)
    predictive = Predictive(model, guide=guide, num_samples=samples, return_sites=("_RETURN",))

    out = predictive(X_t)
    means = out["_RETURN"].detach().cpu().numpy().reshape(-1)

    mean_pred = float(means.mean())
    std_pred = float(means.std())
    confidence = float(max(0.0, min(1.0, 1.0 - std_pred * 50)))

    return {
        "bnn_mean_return": mean_pred,
        "bnn_std_return": std_pred,
        "bnn_confidence": confidence,
        "bnn_direction": "LONG" if mean_pred > 0 else "SHORT",
    }