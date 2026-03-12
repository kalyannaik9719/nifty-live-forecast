from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


SEQ_LEN = 32
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


class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features: list[str], target_col: str, seq_len: int = SEQ_LEN):
        self.features = features
        self.target_col = target_col
        self.seq_len = seq_len

        clean = df.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target_col]).reset_index(drop=True)
        self.df = clean

        self.x = self.df[features].to_numpy(dtype=np.float32)
        self.y = self.df[target_col].to_numpy(dtype=np.float32)

    def __len__(self):
        return max(0, len(self.df) - self.seq_len)

    def __getitem__(self, idx):
        x_seq = self.x[idx : idx + self.seq_len]
        y_next = self.y[idx + self.seq_len]
        return torch.tensor(x_seq), torch.tensor(y_next)


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


@dataclass
class LSTMArtifact:
    model_path: Path
    meta_path: Path


def ensure_regime_code(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "regime_code" not in out.columns:
        mapping = {"bull": 0.5, "bear": -0.5, "sideways": 0.0, "high_vol": 1.0}
        out["regime_code"] = out.get("regime", pd.Series(["sideways"] * len(out))).map(mapping).fillna(0.0)
    return out


def train_lstm(
    df: pd.DataFrame,
    artifact_dir: Path,
    features: list[str] | None = None,
    target_col: str = "target_return",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> LSTMArtifact:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    features = features or DEFAULT_FEATURES

    df = ensure_regime_code(df).copy()

    dataset = SequenceDataset(df, features=features, target_col=target_col, seq_len=SEQ_LEN)
    if len(dataset) < 100:
        raise ValueError("Not enough rows to train LSTM.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = LSTMForecaster(input_size=len(features)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # standardize features using full training set
    x_arr = dataset.x
    x_mean = x_arr.mean(axis=0)
    x_std = x_arr.std(axis=0) + 1e-6

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = (xb - torch.tensor(x_mean)) / torch.tensor(x_std)
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

    model_path = artifact_dir / "lstm_forecaster.pt"
    meta_path = artifact_dir / "lstm_forecaster_meta.json"

    torch.save(model.state_dict(), model_path)
    meta_path.write_text(
        json.dumps(
            {
                "features": features,
                "seq_len": SEQ_LEN,
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
            },
            indent=2,
        )
    )

    return LSTMArtifact(model_path=model_path, meta_path=meta_path)


def load_lstm(artifact_dir: Path, device: str = "cpu"):
    model_path = artifact_dir / "lstm_forecaster.pt"
    meta_path = artifact_dir / "lstm_forecaster_meta.json"

    meta = json.loads(meta_path.read_text())
    model = LSTMForecaster(input_size=len(meta["features"]))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, meta


@torch.no_grad()
def infer_lstm(latest_df: pd.DataFrame, artifact_dir: Path, device: str = "cpu") -> dict:
    model, meta = load_lstm(artifact_dir, device=device)
    df = ensure_regime_code(latest_df).copy()

    features = meta["features"]
    seq_len = meta["seq_len"]
    x_mean = np.array(meta["x_mean"], dtype=np.float32)
    x_std = np.array(meta["x_std"], dtype=np.float32)

    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features).reset_index(drop=True)
    if len(clean) < seq_len:
        raise ValueError(f"LSTM inference needs at least {seq_len} rows.")

    x = clean[features].tail(seq_len).to_numpy(dtype=np.float32)
    x = (x - x_mean) / x_std
    x = torch.tensor(x).unsqueeze(0)

    pred_return = float(model(x).cpu().item())
    direction = "LONG" if pred_return > 0 else "SHORT"
    confidence = min(1.0, abs(pred_return) * 500)  # heuristic

    return {
        "lstm_pred_return": pred_return,
        "lstm_direction": direction,
        "lstm_confidence": confidence,
    }