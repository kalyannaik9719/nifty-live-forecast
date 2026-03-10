from __future__ import annotations

import json
import joblib
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO

from rl.trading_env import NiftyTradingEnv

LIVE_FEATURE_PATH = Path("data/processed/live_features.parquet")
BASELINE_MODEL_PATH = Path("data/processed/baseline_model.pkl")
RL_MODEL_PATH = Path("data/processed/ppo_nifty")
OUT_PATH = Path("data/processed/live_signal.json")

BASELINE_FEATURES = [
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
    "minute"
]


def run():
    live_df = pd.read_parquet(LIVE_FEATURE_PATH).copy()

    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    prob_up = float(baseline_model.predict_proba(live_df[BASELINE_FEATURES])[0, 1])

    if prob_up > 0.55:
        baseline_signal = "LONG"
    elif prob_up < 0.45:
        baseline_signal = "SHORT"
    else:
        baseline_signal = "HOLD"

    rl_df = pd.concat([live_df] * 3, ignore_index=True).copy()
    model = PPO.load(str(RL_MODEL_PATH))
    env = NiftyTradingEnv(rl_df)

    obs, info = env.reset()
    action, _ = model.predict(obs, deterministic=True)

    action_map = {0: "SHORT", 1: "HOLD", 2: "LONG"}
    rl_signal = action_map[int(action)]

    signal = {
        "ts": str(live_df["ts"].iloc[-1]),
        "regime": str(live_df["regime"].iloc[-1]),
        "last_close": float(live_df["close"].iloc[-1]),
        "baseline_signal": baseline_signal,
        "baseline_prob_up": round(prob_up, 4),
        "rl_signal": rl_signal,
    }

    OUT_PATH.write_text(json.dumps(signal, indent=2))
    print("Saved live signal →", OUT_PATH)
    print(json.dumps(signal, indent=2))


if __name__ == "__main__":
    run()