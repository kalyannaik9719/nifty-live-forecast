from pathlib import Path
import pandas as pd
from stable_baselines3 import PPO

from rl.trading_env import NiftyTradingEnv

TREND_MODEL = Path("data/processed/ppo_trend_agent")
MEAN_MODEL = Path("data/processed/ppo_mean_reversion_agent")
VOL_MODEL = Path("data/processed/ppo_volatility_agent")


def model_exists(model_path: Path) -> bool:
    return model_path.exists() or model_path.with_suffix(".zip").exists()


def load_agent_signal(model_path: Path, df: pd.DataFrame) -> str:
    if not model_exists(model_path):
        return "NA"

    model = PPO.load(str(model_path))

    env_df = pd.concat([df.copy()] * 3, ignore_index=True)
    env = NiftyTradingEnv(env_df)

    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)

    action_map = {0: "SHORT", 1: "HOLD", 2: "LONG"}
    return action_map[int(action)]


def ensemble_signal(df: pd.DataFrame) -> dict:
    trend = load_agent_signal(TREND_MODEL, df)
    mean_rev = load_agent_signal(MEAN_MODEL, df)
    vol = load_agent_signal(VOL_MODEL, df)

    votes = {"LONG": 0, "SHORT": 0, "HOLD": 0}

    for sig in [trend, mean_rev, vol]:
        if sig in votes:
            votes[sig] += 1

    if votes["LONG"] == votes["SHORT"] == votes["HOLD"] == 0:
        final_signal = "NA"
    else:
        final_signal = max(votes, key=votes.get)

    return {
        "trend_agent": trend,
        "mean_reversion_agent": mean_rev,
        "volatility_agent": vol,
        "votes": votes,
        "final_agent_vote": final_signal,
    }


if __name__ == "__main__":
    sample = pd.DataFrame([{
        "close": 100,
        "ret1": 0.0,
        "ret5": 0.0,
        "volatility": 0.01,
        "close_sma5_gap": 0.0,
        "close_sma20_gap": 0.0,
        "range": 10,
        "hour": 12,
        "minute": 0,
        "regime_code": 0.0,
        "vix": 15.0,
        "vix_regime_code": 0.0,
        "lstm_pred_return": 0.0,
        "bnn_mean_return": 0.0,
        "bnn_std_return": 0.01,
        "baseline_prob_up": 0.5,
    }])

    print(ensemble_signal(sample))