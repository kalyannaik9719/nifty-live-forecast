from pathlib import Path
import pandas as pd
from stable_baselines3 import PPO

from rl.trading_env import NiftyTradingEnv

DATA_PATH = Path("data/processed/nifty_regimes.parquet")
MODEL_PATH = Path("data/processed/ppo_volatility_agent")


def prepare_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH).copy()

    if "regime_code" not in df.columns:
        mapping = {"bull": 0.5, "bear": -0.5, "sideways": 0.0, "high_vol": 1.0}
        df["regime_code"] = df["regime"].map(mapping).fillna(0.0)

    defaults = {
        "vix": 18.0,
        "vix_regime_code": 0.5,
        "lstm_pred_return": 0.0,
        "bnn_mean_return": 0.0,
        "bnn_std_return": 0.02,
        "baseline_prob_up": 0.5,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    # focus this agent on volatile rows if possible
    if "regime" in df.columns:
        high_vol_df = df[df["regime"] == "high_vol"].copy()
        if len(high_vol_df) > 50:
            df = high_vol_df

    return df.dropna().reset_index(drop=True)


def train():
    df = prepare_data()
    env = NiftyTradingEnv(df)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=8e-5,
        n_steps=512,
        batch_size=64,
        gamma=0.995,
        ent_coef=0.03,
    )

    model.learn(total_timesteps=40000)
    model.save(str(MODEL_PATH))
    print("Saved volatility agent →", MODEL_PATH)


if __name__ == "__main__":
    train()