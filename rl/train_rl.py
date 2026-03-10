import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO

from rl.trading_env import NiftyTradingEnv

DATA_PATH = Path("data/processed/nifty_regimes.parquet")
MODEL_PATH = Path("data/processed/ppo_nifty")

def load_data():
    df = pd.read_parquet(DATA_PATH)
    return df

def train():
    df = load_data()
    env = NiftyTradingEnv(df)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=0.0001,
        n_steps=512,
        batch_size=64,
        gamma=0.995,
        ent_coef=0.01
    )

    model.learn(total_timesteps=50000)
    model.save(str(MODEL_PATH))

    print("RL model saved →", MODEL_PATH)

if __name__ == "__main__":
    train()