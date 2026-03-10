import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO

from rl.trading_env import NiftyTradingEnv

DATA_PATH = Path("data/processed/nifty_regimes.parquet")
MODEL_PATH = Path("data/processed/ppo_nifty")

def run_inference():
    df = pd.read_parquet(DATA_PATH)
    env = NiftyTradingEnv(df)

    model = PPO.load(str(MODEL_PATH))

    obs, info = env.reset()

    done = False
    rewards = []
    actions = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        actions.append(int(action))

        done = terminated or truncated

    total_reward = sum(rewards)

    print("Total RL reward:", round(total_reward, 6))
    print("Average reward:", round(total_reward / len(actions), 6))
    print("Actions taken:", len(actions))
    print("Short actions:", actions.count(0))
    print("Flat actions:", actions.count(1))
    print("Long actions:", actions.count(2))

if __name__ == "__main__":
    run_inference()