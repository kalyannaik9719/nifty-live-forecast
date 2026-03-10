import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO

from rl.trading_env import NiftyTradingEnv

DATA_PATH = Path("data/processed/nifty_regimes.parquet")
MODEL_PATH = Path("data/processed/ppo_nifty")
OUT_PATH = Path("data/processed/rl_trades.parquet")

def run_evaluation():
    df = pd.read_parquet(DATA_PATH).reset_index(drop=True)

    # use last 30% as evaluation set
    split_idx = int(len(df) * 0.7)
    eval_df = df.iloc[split_idx:].reset_index(drop=True)

    env = NiftyTradingEnv(eval_df)
    model = PPO.load(str(MODEL_PATH))

    obs, info = env.reset()

    done = False
    rewards = []
    actions = []
    positions = []
    raw_returns = []

    step_idx = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        actions.append(int(action))
        positions.append(info["position"])
        raw_returns.append(info["raw_return"])

        done = terminated or truncated
        step_idx += 1

    result = eval_df.iloc[:len(rewards)].copy()
    result["action"] = actions
    result["position"] = positions
    result["raw_return"] = raw_returns
    result["reward"] = rewards
    result["equity_curve"] = (1 + pd.Series(rewards)).cumprod()

    # drawdown
    result["rolling_max"] = result["equity_curve"].cummax()
    result["drawdown"] = (
        result["equity_curve"] - result["rolling_max"]
    ) / result["rolling_max"]

    result.to_parquet(OUT_PATH, index=False)

    print("RL evaluation saved →", OUT_PATH)
    print("Final Equity:", round(result["equity_curve"].iloc[-1], 4))
    print("Max Drawdown:", round(result["drawdown"].min(), 4))
    print("Total Reward:", round(sum(rewards), 6))
    print("Actions Taken:", len(actions))
    print("Short:", actions.count(0))
    print("Flat:", actions.count(1))
    print("Long:", actions.count(2))

if __name__ == "__main__":
    run_evaluation()