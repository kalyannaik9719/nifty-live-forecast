import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

FEATURES = [
    "ret1",
    "ret5",
    "volatility",
    "close_sma5_gap",
    "close_sma20_gap",
    "range",
    "hour",
    "minute"
]

class NiftyTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, cost: float = 0.0002, flat_penalty: float = 0.00005):
        super().__init__()

        self.df = df.reset_index(drop=True).copy()
        self.cost = cost
        self.flat_penalty = flat_penalty

        self.current_step = 0
        self.position = 0  # -1 short, 0 flat, 1 long

        self.action_space = spaces.Discrete(3)  # 0=short, 1=flat, 2=long

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(FEATURES) + 1,),
            dtype=np.float32
        )

    def _get_observation(self):
        row = self.df.loc[self.current_step, FEATURES].astype(float).values
        return np.array(list(row) + [self.position], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        return self._get_observation(), {}

    def step(self, action):
        new_position = {0: -1, 1: 0, 2: 1}[int(action)]

        price_now = float(self.df.loc[self.current_step, "close"])
        price_next = float(self.df.loc[self.current_step + 1, "close"])

        raw_return = (price_next / price_now) - 1.0

        trade_penalty = self.cost if new_position != self.position else 0.0

        # Base PnL reward
        reward = new_position * raw_return - trade_penalty

        # Small penalty for staying flat
        if new_position == 0:
            reward -= self.flat_penalty

        # Bonus for correct directional action
        if raw_return > 0 and new_position == 1:
            reward += 0.0001
        elif raw_return < 0 and new_position == -1:
            reward += 0.0001

        self.position = new_position
        self.current_step += 1

        terminated = self.current_step >= len(self.df) - 2
        truncated = False

        obs = self._get_observation()
        info = {
            "position": self.position,
            "raw_return": raw_return,
            "reward": reward
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass