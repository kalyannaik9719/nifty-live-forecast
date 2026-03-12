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
    "minute",
    "regime_code",
    "vix",
    "vix_regime_code",
    "lstm_pred_return",
    "bnn_mean_return",
    "bnn_std_return",
    "baseline_prob_up",
]


class NiftyTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, cost: float = 0.0002):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.cost = cost

        defaults = {
            "regime_code": 0.0,
            "vix": 15.0,
            "vix_regime_code": 0.0,
            "lstm_pred_return": 0.0,
            "bnn_mean_return": 0.0,
            "bnn_std_return": 0.01,
            "baseline_prob_up": 0.5,
        }
        for col, default in defaults.items():
            if col not in self.df.columns:
                self.df[col] = default

        self.df = self.df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"] + FEATURES).reset_index(drop=True)
        if len(self.df) < 3:
            raise ValueError("Not enough rows for RL environment.")

        self.current_step = 0
        self.position = 0  # -1 short, 0 flat, 1 long

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(FEATURES) + 1,),
            dtype=np.float32,
        )

    def _get_obs(self):
        row = self.df.loc[self.current_step, FEATURES].astype(float).values
        return np.array(list(row) + [self.position], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        return self._get_obs(), {}

    def step(self, action):
        action_map = {0: -1, 1: 0, 2: 1}
        new_position = action_map[int(action)]

        price_now = float(self.df.loc[self.current_step, "close"])
        price_next = float(self.df.loc[self.current_step + 1, "close"])
        raw_return = (price_next / price_now) - 1.0

        regime_code = float(self.df.loc[self.current_step, "regime_code"])
        vix = float(self.df.loc[self.current_step, "vix"])
        bnn_std = float(self.df.loc[self.current_step, "bnn_std_return"])

        reward = new_position * raw_return

        if new_position != self.position:
            reward -= self.cost

        # trending markets: discourage flat
        if regime_code in (0.5, -0.5) and new_position == 0:
            reward -= 0.00003

        # sideways: discourage overtrading
        if regime_code == 0.0 and new_position != 0:
            reward -= 0.00005

        # elevated/panic vol: discourage aggressive positions
        if vix >= 18 and new_position != 0:
            reward -= 0.00008

        # uncertainty penalty
        reward -= min(0.0002, bnn_std * 0.5)

        # directional bonus
        if raw_return > 0 and new_position == 1:
            reward += 0.0001
        elif raw_return < 0 and new_position == -1:
            reward += 0.0001

        self.position = new_position
        self.current_step += 1

        terminated = self.current_step >= len(self.df) - 2
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {
            "position": self.position,
            "raw_return": raw_return,
            "reward": reward,
        }

    def render(self):
        pass