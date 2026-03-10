import sys
from pathlib import Path

# ---------- Fix import path for Streamlit Cloud ----------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import json
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from stable_baselines3 import PPO

from rl.trading_env import NiftyTradingEnv


# ----------------------------
# Paths
# ----------------------------
BASELINE_PATH = Path("data/processed/paper_trades.parquet")
RL_PATH = Path("data/processed/rl_trades.parquet")
COMPARE_PATH = Path("data/processed/model_comparison.json")
REGIME_PATH = Path("data/processed/nifty_regimes.parquet")
HIST_PATH = Path("data/processed/nifty_regimes.parquet")

BASELINE_MODEL_PATH = Path("data/processed/baseline_model.pkl")
RL_MODEL_PATH = Path("data/processed/ppo_nifty")


# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="NIFTY50 Research Dashboard", layout="wide")

st.title("NIFTY50 Live Research Dashboard")
st.subheader("Live Signal Engine")


# ----------------------------
# NSE LIVE FETCH
# ----------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/",
}


@st.cache_data(ttl=30)
def fetch_nse():
    session = requests.Session()
    session.headers.update(HEADERS)

    warmup = "https://www.nseindia.com/market-data/live-equity-market"
    session.get(warmup)

    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    r = session.get(url)
    data = r.json()

    row = data["data"][0]

    df = pd.DataFrame(
        [
            {
                "ts": datetime.now(),
                "close": row["lastPrice"],
                "open": row["open"],
                "high": row["dayHigh"],
                "low": row["dayLow"],
                "volume": 0,
                "change": row["change"],
                "pChange": row["pChange"],
            }
        ]
    )

    return df


# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def build_features():

    hist = pd.read_parquet(HIST_PATH)
    live = fetch_nse()

    hist["ts"] = pd.to_datetime(hist["ts"]).dt.tz_localize(None)
    live["ts"] = pd.to_datetime(live["ts"]).dt.tz_localize(None)

    hist = hist[["ts", "open", "high", "low", "close", "volume"]].tail(50)

    df = pd.concat([hist, live], ignore_index=True)
    df = df.sort_values("ts")

    df["ret1"] = df["close"].pct_change()
    df["ret5"] = df["close"].pct_change(5)

    df["sma5"] = df["close"].rolling(5).mean()
    df["sma20"] = df["close"].rolling(20).mean()

    df["ema12"] = df["close"].ewm(span=12).mean()

    df["volatility"] = df["ret1"].rolling(12).std()

    df["close_sma5_gap"] = (df["close"] - df["sma5"]) / df["sma5"]
    df["close_sma20_gap"] = (df["close"] - df["sma20"]) / df["sma20"]

    df["range"] = df["high"] - df["low"]

    df["hour"] = pd.to_datetime(df["ts"]).dt.hour
    df["minute"] = pd.to_datetime(df["ts"]).dt.minute

    vol75 = df["volatility"].quantile(0.75)

    def regime(row):

        if pd.isna(row["volatility"]):
            return "unknown"

        if row["volatility"] > vol75:
            return "high_vol"

        if row["close_sma20_gap"] > 0.002:
            return "bull"

        if row["close_sma20_gap"] < -0.002:
            return "bear"

        return "sideways"

    df["regime"] = df.apply(regime, axis=1)

    return df.tail(1)


# ----------------------------
# SIGNAL ENGINE
# ----------------------------
def generate_signal():

    live = build_features()

    features = [
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
    ]

    baseline = joblib.load(BASELINE_MODEL_PATH)

    prob_up = baseline.predict_proba(live[features])[0][1]

    if prob_up > 0.55:
        baseline_signal = "LONG"

    elif prob_up < 0.45:
        baseline_signal = "SHORT"

    else:
        baseline_signal = "HOLD"

    rl_df = pd.concat([live] * 3)

    env = NiftyTradingEnv(rl_df)

    obs, _ = env.reset()

    model = PPO.load(str(RL_MODEL_PATH))

    action, _ = model.predict(obs, deterministic=True)

    action_map = {0: "SHORT", 1: "HOLD", 2: "LONG"}

    rl_signal = action_map[int(action)]

    return {
        "ts": str(live["ts"].iloc[-1]),
        "close": float(live["close"].iloc[-1]),
        "regime": live["regime"].iloc[-1],
        "baseline_signal": baseline_signal,
        "baseline_prob": prob_up,
        "rl_signal": rl_signal,
    }


# ----------------------------
# SHOW LIVE SIGNAL
# ----------------------------
try:

    signal = generate_signal()

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Last Close", f"{signal['close']:.2f}")
    c2.metric("Regime", signal["regime"])
    c3.metric("Baseline", signal["baseline_signal"])
    c4.metric("RL Signal", signal["rl_signal"])
    c5.metric("Prob Up", f"{signal['baseline_prob']:.2%}")

    st.caption(f"Last update: {signal['ts']}")

except Exception as e:

    st.warning(f"Live engine error: {e}")


# ----------------------------
# BACKTEST DASHBOARD
# ----------------------------
st.divider()

st.header("Backtest Comparison")

baseline_df = pd.read_parquet(BASELINE_PATH)
rl_df = pd.read_parquet(RL_PATH)

compare = {}

if COMPARE_PATH.exists():

    with open(COMPARE_PATH) as f:

        compare = json.load(f)

col1, col2, col3 = st.columns(3)

baseline_equity = baseline_df["equity_curve"].iloc[-1]
rl_equity = rl_df["equity_curve"].iloc[-1]

col1.metric("Baseline Equity", f"{baseline_equity:.4f}")
col2.metric("RL Equity", f"{rl_equity:.4f}")
col3.metric("Winner", compare.get("winner", "unknown"))


# ----------------------------
# EQUITY CURVE
# ----------------------------
baseline_plot = baseline_df.copy()
baseline_plot["model"] = "Baseline"

rl_plot = rl_df.copy()
rl_plot["model"] = "RL"

baseline_plot["ts"] = pd.to_datetime(baseline_plot["ts"])
rl_plot["ts"] = pd.to_datetime(rl_plot["ts"])

plot = pd.concat(
    [
        baseline_plot[["ts", "equity_curve", "model"]],
        rl_plot[["ts", "equity_curve", "model"]],
    ]
)

fig = px.line(plot, x="ts", y="equity_curve", color="model")

st.plotly_chart(fig, width="stretch")


# ----------------------------
# RL ACTION DISTRIBUTION
# ----------------------------
st.subheader("RL Actions")

action_map = {0: "Short", 1: "Flat", 2: "Long"}

dist = rl_df["action"].map(action_map).value_counts().reset_index()

dist.columns = ["action", "count"]

fig2 = px.bar(dist, x="action", y="count")

st.plotly_chart(fig2, width="stretch")


# ----------------------------
# TABLES
# ----------------------------
st.subheader("Latest RL Trades")
st.dataframe(rl_df.tail(20), width="stretch")

st.subheader("Latest Baseline Trades")
st.dataframe(baseline_df.tail(20), width="stretch")