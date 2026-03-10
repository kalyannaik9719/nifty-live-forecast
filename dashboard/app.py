import json
from pathlib import Path
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
# Streamlit page
# ----------------------------
st.set_page_config(page_title="NIFTY50 Research Dashboard", layout="wide")
st.title("NIFTY50 Live Research Dashboard")

# ----------------------------
# NSE live fetch
# ----------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}


@st.cache_data(ttl=30)
def fetch_live_nse_quote():
    session = requests.Session()
    session.headers.update(HEADERS)

    warmup_url = "https://www.nseindia.com/market-data/live-equity-market"
    session.get(warmup_url, timeout=15)

    api_url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    r = session.get(api_url, timeout=15)
    r.raise_for_status()

    data = r.json()
    first_row = data["data"][0]

    row = {
        "ts": datetime.now(),
        "close": first_row.get("lastPrice"),
        "open": first_row.get("open"),
        "high": first_row.get("high", first_row.get("dayHigh")),
        "low": first_row.get("low", first_row.get("dayLow")),
        "change": first_row.get("change"),
        "pChange": first_row.get("pChange"),
        "volume": 0,
    }
    return pd.DataFrame([row])


def make_naive(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    try:
        return s.dt.tz_localize(None)
    except TypeError:
        return s


def build_live_features():
    hist = pd.read_parquet(HIST_PATH).copy()
    live = fetch_live_nse_quote().copy()

    hist["ts"] = make_naive(hist["ts"])
    live["ts"] = make_naive(live["ts"])

    for col in ["open", "high", "low", "close"]:
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
        live[col] = pd.to_numeric(live[col], errors="coerce")

    if "volume" not in hist.columns:
        hist["volume"] = 0
    if "volume" not in live.columns:
        live["volume"] = 0

    hist_base = hist[["ts", "open", "high", "low", "close", "volume"]].copy().tail(50)
    live_base = live[["ts", "open", "high", "low", "close", "volume"]].copy()

    all_df = pd.concat([hist_base, live_base], ignore_index=True)
    all_df["ts"] = make_naive(all_df["ts"])
    all_df = all_df.sort_values("ts").reset_index(drop=True)

    all_df["ret1"] = all_df["close"].pct_change()
    all_df["ret5"] = all_df["close"].pct_change(5)
    all_df["sma5"] = all_df["close"].rolling(5).mean()
    all_df["sma20"] = all_df["close"].rolling(20).mean()
    all_df["ema12"] = all_df["close"].ewm(span=12).mean()
    all_df["volatility"] = all_df["ret1"].rolling(12).std()
    all_df["close_sma5_gap"] = (all_df["close"] - all_df["sma5"]) / all_df["sma5"]
    all_df["close_sma20_gap"] = (all_df["close"] - all_df["sma20"]) / all_df["sma20"]
    all_df["range"] = all_df["high"] - all_df["low"]
    all_df["hour"] = pd.to_datetime(all_df["ts"]).dt.hour
    all_df["minute"] = pd.to_datetime(all_df["ts"]).dt.minute

    vol_q75 = all_df["volatility"].quantile(0.75)

    def get_regime(row):
        if pd.isna(row["volatility"]):
            return "unknown"
        if row["volatility"] > vol_q75:
            return "high_vol"
        elif row["close_sma20_gap"] > 0.002:
            return "bull"
        elif row["close_sma20_gap"] < -0.002:
            return "bear"
        else:
            return "sideways"

    all_df["regime"] = all_df.apply(get_regime, axis=1)
    live_row = all_df.tail(1).dropna().copy()
    return live_row


def generate_live_signal():
    live_df = build_live_features()

    baseline_features = [
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

    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    prob_up = float(baseline_model.predict_proba(live_df[baseline_features])[0, 1])

    if prob_up > 0.55:
        baseline_signal = "LONG"
    elif prob_up < 0.45:
        baseline_signal = "SHORT"
    else:
        baseline_signal = "HOLD"

    rl_df = pd.concat([live_df] * 3, ignore_index=True).copy()
    env = NiftyTradingEnv(rl_df)
    obs, _ = env.reset()

    rl_model = PPO.load(str(RL_MODEL_PATH))
    action, _ = rl_model.predict(obs, deterministic=True)
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
    return signal


# ----------------------------
# Live Signal Engine
# ----------------------------
st.subheader("Live Signal Engine")

try:
    live_signal = generate_live_signal()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Last Close", f"{live_signal.get('last_close', 0):.2f}")
    c2.metric("Live Regime", live_signal.get("regime", "unknown"))
    c3.metric("Baseline Signal", live_signal.get("baseline_signal", "NA"))
    c4.metric("RL Signal", live_signal.get("rl_signal", "NA"))
    c5.metric("Prob Up", f"{live_signal.get('baseline_prob_up', 0):.2%}")

    st.caption(f"Last live update: {live_signal.get('ts', 'NA')}")
except Exception as e:
    st.warning(f"Live signal fetch failed: {e}")

# ----------------------------
# Backtest / comparison section
# ----------------------------
if not BASELINE_PATH.exists():
    st.error("Missing baseline paper trades file.")
    st.stop()

if not RL_PATH.exists():
    st.error("Missing RL trades file.")
    st.stop()

baseline_df = pd.read_parquet(BASELINE_PATH)
rl_df = pd.read_parquet(RL_PATH)

compare = {}
if COMPARE_PATH.exists():
    with open(COMPARE_PATH, "r") as f:
        compare = json.load(f)

latest_regime = "unknown"
if REGIME_PATH.exists():
    regime_df = pd.read_parquet(REGIME_PATH)
    if "regime" in regime_df.columns and len(regime_df) > 0:
        latest_regime = regime_df["regime"].iloc[-1]

col1, col2, col3, col4 = st.columns(4)

baseline_equity = float(baseline_df["equity_curve"].iloc[-1])
rl_equity = float(rl_df["equity_curve"].iloc[-1])

baseline_dd = (
    (baseline_df["equity_curve"] - baseline_df["equity_curve"].cummax())
    / baseline_df["equity_curve"].cummax()
).min()
rl_dd = (
    (rl_df["equity_curve"] - rl_df["equity_curve"].cummax())
    / rl_df["equity_curve"].cummax()
).min()

col1.metric("Baseline Final Equity", f"{baseline_equity:.4f}")
col2.metric("RL Final Equity", f"{rl_equity:.4f}")
col3.metric("Backtest Regime Snapshot", latest_regime)
col4.metric("Winner", compare.get("winner", "unknown"))

col5, col6 = st.columns(2)
col5.metric("Baseline Max Drawdown", f"{baseline_dd:.2%}")
col6.metric("RL Max Drawdown", f"{rl_dd:.2%}")

st.subheader("Equity Curve Comparison")

baseline_plot = baseline_df.copy()
baseline_plot["model"] = "Baseline"

rl_plot = rl_df.copy()
rl_plot["model"] = "RL"

if "ts" in baseline_plot.columns and "ts" in rl_plot.columns:
    baseline_plot["ts"] = pd.to_datetime(baseline_plot["ts"])
    rl_plot["ts"] = pd.to_datetime(rl_plot["ts"])
    plot_df = pd.concat(
        [
            baseline_plot[["ts", "equity_curve", "model"]],
            rl_plot[["ts", "equity_curve", "model"]],
        ],
        ignore_index=True,
    )
    fig = px.line(plot_df, x="ts", y="equity_curve", color="model")
else:
    baseline_plot["idx"] = range(len(baseline_plot))
    rl_plot["idx"] = range(len(rl_plot))
    plot_df = pd.concat(
        [
            baseline_plot[["idx", "equity_curve", "model"]],
            rl_plot[["idx", "equity_curve", "model"]],
        ],
        ignore_index=True,
    )
    fig = px.line(plot_df, x="idx", y="equity_curve", color="model")

st.plotly_chart(fig, width="stretch")

st.subheader("RL Action Distribution")

action_map = {0: "Short", 1: "Flat", 2: "Long"}
rl_actions = rl_df["action"].map(action_map).value_counts().reset_index()
rl_actions.columns = ["action", "count"]

fig2 = px.bar(rl_actions, x="action", y="count")
st.plotly_chart(fig2, width="stretch")

st.subheader("Latest RL Trades")
st.dataframe(rl_df.tail(20), width="stretch")

st.subheader("Latest Baseline Trades")
st.dataframe(baseline_df.tail(20), width="stretch")