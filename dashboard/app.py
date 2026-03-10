import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

LIVE_SIGNAL_PATH = Path("data/processed/live_signal.json")

st.set_page_config(page_title="NIFTY50 Research Dashboard", layout="wide")
st.title("NIFTY50 Live Research Dashboard")
st.subheader("Live Signal Engine")

if LIVE_SIGNAL_PATH.exists():
    with open(LIVE_SIGNAL_PATH, "r") as f:
        live_signal = json.load(f)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Last Close", f"{live_signal.get('last_close', 0):.2f}")
    c2.metric("Regime", live_signal.get("regime", "unknown"))
    c3.metric("Baseline Signal", live_signal.get("baseline_signal", "NA"))
    c4.metric("RL Signal", live_signal.get("rl_signal", "NA"))
    c5.metric("Prob Up", f"{live_signal.get('baseline_prob_up', 0):.2%}")
else:
    st.warning("Live signal not generated yet.")
BASELINE_PATH = Path("data/processed/paper_trades.parquet")
RL_PATH = Path("data/processed/rl_trades.parquet")
COMPARE_PATH = Path("data/processed/model_comparison.json")
REGIME_PATH = Path("data/processed/nifty_regimes.parquet")

# ---------- Load data ----------
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

# ---------- Top metrics ----------
col1, col2, col3, col4 = st.columns(4)

baseline_equity = float(baseline_df["equity_curve"].iloc[-1])
rl_equity = float(rl_df["equity_curve"].iloc[-1])

baseline_dd = ((baseline_df["equity_curve"] - baseline_df["equity_curve"].cummax()) / baseline_df["equity_curve"].cummax()).min()
rl_dd = ((rl_df["equity_curve"] - rl_df["equity_curve"].cummax()) / rl_df["equity_curve"].cummax()).min()

col1.metric("Baseline Final Equity", f"{baseline_equity:.4f}")
col2.metric("RL Final Equity", f"{rl_equity:.4f}")
col3.metric("Latest Regime", latest_regime)
col4.metric("Winner", compare.get("winner", "unknown"))

# ---------- Drawdown metrics ----------
col5, col6 = st.columns(2)
col5.metric("Baseline Max Drawdown", f"{baseline_dd:.2%}")
col6.metric("RL Max Drawdown", f"{rl_dd:.2%}")

# ---------- Equity curves ----------
st.subheader("Equity Curve Comparison")

baseline_plot = baseline_df.copy()
baseline_plot["model"] = "Baseline"

rl_plot = rl_df.copy()
rl_plot["model"] = "RL"

if "ts" in baseline_plot.columns and "ts" in rl_plot.columns:
    baseline_plot["ts"] = pd.to_datetime(baseline_plot["ts"])
    rl_plot["ts"] = pd.to_datetime(rl_plot["ts"])
    plot_df = pd.concat([
        baseline_plot[["ts", "equity_curve", "model"]],
        rl_plot[["ts", "equity_curve", "model"]]
    ], ignore_index=True)

    fig = px.line(plot_df, x="ts", y="equity_curve", color="model")
else:
    baseline_plot["idx"] = range(len(baseline_plot))
    rl_plot["idx"] = range(len(rl_plot))
    plot_df = pd.concat([
        baseline_plot[["idx", "equity_curve", "model"]],
        rl_plot[["idx", "equity_curve", "model"]]
    ], ignore_index=True)

    fig = px.line(plot_df, x="idx", y="equity_curve", color="model")

st.plotly_chart(fig, use_container_width=True)

# ---------- RL action distribution ----------
st.subheader("RL Action Distribution")

action_map = {0: "Short", 1: "Flat", 2: "Long"}
rl_actions = rl_df["action"].map(action_map).value_counts().reset_index()
rl_actions.columns = ["action", "count"]

fig2 = px.bar(rl_actions, x="action", y="count")
st.plotly_chart(fig2, use_container_width=True)

# ---------- Latest rows ----------
st.subheader("Latest RL Trades")
st.dataframe(rl_df.tail(20), use_container_width=True)

st.subheader("Latest Baseline Trades")
st.dataframe(baseline_df.tail(20), use_container_width=True)