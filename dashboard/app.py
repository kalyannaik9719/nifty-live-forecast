import sys
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

LIVE_SIGNAL_PATH = Path("data/processed/live_signal.json")
LEDGER_PATH = Path("data/processed/live_trade_ledger.parquet")
BASELINE_PATH = Path("data/processed/paper_trades.parquet")
RL_PATH = Path("data/processed/rl_trades.parquet")
COMPARE_PATH = Path("data/processed/model_comparison.json")

st.set_page_config(page_title="NIFTY Institutional AI Dashboard", layout="wide")
st.title("NIFTY Institutional AI Dashboard")

# =========================================================
# LIVE SIGNAL ENGINE
# =========================================================
st.subheader("Live Decision Engine")

if LIVE_SIGNAL_PATH.exists():
    with open(LIVE_SIGNAL_PATH, "r") as f:
        signal = json.load(f)

    last_close = signal.get("last_close", 0)
    regime = signal.get("regime", "unknown")
    vix = signal.get("vix", None)
    vix_regime = signal.get("vix_regime", "unknown")

    baseline_signal = signal.get("baseline_signal", "NA")
    baseline_prob = signal.get("baseline_prob_up", None)

    rl_signal = signal.get("rl_signal", "NA")

    trend_agent = signal.get("trend_agent", "NA")
    mean_agent = signal.get("mean_reversion_agent", "NA")
    vol_agent = signal.get("volatility_agent", "NA")
    final_agent_vote = signal.get("final_agent_vote", "NA")

    final_signal = signal.get("final_signal", "NA")
    reason = signal.get("reason", "NA")

    lstm_direction = signal.get("lstm_direction", "NA")
    lstm_pred_return = signal.get("lstm_pred_return", None)

    bnn_direction = signal.get("bnn_direction", "NA")
    bnn_confidence = signal.get("bnn_confidence", None)
    bnn_std = signal.get("bnn_std_return", None)

    buy_level = signal.get("buy_level", None)
    sell_level = signal.get("sell_level", None)

    position_size = signal.get("position_size", None)
    warnings_list = signal.get("warnings", [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close", f"{last_close:.2f}")
    c2.metric("Final Signal", final_signal)
    c3.metric("Reason", reason)
    c4.metric("Regime", regime)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("VIX", "NA" if vix is None else f"{vix:.2f}")
    c6.metric("VIX Regime", vix_regime)
    c7.metric("Baseline Signal", baseline_signal)
    c8.metric("RL Signal", rl_signal)

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Trend Agent", trend_agent)
    c10.metric("Mean Reversion Agent", mean_agent)
    c11.metric("Volatility Agent", vol_agent)
    c12.metric("Agent Vote", final_agent_vote)

    c13, c14, c15, c16 = st.columns(4)
    c13.metric("LSTM Direction", lstm_direction)
    c14.metric("BNN Direction", bnn_direction)
    c15.metric("Buy Level", "NA" if buy_level is None else f"{buy_level:.2f}")
    c16.metric("Sell Level", "NA" if sell_level is None else f"{sell_level:.2f}")

    c17, c18, c19, c20 = st.columns(4)
    c17.metric("Baseline Prob Up", "NA" if baseline_prob is None else f"{baseline_prob:.2%}")
    c18.metric("BNN Confidence", "NA" if bnn_confidence is None else f"{bnn_confidence:.2%}")
    c19.metric("BNN Std", "NA" if bnn_std is None else f"{bnn_std:.5f}")
    c20.metric("Position Size", "NA" if position_size is None else f"{position_size:.2%}")

    if lstm_pred_return is not None:
        st.info(f"LSTM Predicted Return: {lstm_pred_return:.5f}")

    if warnings_list:
        st.warning(" | ".join(warnings_list))

    st.caption(f"Last update: {signal.get('ts', 'unknown')}")

else:
    st.warning("Live signal file not found.")

st.divider()

# =========================================================
# MODEL COMPARISON
# =========================================================
st.subheader("Backtest Model Comparison")

if BASELINE_PATH.exists() and RL_PATH.exists():
    baseline_df = pd.read_parquet(BASELINE_PATH)
    rl_df = pd.read_parquet(RL_PATH)

    compare = {}
    if COMPARE_PATH.exists():
        with open(COMPARE_PATH, "r") as f:
            compare = json.load(f)

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

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baseline Equity", f"{baseline_equity:.4f}")
    m2.metric("RL Equity", f"{rl_equity:.4f}")
    m3.metric("Baseline Drawdown", f"{baseline_dd:.2%}")
    m4.metric("RL Drawdown", f"{rl_dd:.2%}")

    st.metric("Winner", compare.get("winner", "unknown"))

    st.subheader("Equity Curve Comparison")

    bp = baseline_df.copy()
    rp = rl_df.copy()
    bp["model"] = "Baseline"
    rp["model"] = "RL"

    if "ts" in bp.columns and "ts" in rp.columns:
        bp["ts"] = pd.to_datetime(bp["ts"])
        rp["ts"] = pd.to_datetime(rp["ts"])

        plot_df = pd.concat(
            [
                bp[["ts", "equity_curve", "model"]],
                rp[["ts", "equity_curve", "model"]],
            ],
            ignore_index=True,
        )
        fig = px.line(plot_df, x="ts", y="equity_curve", color="model")
    else:
        bp["idx"] = range(len(bp))
        rp["idx"] = range(len(rp))
        plot_df = pd.concat(
            [
                bp[["idx", "equity_curve", "model"]],
                rp[["idx", "equity_curve", "model"]],
            ],
            ignore_index=True,
        )
        fig = px.line(plot_df, x="idx", y="equity_curve", color="model")

    st.plotly_chart(fig, width="stretch")

else:
    st.warning("Backtest files not available yet.")

st.divider()

# =========================================================
# RL ACTION DISTRIBUTION
# =========================================================
if RL_PATH.exists():
    st.subheader("RL Action Distribution")

    rl_df = pd.read_parquet(RL_PATH)
    if "action" in rl_df.columns:
        action_map = {0: "Short", 1: "Flat", 2: "Long"}
        actions = rl_df["action"].map(action_map).value_counts().reset_index()
        actions.columns = ["action", "count"]

        fig2 = px.bar(actions, x="action", y="count")
        st.plotly_chart(fig2, width="stretch")

st.divider()

# =========================================================
# LIVE PAPER TRADE LEDGER
# =========================================================
if LEDGER_PATH.exists():
    st.subheader("Live Paper Trade Ledger")
    ledger = pd.read_parquet(LEDGER_PATH)
    st.dataframe(ledger.tail(50), width="stretch")