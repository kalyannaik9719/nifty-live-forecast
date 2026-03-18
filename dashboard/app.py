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

st.set_page_config(page_title="NIFTY Quant Terminal", layout="wide")
st.title("NIFTY Quant Terminal")

if LIVE_SIGNAL_PATH.exists():
    with open(LIVE_SIGNAL_PATH, "r") as f:
        signal = json.load(f)

    st.subheader("Live Decision Engine")

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("Last Close", f"{signal.get('last_close', 0):.2f}")
    r1c2.metric("Final Signal", signal.get("final_signal", "NA"))
    r1c3.metric("Reason", signal.get("reason", "NA"))
    r1c4.metric("Regime", signal.get("regime", "NA"))

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("VIX", "NA" if signal.get("vix") is None else f"{signal.get('vix'):.2f}")
    r2c2.metric("VIX Regime", signal.get("vix_regime", "NA"))
    r2c3.metric("Gamma Support", "NA" if signal.get("gamma_support") is None else f"{signal.get('gamma_support'):.2f}")
    r2c4.metric("Gamma Resistance", "NA" if signal.get("gamma_resistance") is None else f"{signal.get('gamma_resistance'):.2f}")

    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
    r3c1.metric("Baseline", signal.get("baseline_signal", "NA"))
    r3c2.metric("RL", signal.get("rl_signal", "NA"))
    r3c3.metric("LSTM", signal.get("lstm_direction", "NA"))
    r3c4.metric("BNN", signal.get("bnn_direction", "NA"))

    r4c1, r4c2, r4c3, r4c4 = st.columns(4)
    r4c1.metric("Trend Agent", signal.get("trend_agent", "NA"))
    r4c2.metric("Mean Reversion", signal.get("mean_reversion_agent", "NA"))
    r4c3.metric("Volatility Agent", signal.get("volatility_agent", "NA"))
    r4c4.metric("Agent Vote", signal.get("final_agent_vote", "NA"))

    r5c1, r5c2, r5c3, r5c4 = st.columns(4)
    r5c1.metric("BUY ABOVE", "NA" if signal.get("buy_above") is None else f"{signal.get('buy_above'):.2f}")
    r5c2.metric("SELL BELOW", "NA" if signal.get("sell_below") is None else f"{signal.get('sell_below'):.2f}")
    r5c3.metric("STOP LOSS", "NA" if signal.get("stop_loss") is None else f"{signal.get('stop_loss'):.2f}")
    r5c4.metric("TARGET", "NA" if signal.get("target") is None else f"{signal.get('target'):.2f}")

    r6c1, r6c2, r6c3, r6c4 = st.columns(4)
    bprob = signal.get("baseline_prob_up")
    bconf = signal.get("bnn_confidence")
    bstd = signal.get("bnn_std_return")
    pos = signal.get("position_size")

    r6c1.metric("Baseline Prob Up", "NA" if bprob is None else f"{bprob:.2%}")
    r6c2.metric("BNN Confidence", "NA" if bconf is None else f"{bconf:.2%}")
    r6c3.metric("BNN Std", "NA" if bstd is None else f"{bstd:.5f}")
    r6c4.metric("Position Size", "NA" if pos is None else f"{pos:.2%}")

    if signal.get("lstm_pred_return") is not None:
        st.info(f"LSTM Predicted Return: {signal.get('lstm_pred_return'):.5f}")

    warnings_list = signal.get("warnings", [])
    if warnings_list:
        st.warning(" | ".join(warnings_list))

    st.caption(f"Last update: {signal.get('ts', 'unknown')}")

st.divider()

st.subheader("Backtest Model Comparison")

if BASELINE_PATH.exists() and RL_PATH.exists():
    baseline_df = pd.read_parquet(BASELINE_PATH)
    rl_df = pd.read_parquet(RL_PATH)

    compare = {}
    if COMPARE_PATH.exists():
        with open(COMPARE_PATH, "r") as f:
            compare = json.load(f)

    b_eq = float(baseline_df["equity_curve"].iloc[-1])
    r_eq = float(rl_df["equity_curve"].iloc[-1])

    b_dd = ((baseline_df["equity_curve"] - baseline_df["equity_curve"].cummax()) / baseline_df["equity_curve"].cummax()).min()
    r_dd = ((rl_df["equity_curve"] - rl_df["equity_curve"].cummax()) / rl_df["equity_curve"].cummax()).min()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline Equity", f"{b_eq:.4f}")
    c2.metric("RL Equity", f"{r_eq:.4f}")
    c3.metric("Baseline Drawdown", f"{b_dd:.2%}")
    c4.metric("RL Drawdown", f"{r_dd:.2%}")

    st.metric("Winner", compare.get("winner", "unknown"))

    baseline_plot = baseline_df.copy()
    baseline_plot["model"] = "Baseline"

    rl_plot = rl_df.copy()
    rl_plot["model"] = "RL"

    if "ts" in baseline_plot.columns and "ts" in rl_plot.columns:
        baseline_plot["ts"] = pd.to_datetime(baseline_plot["ts"])
        rl_plot["ts"] = pd.to_datetime(rl_plot["ts"])

        plot_df = pd.concat([
            baseline_plot[["ts", "equity_curve", "model"]],
            rl_plot[["ts", "equity_curve", "model"]],
        ], ignore_index=True)

        fig = px.line(plot_df, x="ts", y="equity_curve", color="model", title="Equity Curve Comparison")
    else:
        baseline_plot["idx"] = range(len(baseline_plot))
        rl_plot["idx"] = range(len(rl_plot))

        plot_df = pd.concat([
            baseline_plot[["idx", "equity_curve", "model"]],
            rl_plot[["idx", "equity_curve", "model"]],
        ], ignore_index=True)

        fig = px.line(plot_df, x="idx", y="equity_curve", color="model", title="Equity Curve Comparison")

    st.plotly_chart(fig, width="stretch")

if RL_PATH.exists():
    st.subheader("RL Action Distribution")
    rl_df = pd.read_parquet(RL_PATH)
    if "action" in rl_df.columns:
        action_map = {0: "Short", 1: "Flat", 2: "Long"}
        actions = rl_df["action"].map(action_map).value_counts().reset_index()
        actions.columns = ["action", "count"]
        fig2 = px.bar(actions, x="action", y="count")
        st.plotly_chart(fig2, width="stretch")

if LEDGER_PATH.exists():
    st.subheader("Live Paper Trade Ledger")
    ledger = pd.read_parquet(LEDGER_PATH)
    st.dataframe(ledger.tail(50), width="stretch")