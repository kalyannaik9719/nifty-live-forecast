import sys
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import json
import math
import joblib
import pandas as pd
from stable_baselines3 import PPO

from rl.trading_env import NiftyTradingEnv

# Optional imports
try:
    from models.lstm_model import infer_lstm
except Exception:
    infer_lstm = None

try:
    from models.bnn_model import infer_bnn
except Exception:
    infer_bnn = None

try:
    from rl_agents.vote_engine import ensemble_signal
except Exception:
    ensemble_signal = None


# ----------------------------
# Paths
# ----------------------------
LIVE_FEATURES = Path("data/processed/live_features.parquet")
HIST_PATH = Path("data/processed/nifty_regimes.parquet")
VIX_PATH = Path("data/raw/nse_vix_quote.parquet")
GAMMA_PATH = Path("data/processed/gamma_levels.json")

BASELINE_MODEL = Path("data/processed/baseline_model.pkl")
RL_MODEL = Path("data/processed/ppo_nifty")

LSTM_ART = Path("artifacts/lstm")
BNN_ART = Path("artifacts/bnn")

OUT_PATH = Path("data/processed/live_signal.json")


# ----------------------------
# Helpers
# ----------------------------
def safe_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def get_vix():
    if not VIX_PATH.exists():
        return None

    df = pd.read_parquet(VIX_PATH)
    if len(df) == 0 or "vix" not in df.columns:
        return None

    return safe_float(df["vix"].iloc[-1], None)


def vix_regime(vix):
    if vix is None:
        return "unknown", 0.0

    if vix < 13:
        return "calm", -0.5
    elif vix < 18:
        return "normal", 0.0
    elif vix < 24:
        return "elevated", 0.5
    else:
        return "panic", 1.0


def ensure_regime_code(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "regime_code" not in out.columns:
        mapping = {
            "bull": 0.5,
            "bear": -0.5,
            "sideways": 0.0,
            "high_vol": 1.0,
        }
        if "regime" in out.columns:
            out["regime_code"] = out["regime"].map(mapping).fillna(0.0)
        else:
            out["regime_code"] = 0.0
    return out


def get_gamma_levels(live_row=None):
    # First try actual gamma file
    if GAMMA_PATH.exists():
        try:
            with open(GAMMA_PATH, "r") as f:
                data = json.load(f)

            gamma_support = data.get("gamma_support")
            gamma_resistance = data.get("gamma_resistance")

            if gamma_support is not None or gamma_resistance is not None:
                return {
                    "gamma_support": gamma_support,
                    "gamma_resistance": gamma_resistance,
                    "gamma_status": data.get("status", "ok"),
                }
        except Exception:
            pass

    # Fallback from live candle range
    if live_row is not None:
        low = safe_float(live_row.get("low"), None)
        high = safe_float(live_row.get("high"), None)

        return {
            "gamma_support": low,
            "gamma_resistance": high,
            "gamma_status": "fallback_from_live_range",
        }

    # If nothing works
    return {
        "gamma_support": None,
        "gamma_resistance": None,
        "gamma_status": "missing",
    }


def baseline_signal(df: pd.DataFrame):
    model = joblib.load(BASELINE_MODEL)

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

    prob_up = float(model.predict_proba(df[features])[0][1])

    if prob_up > 0.55:
        sig = "LONG"
    elif prob_up < 0.45:
        sig = "SHORT"
    else:
        sig = "HOLD"

    return sig, prob_up


def single_rl_signal(df: pd.DataFrame):
    model = PPO.load(str(RL_MODEL))

    env_df = pd.concat([df.copy()] * 3, ignore_index=True)
    env = NiftyTradingEnv(env_df)

    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)

    action_map = {0: "SHORT", 1: "HOLD", 2: "LONG"}
    return action_map[int(action)]


def get_lstm_outputs(seq_df: pd.DataFrame):
    default = {
        "lstm_pred_return": 0.0,
        "lstm_direction": "NA",
        "lstm_confidence": None,
    }

    if infer_lstm is None or not LSTM_ART.exists():
        return default

    try:
        out = infer_lstm(seq_df, artifact_dir=LSTM_ART)
        return {
            "lstm_pred_return": safe_float(out.get("lstm_pred_return"), 0.0),
            "lstm_direction": out.get("lstm_direction", "NA"),
            "lstm_confidence": safe_float(out.get("lstm_confidence"), None),
        }
    except Exception:
        return default


def get_bnn_outputs(latest_df: pd.DataFrame):
    default = {
        "bnn_mean_return": 0.0,
        "bnn_std_return": None,
        "bnn_confidence": None,
        "bnn_direction": "NA",
    }

    if infer_bnn is None:
        print("infer_bnn import failed")
        return default

    if not BNN_ART.exists():
        print("BNN artifact folder missing:", BNN_ART)
        return default

    try:
        out = infer_bnn(latest_df, artifact_dir=BNN_ART)
        print("BNN output:", out)

        return {
            "bnn_mean_return": safe_float(out.get("bnn_mean_return"), 0.0),
            "bnn_std_return": safe_float(out.get("bnn_std_return"), None),
            "bnn_confidence": safe_float(out.get("bnn_confidence"), None),
            "bnn_direction": out.get("bnn_direction", "NA"),
        }
    except Exception as e:
        print("BNN loading error:", e)
        return default


def get_multi_agent_outputs(df: pd.DataFrame):
    default = {
        "trend_agent": "NA",
        "mean_reversion_agent": "NA",
        "volatility_agent": "NA",
        "final_agent_vote": "NA",
    }

    if ensemble_signal is None:
        print("ensemble_signal import failed")
        return default

    try:
        out = ensemble_signal(df)
        print("Multi-agent output:", out)

        return {
            "trend_agent": out.get("trend_agent", "NA"),
            "mean_reversion_agent": out.get("mean_reversion_agent", "NA"),
            "volatility_agent": out.get("volatility_agent", "NA"),
            "final_agent_vote": out.get("final_agent_vote", "NA"),
        }
    except Exception as e:
        print("Multi-agent loading error:", e)
        return default


def build_final_signal(
    regime,
    vix,
    baseline,
    rl,
    lstm,
    bnn_conf,
    final_agent_vote,
):
    final_signal = "HOLD"
    reason = "No alignment"
    warnings = []

    if vix is not None and vix > 22:
        return "HOLD", "VIX too high", ["Avoid trade in extreme volatility"]

    if regime == "high_vol":
        warnings.append("High volatility regime")

    if bnn_conf is not None and bnn_conf < 0.55:
        return "HOLD", "Low BNN confidence", warnings + ["Low model confidence"]

    votes = [baseline, rl, lstm, final_agent_vote]
    long_votes = votes.count("LONG")
    short_votes = votes.count("SHORT")

    if regime in ["bull", "bear"]:
        if long_votes >= 2:
            final_signal = "LONG"
            reason = "Trend alignment"
        elif short_votes >= 2:
            final_signal = "SHORT"
            reason = "Trend alignment"

    elif regime == "sideways":
        if long_votes >= 3:
            final_signal = "LONG"
            reason = "Strong consensus"
        elif short_votes >= 3:
            final_signal = "SHORT"
            reason = "Strong consensus"
        else:
            final_signal = "HOLD"
            reason = "Sideways low edge"

    elif regime == "high_vol":
        if long_votes >= 3:
            final_signal = "LONG"
            reason = "High-vol breakout consensus"
        elif short_votes >= 3:
            final_signal = "SHORT"
            reason = "High-vol breakdown consensus"
        else:
            final_signal = "HOLD"
            reason = "High volatility disagreement"
            warnings.append("Baseline/RL disagreement")

    return final_signal, reason, warnings


def compute_position_size(final_signal, baseline_prob_up, bnn_confidence, vix):
    if final_signal == "HOLD":
        return 0.0

    edge = 0.0
    if baseline_prob_up is not None:
        edge = abs(baseline_prob_up - 0.5) * 2.0

    conf = 0.5 if bnn_confidence is None else max(0.0, min(1.0, bnn_confidence))

    vol_penalty = 1.0
    if vix is not None:
        if vix >= 22:
            vol_penalty = 0.25
        elif vix >= 18:
            vol_penalty = 0.5
        elif vix >= 13:
            vol_penalty = 0.75

    size = edge * conf * 0.02 * vol_penalty
    return round(min(max(size, 0.0), 0.02), 4)


def compute_trade_plan(last_close, final_signal, volatility, gamma_support, gamma_resistance):
    if final_signal == "HOLD":
        return {
            "buy_above": None,
            "sell_below": None,
            "stop_loss": None,
            "target": None,
        }

    if volatility is None:
        sl_points = 80
    else:
        sl_points = max(60, min(150, int(last_close * volatility * 3)))

    target_points = sl_points * 2

    buy_above = None
    sell_below = None
    stop_loss = None
    target = None

    if final_signal == "LONG":
        buy_above = round(last_close + 5, 2)

        if gamma_resistance is not None and buy_above >= float(gamma_resistance) - 20:
            buy_above = round(float(gamma_resistance) + 5, 2)

        stop_loss = round(buy_above - sl_points, 2)
        target = round(buy_above + target_points, 2)

    elif final_signal == "SHORT":
        sell_below = round(last_close - 5, 2)

        if gamma_support is not None and sell_below <= float(gamma_support) + 20:
            sell_below = round(float(gamma_support) - 5, 2)

        stop_loss = round(sell_below + sl_points, 2)
        target = round(sell_below - target_points, 2)

    return {
        "buy_above": buy_above,
        "sell_below": sell_below,
        "stop_loss": stop_loss,
        "target": target,
    }


def run():
    if not LIVE_FEATURES.exists():
        raise FileNotFoundError(f"Missing live features file: {LIVE_FEATURES}")

    latest_live = pd.read_parquet(LIVE_FEATURES).copy().tail(1)
    latest_live = ensure_regime_code(latest_live)

    hist = pd.read_parquet(HIST_PATH).copy() if HIST_PATH.exists() else latest_live.copy()
    hist = ensure_regime_code(hist)
    seq_df = pd.concat([hist.tail(40), latest_live], ignore_index=True)

    # VIX
    vix = get_vix()
    vix_label, vix_code = vix_regime(vix)

    # Baseline
    baseline_sig, baseline_prob_up = baseline_signal(latest_live)

    # Sequence model outputs
    lstm_out = get_lstm_outputs(seq_df)
    bnn_out = get_bnn_outputs(latest_live)

    # Add context for RL and agents
    latest_live["vix"] = 15.0 if vix is None else vix
    latest_live["vix_regime_code"] = vix_code
    latest_live["lstm_pred_return"] = lstm_out["lstm_pred_return"]
    latest_live["bnn_mean_return"] = bnn_out["bnn_mean_return"]
    latest_live["bnn_std_return"] = 0.01 if bnn_out["bnn_std_return"] is None else bnn_out["bnn_std_return"]
    latest_live["baseline_prob_up"] = baseline_prob_up

    # RL
    rl_sig = single_rl_signal(latest_live)

    # Multi-agent vote
    agent_out = get_multi_agent_outputs(latest_live)

    # Final signal
    regime = str(latest_live["regime"].iloc[-1]) if "regime" in latest_live.columns else "unknown"

    final_sig, reason, warnings = build_final_signal(
        regime=regime,
        vix=vix,
        baseline=baseline_sig,
        rl=rl_sig,
        lstm=lstm_out["lstm_direction"],
        bnn_conf=bnn_out["bnn_confidence"],
        final_agent_vote=agent_out["final_agent_vote"],
    )

    # Position sizing
    position_size = compute_position_size(
        final_signal=final_sig,
        baseline_prob_up=baseline_prob_up,
        bnn_confidence=bnn_out["bnn_confidence"],
        vix=vix,
    )

    # Gamma / fallback levels
    live_row = latest_live.iloc[0].to_dict()
    gamma = get_gamma_levels(live_row)

    gamma_support = gamma["gamma_support"]
    gamma_resistance = gamma["gamma_resistance"]
    gamma_status = gamma["gamma_status"]

    # Trade plan
    last_close = float(latest_live["close"].iloc[-1])
    volatility = safe_float(latest_live["volatility"].iloc[-1], None) if "volatility" in latest_live.columns else None

    trade_plan = compute_trade_plan(
        last_close=last_close,
        final_signal=final_sig,
        volatility=volatility,
        gamma_support=gamma_support,
        gamma_resistance=gamma_resistance,
    )

    out = {
        "ts": str(latest_live["ts"].iloc[-1]),
        "last_close": last_close,
        "regime": regime,
        "vix": vix,
        "vix_regime": vix_label,
        "baseline_signal": baseline_sig,
        "baseline_prob_up": baseline_prob_up,
        "rl_signal": rl_sig,
        "trend_agent": agent_out["trend_agent"],
        "mean_reversion_agent": agent_out["mean_reversion_agent"],
        "volatility_agent": agent_out["volatility_agent"],
        "final_agent_vote": agent_out["final_agent_vote"],
        "lstm_direction": lstm_out["lstm_direction"],
        "lstm_pred_return": lstm_out["lstm_pred_return"],
        "lstm_confidence": lstm_out["lstm_confidence"],
        "bnn_direction": bnn_out["bnn_direction"],
        "bnn_mean_return": bnn_out["bnn_mean_return"],
        "bnn_std_return": bnn_out["bnn_std_return"],
        "bnn_confidence": bnn_out["bnn_confidence"],
        "final_signal": final_sig,
        "reason": reason,
        "position_size": position_size,
        "gamma_support": gamma_support,
        "gamma_resistance": gamma_resistance,
        "gamma_status": gamma_status,
        "buy_above": trade_plan["buy_above"],
        "sell_below": trade_plan["sell_below"],
        "stop_loss": trade_plan["stop_loss"],
        "target": trade_plan["target"],
        "warnings": warnings,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))

    print("Saved live signal →", OUT_PATH)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run()