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

# Optional model imports
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

    if infer_bnn is None or not BNN_ART.exists():
        return default

    try:
        out = infer_bnn(latest_df, artifact_dir=BNN_ART)
        return {
            "bnn_mean_return": safe_float(out.get("bnn_mean_return"), 0.0),
            "bnn_std_return": safe_float(out.get("bnn_std_return"), None),
            "bnn_confidence": safe_float(out.get("bnn_confidence"), None),
            "bnn_direction": out.get("bnn_direction", "NA"),
        }
    except Exception:
        return default


def get_multi_agent_outputs(df: pd.DataFrame):
    default = {
        "trend_agent": "NA",
        "mean_reversion_agent": "NA",
        "volatility_agent": "NA",
        "final_agent_vote": "NA",
    }

    if ensemble_signal is None:
        return default

    try:
        out = ensemble_signal(df)
        return {
            "trend_agent": out.get("trend_agent", "NA"),
            "mean_reversion_agent": out.get("mean_reversion_agent", "NA"),
            "volatility_agent": out.get("volatility_agent", "NA"),
            "final_agent_vote": out.get("final_agent_vote", "NA"),
        }
    except Exception:
        return default


def compute_position_size(
    final_signal: str,
    baseline_prob_up: float | None,
    bnn_confidence: float | None,
    vix: float | None,
):
    if final_signal == "HOLD":
        return 0.0

    edge = 0.0
    if baseline_prob_up is not None:
        edge = abs(baseline_prob_up - 0.5) * 2.0  # 0 to 1 scale

    conf = 0.5 if bnn_confidence is None else max(0.0, min(1.0, bnn_confidence))

    vix_penalty = 1.0
    if vix is not None:
        if vix >= 24:
            vix_penalty = 0.25
        elif vix >= 18:
            vix_penalty = 0.5
        elif vix >= 13:
            vix_penalty = 0.75

    size = edge * conf * 0.02 * vix_penalty
    return round(min(max(size, 0.0), 0.02), 4)


def compute_levels(last_close: float, final_signal: str, volatility: float | None):
    if volatility is None:
        band = last_close * 0.003
    else:
        band = max(last_close * 0.002, last_close * min(max(volatility * 2.0, 0.001), 0.01))

    if final_signal == "LONG":
        buy_level = round(last_close, 2)
        sell_level = round(last_close + band, 2)
    elif final_signal == "SHORT":
        buy_level = round(last_close - band, 2)
        sell_level = round(last_close, 2)
    else:
        buy_level = None
        sell_level = None

    return buy_level, sell_level


def final_decision(
    regime: str,
    vix_label: str,
    baseline_sig: str,
    baseline_prob_up: float,
    rl_sig: str,
    trend_agent: str,
    mean_agent: str,
    vol_agent: str,
    final_agent_vote: str,
    lstm_direction: str,
    bnn_direction: str,
    bnn_std: float | None,
):
    warnings = []

    if vix_label == "panic":
        return "HOLD", "VIX panic regime", ["Panic volatility"]

    if regime == "high_vol":
        warnings.append("High volatility regime")

    if bnn_std is not None and bnn_std > 0.01:
        warnings.append("High forecast uncertainty")

    model_votes = {"LONG": 0, "SHORT": 0, "HOLD": 0}

    for sig in [baseline_sig, rl_sig, final_agent_vote, lstm_direction, bnn_direction]:
        if sig in model_votes:
            model_votes[sig] += 1

    # strong high-vol filter
    if vix_label == "elevated":
        if baseline_sig != rl_sig:
            return "HOLD", "High volatility disagreement", warnings + ["Baseline/RL disagreement"]
        if final_agent_vote not in ["NA", baseline_sig]:
            return "HOLD", "Agent conflict in elevated volatility", warnings + ["Multi-agent conflict"]

    # if everything is noisy and baseline is neutral, do nothing
    if baseline_sig == "HOLD" and rl_sig != "HOLD" and vix_label in ["elevated", "panic"]:
        return "HOLD", "Baseline neutral / RL aggressive in high vol", warnings

    # majority vote
    final_signal = max(model_votes, key=model_votes.get)
    reason = "Model vote consensus"

    # if tie-ish behavior or weak consensus, hold
    sorted_votes = sorted(model_votes.values(), reverse=True)
    if len(sorted_votes) >= 2 and sorted_votes[0] == sorted_votes[1]:
        return "HOLD", "No clear model consensus", warnings + ["Vote tie"]

    # sideways override
    if regime == "sideways" and final_signal != "HOLD":
        if abs(baseline_prob_up - 0.5) < 0.08:
            return "HOLD", "Sideways market with weak edge", warnings

    # elevated vol, only allow if baseline and RL agree
    if vix_label == "elevated" and not (baseline_sig == rl_sig == final_signal):
        return "HOLD", "Elevated volatility without full agreement", warnings

    return final_signal, reason, warnings


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

    # Sequence models
    lstm_out = get_lstm_outputs(seq_df)
    bnn_out = get_bnn_outputs(latest_live)

    # Add context for RL / agents
    latest_live["vix"] = 15.0 if vix is None else vix
    latest_live["vix_regime_code"] = vix_code
    latest_live["lstm_pred_return"] = lstm_out["lstm_pred_return"]
    latest_live["bnn_mean_return"] = bnn_out["bnn_mean_return"]
    latest_live["bnn_std_return"] = 0.01 if bnn_out["bnn_std_return"] is None else bnn_out["bnn_std_return"]
    latest_live["baseline_prob_up"] = baseline_prob_up

    # Single RL
    rl_sig = single_rl_signal(latest_live)

    # Multi-agent
    agent_out = get_multi_agent_outputs(latest_live)

    # Final decision
    regime = str(latest_live["regime"].iloc[-1]) if "regime" in latest_live.columns else "unknown"
    final_sig, reason, warnings = final_decision(
        regime=regime,
        vix_label=vix_label,
        baseline_sig=baseline_sig,
        baseline_prob_up=baseline_prob_up,
        rl_sig=rl_sig,
        trend_agent=agent_out["trend_agent"],
        mean_agent=agent_out["mean_reversion_agent"],
        vol_agent=agent_out["volatility_agent"],
        final_agent_vote=agent_out["final_agent_vote"],
        lstm_direction=lstm_out["lstm_direction"],
        bnn_direction=bnn_out["bnn_direction"],
        bnn_std=bnn_out["bnn_std_return"],
    )

    # Position size and levels
    position_size = compute_position_size(
        final_signal=final_sig,
        baseline_prob_up=baseline_prob_up,
        bnn_confidence=bnn_out["bnn_confidence"],
        vix=vix,
    )

    last_close = float(latest_live["close"].iloc[-1])
    volatility = safe_float(latest_live["volatility"].iloc[-1], None) if "volatility" in latest_live.columns else None
    buy_level, sell_level = compute_levels(last_close, final_sig, volatility)

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
        "buy_level": buy_level,
        "sell_level": sell_level,
        "position_size": position_size,
        "warnings": warnings,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))

    print("Saved live signal →", OUT_PATH)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run()