import pandas as pd
import joblib
from pathlib import Path

DATA_PATH = Path("data/processed/nifty_regimes.parquet")
MODEL_PATH = Path("data/processed/baseline_model.pkl")
OUT_PATH = Path("data/processed/paper_trades.parquet")

FEATURES = [
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
    "minute"
]

COST_PER_TRADE = 0.0005  # 0.05%

def load_inputs():
    df = pd.read_parquet(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    return df, model

def simulate(df, model):
    df = df.copy()

    # prediction probability for "up"
    probs = model.predict_proba(df[FEATURES])[:, 1]
    df["prob_up"] = probs

    # basic policy
    # >0.55 => long
    # <0.45 => short
    # else => flat
    df["position"] = 0
    df.loc[df["prob_up"] > 0.55, "position"] = 1
    df.loc[df["prob_up"] < 0.45, "position"] = -1

    # next candle return
    df["future_ret"] = df["close"].shift(-1) / df["close"] - 1

    # trade change penalty
    df["trade_change"] = df["position"].diff().fillna(0).abs()

    # strategy return
    df["strategy_ret"] = (
        df["position"] * df["future_ret"]
        - df["trade_change"] * COST_PER_TRADE
    )

    # cumulative equity curve
    df["equity_curve"] = (1 + df["strategy_ret"].fillna(0)).cumprod()

    return df.dropna().reset_index(drop=True)

def save_results(df):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print("Paper trades saved →", OUT_PATH)
    print("Final Equity:", round(df["equity_curve"].iloc[-1], 4))
    print("Total Trades:", int((df["trade_change"] > 0).sum()))
    print("Long Signals:", int((df["position"] == 1).sum()))
    print("Short Signals:", int((df["position"] == -1).sum()))
    print("Flat Signals:", int((df["position"] == 0).sum()))

if __name__ == "__main__":
    df, model = load_inputs()
    out = simulate(df, model)
    save_results(out)