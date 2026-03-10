from __future__ import annotations

import pandas as pd
from pathlib import Path

HIST_PATH = Path("data/processed/nifty_regimes.parquet")
LIVE_QUOTE_PATH = Path("data/raw/nse_live_quote.parquet")
OUT_PATH = Path("data/processed/live_features.parquet")


def make_naive(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    try:
        return s.dt.tz_localize(None)
    except TypeError:
        return s


def build_live_features():
    hist = pd.read_parquet(HIST_PATH).copy()
    live = pd.read_parquet(LIVE_QUOTE_PATH).copy()

    live = live.rename(columns={"lastPrice": "close"})

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
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    live_row.to_parquet(OUT_PATH, index=False)

    print("Saved live features →", OUT_PATH)
    print(live_row.to_dict(orient="records")[0])


if __name__ == "__main__":
    build_live_features()