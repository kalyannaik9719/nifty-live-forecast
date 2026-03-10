import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

OUT_PATH = Path("data/raw/nse_live_quote.parquet")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}

session = requests.Session()
session.headers.update(HEADERS)


def warmup():
    url = "https://www.nseindia.com/market-data/live-equity-market"
    r = session.get(url, timeout=15)
    print("Warmup status:", r.status_code)


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def fetch_nifty():
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    r = session.get(url, timeout=15)

    if r.status_code != 200:
        raise Exception(f"NSE API error: {r.status_code}")

    data = r.json()

    # Debug once: print first row keys
    first_row = data["data"][0]
    print("Available keys in first row:")
    print(first_row.keys())

    # Use flexible field extraction
    close = first_row.get("lastPrice")
    open_price = first_row.get("open")

    # Some NSE responses use "dayHigh"/"dayLow" instead of "high"/"low"
    high = first_row.get("high", first_row.get("dayHigh"))
    low = first_row.get("low", first_row.get("dayLow"))

    change = first_row.get("change")
    pchange = first_row.get("pChange")

    row = {
        "ts": datetime.now(),
        "close": close,
        "open": open_price,
        "high": high,
        "low": low,
        "change": change,
        "pChange": pchange,
        "volume": 0
    }

    df = pd.DataFrame([row])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print("Saved NSE live quote →", OUT_PATH)
    print(df)


def run():
    warmup()
    fetch_nifty()


if __name__ == "__main__":
    run()