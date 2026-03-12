import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

OUT_PATH = Path("data/raw/nifty_option_chain.parquet")

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/option-chain",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

def fetch_option_chain():
    session = requests.Session()
    session.headers.update(HEADERS)

    # Warm up cookies
    warm = session.get("https://www.nseindia.com/option-chain", timeout=20)
    print("Warmup status:", warm.status_code)

    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    r = session.get(url, timeout=20)
    print("API status:", r.status_code)

    raw = r.json()
    print("Top-level keys:", list(raw.keys()))

    # Save raw debug copy so you can inspect what NSE actually returned
    debug_path = Path("data/raw/nifty_option_chain_raw.json")
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(json.dumps(raw, indent=2))
    print("Saved raw JSON debug →", debug_path)

    if "records" not in raw or "data" not in raw["records"]:
        raise ValueError(
            f"Unexpected NSE option-chain structure. Top-level keys: {list(raw.keys())}. "
            f"Inspect {debug_path}."
        )

    records = raw["records"]["data"]

    rows = []
    for item in records:
        strike = item.get("strikePrice")
        ce = item.get("CE", {})
        pe = item.get("PE", {})

        rows.append({
            "ts": datetime.now(),
            "strike": strike,
            "ce_oi": ce.get("openInterest", 0),
            "ce_change_oi": ce.get("changeinOpenInterest", 0),
            "ce_iv": ce.get("impliedVolatility", 0),
            "ce_ltp": ce.get("lastPrice", 0),
            "pe_oi": pe.get("openInterest", 0),
            "pe_change_oi": pe.get("changeinOpenInterest", 0),
            "pe_iv": pe.get("impliedVolatility", 0),
            "pe_ltp": pe.get("lastPrice", 0),
        })

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print("Saved option chain →", OUT_PATH)
    print(df.head())

if __name__ == "__main__":
    fetch_option_chain()