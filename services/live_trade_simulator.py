import json
from pathlib import Path
from datetime import datetime
import pandas as pd

SIGNAL_PATH = Path("data/processed/live_signal.json")
LEDGER_PATH = Path("data/processed/live_trade_ledger.parquet")


def main():
    signal = json.loads(SIGNAL_PATH.read_text())

    row = {
        "ts": datetime.now(),
        "last_close": signal["last_close"],
        "regime": signal["regime"],
        "vix": signal["vix"],
        "baseline_signal": signal["baseline_signal"],
        "rl_signal": signal["rl_signal"],
        "final_signal": signal["final_signal"],
        "reason": signal["reason"],
        "buy_level": signal["buy_level"],
        "sell_level": signal["sell_level"],
        "warnings": "; ".join(signal.get("warnings", [])),
    }

    if LEDGER_PATH.exists():
        df = pd.read_parquet(LEDGER_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_parquet(LEDGER_PATH, index=False)
    print("Updated live trade ledger →", LEDGER_PATH)
    print(df.tail())


if __name__ == "__main__":
    main()