import time
from datetime import datetime
from services.nse_live_fetch import run as run_fetch
from services.live_feature_builder import build_live_features
from services.live_signal_engine import run as run_signal


def job():
    print(f"\n[{datetime.now().isoformat()}] Running live pipeline...")
    try:
        run_fetch()
        build_live_features()
        run_signal()
        print("Cycle complete.")
    except Exception as e:
        print("Live worker error:", e)


if __name__ == "__main__":
    while True:
        job()
        time.sleep(60)  # every 60 seconds