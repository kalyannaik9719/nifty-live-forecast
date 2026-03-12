import time
from datetime import datetime
import subprocess
import sys

TASKS = [
    ["python", "services/nse_live_fetch.py"],
    ["python", "services/live_feature_builder.py"],
    ["python", "services/nse_vix_fetch.py"],
    ["python", "services/live_signal_engine.py"],
]

def run_task(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Task failed: {' '.join(cmd)}")

def main():
    while True:
        print(f"\n[{datetime.now().isoformat()}] Starting live cycle...")
        try:
            for task in TASKS:
                run_task(task)
            print("Live cycle completed.")
        except Exception as e:
            print("Live worker error:", e)

        time.sleep(60)

if __name__ == "__main__":
    main()