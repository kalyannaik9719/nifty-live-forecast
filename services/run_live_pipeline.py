from services.nse_live_fetch import run as run_fetch
from services.live_feature_builder import build_live_features
from services.live_signal_engine import run as run_signal

if __name__ == "__main__":
    run_fetch()
    build_live_features()
    run_signal()
    print("Live pipeline complete.")