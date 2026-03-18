import time
import os

while True:
    os.system("python services/nse_live_fetch.py")
    os.system("python -m services.live_feature_builder")
    os.system("python services/nse_vix_fetch.py")
    os.system("python services/gamma_levels.py")
    os.system("python services/live_signal_engine.py")
    time.sleep(60)