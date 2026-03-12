import schedule
import time
import os

def retrain():

    print("Retraining RL model")

    os.system("python -m models.train_lstm")
    os.system("python -m models.train_bnn")
    os.system("python -m rl.train_rl")

    print("Retraining complete")

schedule.every().sunday.at("03:00").do(retrain)

while True:
    schedule.run_pending()
    time.sleep(60)