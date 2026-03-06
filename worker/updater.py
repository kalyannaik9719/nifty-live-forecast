import yfinance as yf
from datetime import datetime
from app.db import Base, engine, SessionLocal
from app.models import NiftyCandle, Forecast

Base.metadata.create_all(bind=engine)

SYMBOL = "^NSEI"

def run():
    df = yf.download(SYMBOL, period="5d", interval="15m")

    if df.empty:
        print("No data")
        return

    last = df.iloc[-1]
    now = datetime.now()

    db = SessionLocal()

    candle = NiftyCandle(
        ts=now,
        open=float(last["Open"]),
        high=float(last["High"]),
        low=float(last["Low"]),
        close=float(last["Close"]),
        volume=float(last["Volume"])
    )
    db.add(candle)

    forecast = Forecast(
        ts=now,
        horizon="1h",
        expected_high=float(last["High"]) * 1.002,
        expected_low=float(last["Low"]) * 0.998,
        expected_close=float(last["Close"]) * 1.001,
        trend="Bullish",
        confidence=0.65
    )
    db.add(forecast)

    db.commit()
    db.close()

    print("Forecast saved successfully")

if __name__ == "__main__":
    run()