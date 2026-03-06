from fastapi import FastAPI
from sqlalchemy import desc
from app.db import Base, engine, SessionLocal
from app.models import Forecast, NiftyCandle

app = FastAPI(title="NIFTY50 Live Forecast API")

Base.metadata.create_all(bind=engine)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/forecast/latest")
def latest_forecast():
    db = SessionLocal()
    try:
        row = db.query(Forecast).order_by(desc(Forecast.ts)).first()
        if not row:
            return {"message": "No forecasts yet. Run worker/updater.py"}
        return {
            "ts": row.ts,
            "horizon": row.horizon,
            "expected_high": row.expected_high,
            "expected_low": row.expected_low,
            "expected_close": row.expected_close,
            "trend": row.trend,
            "confidence": row.confidence,
        }
    finally:
        db.close()