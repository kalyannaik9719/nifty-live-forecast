from sqlalchemy import Column, Integer, Float, DateTime, String
from app.db import Base

class NiftyCandle(Base):
    __tablename__ = "nifty_candles"

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime, index=True, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)

class Forecast(Base):
    __tablename__ = "nifty_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime, index=True, nullable=False)
    horizon = Column(String, nullable=False)
    expected_high = Column(Float, nullable=False)
    expected_low = Column(Float, nullable=False)
    expected_close = Column(Float, nullable=False)
    trend = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)