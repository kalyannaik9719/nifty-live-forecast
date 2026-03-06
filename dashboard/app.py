import streamlit as st
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="NIFTY50 Live Forecast", layout="centered")
st.title("NIFTY50 Live Forecast (Cloud Demo)")

SYMBOL = "^NSEI"
INTERVAL = "15m"
PERIOD = "5d"

@st.cache_data(ttl=60)
def fetch_data():
    return yf.download(SYMBOL, period=PERIOD, interval=INTERVAL, progress=False)

df = fetch_data()

if df is None or df.empty:
    st.error("No data received from Yahoo Finance right now.")
    st.stop()

last = df.iloc[-1]
now = datetime.now()

expected_high = float(last["High"]) * 1.002
expected_low = float(last["Low"]) * 0.998
expected_close = float(last["Close"]) * 1.001
trend = "Bullish" if expected_close >= float(last["Close"]) else "Bearish"

st.subheader("Forecast Output")
st.json({
    "ts": now.isoformat(),
    "horizon": "1h",
    "expected_high": expected_high,
    "expected_low": expected_low,
    "expected_close": expected_close,
    "trend": trend,
    "confidence": 0.65
})