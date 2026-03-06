import streamlit as st
import requests

st.title("NIFTY50 Live Forecast")

res = requests.get("http://localhost:8000/forecast/latest")

if res.status_code == 200:
    st.json(res.json())
else:
    st.write("No data yet.")