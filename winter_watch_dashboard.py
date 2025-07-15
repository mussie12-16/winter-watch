
# Winter Watch Dashboard – With Charts + Pattern Matching

import os, datetime as dt, requests, math
from dotenv import load_dotenv
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from transformers import pipeline
from fredapi import Fred
from newsapi import NewsApiClient
import plotly.graph_objects as go

load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')
fred = Fred(api_key=FRED_API_KEY)
NEWS = NewsApiClient(api_key=os.getenv('NEWS_API_KEY')) if os.getenv('NEWS_API_KEY') else None

st.set_page_config(layout='wide')
st.title("❄️ Winter Watch Dashboard – Charts + Pattern Matching")

@st.cache_data(ttl=3600)
def fred_hist(series, start='1990-01-01'):
    return fred.get_series(series, start_date=start).dropna()

@st.cache_data(ttl=3600)
def get_macro():
    yc_hist = fred_hist('DGS10') - fred_hist('DGS2')
    m2_hist = fred_hist('M2SL').pct_change(12)
    pmi_hist = fred_hist('NAPMPI')
    hy_hist = fred_hist('BAMLH0A0HYM2OAS') / 100
    ccc_hist = fred_hist('BAMLHYCCC_OAS') / 100
    return yc_hist, m2_hist, pmi_hist, hy_hist, ccc_hist

@st.cache_data(ttl=1800)
def get_assets():
    tickers = {
        "VIX": "^VIX", "BTC": "BTC-USD", "NVIDIA": "NVDA",
        "Coinbase": "COIN", "MicroStrategy": "MSTR", "DOGE": "DOGE-USD"
    }
    data = {}
    for label, ticker in tickers.items():
        try:
            hist = yf.Ticker(ticker).history(period="5y")["Close"]
            data[label] = hist.dropna()
        except:
            data[label] = pd.Series(dtype=float)
    return data

@st.cache_data(ttl=1800)
def news_risk():
    if not NEWS: return 0
    SENT = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    headlines = NEWS.get_top_headlines(language='en', category='business')['articles'][:20]
    scores = [1 if SENT(h['title'])[0]['label']=='NEGATIVE' else 0 for h in headlines]
    return np.mean(scores)

# -- Compare current levels to past Winter cycles
def pattern_match(series, ref_periods):
    recent = series[-30:].mean()
    for label, ref in ref_periods.items():
        if abs(recent - ref.mean()) / ref.mean() < 0.15:
            return f"Matches {label} pattern"
    return None

# -- Load macro + asset data
yc_hist, m2_hist, pmi_hist, hy_hist, ccc_hist = get_macro()
assets = get_assets()
news_neg = news_risk()

# Historical reference periods
ref_periods = {
    "2008": yc_hist['2007-08':'2008-12'],
    "2000": yc_hist['2000-01':'2001-12'],
    "1974": yc_hist['1973-01':'1975-01'] if len(yc_hist.index) > 10000 else yc_hist.iloc[:0]  # Fallback if no data
}

# --- Risk signals
risk_score = 0
alerts = []

if yc_hist.iloc[-1] < 0: alerts.append("Inverted yield curve"); risk_score += 2
if m2_hist.iloc[-1] < 0: alerts.append("M2 YoY negative"); risk_score += 1
if pmi_hist.iloc[-1] < 50: alerts.append("PMI < 50"); risk_score += 1
if hy_hist.iloc[-1] > 0.05: alerts.append("High-yield spread > 5%"); risk_score += 1
if ccc_hist.iloc[-1] > 0.1: alerts.append("CCC spread > 10%"); risk_score += 1
if news_neg > 0.4: alerts.append("Negative news headlines"); risk_score += 1

# Pattern match alerts
pmatch = pattern_match(yc_hist, ref_periods)
if pmatch: alerts.append(f"Yield curve pattern: {pmatch}"); risk_score += 1

# Layout
st.subheader("📉 Macro Indicators")
col1, col2 = st.columns(2)
col1.metric("Yield Curve (10y-2y)", f"{yc_hist.iloc[-1]:.2f}%")
col1.metric("M2 YoY Growth", f"{m2_hist.iloc[-1]*100:.2f}%")
col1.metric("PMI", f"{pmi_hist.iloc[-1]:.1f}")
col2.metric("HY OAS", f"{hy_hist.iloc[-1]*100:.2f}%")
col2.metric("CCC OAS", f"{ccc_hist.iloc[-1]*100:.2f}%")
col2.metric("News Negativity", f"{news_neg*100:.0f}%")

st.markdown("### 🔔 Alerts")
for a in alerts: st.error(a)
if not alerts: st.success("No critical signals")

st.markdown("### 📊 Macro Charts")
st.plotly_chart(go.Figure().add_trace(go.Scatter(x=yc_hist.index, y=yc_hist.values, name="Yield Curve")), use_container_width=True)
st.plotly_chart(go.Figure().add_trace(go.Scatter(x=m2_hist.index, y=m2_hist.values*100, name="M2 YoY %")), use_container_width=True)
st.plotly_chart(go.Figure().add_trace(go.Scatter(x=pmi_hist.index, y=pmi_hist.values, name="PMI")), use_container_width=True)

st.markdown("### 📈 Assets Over Time")
for name, series in assets.items():
    if not series.empty:
        fig = go.Figure().add_trace(go.Scatter(x=series.index, y=series.values, name=name))
        fig.update_layout(title=name, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

st.caption(f"Updated: {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
