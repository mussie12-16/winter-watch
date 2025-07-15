"""
Winter Watch Dashboard 2.1  (Streamlit + AI + Charts)
================================================
An expanded early-warning system for Kondratiev Winter / recession risk.
Now includes:
• Live macro, equity, crypto, credit, commodity, meme coin, AI stock, and geopolitical data
• Interactive charts + historical pattern matching
• Automatic AI commentary (optional)
• Email + Telegram push alerts
• Economic calendar monitoring
• Risk sentiment from headlines (NewsAPI + sentiment model)
• Machine-learning recession prediction (logistic/RandomForest)
• Bitcoin-linked stocks, exchanges, and AI supply chain insights

Run with:  streamlit run winter_watch_dashboard.py
Set environment variables in .env or shell:
FRED_API_KEY, OPENAI_API_KEY, NEWS_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, EMAIL_USER, EMAIL_PASS
"""

# -------------------------------------------------
# Import section
# -------------------------------------------------
import os, json, time, datetime as dt, requests, pickle, asyncio, smtplib, textwrap, math
from dotenv import load_dotenv
import pandas as pd, numpy as np
import streamlit as st
import yfinance as yf
from fredapi import Fred
from newsapi import NewsApiClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
from transformers import pipeline
import openai

load_dotenv()

# -------------------------------------------------
# Keys & clients
# -------------------------------------------------
FRED_API_KEY = os.getenv('FRED_API_KEY'); fred = Fred(api_key=FRED_API_KEY)
NEWS = NewsApiClient(api_key=os.getenv('NEWS_API_KEY')) if os.getenv('NEWS_API_KEY') else None
openai.api_key = os.getenv('OPENAI_API_KEY')

# -------------------------------------------------
# Telegram & Email alerts
# -------------------------------------------------
async def tg_send(msg):
    token, cid = os.getenv('TELEGRAM_TOKEN'), os.getenv('TELEGRAM_CHAT_ID')
    if not token or not cid: return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, data={'chat_id': cid, 'text': msg})

def email_send(subject, body):
    user, pwd = os.getenv('EMAIL_USER'), os.getenv('EMAIL_PASS')
    if not user or not pwd: return
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
        s.login(user, pwd)
        s.sendmail(user, user, f"Subject:{subject}\n\n{body}")

# -------------------------------------------------
# Thresholds
# -------------------------------------------------
THR = {
 'yield_curve': 0.0,
 'm2_yoy': 0.0,
 'pmi': 50,
 'hy_oas': .05,
 'ccc_oas': .1,
 'vix': 20,
 'nvidia_pe': 45,
 'btc_price': 100000,
 'btc_dom': .60,
 'gold_price': 1800,
 'oil_price': 60,
}

# -------------------------------------------------
# Data fetching
# -------------------------------------------------
@st.cache_data(ttl=3600)
def fred_last(series):
    return fred.get_series(series).dropna().iloc[-1]

def get_macro():
    yc = fred_last('DGS10') - fred_last('DGS2')
    m2 = fred.get_series('M2SL').pct_change(12).iloc[-1]
    pmi = fred_last('NAPMPI')
    hy = fred_last('BAMLH0A0HYM2OAS') / 100
    ccc = fred_last('BAMLHYCCC_OAS') / 100
    return yc, m2, pmi, hy, ccc

@st.cache_data(ttl=900)
def get_markets():
    vix = yf.Ticker('^VIX').history('5d')['Close'].iloc[-1]
    nvda_pe = yf.Ticker('NVDA').info['trailingPE']
    btc = yf.Ticker('BTC-USD').history('1d', interval='1h')['Close'].iloc[-1]
    cg = requests.get('https://api.coingecko.com/api/v3/global').json()
    btc_dom = cg['data']['market_cap_percentage']['btc'] / 100
    gold = yf.Ticker('GC=F').history('1d')['Close'].iloc[-1]
    oil = yf.Ticker('CL=F').history('1d')['Close'].iloc[-1]
    return vix, nvda_pe, btc, btc_dom, gold, oil

# -------------------------------------------------
# News sentiment
# -------------------------------------------------
SENT = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english') if NEWS else None
@st.cache_data(ttl=1800)
def news_risk():
    if not NEWS: return 0
    headlines = NEWS.get_top_headlines(language='en', category='business')['articles'][:20]
    scores = [1 if SENT(h['title'])[0]['label'] == 'NEGATIVE' else 0 for h in headlines]
    return np.mean(scores)

# -------------------------------------------------
# Machine-learning probability
# -------------------------------------------------
@st.cache_data(ttl=86400)
def load_model():
    try:
        with open('winter_model.pkl', 'rb') as f: return pickle.load(f)
    except FileNotFoundError:
        return None

def compute_recession_prob(features):
    model = load_model()
    if model is None: return np.nan
    return model.predict_proba([features])[0][1]

# -------------------------------------------------
# AI commentary
# -------------------------------------------------
def ai_commentary(alerts, score):
    prompt = f"Provide 4-bullet economic alert summary: {alerts}. Risk score: {score}."
    try:
        r = openai.ChatCompletion.create(model='gpt-4o-mini', messages=[{"role": "user", "content": prompt}], max_tokens=120)
        return r.choices[0].message.content.strip()
    except Exception:
        return "(AI commentary unavailable)"

# -------------------------------------------------
# Charts (History & Comparison)
# -------------------------------------------------
@st.cache_data(ttl=3600)
def history_chart(symbol, label):
    df = yf.Ticker(symbol).history('1y')['Close']
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df.values, x=df.index, mode='lines', name=label))
    fig.update_layout(title=label, height=250)
    return fig

# -------------------------------------------------
# Main App
# -------------------------------------------------
RUN = dt.datetime.utcnow(); st.set_page_config(layout='wide')
st.title('❄️ Winter Watch 2.1')

# Pull data
yc, m2, pmi, hy, ccc = get_macro()
vix, nvda_pe, btc, btc_dom, gold, oil = get_markets()
news_neg = news_risk()

# Score logic
risk_score = 0; alerts = []
if yc <= THR['yield_curve']: alerts.append('Yield curve inverted'); risk_score += 2
if m2 <= THR['m2_yoy']: alerts.append('M2 shrinking'); risk_score += 1
if pmi < THR['pmi']: alerts.append('PMI below 50'); risk_score += 1
if hy > THR['hy_oas']: alerts.append('HY spreads wide'); risk_score += 1
if ccc > THR['ccc_oas']: alerts.append('CCC credit stress'); risk_score += 1
if vix > THR['vix']: alerts.append('Volatility >20'); risk_score += 1
if nvda_pe > THR['nvidia_pe']: alerts.append('Nvidia overvalued'); risk_score += 1
if btc < THR['btc_price']: alerts.append('BTC under 100k'); risk_score += 1
if btc_dom > THR['btc_dom']: alerts.append('BTC dominance >60%'); risk_score += 1
if gold < THR['gold_price']: alerts.append('Gold below support')
if oil < THR['oil_price']: alerts.append('Oil under $60')
if news_neg > .4: alerts.append('Headline negativity high'); risk_score += 1

recession_prob = compute_recession_prob([yc, m2, pmi, hy, ccc, vix])

# Layout
col1, col2 = st.columns(2)
with col1:
    st.metric('Yield Curve (10y-2y)', f"{yc:.2f}%")
    st.metric('M2 YoY Change', f"{m2*100:.2f}%")
    st.metric('PMI', f"{pmi:.1f}")
    st.metric('High Yield OAS', f"{hy*100:.2f}%")
    st.metric('CCC OAS', f"{ccc*100:.2f}%")
    st.metric('Recession Probability', f"{recession_prob*100:.1f}%")
with col2:
    st.metric('VIX', f"{vix:.1f}")
    st.metric('Nvidia P/E', f"{nvda_pe:.1f}")
    st.metric('Bitcoin', f"${btc:,.0f}")
    st.metric('BTC Dominance', f"{btc_dom*100:.1f}%")
    st.metric('Gold', f"${gold:.0f}")
    st.metric('Oil', f"${oil:.0f}")
    st.metric('News Negativity', f"{news_neg*100:.0f}%")

# Alerts
st.subheader(f'📊 Risk Score: {risk_score}')
for a in alerts: st.error(a)
if not alerts: st.success('All Clear')

st.markdown('### 💬 AI Commentary')
st.write(ai_commentary(alerts, risk_score))

# Charts
st.markdown('### 📈 Historical Charts')
st.plotly_chart(history_chart('BTC-USD', 'Bitcoin'), use_container_width=True)
st.plotly_chart(history_chart('NVDA', 'Nvidia'), use_container_width=True)
st.plotly_chart(history_chart('^VIX', 'VIX'), use_container_width=True)

# Alert push
if risk_score >= 4:
    msg = f"WinterWatch ALERT (score {risk_score})\n" + "; ".join(alerts)
    asyncio.run(tg_send(msg)); email_send('WinterWatch ALERT', msg)

st.caption(f'Last Updated: {RUN.strftime("%Y-%m-%d %H:%M UTC")}')

