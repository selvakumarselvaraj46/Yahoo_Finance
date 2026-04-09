# ========================================
# Yahoo Finance Real-Time Stock Dashboard
# ========================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
import numpy as np
from streamlit_autorefresh import st_autorefresh


# PAGE CONFIG
st.set_page_config(
    page_title="Yahoo Finance Dashboard",
    page_icon="📈",
    layout="wide",
)

# CONSTANTS
REFRESH_TTL = 1800
DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA"]

# AUTO REFRESH
refresh_min = st.sidebar.slider("Auto-refresh (min)", 5, 60, 30)
st_autorefresh(interval=refresh_min * 60000, key="refresh")


# DATA FUNCTIONS
@st.cache_data(ttl=REFRESH_TTL)
def get_hist(ticker, period):
    df = yf.download(ticker, period=period, interval="1d", progress=False)

    if df.empty:
        return df

    # Fix Multi Index (Yahoo sometimes returns multi index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(7).std() * 100
    df["MA7"] = df["Close"].rolling(7).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    return df


@st.cache_data(ttl=REFRESH_TTL)
def get_multi(tickers, period):
    data = {}
    for t in tickers:
        df = get_hist(t, period)
        if not df.empty:
            data[t] = df
    return data


def detect_anomalies(df):
    df = df.copy()

    mu = df["Returns"].mean()
    sigma = df["Returns"].std()

    if sigma == 0 or pd.isna(sigma):
        df["z"] = 0
        df["anom"] = False
        return df

    df["z"] = (df["Returns"] - mu) / sigma
    df["anom"] = df["z"].abs() > 2.5

    return df


# SIDEBAR
st.sidebar.title("📈 Settings")

chosen = st.sidebar.multiselect(
    "Select Tickers",
    DEFAULT_TICKERS,
    default=["AAPL", "MSFT"]
)

custom = st.sidebar.text_input("Add ticker")

if custom.strip():
    t = custom.strip().upper()
    if t not in chosen:
        chosen.append(t)

period = st.sidebar.selectbox(
    "Period",
    ["1mo", "3mo", "6mo", "1y"]
)

# LOAD DATA
if not chosen:
    st.warning("Select at least one ticker")
    st.stop()

data = get_multi(chosen, period)

if not data:
    st.error("No data found")
    st.stop()

# KPI
st.title("📊 Stock Dashboard")

cols = st.columns(len(data))

for col, (ticker, df) in zip(cols, data.items()):

    price = float(df["Close"].iloc[-1])

    if len(df) > 1:
        prev = float(df["Close"].iloc[-2])
    else:
        prev = price

    change = price - prev

    if prev is not None and prev != 0:
        pct = (change / prev) * 100
    else:
        pct = 0

    col.metric(
        ticker,
        f"${price:.2f}",
        f"{pct:.2f}%"
    )

# CHART
focus = list(data.keys())[0]
df = data[focus]

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05
)

fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MA7"],
        name="MA7"
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MA20"],
        name="MA20"
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume"
    ),
    row=2,
    col=1
)

st.plotly_chart(fig, use_container_width=True)

# ANOMALIES
st.subheader("⚠️ Anomalies")

df_a = detect_anomalies(df)
anom = df_a[df_a["anom"]]

if anom.empty:
    st.success("No anomalies found")
else:
    for i, row in anom.tail(5).iterrows():
        st.write(f"{i.date()} → {row['Returns']*100:.2f}%")


# SUMMARY TABLE
rows = []

for ticker, df in data.items():

    price = float(df["Close"].iloc[-1])

    ret = (
        (df["Close"].iloc[-1] /
         df["Close"].iloc[0] - 1) * 100
    )

    vol = (
        df["Returns"].std() *
        np.sqrt(252) * 100
    )

    rows.append({
        "Ticker": ticker,
        "Price": round(price, 2),
        "Return %": round(ret, 2),
        "Volatility %": round(vol, 2)
    })

summary = pd.DataFrame(rows)

st.subheader("📊 Summary")
st.dataframe(summary, use_container_width=True)
