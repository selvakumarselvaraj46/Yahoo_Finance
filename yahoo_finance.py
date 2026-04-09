# ========================================
# ULTIMATE AI STOCK + CRYPTO DASHBOARD
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Ultimate AI Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# ─────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────
US_STOCKS = ["AAPL","MSFT","TSLA","NVDA"]
INDIAN_STOCKS = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS"]
CRYPTO = ["BTC-USD","ETH-USD","SOL-USD","DOGE-USD"]

REFRESH_TTL = 600

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Dashboard Settings")

refresh = st.sidebar.slider("Auto Refresh",30,600,120)
st_autorefresh(interval=refresh*1000)

market = st.sidebar.selectbox(
    "Select Market",
    ["US Stocks","Indian Stocks","Crypto","Mixed"]
)

if market == "US Stocks":
    tickers = st.sidebar.multiselect("Stocks",US_STOCKS,default=US_STOCKS[:2])

elif market == "Indian Stocks":
    tickers = st.sidebar.multiselect("Stocks",INDIAN_STOCKS,default=INDIAN_STOCKS[:2])

elif market == "Crypto":
    tickers = st.sidebar.multiselect("Crypto",CRYPTO,default=CRYPTO[:2])

else:
    tickers = st.sidebar.multiselect(
        "Mixed",
        US_STOCKS+INDIAN_STOCKS+CRYPTO,
        default=["AAPL","RELIANCE.NS","BTC-USD"]
    )

custom = st.sidebar.text_input("Add Custom Symbol")

if custom:
    tickers.append(custom.upper())

period = st.sidebar.selectbox(
    "Period",
    ["1mo","3mo","6mo","1y","2y","5y"]
)

# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def RSI(data,window=14):
    delta=data.diff()
    gain=(delta.where(delta>0,0)).rolling(window).mean()
    loss=(-delta.where(delta<0,0)).rolling(window).mean()
    rs=gain/loss
    return 100-(100/(1+rs))

def MACD(data):
    exp1=data.ewm(span=12).mean()
    exp2=data.ewm(span=26).mean()
    macd=exp1-exp2
    signal=macd.ewm(span=9).mean()
    return macd,signal

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data(ttl=REFRESH_TTL)
def load_data(ticker):

    df=yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if isinstance(df.columns,pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)

    df["MA20"]=df["Close"].rolling(20).mean()
    df["MA50"]=df["Close"].rolling(50).mean()

    df["Returns"]=df["Close"].pct_change()
    df["Volatility"]=df["Returns"].rolling(7).std()

    df["RSI"]=RSI(df["Close"])

    macd,signal=MACD(df["Close"])
    df["MACD"]=macd
    df["Signal"]=signal

    df["Upper"]=df["MA20"]+2*df["Close"].rolling(20).std()
    df["Lower"]=df["MA20"]-2*df["Close"].rolling(20).std()

    return df


data={}

for t in tickers:
    df=load_data(t)
    if not df.empty:
        data[t]=df

if not data:
    st.error("No Data Found")
    st.stop()

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.title("🚀 Ultimate AI Trading Dashboard")

# ─────────────────────────────────────────────
# KPI
# ─────────────────────────────────────────────
cols=st.columns(len(data))

for col,(ticker,df) in zip(cols,data.items()):

    price=df["Close"].iloc[-1]
    prev=df["Close"].iloc[-2]

    pct=(price-prev)/prev*100

    col.metric(
        ticker,
        f"{price:.2f}",
        f"{pct:.2f}%"
    )

# ─────────────────────────────────────────────
# AI SIGNAL
# ─────────────────────────────────────────────
st.subheader("🤖 AI Buy / Sell Signals")

signals=[]

for ticker,df in data.items():

    rsi=df["RSI"].iloc[-1]
    macd=df["MACD"].iloc[-1]
    signal=df["Signal"].iloc[-1]

    if rsi<30 and macd>signal:
        rec="BUY"
    elif rsi>70 and macd<signal:
        rec="SELL"
    else:
        rec="HOLD"

    signals.append({
        "Ticker":ticker,
        "RSI":round(rsi,2),
        "Recommendation":rec
    })

signal_df=pd.DataFrame(signals)

st.dataframe(signal_df,use_container_width=True)

# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────
focus=st.selectbox("Select Chart",list(data.keys()))
df=data[focus]

fig=make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.5,0.2,0.15,0.15]
)

fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    ),
    row=1,col=1
)

fig.add_trace(go.Scatter(x=df.index,y=df["MA20"],name="MA20"),row=1,col=1)
fig.add_trace(go.Scatter(x=df.index,y=df["MA50"],name="MA50"),row=1,col=1)

fig.add_trace(go.Bar(x=df.index,y=df["Volume"]),row=2,col=1)

fig.add_trace(go.Scatter(x=df.index,y=df["RSI"]),row=3,col=1)

fig.add_trace(go.Scatter(x=df.index,y=df["MACD"]),row=4,col=1)
fig.add_trace(go.Scatter(x=df.index,y=df["Signal"]),row=4,col=1)

fig.update_layout(height=900,xaxis_rangeslider_visible=False)

st.plotly_chart(fig,use_container_width=True)

# ─────────────────────────────────────────────
# PORTFOLIO TRACKER
# ─────────────────────────────────────────────
st.subheader("💼 Portfolio Tracker")

portfolio=[]

for ticker,df in data.items():

    qty=st.number_input(
        f"{ticker} Quantity",
        value=0,
        key=ticker
    )

    price=df["Close"].iloc[-1]

    value=qty*price

    portfolio.append({
        "Ticker":ticker,
        "Qty":qty,
        "Price":round(price,2),
        "Value":round(value,2)
    })

port_df=pd.DataFrame(portfolio)

st.dataframe(port_df)

st.metric(
    "Total Portfolio Value",
    round(port_df["Value"].sum(),2)
)

# ─────────────────────────────────────────────
# RETURNS
# ─────────────────────────────────────────────
st.subheader("📈 Returns")

ret=pd.DataFrame()

for t,df in data.items():
    ret[t]=df["Close"]/df["Close"].iloc[0]

st.line_chart(ret)

# ─────────────────────────────────────────────
# CORRELATION
# ─────────────────────────────────────────────
st.subheader("🔥 Correlation")

corr=ret.corr()

fig_corr=px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu"
)

st.plotly_chart(fig_corr)

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
rows=[]

for t,df in data.items():

    price=df["Close"].iloc[-1]
    first=df["Close"].iloc[0]

    ret=(price/first-1)*100
    vol=df["Returns"].std()*np.sqrt(252)*100

    rows.append({
        "Ticker":t,
        "Price":round(price,2),
        "Return %":round(ret,2),
        "Volatility":round(vol,2)
    })

summary=pd.DataFrame(rows)

st.subheader("📊 Summary")

st.dataframe(summary)

st.download_button(
    "Download CSV",
    summary.to_csv().encode(),
    "stocks.csv"
)

st.success("Ultimate Dashboard Ready 🚀")
