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

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Yahoo Finance Dashboard",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
REFRESH_TTL = 1800
DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA"]

# ─────────────────────────────────────────────
# DATA FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_data(ttl=REFRESH_TTL)
def get_hist(ticker, period):
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        return df

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["Close"] = df["Close"].squeeze()

    # Basic indicators
    df["Returns"]   = df["Close"].pct_change()
    df["Volatility"]= df["Returns"].rolling(7).std() * 100
    df["MA7"]       = df["Close"].rolling(7).mean()
    df["MA20"]      = df["Close"].rolling(20).mean()
    df["MA50"]      = df["Close"].rolling(50).mean()

    # ── RSI (14-period) ──────────────────────
    delta  = df["Close"].diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=13, adjust=False).mean()
    avg_l  = loss.ewm(com=13, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── MACD (12 / 26 / 9) ───────────────────
    ema12        = df["Close"].ewm(span=12, adjust=False).mean()
    ema26        = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]   = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"]   = df["MACD"] - df["Signal"]

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
    returns = df["Returns"].dropna()
    if len(returns) < 2:
        df["z"] = 0.0
        df["anom"] = False
        return df
    mu    = returns.mean()
    sigma = returns.std()
    if sigma == 0 or pd.isna(sigma):
        df["z"] = 0.0
        df["anom"] = False
        return df
    df["z"]    = (df["Returns"] - mu) / sigma
    df["anom"] = df["z"].abs() > 2.5
    return df


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("📈 Settings")

chosen = st.sidebar.multiselect(
    "Select Tickers", DEFAULT_TICKERS, default=["AAPL", "MSFT"]
)

custom = st.sidebar.text_input("Add ticker (e.g. GOOG)")
if custom.strip():
    t = custom.strip().upper()
    if t not in chosen:
        chosen.append(t)

period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y"])

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
if not chosen:
    st.warning("Select at least one ticker.")
    st.stop()

data = get_multi(chosen, period)

if not data:
    st.error("No data found. Check ticker symbols.")
    st.stop()

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
st.title("📊 Stock Dashboard")

cols = st.columns(len(data))
for col, (ticker, df) in zip(cols, data.items()):
    price  = float(df["Close"].iloc[-1])
    prev   = float(df["Close"].iloc[-2]) if len(df) > 1 else price
    change = price - prev
    pct    = (change / prev) * 100 if prev else 0.0
    col.metric(ticker, f"${price:.2f}", f"{pct:+.2f}%")

st.markdown("---")

# ─────────────────────────────────────────────
# 1. CLOSING PRICE TREND — MULTI TICKER
# ─────────────────────────────────────────────
st.subheader("📈 Closing Price Trend (All Tickers)")

fig_trend = go.Figure()
for ticker, df in data.items():
    fig_trend.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name=ticker,
        line=dict(width=2),
    ))

fig_trend.update_layout(
    xaxis_title="Date",
    yaxis_title="Close Price (USD)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
    margin=dict(l=40, r=20, t=30, b=40),
)
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# 2. CANDLESTICK + VOLUME  (focused ticker)
# ─────────────────────────────────────────────
focus_ticker = st.selectbox("Select ticker for detailed charts", list(data.keys()))
df = data[focus_ticker]

st.subheader(f"🕯️ {focus_ticker} — Candlestick & Volume")

fig_candle = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.05,
)

fig_candle.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"], high=df["High"],
    low=df["Low"],   close=df["Close"],
    name="Price",
), row=1, col=1)

fig_candle.add_trace(go.Scatter(x=df.index, y=df["MA7"],  name="MA7",  line=dict(width=1, dash="dot")),  row=1, col=1)
fig_candle.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(width=1, dash="dash")), row=1, col=1)

colors = ["green" if c >= o else "red"
          for c, o in zip(df["Close"], df["Open"])]
fig_candle.add_trace(go.Bar(
    x=df.index, y=df["Volume"],
    name="Volume",
    marker_color=colors,
), row=2, col=1)

fig_candle.update_layout(
    xaxis_rangeslider_visible=False,
    height=520,
    margin=dict(l=40, r=20, t=30, b=40),
    hovermode="x unified",
)
fig_candle.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig_candle.update_yaxes(title_text="Volume",      row=2, col=1)
st.plotly_chart(fig_candle, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# 3. RSI
# ─────────────────────────────────────────────
st.subheader(f"📉 {focus_ticker} — RSI (14)")

fig_rsi = go.Figure()

fig_rsi.add_trace(go.Scatter(
    x=df.index, y=df["RSI"],
    mode="lines",
    name="RSI",
    line=dict(color="#f59e0b", width=2),
))

# Overbought / oversold bands
fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red",   opacity=0.08, line_width=0, annotation_text="Overbought",  annotation_position="top left")
fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor="green", opacity=0.08, line_width=0, annotation_text="Oversold",    annotation_position="bottom left")
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",   line_width=1)
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", line_width=1)

fig_rsi.update_layout(
    xaxis_title="Date",
    yaxis_title="RSI",
    yaxis=dict(range=[0, 100]),
    height=320,
    margin=dict(l=40, r=20, t=30, b=40),
    hovermode="x unified",
)
st.plotly_chart(fig_rsi, use_container_width=True)

# RSI signal summary
latest_rsi = float(df["RSI"].iloc[-1]) if not df["RSI"].isna().all() else None
if latest_rsi is not None:
    if latest_rsi >= 70:
        st.warning(f"⚠️ RSI = {latest_rsi:.1f} — **Overbought** territory. Potential pullback ahead.")
    elif latest_rsi <= 30:
        st.success(f"✅ RSI = {latest_rsi:.1f} — **Oversold** territory. Potential bounce ahead.")
    else:
        st.info(f"ℹ️ RSI = {latest_rsi:.1f} — Neutral zone.")

st.markdown("---")

# ─────────────────────────────────────────────
# 4. MACD
# ─────────────────────────────────────────────
st.subheader(f"📊 {focus_ticker} — MACD (12 / 26 / 9)")

fig_macd = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.6, 0.4],
    vertical_spacing=0.05,
)

# MACD & Signal line
fig_macd.add_trace(go.Scatter(
    x=df.index, y=df["MACD"],
    name="MACD",
    line=dict(color="#3b82f6", width=2),
), row=1, col=1)

fig_macd.add_trace(go.Scatter(
    x=df.index, y=df["Signal"],
    name="Signal",
    line=dict(color="#ef4444", width=2),
), row=1, col=1)

# Histogram with color based on positive/negative
hist_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in df["Hist"].fillna(0)]
fig_macd.add_trace(go.Bar(
    x=df.index, y=df["Hist"],
    name="Histogram",
    marker_color=hist_colors,
    opacity=0.7,
), row=2, col=1)

fig_macd.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=2, col=1)

fig_macd.update_layout(
    xaxis_rangeslider_visible=False,
    height=420,
    margin=dict(l=40, r=20, t=30, b=40),
    hovermode="x unified",
)
fig_macd.update_yaxes(title_text="MACD",      row=1, col=1)
fig_macd.update_yaxes(title_text="Histogram", row=2, col=1)

st.plotly_chart(fig_macd, use_container_width=True)

# MACD signal summary
latest_macd   = float(df["MACD"].iloc[-1])   if not df["MACD"].isna().all()   else None
latest_signal = float(df["Signal"].iloc[-1]) if not df["Signal"].isna().all() else None
if latest_macd is not None and latest_signal is not None:
    if latest_macd > latest_signal:
        st.success(f"✅ MACD ({latest_macd:.3f}) > Signal ({latest_signal:.3f}) — **Bullish** crossover signal.")
    else:
        st.warning(f"⚠️ MACD ({latest_macd:.3f}) < Signal ({latest_signal:.3f}) — **Bearish** crossover signal.")

st.markdown("---")

# ─────────────────────────────────────────────
# 5. ANOMALIES
# ─────────────────────────────────────────────
st.subheader("⚠️ Return Anomalies")

df_a  = detect_anomalies(df)
anom  = df_a[df_a["anom"]]

if anom.empty:
    st.success(f"No anomalies detected for {focus_ticker}.")
else:
    for idx, row in anom.tail(5).iterrows():
        date_str = pd.Timestamp(idx).date()
        ret_val  = float(row["Returns"]) * 100
        icon     = "🔴" if ret_val < 0 else "🟢"
        st.write(f"{icon} {date_str} → {ret_val:.2f}%")

st.markdown("---")

# ─────────────────────────────────────────────
# 6. SUMMARY TABLE
# ─────────────────────────────────────────────
st.subheader("📋 Summary Table")

summary_rows = []
for ticker, df_s in data.items():
    price  = float(df_s["Close"].iloc[-1])
    first  = float(df_s["Close"].iloc[0])
    ret    = (price / first - 1) * 100 if first else 0.0
    vol    = float(df_s["Returns"].std()) * np.sqrt(252) * 100
    rsi    = float(df_s["RSI"].iloc[-1]) if not df_s["RSI"].isna().all() else None
    macd_v = float(df_s["MACD"].iloc[-1]) if not df_s["MACD"].isna().all() else None

    summary_rows.append({
        "Ticker":        ticker,
        "Price ($)":     round(price, 2),
        "Return %":      round(ret,   2),
        "Ann. Vol %":    round(vol,   2),
        "RSI (14)":      round(rsi,   1) if rsi is not None else "N/A",
        "MACD":          round(macd_v, 4) if macd_v is not None else "N/A",
    })

summary = pd.DataFrame(summary_rows)
st.dataframe(summary, use_container_width=True)
