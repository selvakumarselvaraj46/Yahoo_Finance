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
# INDICATOR HELPERS  (operate on plain Series)
# ─────────────────────────────────────────────
def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l  = loss.ewm(com=period - 1, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).rename("RSI")


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = (ema_fast - ema_slow).rename("MACD")
    signal_line = macd_line.ewm(span=signal, adjust=False).mean().rename("Signal")
    hist        = (macd_line - signal_line).rename("Hist")
    return macd_line, signal_line, hist


# ─────────────────────────────────────────────
# DATA FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_data(ttl=REFRESH_TTL)
def get_hist(ticker: str, period: str) -> pd.DataFrame:
    raw = yf.download(
        ticker, period=period, interval="1d",
        progress=False, auto_adjust=True
    )
    if raw.empty:
        return pd.DataFrame()

    # ── Flatten MultiIndex columns (yfinance >= 0.2.x) ──────────────
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ["_".join([c for c in col if c]).strip("_")
                       for col in raw.columns.to_flat_index()]
        rename_map = {}
        for col in raw.columns:
            for base in ["Open", "High", "Low", "Close", "Volume"]:
                if col.startswith(base):
                    rename_map[col] = base
        raw.rename(columns=rename_map, inplace=True)

    # Keep only OHLCV columns
    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    df = raw[needed].copy()

    # Ensure all columns are plain float Series
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].squeeze(), errors="coerce")

    df.dropna(subset=["Close"], inplace=True)

    if df.empty:
        return pd.DataFrame()

    close = df["Close"].copy()

    # ── Moving averages & returns ────────────────────────────────────
    df["Returns"]    = close.pct_change()
    df["Volatility"] = df["Returns"].rolling(7).std() * 100
    df["MA7"]        = close.rolling(7).mean()
    df["MA20"]       = close.rolling(20).mean()
    df["MA50"]       = close.rolling(50).mean()

    # ── RSI — assign via numpy array to avoid index alignment issues ─
    rsi_vals        = calc_rsi(close)
    df["RSI"]       = rsi_vals.to_numpy()

    # ── MACD ────────────────────────────────────────────────────────
    macd, signal, hist = calc_macd(close)
    df["MACD"]   = macd.to_numpy()
    df["Signal"] = signal.to_numpy()
    df["Hist"]   = hist.to_numpy()

    return df


@st.cache_data(ttl=REFRESH_TTL)
def get_multi(tickers: tuple, period: str) -> dict:
    result = {}
    for t in tickers:
        df = get_hist(t, period)
        if not df.empty:
            result[t] = df
    return result


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    returns = df["Returns"].dropna()
    if len(returns) < 2:
        df["z"] = 0.0
        df["anom"] = False
        return df
    mu, sigma = returns.mean(), returns.std()
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

custom = st.sidebar.text_input("Add custom ticker (e.g. GOOG)")
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

with st.spinner("Fetching data..."):
    data = get_multi(tuple(chosen), period)

if not data:
    st.error("No data found. Check ticker symbols and try again.")
    st.stop()

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
st.title("📊 Stock Dashboard")

cols = st.columns(len(data))
for col, (ticker, df) in zip(cols, data.items()):
    price = float(df["Close"].iloc[-1])
    prev  = float(df["Close"].iloc[-2]) if len(df) > 1 else price
    pct   = ((price - prev) / prev * 100) if prev else 0.0
    col.metric(ticker, f"${price:.2f}", f"{pct:+.2f}%")

st.markdown("---")

# ─────────────────────────────────────────────
# 1. CLOSING PRICE TREND — MULTI TICKER
# ─────────────────────────────────────────────
st.subheader("📈 Closing Price Trend (All Tickers)")

fig_trend = go.Figure()
for ticker, df in data.items():
    fig_trend.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", name=ticker, line=dict(width=2),
    ))

fig_trend.update_layout(
    xaxis_title="Date", yaxis_title="Close Price (USD)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400, margin=dict(l=40, r=20, t=30, b=40),
)
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# FOCUSED TICKER SELECTOR
# ─────────────────────────────────────────────
focus_ticker = st.selectbox("Select ticker for detailed charts", list(data.keys()))
df = data[focus_ticker].copy()

# ─────────────────────────────────────────────
# 2. CANDLESTICK + VOLUME
# ─────────────────────────────────────────────
st.subheader(f"🕯️ {focus_ticker} — Candlestick & Volume")

fig_candle = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.7, 0.3], vertical_spacing=0.05,
)
fig_candle.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"], high=df["High"],
    low=df["Low"],   close=df["Close"],
    name="Price",
), row=1, col=1)
fig_candle.add_trace(go.Scatter(
    x=df.index, y=df["MA7"], name="MA7",
    line=dict(width=1, dash="dot")), row=1, col=1)
fig_candle.add_trace(go.Scatter(
    x=df.index, y=df["MA20"], name="MA20",
    line=dict(width=1, dash="dash")), row=1, col=1)

vol_colors = ["green" if c >= o else "red"
              for c, o in zip(df["Close"], df["Open"])]
fig_candle.add_trace(go.Bar(
    x=df.index, y=df["Volume"],
    name="Volume", marker_color=vol_colors,
), row=2, col=1)

fig_candle.update_layout(
    xaxis_rangeslider_visible=False, height=500,
    margin=dict(l=40, r=20, t=30, b=40), hovermode="x unified",
)
fig_candle.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig_candle.update_yaxes(title_text="Volume",      row=2, col=1)
st.plotly_chart(fig_candle, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# 3. RSI
# ─────────────────────────────────────────────
st.subheader(f"📉 {focus_ticker} — RSI (14)")

# Safety: recompute if column missing or all-NaN
if "RSI" not in df.columns or df["RSI"].isna().all():
    df["RSI"] = calc_rsi(df["Close"]).to_numpy()

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(
    x=df.index, y=df["RSI"],
    mode="lines", name="RSI",
    line=dict(color="#f59e0b", width=2),
))
fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red",   opacity=0.08, line_width=0,
                  annotation_text="Overbought (70)", annotation_position="top left")
fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor="green", opacity=0.08, line_width=0,
                  annotation_text="Oversold (30)",   annotation_position="bottom left")
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",   line_width=1)
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", line_width=1)
fig_rsi.update_layout(
    xaxis_title="Date", yaxis_title="RSI",
    yaxis=dict(range=[0, 100]),
    height=320, margin=dict(l=40, r=20, t=30, b=40),
    hovermode="x unified",
)
st.plotly_chart(fig_rsi, use_container_width=True)

rsi_clean = df["RSI"].dropna()
if not rsi_clean.empty:
    latest_rsi = float(rsi_clean.iloc[-1])
    if latest_rsi >= 70:
        st.warning(f"⚠️ RSI = {latest_rsi:.1f} — Overbought. Potential pullback ahead.")
    elif latest_rsi <= 30:
        st.success(f"✅ RSI = {latest_rsi:.1f} — Oversold. Potential bounce ahead.")
    else:
        st.info(f"ℹ️ RSI = {latest_rsi:.1f} — Neutral zone.")

st.markdown("---")

# ─────────────────────────────────────────────
# 4. MACD
# ─────────────────────────────────────────────
st.subheader(f"📊 {focus_ticker} — MACD (12 / 26 / 9)")

# Safety: recompute if columns missing or all-NaN
if any(c not in df.columns or df[c].isna().all() for c in ["MACD", "Signal", "Hist"]):
    m, s, h      = calc_macd(df["Close"])
    df["MACD"]   = m.to_numpy()
    df["Signal"] = s.to_numpy()
    df["Hist"]   = h.to_numpy()

fig_macd = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.6, 0.4], vertical_spacing=0.05,
)
fig_macd.add_trace(go.Scatter(
    x=df.index, y=df["MACD"],
    name="MACD", line=dict(color="#3b82f6", width=2),
), row=1, col=1)
fig_macd.add_trace(go.Scatter(
    x=df.index, y=df["Signal"],
    name="Signal", line=dict(color="#ef4444", width=2),
), row=1, col=1)

hist_colors = ["#22c55e" if v >= 0 else "#ef4444"
               for v in df["Hist"].fillna(0)]
fig_macd.add_trace(go.Bar(
    x=df.index, y=df["Hist"],
    name="Histogram", marker_color=hist_colors, opacity=0.7,
), row=2, col=1)
fig_macd.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=2, col=1)
fig_macd.update_layout(
    xaxis_rangeslider_visible=False, height=420,
    margin=dict(l=40, r=20, t=30, b=40), hovermode="x unified",
)
fig_macd.update_yaxes(title_text="MACD",      row=1, col=1)
fig_macd.update_yaxes(title_text="Histogram", row=2, col=1)
st.plotly_chart(fig_macd, use_container_width=True)

macd_clean = df["MACD"].dropna()
sig_clean  = df["Signal"].dropna()
if not macd_clean.empty and not sig_clean.empty:
    lm, ls = float(macd_clean.iloc[-1]), float(sig_clean.iloc[-1])
    if lm > ls:
        st.success(f"✅ MACD ({lm:.3f}) > Signal ({ls:.3f}) — Bullish crossover signal.")
    else:
        st.warning(f"⚠️ MACD ({lm:.3f}) < Signal ({ls:.3f}) — Bearish crossover signal.")

st.markdown("---")

# ─────────────────────────────────────────────
# 5. ANOMALIES
# ─────────────────────────────────────────────
st.subheader(f"⚠️ Return Anomalies — {focus_ticker}")

df_a = detect_anomalies(df)
anom = df_a[df_a["anom"]]

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
    price = float(df_s["Close"].iloc[-1])
    first = float(df_s["Close"].iloc[0])
    ret   = (price / first - 1) * 100 if first else 0.0
    vol   = float(df_s["Returns"].std()) * np.sqrt(252) * 100

    rsi_s  = df_s["RSI"].dropna()  if "RSI"  in df_s.columns else pd.Series(dtype=float)
    macd_s = df_s["MACD"].dropna() if "MACD" in df_s.columns else pd.Series(dtype=float)

    summary_rows.append({
        "Ticker":     ticker,
        "Price ($)":  round(price, 2),
        "Return %":   round(ret,   2),
        "Ann. Vol %": round(vol,   2),
        "RSI (14)":   round(float(rsi_s.iloc[-1]),  1) if not rsi_s.empty  else "N/A",
        "MACD":       round(float(macd_s.iloc[-1]), 4) if not macd_s.empty else "N/A",
    })

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
