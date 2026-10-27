from concurrent.futures import as_completed, ThreadPoolExecutor
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Quarterly Catalyst & PEGY Radar",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ Multi-Market Quarterly Breakout & PEGY Radar")
st.caption(
    "Institutional scanning across S&P 100, Nasdaq-100 & Metals: Strat Inside"
    " Bars + TTM Squeeze + Lynch PEGY."
)

# --- Visual Legend ---
with st.expander(
    "🎨 Color Legend & Reading Guide (What is Good vs. Not)", expanded=True
):
  col_l1, col_l2, col_l3, col_l4 = st.columns(4)
  with col_l1:
    st.markdown("**Setup Score / Grade**")
    st.markdown("🟢 **80–100:** Prime A+ (High Confluence)")
    st.markdown("🟡 **60–79:** Stage 2 Setup")
    st.markdown("🔴 **< 60:** Low Probability / Watchlist")
  with col_l2:
    st.markdown("**PEGY / Valuation**")
    st.markdown("🟢 **< 1.0:** Deep Bargain (Peter Lynch)")
    st.markdown("🟡 **1.0–1.5:** Fair Compounding Zone")
    st.markdown("🔴 **> 1.5:** Stretched / Overvalued")
  with col_l3:
    st.markdown("**Strat & Continuity**")
    st.markdown("🟢 **1 (Inside Bar):** Coiled spring")
    st.markdown("🟢 **2U (Bullish):** Active upside continuity")
    st.markdown("🔴 **2D (Pullback):** Red monthly / trap risk")
  with col_l4:
    st.markdown("**Squeeze Status**")
    st.markdown(
        "🔥 **Active Squeeze:** Bollinger Bands inside Keltner (Coiling)"
    )
    st.markdown("⚪ **Standard:** Open / Normal expansion")

# --- Preset Universes ---
METALS_MINERS = [
    "GLD",
    "SLV",
    "IAU",
    "PPLT",
    "CPER",
    "NEM",
    "GOLD",
    "AEM",
    "PAAS",
    "AG",
    "FSM",
    "HL",
    "CDE",
    "FCX",
    "SCCO",
    "NUE",
    "STLD",
    "AA",
    "CLF",
]


@st.cache_data(ttl=86400)
def load_sp100_tickers():
  try:
    tables = pd.read_html("https://en.wikipedia.org/wiki/S%26P_100")
    df_sp = tables[2] if len(tables) > 2 else tables[1]
    col = "Symbol" if "Symbol" in df_sp.columns else "Ticker"
    return [
        t.strip().upper()
        for t in df_sp[col].str.replace(".", "-", regex=False)
        if isinstance(t, str)
    ]
  except Exception:
    return [
        "AAPL",
        "ABBV",
        "AMD",
        "AMZN",
        "AVGO",
        "CAT",
        "COP",
        "CVX",
        "DE",
        "GS",
        "JPM",
        "META",
        "MSFT",
        "NVDA",
        "XOM",
    ]


@st.cache_data(ttl=86400)
def load_nasdaq100_tickers():
  try:
    tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
    df_ndx = tables[4] if len(tables) > 4 else tables[3]
    col = "Ticker" if "Ticker" in df_ndx.columns else "Symbol"
    return [
        t.strip().upper()
        for t in df_ndx[col].str.replace(".", "-", regex=False)
        if isinstance(t, str)
    ]
  except Exception:
    return [
        "AAPL",
        "AMAT",
        "AMD",
        "AMZN",
        "APP",
        "ARM",
        "AVGO",
        "CRWD",
        "GOOGL",
        "KLAC",
        "LRCX",
        "META",
        "MSFT",
        "NVDA",
        "PANW",
        "TSLA",
    ]


# --- Sidebar Controls ---
st.sidebar.header("🎯 Universe Selection")
include_sp100 = st.sidebar.checkbox("S&P 100 Large-Caps", value=True)
include_ndx = st.sidebar.checkbox("Nasdaq-100 Growth / Tech", value=True)
include_metals = st.sidebar.checkbox("Metals & Commodities", value=True)
custom_extra = st.sidebar.text_input(
    "Add Custom Tickers", value="FORM, ONTO, COHU, PLTR, URI"
)

selected_universe = set()
if include_sp100:
  selected_universe.update(load_sp100_tickers())
if include_ndx:
  selected_universe.update(load_nasdaq100_tickers())
if include_metals:
  selected_universe.update(METALS_MINERS)
if custom_extra:
  for t in custom_extra.split(","):
    if t.strip():
      selected_universe.add(t.strip().upper())

universe_list = sorted(list(selected_universe))
st.sidebar.markdown(f"**Total Monitored:** `{len(universe_list)}` tickers")

st.sidebar.markdown("---")
st.sidebar.header("⚡ Filters")
min_score = st.sidebar.slider("Min Setup Score", 0, 100, 50, 5)
filter_squeeze = st.sidebar.checkbox("Active Squeeze Only", value=False)
filter_inside = st.sidebar.checkbox("Monthly '1' Inside Bar Only", value=False)
filter_ftc = st.sidebar.checkbox("Monthly FTC UP Only", value=False)


# --- Data Pipeline ---
@st.cache_data(ttl=1800)
def fetch_info(ticker):
  try:
    info = yf.Ticker(ticker).info or {}
    return (
        ticker,
        info.get("forwardPE", np.nan),
        info.get("trailingPE", np.nan),
        info.get("pegRatio", np.nan),
        (info.get("dividendYield") or 0.0) * 100.0,
        (
            "Metals/ETF"
            if ticker in METALS_MINERS
            else info.get("sector", "Other")
        ),
    )
  except Exception:
    return ticker, np.nan, np.nan, np.nan, 0.0, "Other"


@st.cache_data(ttl=1800)
def run_scan(tickers):
  if not tickers:
    return pd.DataFrame()

  df_daily = yf.download(
      tickers,
      period="1y",
      interval="1d",
      group_by="ticker",
      auto_adjust=True,
      threads=True,
      progress=False,
  )
  df_monthly = yf.download(
      tickers,
      period="2y",
      interval="1mo",
      group_by="ticker",
      auto_adjust=True,
      threads=True,
      progress=False,
  )

  fundamentals = {}
  with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(fetch_info, t): t for t in tickers}
    for fut in as_completed(futures):
      t, fwd_pe, trail_pe, peg, div_yield, sector = fut.result()
      fundamentals[t] = {
          "pe": fwd_pe if pd.notnull(fwd_pe) and fwd_pe > 0 else trail_pe,
          "peg": peg,
          "yield": div_yield,
          "sector": sector,
      }

  results = []
  for t in tickers:
    try:
      d = df_daily[t].dropna(subset=["Close"]) if len(tickers) > 1 else df_daily
      m = (
          df_monthly[t].dropna(subset=["Close"])
          if len(tickers) > 1
          else df_monthly
      )
      if len(d) < 50 or len(m) < 3:
        continue

      close_d = d["Close"]
      price = close_d.iloc[-1]

      sma50 = close_d.rolling(50).mean().iloc[-1]
      sma200 = (
          close_d.rolling(200).mean().iloc[-1] if len(d) >= 200 else sma50
      )
      ema21 = close_d.ewm(span=21, adjust=False).mean().iloc[-1]

      # Volatility Squeeze
      bb_mid = close_d.rolling(20).mean()
      bb_std = close_d.rolling(20).std()
      bb_upper = bb_mid + (2 * bb_std)
      bb_lower = bb_mid - (2 * bb_std)

      tr = pd.concat([
          d["High"] - d["Low"],
          (d["High"] - close_d.shift(1)).abs(),
          (d["Low"] - close_d.shift(1)).abs(),
      ],
                     axis=1).max(axis=1)
      atr20 = tr.rolling(20).mean()
      kc_upper = bb_mid + (1.5 * atr20)
      kc_lower = bb_mid - (1.5 * atr20)
      is_squeezed = (bb_lower.iloc[-1] > kc_lower.iloc[-1]) and (
          bb_upper.iloc[-1] < kc_upper.iloc[-1]
      )

      curr_m, prev_m = m.iloc[-1], m.iloc[-2]
      ftc_up = curr_m["Close"] > curr_m["Open"]
      inside = (curr_m["High"] <= prev_m["High"]) and (
          curr_m["Low"] >= prev_m["Low"]
      )
      strat = (
          "1 (Inside Coil)"
          if inside
          else (
              "2U (Continuation)"
              if curr_m["High"] > prev_m["High"]
              else "2D (Pullback)"
          )
      )

      f = fundamentals.get(t, {})
      pe, peg, dy = f.get("pe"), f.get("peg"), f.get("yield", 0.0)
      pegy = np.nan
      if pd.notnull(peg) and peg > 0 and pd.notnull(pe) and pe > 0:
        growth = pe / peg
        if (growth + dy) > 0:
          pegy = pe / (growth + dy)

      # Score Engine
      score = 0
      if price > sma200:
        score += 10
      if price > sma50:
        score += 10
      if price > ema21:
        score += 10
      if ftc_up:
        score += 15
      if inside:
        score += 15
      elif strat == "2U (Continuation)":
        score += 10
      if is_squeezed:
        score += 20

      metric = pegy if pd.notnull(pegy) else peg
      if pd.notnull(metric) and metric <= 1.0:
        score += 20
      elif pd.notnull(metric) and metric <= 1.5:
        score += 10
      elif t in METALS_MINERS:
        score += 15 if ftc_up and price > sma50 else 5

      results.append({
          "Ticker": t,
          "Sector": f.get("sector", "Other"),
          "Price": round(price, 2),
          "Score": int(score),
          "Squeeze": "🔥 SQUEEZE" if is_squeezed else "Standard",
          "Monthly Strat": strat,
          "Monthly FTC": "Bullish 🟢" if ftc_up else "Bearish 🔴",
          "P/E": round(pe, 1) if pd.notnull(pe) else np.nan,
          "Lynch PEGY": round(pegy, 2) if pd.notnull(pegy) else np.nan,
          "Standard PEG": round(peg, 2) if pd.notnull(peg) else np.nan,
          "Yield (%)": round(dy, 2)
