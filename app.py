from concurrent.futures import as_completed, ThreadPoolExecutor
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="High-Scale Quarterly Breakout & PEGY Radar",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Multi-Market Quarterly Breakout & PEGY Radar")
st.caption(
    "Batched institutional scanning engine across S&P 100, Nasdaq, &"
    " Metals/Miners."
)

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


# --- Ticker Resolvers with Wikipedia Scrapers + Offline Fallbacks ---
@st.cache_data(ttl=86400)
def load_sp100_tickers():
  try:
    tables = pd.read_html("https://en.wikipedia.org/wiki/S%26P_100")
    df = tables[2] if len(tables) > 2 else tables[1]
    col = "Symbol" if "Symbol" in df.columns else "Ticker"
    return [
        t.strip().upper()
        for t in df[col].str.replace(".", "-", regex=False)
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
    df = tables[4] if len(tables) > 4 else tables[3]
    col = "Ticker" if "Ticker" in df.columns else "Symbol"
    return [
        t.strip().upper()
        for t in df[col].str.replace(".", "-", regex=False)
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


# --- Sidebar Selection ---
st.sidebar.header("🎯 Universe Selection")
inc_sp = st.sidebar.checkbox("S&P 100 Large-Caps", value=True)
inc_ndx = st.sidebar.checkbox("Nasdaq Growth / Tech", value=True)
inc_metals = st.sidebar.checkbox("Metals, Miners & Commodities", value=True)
custom_extra = st.sidebar.text_input(
    "Add Custom Tickers", value="FORM, ONTO, COHU, PLTR, URI"
)

selected_universe = set()
if inc_sp:
  selected_universe.update(load_sp100_tickers())
if inc_ndx:
  selected_universe.update(load_nasdaq100_tickers())
if inc_metals:
  selected_universe.update(METALS_MINERS)
if custom_extra:
  for t in custom_extra.split(","):
    if t.strip():
      selected_universe.add(t.strip().upper())

universe_list = sorted(list(selected_universe))
st.sidebar.markdown(f"**Total Active Watchlist:** `{len(universe_list)}` names")

min_score = st.sidebar.slider("Min Setup Score", 0, 100, 50, 5)
filter_squeeze = st.sidebar.checkbox("Active Daily Squeeze Only", value=False)
filter_inside = st.sidebar.checkbox("Monthly '1' Inside Bar Only", value=False)
filter_ftc = st.sidebar.checkbox(
    "Full Timeframe Continuity UP Only", value=False
)


# --- High-Speed Vectorized Scanner ---
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
  # 1. Batch download daily & monthly price arrays in parallel
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

  # 2. Concurrently fetch fundamental data
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

      # Moving Averages & Trend
      sma50 = close_d.rolling(50).mean().iloc[-1]
      sma200 = (
          close_d.rolling(200).mean().iloc[-1] if len(d) >= 200 else sma50
      )
      ema21 = close_d.ewm(span=21, adjust=False).mean().iloc[-1]

      # Volatility Squeeze (20-day BB vs 20-day KC)
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

      # Monthly Strat & FTC
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

      # Valuation & PEGY Calculation
      f = fundamentals.get(t, {})
      pe, peg, dy = f.get("pe"), f.get("peg"), f.get("yield", 0.0)
      pegy = np.nan
      if pd.notnull(peg) and peg > 0 and pd.notnull(pe) and pe > 0:
        growth = pe / peg
        if (growth + dy) > 0:
          pegy = pe / (growth + dy)

      # Score Engine (0-100)
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
          "Squeeze": "🔥 Squeeze" if is_squeezed else "Standard",
          "Monthly Strat": strat,
          "Monthly FTC": "Bullish 🟢" if ftc_up else "Bearish 🔴",
          "P/E": round(pe, 1) if pd.notnull(pe) else "-",
          "Lynch PEGY": round(pegy, 2) if pd.notnull(pegy) else "-",
          "Standard PEG": round(peg, 2) if pd.notnull(peg) else "-",
          "Yield (%)": f"{round(dy, 2)}%" if dy > 0 else "-",
      })
    except Exception:
      continue

  return pd.DataFrame(results)


# --- Execute Scan ---
with st.spinner(f"Scanning {len(universe_list)} institutional names..."):
  df = run_scan(universe_list)

if not df.empty:
  out = df[df["Score"] >= min_score]
  if filter_squeeze:
    out = out[out["Squeeze"] == "🔥 Squeeze"]
  if filter_inside:
    out = out[out["Monthly Strat"] == "1 (Inside Coil)"]
  if filter_ftc:
    out = out[out["Monthly FTC"] == "Bullish 🟢"]

  st.dataframe(
      out.sort_values(by="Score", ascending=False),
      use_container_width=True,
      hide_index=True,
  )
