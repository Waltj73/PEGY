from concurrent.futures import as_completed, ThreadPoolExecutor
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Monthly OTM Breakout Radar",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .metric-card {
        background-color: #1E222D;
        border-radius: 8px;
        padding: 14px;
        border-left: 4px solid #2962FF;
        margin-bottom: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("⚡ Monthly OTM Breakout & Probability Radar")
st.caption(
    "Active 20–35 DTE Swing Scanner: Weekly Coils + High Implied Velocity +"
    " Asymmetric OTM Call Debit Architecture."
)

with st.expander(
    "🎓 The OTM Debit Spread Blueprint (High-R:R Monthly Model)", expanded=False
):
  col_t1, col_t2, col_t3 = st.columns(3)
  with col_t1:
    st.markdown("### 🎯 Strike Architecture")
    st.markdown(
        "• **Buy Long Call:** ~40–45 Delta (First OTM strike right at breakout"
        " resistance)."
    )
    st.markdown(
        "• **Sell Short Call:** ~20–25 Delta (Next technical target or 1"
        " Expected Move away)."
    )
    st.markdown(
        "• **Cost Target:** 25%–35% of spread width (Yielding **2:1 to 3:1"
        " R:R**)."
    )
  with col_t2:
    st.markdown("### ⏱️ Timeframe Alignment")
    st.markdown("• **Duration:** 21 to 35 DTE (Front monthly expiration).")
    st.markdown(
        "• **Anchor:** Weekly Strat `1` or `2U` in line with **Monthly Bullish"
        " FTC**."
    )
    st.markdown(
        "• **Velocity Requirement:** EPS Velocity > 15% (Ensures real momentum"
        " drivers)."
    )
  with col_t3:
    st.markdown("### 🛑 Trade Management")
    st.markdown(
        "• **Exit at 100%–150% Profit:** Don't hold to max expiration; harvest"
        " early when Delta expands."
    )
    st.markdown(
        "• **Stop-Loss:** Daily close below the rising 8 or 21 EMA cuts the"
        " trade at a defined 40%–50% debit loss."
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
        "ABT",
        "ACN",
        "ADBE",
        "AIG",
        "AMD",
        "AMGN",
        "AMZN",
        "AVGO",
        "AXP",
        "BA",
        "BAC",
        "BK",
        "BKNG",
        "BLK",
        "BMY",
        "C",
        "CAT",
        "CL",
        "CMCSA",
        "COF",
        "COP",
        "COST",
        "CRM",
        "CSCO",
        "CVS",
        "CVX",
        "DE",
        "DHR",
        "DIS",
        "DUK",
        "EMR",
        "FDX",
        "GD",
        "GE",
        "GILD",
        "GM",
        "GOOG",
        "GOOGL",
        "GS",
        "HD",
        "HON",
        "IBM",
        "INTC",
        "INTU",
        "JNJ",
        "JPM",
        "KHC",
        "KO",
        "LIN",
        "LLY",
        "LMT",
        "LOW",
        "MA",
        "MCD",
        "MDLZ",
        "MDT",
        "MET",
        "META",
        "MMM",
        "MO",
        "MRK",
        "MS",
        "MSFT",
        "NEE",
        "NFLX",
        "NKE",
        "NVDA",
        "ORCL",
        "PEP",
        "PFE",
        "PG",
        "PM",
        "PYPL",
        "QCOM",
        "RTX",
        "SBUX",
        "SCHW",
        "SO",
        "SPG",
        "T",
        "TGT",
        "TJX",
        "TMO",
        "TMUS",
        "TSLA",
        "TXN",
        "UNH",
        "UNP",
        "UPS",
        "USB",
        "V",
        "VZ",
        "WFC",
        "WMT",
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
        "ABNB",
        "ADBE",
        "ADI",
        "ADP",
        "ADSK",
        "AEP",
        "AMAT",
        "AMD",
        "AMGN",
        "AMZN",
        "ANSS",
        "APP",
        "ARM",
        "ASML",
        "AVGO",
        "AZN",
        "BIIB",
        "BKNG",
        "BKR",
        "CDNS",
        "CEG",
        "CHTR",
        "CMCSA",
        "COST",
        "CPRT",
        "CRWD",
        "CSCO",
        "CSGP",
        "CSX",
        "CTAS",
        "CTSH",
        "DASH",
        "DDOG",
        "DLTR",
        "DXCM",
        "EA",
        "EXC",
        "FANG",
        "FAST",
        "FTNT",
        "GEHC",
        "GFS",
        "GILD",
        "GOOG",
        "GOOGL",
        "HON",
        "IDXX",
        "ILMN",
        "INTC",
        "INTU",
        "ISRG",
        "KDP",
        "KHC",
        "KLAC",
        "LRCX",
        "LULU",
        "MAR",
        "MCHP",
        "MDB",
        "MDLZ",
        "MELI",
        "META",
        "MNST",
        "MRNA",
        "MRVL",
        "MSFT",
        "MU",
        "NFLX",
        "NVDA",
        "NXPI",
        "ODFL",
        "ON",
        "ORLY",
        "PANW",
        "PAYX",
        "PCAR",
        "PDD",
        "PEP",
        "PYPL",
        "QCOM",
        "REGN",
        "ROP",
        "ROST",
        "SBUX",
        "SNPS",
        "TEAM",
        "TMUS",
        "TSLA",
        "TTD",
        "TTWO",
        "TXN",
        "VRSK",
        "VRTX",
        "WBD",
        "WDAY",
        "XEL",
        "ZS",
    ]


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
st.sidebar.header("⚡ Monthly Filters")
tier_filter = st.sidebar.multiselect(
    "Filter by Trade Tier",
    [
        "🏆 Tier 1 (Prime Setup)",
        "🐢 Tier 2 (Slow Clunker)",
        "⚠️ Tier 3 (Cyclical Trap)",
    ],
    default=["🏆 Tier 1 (Prime Setup)"],
)
min_score = st.sidebar.slider("Min Setup Score", 0, 100, 50, 5)
min_velocity = st.sidebar.slider("Min EPS Velocity (%)", 0, 50, 15, 5)
filter_weekly_inside = st.sidebar.checkbox(
    "Weekly '1' Inside Coil Only", value=False
)
filter_squeeze = st.sidebar.checkbox("Active Daily Squeeze Only", value=False)


@st.cache_data(ttl=1800)
def fetch_info(ticker):
  try:
    info = yf.Ticker(ticker).info or {}
    fwd_pe = info.get("forwardPE", np.nan)
    trail_pe = info.get("trailingPE", np.nan)
    peg = info.get("pegRatio", np.nan)
    beta = info.get("beta", 1.0)
    payout = (info.get("payoutRatio") or 0.0) * 100.0
    rev_growth = (info.get("revenueGrowth") or 0.0) * 100.0
    profit_margins = (info.get("profitMargins") or 0.0) * 100.0
    free_cashflow = info.get("freeCashflow", 0)

    raw_yield = info.get("dividendYield") or 0.0
    div_yield = raw_yield * 100.0 if raw_yield < 0.20 else raw_yield

    sector = (
        "Metals/ETF" if ticker in METALS_MINERS else info.get("sector", "Other")
    )
    industry = info.get("industry", "")
    short_name = info.get("shortName", ticker)
    return (
        ticker,
        fwd_pe,
        trail_pe,
        peg,
        div_yield,
        beta,
        payout,
        rev_growth,
        profit_margins,
        free_cashflow,
        sector,
        industry,
        short_name,
    )
  except Exception:
    return (
        ticker,
        np.nan,
        np.nan,
        np.nan,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0,
        "Other",
        "",
        ticker,
    )


@st.cache_data(ttl=1800)
def run_scan(tickers):
  if not tickers:
    return pd.DataFrame(), {}

  df_daily = yf.download(
      tickers,
      period="1y",
      interval="1d",
      group_by="ticker",
      auto_adjust=True,
      threads=True,
      progress=False,
  )
  df_weekly = yf.download(
      tickers,
      period="2y",
      interval="1wk",
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
      (
          t,
          fwd_pe,
          trail_pe,
          peg,
          div_yield,
          beta,
          payout,
          rev_growth,
          profit_margins,
          free_cashflow,
          sector,
          industry,
          short_name,
      ) = fut.result()
      fundamentals[t] = {
          "pe": fwd_pe if pd.notnull(fwd_pe) and fwd_pe > 0 else trail_pe,
          "peg": peg,
          "yield": div_yield,
          "beta": beta if pd.notnull(beta) else 1.0,
          "payout": payout,
          "rev_growth": rev_growth,
          "profit_margins": profit_margins,
          "free_cashflow": free_cashflow,
          "sector": sector,
          "industry": industry,
          "short_name": short_name,
      }

  results = []
  deep_dive_records = {}

  for t in tickers:
    try:
      d = df_daily[t].dropna(subset=["Close"]) if len(tickers) > 1 else df_daily
      w = (
          df_weekly[t].dropna(subset=["Close"])
          if len(tickers) > 1
          else df_weekly
      )
      m = (
          df_monthly[t].dropna(subset=["Close"])
          if len(tickers) > 1
          else df_monthly
      )

      if len(d) < 50 or len(w) < 4 or len(m) < 3:
        continue

      close_d = d["Close"]
      high_d = d["High"]
      low_d = d["Low"]
      price = close_d.iloc[-1]

      sma50 = close_d.rolling(50).mean().iloc[-1]
      sma200 = (
          close_d.rolling(200).mean().iloc[-1] if len(d) >= 200 else sma50
      )
      ema8 = close_d.ewm(span=8, adjust=False).mean().iloc[-1]
      ema21 = close_d.ewm(span=21, adjust=False).mean().iloc[-1]

      # Volatility Squeeze
      bb_mid = close_d.rolling(20).mean()
      bb_std = close_d.rolling(20).std()
      bb_upper = bb_mid + (2 * bb_std)
      bb_lower = bb_mid - (2 * bb_std)

      tr = pd.concat([
          high_d - low_d,
          (high_d - close_d.shift(1)).abs(),
          (low_d - close_d.shift(1)).abs(),
      ],
                     axis=1).max(axis=1)
      atr20 = tr.rolling(20).mean().iloc[-1]
      kc_upper = bb_mid + (1.5 * tr.rolling(20).mean())
      kc_lower = bb_mid - (1.5 * tr.rolling(20).mean())
      is_squeezed = bool(
          (bb_lower.iloc[-1] > kc_lower.iloc[-1])
          and (bb_upper.iloc[-1] < kc_upper.iloc[-1])
      )

      # Weekly Strat Structure (For Monthly Timelines)
      curr_w, prev_w = w.iloc[-1], w.iloc[-2]
      weekly_inside = bool(
          (curr_w["High"] <= prev_w["High"]) and (curr_w["Low"] >= prev_w["Low"])
      )
      weekly_strat = (
          "1 (Inside Coil)"
          if weekly_inside
          else (
              "2U (Breakout)"
              if curr_w["High"] > prev_w["High"]
              else "2D (Pullback)"
          )
      )

      # Monthly Strat & FTC (Anchor)
      curr_m, prev_m = m.iloc[-1], m.iloc[-2]
      m_ftc_up = bool(curr_m["Close"] > curr_m["Open"])
      m_strat = (
          "1 (Inside)"
          if bool(
              (curr_m["High"] <= prev_m["High"])
              and (curr_m["Low"] >= prev_m["Low"])
          )
          else ("2U" if curr_m["High"] > prev_m["High"] else "2D")
      )

      f = fundamentals.get(t, {})
      pe = f.get("pe")
      peg = f.get("peg")
      dy = f.get("yield", 0.0)
      beta = f.get("beta", 1.0)
      sector = f.get("sector", "Other")

      implied_growth = np.nan
      pegy = np.nan
      if pd.notnull(peg) and peg > 0 and pd.notnull(pe) and pe > 0:
        implied_growth = pe / peg
        tot_engine = implied_growth + dy
        if tot_engine > 0:
          pegy = pe / tot_engine

      # 30-Day Expected Move & Implied Volatility Proxy
      returns = close_d.pct_change().dropna()
      hist_vol = (
          returns.iloc[-30:].std() * np.sqrt(252)
          if len(returns) >= 30
          else 0.30
      )
      em_30d = price * hist_vol * np.sqrt(30 / 365)

      # OTM Strike Calculations (~42 Delta Long / ~22 Delta Short)
      otm_long_strike = (
          round((price + (0.35 * em_30d)) / 2.5) * 2.5
          if price < 100
          else round((price + (0.35 * em_30d)) / 5.0) * 5.0
      )
      if otm_long_strike <= price:
        otm_long_strike = price + 2.5 if price < 100 else price + 5.0

      otm_short_strike = (
          round((price + (1.0 * em_30d)) / 2.5) * 2.5
          if price < 100
          else round((price + (1.0 * em_30d)) / 5.0) * 5.0
      )
      if otm_short_strike <= otm_long_strike:
        otm_short_strike = otm_long_strike + 5.0

      # Probability of Touching Long Strike
      prob_touch_long = (
          min(85, int(round((1 - (otm_long_strike - price) / em_30d) * 50 + 40)))
          if em_30d > 0
          else 50
      )
      prob_touch_long = max(35, prob_touch_long)

      is_cyclical = (
          sector in ["Energy", "Basic Materials"]
          or t in METALS_MINERS
          or "Steel" in f.get("industry", "")
          or "Mining" in f.get("industry", "")
      )

      if is_cyclical and t not in ["GLD", "SLV", "IAU"]:
        tier = "⚠️ Tier 3 (Cyclical Trap)"
        tier_reason = "Commodity Price-Taker / Cycle Peak Risk"
      elif (pd.notnull(implied_growth) and implied_growth < 8.0) or (
          beta < 0.70 and dy > 2.5
      ):
        tier = "🐢 Tier 2 (Slow Clunker)"
        tier_reason = (
            "Sluggish EPS Velocity"
            f" ({round(implied_growth, 1) if pd.notnull(implied_growth) else '-'}%)"
        )
      elif (
          (pd.notnull(pegy) and pegy <= 1.1)
          or (pd.notnull(peg) and peg <= 1.1)
          or (t in ["GLD", "SLV"] and m_ftc_up)
      ):
        tier = "🏆 Tier 1 (Prime Setup)"
        tier_reason = (
            f"Strong EPS Velocity"
            f" ({round(implied_growth, 1) if pd.notnull(implied_growth) else '-'}%)"
            " + Lynch Bargain"
        )
      else:
        tier = "🐢 Tier 2 (Slow Clunker)"
        tier_reason = "Moderate Velocity"

      # Score Engine
      score = 0
      if price > sma200:
        score += 10
      if price > sma50:
        score += 10
      if price > ema21:
        score += 10
      if m_ftc_up:
        score += 15
      if weekly_inside:
        score += 15
      elif weekly_strat == "2U (Breakout)":
        score += 10
      if is_squeezed:
        score += 20

      metric = pegy if pd.notnull(pegy) else peg
      if pd.notnull(metric) and metric <= 1.0:
        score += 20
      elif pd.notnull(metric) and metric <= 1.5:
        score += 10
      elif t in METALS_MINERS:
        score += 15 if m_ftc_up and price > sma50 else 5

      deep_dive_records[t] = {
          "name": f.get("short_name", t),
          "tier": tier,
          "tier_reason": tier_reason,
          "price": price,
          "pe": pe,
          "peg": peg,
          "pegy": pegy,
          "yield": dy,
          "velocity": implied_growth,
          "score": score,
          "weekly_strat": weekly_strat,
          "monthly_strat": m_strat,
          "ftc": "Bullish 🟢" if m_ftc_up else "Bearish 🔴",
          "squeeze": is_squeezed,
          "sma50": sma50,
          "sma200": sma200,
          "ema8": ema8,
          "ema21": ema21,
          "beta": beta,
          "hist_vol": hist_vol * 100,
          "em_30d": em_30d,
          "long_strike": otm_long_strike,
          "short_strike": otm_short_strike,
          "prob_touch": prob_touch_long,
          "sector": sector,
          "industry": f.get("industry", ""),
          "prev_w_high": prev_w["High"],
          "prev_w_low": prev_w["Low"],
      }

      results.append({
          "Tier": tier,
          "Ticker": t,
          "Price": round(price, 2),
          "OTM Long Strike": round(otm_long_strike, 1),
          "OTM Short Strike": round(otm_short_strike, 1),
          "Est. Prob Touch (%)": f"{prob_touch_long}%",
          "Lynch PEGY": round(pegy, 2) if pd.notnull(pegy) else np.nan,
          "EPS Velocity (%)": (
              round(implied_growth, 1) if pd.notnull(implied_growth) else np.nan
          ),
          "Score": int(score),
          "Squeeze": "🔥 SQUEEZE" if is_squeezed else "Standard",
          "Weekly Strat": weekly_strat,
          "Monthly FTC": "Bullish 🟢" if m_ftc_up else "Bearish 🔴",
          "Beta": round(beta, 2),
          "Sector": sector,
          "Diagnosis / Warning": tier_reason,
      })
    except Exception:
      continue

  return pd.DataFrame(results), deep_dive_records


with st.spinner(
    "Scanning universe across S&P 100, Nasdaq, & Metals for monthly OTM"
    " breakout setups..."
):
  df, deep_dive_data = run_scan(universe_list)

if not df.empty:
  filtered = df[df["Score"] >= min_score]
  if tier_filter:
    filtered = filtered[filtered["Tier"].isin(tier_filter)]
  if min_velocity > 0:
    filtered = filtered[
        (
            pd.to_numeric(filtered["EPS Velocity (%)"], errors="coerce")
            >= min_velocity
        )
        | (filtered["Sector"].str.contains("Metals", na=False))
    ]
  if filter_weekly_inside:
    filtered = filtered[filtered["Weekly Strat"] == "1 (Inside Coil)"]
  if filter_squeeze:
    filtered = filtered[filtered["Squeeze"] == "🔥 SQUEEZE"]

  filtered = filtered.sort_values(
      by=["Tier", "Score"], ascending=[True, False]
  ).reset_index(drop=True)

  c1, c2, c3, c4 = st.columns(4)
  with c1:
    st.metric("Monthly Candidates", len(filtered))
  with c2:
    st.metric(
        "🏆 Prime Tier 1 Setups",
        len(filtered[filtered["Tier"] == "🏆 Tier 1 (Prime Setup)"]),
    )
  with c3:
    st.metric(
        "⚡ Weekly 1-Bars Coiling",
        len(filtered[filtered["Weekly Strat"] == "1 (Inside Coil)"]),
    )
  with c4:
    st.metric(
        "🔥 Active Squeezes", len(filtered[filtered["Squeeze"] == "🔥 SQUEEZE"])
    )


  def style_breakout_table(val, col_type):
    if col_type == "tier":
      if "Tier 1" in str(val):
        return (
            "background-color: rgba(0, 230, 118, 0.35); color: #00E676;"
            " font-weight: bold; border-left: 4px solid #00E676;"
        )
      elif "Tier 2" in str(val):
        return (
            "background-color: rgba(255, 214, 0, 0.22); color: #FFD600;"
            " font-weight: bold;"
        )
      elif "Tier 3" in str(val):
        return (
            "background-color: rgba(255, 82, 82, 0.25); color: #FF5252;"
            " font-weight: bold; border-left: 4px solid #FF5252;"
        )
      return ""
    elif col_type == "score":
      if val >= 80:
        return (
            "background-color: rgba(0, 230, 118, 0.28); color: #00E676;"
            " font-weight: bold;"
        )
      elif val >= 60:
        return (
            "background-color: rgba(255, 214, 0, 0.22); color: #FFD600;"
            " font-weight: bold;"
        )
      else:
        return "background-color: rgba(255, 82, 82, 0.20); color: #FF5252;"
    elif col_type == "pegy":
      if pd.isna(val) or val <= 0:
        return ""
      if val < 1.0:
        return (
            "background-color: rgba(0, 230, 118, 0.32); color: #00E676;"
            " font-weight: bold;"
        )
      elif val <= 1.5:
        return "background-color: rgba(255, 214, 0, 0.22); color: #FFD600;"
      else:
        return "background-color: rgba(255, 82, 82, 0.20); color: #FF5252;"
    elif col_type == "squeeze":
      if "SQUEEZE" in str(val):
        return (
            "background-color: rgba(255, 145, 0, 0.35); color: #FF9100;"
            " font-weight: bold;"
        )
      return "color: #888888;"
    elif col_type == "strat":
      if "1" in str(val):
        return (
            "background-color: rgba(0, 230, 118, 0.25); color: #00E676;"
            " font-weight: bold;"
        )
      elif "2U" in str(val):
        return "color: #00E676; font-weight: bold;"
      elif "2D" in str(val):
        return "color: #FF5252;"
      return ""
    elif col_type == "ftc":
      if "Bullish" in str(val):
        return "background-color: rgba(0, 230, 118, 0.20); color: #00E676;"
      return "background-color: rgba(255, 82, 82, 0.20); color: #FF5252;"
    return ""


  styled_df = (
      filtered.style.map(
          lambda v: style_breakout_table(v, "tier"), subset=["Tier"]
      )
      .map(lambda v: style_breakout_table(v, "score"), subset=["Score"])
      .map(
          lambda v: style_breakout_table(v, "pegy"),
          subset=["Lynch PEGY"],
      )
      .map(lambda v: style_breakout_table(v, "squeeze"), subset=["Squeeze"])
      .map(lambda v: style_breakout_table(v, "strat"), subset=["Weekly Strat"])
      .map(lambda v: style_breakout_table(v, "ftc"), subset=["Monthly FTC"])
      .format(
          {
              "Price": "${:.2f}",
              "OTM Long Strike": "${:.2f}",
              "OTM Short Strike": "${:.2f}",
              "EPS Velocity (%)": "{:.1f}%",
              "Lynch PEGY": "{:.2f}",
              "Beta": "{:.2f}",
          },
          na_rep="-",
      )
  )

  st.subheader("📊 Monthly OTM Breakout Radar (21–35 DTE Engine)")
  st.dataframe(styled_df, use_container_width=True, height=520)

  # =========================================================================
  # SINGLE TICKER OTM DEBIT SPREAD DOSSIER
  # =========================================================================
  st.markdown("---")
  st.subheader("🔎 OTM Vertical Debit Blueprint & Execution Parameters")

  col_sel1, col_sel2 = st.columns([2, 1])
  with col_sel1:
    dropdown_options = (
        filtered["Ticker"].tolist() if not filtered.empty else universe_list
    )
    selected_ticker = st.selectbox(
        "Select a Candidate for Instant OTM Trade Architecture:",
        dropdown_options,
    )
  with col_sel2:
    custom_lookup = (
        st.text_input(
            "Or Input ANY Custom Ticker:", placeholder="e.g. NVDA, FORM, AAPL"
        )
        .strip()
        .upper()
    )

  active_t = custom_lookup if custom_lookup else selected_ticker

  if active_t and active_t in deep_dive_data:
    dossier = deep_dive_data[active_t]
    st.markdown(f"### 📋 Trade Blueprint: **{dossier['name']} ({active_t})**")
    st.markdown(
        f"**Classification:** `{dossier['tier']}` —"
        f" *{dossier['tier_reason']}* | **30-Day Expected Move (1 SD):**"
        f" `±${dossier['em_30d']:.2f}` (`{dossier['hist_vol']:.1f}% IV Proxy`)"
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
      st.metric("Current Price", f"${dossier['price']:.2f}")
    with k2:
      st.metric("Long Strike (~42 Δ)", f"${dossier['long_strike']:.2f}")
    with k3:
      st.metric("Short Strike (~22 Δ)", f"${dossier['short_strike']:.2f}")
    with k4:
      st.metric("Est. Prob of Touch", f"{dossier['prob_touch']}%")
    with k5:
      st.metric("Weekly Strat", dossier['weekly_strat'])

    c_ex1, c_ex2 = st.columns(2)
    with c_ex1:
      st.markdown("#### ⚡ Asymmetric OTM Debit Setup (20–35 DTE)")
      spread_w = dossier['short_strike'] - dossier['long_strike']
      max_cost = spread_w * 0.32
      max_p = spread_w - max_cost
      st.markdown(f"""
            * **Structure:** Buy +1 **${dossier['long_strike']:.2f} Call** / Sell -1 **${dossier['short_strike']:.2f} Call**
            * **Target Expiration:** 21 to 35 DTE (Next monthly expiration cycle)
            * **Spread Width:** **${spread_w:.2f}**
            * **Target Cost Basis ($\le$ 32% of width):** **$\le ${max_cost:.2f} Debit**
            * **Max Profit Potential:** **${max_p:.2f}** per contract (**~{max_p / max_cost * 100:.0f}% ROI**)
            * **Why it Works:** You risk a small amount of capital ($\approx 30\%$ of width) for a 2:1 or 3:1 payout if momentum triggers out of the weekly compression bar.
            """)

    with c_ex2:
      st.markdown("#### 🛑 Trade Management & Exit Rules")
      st.markdown(f"""
            * **Take-Profit Rule (Harvest at 100%–120% ROI):** When price hits or exceeds the Long Strike (**${dossier['long_strike']:.2f}**), implied volatility expansion will boost the spread. Take profit at **${max_cost * 2.1:.2f}**; do not wait until expiration Friday.
            * **Stop-Loss / Invalidation:** Close if the underlying price closes below the rising **Daily 8 EMA (${dossier['ema8']:.2f})** or prior week's low (**${dossier['prev_w_low']:.2f}**).
            * **Time Stop:** If price chops sideways for 10 trading days with no expansion, exit at 50% loss to preserve capital.
            """)
