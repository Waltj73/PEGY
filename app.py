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

st.markdown(
    """
<style>
    .metric-card {
        background-color: #1E222D;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #2962FF;
        margin-bottom: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("⚡ Multi-Market Quarterly Breakout & Trade Tier Radar")
st.caption(
    "Automated Tier Filtering: Prime Growth vs. Slow Clunkers vs. Cyclical Value"
    " Traps + Implied Growth & Beta."
)

with st.expander(
    "🎓 How the Dashboard Classifies Tiers (Never Guess Again)", expanded=True
):
  col_t1, col_t2, col_t3 = st.columns(3)
  with col_t1:
    st.markdown("### 🏆 Tier 1: Prime Compounders")
    st.markdown(
        "**What it is:** High-quality secular growth, wide moats, and real"
        " earnings velocity."
    )
    st.markdown(
        "• **Rules:** PEG < 1.1 **AND** Implied Growth > 10% **AND** Beta >="
        " 0.80."
    )
    st.markdown(
        "• **Action:** Immediate candidates for 60–90 DTE In-Out Call Debit"
        " Spreads."
    )
  with col_t2:
    st.markdown("### 🐢 Tier 2: Slow Clunkers / Yield Drag")
    st.markdown(
        "**What it is:** Low beta, slow-moving dividend payers where dividend"
        " carries the entire valuation."
    )
    st.markdown("• **Rules:** Implied Growth < 8% **OR** Beta < 0.70.")
    st.markdown(
        "• **Action:** Skip for active quarterly swing trades; price takes too"
        " long to expand."
    )
  with col_t3:
    st.markdown("### ⚠️ Tier 3: Cyclical / Value Traps")
    st.markdown(
        "**What it is:** Steel, energy, or miners showing fake cheap P/E at"
        " cyclical tops."
    )
    st.markdown(
        "• **Rules:** Highly cyclical sector / commodity price-takers (CLF,"
        " NUE, FCX)."
    )
    st.markdown(
        "• **Action:** Avoid unless macro underlying spot commodity is"
        " exploding."
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
st.sidebar.header("⚡ Actionable Tier Filters")
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
filter_squeeze = st.sidebar.checkbox("Active Squeeze Only", value=False)
filter_inside = st.sidebar.checkbox("Monthly '1' Inside Bar Only", value=False)
filter_ftc = st.sidebar.checkbox("Monthly FTC UP Only", value=False)


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

    raw_yield = info.get("dividendYield") or 0.0
    div_yield = raw_yield * 100.0 if raw_yield < 0.20 else raw_yield

    sector = (
        "Metals/ETF" if ticker in METALS_MINERS else info.get("sector", "Other")
    )
    industry = info.get("industry", "")
    return (
        ticker,
        fwd_pe,
        trail_pe,
        peg,
        div_yield,
        beta,
        payout,
        rev_growth,
        sector,
        industry,
    )
  except Exception:
    return ticker, np.nan, np.nan, np.nan, 0.0, 1.0, 0.0, 0.0, "Other", ""


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
      (
          t,
          fwd_pe,
          trail_pe,
          peg,
          div_yield,
          beta,
          payout,
          rev_growth,
          sector,
          industry,
      ) = fut.result()
      fundamentals[t] = {
          "pe": fwd_pe if pd.notnull(fwd_pe) and fwd_pe > 0 else trail_pe,
          "peg": peg,
          "yield": div_yield,
          "beta": beta if pd.notnull(beta) else 1.0,
          "payout": payout,
          "rev_growth": rev_growth,
          "sector": sector,
          "industry": industry,
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
      is_squeezed = bool(
          (bb_lower.iloc[-1] > kc_lower.iloc[-1])
          and (bb_upper.iloc[-1] < kc_upper.iloc[-1])
      )

      curr_m, prev_m = m.iloc[-1], m.iloc[-2]
      ftc_up = bool(curr_m["Close"] > curr_m["Open"])
      inside = bool(
          (curr_m["High"] <= prev_m["High"]) and (curr_m["Low"] >= prev_m["Low"])
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

      # AUTOMATED TIER CLASSIFICATION
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
            "Sluggish EPS Growth"
            f" ({round(implied_growth, 1) if pd.notnull(implied_growth) else '-'}%)"
            " / Low Beta"
        )
      elif (
          (pd.notnull(pegy) and pegy <= 1.1)
          or (pd.notnull(peg) and peg <= 1.1)
          or (t in ["GLD", "SLV"] and ftc_up)
      ):
        tier = "🏆 Tier 1 (Prime Setup)"
        tier_reason = (
            f"EPS Velocity"
            f" ({round(implied_growth, 1) if pd.notnull(implied_growth) else '-'}%)"
            " + Deep Lynch Valuation"
        )
      else:
        tier = "🐢 Tier 2 (Slow Clunker)"
        tier_reason = "Average Multiple / Moderate Velocity"

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
          "Tier": tier,
          "Ticker": t,
          "Diagnosis / Warning": tier_reason,
          "Sector": sector,
          "Price": round(price, 2),
          "Score": int(score),
          "Squeeze": "🔥 SQUEEZE" if is_squeezed else "Standard",
          "Monthly Strat": strat,
          "Monthly FTC": "Bullish 🟢" if ftc_up else "Bearish 🔴",
          "Implied Growth (%)": (
              round(implied_growth, 1) if pd.notnull(implied_growth) else np.nan
          ),
          "Lynch PEGY": round(pegy, 2) if pd.notnull(pegy) else np.nan,
          "Standard PEG": round(peg, 2) if pd.notnull(peg) else np.nan,
          "Yield (%)": round(dy, 2) if dy > 0 else 0.0,
          "Beta": round(beta, 2),
      })
    except Exception:
      continue
  return pd.DataFrame(results)


with st.spinner("Scanning universe across S&P 100, Nasdaq, & Metals..."):
  df = run_scan(universe_list)

if not df.empty:
  filtered = df[df["Score"] >= min_score]
  if tier_filter:
    filtered = filtered[filtered["Tier"].isin(tier_filter)]
  if filter_squeeze:
    filtered = filtered[filtered["Squeeze"] == "🔥 SQUEEZE"]
  if filter_inside:
    filtered = filtered[filtered["Monthly Strat"] == "1 (Inside Coil)"]
  if filter_ftc:
    filtered = filtered[filtered["Monthly FTC"] == "Bullish 🟢"]

  filtered = filtered.sort_values(
      by=["Tier", "Score"], ascending=[True, False]
  ).reset_index(drop=True)

  c1, c2, c3, c4 = st.columns(4)
  with c1:
    st.metric("Filtered Candidates", len(filtered))
  with c2:
    st.metric(
        "🏆 Prime Tier 1 Setups",
        len(filtered[filtered["Tier"] == "🏆 Tier 1 (Prime Setup)"]),
    )
  with c3:
    st.metric(
        "🐢 Tier 2 Clunkers",
        len(filtered[filtered["Tier"] == "🐢 Tier 2 (Slow Clunker)"]),
    )
  with c4:
    st.metric(
        "⚠️ Tier 3 Cyclical Traps",
        len(filtered[filtered["Tier"] == "⚠️ Tier 3 (Cyclical Trap)"]),
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
          subset=["Lynch PEGY", "Standard PEG"],
      )
      .map(lambda v: style_breakout_table(v, "squeeze"), subset=["Squeeze"])
      .map(lambda v: style_breakout_table(v, "strat"), subset=["Monthly Strat"])
      .map(lambda v: style_breakout_table(v, "ftc"), subset=["Monthly FTC"])
      .format(
          {
              "Price": "${:.2f}",
              "Implied Growth (%)": "{:.1f}%",
              "Lynch PEGY": "{:.2f}",
              "Standard PEG": "{:.2f}",
              "Yield (%)": "{:.2f}%",
              "Beta": "{:.2f}",
          },
          na_rep="-",
      )
  )

  st.subheader("📊 Trade Tier Intelligence Radar")
  st.dataframe(styled_df, use_container_width=True, height=650)
