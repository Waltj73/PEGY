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
        padding: 14px;
        border-left: 4px solid #2962FF;
        margin-bottom: 12px;
    }
    .deep-dive-box {
        background-color: #131722;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #2A2E39;
        margin-top: 15px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("⚡ Multi-Market Quarterly Breakout & Trade Tier Radar")
st.caption(
    "Automated Tier Filtering: Prime Growth vs. Slow Clunkers vs. Cyclical Value"
    " Traps + Single Ticker Deep-Dive."
)

with st.expander(
    "🎓 How the Dashboard Classifies Tiers (Never Guess Again)", expanded=False
):
  col_t1, col_t2, col_t3 = st.columns(3)
  with col_t1:
    st.markdown("### 🏆 Tier 1: Prime Compounders")
    st.markdown(
        "• **Rules:** PEG < 1.1 **AND** EPS Velocity > 10% **AND** Beta >= 0.80."
    )
    st.markdown(
        "• **Action:** Immediate candidates for 60–90 DTE In-Out Call Debit"
        " Spreads."
    )
  with col_t2:
    st.markdown("### 🐢 Tier 2: Slow Clunkers / Yield Drag")
    st.markdown("• **Rules:** EPS Velocity < 8% **OR** Beta < 0.70.")
    st.markdown(
        "• **Action:** Skip for active quarterly swing trades; price takes too"
        " long to expand."
    )
  with col_t3:
    st.markdown("### ⚠️ Tier 3: Cyclical / Value Traps")
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
      m = (
          df_monthly[t].dropna(subset=["Close"])
          if len(tickers) > 1
          else df_monthly
      )
      if len(d) < 50 or len(m) < 3:
        continue

      close_d = d["Close"]
      high_d = d["High"]
      low_d = d["Low"]
      price = close_d.iloc[-1]

      sma50 = close_d.rolling(50).mean().iloc[-1]
      sma200 = (
          close_d.rolling(200).mean().iloc[-1] if len(d) >= 200 else sma50
      )
      ema21 = close_d.ewm(span=21, adjust=False).mean().iloc[-1]

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
            " / Low Beta"
        )
      elif (
          (pd.notnull(pegy) and pegy <= 1.1)
          or (pd.notnull(peg) and peg <= 1.1)
          or (t in ["GLD", "SLV"] and ftc_up)
      ):
        tier = "🏆 Tier 1 (Prime Setup)"
        tier_reason = (
            f"Strong EPS Velocity"
            f" ({round(implied_growth, 1) if pd.notnull(implied_growth) else '-'}%)"
            " + Lynch Bargain"
        )
      else:
        tier = "🐢 Tier 2 (Slow Clunker)"
        tier_reason = "Moderate EPS Velocity"

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

      # Save full technical and fundamental record for the dossier
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
          "strat": strat,
          "ftc": "Bullish 🟢" if ftc_up else "Bearish 🔴",
          "squeeze": is_squeezed,
          "sma50": sma50,
          "sma200": sma200,
          "ema21": ema21,
          "beta": beta,
          "payout": f.get("payout", 0.0),
          "profit_margins": f.get("profit_margins", 0.0),
          "fcf": f.get("free_cashflow", 0),
          "sector": sector,
          "industry": f.get("industry", ""),
          "prior_m_high": prev_m["High"],
          "prior_m_low": prev_m["Low"],
          "curr_m_open": curr_m["Open"],
      }

      results.append({
          "Tier": tier,
          "Ticker": t,
          "Price": round(price, 2),
          "Lynch PEGY": round(pegy, 2) if pd.notnull(pegy) else np.nan,
          "Standard PEG": round(peg, 2) if pd.notnull(peg) else np.nan,
          "Score": int(score),
          "Squeeze": "🔥 SQUEEZE" if is_squeezed else "Standard",
          "Monthly Strat": strat,
          "Monthly FTC": "Bullish 🟢" if ftc_up else "Bearish 🔴",
          "EPS Velocity (%)": (
              round(implied_growth, 1) if pd.notnull(implied_growth) else np.nan
          ),
          "Yield (%)": round(dy, 2) if dy > 0 else 0.0,
          "Beta": round(beta, 2),
          "Sector": sector,
          "Diagnosis / Warning": tier_reason,
      })
    except Exception:
      continue
  return pd.DataFrame(results), deep_dive_records


with st.spinner("Scanning universe across S&P 100, Nasdaq, & Metals..."):
  df, deep_dive_data = run_scan(universe_list)

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
              "EPS Velocity (%)": "{:.1f}%",
              "Lynch PEGY": "{:.2f}",
              "Standard PEG": "{:.2f}",
              "Yield (%)": "{:.2f}%",
              "Beta": "{:.2f}",
          },
          na_rep="-",
      )
  )

  st.subheader("📊 Trade Tier Intelligence Radar")
  st.dataframe(styled_df, use_container_width=True, height=520)

  # =========================================================================
  # SINGLE TICKER DEEP-DIVE DOSSIER SECTION
  # =========================================================================
  st.markdown("---")
  st.subheader("🔎 Single Ticker Intelligence Dossier & Trade Playbook")

  col_sel1, col_sel2 = st.columns([2, 1])
  with col_sel1:
    dropdown_options = (
        filtered["Ticker"].tolist() if not filtered.empty else universe_list
    )
    selected_ticker = st.selectbox(
        "Select a Scanned Candidate for Instant Dossier:", dropdown_options
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

  if active_t:
    if active_t in deep_dive_data:
      dossier = deep_dive_data[active_t]
    else:
      with st.spinner(
          f"Generating on-demand intelligence dossier for {active_t}..."
      ):
        tk = yf.Ticker(active_t)
        h_d = tk.history(period="1y", interval="1d")
        h_m = tk.history(period="2y", interval="1mo")
        info = tk.info or {}
        if not h_d.empty and not h_m.empty:
          c_p = h_d["Close"].iloc[-1]
          s50 = h_d["Close"].rolling(50).mean().iloc[-1]
          s200 = (
              h_d["Close"].rolling(200).mean().iloc[-1]
              if len(h_d) >= 200
              else s50
          )
          e21 = h_d["Close"].ewm(span=21, adjust=False).mean().iloc[-1]
          f_pe = info.get("forwardPE", np.nan)
          t_pe = info.get("trailingPE", np.nan)
          p_peg = info.get("pegRatio", np.nan)
          r_yield = info.get("dividendYield") or 0.0
          d_yield = r_yield * 100.0 if r_yield < 0.20 else r_yield
          pe_val = f_pe if pd.notnull(f_pe) and f_pe > 0 else t_pe

          imp_gr = (
              pe_val / p_peg
              if (pd.notnull(p_peg) and p_peg > 0 and pd.notnull(pe_val))
              else np.nan
          )
          l_pegy = (
              pe_val / (imp_gr + d_yield)
              if (pd.notnull(imp_gr) and (imp_gr + d_yield) > 0)
              else np.nan
          )

          cur_m = h_m.iloc[-1]
          pr_m = h_m.iloc[-2]
          f_up = cur_m["Close"] > cur_m["Open"]
          ins = (cur_m["High"] <= pr_m["High"]) and (cur_m["Low"] >= pr_m["Low"])
          st_pat = (
              "1 (Inside Coil)"
              if ins
              else (
                  "2U (Continuation)"
                  if cur_m["High"] > pr_m["High"]
                  else "2D (Pullback)"
              )
          )

          bb_m = h_d["Close"].rolling(20).mean()
          bb_s = h_d["Close"].rolling(20).std()
          b_u = bb_m + (2 * bb_s)
          b_l = bb_m - (2 * bb_s)
          tr_s = pd.concat([
              h_d["High"] - h_d["Low"],
              (h_d["High"] - h_d["Close"].shift(1)).abs(),
              (h_d["Low"] - h_d["Close"].shift(1)).abs(),
          ],
                           axis=1).max(axis=1)
          k_u = bb_m + (1.5 * tr_s.rolling(20).mean())
          k_l = bb_m - (1.5 * tr_s.rolling(20).mean())
          is_sq = bool(
              (b_l.iloc[-1] > k_l.iloc[-1]) and (b_u.iloc[-1] < k_u.iloc[-1])
          )

          dossier = {
              "name": info.get("shortName", active_t),
              "tier": (
                  "🏆 Tier 1 (Prime Setup)"
                  if pd.notnull(l_pegy) and l_pegy <= 1.1
                  else "🐢 Tier 2 (Slow Clunker)"
              ),
              "tier_reason": "On-Demand Screen",
              "price": c_p,
              "pe": pe_val,
              "peg": p_peg,
              "pegy": l_pegy,
              "yield": d_yield,
              "velocity": imp_gr,
              "score": 80,
              "strat": st_pat,
              "ftc": "Bullish 🟢" if f_up else "Bearish 🔴",
              "squeeze": is_sq,
              "sma50": s50,
              "sma200": s200,
              "ema21": e21,
              "beta": info.get("beta", 1.0),
              "payout": (info.get("payoutRatio") or 0.0) * 100.0,
              "profit_margins": (info.get("profitMargins") or 0.0) * 100.0,
              "fcf": info.get("freeCashflow", 0),
              "sector": info.get("sector", "Other"),
              "industry": info.get("industry", ""),
              "prior_m_high": pr_m["High"],
              "prior_m_low": pr_m["Low"],
              "curr_m_open": cur_m["Open"],
          }
        else:
          dossier = None

    if dossier:
      st.markdown(
          f"### 📋 Institutional Dossier: **{dossier['name']} ({active_t})**"
      )
      st.markdown(
          f"**Classification:** `{dossier['tier']}` —"
          f" *{dossier['tier_reason']}* | **Sector:** `{dossier['sector']}`"
          f" ({dossier['industry']})"
      )

      kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
      with kpi1:
        st.metric("Current Price", f"${dossier['price']:.2f}")
      with kpi2:
        pegy_str = (
            f"{dossier['pegy']:.2f}" if pd.notnull(dossier['pegy']) else "-"
        )
        st.metric(
            "Lynch PEGY",
            pegy_str,
            delta=(
                "Bargain (<1.0)"
                if pd.notnull(dossier['pegy']) and dossier['pegy'] < 1.0
                else "Stretched"
            ),
            delta_color="normal",
        )
      with kpi3:
        peg_str = f"{dossier['peg']:.2f}" if pd.notnull(dossier['peg']) else "-"
        st.metric("Standard PEG", peg_str)
      with kpi4:
        velo_str = (
            f"{dossier['velocity']:.1f}%"
            if pd.notnull(dossier['velocity'])
            else "-"
        )
        st.metric("EPS Velocity", velo_str)
      with kpi5:
        sq_disp = "🔥 In Squeeze" if dossier['squeeze'] else "Standard"
        st.metric("Volatility Squeeze", sq_disp)

      tab_exec, tab_strat, tab_fund = st.tabs([
          "🎯 Tactical Options Architecture",
          "📐 Multi-Timeframe Strat Geometry",
          "🏦 Fundamentals & Cash Conversion",
      ])

      with tab_exec:
        c_ex1, c_ex2 = st.columns(2)
        with c_ex1:
          st.markdown("#### ⚡ Recommended Execution Architecture")
          if "Tier 1" in dossier['tier']:
            st.success(
                "🟢 **Green Light for Defined-Risk In-Out Spread (60–90 DTE)**"
            )
            st.markdown(f"""
                        * **Trade Vehicle:** In-Out Call Debit Spread (Vertical)
                        * **Target Expiration:** 60 to 90 DTE (Capture full quarterly cycle)
                        * **Optimal Strike Placement:**
                          * **Buy Long Call:** ~65 Delta strike (Approx. **${dossier['price'] * 0.96:.2f}**)
                          * **Sell Short Call:** ~35 Delta strike (Approx. **${dossier['price'] * 1.06:.2f}**)
                        * **Target Cost Basis:** $\le$ 50% of the total spread width.
                        * **Theoretical Edge:** Greek neutralization against implied volatility crush while earnings momentum drives directional delta.
                        """)
          elif "Tier 3" in dossier['tier']:
            st.error("🔴 **Stand Aside / Execution Trap**")
            st.markdown(
                "• Commodity price-taker risk detected. Spot commodities must"
                " confirm prior to taking directional positions."
            )
          else:
            st.warning("🟡 **Tier 2 Clunker (Slow Move Warning)**")
            st.markdown(
                "• EPS velocity is sluggish; options premium may suffer from"
                " theta decay before price reaches upper targets."
            )

        with c_ex2:
          st.markdown("#### 🛑 Invalidation & Risk Parameters")
          st.markdown(f"""
                    * **Daily Invalidation Stop:** Daily closing candle below 21 EMA (**${dossier['ema21']:.2f}**) or 50 SMA (**${dossier['sma50']:.2f}**).
                    * **Monthly Pattern Invalidation:** Price prints below prior month's low (**${dossier['prior_m_low']:.2f}**).
                    * **Breakout Trigger Level:** Daily close above prior month's high (**${dossier['prior_m_high']:.2f}**).
                    * **Profit Harvest Rule:** Close 50% of the position when the spread reaches 75% of maximum profit.
                    """)

      with tab_strat:
        st.markdown("#### 📐 The Strat Higher-Timeframe Geometry")
        cs1, cs2, cs3 = st.columns(3)
        with cs1:
          st.metric("Monthly Candle Pattern", dossier['strat'])
          st.caption(
              "1 = Inside equilibrium coil | 2U = Directional continuation | 2D"
              " = Failed breakdown"
          )
        with cs2:
          st.metric("Full Timeframe Continuity (FTC)", dossier['ftc'])
          st.caption(
              "Current month trading relative to open:"
              f" ${dossier['curr_m_open']:.2f}"
          )
        with cs3:
          st.metric(
              "Daily Moving Average Stack",
              "Bullish Stack"
              if dossier['price'] > dossier['ema21'] > dossier['sma50']
              else "Mixed",
          )
          st.caption(
              f"21 EMA: ${dossier['ema21']:.2f} | 50 SMA:"
              f" ${dossier['sma50']:.2f} | 200 SMA: ${dossier['sma200']:.2f}"
          )

      with tab_fund:
        st.markdown("#### 🏦 Balance Sheet & Cash Quality Check")
        cf1, cf2, cf3, cf4 = st.columns(4)
        with cf1:
          st.metric(
              "Forward / Trailing P/E",
              (
                  f"{dossier['pe']:.1f}x"
                  if pd.notnull(dossier['pe'])
                  else "-"
              ),
          )
        with cf2:
          st.metric(
              "Dividend Yield & Payout",
              f"{dossier['yield']:.2f}%",
              f"{dossier['payout']:.1f}% Payout",
          )
        with cf3:
          st.metric("Net Profit Margin", f"{dossier['profit_margins']:.1f}%")
        with cf4:
          fcf_billions = dossier['fcf'] / 1e9 if dossier['fcf'] else 0.0
          st.metric("Trailing Free Cash Flow", f"${fcf_billions:.2f}B")
    else:
      st.error(f"Could not load data for '{active_t}'. Verify ticker symbol.")
