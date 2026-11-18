from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Dual Engine: OTM Radar, Pre-Flight & Core Equity Picker",
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

METALS_MINERS = [
    "GLD", "SLV", "IAU", "PPLT", "CPER", "NEM", "GOLD", "AEM", "PAAS", "AG",
    "FSM", "HL", "CDE", "FCX", "SCCO", "NUE", "STLD", "AA", "CLF",
]


# =============================================================================
# VERIFIED LOGNORMAL FIRST-PASSAGE BARRIER MATH
# =============================================================================
def exact_touch_prob(spot, barrier, sigma, t_years, r=0.0425):
    """Exact analytical First-Passage Probability of Touch under GBM with drift."""
    if barrier == spot or t_years <= 0 or sigma <= 0:
        return 1.0
    mu = r - 0.5 * (sigma ** 2)
    sig_sqrt_t = sigma * np.sqrt(t_years)
    if barrier > spot:
        ln_ratio = np.log(barrier / spot)
        d1 = (-ln_ratio + mu * t_years) / sig_sqrt_t
        d2 = (-ln_ratio - mu * t_years) / sig_sqrt_t
        power = (2.0 * mu) / (sigma ** 2)
        prob = norm.cdf(d1) + ((barrier / spot) ** power) * norm.cdf(d2)
    else:
        ln_ratio = np.log(spot / barrier)
        d1 = (-ln_ratio - mu * t_years) / sig_sqrt_t
        d2 = (-ln_ratio + mu * t_years) / sig_sqrt_t
        power = (2.0 * mu) / (sigma ** 2)
        prob = norm.cdf(d1) + ((barrier / spot) ** power) * norm.cdf(d2)
    return float(np.clip(prob, 0.0, 1.0))


def solve_barrier_for_prob(spot, target_prob, sigma, t_years, direction="UP", r=0.0425):
    """Solves exact barrier level with `target_prob` of being touched over `t_years`."""
    try:
        if direction == "UP":
            return float(
                brentq(
                    lambda b: exact_touch_prob(spot, b, sigma, t_years, r) - target_prob,
                    spot * 1.0001,
                    spot * 5.0,
                )
            )
        else:
            return float(
                brentq(
                    lambda b: exact_touch_prob(spot, b, sigma, t_years, r) - target_prob,
                    spot * 0.05,
                    spot * 0.9999,
                )
            )
    except Exception:
        return spot


# =============================================================================
# TICKER UNIVERSES
# =============================================================================
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
            "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMZN",
            "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "C", "CAT",
            "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX",
            "DE", "DHR", "DIS", "DUK", "EMR", "FDX", "GD", "GE", "GILD", "GM",
            "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU", "JNJ",
            "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ",
            "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE",
            "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL",
            "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TJX",
            "TMO", "TMUS", "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V",
            "VZ", "WFC", "WMT", "XOM",
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
            "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD",
            "AMGN", "AMZN", "ANSS", "APP", "ARM", "ASML", "AVGO", "AZN", "BIIB",
            "BKNG", "BKR", "CDNS", "CEG", "CHTR", "CMCSA", "COST", "CPRT",
            "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTSH", "DASH", "DDOG",
            "DLTR", "DXCM", "EA", "EXC", "FANG", "FAST", "FTNT", "GEHC", "GFS",
            "GILD", "GOOG", "GOOGL", "HON", "IDXX", "ILMN", "INTC", "INTU",
            "ISRG", "KDP", "KHC", "KLAC", "LRCX", "LULU", "MAR", "MCHP", "MDB",
            "MDLZ", "MELI", "META", "MNST", "MRNA", "MRVL", "MSFT", "MU",
            "NFLX", "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX",
            "PCAR", "PDD", "PEP", "PYPL", "QCOM", "REGN", "ROP", "ROST",
            "SBUX", "SNPS", "TEAM", "TMUS", "TSLA", "TTD", "TTWO", "TXN",
            "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS",
        ]


# =============================================================================
# DATA FETCHING CACHES
# =============================================================================
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
        free_cashflow = info.get("freeCashflow", 0) or 0
        de_ratio = info.get("debtToEquity", np.nan)

        raw_yield = info.get("dividendYield") or 0.0
        div_yield = raw_yield * 100.0 if raw_yield < 0.20 else raw_yield

        sector = "Metals/ETF" if ticker in METALS_MINERS else info.get("sector", "Other")
        industry = info.get("industry", "")
        short_name = info.get("shortName", ticker)
        return (
            ticker, fwd_pe, trail_pe, peg, div_yield, beta, payout,
            rev_growth, profit_margins, free_cashflow, de_ratio, sector, industry, short_name
        )
    except Exception:
        return (
            ticker, np.nan, np.nan, np.nan, 0.0, 1.0, 0.0,
            0.0, 0.0, 0, np.nan, "Other", "", ticker
        )


@st.cache_data(ttl=1800)
def run_scan(tickers):
    if not tickers:
        return pd.DataFrame(), {}

    scan_list = list(set(tickers + ["SPY"]))

    df_daily = yf.download(
        scan_list, period="1y", interval="1d", group_by="ticker",
        auto_adjust=True, threads=True, progress=False
    )
    df_weekly = yf.download(
        tickers, period="2y", interval="1wk", group_by="ticker",
        auto_adjust=True, threads=True, progress=False
    )
    df_monthly = yf.download(
        tickers, period="2y", interval="1mo", group_by="ticker",
        auto_adjust=True, threads=True, progress=False
    )

    # SPY benchmark returns for Relative Strength
    try:
        spy_close = df_daily["SPY"]["Close"].dropna()
        spy_ret_30d = (spy_close.iloc[-1] / spy_close.iloc[-30] - 1.0) * 100.0 if len(spy_close) >= 30 else 0.0
        spy_ret_60d = (spy_close.iloc[-1] / spy_close.iloc[-60] - 1.0) * 100.0 if len(spy_close) >= 60 else 0.0
    except Exception:
        spy_ret_30d, spy_ret_60d = 0.0, 0.0

    fundamentals = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_info, t): t for t in tickers}
        for fut in as_completed(futures):
            (
                t, fwd_pe, trail_pe, peg, div_yield, beta, payout,
                rev_growth, profit_margins, free_cashflow, de_ratio, sector, industry, short_name
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
                "de_ratio": de_ratio,
                "sector": sector,
                "industry": industry,
                "short_name": short_name,
            }

    results = []
    deep_dive_records = {}

    for t in tickers:
        try:
            d = df_daily[t].dropna(subset=["Close"])
            w = df_weekly[t].dropna(subset=["Close"]) if len(tickers) > 1 else df_weekly
            m = df_monthly[t].dropna(subset=["Close"]) if len(tickers) > 1 else df_monthly

            if len(d) < 65 or len(w) < 4 or len(m) < 3:
                continue

            close_d = d["Close"]
            high_d = d["High"]
            low_d = d["Low"]
            price = float(close_d.iloc[-1])

            sma20 = float(close_d.rolling(20).mean().iloc[-1])
            sma50 = float(close_d.rolling(50).mean().iloc[-1])
            sma50_prev = float(close_d.rolling(50).mean().iloc[-10])
            sma200 = float(close_d.rolling(200).mean().iloc[-1]) if len(d) >= 200 else sma50
            sma200_prev = float(close_d.rolling(200).mean().iloc[-20]) if len(d) >= 220 else sma200
            ema8 = float(close_d.ewm(span=8, adjust=False).mean().iloc[-1])
            ema21 = float(close_d.ewm(span=21, adjust=False).mean().iloc[-1])

            # Relative Strength vs SPY
            ret_30d = (price / float(close_d.iloc[-30]) - 1.0) * 100.0
            ret_60d = (price / float(close_d.iloc[-60]) - 1.0) * 100.0
            rs_30d = ret_30d - spy_ret_30d
            rs_60d = ret_60d - spy_ret_60d

            # Volatility Squeeze & Contraction Ratio (10D ATR vs 50D ATR)
            bb_mid = close_d.rolling(20).mean()
            bb_std = close_d.rolling(20).std()
            bb_upper = bb_mid + (2 * bb_std)
            bb_lower = bb_mid - (2 * bb_std)

            tr = pd.concat([
                high_d - low_d,
                (high_d - close_d.shift(1)).abs(),
                (low_d - close_d.shift(1)).abs(),
            ], axis=1).max(axis=1)
            kc_upper = bb_mid + (1.5 * tr.rolling(20).mean())
            kc_lower = bb_mid - (1.5 * tr.rolling(20).mean())
            is_squeezed = bool(
                (bb_lower.iloc[-1] > kc_lower.iloc[-1])
                and (bb_upper.iloc[-1] < kc_upper.iloc[-1])
            )

            atr10 = float(tr.rolling(10).mean().iloc[-1])
            atr50 = float(tr.rolling(50).mean().iloc[-1])
            vol_contraction_ratio = (atr10 / atr50) if atr50 > 0 else 1.0

            # Weekly Strat Structure
            curr_w, prev_w = w.iloc[-1], w.iloc[-2]
            weekly_inside = bool((curr_w["High"] <= prev_w["High"]) and (curr_w["Low"] >= prev_w["Low"]))
            weekly_strat = (
                "1 (Inside Coil)"
                if weekly_inside
                else ("2U (Breakout)" if curr_w["High"] > prev_w["High"] else "2D (Pullback)")
            )

            # Monthly Strat & FTC
            curr_m, prev_m = m.iloc[-1], m.iloc[-2]
            m_ftc_up = bool(curr_m["Close"] > curr_m["Open"])
            m_strat = (
                "1 (Inside)"
                if bool((curr_m["High"] <= prev_m["High"]) and (curr_m["Low"] >= prev_m["Low"]))
                else ("2U" if curr_m["High"] > prev_m["High"] else "2D")
            )

            f = fundamentals.get(t, {})
            pe = f.get("pe")
            peg = f.get("peg")
            dy = f.get("yield", 0.0)
            beta = f.get("beta", 1.0)
            fcf = f.get("free_cashflow", 0)
            de_ratio = f.get("de_ratio", np.nan)
            sector = f.get("sector", "Other")

            implied_growth = np.nan
            pegy = np.nan
            if pd.notnull(peg) and peg > 0 and pd.notnull(pe) and pe > 0:
                implied_growth = pe / peg
                tot_engine = implied_growth + dy
                if tot_engine > 0:
                    pegy = pe / tot_engine

            # 60-Day Log-Return Volatility (Stabilized for Verified First-Passage Barriers)
            log_returns = np.log(close_d / close_d.shift(1)).dropna()
            hv_60 = float(log_returns.tail(60).std() * np.sqrt(252)) if len(log_returns) >= 60 else 0.30
            t_30 = 30.0 / 252.0

            up_80_touch = solve_barrier_for_prob(price, 0.80, hv_60, t_30, "UP")
            up_68_touch = solve_barrier_for_prob(price, 0.68, hv_60, t_30, "UP")
            down_80_floor = solve_barrier_for_prob(price, 0.80, hv_60, t_30, "DOWN")

            em_30d = price * hv_60 * np.sqrt(t_30)

            otm_long_strike = (
                round(up_80_touch / 2.5) * 2.5
                if price < 100
                else round(up_80_touch / 5.0) * 5.0
            )
            if otm_long_strike <= price:
                otm_long_strike = price + 2.5 if price < 100 else price + 5.0

            otm_short_strike = (
                round(up_68_touch / 2.5) * 2.5
                if price < 100
                else round(up_68_touch / 5.0) * 5.0
            )
            if otm_short_strike <= otm_long_strike:
                otm_short_strike = otm_long_strike + (2.5 if price < 100 else 5.0)

            prob_touch_long = int(round(exact_touch_prob(price, otm_long_strike, hv_60, t_30) * 100))

            is_cyclical = (
                sector in ["Energy", "Basic Materials"]
                or t in METALS_MINERS
                or "Steel" in f.get("industry", "")
                or "Mining" in f.get("industry", "")
            )

            if is_cyclical and t not in ["GLD", "SLV", "IAU"]:
                tier = "⚠️ Tier 3 (Cyclical Trap)"
                tier_reason = "Commodity Price-Taker / Cycle Peak Risk"
            elif (pd.notnull(implied_growth) and implied_growth < 8.0) or (beta < 0.70 and dy > 2.5):
                tier = "🐢 Tier 2 (Slow Clunker)"
                tier_reason = f"Sluggish EPS Velocity ({round(implied_growth, 1) if pd.notnull(implied_growth) else '-'}%)"
            elif (
                (pd.notnull(pegy) and pegy <= 1.1)
                or (pd.notnull(peg) and peg <= 1.1)
                or (t in ["GLD", "SLV"] and m_ftc_up)
            ):
                tier = "🏆 Tier 1 (Prime Setup)"
                tier_reason = f"Strong EPS Velocity ({round(implied_growth, 1) if pd.notnull(implied_growth) else '-'}%) + Lynch Bargain"
            else:
                tier = "🐢 Tier 2 (Slow Clunker)"
                tier_reason = "Moderate Velocity"

            score = 0
            if price > sma200: score += 10
            if price > sma50: score += 10
            if price > ema21: score += 10
            if m_ftc_up: score += 15
            if weekly_inside: score += 15
            elif weekly_strat == "2U (Breakout)": score += 10
            if is_squeezed: score += 20

            metric = pegy if pd.notnull(pegy) else peg
            if pd.notnull(metric) and metric <= 1.0: score += 20
            elif pd.notnull(metric) and metric <= 1.5: score += 10
            elif t in METALS_MINERS: score += 15 if m_ftc_up and price > sma50 else 5

            # Core Equity Trend + GARP Evaluation
            trend_aligned = (price > sma50) and (price > sma200) and (sma50 > sma50_prev)
            rs_positive = (rs_30d > 0) and (rs_60d > 0)
            garp_valid = pd.notnull(metric) and (0 < metric <= 1.35)
            fcf_positive = (fcf > 0) or (t in ["GLD", "SLV", "IAU"])

            equity_grade = (
                "🟢 CORE BUY CANDIDATE"
                if (trend_aligned and rs_positive and garp_valid and fcf_positive)
                else (
                    "🟡 WATCHLIST (Trend OK)"
                    if (trend_aligned and fcf_positive)
                    else "🔴 AVOID / DOWNTREND"
                )
            )

            deep_dive_records[t] = {
                "name": f.get("short_name", t),
                "tier": tier,
                "tier_reason": tier_reason,
                "equity_grade": equity_grade,
                "price": price,
                "pe": pe,
                "peg": peg,
                "pegy": pegy,
                "yield": dy,
                "velocity": implied_growth,
                "fcf": fcf,
                "de_ratio": de_ratio,
                "score": score,
                "weekly_strat": weekly_strat,
                "monthly_strat": m_strat,
                "ftc": "Bullish 🟢" if m_ftc_up else "Bearish 🔴",
                "squeeze": is_squeezed,
                "vol_contraction": vol_contraction_ratio,
                "sma20": sma20,
                "sma50": sma50,
                "sma200": sma200,
                "ema8": ema8,
                "ema21": ema21,
                "trend_aligned": trend_aligned,
                "rs_30d": rs_30d,
                "rs_60d": rs_60d,
                "beta": beta,
                "hist_vol": hv_60 * 100,
                "em_30d": em_30d,
                "up_80_touch": up_80_touch,
                "up_68_touch": up_68_touch,
                "down_80_floor": down_80_floor,
                "long_strike": otm_long_strike,
                "short_strike": otm_short_strike,
                "prob_touch": prob_touch_long,
                "sector": sector,
                "industry": f.get("industry", ""),
                "prev_w_high": float(prev_w["High"]),
                "prev_w_low": float(prev_w["Low"]),
            }

            results.append({
                "Tier": tier,
                "Equity Grade": equity_grade,
                "Ticker": t,
                "Price": round(price, 2),
                "OTM Long Strike": round(otm_long_strike, 1),
                "OTM Short Strike": round(otm_short_strike, 1),
                "Est. Prob Touch (%)": f"{prob_touch_long}%",
                "Lynch PEGY": round(pegy, 2) if pd.notnull(pegy) else np.nan,
                "EPS Velocity (%)": round(implied_growth, 1) if pd.notnull(implied_growth) else np.nan,
                "RS vs SPY 60D (%)": round(rs_60d, 1),
                "80% Down Floor": round(down_80_floor, 2),
                "80% Up Touch": round(up_80_touch, 2),
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


def fetch_preflight_data(ticker_symbol):
    """Pulls fundamental metrics and assigns Go/No-Go score."""
    try:
        t = yf.Ticker(ticker_symbol)
        info = t.info or {}
        price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
        q_eps_growth = info.get("earningsQuarterlyGrowth", None)
        q_rev_growth = info.get("revenueGrowth", None)
        op_margin = info.get("operatingMargins", None)
        peg = info.get("pegRatio", None)
        fwd_pe = info.get("forwardPE", None)
        trail_pe = info.get("trailingPE", None)
        de_ratio = info.get("debtToEquity", None)
        inst_ownership = info.get("heldPercentInstitutions", None)

        score = 0.0
        if q_eps_growth is not None:
            if q_eps_growth >= 0.25: score += 1.5
            elif q_eps_growth >= 0.10: score += 0.75

        if q_rev_growth is not None:
            if q_rev_growth >= 0.20: score += 1.0
            elif q_rev_growth >= 0.10: score += 0.5

        if op_margin is not None and op_margin >= 0.15:
            score += 0.5

        if peg is not None and 0 < peg <= 2.0:
            score += 1.0
        elif fwd_pe and trail_pe and fwd_pe < trail_pe:
            score += 0.75

        if de_ratio is not None and de_ratio <= 150:
            score += 1.0
        elif de_ratio is None:
            score += 0.5

        if score >= 3.5:
            status = "🟢 GO"
        elif score >= 2.0:
            status = "🟡 WATCH"
        else:
            status = "🔴 NO-GO"

        return {
            "Status": status,
            "Ticker": ticker_symbol,
            "Score": round(score, 1),
            "Price": f"${price:.2f}" if price else "N/A",
            "YoY EPS Gr": f"{q_eps_growth * 100:.1f}%" if q_eps_growth is not None else "N/A",
            "YoY Rev Gr": f"{q_rev_growth * 100:.1f}%" if q_rev_growth is not None else "N/A",
            "Op Margin": f"{op_margin * 100:.1f}%" if op_margin is not None else "N/A",
            "PEG": f"{peg:.2f}" if peg is not None else "N/A",
            "Fwd P/E": f"{fwd_pe:.1f}" if fwd_pe is not None else "N/A",
            "D/E Ratio": f"{de_ratio:.1f}" if de_ratio is not None else "N/A",
            "Inst Own": f"{inst_ownership * 100:.0f}%" if inst_ownership is not None else "N/A",
            "_raw_score": score
        }
    except Exception:
        return None


# =============================================================================
# SHARED UNIVERSE & SCAN EXECUTION
# =============================================================================
st.sidebar.markdown("### 🎯 Universe Selection")
include_sp100 = st.sidebar.checkbox("S&P 100 Large-Caps", value=True, key="otm_sp100")
include_ndx = st.sidebar.checkbox("Nasdaq-100 Growth / Tech", value=True, key="otm_ndx")
include_metals = st.sidebar.checkbox("Metals & Commodities", value=True, key="otm_metals")
custom_extra = st.sidebar.text_input("Add Custom Tickers", value="FORM, ONTO, COHU, PLTR, URI", key="otm_custom")

selected_universe = set()
if include_sp100: selected_universe.update(load_sp100_tickers())
if include_ndx: selected_universe.update(load_nasdaq100_tickers())
if include_metals: selected_universe.update(METALS_MINERS)
if custom_extra:
    for t in custom_extra.split(","):
        if t.strip():
            selected_universe.add(t.strip().upper())

universe_list = sorted(list(selected_universe))
st.sidebar.markdown(f"**Total Monitored:** `{len(universe_list)}` tickers")

with st.spinner("Scanning universe across S&P 100, Nasdaq, & Metals..."):
    df, deep_dive_data = run_scan(universe_list)


# =============================================================================
# TABS INTERFACE
# =============================================================================
tab_equity, tab_otm, tab_preflight = st.tabs([
    "🏛️ Core Equity Stock Picker (GARP + Trend)",
    "⚡ Monthly OTM Breakout Radar",
    "🛡️ Fundamental Pre-Flight (Go / No-Go)"
])


# -----------------------------------------------------------------------------
# TAB 1 (NEW): CORE EQUITY STOCK PICKER (COMMON SHARES ONLY — ZERO EXPIRATION)
# -----------------------------------------------------------------------------
with tab_equity:
    st.title("🏛️ Core Equity Stock Picker (GARP + Institutional Trend)")
    st.caption(
        "Common-Share Stock Selection System: Combines Peter Lynch PEGY & Free Cash Flow with Rising 50/200 SMA Uptrends, "
        "Relative Strength vs. SPY, and Exact First-Passage Barrier Position Sizing."
    )

    with st.expander("📘 The 4-Pillar Equity Selection Rules (No Expiring Options)", expanded=False):
        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            st.markdown("### 1. GARP Valuation")
            st.markdown("• **Lynch PEGY ≤ 1.25:** Growth + dividend yield justifies the P/E multiple.")
            st.markdown("• **Positive Free Cash Flow:** Eliminates speculative cash-burning companies.")
        with ec2:
            st.markdown("### 2. Trend Alignment")
            st.markdown("• **Above 50 & 200 SMA:** Never catch falling knives in downtrends.")
            st.markdown("• **Rising 50-Day SMA:** Confirms active institutional accumulation.")
        with ec3:
            st.markdown("### 3. Relative Strength")
            st.markdown("• **Beating SPY (30D & 60D):** Focus only on market leaders, not laggards.")
            st.markdown("• **Volatility Contraction:** Tighter ATR coils offer low-risk entry pivots.")
        with ec4:
            st.markdown("### 4. Hard Risk Sizing")
            st.markdown("• **Common Shares Only:** Zero theta decay and zero expiration clock.")
            st.markdown("• **1% Account Risk Cap:** Position size derived directly from stop distance.")

    col_eq_f1, col_eq_f2, col_eq_f3, col_eq_f4 = st.columns(4)
    with col_eq_f1:
        max_pegy_eq = st.slider("Max Lynch PEGY", 0.5, 2.5, 1.25, 0.05, key="eq_max_pegy")
    with col_eq_f2:
        min_vel_eq = st.slider("Min EPS Velocity (%)", 0, 40, 10, 2, key="eq_min_vel")
    with col_eq_f3:
        require_uptrend = st.checkbox("Require Price > Rising 50 & 200 SMA", value=True, key="eq_uptrend")
    with col_eq_f4:
        require_rs_pos = st.checkbox("Require Outperforming SPY (60D)", value=True, key="eq_rs")

    if not df.empty:
        eq_df = df.copy()
        eq_df["PEGY_num"] = pd.to_numeric(eq_df["Lynch PEGY"], errors="coerce")
        eq_df["Vel_num"] = pd.to_numeric(eq_df["EPS Velocity (%)"], errors="coerce")

        # Apply GARP + Trend filters
        eq_filtered = eq_df[
            (eq_df["PEGY_num"].notna())
            & (eq_df["PEGY_num"] > 0)
            & (eq_df["PEGY_num"] <= max_pegy_eq)
            & (eq_df["Vel_num"] >= min_vel_eq)
        ]

        if require_uptrend:
            eq_filtered = eq_filtered[
                eq_filtered["Ticker"].map(lambda x: deep_dive_data.get(x, {}).get("trend_aligned", False))
            ]
        if require_rs_pos:
            eq_filtered = eq_filtered[eq_filtered["RS vs SPY 60D (%)"] > 0]

        eq_filtered = eq_filtered.sort_values(
            by=["Equity Grade", "RS vs SPY 60D (%)", "Lynch PEGY"],
            ascending=[True, False, True]
        ).reset_index(drop=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Passing Equity Candidates", len(eq_filtered))
        m2.metric(
            "🟢 Core Buy Candidates",
            len(eq_filtered[eq_filtered["Equity Grade"] == "🟢 CORE BUY CANDIDATE"])
        )
        m3.metric(
            "Avg Lynch PEGY",
            f"{eq_filtered['PEGY_num'].mean():.2f}" if not eq_filtered.empty else "-"
        )
        m4.metric(
            "Avg 60D RS vs SPY",
            f"+{eq_filtered['RS vs SPY 60D (%)'].mean():.1f}%" if not eq_filtered.empty else "-"
        )

        display_cols = [
            "Equity Grade", "Ticker", "Price", "Lynch PEGY", "EPS Velocity (%)",
            "RS vs SPY 60D (%)", "80% Down Floor", "80% Up Touch",
            "Weekly Strat", "Monthly FTC", "Squeeze", "Sector"
        ]

        st.subheader("📋 Ranked Common-Share Candidates (GARP + Institutional Trend)")
        st.dataframe(
            eq_filtered[display_cols].style.format({
                "Price": "${:.2f}",
                "Lynch PEGY": "{:.2f}",
                "EPS Velocity (%)": "{:.1f}%",
                "RS vs SPY 60D (%)": "{:+.1f}%",
                "80% Down Floor": "${:.2f}",
                "80% Up Touch": "${:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
            height=420,
        )

        # Position Sizing & Verified Barrier Audit Calculator
        st.markdown("---")
        st.subheader("🧮 Common-Share Position Sizer & Verified Barrier Audit")

        ps_col1, ps_col2, ps_col3, ps_col4 = st.columns([2, 1, 1, 1])
        with ps_col1:
            eq_options = eq_filtered["Ticker"].tolist() if not eq_filtered.empty else universe_list
            eq_selected = st.selectbox("Select Stock to Size:", eq_options, key="eq_size_ticker")
        with ps_col2:
            eq_custom = st.text_input("Or Any Ticker:", placeholder="e.g. NVDA, MSFT", key="eq_custom_t").strip().upper()
        with ps_col3:
            acct_size = st.number_input("Account Capital ($)", min_value=1000, value=25000, step=1000, key="eq_acct")
        with ps_col4:
            risk_pct = st.number_input("Max Risk / Trade (%)", min_value=0.25, max_value=5.0, value=1.0, step=0.25, key="eq_risk")

        target_sym = eq_custom if eq_custom else eq_selected

        if target_sym and target_sym in deep_dive_data:
            rec = deep_dive_data[target_sym]
            entry_px = rec["price"]
            # Use the tighter of 80% Downside Barrier or 50-Day SMA as structural stop
            stop_px = round(min(rec["down_80_floor"], entry_px * 0.96), 2)
            risk_per_share = max(entry_px - stop_px, 0.25)
            max_dollar_risk = acct_size * (risk_pct / 100.0)
            rec_shares = int(max_dollar_risk // risk_per_share)
            position_dollars = rec_shares * entry_px
            pct_of_acct = (position_dollars / acct_size) * 100.0

            st.markdown(
                f"### **{rec['name']} ({target_sym})** — `{rec['equity_grade']}` | "
                f"Sector: **{rec['sector']}** | 60D Stabilized Vol: **`{rec['hist_vol']:.1f}%`**"
            )

            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("Spot Price", f"${entry_px:.2f}")
            sc2.metric("80% Down Floor (Stop)", f"${stop_px:.2f}", f"-{(entry_px - stop_px) / entry_px * 100:.1f}%")
            sc3.metric("80% Up Touch (30D)", f"${rec['up_80_touch']:.2f}", f"+{(rec['up_80_touch'] - entry_px) / entry_px * 100:.1f}%")
            sc4.metric("68% Up Touch (30D)", f"${rec['up_68_touch']:.2f}", f"+{(rec['up_68_touch'] - entry_px) / entry_px * 100:.1f}%")
            sc5.metric("Recommended Shares", f"{rec_shares} shs", f"${position_dollars:,.0f} ({pct_of_acct:.1f}% acct)")

            st.info(
                f"**Risk Guardrail:** Buying **{rec_shares} common shares** of `{target_sym}` at **${entry_px:.2f}** "
                f"(**${position_dollars:,.2f}** total equity) with a hard stop at **${stop_px:.2f}** caps your "
                f"worst-case loss at **${rec_shares * risk_per_share:,.2f}** (**{risk_pct:.2f}%** of your ${acct_size:,.0f} account)."
            )


# -----------------------------------------------------------------------------
# TAB 2: MONTHLY OTM RADAR
# -----------------------------------------------------------------------------
with tab_otm:
    st.title("⚡ Monthly OTM Breakout & Probability Radar")
    st.caption("Active 20–35 DTE Swing Scanner: Weekly Coils + High Implied Velocity + Verified First-Passage Call Debit Architecture.")

    with st.expander("🎓 The OTM Debit Spread Blueprint (High-R:R Monthly Model)", expanded=False):
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.markdown("### 🎯 Strike Architecture")
            st.markdown("• **Buy Long Call:** Aligned near the 80% First-Passage Touch level.")
            st.markdown("• **Sell Short Call:** Aligned near the 68% First-Passage Touch level.")
            st.markdown("• **Cost Target:** 25%–35% of spread width (Yielding **2:1 to 3:1 R:R**).")
        with col_t2:
            st.markdown("### ⏱️ Timeframe Alignment")
            st.markdown("• **Duration:** 21 to 35 DTE (Front monthly expiration).")
            st.markdown("• **Anchor:** Weekly Strat `1` or `2U` in line with **Monthly Bullish FTC**.")
            st.markdown("• **Velocity Requirement:** EPS Velocity > 15% (Ensures real momentum drivers).")
        with col_t3:
            st.markdown("### 🛑 Trade Management")
            st.markdown("• **Exit at 100%–150% Profit:** Don't hold to max expiration; harvest early when Delta expands.")
            st.markdown("• **Stop-Loss:** Daily close below rising 8 or 21 EMA cuts trade at a defined 40%–50% loss.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Tab 2: Radar Filters")
    tier_filter = st.sidebar.multiselect(
        "Filter by Trade Tier",
        ["🏆 Tier 1 (Prime Setup)", "🐢 Tier 2 (Slow Clunker)", "⚠️ Tier 3 (Cyclical Trap)"],
        default=["🏆 Tier 1 (Prime Setup)"],
        key="otm_tier"
    )
    min_score = st.sidebar.slider("Min Setup Score", 0, 100, 50, 5, key="otm_score")
    min_velocity = st.sidebar.slider("Min EPS Velocity (%)", 0, 50, 15, 5, key="otm_velocity")
    filter_weekly_inside = st.sidebar.checkbox("Weekly '1' Inside Coil Only", value=False, key="otm_weekly_1")
    filter_squeeze = st.sidebar.checkbox("Active Daily Squeeze Only", value=False, key="otm_sqz")

    if not df.empty:
        filtered = df[df["Score"] >= min_score]
        if tier_filter:
            filtered = filtered[filtered["Tier"].isin(tier_filter)]
        if min_velocity > 0:
            filtered = filtered[
                (pd.to_numeric(filtered["EPS Velocity (%)"], errors="coerce") >= min_velocity)
                | (filtered["Sector"].str.contains("Metals", na=False))
            ]
        if filter_weekly_inside:
            filtered = filtered[filtered["Weekly Strat"] == "1 (Inside Coil)"]
        if filter_squeeze:
            filtered = filtered[filtered["Squeeze"] == "🔥 SQUEEZE"]

        filtered = filtered.sort_values(by=["Tier", "Score"], ascending=[True, False]).reset_index(drop=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Monthly Candidates", len(filtered))
        c2.metric("🏆 Prime Tier 1 Setups", len(filtered[filtered["Tier"] == "🏆 Tier 1 (Prime Setup)"]))
        c3.metric("⚡ Weekly 1-Bars Coiling", len(filtered[filtered["Weekly Strat"] == "1 (Inside Coil)"]))
        c4.metric("🔥 Active Squeezes", len(filtered[filtered["Squeeze"] == "🔥 SQUEEZE"]))

        def style_breakout_table(val, col_type):
            if col_type == "tier":
                if "Tier 1" in str(val):
                    return "background-color: rgba(0, 230, 118, 0.35); color: #00E676; font-weight: bold; border-left: 4px solid #00E676;"
                elif "Tier 2" in str(val):
                    return "background-color: rgba(255, 214, 0, 0.22); color: #FFD600; font-weight: bold;"
                elif "Tier 3" in str(val):
                    return "background-color: rgba(255, 82, 82, 0.25); color: #FF5252; font-weight: bold; border-left: 4px solid #FF5252;"
            elif col_type == "score":
                if val >= 80: return "background-color: rgba(0, 230, 118, 0.28); color: #00E676; font-weight: bold;"
                elif val >= 60: return "background-color: rgba(255, 214, 0, 0.22); color: #FFD600; font-weight: bold;"
                else: return "background-color: rgba(255, 82, 82, 0.20); color: #FF5252;"
            elif col_type == "pegy":
                if pd.isna(val) or val <= 0: return ""
                if val < 1.0: return "background-color: rgba(0, 230, 118, 0.32); color: #00E676; font-weight: bold;"
                elif val <= 1.5: return "background-color: rgba(255, 214, 0, 0.22); color: #FFD600; font-weight: bold;"
                else: return "background-color: rgba(255, 82, 82, 0.20); color: #FF5252;"
            elif col_type == "squeeze":
                if "SQUEEZE" in str(val): return "background-color: rgba(255, 145, 0, 0.35); color: #FF9100; font-weight: bold;"
                return "color: #888888;"
            elif col_type == "strat":
                if "1" in str(val): return "background-color: rgba(0, 230, 118, 0.25); color: #00E676; font-weight: bold;"
                elif "2U" in str(val): return "color: #00E676; font-weight: bold;"
                elif "2D" in str(val): return "color: #FF5252;"
            elif col_type == "ftc":
                if "Bullish" in str(val): return "background-color: rgba(0, 230, 118, 0.20); color: #00E676;"
                return "background-color: rgba(255, 82, 82, 0.20); color: #FF5252;"
            return ""

        styled_df = (
            filtered.style.map(lambda v: style_breakout_table(v, "tier"), subset=["Tier"])
            .map(lambda v: style_breakout_table(v, "score"), subset=["Score"])
            .map(lambda v: style_breakout_table(v, "pegy"), subset=["Lynch PEGY"])
            .map(lambda v: style_breakout_table(v, "squeeze"), subset=["Squeeze"])
            .map(lambda v: style_breakout_table(v, "strat"), subset=["Weekly Strat"])
            .map(lambda v: style_breakout_table(v, "ftc"), subset=["Monthly FTC"])
            .format({
                "Price": "${:.2f}",
                "OTM Long Strike": "${:.2f}",
                "OTM Short Strike": "${:.2f}",
                "EPS Velocity (%)": "{:.1f}%",
                "Lynch PEGY": "{:.2f}",
                "RS vs SPY 60D (%)": "{:+.1f}%",
                "80% Down Floor": "${:.2f}",
                "80% Up Touch": "${:.2f}",
                "Beta": "{:.2f}",
            }, na_rep="-")
        )

        st.subheader("📊 Monthly OTM Breakout Radar (21–35 DTE Engine)")
        st.dataframe(styled_df, use_container_width=True, height=520)

        st.markdown("---")
        st.subheader("🔎 OTM Vertical Debit Blueprint & Execution Parameters")

        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            dropdown_options = filtered["Ticker"].tolist() if not filtered.empty else universe_list
            selected_ticker = st.selectbox("Select a Candidate for Instant OTM Trade Architecture:", dropdown_options)
        with col_sel2:
            custom_lookup = st.text_input("Or Input ANY Custom Ticker:", placeholder="e.g. NVDA, FORM, AAPL").strip().upper()

        active_t = custom_lookup if custom_lookup else selected_ticker

        if active_t and active_t in deep_dive_data:
            dossier = deep_dive_data[active_t]
            st.markdown(f"### 📋 Trade Blueprint: **{dossier['name']} ({active_t})**")
            st.markdown(
                f"**Classification:** `{dossier['tier']}` — *{dossier['tier_reason']}* | "
                f"**60D Stabilized Vol:** `{dossier['hist_vol']:.1f}%`"
            )

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Current Price", f"${dossier['price']:.2f}")
            k2.metric("Long Strike (~80% Touch)", f"${dossier['long_strike']:.2f}")
            k3.metric("Short Strike (~68% Touch)", f"${dossier['short_strike']:.2f}")
            k4.metric("Verified Touch Prob", f"{dossier['prob_touch']}%")
            k5.metric("Weekly Strat", dossier['weekly_strat'])

            c_ex1, c_ex2 = st.columns(2)
            with c_ex1:
                st.markdown("#### ⚡ Asymmetric OTM Debit Setup (20–35 DTE)")
                spread_w = dossier['short_strike'] - dossier['long_strike']
                max_cost = spread_w * 0.32
                max_p = spread_w - max_cost
                st.markdown(f"""
- **Structure:** Buy +1 **${dossier['long_strike']:.2f} Call** / Sell -1 **${dossier['short_strike']:.2f} Call**
- **Target Expiration:** 21 to 35 DTE (Next monthly expiration cycle)
- **Spread Width:** **${spread_w:.2f}**
- **Target Cost Basis ($\le$ 32% of width):** **$\le ${max_cost:.2f} Debit**
- **Max Profit Potential:** **${max_p:.2f}** per contract (**~{max_p / max_cost * 100:.0f}% ROI**)
- **Why it Works:** Strikes are anchored to exact continuous-time first-passage barriers rather than arbitrary multiples.
                """)

            with c_ex2:
                st.markdown("#### 🛑 Trade Management & Exit Rules")
                st.markdown(f"""
- **Take-Profit Rule (Harvest at 100%–120% ROI):** When price reaches Long Strike (**${dossier['long_strike']:.2f}**), harvest at **${max_cost * 2.1:.2f}**; do not wait until expiration Friday.
- **Stop-Loss / Invalidation:** Close if the underlying closes below the **Daily 8 EMA (${dossier['ema8']:.2f})** or prior week low (**${dossier['prev_w_low']:.2f}**).
- **Time Stop:** If price chops sideways for 10 trading days with no expansion, exit at 50% loss.
                """)


# -----------------------------------------------------------------------------
# TAB 3: FUNDAMENTAL PRE-FLIGHT (GO / NO-GO)
# -----------------------------------------------------------------------------
with tab_preflight:
    st.title("🛡️ Institutional Fundamental Pre-Flight Screener")
    st.caption("Verify fundamental health & institutional sponsorship before stalking chart bases and pattern entries.")

    with st.expander("ℹ️ The Institutional Scoring Engine (Max 5.0 Points)", expanded=False):
        st.markdown("""
- **1. EPS Acceleration (+1.5 pts):** YoY EPS Growth $\ge$ +25% (+0.75 pts if $\ge$ +10%).
- **2. Top-Line Fuel (+1.0 pt):** YoY Sales Growth $\ge$ +20% (+0.5 pts if $\ge$ +10%).
- **3. Margin Expansion (+0.5 pt):** Operating Margin $\ge$ 15%.
- **4. Forward Traction (+1.0 pt):** PEG $\le$ 2.0 or Forward P/E shows significant earnings expansion.
- **5. Solvency / Balance Sheet (+1.0 pt):** Debt-to-Equity $\le$ 150 (clean balance sheet).
        
        **Thresholds:**
- 🟢 **GO ($\ge$ 3.5):** Strong institutional backing. Chart breakout setups are high-conviction.
- 🟡 **WATCH (2.0 – 3.0):** Decent metrics with one or two flaws. Demands tighter price confirmation.
- 🔴 **NO-GO (< 2.0):** Weak or decelerating fundamentals. High probability of false breakouts.
        """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛡️ Tab 3: Pre-Flight Controls")

    pf_universe_choice = st.sidebar.selectbox(
        "Screener Universe",
        ["Nasdaq-100 Top Liquid", "S&P 100 Large-Caps", "Metals & Commodities", "Custom List"],
        key="pf_univ_choice"
    )

    if pf_universe_choice == "Nasdaq-100 Top Liquid":
        pf_tickers = load_nasdaq100_tickers()[:35]
    elif pf_universe_choice == "S&P 100 Large-Caps":
        pf_tickers = load_sp100_tickers()[:35]
    elif pf_universe_choice == "Metals & Commodities":
        pf_tickers = METALS_MINERS
    else:
        pf_custom_input = st.sidebar.text_area(
            "Enter Tickers to Audit:",
            value="NVDA, TSLA, AAPL, PLTR, AMD, FORM, ONTO, COHU, CAVA",
            key="pf_custom_input"
        )
        pf_tickers = [t.strip().upper() for t in pf_custom_input.replace(",", " ").split() if t.strip()]

    pf_status_filter = st.sidebar.multiselect(
        "Filter by Verdict",
        ["🟢 GO", "🟡 WATCH", "🔴 NO-GO"],
        default=["🟢 GO", "🟡 WATCH", "🔴 NO-GO"],
        key="pf_status_filter"
    )
    pf_min_score = st.sidebar.slider("Min Pre-Flight Score", 0.0, 5.0, 0.0, 0.5, key="pf_min_score")

    if st.button("🚀 Audit Fundamentals (Run Scan)", use_container_width=True, key="pf_run_btn"):
        with st.spinner(f"Auditing {len(pf_tickers)} companies across institutional growth & solvency metrics..."):
            pf_results = []
            with ThreadPoolExecutor(max_workers=15) as executor:
                futures = [executor.submit(fetch_preflight_data, sym) for sym in pf_tickers]
                for f in as_completed(futures):
                    res = f.result()
                    if res:
                        pf_results.append(res)

            if pf_results:
                pf_df = pd.DataFrame(pf_results)
                pf_df = pf_df.sort_values(by="_raw_score", ascending=False).drop(columns=["_raw_score"])

                filtered_pf = pf_df[
                    (pf_df["Status"].isin(pf_status_filter)) &
                    (pf_df["Score"] >= pf_min_score)
                ]

                go_cnt = (pf_df["Status"] == "🟢 GO").sum()
                watch_cnt = (pf_df["Status"] == "🟡 WATCH").sum()
                nogo_cnt = (pf_df["Status"] == "🔴 NO-GO").sum()

                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                col_p1.metric("Audited Symbols", len(pf_df))
                col_p2.metric("🟢 GO (Institutional Buy)", go_cnt)
                col_p3.metric("🟡 WATCH (Conditional)", watch_cnt)
                col_p4.metric("🔴 NO-GO (Avoid Setup)", nogo_cnt)

                st.divider()

                st.dataframe(
                    filtered_pf,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Status": st.column_config.TextColumn("Verdict", width="small"),
                        "Score": st.column_config.ProgressColumn("Institutional Score", min_value=0, max_value=5, format="%.1f"),
                        "Ticker": st.column_config.TextColumn("Symbol", width="small"),
                        "YoY EPS Gr": "YoY EPS",
                        "YoY Rev Gr": "YoY Sales",
                        "Op Margin": "Op Margin",
                        "D/E Ratio": "Debt/Eq",
                        "Inst Own": "Inst Own"
                    }
                )

                csv_data = filtered_pf.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export Audit to CSV", csv_data, "preflight_audit.csv", "text/csv")
            else:
                st.error("No fundamental data returned. Check symbols or internet connection.")
