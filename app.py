"""
TEFAS Fund Screener — fetch-on-the-fly Streamlit app (no database)

ARCHITECTURE PIVOT from the earlier storage-backed version: no local disk,
no S3, no retention/pruning, no daily-CSV files at all. Instead, on demand
(or once per cache TTL), this fetches up to MAX_LOOKBACK_TDAYS (250) trading
days of price/investor data for the WHOLE fund universe in a single call:

    Crawler().fetch(start_date, end_date, kind="YAT", columns="info")

pytefas automatically chunks a long date range into ~28-day pieces
internally and manages TEFAS's own rate limit for you (confirmed by
inspecting the installed package directly) -- so ~250 trading days is
roughly 12-13 chunked requests, taking on the order of 2-3 minutes, not
250 separate one-day calls. That's what makes "no database" practical.

CACHING: the expensive fetch is cached via st.cache_data with a TTL (see
CACHE_TTL_HOURS) keyed only on fund `kind` -- so it always pulls the full
MAX_LOOKBACK_TDAYS window regardless of what the lookback slider is set
to. Adjusting the lookback slider, return threshold, investor-count
filter, or TOP_N afterward is instant, because those just re-slice/
re-filter the already-cached DataFrame rather than hitting the network
again. A "Refresh Data" button lets you force a new fetch before the TTL
expires.

SCREENING LOGIC (single window, not multi-horizon like the earlier
scripts): for each fund, take its last min(MAX_LOOKBACK_TDAYS, however
many days are actually available) daily log returns -- "250 or fewer" as
requested, with a MIN_OBS_FOR_SHARPE floor below which a fund is excluded
outright (too little history for a meaningful Sharpe estimate). Compute
annualized return and Sharpe ratio over that window. A fund must clear
BOTH the minimum investor count AND the minimum annualized return
threshold (the "initial filter") to be ranked at all; among survivors,
rank by Sharpe descending and show the top N.

Run with:  streamlit run tefas_streamlit_app.py
Requirements: streamlit, pytefas, pandas, numpy
"""

import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

try:
    from pytefas import Crawler, TefasError, TefasRateLimitError
except ImportError:
    Crawler = None
    TefasError = TefasRateLimitError = Exception


# ════════════════════════════════════════════════════════════════════
#  CONFIG (fixed constants; the tunable ones are sidebar controls below)
# ════════════════════════════════════════════════════════════════════
MAX_LOOKBACK_TDAYS = 250
# Hard ceiling on how much history is ever fetched -- also the size of
# the window the lookback slider can select from. Fetching is always
# sized to cover this full amount regardless of the slider, so the
# slider itself never triggers a re-fetch.

CACHE_TTL_HOURS = 12
# TEFAS publishes NAVs once per business day, so there's no point
# re-fetching more often than this. "Refresh Data" below forces an
# early refresh if you want it anyway.

ANNUALIZATION = 252

# ── Data-quality guards (same spirit as the backtest scripts) ─────────
MAX_ABS_DAILY_RETURN = 0.30
MAX_BAD_TICK_FRACTION = 0.05
MAX_INTERP_GAP_TDAYS = 5


def find_col(df, patterns):
    for pat in patterns:
        for c in df.columns:
            if re.search(pat, c, re.I):
                return c
    return None


@st.cache_resource
def get_crawler():
    return Crawler()


@st.cache_data(ttl=CACHE_TTL_HOURS * 3600, show_spinner=False)
def fetch_universe(kind, max_lookback_tdays):
    """The one expensive network call. Cached by `kind` only (and an
    implicit TTL) -- NOT by any of the screening/filter parameters, so
    tweaking those afterward never re-triggers this."""
    crawler = get_crawler()
    calendar_days_back = int(max_lookback_tdays * 1.55) + 20   # buffer for weekends/holidays
    end = date.today()
    start = end - timedelta(days=calendar_days_back)
    raw = crawler.fetch(start.isoformat(), end.isoformat(), kind=kind, columns="info")
    return raw, datetime.now()


def clean_bad_ticks(price_df):
    """Same logic as the backtest scripts: null out single-day moves
    bigger than MAX_ABS_DAILY_RETURN, interpolate short gaps, drop funds
    with too many bad ticks to trust. Returns (clean_df, dropped_funds)."""
    raw_ret = price_df.pct_change(fill_method=None)
    bad_tick = raw_ret.abs() > MAX_ABS_DAILY_RETURN
    bad_frac = bad_tick.sum() / raw_ret.notna().sum().replace(0, np.nan)
    dropped = bad_frac[bad_frac > MAX_BAD_TICK_FRACTION].index.tolist()

    clean = price_df.drop(columns=dropped) if dropped else price_df.copy()
    bad_tick = bad_tick.drop(columns=dropped) if dropped else bad_tick
    clean[bad_tick.reindex(columns=clean.columns, fill_value=False)] = np.nan
    clean = clean.interpolate(method="time", limit_area="inside", limit=MAX_INTERP_GAP_TDAYS)
    return clean, dropped


def screen_funds(price_df, investor_snapshot, lookback_tdays, min_obs,
                  min_ann_return, min_investors, risk_free, top_n):
    """The cheap, reactive part -- no network calls, just math on the
    already-fetched (and cached) price matrix."""
    daily_log_ret = np.log(price_df / price_df.shift(1))
    rows = []
    for code in price_df.columns:
        s = daily_log_ret[code].dropna()
        if len(s) < min_obs:
            continue
        window = s.iloc[-lookback_tdays:]   # last <=lookback_tdays observations
        n_obs = len(window)
        mean_daily = window.mean()
        std_daily = window.std(ddof=1)
        if not std_daily or pd.isna(std_daily) or std_daily <= 0:
            continue

        ann_ret = np.exp(mean_daily * ANNUALIZATION) - 1
        ann_vol = std_daily * np.sqrt(ANNUALIZATION)
        sharpe = (ann_ret - risk_free) / ann_vol
        total_ret = np.exp(window.sum()) - 1
        investors = int(investor_snapshot.get(code, 0))

        if ann_ret < min_ann_return:
            continue
        if investors < min_investors:
            continue

        rows.append({
            "fund_code": code,
            "n_days_used": n_obs,
            "total_return_%": round(total_ret * 100, 2),
            "ann_return_%": round(ann_ret * 100, 2),
            "ann_vol_%": round(ann_vol * 100, 2),
            "sharpe": round(sharpe, 3),
            "investors": investors,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("sharpe", ascending=False).head(top_n).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════
#  PAGE
# ════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="TEFAS Fund Screener", layout="wide")
st.title("📊 TEFAS Fund Screener — Best Sharpe Ratio Funds")
st.caption("Fetches fresh data on demand -- no database, no stored files. "
           f"Cached in memory for {CACHE_TTL_HOURS}h at a time so repeated "
           "interactions don't re-hit the network.")

if Crawler is None:
    st.error("`pytefas` is not installed in this environment. Add it to requirements.txt.")
    st.stop()

st.sidebar.header("Universe")
KIND = st.sidebar.selectbox("Fund kind", ["YAT", "EMK", "BYF"], index=0,
                             help="YAT=mutual funds, EMK=pension, BYF=ETF")

st.sidebar.header("Screening parameters")
LOOKBACK_TDAYS = st.sidebar.slider(
    "Lookback window (trading days)", min_value=30, max_value=MAX_LOOKBACK_TDAYS,
    value=MAX_LOOKBACK_TDAYS, step=10,
    help="Return and Sharpe are computed over each fund's last N trading "
         "days -- or however many are actually available if fewer than N "
         "(see 'Min days required' below for the cutoff on how few is too "
         "few to trust). Adjusting this doesn't re-fetch anything, it just "
         "re-slices the already-fetched data."
)
MIN_OBS_FOR_SHARPE = st.sidebar.slider(
    "Min days required to include a fund", min_value=20, max_value=MAX_LOOKBACK_TDAYS,
    value=60, step=10,
    help="Funds with fewer valid trading days than this are excluded "
         "entirely -- too little history for a meaningful Sharpe estimate, "
         "regardless of how good it looks."
)
MIN_ANN_RETURN_PCT = st.sidebar.number_input(
    "Min annualized return (%) -- initial filter", min_value=-50.0, max_value=500.0,
    value=40.0, step=5.0,
    help="A fund must clear this annualized-return bar over its lookback "
         "window to be considered at all. Annualized (not raw total "
         "return) so funds with less than the full lookback available are "
         "compared fairly against funds with a full window."
)
MIN_INVESTOR_COUNT = st.sidebar.number_input(
    "Min investor count", min_value=0, max_value=100000, value=200, step=50
)
TOP_N = st.sidebar.number_input("Funds to show", min_value=5, max_value=200, value=50, step=5)
RISK_FREE_RATE = st.sidebar.number_input(
    "Risk-free rate (annualized, %)", min_value=0.0, max_value=100.0, value=0.0, step=1.0,
    help="Subtracted from annualized return in the Sharpe numerator. Set "
         "to your actual TRY risk-free rate if you want a proper excess-return Sharpe."
) / 100.0

col_a, col_b = st.columns([1, 4])
refresh = col_a.button("🔄 Refresh Data", help="Force a new fetch, bypassing the cache TTL.")
if refresh:
    fetch_universe.clear()

with st.spinner(f"Fetching up to {MAX_LOOKBACK_TDAYS} trading days of {KIND} fund data "
                 f"-- first load can take a couple of minutes ..."):
    try:
        raw, fetched_at = fetch_universe(KIND, MAX_LOOKBACK_TDAYS)
    except TefasRateLimitError as e:
        st.error(f"TEFAS rate-limited this request: {e}. Wait a bit and try again.")
        st.stop()
    except TefasError as e:
        st.error(f"TEFAS API error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error fetching data: {e}")
        st.stop()

col_b.caption(f"Last fetched: {fetched_at.strftime('%Y-%m-%d %H:%M:%S')} "
              f"(cached up to {CACHE_TTL_HOURS}h)")

if raw is None or raw.empty:
    st.error("No data returned. Try Refresh Data, or check back later.")
    st.stop()

# ── Detect columns defensively (pytefas's own docs list these as fixed
# names, but a light regex check costs nothing and protects against a
# future schema tweak) ─────────────────────────────────────────────
date_col = find_col(raw, [r'^date$'])
code_col = find_col(raw, [r'fund_code'])
price_col = find_col(raw, [r'^price$'])
investor_col = find_col(raw, [r'investor_count'])

missing = [(l, c) for l, c in [("date", date_col), ("code", code_col), ("price", price_col)] if c is None]
if missing:
    st.error(f"Cannot detect required column(s): {[l for l, _ in missing]}. "
             f"Columns returned: {raw.columns.tolist()}")
    st.stop()

raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
raw[price_col] = pd.to_numeric(raw[price_col], errors="coerce")
raw[code_col] = raw[code_col].astype(str).str.strip().str.upper()
raw = raw.dropna(subset=[date_col, price_col])
if investor_col:
    raw[investor_col] = pd.to_numeric(raw[investor_col], errors="coerce")

price_df = (raw[[date_col, code_col, price_col]]
            .drop_duplicates(subset=[date_col, code_col], keep="last")
            .pivot(index=date_col, columns=code_col, values=price_col)
            .sort_index())
price_df.index = pd.to_datetime(price_df.index)
price_df.columns.name = None

investor_snapshot = pd.Series(dtype=float)
if investor_col:
    latest_date = raw[date_col].max()
    investor_snapshot = (
        raw[raw[date_col] == latest_date][[code_col, investor_col]]
        .dropna(subset=[investor_col])
        .drop_duplicates(subset=[code_col], keep="last")
        .set_index(code_col)[investor_col]
    )
else:
    st.warning("No investor_count column detected -- investor filter will exclude everything. "
               "Set 'Min investor count' to 0 if you want to proceed without it.")

price_df, dropped_funds = clean_bad_ticks(price_df)

st.write(f"📂 Universe: **{price_df.shape[1]}** fund codes, **{price_df.shape[0]}** trading days "
         f"loaded ({price_df.index.min().date()} → {price_df.index.max().date()})"
         + (f" · dropped {len(dropped_funds)} fund(s) with unreliable price data" if dropped_funds else ""))

results = screen_funds(
    price_df, investor_snapshot, LOOKBACK_TDAYS, MIN_OBS_FOR_SHARPE,
    MIN_ANN_RETURN_PCT / 100.0, MIN_INVESTOR_COUNT, RISK_FREE_RATE, TOP_N
)

if results.empty:
    st.warning(f"No funds passed at {MIN_ANN_RETURN_PCT:.1f}% min annualized return "
               f"and ≥{MIN_INVESTOR_COUNT} investors. Try loosening either filter.")
    st.stop()

st.divider()
st.subheader(f"🏆 Top {len(results)} by Sharpe ratio")
st.caption(f"Window: up to {LOOKBACK_TDAYS} trading days | "
           f"min annualized return ≥{MIN_ANN_RETURN_PCT:.1f}% | "
           f"min investors ≥{MIN_INVESTOR_COUNT}")
st.dataframe(results, width="stretch", hide_index=True)
