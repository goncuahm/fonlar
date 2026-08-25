"""
TEFAS Fund Screener — fetch-on-the-fly Streamlit app (no database)

No local disk, no S3, no daily-CSV files. On demand, this fetches trading
data for the WHOLE fund universe in a single call:

    Crawler().fetch(start_date, end_date, kind="YAT", columns="info")

pytefas auto-chunks a long date range into ~28-day pieces internally and
manages TEFAS's own rate limit for you (confirmed by inspecting the
installed package) -- so 200 trading days is ~10-11 chunked requests
under the hood, not 200 separate calls.

FETCH: always pulls the full MAX_LOOKBACK_TDAYS (200 trading days) window,
regardless of where the lookback slider is set. This is a direct
consequence of requiring every included fund to have at least
MIN_HISTORY_TDAYS (200) days of history (see below) -- there's no way to
verify a fund has 200 days of data without having fetched at least that
much, so a "fetch only 125 days for speed" shortcut is no longer
compatible with that requirement. The lookback slider is still free and
instant to move (30-200, 5-day increments) since it only re-slices the
already-fetched data -- it just means the FIRST load each session always
takes roughly proportionally as long as MAX_LOOKBACK_TDAYS implies, not a
faster 125-day path.

MATURITY FILTER: a fund needs at least MIN_HISTORY_TDAYS (200) valid
trading days of price history to be considered AT ALL, independent of
the lookback slider. This is a fixed constant, not a UI control -- it's
a data-quality/maturity floor, not something meant to be tuned away.
Because MIN_HISTORY_TDAYS equals MAX_LOOKBACK_TDAYS, any fund that
passes this gate automatically has enough history for ANY lookback
selection up to 200 -- no partial/truncated windows are possible anymore.

SCREENING: a fund needs enough investors AND to clear an annualized
return floor (the "initial filter") over its own lookback window (up to
the slider value, or fewer days if that's all that's available -- subject
to a minimum-days floor below which it's excluded outright) to be ranked
at all. Survivors are ranked by Sharpe ratio.

TOP-10 DEEP DIVE: independent of how many rows the main table shows,
the top 10 by Sharpe get a scaled cumulative-price chart (all starting
at 1.0), a money-flow table (1/5/10/20-day, as %-of-AUM using the
shares-outstanding signal), and a return-stats table (Sharpe, max
drawdown, Calmar, annualized return, annualized vol).

Run with:  streamlit run tefas_streamlit_app.py
Requirements: streamlit, pytefas, pandas, numpy, matplotlib
"""

import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    from pytefas import Crawler, TefasError, TefasRateLimitError
except ImportError:
    Crawler = None
    TefasError = TefasRateLimitError = Exception


# ════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════
MAX_LOOKBACK_TDAYS = 200
# Also the fetch size, always -- see module docstring for why the tiered
# fetch approach is no longer compatible with MIN_HISTORY_TDAYS below.

MIN_HISTORY_TDAYS = 200
# NEW: a fund must have at least this many valid trading days of price
# history to be included AT ALL, regardless of the lookback slider.
# Fixed on purpose, not a sidebar control -- this is a maturity/data-
# quality floor ("don't show funds that haven't been around long
# enough to trust"), not something meant to be dialed down.

CACHE_TTL_HOURS = 12
ANNUALIZATION = 252

MAX_ABS_DAILY_RETURN = 0.30
MAX_BAD_TICK_FRACTION = 0.05
MAX_INTERP_GAP_TDAYS = 5

FLOW_WINDOWS_TDAYS = [1, 5, 10, 20]


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
def fetch_universe(kind, lookback_tdays):
    """The one expensive network call -- always fetches the full
    MAX_LOOKBACK_TDAYS window (see module docstring), cached per `kind`."""
    crawler = get_crawler()
    calendar_days_back = int(lookback_tdays * 1.55) + 20
    end = date.today()
    start = end - timedelta(days=calendar_days_back)
    raw = crawler.fetch(start.isoformat(), end.isoformat(), kind=kind, columns="info")
    return raw, datetime.now()


def clean_bad_ticks(price_df):
    raw_ret = price_df.pct_change(fill_method=None)
    bad_tick = raw_ret.abs() > MAX_ABS_DAILY_RETURN
    bad_frac = bad_tick.sum() / raw_ret.notna().sum().replace(0, np.nan)
    dropped = bad_frac[bad_frac > MAX_BAD_TICK_FRACTION].index.tolist()

    clean = price_df.drop(columns=dropped) if dropped else price_df.copy()
    bad_tick = bad_tick.drop(columns=dropped) if dropped else bad_tick
    clean[bad_tick.reindex(columns=clean.columns, fill_value=False)] = np.nan
    clean = clean.interpolate(method="time", limit_area="inside", limit=MAX_INTERP_GAP_TDAYS)
    return clean, dropped


def screen_funds(price_df, investor_snapshot, lookback_tdays, min_history_tdays,
                  min_ann_return, min_investors, risk_free, top_n):
    """Returns a DataFrame with one row per qualifying fund: return,
    vol, Sharpe, max drawdown, Calmar -- ranked by Sharpe descending."""
    daily_log_ret = np.log(price_df / price_df.shift(1))
    rows = []
    for code in price_df.columns:
        # NEW: check history in terms of PRICE observations, not the
        # derived daily-return series -- returns are day-over-day diffs,
        # so a fund with exactly N valid prices only has N-1 valid
        # returns. Checking the return series' length here would
        # silently require N+1 days of price data to pass, not N.
        if price_df[code].notna().sum() < min_history_tdays:
            continue
        s = daily_log_ret[code].dropna()
        window = s.iloc[-lookback_tdays:]
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

        nav = np.exp(window.cumsum())
        max_dd = (nav / nav.cummax() - 1).min()
        calmar = (ann_ret / abs(max_dd)) if max_dd < 0 else np.nan

        rows.append({
            "fund_code": code,
            "n_days_used": n_obs,
            "total_return_%": round(total_ret * 100, 2),
            "ann_return_%": round(ann_ret * 100, 2),
            "ann_vol_%": round(ann_vol * 100, 2),
            "sharpe": round(sharpe, 3),
            "max_drawdown_%": round(max_dd * 100, 2),
            "calmar": round(calmar, 3) if pd.notna(calmar) else None,
            "investors": investors,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("sharpe", ascending=False).head(top_n).reset_index(drop=True)


def money_flow_table(price_df, shares_df, fund_codes, windows_tdays):
    """Daily flow = Δshares_outstanding * price (independent of price
    moves -- the cleanest signal, same convention as the backtest
    scripts), normalized to %-of-prior-day-AUM, then SUMMED over each
    window (not averaged) since these are different-length snapshots,
    not one rolling figure."""
    codes = [c for c in fund_codes if c in shares_df.columns and c in price_df.columns]
    if not codes:
        return pd.DataFrame()

    shares = shares_df[codes]
    price = price_df[codes]
    daily_flow = shares.diff() * price
    prior_aum = shares.shift(1) * price.shift(1)
    flow_pct = daily_flow / prior_aum.replace(0, np.nan)

    rows = []
    for code in codes:
        s = flow_pct[code].dropna()
        row = {"fund_code": code}
        for w in windows_tdays:
            window = s.iloc[-w:]
            row[f"flow_{w}d_%"] = round(window.sum() * 100, 2) if len(window) else None
        rows.append(row)
    return pd.DataFrame(rows)


def plot_scaled_prices(price_df, fund_codes, lookback_tdays, title):
    codes = [c for c in fund_codes if c in price_df.columns]
    window_prices = price_df[codes].iloc[-lookback_tdays:]
    first_valid = window_prices.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    scaled = window_prices.div(first_valid)

    fig, ax = plt.subplots(figsize=(13, 6))
    cmap = plt.cm.tab10
    for idx, code in enumerate(codes):
        s = scaled[code].dropna()
        ax.plot(s.index, s.values, lw=1.8, alpha=0.9, color=cmap(idx % 10), label=code)

    ax.axhline(1.0, color="black", lw=0.8, linestyle="--", alpha=0.35)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative scaled price (start = 1.0)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════
#  PAGE
# ════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="TEFAS Fund Screener", layout="wide")
st.title("📊 TEFAS Fund Screener — Best Sharpe Ratio Funds")
st.caption("Fetches fresh data on demand -- no database, no stored files. "
           f"Cached in memory for {CACHE_TTL_HOURS}h at a time so repeated "
           "interactions don't re-hit the network.")
st.caption(f"ℹ️ Only funds with at least {MIN_HISTORY_TDAYS} trading days of "
           f"price history are ever considered (fixed, not adjustable).")

if Crawler is None:
    st.error("`pytefas` is not installed in this environment. Add it to requirements.txt.")
    st.stop()

st.sidebar.header("Universe")
KIND = st.sidebar.selectbox("Fund kind", ["YAT", "EMK", "BYF"], index=0,
                             help="YAT=mutual funds, EMK=pension, BYF=ETF")

st.sidebar.header("Screening parameters")
LOOKBACK_TDAYS = st.sidebar.slider(
    "Lookback window (trading days)", min_value=30, max_value=MAX_LOOKBACK_TDAYS,
    value=125, step=5,
    help=f"Return/Sharpe computed over each fund's last N trading days. "
         f"Every fund shown already has at least {MIN_HISTORY_TDAYS} days of "
         f"history (see the maturity filter above), so this is always a "
         f"full, untruncated window -- and moving this slider is instant, "
         f"it never re-fetches (the full {MAX_LOOKBACK_TDAYS}-day dataset "
         f"is always fetched upfront)."
)
MIN_ANN_RETURN_PCT = st.sidebar.number_input(
    "Min annualized return (%) -- initial filter", min_value=-50.0, max_value=500.0,
    value=60.0, step=5.0,
    help="A fund must clear this annualized-return bar over its lookback "
         "window to be considered at all."
)
MIN_INVESTOR_COUNT = st.sidebar.number_input(
    "Min investor count", min_value=0, max_value=100000, value=200, step=50
)
TOP_N = st.sidebar.number_input("Funds to show in main table", min_value=5, max_value=200, value=50, step=5)
RISK_FREE_RATE = st.sidebar.number_input(
    "Risk-free rate (annualized, %)", min_value=0.0, max_value=100.0, value=0.0, step=1.0
) / 100.0

col_a, col_b = st.columns([1, 4])
if col_a.button("🔄 Refresh Data", help="Force a new fetch, bypassing the cache TTL."):
    fetch_universe.clear()

with st.spinner(f"Fetching {MAX_LOOKBACK_TDAYS} trading days of {KIND} fund data "
                 f"-- this can take a couple of minutes ..."):
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

date_col = find_col(raw, [r'^date$'])
code_col = find_col(raw, [r'fund_code'])
price_col = find_col(raw, [r'^price$'])
investor_col = find_col(raw, [r'investor_count'])
shares_col = find_col(raw, [r'shares_outstanding'])

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
if shares_col:
    raw[shares_col] = pd.to_numeric(raw[shares_col], errors="coerce")

price_df = (raw[[date_col, code_col, price_col]]
            .drop_duplicates(subset=[date_col, code_col], keep="last")
            .pivot(index=date_col, columns=code_col, values=price_col)
            .sort_index())
price_df.index = pd.to_datetime(price_df.index)
price_df.columns.name = None

shares_df = None
if shares_col:
    shares_df = (raw[[date_col, code_col, shares_col]]
                 .drop_duplicates(subset=[date_col, code_col], keep="last")
                 .pivot(index=date_col, columns=code_col, values=shares_col)
                 .sort_index()
                 .reindex(columns=price_df.columns))
    shares_df.index = pd.to_datetime(shares_df.index)
    shares_df.columns.name = None
else:
    st.info("No shares_outstanding column detected -- money-flow table will be skipped.")

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
    price_df, investor_snapshot, LOOKBACK_TDAYS, MIN_HISTORY_TDAYS,
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

# ── Top-10 deep dive ────────────────────────────────────────────────
top10 = results.head(10)
top10_codes = top10["fund_code"].tolist()

st.divider()
st.subheader(f"🔍 Top {len(top10)} deep dive")

st.pyplot(plot_scaled_prices(
    price_df, top10_codes, LOOKBACK_TDAYS,
    f"Top {len(top10)} funds by Sharpe — cumulative scaled price "
    f"(last {LOOKBACK_TDAYS} trading days, start = 1.0)"
))

st.markdown("**Money flow (% of prior-day AUM, summed over each window)**")
if shares_df is not None:
    flow_df = money_flow_table(price_df, shares_df, top10_codes, FLOW_WINDOWS_TDAYS)
    st.dataframe(flow_df, width="stretch", hide_index=True)
else:
    st.caption("Skipped -- no shares_outstanding data available this fetch.")

st.markdown("**Return stats**")
st.dataframe(
    top10[["fund_code", "sharpe", "max_drawdown_%", "calmar", "ann_return_%", "ann_vol_%"]],
    width="stretch", hide_index=True
)
