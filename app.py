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
import matplotlib.ticker as mticker
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
MONTH_TDAYS = 22   # "last month" convention used in the summary table below


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
            "total_return_%": round(total_ret * 100, 2),   # actual return over the window used -- see it first
            "n_days_used": n_obs,
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
    scripts).

    FIXED: previously summed daily flow-as-%-of-PRIOR-DAY-AUM values
    directly. That's invalid -- each day's percentage is relative to a
    DIFFERENT (shrinking or growing) base, so summing them isn't a
    meaningful cumulative figure and can produce results below -100%
    even though a fund can never lose more than 100% of its assets
    (verified: 7 days of ~-20%/day outflows summed to -132%, while the
    true compounded decline was -76.9%). Now: sum the raw TRY flow
    (dollars ARE additive, no issue there), then divide by ONE fixed
    reference -- the AUM immediately before the window started -- same
    convention as every other return figure in this app (ending vs. a
    single starting point, not a chain of shifting bases)."""
    codes = [c for c in fund_codes if c in shares_df.columns and c in price_df.columns]
    if not codes:
        return pd.DataFrame()

    shares = shares_df[codes]
    price = price_df[codes]
    daily_flow_try = shares.diff() * price
    prior_aum = shares.shift(1) * price.shift(1)

    rows = []
    for code in codes:
        flow_s = daily_flow_try[code].dropna()
        row = {"fund_code": code}
        for w in windows_tdays:
            window = flow_s.iloc[-w:]
            if len(window) == 0:
                row[f"flow_{w}d_try"] = None
                row[f"flow_{w}d_%"] = None
                continue
            total_flow = window.sum()
            base_aum = prior_aum[code].get(window.index[0], np.nan)   # AUM right before the window began
            row[f"flow_{w}d_try"] = round(total_flow, 0)
            row[f"flow_{w}d_%"] = (round(100 * total_flow / base_aum, 2)
                                    if pd.notna(base_aum) and base_aum != 0 else None)
        rows.append(row)
    return pd.DataFrame(rows)


def format_money_try(value):
    """Abbreviate a raw TRY amount to k/m/bn, 1 decimal -- e.g. 2.0m, 4.2bn,
    -1.5m for outflows. None/NaN pass through as None."""
    if value is None or pd.isna(value):
        return None
    sign = "-" if value < 0 else ""
    abs_val = abs(value)
    if abs_val >= 1e9:
        return f"{sign}{abs_val / 1e9:.1f}bn"
    elif abs_val >= 1e6:
        return f"{sign}{abs_val / 1e6:.1f}m"
    elif abs_val >= 1e3:
        return f"{sign}{abs_val / 1e3:.1f}k"
    return f"{sign}{abs_val:.1f}"


def fund_summary_table(price_df, shares_df, name_map, results_df, month_tdays=MONTH_TDAYS):
    """fund_name, fund_code, sharpe, last-month (month_tdays) return, and
    NOMINAL (raw TRY, not %-of-AUM) money flow: last 1 day, plus the
    AVERAGE daily flow over the last 5 and last {month_tdays} days --
    averages here, not sums, per what was asked (distinct from the
    windowed money_flow_table above, which sums)."""
    daily_flow_try = None
    if shares_df is not None:
        common = [c for c in results_df["fund_code"] if c in shares_df.columns and c in price_df.columns]
        if common:
            daily_flow_try = shares_df[common].diff() * price_df[common]

    rows = []
    for _, r in results_df.iterrows():
        code = r["fund_code"]
        p = price_df[code].dropna() if code in price_df.columns else pd.Series(dtype=float)
        if len(p) >= month_tdays + 1:
            ret_month_pct = round((p.iloc[-1] / p.iloc[-(month_tdays + 1)] - 1) * 100, 2)
        else:
            ret_month_pct = None

        row = {
            "fund_name": name_map.get(code, ""),
            "fund_code": code,
            "sharpe": r["sharpe"],
            f"return_{month_tdays}d_%": ret_month_pct,
        }

        if daily_flow_try is not None and code in daily_flow_try.columns:
            flow_s = daily_flow_try[code].dropna()
            row["flow_1d_try"] = format_money_try(flow_s.iloc[-1]) if len(flow_s) >= 1 else None
            row["flow_5d_avg_try"] = format_money_try(flow_s.iloc[-5:].mean()) if len(flow_s) >= 1 else None
            row[f"flow_{month_tdays}d_avg_try"] = (
                format_money_try(flow_s.iloc[-month_tdays:].mean()) if len(flow_s) >= 1 else None
            )
        else:
            row["flow_1d_try"] = None
            row["flow_5d_avg_try"] = None
            row[f"flow_{month_tdays}d_avg_try"] = None

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


def plot_money_flow_bars(price_df, shares_df, fund_codes, title):
    """Grouped bar chart: one group per fund, 3 bars = last 1-day flow,
    last 5-day avg flow, last 22-day avg flow (raw TRY, same figures as
    the Fund summary table). Funds sorted by the 1-day flow, descending."""
    if shares_df is None:
        return None
    codes = [c for c in fund_codes if c in shares_df.columns and c in price_df.columns]
    if not codes:
        return None

    daily_flow_try = shares_df[codes].diff() * price_df[codes]

    rows = []
    for code in codes:
        flow_s = daily_flow_try[code].dropna()
        if len(flow_s) == 0:
            continue
        rows.append((
            code,
            flow_s.iloc[-1],
            flow_s.iloc[-5:].mean(),
            flow_s.iloc[-MONTH_TDAYS:].mean(),
        ))
    if not rows:
        return None

    rows.sort(key=lambda r: r[1], reverse=True)   # sort by 1-day flow, descending
    codes_sorted = [r[0] for r in rows]
    flow_1d = [r[1] for r in rows]
    flow_5d = [r[2] for r in rows]
    flow_22d = [r[3] for r in rows]

    x = np.arange(len(codes_sorted))
    width = 0.26

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width, flow_1d, width, label="Last 1 day", color="tab:blue")
    ax.bar(x, flow_5d, width, label="Last 5 days (avg)", color="tab:orange")
    ax.bar(x + width, flow_22d, width, label=f"Last {MONTH_TDAYS} days (avg)", color="tab:green")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(codes_sorted)
    ax.set_ylabel("Money flow (TRY)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: format_money_try(y) or "0"))
    ax.set_title(title, fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")
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
name_col = find_col(raw, [r'fund_name'])

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

name_map = {}
if name_col:
    name_map = (raw.sort_values(date_col).dropna(subset=[name_col])
                   .groupby(code_col)[name_col].last().to_dict())

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
st.caption("ℹ️ `total_return_%` is the actual return over the window used. "
           "`ann_return_%` is a compounded (CAGR-style) extrapolation of that "
           "same return to a full year, which can look extreme for short "
           "windows with large returns -- e.g. a fund up 100% in 3 months "
           "annualizes to +1500%, since compounding assumes that exact rate "
           "repeats 4 times over. Treat `total_return_%` as the grounded, "
           "actually-observed figure.")
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

st.markdown("**Money flow ranking (nominal TRY, sorted by last 1-day flow)**")
flow_fig = plot_money_flow_bars(
    price_df, shares_df, top10_codes,
    f"Top {len(top10)} funds — money flow: last 1 day vs 5-day avg vs {MONTH_TDAYS}-day avg"
)
if flow_fig is not None:
    st.pyplot(flow_fig)
else:
    st.caption("Skipped -- no shares_outstanding data available this fetch.")

st.markdown("**Return stats**")
st.dataframe(
    top10[["fund_code", "total_return_%", "sharpe", "max_drawdown_%", "calmar", "ann_return_%", "ann_vol_%"]],
    width="stretch", hide_index=True
)

st.markdown(f"**Fund summary — name, Sharpe, last {MONTH_TDAYS}-day return, nominal money flow**")
st.caption(f"`flow_1d_try` is a single day's raw flow (TRY). `flow_5d_avg_try` and "
           f"`flow_{MONTH_TDAYS}d_avg_try` are the AVERAGE daily flow over each "
           f"window (not summed) -- distinct from the money-flow table above, "
           f"which sums cumulative flow as %-of-AUM.")
summary_df = fund_summary_table(price_df, shares_df, name_map, top10, MONTH_TDAYS)
st.dataframe(summary_df, width="stretch", hide_index=True)
