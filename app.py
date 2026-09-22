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

MAX DRAWDOWN FILTER (optional): a sidebar checkbox, OFF by default, lets
the user cap how much drawdown (over the lookback window, same field as
`max_drawdown_%` in the results table) a fund is allowed to have. If
enabled, any fund whose max drawdown magnitude exceeds the chosen limit
(default 10%) is dropped from the qualifying set entirely -- it disappears
from the main table, the Top-10 deep dive, and the inflow/outflow universe,
since all of those are derived from the same filtered set. This is applied
on top of the initial filter (history floor + min annualized return + min
investors), not instead of it.

HOUSE KEYWORD FILTER (optional): a sidebar text input of comma-separated
keywords (default "Pusula, Tera, Atlas" -- portfolio management companies
whose funds recently declared default), matched case-insensitively as a
substring against each fund's name. Any fund whose name contains ANY of
these keywords is excluded from the qualifying set outright -- same
treatment as the max drawdown filter above (main table, Top-10 deep dive,
and inflow/outflow universe all shrink accordingly). Matching against
fund NAME (not category) because TEFAS's naming convention puts the
portfolio management company's name directly in the fund's title (e.g.
"Tera Portföy Birinci ... Fonu"), and unlike fund category, fund_name is
a field the API actually returns. Like the max drawdown filter, this is
a Screener-only filter: the Custom Portfolio section explicitly bypasses
it, since that section is a user-driven, explicit ticker list against
the full fetched universe, not a screened/ranked set. Clear the keyword
field entirely to disable this filter.

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

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None


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

DEFAULT_MAX_DRAWDOWN_PCT = 10.0
# Default value shown in the (optional, off-by-default) max drawdown
# filter number input -- NOT applied unless the user ticks the checkbox.

CACHE_TTL_HOURS = 12
ANNUALIZATION = 252

MAX_ABS_DAILY_RETURN = 0.30
MAX_BAD_TICK_FRACTION = 0.05
MAX_INTERP_GAP_TDAYS = 5

FLOW_WINDOWS_TDAYS = [1, 5, 10, 20]
MONTH_TDAYS = 22   # "last month" convention used in the summary table below

DEFAULT_EXCLUDED_HOUSE_KEYWORDS = "Pusula, Tera, Atlas"
# Comma-separated default for the sidebar's investment-house keyword
# filter -- these three portfolio management companies' funds recently
# declared default. Matched case-insensitively as a substring against
# each fund's NAME (TEFAS's naming convention puts the portfolio
# management company's name directly in the fund title). Editable in
# the sidebar; clear the field entirely to disable this filter.


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


def screen_funds_all(price_df, investor_snapshot, lookback_tdays, min_history_tdays,
                      min_ann_return, min_investors, risk_free,
                      max_drawdown_limit_pct=None, excluded_fund_codes=None):
    """Returns a DataFrame with one row per fund that clears the initial
    filter (history floor + min annualized return + min investors) --
    the FULL qualifying set, sorted by Sharpe descending but NOT
    truncated to any Top-N. `screen_funds` below just slices this for
    the main table; anything that should look beyond the Sharpe-ranked
    Top-N (e.g. an inflow/outflow screen) should call this directly.

    `max_drawdown_limit_pct`: optional. When provided (not None), any
    fund whose max drawdown magnitude over the lookback window exceeds
    this percentage is excluded from the qualifying set entirely --
    same "excluded outright" treatment as the other initial-filter
    conditions, not a post-hoc dimming/highlighting.

    `excluded_fund_codes`: optional set of fund codes to exclude outright
    (e.g. funds matched against the investment-house keyword filter) --
    same treatment as the other initial-filter conditions."""
    daily_log_ret = np.log(price_df / price_df.shift(1))
    rows = []
    for code in price_df.columns:
        # Excluded-code filter: check first, cheapest possible short-circuit
        # before any numeric work on this fund at all.
        if excluded_fund_codes and code in excluded_fund_codes:
            continue

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

        # Optional max-drawdown cap: drop the fund outright if its
        # drawdown magnitude exceeds the user-chosen limit. max_dd is
        # <= 0 (0 = no drawdown), so we compare its magnitude.
        if max_drawdown_limit_pct is not None and abs(max_dd) * 100 > max_drawdown_limit_pct:
            continue

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
    return result.sort_values("sharpe", ascending=False).reset_index(drop=True)


def screen_funds(price_df, investor_snapshot, lookback_tdays, min_history_tdays,
                  min_ann_return, min_investors, risk_free, top_n,
                  max_drawdown_limit_pct=None, excluded_fund_codes=None):
    """Same as screen_funds_all, truncated to the top_n by Sharpe --
    kept as a separate thin wrapper so existing call sites/behavior
    don't change."""
    all_qualified = screen_funds_all(
        price_df, investor_snapshot, lookback_tdays, min_history_tdays,
        min_ann_return, min_investors, risk_free,
        max_drawdown_limit_pct=max_drawdown_limit_pct,
        excluded_fund_codes=excluded_fund_codes
    )
    if all_qualified.empty:
        return all_qualified
    return all_qualified.head(top_n).reset_index(drop=True)


def excluded_codes_by_house_keywords(name_map, keywords_str):
    """Given fund_code -> fund_name and a comma-separated keyword string,
    return the set of fund codes whose NAME contains any keyword
    (case-insensitive substring match). Empty/blank keywords_str returns
    an empty set (filter disabled). Matching is against fund_name because
    TEFAS naming convention puts the portfolio management company's name
    directly in the fund title -- e.g. a fund named "Tera Portföy Birinci
    Değişken Fon" is caught by the keyword "Tera"."""
    keywords = [k.strip().lower() for k in keywords_str.split(",") if k.strip()]
    if not keywords:
        return set()
    return {
        code for code, name in name_map.items()
        if name and any(kw in name.lower() for kw in keywords)
    }


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


def optimize_max_sharpe(daily_ret_df, risk_free_annual, annualization=ANNUALIZATION):
    """Long-only, fully-invested (weights >= 0, sum to 1) max-Sharpe
    ("tangency") portfolio. No closed-form solution exists once you add
    the long-only constraint -- the classic w ∝ Σ⁻¹μ formula can produce
    negative weights, so this needs a real constrained optimizer
    (scipy's SLSQP). Verified against two synthetic cases before wiring
    this in: (1) one clearly-dominant asset -> optimizer correctly puts
    ~100% there; (2) two similar-quality but UNCORRELATED assets ->
    optimizer finds a genuinely diversified blend that beats either
    asset alone AND beats a naive 50/50 split.

    daily_ret_df: DataFrame of daily SIMPLE returns (columns = tickers),
    already aligned (no NaNs -- caller should .dropna() first so the
    covariance matrix is well-formed, not just pairwise-complete).

    Returns (weights_dict, success_bool). weights_dict is None on failure.
    """
    if minimize is None or daily_ret_df.shape[0] < 2 or daily_ret_df.shape[1] < 1:
        return None, False

    tickers = daily_ret_df.columns.tolist()
    n = len(tickers)
    mean_daily = daily_ret_df.mean().values
    cov_daily = daily_ret_df.cov().values
    rf_daily = risk_free_annual / annualization   # simple daily-rate approximation

    def neg_sharpe(w):
        port_mean = np.dot(w, mean_daily)
        port_var = np.dot(w, np.dot(cov_daily, w))
        port_vol = np.sqrt(max(port_var, 1e-16))
        return -(port_mean - rf_daily) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    w0 = np.repeat(1.0 / n, n)

    result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 1000, "ftol": 1e-12})
    if not result.success:
        return None, False

    w = np.clip(result.x, 0, None)
    w = w / w.sum()
    return dict(zip(tickers, w)), True


def plot_correlation_matrix(daily_ret_df, title):
    corr = daily_ret_df.corr()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(4, 1.1 * n + 1.5), max(3.5, 1.1 * n + 1)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if abs(val) > 0.55 else "black", fontsize=9)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    plt.tight_layout()
    return fig


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


def plot_fund_daily_money_flow_with_ma(price_df, shares_df, code, lookback_tdays, title):
    """Day-by-day money flow bar chart for a SINGLE fund (raw TRY,
    Δshares_outstanding * price -- same signal used everywhere else in
    this app), overlaid with its 5-day and 22-day simple moving
    averages as lines on top of the bars. Sliced to the last
    `lookback_tdays` of available flow observations (same window
    convention as the rest of the custom-portfolio section). Returns
    None if there's no shares_outstanding data for this fund."""
    if shares_df is None or code not in shares_df.columns or code not in price_df.columns:
        return None

    daily_flow_try = (shares_df[code].diff() * price_df[code]).dropna()
    if daily_flow_try.empty:
        return None

    flow_window = daily_flow_try.iloc[-lookback_tdays:]
    ma5 = flow_window.rolling(5).mean()
    ma22 = flow_window.rolling(MONTH_TDAYS).mean()

    colors = ["tab:green" if v >= 0 else "tab:red" for v in flow_window.values]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(flow_window.index, flow_window.values, width=1.0, color=colors, alpha=0.6,
           label="Daily flow")
    ax.plot(ma5.index, ma5.values, color="tab:blue", lw=1.8, label="5-day MA")
    ax.plot(ma22.index, ma22.values, color="black", lw=1.8, label=f"{MONTH_TDAYS}-day MA")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Money flow (TRY)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: format_money_try(y) or "0"))
    ax.set_xlabel("Date")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    return fig


def top_inflow_outflow_codes(price_df, shares_df, fund_codes, n=5):
    """From `fund_codes` (any universe -- e.g. everything that passed
    the initial filter, independent of Sharpe ranking), rank by last
    1-day money flow and return up to n biggest inflow codes and up to
    n biggest outflow codes. If the universe has fewer than 2n funds,
    the inflow/outflow sets are de-duplicated (a fund can't be both)
    while preserving rank order."""
    if shares_df is None:
        return [], []
    codes = [c for c in fund_codes if c in shares_df.columns and c in price_df.columns]
    if not codes:
        return [], []

    daily_flow_try = shares_df[codes].diff() * price_df[codes]
    latest_flow = {}
    for code in codes:
        s = daily_flow_try[code].dropna()
        if len(s):
            latest_flow[code] = s.iloc[-1]
    if not latest_flow:
        return [], []

    ranked = sorted(latest_flow, key=latest_flow.get, reverse=True)
    top_in = ranked[:n]
    top_out = list(reversed(ranked[-n:]))
    top_out = [c for c in top_out if c not in top_in]   # avoid overlap when universe is small
    return top_in, top_out


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
    "Min investor count", min_value=0, max_value=100000, value=250, step=50
)

ENABLE_MAX_DRAWDOWN_FILTER = st.sidebar.checkbox(
    "Enable max drawdown filter", value=False,
    help="Off by default. When on, funds whose max drawdown (over the "
         "lookback window) exceeds the limit below are removed from the "
         "results entirely -- main table, Top-10 deep dive, and the "
         "inflow/outflow universe."
)
MAX_DRAWDOWN_LIMIT_PCT = st.sidebar.number_input(
    "Max allowed drawdown (%)", min_value=0.0, max_value=100.0,
    value=DEFAULT_MAX_DRAWDOWN_PCT, step=1.0,
    disabled=not ENABLE_MAX_DRAWDOWN_FILTER,
    help="A fund is excluded if its max drawdown magnitude over the "
         "lookback window is larger than this. E.g. 10 excludes any fund "
         "that ever fell more than 10% from a prior peak within the window."
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

# ── Investment-house keyword filter (sidebar) ───────────────────────
# Screener-only "initial filter" condition, same treatment as the max
# drawdown filter above: applied inside screen_funds_all via
# excluded_fund_codes, NOT by truncating price_df -- so an excluded
# fund still shows up fine if the user types its ticker directly into
# Custom Portfolio below. Matches against fund NAME (a field the API
# actually returns), not category (which it doesn't -- see git history
# /prior attempts). Default keywords: Pusula, Tera, Atlas.
st.sidebar.header("🏢 Exclude by investment house")
HOUSE_KEYWORDS_STR = st.sidebar.text_input(
    "Exclude funds whose name contains (comma-separated)",
    value=DEFAULT_EXCLUDED_HOUSE_KEYWORDS,
    help="Case-insensitive substring match against each fund's name. "
         "E.g. \"Tera\" excludes any fund named like \"Tera Portföy ... Fonu\". "
         "Clear this field entirely to disable the filter. Screener-only -- "
         "Custom Portfolio below is unaffected, you can still type any "
         "ticker directly."
)
EXCLUDED_FUND_CODES = excluded_codes_by_house_keywords(name_map, HOUSE_KEYWORDS_STR)
if EXCLUDED_FUND_CODES:
    st.sidebar.caption(f"🚫 {len(EXCLUDED_FUND_CODES)} fund(s) excluded by name match.")

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
         + (f" · dropped {len(dropped_funds)} fund(s) with unreliable price data" if dropped_funds else "")
         + (f" · {len(EXCLUDED_FUND_CODES)} fund(s) excluded by investment-house keyword"
            if EXCLUDED_FUND_CODES else ""))

all_qualified = screen_funds_all(
    price_df, investor_snapshot, LOOKBACK_TDAYS, MIN_HISTORY_TDAYS,
    MIN_ANN_RETURN_PCT / 100.0, MIN_INVESTOR_COUNT, RISK_FREE_RATE,
    max_drawdown_limit_pct=(MAX_DRAWDOWN_LIMIT_PCT if ENABLE_MAX_DRAWDOWN_FILTER else None),
    excluded_fund_codes=EXCLUDED_FUND_CODES
)
results = all_qualified.head(TOP_N).reset_index(drop=True) if not all_qualified.empty else all_qualified

if results.empty:
    dd_clause = (f" and max drawdown ≤{MAX_DRAWDOWN_LIMIT_PCT:.1f}%"
                 if ENABLE_MAX_DRAWDOWN_FILTER else "")
    house_clause = (f" ({len(EXCLUDED_FUND_CODES)} fund(s) excluded by investment-house keyword)"
                     if EXCLUDED_FUND_CODES else "")
    st.warning(f"No funds passed at {MIN_ANN_RETURN_PCT:.1f}% min annualized return, "
               f"≥{MIN_INVESTOR_COUNT} investors{dd_clause}{house_clause}. Try loosening the filters.")
    st.stop()

st.divider()
st.subheader(f"🏆 Top {len(results)} by Sharpe ratio")
dd_caption = (f" | max drawdown ≤{MAX_DRAWDOWN_LIMIT_PCT:.1f}%"
              if ENABLE_MAX_DRAWDOWN_FILTER else "")
house_caption = (f" | {len(EXCLUDED_FUND_CODES)} fund(s) excluded by investment-house keyword"
                  if EXCLUDED_FUND_CODES else "")
st.caption(f"Window: up to {LOOKBACK_TDAYS} trading days | "
           f"min annualized return ≥{MIN_ANN_RETURN_PCT:.1f}% | "
           f"min investors ≥{MIN_INVESTOR_COUNT}{dd_caption}{house_caption}")
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

st.markdown("**Top 5 inflow vs top 5 outflow funds (initial-filter universe, no Sharpe ranking)**")
st.caption(f"Drawn from all **{len(all_qualified)}** fund(s) that passed the initial filter "
           f"(history floor + min annualized return + min investors"
           f"{' + max drawdown cap' if ENABLE_MAX_DRAWDOWN_FILTER else ''}"
           f"{' + investment-house keyword filter' if EXCLUDED_FUND_CODES else ''}) -- not limited to the "
           f"Top {len(top10)} by Sharpe above. Ranked by last 1-day money flow.")
if shares_df is not None:
    all_qualified_codes = all_qualified["fund_code"].tolist()
    top5_in, top5_out = top_inflow_outflow_codes(price_df, shares_df, all_qualified_codes, n=5)
    inflow_outflow_codes = top5_in + top5_out
    if inflow_outflow_codes:
        io_fig = plot_money_flow_bars(
            price_df, shares_df, inflow_outflow_codes,
            "Top 5 inflow vs top 5 outflow funds — money flow: last 1 day vs 5-day avg vs "
            f"{MONTH_TDAYS}-day avg"
        )
        if io_fig is not None:
            st.pyplot(io_fig)
        else:
            st.caption("No flow data available for this universe.")
    else:
        st.caption("No flow data available for this universe.")
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


# ════════════════════════════════════════════════════════════════════
#  CUSTOM PORTFOLIO — user-defined ticker list + weights
#  Reuses the SAME already-fetched universe (price_df, shares_df,
#  name_map) as the Screener above -- no extra network fetch needed,
#  so "Run" is instant. Uses the same LOOKBACK_TDAYS / RISK_FREE_RATE
#  already selected in the sidebar for "the selected period". The
#  custom portfolio is a user-driven, explicit ticker list, so the
#  optional max-drawdown filter AND the investment-house keyword filter
#  above are NOT applied here -- they only govern the Screener's own
#  qualifying set.
# ════════════════════════════════════════════════════════════════════
if "custom_tickers" not in st.session_state:
    st.session_state.custom_tickers = "ALE, GTZ, BSM, GZM, LPH, ECA"


def _add_ticker_callback(code):
    """Runs BEFORE the script re-renders the ticker text_input, so this
    is safe -- mutating st.session_state[key] AFTER that widget has
    already been instantiated in the same run raises a StreamlitAPIException;
    on_click callbacks run prior to the main script body specifically to
    avoid that (verified with an isolated test before wiring this in)."""
    current = [t.strip().upper() for t in st.session_state.custom_tickers.split(",") if t.strip()]
    if code not in current:
        current.append(code)
        st.session_state.custom_tickers = ", ".join(current)


st.sidebar.divider()
st.sidebar.header("📁 Custom Portfolio")
st.sidebar.text_input(
    "Tickers (comma-separated)", key="custom_tickers",
    help="Uses the SAME fetched fund universe as the Screener above (same "
         "Fund kind) -- no extra fetch needed. Not affected by the "
         "investment-house keyword filter or max drawdown filter above."
)

with st.sidebar.expander("🔍 Search fund name to find a ticker"):
    name_query = st.text_input("Search by fund name", value="", key="fund_name_search")
    if name_query:
        matches = [(c, n) for c, n in name_map.items() if name_query.lower() in n.lower()]
        if matches:
            for code, fname in matches[:8]:
                c1, c2 = st.columns([3, 1])
                c1.write(f"**{code}** — {fname}")
                c2.button("Add", key=f"add_ticker_{code}", on_click=_add_ticker_callback, args=(code,))
            if len(matches) > 8:
                st.caption(f"...and {len(matches) - 8} more match(es). Refine your search.")
        else:
            st.caption("No matches.")

EQUAL_WEIGHT = st.sidebar.checkbox("Equal weight", value=True)
manual_weights_str = ""
if not EQUAL_WEIGHT:
    manual_weights_str = st.sidebar.text_input(
        "Weights (comma-separated, same order as tickers)", value="",
        help="E.g. 0.4, 0.3, 0.2, 0.1 -- doesn't need to sum to exactly 1, "
             "it's normalized automatically."
    )

run_portfolio_clicked = st.sidebar.button("▶️ Run Portfolio Analysis", type="primary")

if run_portfolio_clicked:
    st.session_state.portfolio_has_run = True
    st.session_state.portfolio_tickers_str = st.session_state.custom_tickers
    st.session_state.portfolio_equal_weight = EQUAL_WEIGHT
    st.session_state.portfolio_weights_str = manual_weights_str

st.divider()
st.header("📁 Custom Portfolio")

if not st.session_state.get("portfolio_has_run"):
    st.info("Enter tickers (and weights, if not equal-weighting) in the sidebar, then click "
            "**▶️ Run Portfolio Analysis**.")
else:
    tickers = [t.strip().upper() for t in st.session_state.portfolio_tickers_str.split(",") if t.strip()]
    if not tickers:
        st.warning("No tickers entered.")
    else:
        found = [t for t in tickers if t in price_df.columns]
        missing = [t for t in tickers if t not in price_df.columns]
        if missing:
            st.warning(f"Not found in the current fetch (check spelling, or that they're "
                       f"covered by the selected Fund kind): {missing}")

        if not found:
            st.error("None of the entered tickers were found.")
        else:
            weights = None
            if st.session_state.portfolio_equal_weight:
                weights = {t: 1.0 / len(found) for t in found}
            else:
                weight_strs = [w.strip() for w in st.session_state.portfolio_weights_str.split(",") if w.strip()]
                if len(weight_strs) != len(tickers):
                    st.error(f"Entered {len(weight_strs)} weight(s) for {len(tickers)} ticker(s) -- "
                             f"these must match 1:1, in the same order as the tickers field.")
                else:
                    try:
                        raw_weights = {t: float(w) for t, w in zip(tickers, weight_strs)}
                        found_weights_raw = {t: raw_weights[t] for t in found}
                        wsum = sum(found_weights_raw.values())
                        if wsum == 0:
                            st.error("Weights for the found tickers sum to zero -- can't normalize.")
                        else:
                            weights = {t: w / wsum for t, w in found_weights_raw.items()}
                    except ValueError:
                        st.error("Weights must be plain numbers, comma-separated.")

            if weights:
                # sliced to LOOKBACK_TDAYS+1 prices -> exactly LOOKBACK_TDAYS
                # valid daily returns after pct_change + dropping the
                # guaranteed-NaN first row (same off-by-one care as
                # elsewhere in this app)
                sub_price = price_df[found].iloc[-(LOOKBACK_TDAYS + 1):]
                daily_ret = sub_price.pct_change(fill_method=None).iloc[1:]

                # long-only max-Sharpe ("tangency") portfolio, for comparison
                # against the user's chosen weights -- uses rows where ALL
                # selected funds have simultaneous valid data (dropna, not
                # pairwise), so the covariance matrix is well-formed
                common_ret = daily_ret[found].dropna()
                opt_weights, opt_success = optimize_max_sharpe(common_ret, RISK_FREE_RATE)
                if not opt_success:
                    st.caption("ℹ️ Could not compute the optimal (max-Sharpe) allocation "
                               "for this fund set -- showing your chosen weights only.")

                # simple (not log) returns -- portfolio return = weighted
                # sum of constituent returns only holds for simple returns
                port_daily_ret = sum(daily_ret[t].fillna(0) * w for t, w in weights.items())
                port_nav = (1 + port_daily_ret).cumprod()

                opt_nav = None
                if opt_success:
                    opt_daily_ret = sum(daily_ret[t].fillna(0) * w for t, w in opt_weights.items())
                    opt_nav = (1 + opt_daily_ret).cumprod()

                def _perf_stats_simple(nav_series):
                    n = len(nav_series)
                    if n == 0:
                        return None
                    total_ret = nav_series.iloc[-1] - 1
                    if n > MONTH_TDAYS:
                        ret_1m = nav_series.iloc[-1] / nav_series.iloc[-(MONTH_TDAYS + 1)] - 1
                    else:
                        ret_1m = np.nan
                    r = nav_series.pct_change().dropna()
                    ann_vol = r.std(ddof=1) * np.sqrt(ANNUALIZATION) if len(r) > 1 else np.nan
                    ann_ret = (1 + total_ret) ** (ANNUALIZATION / n) - 1
                    sharpe = ((ann_ret - RISK_FREE_RATE) / ann_vol
                              if pd.notna(ann_vol) and ann_vol > 0 else np.nan)
                    dd = (nav_series / nav_series.cummax() - 1).min()
                    return {
                        "total_return_%": round(total_ret * 100, 2),
                        f"return_{MONTH_TDAYS}d_%": round(ret_1m * 100, 2) if pd.notna(ret_1m) else None,
                        "ann_vol_%": round(ann_vol * 100, 2) if pd.notna(ann_vol) else None,
                        "sharpe": round(sharpe, 3) if pd.notna(sharpe) else None,
                        "max_drawdown_%": round(dd * 100, 2),
                    }

                rows = []
                port_stats = _perf_stats_simple(port_nav)
                port_stats["name"] = "PORTFOLIO"
                port_stats["weight_%"] = 100.0
                port_stats["optimal_weight_%"] = None
                rows.append(port_stats)

                if opt_success:
                    opt_stats = _perf_stats_simple(opt_nav)
                    opt_stats["name"] = "OPTIMAL (Max Sharpe, in-sample)"
                    opt_stats["weight_%"] = None
                    opt_stats["optimal_weight_%"] = 100.0
                    rows.append(opt_stats)

                fund_navs = {}
                for t in found:
                    nav_t = (1 + daily_ret[t].fillna(0)).cumprod()
                    fund_navs[t] = nav_t
                    stats_t = _perf_stats_simple(nav_t)
                    stats_t["name"] = t
                    stats_t["weight_%"] = round(weights[t] * 100, 1)
                    stats_t["optimal_weight_%"] = (round(opt_weights[t] * 100, 1)
                                                    if opt_success and t in opt_weights else None)
                    rows.append(stats_t)

                stats_df = pd.DataFrame(rows)[
                    ["name", "weight_%", "optimal_weight_%", "total_return_%", f"return_{MONTH_TDAYS}d_%",
                     "ann_vol_%", "sharpe", "max_drawdown_%"]
                ]

                st.subheader("Cumulative performance")
                fig, ax = plt.subplots(figsize=(13, 6))
                ax.plot(port_nav.index, port_nav.values, lw=2.5, color="black", label="PORTFOLIO")
                if opt_success:
                    ax.plot(opt_nav.index, opt_nav.values, lw=2.2, color="tab:green",
                            linestyle="--", label="OPTIMAL (Max Sharpe)")
                cmap = plt.cm.tab10
                for idx, t in enumerate(found):
                    ax.plot(fund_navs[t].index, fund_navs[t].values, lw=1.2, alpha=0.7,
                            color=cmap(idx % 10), label=t)
                ax.axhline(1.0, color="gray", lw=0.8, linestyle="--", alpha=0.4)
                ax.set_title(f"Custom portfolio vs constituents — cumulative return "
                             f"(last {LOOKBACK_TDAYS} trading days, start = 1.0)", fontsize=11)
                ax.set_xlabel("Date")
                ax.set_ylabel("Cumulative return (start = 1.0)")
                ax.legend(loc="upper left", fontsize=8, ncol=2)
                ax.grid(True, alpha=0.25)
                plt.tight_layout()
                st.pyplot(fig)

                st.subheader("Performance stats")
                st.dataframe(stats_df, width="stretch", hide_index=True)
                if opt_success:
                    st.caption("⚠️ 'OPTIMAL (Max Sharpe)' is the best long-only allocation "
                               "IN HINDSIGHT over this exact historical window -- a retrospective "
                               "illustration of what would have maximized Sharpe, not a "
                               "forward-looking recommendation.")

                st.subheader("Correlation matrix")
                if len(found) >= 2:
                    st.pyplot(plot_correlation_matrix(
                        common_ret, f"Daily-return correlation — last {LOOKBACK_TDAYS} trading days"
                    ))
                else:
                    st.caption("Need at least 2 funds to show a correlation matrix.")

                st.subheader("Per-fund price & AUM (both scaled to start = 1.0)")
                for t in found:
                    p = price_df[t].iloc[-LOOKBACK_TDAYS:]
                    p_valid = p.dropna()
                    p_scaled = p / p_valid.iloc[0] if len(p_valid) else p
                    price_dd = ((p_scaled / p_scaled.cummax() - 1).min() * 100
                                if len(p_valid) else np.nan)

                    fig2, ax1 = plt.subplots(figsize=(12, 4.5))
                    ax1.plot(p_scaled.index, p_scaled.values, color="tab:blue", lw=1.8, label="Price")
                    ax1.set_ylabel("Price (scaled, start = 1.0)", color="tab:blue")
                    ax1.tick_params(axis="y", labelcolor="tab:blue")
                    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}x"))
                    ax1.set_xlabel("Date")
                    ax1.axhline(1.0, color="gray", lw=0.7, linestyle=":", alpha=0.5)

                    aum_dd = np.nan
                    if shares_df is not None and t in shares_df.columns:
                        aum = (shares_df[t] * price_df[t]).iloc[-LOOKBACK_TDAYS:]
                        aum_valid = aum.dropna()
                        aum_scaled = aum / aum_valid.iloc[0] if len(aum_valid) else aum
                        aum_dd = ((aum_scaled / aum_scaled.cummax() - 1).min() * 100
                                  if len(aum_valid) else np.nan)
                        ax2 = ax1.twinx()
                        ax2.plot(aum_scaled.index, aum_scaled.values, color="tab:red", lw=1.4, alpha=0.8, label="AUM")
                        ax2.set_ylabel("AUM (scaled, start = 1.0)", color="tab:red")
                        ax2.tick_params(axis="y", labelcolor="tab:red")
                        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}x"))

                    dd_lines = [
                        f"Price: {price_dd:.1f}%" if pd.notna(price_dd) else "Price: n/a",
                        f"AUM: {aum_dd:.1f}%" if pd.notna(aum_dd) else "AUM: n/a",
                    ]
                    ax1.text(
                        0.02, 0.97, "Max drawdown\n" + "\n".join(dd_lines),
                        transform=ax1.transAxes, fontsize=9,
                        verticalalignment="top", horizontalalignment="left",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray")
                    )

                    ax1.set_title(f"{t} — {name_map.get(t, '')}", fontsize=11)
                    ax1.grid(True, alpha=0.2)
                    plt.tight_layout()
                    st.pyplot(fig2)

                # ── Per-fund daily money flow, day-by-day, with 5d/22d MAs ──
                st.subheader("Per-fund daily money flow (with 5-day and 22-day moving averages)")
                if shares_df is None:
                    st.caption("Skipped -- no shares_outstanding data available this fetch.")
                else:
                    for t in found:
                        flow_fig_t = plot_fund_daily_money_flow_with_ma(
                            price_df, shares_df, t, LOOKBACK_TDAYS,
                            f"{t} — {name_map.get(t, '')} — daily money flow "
                            f"(last {LOOKBACK_TDAYS} trading days)"
                        )
                        if flow_fig_t is not None:
                            st.pyplot(flow_fig_t)
                        else:
                            st.caption(f"{t}: no shares_outstanding data available -- skipped.")
