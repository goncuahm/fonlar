"""
TEFAS Fund Screener — Streamlit app
Port of the Colab "Part 1: update dataset / Part 2: screen & rank" script.

Key differences from the Colab version:
  - No Google Drive mount -- point DATA_FOLDER at any local folder (sidebar).
  - Part 1 (incremental daily CSV backfill) becomes a button: click
    "Update Dataset" and it fetches missing trading days one by one with
    a live progress bar + log, instead of running unattended in a cell.
  - Because the fetch loop is a genuine blocking loop (rate-limited with
    time.sleep between requests, same as the original), a single click
    could otherwise block the browser for a very long time if many days
    are missing (e.g. a 215-day bootstrap at 20s/day is over an hour).
    A "max days per click" cap lets you backfill incrementally across
    several clicks instead -- see MAX_DAYS_PER_RUN in the Update tab.
  - Part 2 (screen & rank) is reactive: adjusting the threshold/TOP_N/
    investor-count controls re-filters and re-ranks instantly, because
    the (comparatively expensive) CSV load+clean step is cached
    separately via st.cache_data and only re-runs when the on-disk file
    list actually changes.

Run with:  streamlit run tefas_streamlit_app.py
Requirements: streamlit, pytefas, pandas, numpy, matplotlib
"""

import os
import re
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

try:
    from pytefas import Crawler
except ImportError:
    Crawler = None


# ════════════════════════════════════════════════════════════════════
#  HELPERS — same logic as the original Colab script, unchanged
# ════════════════════════════════════════════════════════════════════
def all_weekdays_between(start, end):
    """Every Mon-Fri between start and end inclusive."""
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def last_trading_day(ref=None):
    """Most recent weekday on or before ref (default: yesterday)."""
    d = (ref or date.today()) - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def last_confirmed_date(price_df):
    """Last date where >=50% of funds have a price (partial-holiday safe)."""
    coverage = price_df.notna().sum(axis=1)
    threshold = max(len(price_df.columns) * 0.5, 1)
    valid = coverage[coverage >= threshold].index
    return valid[-1] if len(valid) else price_df.index[-1]


def period_return_tdays(series, trading_days):
    """Holiday-safe return using actual index positions."""
    s = series.dropna()
    if len(s) < trading_days + 1:
        return np.nan
    return (s.iloc[-1] / s.iloc[-(trading_days + 1)]) - 1


def find_col(df, patterns):
    for pat in patterns:
        for c in df.columns:
            if re.search(pat, c, re.I):
                return c
    return None


def fetch_one_day(crawler, day_str, kind):
    """Returns (df_or_None, error_or_None)."""
    try:
        df_day = crawler.fetch(day_str, columns="info", kind=kind)
        if df_day is None or df_day.empty:
            return None, None
        return df_day, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def get_crawler():
    return Crawler()


def list_existing_files(data_folder):
    if not os.path.isdir(data_folder):
        return []
    return sorted([
        f for f in os.listdir(data_folder)
        if f.startswith("tefas_daily_") and f.endswith(".csv")
    ])


def folder_size_bytes(data_folder, files):
    total = 0
    for f in files:
        try:
            total += os.path.getsize(os.path.join(data_folder, f))
        except OSError:
            pass
    return total


def human_size(n_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def files_older_than(data_folder, files, retention_years):
    """Daily files whose DATE (from the filename) is older than the
    rolling retention window, oldest-first."""
    cutoff = date.today() - timedelta(days=int(retention_years * 365.25))
    old = [f for f in files if date.fromisoformat(f[12:22]) < cutoff]
    return sorted(old)   # filenames sort chronologically since they embed ISO dates


def prune_old_files(data_folder, files_to_remove):
    """Deletes the given files from data_folder. Returns (removed, failed)."""
    removed, failed = [], []
    for f in files_to_remove:
        try:
            os.remove(os.path.join(data_folder, f))
            removed.append(f)
        except OSError as e:
            failed.append((f, str(e)))
    return removed, failed


HORIZONS_TDAYS = {"1m": 21, "2m": 42, "3m": 63, "4m": 84, "5m": 105, "6m": 126}
MIN_OBS = HORIZONS_TDAYS["6m"] + 5


class ColumnDetectionError(Exception):
    """Raised when a required column (date/code/price) can't be found in the
    loaded CSVs -- surfaced as a clean st.error() instead of a raw traceback."""
    pass


# ════════════════════════════════════════════════════════════════════
#  PAGE SETUP
# ════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="TEFAS Fund Screener", layout="wide")
st.title("📊 TEFAS Fund Screener")

st.sidebar.header("Configuration")
DATA_FOLDER = st.sidebar.text_input(
    "Local data folder", value="./tefas_data/",
    help="Any local folder -- there's no Google Drive mount here. "
         "If you're running this inside Colab with Drive already mounted, "
         "you can point this at /content/drive/MyDrive/Tefas/data/ instead."
)
KIND = st.sidebar.selectbox("Fund kind", ["YAT", "EMK", "BYF"], index=0,
                             help="YAT=mutual funds, EMK=pension, BYF=ETF")
os.makedirs(DATA_FOLDER, exist_ok=True)

if Crawler is None:
    st.sidebar.error("`pytefas` is not installed in this environment. "
                      "Add it to requirements.txt to enable data fetching. "
                      "The Screener tab still works on data already on disk.")

tab_update, tab_screen = st.tabs(["📥 Update Data", "🏆 Screener"])


# ════════════════════════════════════════════════════════════════════
#  TAB 1 — Update Data  (Part 1 of the original script, as a button)
# ════════════════════════════════════════════════════════════════════
with tab_update:
    st.subheader("Retention & fetch settings")
    r1, r2, r3 = st.columns(3)
    RETENTION_YEARS = r1.number_input(
        "Keep last N years of data", min_value=0.5, max_value=10.0, value=3.0, step=0.5,
        help="Daily files older than this rolling window get pruned so the "
             "on-disk dataset doesn't grow without bound. Set generously if "
             "you're not sure -- pruning only ever happens when you click "
             "Update Dataset or Prune Now below, never automatically on page load."
    )
    MAX_DAYS_PER_RUN = r2.number_input(
        "Max days to fetch per click", min_value=1, max_value=500, value=10,
        help="Caps how long a single click can block the app (at "
             "RATE_LIMIT_PAUSE seconds/day, 10 days ≈ 3-4 minutes). If more "
             "days are missing than this, click Update again afterward to "
             "keep backfilling -- it always resumes from the last date "
             "actually on disk, so nothing is skipped."
    )
    ENABLE_AUTO_PRUNE = r3.checkbox(
        "Auto-prune on each update", value=True,
        help="If on, every 'Update Dataset' click also deletes any files "
             "older than the retention window above, right after fetching. "
             "Turn off if you'd rather prune manually via 'Prune Now'."
    )
    RATE_LIMIT_PAUSE = st.slider(
        "Seconds between requests", 5, 60, 20,
        help="Same purpose as the original script's RATE_LIMIT_PAUSE -- "
             "too low risks TEFAS rate-limiting or blocking requests."
    )
    also_attempt_today = st.checkbox(
        "Also attempt today's same-day fetch after backfill", value=True,
        help="TEFAS usually finalizes NAVs the next business day, so this "
             "will often come back empty before ~09:00-10:00 the following "
             "morning -- that's expected, not an error."
    )

    st.divider()
    st.subheader("Local dataset status")

    existing_files = list_existing_files(DATA_FOLDER)
    today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    yesterday = last_trading_day()   # guaranteed-safe ceiling for backfill

    if existing_files:
        last_on_disk_str = existing_files[-1][12:22]
        last_on_disk = date.fromisoformat(last_on_disk_str)
        fetch_from = last_on_disk + timedelta(days=1)
    else:
        last_on_disk = None
        fetch_from = today - timedelta(days=215)

    existing_dates = {f[12:22] for f in existing_files}
    missing_days = [
        d for d in all_weekdays_between(fetch_from, yesterday)
        if d.strftime("%Y-%m-%d") not in existing_dates
    ]
    today_file = os.path.join(DATA_FOLDER, f"tefas_daily_{today_str}.csv")
    today_exists = os.path.exists(today_file)

    disk_bytes = folder_size_bytes(DATA_FOLDER, existing_files)
    prune_preview = files_older_than(DATA_FOLDER, existing_files, RETENTION_YEARS)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files on disk", len(existing_files))
    c2.metric("Last date on disk", last_on_disk.isoformat() if last_on_disk else "—")
    c3.metric("Missing trading days", len(missing_days))
    c4.metric("Today's file", "✅ saved" if today_exists else "not yet")

    c5, c6 = st.columns(2)
    c5.metric("Disk usage", human_size(disk_bytes))
    c6.metric(f"Older than {RETENTION_YEARS:g}y (prunable)", len(prune_preview))

    if not existing_files:
        st.info(f"No data on disk yet — the first update will bootstrap "
                 f"~7 months of history ({fetch_from} → {yesterday}).")
    elif prune_preview:
        st.caption(f"Oldest prunable file: {prune_preview[0][12:22]} · "
                   f"newest prunable file: {prune_preview[-1][12:22]} — "
                   f"{'will be removed automatically on the next update' if ENABLE_AUTO_PRUNE else 'auto-prune is off; use Prune Now to remove them'}.")

    if st.button("🔄 Update Dataset", type="primary", disabled=(Crawler is None)):
        crawler = get_crawler()
        days_to_fetch = missing_days[:MAX_DAYS_PER_RUN]
        remaining_after = len(missing_days) - len(days_to_fetch)

        saved, still_failed = [], []

        if days_to_fetch:
            st.write(f"Fetching {len(days_to_fetch)} day(s) "
                     f"({days_to_fetch[0]} → {days_to_fetch[-1]}) ...")
            progress = st.progress(0.0)
            log_box = st.container(height=260, border=True)
            errors = []

            for i, day in enumerate(days_to_fetch):
                day_str = day.strftime("%Y-%m-%d")
                out_path = os.path.join(DATA_FOLDER, f"tefas_daily_{day_str}.csv")
                df_day, err = fetch_one_day(crawler, day_str, KIND)
                if err:
                    log_box.write(f"⚠️ {day_str}: {err}")
                    errors.append((day_str, err))
                elif df_day is None:
                    log_box.write(f"⏭️ {day_str}: no data (holiday?)")
                else:
                    df_day.to_csv(out_path, index=False, encoding="utf-8-sig")
                    saved.append(day_str)
                    log_box.write(f"✅ {day_str}: {len(df_day):,} funds")
                progress.progress((i + 1) / len(days_to_fetch))
                if i < len(days_to_fetch) - 1:
                    time.sleep(RATE_LIMIT_PAUSE)

            if errors:
                log_box.write(f"🔁 Retrying {len(errors)} failed day(s) ...")
                for day_str, _ in errors:
                    time.sleep(RATE_LIMIT_PAUSE)
                    df_day, err2 = fetch_one_day(crawler, day_str, KIND)
                    if df_day is not None:
                        out_path = os.path.join(DATA_FOLDER, f"tefas_daily_{day_str}.csv")
                        df_day.to_csv(out_path, index=False, encoding="utf-8-sig")
                        saved.append(day_str)
                        log_box.write(f"  {day_str} ✅ recovered")
                    elif err2:
                        log_box.write(f"  {day_str} ❌ {err2}")
                        still_failed.append(day_str)
                    else:
                        log_box.write(f"  {day_str} (still empty on retry)")

            st.success(f"Backfill run complete — saved {len(saved)}, "
                       f"failed {len(still_failed)}.")
            if still_failed:
                st.warning(f"Still failed: {still_failed}")
            if remaining_after > 0:
                st.info(f"{remaining_after} more missing day(s) remain — "
                        f"click **Update Dataset** again to continue.")
        else:
            st.success(f"Backfill already up to date through {yesterday}.")

        if also_attempt_today:
            st.write("---")
            st.write(f"**Same-day fetch attempt for {today_str}:**")
            if today.weekday() >= 5:
                st.info(f"Skipped — {today_str} is a weekend.")
            elif os.path.exists(today_file):
                st.info("Already saved.")
            else:
                df_today, err_today = fetch_one_day(crawler, today_str, KIND)
                if err_today:
                    st.warning(f"Fetch error: {err_today}")
                elif df_today is None:
                    st.info(f"No data published yet for {today_str} — normal "
                            f"if TEFAS hasn't finalized today's NAVs. Try again later.")
                else:
                    df_today.to_csv(today_file, index=False, encoding="utf-8-sig")
                    st.success(f"{today_str} NAVs are live — saved {len(df_today):,} funds.")

        if ENABLE_AUTO_PRUNE:
            st.write("---")
            fresh_files = list_existing_files(DATA_FOLDER)
            to_remove = files_older_than(DATA_FOLDER, fresh_files, RETENTION_YEARS)
            if to_remove:
                removed, failed = prune_old_files(DATA_FOLDER, to_remove)
                st.write(f"**Retention pruning** (keeping last {RETENTION_YEARS:g} years):")
                if removed:
                    st.success(f"🗑️ Pruned {len(removed)} file(s) older than the retention "
                               f"window ({removed[0][12:22]} → {removed[-1][12:22]}).")
                if failed:
                    st.warning(f"Could not remove {len(failed)} file(s): {failed}")
            else:
                st.write(f"**Retention pruning:** nothing older than "
                         f"{RETENTION_YEARS:g} years — no files removed.")

        st.divider()
        fresh_files = list_existing_files(DATA_FOLDER)
        fresh_bytes = folder_size_bytes(DATA_FOLDER, fresh_files)
        st.write(f"**Updated status:** {len(fresh_files)} file(s) on disk, "
                 f"{human_size(fresh_bytes)} total.")

    st.divider()
    st.subheader("Manual pruning")
    st.caption(
        "Runs independently of Update Dataset -- useful right after lowering "
        "the retention window, or for one-off cleanup. Note: TEFAS/pytefas "
        "may or may not support re-fetching very old historical dates later, "
        "so treat pruning as a genuine, not-easily-reversed deletion."
    )
    if prune_preview:
        st.write(f"**{len(prune_preview)} file(s)** would be removed "
                 f"({prune_preview[0][12:22]} → {prune_preview[-1][12:22]}).")
        if st.button(f"🗑️ Prune Now ({len(prune_preview)} file(s))"):
            removed, failed = prune_old_files(DATA_FOLDER, prune_preview)
            if removed:
                st.success(f"Removed {len(removed)} file(s).")
            if failed:
                st.warning(f"Could not remove {len(failed)} file(s): {failed}")
    else:
        st.write("Nothing to prune at the current retention setting.")


# ════════════════════════════════════════════════════════════════════
#  TAB 2 — Screener  (Part 2 of the original script, reactive)
# ════════════════════════════════════════════════════════════════════
with tab_screen:
    st.subheader("Screening parameters")
    p1, p2, p3, p4 = st.columns(4)
    MONTHLY_THRESHOLD = p1.number_input(
        "Min per-month return hurdle (%)", min_value=0.0, max_value=20.0,
        value=3.75, step=0.05, help="Same as MONTHLY_THRESHOLD in the original script."
    ) / 100.0
    TOP_N = p2.number_input("Funds per table (TOP_N)", min_value=5, max_value=200, value=50, step=5)
    MIN_INVESTOR_COUNT = p3.number_input("Min investor count", min_value=0, max_value=100000, value=200, step=50)
    LOOKBACK_DAYS = p4.number_input(
        "CSV lookback window (days)", min_value=200, max_value=1200, value=215, step=5,
        help="Only load CSVs from this many calendar days back -- must be "
             "comfortably more than the 6-month (126 trading-day) horizon. "
             "Capped at 1200 (~3.3 years) to match the Update Data tab's "
             "default retention window -- there's no point setting this "
             "higher than however much history you're actually keeping on disk."
    )

    existing_files = list_existing_files(DATA_FOLDER)
    cutoff = date.today() - timedelta(days=int(LOOKBACK_DAYS))
    csv_files = sorted([
        f for f in existing_files
        if date.fromisoformat(f[12:22]) >= cutoff
    ])

    if not csv_files:
        st.warning("No CSV files in the lookback window yet — use the "
                   "**Update Data** tab to fetch some first.")
        st.stop()

    # ── Cached, expensive step: load + clean raw CSVs ──────────────
    @st.cache_data(show_spinner="Loading and cleaning local CSVs ...")
    def load_raw(data_folder, csv_files_tuple):
        frames = []
        for fname in csv_files_tuple:
            try:
                frames.append(pd.read_csv(
                    os.path.join(data_folder, fname),
                    encoding="utf-8-sig", low_memory=False
                ))
            except Exception as e:
                st.warning(f"Could not read {fname}: {e}")
        if not frames:
            return None, None, None, None, None
        raw = pd.concat(frames, ignore_index=True)

        date_col = find_col(raw, [r'^date$', r'tarih'])
        code_col = find_col(raw, [r'fund_code', r'^code$', r'fonkodu', r'\bkod\b'])
        price_col = find_col(raw, [r'^price$', r'fiyat'])
        name_col = find_col(raw, [r'fund_name', r'^title$', r'^name$', r'fonunvani', r'unvan'])
        investor_col = find_col(raw, [r'investor_count', r'investor', r'yatirimci',
                                       r'katilimci', r'number_of_investors'])

        for label, col in [("date", date_col), ("code", code_col), ("price", price_col)]:
            if col is None:
                raise ColumnDetectionError(
                    f"Cannot detect '{label}' column. "
                    f"Columns found across all loaded files: {raw.columns.tolist()}"
                )

        raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
        raw[price_col] = pd.to_numeric(raw[price_col], errors="coerce")
        raw[code_col] = raw[code_col].astype(str).str.strip().str.upper()
        raw = raw.dropna(subset=[date_col, price_col])
        if investor_col:
            raw[investor_col] = pd.to_numeric(raw[investor_col], errors="coerce")

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

        investor_snapshot = pd.Series(dtype=float)
        if investor_col:
            latest_date = raw[date_col].max()
            raw_latest = raw[raw[date_col] == latest_date]
            investor_snapshot = (
                raw_latest[[code_col, investor_col]]
                .dropna(subset=[investor_col])
                .drop_duplicates(subset=[code_col], keep="last")
                .set_index(code_col)[investor_col]
            )

        return price_df, name_map, investor_snapshot, investor_col, raw[date_col].max()

    try:
        price_df_full, name_map, investor_snapshot, investor_col, data_latest_date = load_raw(
            DATA_FOLDER, tuple(csv_files)
        )
    except ColumnDetectionError as e:
        st.error(
            f"⚠️ Problem reading the local CSV files: {e}\n\n"
            f"This usually means one of the daily files has an unexpected "
            f"format (a corrupted download, an interrupted write, or a "
            f"pytefas schema change). Check the files in `{DATA_FOLDER}`, "
            f"or try re-fetching the affected day(s) from the **Update Data** tab."
        )
        st.stop()

    if price_df_full is None:
        st.error("No CSV files could be loaded.")
        st.stop()

    st.caption(f"Loaded {len(csv_files)} daily file(s) | "
               f"{price_df_full.shape[1]} distinct fund codes | "
               f"latest data: {data_latest_date.date()}")

    # ── Reactive filtering: investor count -> price matrix ─────────
    price_df = price_df_full.copy()
    if investor_col and len(investor_snapshot):
        qualified = investor_snapshot[investor_snapshot >= MIN_INVESTOR_COUNT].index.tolist()
        before_n = len(price_df.columns)
        price_df = price_df[[c for c in price_df.columns if c in qualified]]
        st.write(f"👥 Investor filter (≥{MIN_INVESTOR_COUNT}): "
                 f"{len(price_df.columns)} / {before_n} funds qualify "
                 f"(as of {investor_snapshot.name if hasattr(investor_snapshot, 'name') else ''} "
                 f"latest snapshot)")
    elif not investor_col:
        st.info("No investor_count column detected — investor filter skipped.")

    obs_count = price_df.notna().sum()
    price_df = price_df[obs_count[obs_count >= MIN_OBS].index]
    true_last = last_confirmed_date(price_df)
    st.write(f"📊 Funds passing investor + history filters: **{len(price_df.columns)}** "
             f"| Last confirmed trading day: **{true_last.date()}**")

    if price_df.shape[1] == 0:
        st.error("No funds survive the investor/history filters — loosen "
                 "MIN_INVESTOR_COUNT or check your data.")
        st.stop()

    # ── Returns across all 6 horizons ───────────────────────────────
    ret = pd.DataFrame(index=price_df.columns)
    for label, tdays in HORIZONS_TDAYS.items():
        ret[label] = price_df.apply(lambda s: period_return_tdays(s, tdays))
    ret = ret.dropna()

    thresholds = {label: MONTHLY_THRESHOLD * int(label[0]) for label in HORIZONS_TDAYS}
    mask = pd.Series(True, index=ret.index)
    for label, thresh in thresholds.items():
        mask &= (ret[label] >= thresh)
    passed = ret[mask].index.tolist()

    st.write(f"🎯 Funds passing all 6 horizon filters: **{len(passed)}** "
             f"(of {len(ret)} with complete 6-horizon data)")

    if not passed:
        st.warning(f"No funds passed at a {MONTHLY_THRESHOLD*100:.2f}%/month hurdle. "
                   f"Top 10 by mean horizon return, for reference:")
        top10 = ret.mean(axis=1).sort_values(ascending=False).head(10)
        rows = []
        for c in top10.index:
            row = ret.loc[c]
            rows.append({
                "fund_code": c,
                "investors": int(investor_snapshot.get(c, 0)) if investor_col else "n/a",
                **{f"{l}_%": round(row[l] * 100, 2) for l in HORIZONS_TDAYS},
            })
        st.dataframe(pd.DataFrame(rows), width='stretch')
        st.stop()

    # ── Cross-horizon average return / volatility / Sharpe proxy ───
    avg_cols = [f"avg_{l}" for l in HORIZONS_TDAYS]
    avg_m = pd.DataFrame(index=passed)
    for label in HORIZONS_TDAYS:
        n = int(label[0])
        avg_m[f"avg_{label}"] = ret.loc[passed, label] / n
    avg_m["mean_monthly"] = avg_m[avg_cols].mean(axis=1)
    avg_m["vol_monthly"] = avg_m[avg_cols].std(axis=1)
    avg_m["sharpe_proxy"] = np.where(
        avg_m["vol_monthly"] > 0, avg_m["mean_monthly"] / avg_m["vol_monthly"], np.nan
    )

    def build_results(df_ranked):
        out = df_ranked.copy()
        for label in HORIZONS_TDAYS:
            out[f"ret_{label}_%"] = (ret.loc[out.index, label] * 100).round(2)
        out["avg_monthly_%"] = (out["mean_monthly"] * 100).round(2)
        out["vol_%"] = (out["vol_monthly"] * 100).round(4)
        out["sharpe_proxy"] = out["sharpe_proxy"].round(3)
        if investor_col and len(investor_snapshot):
            out["investors"] = out.index.map(lambda c: int(investor_snapshot.get(c, 0)))
        out.index.name = "fund_code"
        out = out.reset_index()
        out["fund_name"] = out["fund_code"].map(name_map).fillna("")
        out.insert(0, "rank", range(1, len(out) + 1))
        return out

    display_cols = (
        ["rank", "fund_code", "fund_name"]
        + [f"ret_{l}_%" for l in HORIZONS_TDAYS]
        + ["avg_monthly_%", "vol_%", "sharpe_proxy"]
        + (["investors"] if investor_col else [])
    )

    results_vol = build_results(avg_m.sort_values("vol_monthly").head(TOP_N))
    results_sharpe = build_results(
        avg_m.dropna(subset=["sharpe_proxy"]).sort_values("sharpe_proxy", ascending=False).head(TOP_N)
    )

    st.divider()
    st.subheader(f"🏆 Table 1 — Top {TOP_N} by LOWEST monthly return volatility")
    st.caption(f"Most consistent compounders | ≥{MIN_INVESTOR_COUNT} investors")
    st.dataframe(results_vol[display_cols], width='stretch', hide_index=True)

    st.subheader(f"🏆 Table 2 — Top {TOP_N} by HIGHEST Sharpe proxy")
    st.caption(f"Best return per unit of volatility | ≥{MIN_INVESTOR_COUNT} investors")
    st.dataframe(results_sharpe[display_cols], width='stretch', hide_index=True)

    overlap = sorted(set(results_vol["fund_code"]) & set(results_sharpe["fund_code"]))
    st.write(f"📌 Funds in **both** tables: {len(overlap)}  {overlap}")

    # ── Scaled price charts ──────────────────────────────────────────
    def plot_scaled(fund_codes, title_suffix, price_df, top_n):
        codes = [c for c in fund_codes if c in price_df.columns]
        plot_prices = price_df[codes].copy()
        first_valid = plot_prices.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
        scaled = plot_prices.div(first_valid)

        fig, ax = plt.subplots(figsize=(14, 7))
        cmap = plt.cm.tab20
        colors = [cmap(i / max(top_n, 1)) for i in range(len(codes))]
        for idx, code in enumerate(codes):
            s = scaled[code].dropna()
            lw = 2.5 if idx == 0 else 1.0
            alp = 0.95 if idx == 0 else 0.65
            ax.plot(s.index, s.values, lw=lw, alpha=alp, color=colors[idx], label=f"#{idx+1} {code}")

        ax.axhline(1.0, color="black", lw=0.8, linestyle="--", alpha=0.35)
        ax.set_title(f"Top {top_n} TEFAS Funds — Scaled Price (base = 1.0)\n"
                     f"Filter ≥{MONTHLY_THRESHOLD*100:.2f}%/month | ≥{MIN_INVESTOR_COUNT} investors | "
                     f"1-6m horizons | {title_suffix} | Last: {true_last.date()}", fontsize=11)
        ax.set_xlabel("Date")
        ax.set_ylabel("Scaled Price")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}x"))
        ax.legend(loc="upper left", fontsize=6.5, ncol=3, framealpha=0.6)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        return fig

    st.divider()
    st.subheader("Scaled price charts")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.pyplot(plot_scaled(results_vol["fund_code"].tolist(),
                               "Ranked by lowest volatility", price_df, TOP_N))
    with chart_col2:
        st.pyplot(plot_scaled(results_sharpe["fund_code"].tolist(),
                               "Ranked by highest Sharpe", price_df, TOP_N))
