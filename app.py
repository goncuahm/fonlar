"""
TEFAS Fund Screener — Streamlit app
Port of the Colab "Part 1: update dataset / Part 2: screen & rank" script.

STORAGE BACKENDS (new):
  Local disk works fine for running this yourself or on your own server.
  On Streamlit Community Cloud specifically, the filesystem is EPHEMERAL --
  it does not survive app restarts, redeploys, or the sleep/wake cycle free
  apps go through. So if you're deploying there, pick the "S3-compatible
  bucket" backend instead (works with AWS S3, Cloudflare R2, Backblaze B2,
  MinIO, or anything else that speaks the S3 API via boto3). Configure
  credentials via st.secrets (see the sample secrets.toml in the sidebar
  help text) rather than typing them into the UI in production.

  Both backends implement the same tiny interface (list_files_with_sizes,
  exists, read_csv, write_csv, delete_files), so every other part of this
  app -- the fetch loop, the retention/pruning logic, the screener's
  loading and ranking -- is completely unaware of which one is active.

Key differences from the Colab version:
  - Part 1 (incremental daily CSV backfill) becomes a button: click
    "Update Dataset" and it fetches missing trading days one by one with
    a live progress bar + log, instead of running unattended in a cell.
  - Because the fetch loop is a genuine blocking loop (rate-limited with
    time.sleep between requests, same as the original), a "max days per
    click" cap lets you backfill incrementally across several clicks
    instead of blocking the browser for a very long time.
  - A rolling retention window auto-prunes files older than N years so
    storage doesn't grow without bound (see the Update Data tab).
  - Part 2 (screen & rank) is reactive: adjusting the threshold/TOP_N/
    investor-count controls re-filters and re-ranks instantly, because
    the (comparatively expensive) load+clean step is cached separately
    via st.cache_data and only re-runs when the file list actually changes.

Run with:  streamlit run tefas_streamlit_app.py
Requirements: streamlit, pytefas, pandas, numpy, matplotlib, boto3 (only
              needed if you use the S3-compatible backend)
"""

import io
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

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
except ImportError:
    boto3 = None
    ClientError = BotoCoreError = Exception


def safe_secrets(section):
    """st.secrets raises if no secrets.toml exists at all (common in local
    dev) -- swallow that and just return {} instead of crashing the app."""
    try:
        return dict(st.secrets.get(section, {}))
    except Exception:
        return {}


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


def human_size(n_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def files_older_than(files, retention_years):
    """Daily filenames whose DATE (embedded in the name) is older than the
    rolling retention window, oldest-first. Pure string/date logic --
    doesn't touch storage, works the same for either backend."""
    cutoff = date.today() - timedelta(days=int(retention_years * 365.25))
    old = [f for f in files if date.fromisoformat(f[12:22]) < cutoff]
    return sorted(old)   # filenames sort chronologically since they embed ISO dates


class StorageError(Exception):
    """Raised for backend-level failures (bad credentials, unreachable
    bucket, etc.) -- surfaced as a clean st.error() instead of a traceback."""
    pass


# ════════════════════════════════════════════════════════════════════
#  STORAGE BACKENDS
# ════════════════════════════════════════════════════════════════════
class StorageBackend:
    """Common interface -- the rest of the app only ever talks to this,
    never to os.* or boto3 directly."""

    def list_files_with_sizes(self):
        """-> {filename: size_bytes} for every tefas_daily_*.csv file."""
        raise NotImplementedError

    def exists(self, filename):
        raise NotImplementedError

    def read_csv(self, filename):
        raise NotImplementedError

    def write_csv(self, filename, df):
        raise NotImplementedError

    def delete_files(self, filenames):
        """-> (removed: list[str], failed: list[(str, str)])"""
        raise NotImplementedError


class LocalStorage(StorageBackend):
    def __init__(self, folder):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

    def _path(self, filename):
        return os.path.join(self.folder, filename)

    def list_files_with_sizes(self):
        result = {}
        for f in os.listdir(self.folder):
            if f.startswith("tefas_daily_") and f.endswith(".csv"):
                try:
                    result[f] = os.path.getsize(self._path(f))
                except OSError:
                    result[f] = 0
        return result

    def exists(self, filename):
        return os.path.exists(self._path(filename))

    def read_csv(self, filename):
        return pd.read_csv(self._path(filename), encoding="utf-8-sig", low_memory=False)

    def write_csv(self, filename, df):
        df.to_csv(self._path(filename), index=False, encoding="utf-8-sig")

    def delete_files(self, filenames):
        removed, failed = [], []
        for f in filenames:
            try:
                os.remove(self._path(f))
                removed.append(f)
            except OSError as e:
                failed.append((f, str(e)))
        return removed, failed


class S3Storage(StorageBackend):
    def __init__(self, bucket, prefix="", region_name=None, endpoint_url=None,
                 access_key_id=None, secret_access_key=None):
        if boto3 is None:
            raise StorageError("boto3 is not installed -- add it to requirements.txt "
                                "to use the S3-compatible backend.")
        self.bucket = bucket
        self.prefix = (prefix.rstrip("/") + "/") if prefix else ""
        client_kwargs = {}
        if region_name:
            client_kwargs["region_name"] = region_name
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        if access_key_id and secret_access_key:
            client_kwargs["aws_access_key_id"] = access_key_id
            client_kwargs["aws_secret_access_key"] = secret_access_key
        try:
            self.client = boto3.client("s3", **client_kwargs)
        except (ClientError, BotoCoreError, ValueError) as e:
            raise StorageError(f"Could not create S3 client: {e}")

    def _key(self, filename):
        return f"{self.prefix}{filename}"

    def list_files_with_sizes(self):
        result = {}
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket,
                                            Prefix=self.prefix + "tefas_daily_"):
                for obj in page.get("Contents", []):
                    filename = obj["Key"][len(self.prefix):]
                    if filename.endswith(".csv"):
                        result[filename] = obj["Size"]
        except (ClientError, BotoCoreError) as e:
            raise StorageError(f"Could not list bucket contents: {e}")
        return result

    def exists(self, filename):
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(filename))
            return True
        except ClientError:
            return False
        except BotoCoreError as e:
            raise StorageError(f"Could not check for {filename}: {e}")

    def read_csv(self, filename):
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self._key(filename))
            body = obj["Body"].read()
        except (ClientError, BotoCoreError) as e:
            raise StorageError(f"Could not read {filename}: {e}")
        return pd.read_csv(io.BytesIO(body), encoding="utf-8-sig", low_memory=False)

    def write_csv(self, filename, df):
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        try:
            self.client.put_object(Bucket=self.bucket, Key=self._key(filename), Body=csv_bytes)
        except (ClientError, BotoCoreError) as e:
            raise StorageError(f"Could not write {filename}: {e}")

    def delete_files(self, filenames):
        removed, failed = [], []
        keys = [self._key(f) for f in filenames]
        # S3 batch delete accepts up to 1000 keys per call
        for i in range(0, len(keys), 1000):
            batch_files = filenames[i:i + 1000]
            batch_keys = keys[i:i + 1000]
            try:
                resp = self.client.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": [{"Key": k} for k in batch_keys]}
                )
                deleted_keys = {d["Key"] for d in resp.get("Deleted", [])}
                error_keys = {e["Key"]: e.get("Message", "unknown error")
                              for e in resp.get("Errors", [])}
                for f, k in zip(batch_files, batch_keys):
                    if k in deleted_keys:
                        removed.append(f)
                    else:
                        failed.append((f, error_keys.get(k, "not confirmed deleted")))
            except (ClientError, BotoCoreError) as e:
                for f in batch_files:
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
KIND = st.sidebar.selectbox("Fund kind", ["YAT", "EMK", "BYF"], index=0,
                             help="YAT=mutual funds, EMK=pension, BYF=ETF")

st.sidebar.subheader("Storage backend")
BACKEND_CHOICE = st.sidebar.radio(
    "Where the daily CSVs live", ["Local folder", "S3-compatible bucket"], index=0,
    help="Local folder is fine for running this yourself or on your own "
         "server with a persistent disk. On Streamlit Community Cloud the "
         "filesystem is ephemeral and does NOT survive restarts/redeploys "
         "-- use S3-compatible storage there instead (AWS S3, Cloudflare "
         "R2, Backblaze B2, MinIO, etc. -- anything boto3 can talk to)."
)

storage = None
storage_cache_key = "unset"

if BACKEND_CHOICE == "Local folder":
    DATA_FOLDER = st.sidebar.text_input("Local data folder", value="./tefas_data/")
    storage = LocalStorage(DATA_FOLDER)
    storage_cache_key = f"local:{DATA_FOLDER}"
else:
    if boto3 is None:
        st.sidebar.error("`boto3` is not installed in this environment. "
                          "Add it to requirements.txt to use S3-compatible storage.")
    s3_secrets = safe_secrets("s3")
    with st.sidebar.expander("S3 settings", expanded=(not s3_secrets)):
        st.caption(
            "Prefer setting these via **st.secrets** in production (Streamlit "
            "Cloud: Settings → Secrets) rather than typing them here:\n\n"
            "```toml\n[s3]\nbucket = \"your-bucket\"\nprefix = \"tefas/\"\n"
            "region = \"auto\"\nendpoint_url = \"\"  # blank for AWS S3\n"
            "access_key_id = \"...\"\nsecret_access_key = \"...\"\n```"
        )
        bucket = st.text_input("Bucket name", value=s3_secrets.get("bucket", ""))
        prefix = st.text_input("Key prefix", value=s3_secrets.get("prefix", "tefas/"))
        region = st.text_input("Region (optional)", value=s3_secrets.get("region", ""))
        endpoint_url = st.text_input(
            "Custom endpoint URL (blank for AWS S3; needed for R2/B2/MinIO)",
            value=s3_secrets.get("endpoint_url", "")
        )
        access_key_id = s3_secrets.get("access_key_id") or st.text_input(
            "Access key ID", type="password")
        secret_access_key = s3_secrets.get("secret_access_key") or st.text_input(
            "Secret access key", type="password")

    if not (bucket and access_key_id and secret_access_key):
        st.sidebar.warning("Fill in bucket + credentials above (or via "
                            "st.secrets) to enable S3 storage.")
    else:
        try:
            storage = S3Storage(bucket=bucket, prefix=prefix, region_name=region or None,
                                 endpoint_url=endpoint_url or None,
                                 access_key_id=access_key_id, secret_access_key=secret_access_key)
            storage_cache_key = f"s3:{bucket}:{prefix}"
        except StorageError as e:
            st.sidebar.error(f"Could not connect: {e}")

if Crawler is None:
    st.sidebar.error("`pytefas` is not installed in this environment. "
                      "Add it to requirements.txt to enable data fetching. "
                      "The Screener tab still works on data already stored.")

if storage is None:
    st.warning("Configure a storage backend in the sidebar to continue.")
    st.stop()

tab_update, tab_screen = st.tabs(["📥 Update Data", "🏆 Screener"])


# ════════════════════════════════════════════════════════════════════
#  TAB 1 — Update Data  (Part 1 of the original script, as a button)
# ════════════════════════════════════════════════════════════════════
with tab_update:
    st.subheader("Retention & fetch settings")
    r1, r2, r3 = st.columns(3)
    RETENTION_YEARS = r1.number_input(
        "Keep last N years of data", min_value=0.5, max_value=10.0, value=3.0, step=0.5,
        help="Files older than this rolling window get pruned so storage "
             "doesn't grow without bound. Pruning only ever happens when "
             "you click Update Dataset or Prune Now below, never "
             "automatically on page load."
    )
    MAX_DAYS_PER_RUN = r2.number_input(
        "Max days to fetch per click", min_value=1, max_value=500, value=10,
        help="Caps how long a single click can block the app (at "
             "RATE_LIMIT_PAUSE seconds/day, 10 days ≈ 3-4 minutes). If more "
             "days are missing than this, click Update again afterward to "
             "keep backfilling -- it always resumes from the last date "
             "actually stored, so nothing is skipped."
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
             "too low risks TEFAS rate-limiting or blocking requests "
             "(TEFAS's own API applies a ~6 requests/minute limit)."
    )
    also_attempt_today = st.checkbox(
        "Also attempt today's same-day fetch after backfill", value=True,
        help="TEFAS usually finalizes NAVs the next business day, so this "
             "will often come back empty before ~09:00-10:00 the following "
             "morning -- that's expected, not an error."
    )

    st.divider()
    st.subheader("Dataset status")

    try:
        files_with_sizes = storage.list_files_with_sizes()
    except StorageError as e:
        st.error(f"Could not read from storage: {e}")
        st.stop()

    existing_files = sorted(files_with_sizes.keys())
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
    today_file = f"tefas_daily_{today_str}.csv"
    today_exists = today_file in existing_dates or storage.exists(today_file)

    disk_bytes = sum(files_with_sizes.values())
    prune_preview = files_older_than(existing_files, RETENTION_YEARS)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files stored", len(existing_files))
    c2.metric("Last date stored", last_on_disk.isoformat() if last_on_disk else "—")
    c3.metric("Missing trading days", len(missing_days))
    c4.metric("Today's file", "✅ saved" if today_exists else "not yet")

    c5, c6 = st.columns(2)
    c5.metric("Storage used", human_size(disk_bytes))
    c6.metric(f"Older than {RETENTION_YEARS:g}y (prunable)", len(prune_preview))

    if not existing_files:
        st.info(f"Nothing stored yet — the first update will bootstrap "
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
        storage_errors = []

        if days_to_fetch:
            st.write(f"Fetching {len(days_to_fetch)} day(s) "
                     f"({days_to_fetch[0]} → {days_to_fetch[-1]}) ...")
            progress = st.progress(0.0)
            log_box = st.container(height=260, border=True)
            errors = []

            for i, day in enumerate(days_to_fetch):
                day_str = day.strftime("%Y-%m-%d")
                filename = f"tefas_daily_{day_str}.csv"
                df_day, err = fetch_one_day(crawler, day_str, KIND)
                if err:
                    log_box.write(f"⚠️ {day_str}: {err}")
                    errors.append((day_str, err))
                elif df_day is None:
                    log_box.write(f"⏭️ {day_str}: no data (holiday?)")
                else:
                    try:
                        storage.write_csv(filename, df_day)
                        saved.append(day_str)
                        log_box.write(f"✅ {day_str}: {len(df_day):,} funds")
                    except StorageError as se:
                        log_box.write(f"❌ {day_str}: fetched OK but could not save — {se}")
                        storage_errors.append((day_str, str(se)))
                progress.progress((i + 1) / len(days_to_fetch))
                if i < len(days_to_fetch) - 1:
                    time.sleep(RATE_LIMIT_PAUSE)

            if errors:
                log_box.write(f"🔁 Retrying {len(errors)} failed day(s) ...")
                for day_str, _ in errors:
                    time.sleep(RATE_LIMIT_PAUSE)
                    filename = f"tefas_daily_{day_str}.csv"
                    df_day, err2 = fetch_one_day(crawler, day_str, KIND)
                    if df_day is not None:
                        try:
                            storage.write_csv(filename, df_day)
                            saved.append(day_str)
                            log_box.write(f"  {day_str} ✅ recovered")
                        except StorageError as se:
                            log_box.write(f"  {day_str} ❌ fetched but could not save — {se}")
                            storage_errors.append((day_str, str(se)))
                    elif err2:
                        log_box.write(f"  {day_str} ❌ {err2}")
                        still_failed.append(day_str)
                    else:
                        log_box.write(f"  {day_str} (still empty on retry)")

            st.success(f"Backfill run complete — saved {len(saved)}, "
                       f"failed {len(still_failed)}.")
            if storage_errors:
                st.error(f"{len(storage_errors)} day(s) fetched successfully but "
                         f"could not be saved to storage: {storage_errors}")
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
            elif storage.exists(today_file):
                st.info("Already saved.")
            else:
                df_today, err_today = fetch_one_day(crawler, today_str, KIND)
                if err_today:
                    st.warning(f"Fetch error: {err_today}")
                elif df_today is None:
                    st.info(f"No data published yet for {today_str} — normal "
                            f"if TEFAS hasn't finalized today's NAVs. Try again later.")
                else:
                    try:
                        storage.write_csv(today_file, df_today)
                        st.success(f"{today_str} NAVs are live — saved {len(df_today):,} funds.")
                    except StorageError as se:
                        st.error(f"Fetched today's data but could not save it: {se}")

        if ENABLE_AUTO_PRUNE:
            st.write("---")
            try:
                fresh_files = sorted(storage.list_files_with_sizes().keys())
                to_remove = files_older_than(fresh_files, RETENTION_YEARS)
                if to_remove:
                    removed, failed = storage.delete_files(to_remove)
                    st.write(f"**Retention pruning** (keeping last {RETENTION_YEARS:g} years):")
                    if removed:
                        st.success(f"🗑️ Pruned {len(removed)} file(s) older than the retention "
                                   f"window ({removed[0][12:22]} → {removed[-1][12:22]}).")
                    if failed:
                        st.warning(f"Could not remove {len(failed)} file(s): {failed}")
                else:
                    st.write(f"**Retention pruning:** nothing older than "
                             f"{RETENTION_YEARS:g} years — no files removed.")
            except StorageError as e:
                st.warning(f"Could not run retention pruning: {e}")

        st.divider()
        try:
            fresh = storage.list_files_with_sizes()
            st.write(f"**Updated status:** {len(fresh)} file(s) stored, "
                     f"{human_size(sum(fresh.values()))} total.")
        except StorageError as e:
            st.warning(f"Could not refresh status: {e}")

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
            try:
                removed, failed = storage.delete_files(prune_preview)
                if removed:
                    st.success(f"Removed {len(removed)} file(s).")
                if failed:
                    st.warning(f"Could not remove {len(failed)} file(s): {failed}")
            except StorageError as e:
                st.error(f"Pruning failed: {e}")
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
        help="Only load files from this many calendar days back -- must be "
             "comfortably more than the 6-month (126 trading-day) horizon. "
             "Capped at 1200 (~3.3 years) to match the Update Data tab's "
             "default retention window."
    )

    try:
        existing_files = sorted(storage.list_files_with_sizes().keys())
    except StorageError as e:
        st.error(f"Could not read from storage: {e}")
        st.stop()

    cutoff = date.today() - timedelta(days=int(LOOKBACK_DAYS))
    csv_files = sorted([
        f for f in existing_files
        if date.fromisoformat(f[12:22]) >= cutoff
    ])

    if not csv_files:
        st.warning("No files in the lookback window yet — use the "
                   "**Update Data** tab to fetch some first.")
        st.stop()

    # ── Cached, expensive step: load + clean raw CSVs ──────────────
    # _storage has a leading underscore so Streamlit doesn't try to hash
    # the backend object itself (it isn't hashable in a meaningful way,
    # especially the S3 client) -- cache_key + the file tuple do the
    # actual cache-invalidation work instead.
    @st.cache_data(show_spinner="Loading and cleaning data ...")
    def load_raw(_storage, cache_key, csv_files_tuple):
        frames = []
        for fname in csv_files_tuple:
            try:
                frames.append(_storage.read_csv(fname))
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
            storage, storage_cache_key, tuple(csv_files)
        )
    except ColumnDetectionError as e:
        st.error(
            f"⚠️ Problem reading the stored CSV files: {e}\n\n"
            f"This usually means one of the daily files has an unexpected "
            f"format (a corrupted download, an interrupted write, or a "
            f"pytefas schema change). Try re-fetching the affected day(s) "
            f"from the **Update Data** tab."
        )
        st.stop()

    if price_df_full is None:
        st.error("No files could be loaded.")
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
                 f"(latest snapshot)")
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
