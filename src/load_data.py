"""Load and clean Vadimkin volunteer air raid alerts dataset.

Source: https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset
File: datasets/volunteer_data_en.csv
"""
from pathlib import Path
import pandas as pd

RAW_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "volunteer_data_en.csv"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, parse_dates=["started_at", "finished_at"])
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)
    df["duration_min"] = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60
    return df


def filter_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Filter alerts for a specific region (e.g. 'Kyiv City', 'Kyivska oblast')."""
    sub = df[df["region"] == region].copy()
    if sub.empty:
        available = sorted(df["region"].unique())
        raise ValueError(
            f"No rows for region={region!r}. Available: {available[:10]}..."
        )
    return sub.sort_values("started_at").reset_index(drop=True)


def to_daily_counts(df_region: pd.DataFrame, kyiv_local: bool = True) -> pd.DataFrame:
    """Aggregate to daily alert counts.

    kyiv_local=True converts UTC timestamps to Europe/Kyiv before bucketing —
    important so the 'day' boundary matches civilian intuition.
    """
    starts = df_region["started_at"]
    if kyiv_local:
        starts = starts.dt.tz_convert("Europe/Kyiv")
    daily = (
        starts.dt.tz_localize(None).dt.floor("D")
        .value_counts()
        .sort_index()
        .rename_axis("date")
        .rename("alert_count")
        .reset_index()
    )
    full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = (
        daily.set_index("date")
        .reindex(full_range, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )
    return daily


def save_processed(daily: pd.DataFrame, region_slug: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{region_slug}_daily.csv"
    daily.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    df = load_raw()
    print(f"Rows: {len(df):,}")
    print(f"Date range: {df['started_at'].min()} .. {df['started_at'].max()}")
    print(f"Unique regions: {df['region'].nunique()}")
    print("\nTop 10 regions by alert count:")
    print(df["region"].value_counts().head(10))
