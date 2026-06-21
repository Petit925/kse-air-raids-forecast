"""Quick EDA on a single region's daily series.

Produces 3 figures in reports/figures/:
  1. daily_series.png        — full daily count series with rolling mean
  2. weekday_seasonality.png — mean count by weekday
  3. hourly_seasonality.png  — mean alert start count by hour-of-day (local)
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

FIGURES = Path(__file__).resolve().parents[1] / "reports" / "figures"


def plot_daily(daily: pd.DataFrame, region_label: str, slug: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily["date"], daily["alert_count"], color="#888", lw=0.6, label="daily")
    ax.plot(
        daily["date"],
        daily["alert_count"].rolling(28, min_periods=1).mean(),
        color="#c00",
        lw=1.6,
        label="28-day rolling mean",
    )
    ax.set_title(f"Daily air-raid alerts — {region_label}")
    ax.set_xlabel("date")
    ax.set_ylabel("alerts per day")
    ax.legend()
    fig.tight_layout()
    out = FIGURES / f"daily_series__{slug}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_weekday(daily: pd.DataFrame, region_label: str, slug: str) -> Path:
    by_dow = daily.assign(dow=daily["date"].dt.dayofweek).groupby("dow")["alert_count"].mean()
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(labels, by_dow.values, color="#356")
    ax.set_title(f"Mean alerts by weekday — {region_label}")
    ax.set_ylabel("mean daily alerts")
    fig.tight_layout()
    out = FIGURES / f"weekday_seasonality__{slug}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_hourly(df_region: pd.DataFrame, region_label: str, slug: str) -> Path:
    """Alert START times — mean per hour-of-day across whole period (Kyiv local)."""
    s = df_region["started_at"].dt.tz_convert("Europe/Kyiv")
    by_hour = s.dt.hour.value_counts().sort_index()
    total_days = max((s.max() - s.min()).days, 1)
    rate = by_hour / total_days
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(rate.index, rate.values, color="#5b3")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title(f"Mean alerts starting in each hour-of-day — {region_label}")
    ax.set_xlabel("hour (Europe/Kyiv)")
    ax.set_ylabel("alerts started / day")
    fig.tight_layout()
    out = FIGURES / f"hourly_seasonality__{slug}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
