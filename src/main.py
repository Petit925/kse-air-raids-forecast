"""End-to-end pipeline: load → EDA figures → baseline & Prophet → metrics.

Run from project root:
    python -m src.main --region "Kyiv City" --test-days 14 --horizon 7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from src.load_data import load_raw, filter_region, to_daily_counts, save_processed
from src.eda import plot_daily, plot_weekday, plot_hourly
from src.baseline import seasonal_naive_7d
from src.forecast import fit_prophet, predict_prophet, evaluate

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def slugify(region: str) -> str:
    return region.lower().replace(" ", "_").replace("'", "")


def run(region: str, test_days: int, horizon: int) -> dict:
    print(f"[1/5] Load raw...")
    df = load_raw()
    print(f"      rows={len(df):,}  range={df['started_at'].min().date()}..{df['started_at'].max().date()}")

    print(f"[2/5] Filter region={region!r}, aggregate daily...")
    df_region = filter_region(df, region)
    daily = to_daily_counts(df_region, kyiv_local=True)
    save_processed(daily, slugify(region))
    print(f"      days={len(daily)}  total_alerts={int(daily['alert_count'].sum()):,}  mean/day={daily['alert_count'].mean():.1f}")

    slug = slugify(region)
    print(f"[3/5] EDA figures → reports/figures/  (slug={slug})")
    plot_daily(daily, region, slug)
    plot_weekday(daily, region, slug)
    plot_hourly(df_region, region, slug)

    print(f"[4/5] Split train/test (last {test_days} days held out), fit baseline + Prophet...")
    train = daily.iloc[:-test_days].copy()
    test = daily.iloc[-test_days:].copy()
    last_train_date = train["date"].max()

    base = seasonal_naive_7d(train, horizon=test_days)
    model = fit_prophet(train)
    prop = predict_prophet(model, horizon=test_days, last_train_date=last_train_date)

    y_true = test["alert_count"].values
    metrics = {
        "baseline_seasonal_naive_7d": evaluate(y_true, base["yhat"].values),
        "prophet": evaluate(y_true, prop["yhat"].values),
        "test_window": {
            "start": test["date"].min().date().isoformat(),
            "end": test["date"].max().date().isoformat(),
            "n_days": int(test_days),
            "mean_actual": float(y_true.mean()),
        },
        "region": region,
    }

    print(f"[5/5] Refit on full history, produce {horizon}-day forward forecast...")
    full_model = fit_prophet(daily)
    fwd = predict_prophet(full_model, horizon=horizon, last_train_date=daily["date"].max())
    fwd.to_csv(REPORTS / f"forecast_next_7d__{slug}.csv", index=False)

    # backtest figure
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    history_tail = daily.iloc[-90:]
    ax.plot(history_tail["date"], history_tail["alert_count"], color="#222", lw=1.0, label="actual")
    ax.plot(prop["date"], prop["yhat"], color="#c00", lw=1.5, label="prophet (backtest)")
    ax.plot(base["date"], base["yhat"], color="#3a7", lw=1.5, ls="--", label="seasonal-naive (backtest)")
    ax.axvline(last_train_date, color="#888", ls=":", lw=1, label="train/test split")
    ax.plot(fwd["date"], fwd["yhat"], color="#06c", lw=1.8, label="prophet forward 7d")
    ax.fill_between(fwd["date"], fwd["yhat_lower"], fwd["yhat_upper"], color="#06c", alpha=0.15)
    ax.set_title(f"Backtest + 7-day forecast — {region}")
    ax.legend(loc="upper left")
    ax.set_xlabel("date"); ax.set_ylabel("alerts per day")
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / f"backtest_forecast__{slug}.png", dpi=120)
    plt.close(fig)

    metrics_path = REPORTS / f"metrics__{slug}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\n=== METRICS ===")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="Kyiv City")
    ap.add_argument("--test-days", type=int, default=14)
    ap.add_argument("--horizon", type=int, default=7)
    args = ap.parse_args()
    run(args.region, args.test_days, args.horizon)


if __name__ == "__main__":
    main()
