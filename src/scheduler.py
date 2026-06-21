"""Optimal work-window scheduler.

Given a daily alert forecast and the historical hour-of-day distribution
for a region, compute the work window (default 9 hours = 8h work + 1h lunch)
that minimises expected employee-exposure to air-raid alerts.

Honest about what this does NOT model:
  * Commute time before/after the window.
  * Same-shift consistency (each day is optimised independently).
  * Curfew exceptions (default applied as a hard window the schedule
    must fit inside).

The model is a contiguous-window argmin over expected hourly alerts.
Simple, transparent, easy to defend in an interview.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class CurfewWindow:
    """Hours during which work is NOT allowed.

    start_hour, end_hour are inclusive of start, exclusive of end (Europe/Kyiv).
    Default: 00:00–05:00 — Kyiv-city curfew as of mid-2026.
    """
    start_hour: int = 0
    end_hour: int = 5

    def allowed_hours(self) -> list[int]:
        """The hours (0..23) when work IS allowed."""
        return [h for h in range(24) if not (self.start_hour <= h < self.end_hour)]


def hourly_distribution(df_region: pd.DataFrame) -> np.ndarray:
    """Probability mass of an alert start falling in each hour-of-day.

    Returns a length-24 array summing to 1.0. Uses Europe/Kyiv local time.
    """
    s = df_region["started_at"].dt.tz_convert("Europe/Kyiv")
    counts = s.dt.hour.value_counts().sort_index()
    arr = np.zeros(24, dtype=float)
    for h, c in counts.items():
        arr[h] = c
    total = arr.sum()
    return arr / total if total > 0 else arr


def expected_alerts_per_hour(daily_forecast: float, hourly_dist: np.ndarray) -> np.ndarray:
    """Disaggregate a daily forecast into an hourly expected-count vector."""
    return hourly_dist * float(daily_forecast)


def find_best_window(
    hourly_expected: np.ndarray,
    window_hours: int = 9,
    curfew: CurfewWindow | None = None,
) -> tuple[int, int, float]:
    """Find the contiguous `window_hours`-long block that minimises the sum of
    expected alerts AND fits entirely inside the curfew-allowed range.

    Returns (start_hour, end_hour_exclusive, risk_sum).
    """
    if curfew is None:
        curfew = CurfewWindow()
    allowed = set(curfew.allowed_hours())

    best = None
    for start in range(24 - window_hours + 1):
        hours = range(start, start + window_hours)
        # Reject if any hour in the proposed block falls inside curfew.
        if not all(h in allowed for h in hours):
            continue
        risk = float(hourly_expected[start : start + window_hours].sum())
        if best is None or risk < best[2]:
            best = (start, start + window_hours, risk)

    if best is None:
        raise ValueError(
            f"No {window_hours}-hour window fits in the allowed hours "
            f"(curfew {curfew.start_hour:02d}:00–{curfew.end_hour:02d}:00). "
            f"Reduce window_hours or relax curfew."
        )
    return best


def build_schedule(
    forecast: pd.DataFrame,
    df_region_full: pd.DataFrame,
    window_hours: int = 9,
    curfew: CurfewWindow | None = None,
) -> pd.DataFrame:
    """Build a per-day work-schedule from a forecast DataFrame.

    Parameters
    ----------
    forecast : DataFrame with columns [date, yhat] — Prophet output.
    df_region_full : raw region rows (for the hourly distribution).
    window_hours : length of work block (8h work + 1h lunch = 9).
    curfew : CurfewWindow or None (defaults to 00–05).

    Returns
    -------
    DataFrame[date, weekday, yhat, start, end, expected_alerts_in_block,
              expected_alerts_outside_block, risk_reduction_pct]
    """
    dist = hourly_distribution(df_region_full)
    rows = []
    weekday_uk = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
    for _, r in forecast.iterrows():
        d = pd.to_datetime(r["date"])
        yhat = float(r["yhat"])
        hourly = expected_alerts_per_hour(yhat, dist)
        start, end, risk_in = find_best_window(hourly, window_hours, curfew)
        risk_out = float(hourly.sum() - risk_in)
        baseline_uniform_risk = (window_hours / 24.0) * yhat
        reduction = (
            (baseline_uniform_risk - risk_in) / baseline_uniform_risk * 100
            if baseline_uniform_risk > 0 else 0.0
        )
        rows.append({
            "date": d.date(),
            "weekday": weekday_uk[d.weekday()],
            "yhat": round(yhat, 2),
            "start": f"{start:02d}:00",
            "end": f"{end:02d}:00",
            "alerts_in_block": round(risk_in, 2),
            "alerts_outside_block": round(risk_out, 2),
            "vs_naive_pct": round(reduction, 1),
        })
    return pd.DataFrame(rows)
