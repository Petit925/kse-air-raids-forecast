"""Seasonal-naive baseline: forecast(t) = observed(t - 7 days).

This is the standard sanity check for weekly-seasonal data — any model that
can't beat it is not worth shipping.
"""
import numpy as np
import pandas as pd


def seasonal_naive_7d(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Predict each day in the next `horizon` days as the value 7 days earlier
    (in the training tail). Returns DataFrame[date, yhat]."""
    last_date = train["date"].max()
    tail_7 = train.set_index("date")["alert_count"].iloc[-7:].values
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    yhat = np.array([tail_7[i % 7] for i in range(horizon)], dtype=float)
    return pd.DataFrame({"date": future_dates, "yhat": yhat})
