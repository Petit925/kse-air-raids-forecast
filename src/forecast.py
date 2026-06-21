"""Prophet forecasting wrapper + evaluation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from prophet import Prophet


def fit_prophet(train: pd.DataFrame) -> Prophet:
    """train: DataFrame[date, alert_count]."""
    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.2,
    )
    df = train.rename(columns={"date": "ds", "alert_count": "y"})
    m.fit(df)
    return m


def predict_prophet(model: Prophet, horizon: int, last_train_date: pd.Timestamp) -> pd.DataFrame:
    future = pd.DataFrame(
        {"ds": pd.date_range(last_train_date + pd.Timedelta(days=1), periods=horizon, freq="D")}
    )
    fc = model.predict(future)
    out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date"})
    out["yhat"] = out["yhat"].clip(lower=0)
    out["yhat_lower"] = out["yhat_lower"].clip(lower=0)
    return out


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # MAPE undefined when y_true==0 (frequent on quiet days) — use SMAPE instead.
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom > 0
    smape = float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100) if mask.any() else float("nan")
    return {"mae": mae, "rmse": rmse, "smape_pct": smape}
