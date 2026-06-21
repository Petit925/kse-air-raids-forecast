# 🛡️ Ukraine Air-Raid Alerts — Forecast & Workforce Planner

Mini pet-project for KSE AI Agentic Summer School (Stage 2).
**Task:** time-series analysis and short-horizon forecast of daily air-raid alert counts in Ukraine, plus an operational layer that turns the forecast into a 7-day work schedule for a critical-infrastructure / civil-defense team.

> Author: galabitskiy@gmail.com · 2026-06-21

## What's in here

1. **Static report** — `report.html`. Self-contained, no install required — just open it. **This is what an external reviewer should look at first.** Pre-rendered for 3 contrasting regions (Kyiv City, Dnipropetrovska, Kharkivska).
2. **Pipeline** — `src/` modules + `python -m src.main` for offline backtest and metrics.
3. **Interactive dashboard** — `app.py` (Streamlit) with 3 tabs: history overview, +7-day forecast, optimal work-schedule. Region/curfew/window-length are user-controlled.
4. **Launchers (Windows, one click)** —
   - `start_dashboard.bat` — opens the interactive Streamlit app in your browser.
   - `build_report.bat` — regenerates `report.html` from current data.

```
zero-install reviewer        →  double-click report.html  →  read
interactive demo / owner     →  double-click start_dashboard.bat  →  pick region & curfew
```

---

## Problem

Given a daily count of air-raid alerts for one Ukrainian region, forecast the count for each of the next 7 days. Hold out the last 14 days as a test window and compare two models against ground truth.

The volume varies by two orders of magnitude across regions (Kyiv City: ~1.5 alerts/day, frontline oblasts: 7-15/day). Any honest evaluation has to look at both ends of that distribution — a model that wins on a dense series can lose on a sparse one. We therefore run the pipeline for two contrasting regions:

| Region | Total alerts (since 2022-02-25) | Mean/day | Profile |
|---|---:|---:|---|
| **Kyiv City** | 2,294 | 1.5 | Capital, dense air defense, sparse and bursty |
| **Dnipropetrovska oblast** | 11,731 | 7.4 | Frontline-adjacent, dense and high-variance |

## Data

- **Source:** [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset), `datasets/volunteer_data_en.csv` (MIT).
- **Coverage:** 2022-02-25 → 2026-06-21 (1,578 days), 25 oblasts/cities, 101,448 rows.
- **Schema:** `region, started_at (UTC), finished_at (UTC), naive (bool)`.
- **Why this dataset:** uniform per-oblast granularity across the whole period (the official feed switched to raion-level mid-period); no API key; cited in academic forecasting work (arxiv:2411.14625).
- **Preprocessing:** convert UTC → Europe/Kyiv before bucketing to days so the day boundary matches civilian intuition; reindex to a continuous date range, filling missing days with 0 (an absent row = no alert).

## Method

```
raw csv  →  filter region  →  aggregate to daily count (Europe/Kyiv)
                                    ↓
                  train (everything except last 14 days)
                                    ↓
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
   baseline: seasonal naive (7d)          Prophet (weekly + yearly seasonality,
   forecast(t) = observed(t-7d)            additive, changepoint_prior_scale=0.2)
              │                                     │
              ▼                                     ▼
            evaluate on the test window: MAE, RMSE, SMAPE
                                    ↓
                       refit Prophet on full history
                                    ↓
                       7-day forward forecast
```

Baseline is a deliberately strong sanity check — for a series with any weekly pattern, "last week, same weekday" is a hard floor to beat. If Prophet doesn't beat it, the model isn't earning its complexity.

SMAPE is used in place of MAPE because zero-actual days are common in low-volume regions, and MAPE blows up to infinity on them. SMAPE is symmetric and bounded at 200%.

## Results

| Region | Model | MAE | RMSE | SMAPE % | Test-window mean (actual) |
|---|---|---:|---:|---:|---:|
| Kyiv City | seasonal-naive(7d) | 1.71 | 2.17 | 107.6 | 1.14 |
| Kyiv City | **Prophet** | **1.20** | **1.33** | **103.1** | 1.14 |
| Dnipropetrovska | seasonal-naive(7d) | 3.64 | 4.62 | 30.0 | 13.07 |
| Dnipropetrovska | **Prophet** | **2.94** | **3.83** | **25.4** | 13.07 |

Prophet beats the seasonal-naive baseline on every metric in every region. Improvement on MAE: **~30% for Kyiv City, ~19% for Dnipropetrovska**.

### Backtest visuals

- `reports/figures/backtest_forecast__kyiv_city.png`
- `reports/figures/backtest_forecast__dnipropetrovska_oblast.png`

### What the figures tell us beyond the metrics

- **Weekly seasonality is weak.** Mean alerts per weekday differ by <10% across Mon–Sun. War doesn't respect weekends — the model's edge comes from trend and yearly seasonality, not from the weekly cycle.
- **Strong intra-day pattern.** Alert *starts* peak around 10:00–15:00 Europe/Kyiv local and bottom out at 04:00–06:00. This is invisible at the daily aggregation we forecast; it would matter if we forecasted hourly.
- **The SMAPE gap is the story.** ~25% on a frontline oblast vs ~103% on the capital. Sparse series are genuinely hard — most days have 0 or 1 alert, so any small absolute error is huge in percentage terms. Honest reporting matters here; an MAPE-only table would have looked terrible.

## Reproduce — backtest pipeline

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# data lives in data/raw/volunteer_data_en.csv (committed)
python -m src.main --region "Kyiv City" --test-days 14 --horizon 7
python -m src.main --region "Dnipropetrovska oblast" --test-days 14 --horizon 7
```

Artifacts land in `reports/`:

- `metrics__<region>.json` — backtest metrics
- `forecast_next_7d__<region>.csv` — point + interval forecast
- `figures/*__<region>.png` — EDA + backtest plots

## Dashboard — `app.py`

```bash
streamlit run app.py
# or just double-click start_dashboard.bat on Windows
```

Three tabs:

1. **📊 Огляд історії** — KPI cards, daily series with 28-day rolling mean, weekday and hour-of-day distributions, year/month heatmap.
2. **🔮 Прогноз +7 днів** — Prophet forward forecast with 80% uncertainty band, weekday-labelled table.
3. **📅 Графік роботи на тиждень** — for each of the next 7 days, the dashboard finds the contiguous **9-hour work block** (8h work + 1h lunch) that minimises expected employee-exposure to air-raid alerts, subject to a configurable **curfew** (default 00:00–05:00). Each day shows a 24-hour bar chart with the curfew (black), chosen work block (green), and out-of-block hours (grey). Schedule is downloadable as CSV.

### How the schedule algorithm works (defense-scenario framing)

For every day `d` in the forecast:
1. Take the Prophet point forecast `yhat[d]` (expected alerts that day).
2. Take the historical hour-of-day probability distribution `p[h]` for the region (sums to 1, computed from the full dataset).
3. Compute expected alerts per hour: `e[h, d] = yhat[d] · p[h]`.
4. Among all 9-hour contiguous windows that fit *entirely inside the curfew-allowed range*, pick the one minimising `sum(e[h, d] for h in window)`.

This is a transparent grid search (16 candidate windows max per day) — easy to inspect, easy to defend in an interview, no opaque optimiser. The "vs naive %" column shows how much exposure is reduced compared to a manager who picked the work block uniformly at random across the 24-hour day.

**What this model does NOT capture** (stated honestly in the dashboard):
- Commute exposure before/after the block.
- Same-shift consistency across days (each day is optimised independently — actual deployments would want a smoothing constraint).
- Day-specific deviations from the historical hourly pattern.
- Sudden regime shifts (e.g. a new drone-strike campaign that arrives mid-week).

## Layout

```
src/
  load_data.py    pandas loader + region filter + daily aggregation (UTC → Europe/Kyiv)
  eda.py          daily / weekday / hourly figures
  baseline.py     seasonal-naive(7d)
  forecast.py     Prophet fit + predict + metrics (MAE/RMSE/SMAPE)
  scheduler.py    optimal 9h work-window finder with curfew constraint
  build_report.py self-contained static HTML generator
  main.py         end-to-end backtest pipeline runner
app.py              Streamlit dashboard (history / forecast / schedule)
report.html         pre-rendered static report (open directly, no install)
start_dashboard.bat one-click Windows launcher for the dashboard
build_report.bat    one-click regenerate of report.html
data/raw/           volunteer_data_en.csv
data/processed/     daily counts per region
reports/            metrics, forecasts, figures
```

## Limitations & honest next steps

- **One model family.** Compared Prophet vs naive. SARIMA / LightGBM with lag features / a small recurrent net would be the next ablation.
- **Daily horizon only.** Hourly forecasting is the more useful product for civil-defense scheduling but needs heavier modelling.
- **No regime-shift handling.** The series is non-stationary by definition (war intensity changes). A changepoint analysis on `changepoint_prior_scale` would tighten the forecast.
- **Volunteer data has gaps.** A few-day window may show under-counts when an oblast contributor goes offline. Joining the official Vadimkin file as a cross-check is the obvious next step.
- **Per-oblast point estimates, no fan chart for the test window.** Prophet returns intervals only for the future forecast in my current pipeline — extending to backtest intervals is straightforward.

## Reflection (~100 words)

See `REFLECTION.txt` for the submission-form text.

## Acknowledgements

Data: [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) (MIT).
Modelling: [Prophet](https://facebook.github.io/prophet/) by Meta (BSD).
