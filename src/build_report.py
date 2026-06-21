"""Generate a self-contained INTERACTIVE static HTML.

Looks and behaves like the Streamlit dashboard (app.py): sidebar with region /
curfew / window controls, three tabs, KPI cards, charts, schedule. But it is a
single .html file — no Python, no Streamlit, no install on the reviewer's side.

How it works
------------
* Server side (this script): for each of the 25 regions, pre-compute daily
  history, weekday/hourly seasonality, +7-day Prophet forecast, monthly heatmap,
  KPI numbers, and the raw inputs needed to compute schedules client-side
  (forecast yhat array + historical hourly probability distribution).
* All five Vega-Lite chart specs per region are serialised into one big
  REGIONS_DATA JSON object embedded inside the HTML.
* Client side (vanilla JS in the HTML): when the reviewer changes region,
  curfew, or window length, JS swaps the Vega specs and re-runs the
  scheduler (a 16-window grid-search per day — trivial in JS).

Run:
    python -m src.build_report
or double-click build_report.bat.
"""
from __future__ import annotations

import datetime as dt
import json
from html import escape
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from prophet import Prophet

from src.load_data import load_raw, filter_region, to_daily_counts
from src.scheduler import hourly_distribution

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report.html"

HORIZON = 7
WEEKDAY_UK = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
DEFAULT_REGION = "Kyiv City"

FULL_W = 1040
HALF_W = 500
SCHED_ROW_W = 940


# ───────────────────── chart spec builders ─────────────────────

def chart_daily_spec(daily: pd.DataFrame, region: str) -> dict:
    df = daily.assign(
        rolling_28=daily["alert_count"].rolling(28, min_periods=1).mean()
    )
    base = alt.Chart(df).encode(x=alt.X("date:T", title="Дата"))
    line_daily = base.mark_line(color="#bbb", opacity=0.6).encode(
        y=alt.Y("alert_count:Q", title="Тривог/день"),
        tooltip=["date:T", "alert_count:Q"],
    )
    line_smooth = base.mark_line(color="#ff4b4b", size=2).encode(y="rolling_28:Q")
    return (line_daily + line_smooth).properties(
        height=260, width=FULL_W, title=f"Денна історія — {region}"
    ).to_dict()


def chart_weekday_spec(daily: pd.DataFrame) -> dict:
    by = (
        daily.assign(dow=daily["date"].dt.dayofweek)
        .groupby("dow")["alert_count"].mean().reset_index()
        .assign(day=lambda d: d["dow"].map(lambda i: WEEKDAY_UK[i]))
    )
    return alt.Chart(by).mark_bar(color="#395676").encode(
        x=alt.X("day:N", sort=WEEKDAY_UK, title=""),
        y=alt.Y("alert_count:Q", title="Середнє/день"),
        tooltip=["day:N", alt.Tooltip("alert_count:Q", format=".2f")],
    ).properties(height=240, width=HALF_W, title="Сезонність по днях тижня").to_dict()


def chart_hourly_spec(df_region: pd.DataFrame) -> dict:
    s = df_region["started_at"].dt.tz_convert("Europe/Kyiv")
    counts = s.dt.hour.value_counts().sort_index()
    arr = np.zeros(24, dtype=float)
    for h, c in counts.items():
        arr[h] = c
    total_days = max((s.max() - s.min()).days, 1)
    rate = arr / total_days
    df = pd.DataFrame({"hour": np.arange(24), "rate": rate})
    return alt.Chart(df).mark_bar(color="#5a9b3a").encode(
        x=alt.X("hour:O", title="Година (Європа/Київ)"),
        y=alt.Y("rate:Q", title="Тривог стартує / день"),
        tooltip=["hour:O", alt.Tooltip("rate:Q", format=".2f")],
    ).properties(height=240, width=HALF_W, title="Сезонність по годинах").to_dict()


def chart_heatmap_spec(daily: pd.DataFrame) -> dict:
    heat = daily.assign(
        year=daily["date"].dt.year,
        month=daily["date"].dt.month,
    ).groupby(["year", "month"])["alert_count"].sum().reset_index()
    return alt.Chart(heat).mark_rect().encode(
        x=alt.X("month:O", title="Місяць"),
        y=alt.Y("year:O", title="Рік", sort="descending"),
        color=alt.Color("alert_count:Q", scale=alt.Scale(scheme="reds"),
                        title="Тривог/місяць"),
        tooltip=["year:O", "month:O", "alert_count:Q"],
    ).properties(height=180, width=FULL_W, title="Місячна теплова карта").to_dict()


def chart_forecast_spec(daily: pd.DataFrame, fc: pd.DataFrame, region: str) -> dict:
    tail = daily.tail(60).rename(columns={"alert_count": "y"})
    hist = alt.Chart(tail).mark_line(color="#262730").encode(
        x=alt.X("date:T", title="Дата"),
        y=alt.Y("y:Q", title="Тривог/день"),
        tooltip=["date:T", "y:Q"],
    )
    line = alt.Chart(fc).mark_line(color="#0068c9", size=2).encode(
        x="date:T", y="yhat:Q",
        tooltip=["date:T",
                 alt.Tooltip("yhat:Q", format=".2f"),
                 alt.Tooltip("yhat_lower:Q", format=".2f"),
                 alt.Tooltip("yhat_upper:Q", format=".2f")],
    )
    band = alt.Chart(fc).mark_area(color="#0068c9", opacity=0.2).encode(
        x="date:T", y=alt.Y("yhat_lower:Q", title=""), y2="yhat_upper:Q",
    )
    return (hist + band + line).properties(
        height=280, width=FULL_W,
        title=f"Прогноз +7 днів — {region}"
    ).to_dict()


# ───────────────────── Prophet ─────────────────────

def fit_and_forecast(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.2,
    )
    m.fit(daily.rename(columns={"date": "ds", "alert_count": "y"}))
    last = daily["date"].max()
    future = pd.DataFrame({"ds": pd.date_range(last + pd.Timedelta(days=1),
                                                periods=horizon, freq="D")})
    fc = m.predict(future)
    out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date"})
    out["yhat"] = out["yhat"].clip(lower=0)
    out["yhat_lower"] = out["yhat_lower"].clip(lower=0)
    return out


# ───────────────────── per-region payload ──────────────────────

def build_region_payload(df_full: pd.DataFrame, region: str) -> dict:
    df_region = filter_region(df_full, region)
    daily = to_daily_counts(df_region, kyiv_local=True)
    fc = fit_and_forecast(daily, HORIZON)
    h_dist = hourly_distribution(df_region)

    total = int(daily["alert_count"].sum())
    last_30 = int(daily.tail(30)["alert_count"].sum())
    prev_30 = int(daily.tail(60).head(30)["alert_count"].sum())
    last_7 = int(daily.tail(7)["alert_count"].sum())
    mean_day = float(daily["alert_count"].mean())

    forecast_rows = [
        {
            "weekday": WEEKDAY_UK[pd.Timestamp(r["date"]).dayofweek],
            "date": pd.Timestamp(r["date"]).date().isoformat(),
            "yhat": round(float(r["yhat"]), 2),
            "yhat_lower": round(float(r["yhat_lower"]), 2),
            "yhat_upper": round(float(r["yhat_upper"]), 2),
        }
        for _, r in fc.iterrows()
    ]

    return {
        "region": region,
        "period_start": daily["date"].min().date().isoformat(),
        "period_end": daily["date"].max().date().isoformat(),
        "kpis": {
            "total": total,
            "last_30": last_30,
            "delta_30": last_30 - prev_30,
            "last_7": last_7,
            "mean_day": round(mean_day, 2),
        },
        "charts": {
            "daily": chart_daily_spec(daily, region),
            "weekday": chart_weekday_spec(daily),
            "hourly": chart_hourly_spec(df_region),
            "heatmap": chart_heatmap_spec(daily),
            "forecast": chart_forecast_spec(daily, fc, region),
        },
        # Data for client-side scheduler:
        "yhat": [round(r["yhat"], 4) for r in forecast_rows],
        "forecast_dates": [r["date"] for r in forecast_rows],
        "forecast_weekdays": [r["weekday"] for r in forecast_rows],
        "forecast_rows": forecast_rows,
        "hourly_dist": [round(float(x), 6) for x in h_dist.tolist()],
    }


# ───────────────────── HTML ───────────────────────────

CSS = r"""
:root {
  --bg: #ffffff;
  --bg2: #f0f2f6;
  --ink: #262730;
  --muted: #6b6b6b;
  --border: #e6e6e6;
  --primary: #ff4b4b;
  --accent: #0068c9;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: "Source Sans Pro", -apple-system, "Segoe UI", Roboto, sans-serif;
  color: var(--ink);
  background: var(--bg);
  line-height: 1.5;
}
.layout { display: grid; grid-template-columns: 280px 1fr; min-height: 100vh; }
.sidebar {
  background: var(--bg2);
  border-right: 1px solid var(--border);
  padding: 20px 18px;
}
.sidebar h2 { font-size: 18px; margin: 0 0 14px; }
.sidebar h3 { font-size: 14px; margin: 18px 0 8px; color: var(--muted);
              font-weight: 600; text-transform: none; }
.sidebar label { display:block; font-size: 13px; margin: 8px 0 4px;
                 color: var(--muted); }
.sidebar select, .sidebar input[type="number"] {
  width: 100%; padding: 6px 8px; border: 1px solid var(--border);
  border-radius: 6px; background: var(--bg); font-size: 14px;
  font-family: inherit; color: var(--ink);
}
.sidebar input[type="range"] { width: 100%; }
.range-row { display:flex; justify-content: space-between; font-size: 12px;
             color: var(--muted); margin-top: -4px; }
.sidebar .footer-note { color: var(--muted); font-size: 12px; margin-top: 24px;
                        border-top: 1px solid var(--border); padding-top: 14px; }
.main { padding: 22px 28px; max-width: 1180px; }
h1 { font-size: 30px; margin: 0 0 4px; font-weight: 700; }
.subtitle { color: var(--muted); font-size: 14px; margin-bottom: 18px; }
.kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
           margin: 10px 0 22px; }
.kpi { background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
       padding: 12px 14px; }
.kpi .label { color: var(--muted); font-size: 13px; }
.kpi .value { font-size: 28px; font-weight: 600; margin-top: 4px; }
.kpi .delta { font-size: 12px; margin-top: 4px; display: inline-block; }
.kpi .delta.down { color: #07a700; }   /* down = good for alerts */
.kpi .delta.up { color: var(--primary); }

.tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--border);
        margin-bottom: 18px; }
.tab-button {
  background: transparent; border: 0; padding: 10px 16px; font-size: 15px;
  cursor: pointer; color: var(--muted); font-family: inherit;
  border-bottom: 3px solid transparent; transform: translateY(1px);
}
.tab-button.active { color: var(--primary); border-bottom-color: var(--primary); font-weight: 600; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }

.chart {
  background: var(--bg); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px; margin: 10px 0 18px;
  overflow-x: auto;
}
.row-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 1000px) {
  .layout { grid-template-columns: 1fr; }
  .sidebar { border-right: 0; border-bottom: 1px solid var(--border); }
  .kpi-row { grid-template-columns: repeat(2, 1fr); }
  .row-2 { grid-template-columns: 1fr; }
}
.caption { color: var(--muted); font-size: 13px; margin-top: -8px;
           margin-bottom: 18px; }
table { width: 100%; border-collapse: collapse; background: var(--bg);
        border: 1px solid var(--border); border-radius: 8px;
        overflow: hidden; margin: 6px 0 18px; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border);
         font-size: 13px; }
th { background: var(--bg2); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.legend { display: flex; gap: 14px; font-size: 12px; color: var(--muted);
          margin: 6px 0 12px; flex-wrap: wrap; }
.legend .swatch { display: inline-block; width: 10px; height: 10px;
                  vertical-align: middle; margin-right: 4px; border-radius: 2px; }
details { background: var(--bg); border: 1px solid var(--border);
          border-radius: 8px; padding: 10px 14px; margin: 14px 0; }
summary { cursor: pointer; font-weight: 600; }
.download-btn {
  display: inline-block; padding: 8px 14px; background: var(--bg);
  border: 1px solid var(--border); border-radius: 6px; cursor: pointer;
  font-family: inherit; font-size: 14px; color: var(--ink); margin: 6px 0;
}
.download-btn:hover { background: var(--bg2); border-color: var(--muted); }
.warn { background: #fffbe6; border: 1px solid #ffe69c; padding: 8px 12px;
        border-radius: 6px; font-size: 13px; color: #806600; margin: 10px 0; }
"""

JS = r"""
let CURRENT_REGION = DEFAULT_REGION;
const VEGA_OPTS = { actions: false, theme: 'none' };
const EMBED_KEYS = ['daily', 'weekday', 'hourly', 'forecast', 'heatmap'];

function fmtInt(x) { return String(x).replace(/\B(?=(\d{3})+(?!\d))/g, ' '); }
function fmt1(x) { return Number(x).toFixed(1); }
function fmt2(x) { return Number(x).toFixed(2); }
function el(id) { return document.getElementById(id); }

function setKPI(kpis) {
  el('kpi-total').textContent = fmtInt(kpis.total);
  el('kpi-last30').textContent = fmtInt(kpis.last_30);
  el('kpi-last7').textContent = fmtInt(kpis.last_7);
  el('kpi-mean').textContent = fmt1(kpis.mean_day);
  const d = kpis.delta_30;
  const sign = d > 0 ? '+' : '';
  const cls = d <= 0 ? 'down' : 'up';
  el('kpi-delta30').textContent = `${sign}${d} vs попередні 30`;
  el('kpi-delta30').className = `delta ${cls}`;
}

function setPeriod(region, start, end) {
  el('header-region').textContent = region;
  el('header-period').textContent = `Період: ${start} → ${end}`;
}

function renderCharts(payload) {
  for (const k of EMBED_KEYS) {
    const div = el(`chart-${k}`);
    div.innerHTML = '';
    vegaEmbed(div, payload.charts[k], VEGA_OPTS).catch(console.error);
  }
}

// ─── client-side scheduler ──────────────────────────────────────
function isInCurfew(hour, curfewStart, curfewEnd) {
  // Handles wrap-around (e.g. 22..6 means 22:00–06:00 next day)
  if (curfewEnd > curfewStart) return hour >= curfewStart && hour < curfewEnd;
  return hour >= curfewStart || hour < curfewEnd;
}

function findBestWindow(hourlyExpected, windowHours, curfewStart, curfewEnd) {
  let best = null;
  for (let start = 0; start + windowHours <= 24; start++) {
    let inCurfew = false;
    for (let h = start; h < start + windowHours; h++) {
      if (isInCurfew(h, curfewStart, curfewEnd)) { inCurfew = true; break; }
    }
    if (inCurfew) continue;
    let risk = 0;
    for (let h = start; h < start + windowHours; h++) risk += hourlyExpected[h];
    if (best === null || risk < best.risk) {
      best = { start, end: start + windowHours, risk };
    }
  }
  return best;
}

function computeSchedule(payload, windowHours, curfewStart, curfewEnd) {
  const dist = payload.hourly_dist;
  const rows = [];
  for (let i = 0; i < payload.yhat.length; i++) {
    const yhat = payload.yhat[i];
    const hourlyExp = dist.map(p => p * yhat);
    const best = findBestWindow(hourlyExp, windowHours, curfewStart, curfewEnd);
    if (!best) {
      rows.push({
        weekday: payload.forecast_weekdays[i],
        date: payload.forecast_dates[i],
        yhat,
        start: '—', end: '—',
        alerts_in_block: 0,
        alerts_outside_block: yhat,
        vs_naive_pct: 0,
        bestStart: null, bestEnd: null, hourlyExp,
        error: 'не вміщається у дозволені години',
      });
      continue;
    }
    const inSum = best.risk;
    const totalSum = hourlyExp.reduce((a, b) => a + b, 0);
    const outSum = totalSum - inSum;
    const naive = (windowHours / 24) * yhat;
    const vs = naive > 0 ? ((naive - inSum) / naive * 100) : 0;
    rows.push({
      weekday: payload.forecast_weekdays[i],
      date: payload.forecast_dates[i],
      yhat, start: `${String(best.start).padStart(2,'0')}:00`,
      end: `${String(best.end).padStart(2,'0')}:00`,
      alerts_in_block: inSum,
      alerts_outside_block: outSum,
      vs_naive_pct: vs,
      bestStart: best.start, bestEnd: best.end,
      hourlyExp,
    });
  }
  return rows;
}

function renderScheduleTable(rows) {
  const head = `<tr>
    <th>День</th><th>Дата</th><th class="num">Прогноз</th>
    <th>Початок</th><th>Кінець</th>
    <th class="num">Тривог у блоці</th><th class="num">Поза блоком</th>
    <th class="num">vs наївний, %</th>
  </tr>`;
  const body = rows.map(r => `<tr>
    <td>${r.weekday}</td><td>${r.date}</td>
    <td class="num">${fmt2(r.yhat)}</td>
    <td>${r.start}</td><td>${r.end}</td>
    <td class="num">${fmt2(r.alerts_in_block)}</td>
    <td class="num">${fmt2(r.alerts_outside_block)}</td>
    <td class="num">${fmt1(r.vs_naive_pct)}</td>
  </tr>`).join('');
  el('schedule-table').innerHTML = `<table>${head}${body}</table>`;
}

function renderForecastTable(rows) {
  const head = `<tr>
    <th>День</th><th>Дата</th><th class="num">Прогноз</th>
    <th class="num">Мін</th><th class="num">Макс</th>
  </tr>`;
  const body = rows.map(r => `<tr>
    <td>${r.weekday}</td><td>${r.date}</td>
    <td class="num">${fmt2(r.yhat)}</td>
    <td class="num">${fmt2(r.yhat_lower)}</td>
    <td class="num">${fmt2(r.yhat_upper)}</td>
  </tr>`).join('');
  el('forecast-table').innerHTML = `<table>${head}${body}</table>`;
}

function renderScheduleCharts(rows, curfewStart, curfewEnd) {
  // Build long-form data, then a single vega-lite spec with row faceting.
  const data = [];
  for (const r of rows) {
    for (let h = 0; h < 24; h++) {
      const inCurfew = h >= curfewStart && h < curfewEnd;
      const inBlock = (r.bestStart !== null) && (h >= r.bestStart && h < r.bestEnd);
      const bucket = inCurfew ? "🌙 комендантська"
                     : inBlock ? "👔 робочий блок"
                               : "поза блоком";
      data.push({
        day_label: `${r.weekday} ${r.date} (${r.start}–${r.end})`,
        hour: h, expected: r.hourlyExp[h], bucket,
      });
    }
  }
  const labels = rows.map(r => `${r.weekday} ${r.date} (${r.start}–${r.end})`);
  const spec = {
    $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
    data: { values: data },
    mark: 'bar',
    encoding: {
      x: { field: 'hour', type: 'ordinal', axis: { labelAngle: 0 }, title: null },
      y: { field: 'expected', type: 'quantitative', title: 'Очікувано тривог' },
      color: {
        field: 'bucket', type: 'nominal',
        scale: { domain: ['🌙 комендантська', '👔 робочий блок', 'поза блоком'],
                 range: ['#262730', '#2a7', '#bbb'] },
        legend: { orient: 'bottom' },
      },
      row: { field: 'day_label', type: 'nominal', title: null, sort: labels },
      tooltip: [
        { field: 'day_label', type: 'nominal' },
        { field: 'hour', type: 'ordinal' },
        { field: 'expected', type: 'quantitative', format: '.3f' },
        { field: 'bucket', type: 'nominal' },
      ],
    },
    height: 80, width: SCHED_ROW_W,
    config: { view: { stroke: null } },
  };
  const div = el('chart-schedule');
  div.innerHTML = '';
  vegaEmbed(div, spec, VEGA_OPTS).catch(console.error);
}

// ─── orchestration ──────────────────────────────────────────────
function rerender() {
  const payload = REGIONS_DATA[CURRENT_REGION];
  const windowHours = parseInt(el('input-window').value, 10);
  const curfewStart = parseInt(el('input-curfew-start').value, 10);
  const curfewEnd = parseInt(el('input-curfew-end').value, 10);
  el('window-display').textContent = windowHours;
  el('curfew-display').textContent =
    `${String(curfewStart).padStart(2,'0')}:00–${String(curfewEnd).padStart(2,'0')}:00`;
  setKPI(payload.kpis);
  setPeriod(payload.region, payload.period_start, payload.period_end);
  renderCharts(payload);
  renderForecastTable(payload.forecast_rows);
  const sched = computeSchedule(payload, windowHours, curfewStart, curfewEnd);
  renderScheduleTable(sched);
  renderScheduleCharts(sched, curfewStart, curfewEnd);
  // Cache for CSV export
  window._currentSchedule = sched;
}

function setupTabs() {
  document.querySelectorAll('.tab-button').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      document.querySelectorAll('.tab-button').forEach(b => b.classList.toggle('active', b === btn));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${target}`));
    });
  });
}

function setupControls() {
  el('input-region').addEventListener('change', e => {
    CURRENT_REGION = e.target.value;
    rerender();
  });
  el('input-curfew-start').addEventListener('change', rerender);
  el('input-curfew-end').addEventListener('change', rerender);
  el('input-window').addEventListener('input', rerender);

  el('download-csv').addEventListener('click', () => {
    const rows = window._currentSchedule || [];
    if (!rows.length) return;
    const header = ['day', 'date', 'yhat', 'start', 'end',
                    'alerts_in_block', 'alerts_outside_block', 'vs_naive_pct'];
    const csv = [header.join(',')].concat(rows.map(r => [
      r.weekday, r.date, fmt2(r.yhat), r.start, r.end,
      fmt2(r.alerts_in_block), fmt2(r.alerts_outside_block), fmt1(r.vs_naive_pct),
    ].join(','))).join('\n');
    const blob = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `work_schedule__${CURRENT_REGION.replace(/ /g, '_').toLowerCase()}.csv`;
    a.click();
  });
}

window.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupControls();
  rerender();
});
"""

VEGA_CDN = """
<script src="https://cdn.jsdelivr.net/npm/vega@5.25.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5.16.3"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.22.2"></script>
"""


def render_html(regions_data: dict, dataset_meta: dict, region_order: list) -> str:
    today = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    region_options = "\n".join(
        f'      <option value="{escape(r)}"{" selected" if r == DEFAULT_REGION else ""}>{escape(r)}</option>'
        for r in region_order
    )
    data_json = json.dumps(regions_data, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"""<!doctype html>
<html lang="uk">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>🛡️ Air Raid Workforce Planner</title>
<style>{CSS}</style>
{VEGA_CDN}
</head>
<body>
<div class="layout">

<aside class="sidebar">
  <h2>⚙️ Параметри</h2>

  <label for="input-region">Регіон</label>
  <select id="input-region">
{region_options}
  </select>

  <h3>🌙 Комендантська</h3>
  <label for="input-curfew-start">Початок (год)</label>
  <input id="input-curfew-start" type="number" min="0" max="23" value="0" />
  <label for="input-curfew-end">Кінець (год)</label>
  <input id="input-curfew-end" type="number" min="1" max="24" value="5" />
  <div class="range-row">
    <span>зараз: <strong id="curfew-display">00:00–05:00</strong></span>
  </div>

  <h3>👔 Робочий блок</h3>
  <label for="input-window">Годин (8 роб + 1 обід = 9)</label>
  <input id="input-window" type="range" min="6" max="12" value="9" />
  <div class="range-row"><span>6</span><span><strong id="window-display">9</strong></span><span>12</span></div>

  <div class="footer-note">
    Дані: Vadimkin/ukrainian-air-raid-sirens-dataset · {dataset_meta['total_rows']:,} рядків · {dataset_meta['n_regions']} регіонів<br/>
    Згенеровано: {today}<br/>
    Період: {dataset_meta['period_start']} → {dataset_meta['period_end']}
  </div>
</aside>

<main class="main">
  <h1>🛡️ Air Raid Workforce Planner</h1>
  <p class="subtitle">Регіон: <strong id="header-region">—</strong> · <span id="header-period">—</span></p>

  <div class="kpi-row">
    <div class="kpi"><div class="label">Усього тривог</div><div class="value" id="kpi-total">—</div></div>
    <div class="kpi"><div class="label">Останні 30 днів</div><div class="value" id="kpi-last30">—</div><div class="delta down" id="kpi-delta30"></div></div>
    <div class="kpi"><div class="label">Останні 7 днів</div><div class="value" id="kpi-last7">—</div></div>
    <div class="kpi"><div class="label">Середнє/день</div><div class="value" id="kpi-mean">—</div></div>
  </div>

  <div class="tabs">
    <button class="tab-button active" data-tab="overview">📊 Огляд історії</button>
    <button class="tab-button" data-tab="forecast">🔮 Прогноз +7 днів</button>
    <button class="tab-button" data-tab="schedule">📅 Графік роботи на тиждень</button>
  </div>

  <section id="tab-overview" class="tab-panel active">
    <h3>Денна історія</h3>
    <div class="chart" id="chart-daily"></div>
    <p class="caption">Сіра — щодня; червона — 28-денне ковзне середнє (тренд).</p>

    <div class="row-2">
      <div class="chart" id="chart-weekday"></div>
      <div class="chart" id="chart-hourly"></div>
    </div>

    <div class="chart" id="chart-heatmap"></div>
  </section>

  <section id="tab-forecast" class="tab-panel">
    <h3>Прогноз на наступні 7 днів (Prophet)</h3>
    <div class="chart" id="chart-forecast"></div>
    <p class="caption">Чорна — фактична історія (останні 60 днів). Синя — Prophet прогноз. Смуга — діапазон невизначеності 80%.</p>
    <div id="forecast-table"></div>
  </section>

  <section id="tab-schedule" class="tab-panel">
    <h3>Оптимальний графік роботи</h3>
    <p class="caption">Параметри з sidebar: блок <strong><span id="schedule-window-mirror"></span></strong>,
       комендантська <strong><span id="schedule-curfew-mirror"></span></strong>.
       Алгоритм перебирає всі 9-годинні вікна що повністю лежать поза комендантською і обирає те, де сума очікуваних тривог найменша.</p>
    <div id="schedule-table"></div>
    <div class="legend">
      <span><span class="swatch" style="background:#262730"></span>комендантська</span>
      <span><span class="swatch" style="background:#2a7"></span>обраний робочий блок</span>
      <span><span class="swatch" style="background:#bbb"></span>дозволено, але не обрано</span>
    </div>
    <button class="download-btn" id="download-csv">⬇️ Завантажити CSV</button>
    <h3>Погодинний ризик з виділеним блоком</h3>
    <div class="chart" id="chart-schedule"></div>

    <details>
      <summary>Як читати ці числа</summary>
      <p><strong>Прогноз тривог</strong> — Prophet'ове очікування тривог на цілий день.</p>
      <p><strong>Тривог у блоці</strong> — скільки тривог потрапить у обрані 9 робочих годин (помножений на історичний погодинний розподіл регіону).</p>
      <p><strong>vs наївний, %</strong> — наскільки менше тривог у обраному вікні порівняно з рівномірним розподілом 9/24 від денного прогнозу.</p>
      <div class="warn"><strong>Це СЕРЕДНЄ очікування.</strong> У конкретний день може бути значно більше або менше — алгоритм мінімізує очікувану експозицію, не гарантує тиху зміну.</div>
    </details>
  </section>
</main>

</div>

<script>
const DEFAULT_REGION = {json.dumps(DEFAULT_REGION)};
const SCHED_ROW_W = {SCHED_ROW_W};
const REGIONS_DATA = {data_json};
{JS}
</script>
</body>
</html>
"""


def main() -> None:
    print("[1/3] Loading dataset...")
    df = load_raw()
    all_regions = sorted(df["region"].unique())
    dataset_meta = {
        "total_rows": len(df),
        "n_regions": df["region"].nunique(),
        "period_start": df["started_at"].min().date().isoformat(),
        "period_end": df["started_at"].max().date().isoformat(),
    }

    payloads = {}
    for i, region in enumerate(all_regions, 1):
        print(f"[2/3] Building region {i}/{len(all_regions)}: {region}")
        payloads[region] = build_region_payload(df, region)

    print("[3/3] Rendering HTML...")
    html = render_html(payloads, dataset_meta, all_regions)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"\nWrote {OUTPUT}  ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
