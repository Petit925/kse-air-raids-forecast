"""Generate a self-contained static HTML report.

Why this exists. The Streamlit dashboard requires Python + Prophet + Streamlit.
An external reviewer (e.g. a course evaluator) should be able to read the
report by double-clicking a single .html file — no setup, no install.

Output: report.html in the project root. All charts are embedded as
Vega-Lite specs and rendered by vega-embed loaded from a CDN (~30 KB JS),
so the file is reproducible offline-ish (needs internet to load vega-embed).

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
from src.scheduler import CurfewWindow, build_schedule, hourly_distribution

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report.html"

REGIONS = ["Kyiv City", "Dnipropetrovska oblast", "Kharkivska oblast"]
CURFEW = CurfewWindow(start_hour=0, end_hour=5)
WINDOW_HOURS = 9
HORIZON = 7
WEEKDAY_UK = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]


# ───────────────────── model + chart helpers ─────────────────────

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
    future = pd.DataFrame({"ds": pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")})
    fc = m.predict(future)
    out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date"})
    out["yhat"] = out["yhat"].clip(lower=0)
    out["yhat_lower"] = out["yhat_lower"].clip(lower=0)
    return out


FULL_W = 1080
HALF_W = 520
SCHED_ROW_W = 980


def chart_daily(daily: pd.DataFrame, region: str) -> dict:
    df = daily.assign(rolling_28=daily["alert_count"].rolling(28, min_periods=1).mean())
    base = alt.Chart(df).encode(x=alt.X("date:T", title="Дата"))
    line_daily = base.mark_line(color="#aaa", opacity=0.6).encode(
        y=alt.Y("alert_count:Q", title="Тривог/день"),
        tooltip=["date:T", "alert_count:Q"],
    )
    line_smooth = base.mark_line(color="#c00", size=2).encode(y="rolling_28:Q")
    return (line_daily + line_smooth).properties(
        height=260, width=FULL_W, title=f"Денна історія тривог — {region}"
    ).to_dict()


def chart_weekday(daily: pd.DataFrame, region: str) -> dict:
    by = (
        daily.assign(dow=daily["date"].dt.dayofweek)
        .groupby("dow")["alert_count"].mean().reset_index()
        .assign(day=lambda d: d["dow"].map(lambda i: WEEKDAY_UK[i]))
    )
    return alt.Chart(by).mark_bar(color="#356").encode(
        x=alt.X("day:N", sort=WEEKDAY_UK, title=""),
        y=alt.Y("alert_count:Q", title="Середнє/день"),
        tooltip=["day:N", alt.Tooltip("alert_count:Q", format=".2f")],
    ).properties(height=240, width=HALF_W, title="По днях тижня").to_dict()


def chart_hourly(df_region: pd.DataFrame, region: str) -> dict:
    s = df_region["started_at"].dt.tz_convert("Europe/Kyiv")
    counts = s.dt.hour.value_counts().sort_index()
    arr = np.zeros(24, dtype=float)
    for h, c in counts.items():
        arr[h] = c
    total_days = max((s.max() - s.min()).days, 1)
    rate = arr / total_days
    df = pd.DataFrame({"hour": np.arange(24), "rate": rate})
    return alt.Chart(df).mark_bar(color="#5b3").encode(
        x=alt.X("hour:O", title="Година (Київ-час)"),
        y=alt.Y("rate:Q", title="Тривог стартує / день"),
        tooltip=["hour:O", alt.Tooltip("rate:Q", format=".2f")],
    ).properties(height=240, width=HALF_W, title="По годинах доби").to_dict()


def chart_forecast(daily: pd.DataFrame, fc: pd.DataFrame, region: str) -> dict:
    tail = daily.tail(60).rename(columns={"alert_count": "y"})
    hist = alt.Chart(tail).mark_line(color="#222").encode(
        x=alt.X("date:T", title="Дата"),
        y=alt.Y("y:Q", title="Тривог/день"),
        tooltip=["date:T", "y:Q"],
    )
    line = alt.Chart(fc).mark_line(color="#06c", size=2).encode(
        x="date:T", y="yhat:Q",
        tooltip=["date:T",
                 alt.Tooltip("yhat:Q", format=".2f"),
                 alt.Tooltip("yhat_lower:Q", format=".2f"),
                 alt.Tooltip("yhat_upper:Q", format=".2f")],
    )
    band = alt.Chart(fc).mark_area(color="#06c", opacity=0.2).encode(
        x="date:T", y=alt.Y("yhat_lower:Q", title=""), y2="yhat_upper:Q",
    )
    return (hist + band + line).properties(
        height=280, width=FULL_W,
        title=f"Прогноз на наступні 7 днів — {region}"
    ).to_dict()


def chart_schedule_per_day(schedule: pd.DataFrame, hourly_dist: np.ndarray, curfew: CurfewWindow) -> dict:
    rows = []
    for _, row in schedule.iterrows():
        hourly = hourly_dist * row["yhat"]
        start_h = int(row["start"].split(":")[0])
        end_h = int(row["end"].split(":")[0])
        for h in range(24):
            in_curfew = curfew.start_hour <= h < curfew.end_hour
            in_block = start_h <= h < end_h
            bucket = (
                "🌙 комендантська" if in_curfew
                else "👔 робочий блок" if in_block
                else "поза блоком"
            )
            rows.append({
                "day_label": f"{row['weekday']} {row['date']} ({row['start']}–{row['end']})",
                "hour": h,
                "expected": float(hourly[h]),
                "bucket": bucket,
            })
    df = pd.DataFrame(rows)
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("hour:O", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("expected:Q", title="Очікувано тривог"),
        color=alt.Color("bucket:N", scale=alt.Scale(
            domain=["🌙 комендантська", "👔 робочий блок", "поза блоком"],
            range=["#222", "#2a7", "#bbb"],
        ), legend=alt.Legend(orient="bottom")),
        row=alt.Row("day_label:N", title=None, sort=list(df["day_label"].unique())),
        tooltip=["day_label:N", "hour:O", alt.Tooltip("expected:Q", format=".3f"), "bucket:N"],
    ).properties(height=80, width=SCHED_ROW_W).to_dict()


# ───────────────────── per-region payload ──────────────────────

def build_region_section(df_full: pd.DataFrame, region: str) -> dict:
    df_region = filter_region(df_full, region)
    daily = to_daily_counts(df_region, kyiv_local=True)

    total = int(daily["alert_count"].sum())
    last_30 = int(daily.tail(30)["alert_count"].sum())
    prev_30 = int(daily.tail(60).head(30)["alert_count"].sum())
    last_7 = int(daily.tail(7)["alert_count"].sum())
    mean_day = float(daily["alert_count"].mean())

    fc = fit_and_forecast(daily, HORIZON)
    schedule = build_schedule(fc, df_region, window_hours=WINDOW_HOURS, curfew=CURFEW)
    hourly_dist = hourly_distribution(df_region)

    return {
        "region": region,
        "kpis": {
            "total": total,
            "last_30": last_30,
            "delta_30": last_30 - prev_30,
            "last_7": last_7,
            "mean_day": round(mean_day, 2),
            "period_start": daily["date"].min().date().isoformat(),
            "period_end": daily["date"].max().date().isoformat(),
        },
        "schedule_rows": schedule.to_dict(orient="records"),
        "forecast_rows": [
            {
                "day": WEEKDAY_UK[pd.Timestamp(r["date"]).dayofweek],
                "date": pd.Timestamp(r["date"]).date().isoformat(),
                "yhat": round(float(r["yhat"]), 1),
                "lower": round(float(r["yhat_lower"]), 1),
                "upper": round(float(r["yhat_upper"]), 1),
            }
            for _, r in fc.iterrows()
        ],
        "charts": {
            "daily": chart_daily(daily, region),
            "weekday": chart_weekday(daily, region),
            "hourly": chart_hourly(df_region, region),
            "forecast": chart_forecast(daily, fc, region),
            "schedule": chart_schedule_per_day(schedule, hourly_dist, CURFEW),
        },
    }


# ───────────────────── HTML rendering ───────────────────────────

CSS = """
:root { --bg:#fafafa; --card:#fff; --ink:#222; --muted:#666; --border:#e3e3e3;
       --accent:#06c; --good:#2a7; --bad:#c33; }
* { box-sizing:border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       background:var(--bg); color:var(--ink); margin:0; padding:0;
       line-height:1.5; }
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
h1 { font-size: 28px; margin: 0 0 6px; }
h2 { font-size: 22px; margin: 32px 0 10px; padding-top: 16px;
     border-top: 2px solid var(--border); }
h3 { font-size: 16px; margin: 18px 0 8px; color: var(--muted); font-weight:600; }
.meta { color: var(--muted); font-size: 13px; }
.kpi-row { display:grid; grid-template-columns: repeat(4, 1fr);
          gap: 12px; margin: 14px 0 22px; }
.kpi { background:var(--card); border:1px solid var(--border); border-radius:8px;
       padding:12px 14px; }
.kpi .label { color: var(--muted); font-size: 12px; }
.kpi .value { font-size: 24px; font-weight:600; margin-top:4px; }
.kpi .delta-bad { color: var(--bad); font-size:12px; margin-top:2px; }
.kpi .delta-good { color: var(--good); font-size:12px; margin-top:2px; }
.row-2 { display:grid; grid-template-columns: 1fr 1fr; gap:18px; }
@media (max-width: 800px) {
  .kpi-row { grid-template-columns: repeat(2,1fr); }
  .row-2 { grid-template-columns: 1fr; }
}
.chart { background:var(--card); border:1px solid var(--border); border-radius:8px;
         padding:14px; margin:10px 0 18px; min-height: 280px; }
table { width:100%; border-collapse: collapse; background:var(--card);
        border:1px solid var(--border); border-radius:8px; overflow:hidden;
        margin: 6px 0 18px; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border);
         font-size: 13px; }
th { background: #f0f0f0; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.legend { display:flex; gap:14px; font-size: 12px; color: var(--muted);
          margin: 6px 0 0; flex-wrap: wrap;}
.legend .swatch { display:inline-block; width:10px; height:10px;
                  vertical-align:middle; margin-right:4px; border-radius:2px; }
details { background:var(--card); border:1px solid var(--border); border-radius:8px;
          padding: 10px 14px; margin: 14px 0; }
summary { cursor: pointer; font-weight:600; }
footer { color: var(--muted); font-size: 13px; margin-top: 40px;
         padding-top: 16px; border-top: 1px solid var(--border); }
code { background: #eee; padding: 1px 4px; border-radius: 3px; font-size: 12px; }
"""

VEGA_CDN = """
<script src="https://cdn.jsdelivr.net/npm/vega@5.25.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5.16.3"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.22.2"></script>
"""


def render_kpi(label: str, value: str, delta: str | None = None, delta_good: bool = False) -> str:
    delta_html = ""
    if delta:
        cls = "delta-good" if delta_good else "delta-bad"
        delta_html = f'<div class="{cls}">{escape(delta)}</div>'
    return f'<div class="kpi"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div>{delta_html}</div>'


def render_chart_div(chart_id: str, spec: dict) -> str:
    spec_json = json.dumps(spec, default=str)
    return f"""
<div class="chart" id="{chart_id}"></div>
<script>
vegaEmbed("#{chart_id}", {spec_json}, {{actions:false, theme:"none"}}).catch(console.error);
</script>
"""


def render_schedule_table(rows: list[dict]) -> str:
    head = "<tr><th>День</th><th>Дата</th><th>Прогноз</th><th>Початок</th><th>Кінець</th><th>Тривог у блоці</th><th>Поза блоком</th><th>vs наївний, %</th></tr>"
    body = "".join(
        f"<tr><td>{escape(str(r['weekday']))}</td>"
        f"<td>{escape(str(r['date']))}</td>"
        f"<td class='num'>{r['yhat']:.2f}</td>"
        f"<td class='num'>{escape(str(r['start']))}</td>"
        f"<td class='num'>{escape(str(r['end']))}</td>"
        f"<td class='num'>{r['alerts_in_block']:.2f}</td>"
        f"<td class='num'>{r['alerts_outside_block']:.2f}</td>"
        f"<td class='num'>{r['vs_naive_pct']:.1f}</td></tr>"
        for r in rows
    )
    return f"<table>{head}{body}</table>"


def render_forecast_table(rows: list[dict]) -> str:
    head = "<tr><th>День</th><th>Дата</th><th>Прогноз</th><th>Мін</th><th>Макс</th></tr>"
    body = "".join(
        f"<tr><td>{escape(r['day'])}</td>"
        f"<td>{escape(r['date'])}</td>"
        f"<td class='num'>{r['yhat']:.1f}</td>"
        f"<td class='num'>{r['lower']:.1f}</td>"
        f"<td class='num'>{r['upper']:.1f}</td></tr>"
        for r in rows
    )
    return f"<table>{head}{body}</table>"


def render_region_section(section: dict, idx: int) -> str:
    k = section["kpis"]
    region = section["region"]
    delta = k["delta_30"]
    delta_str = f"{'+' if delta >= 0 else ''}{delta} vs попередні 30"
    return f"""
<h2 id="region-{idx}">{escape(region)}</h2>
<p class="meta">Період: {k['period_start']} → {k['period_end']}</p>
<div class="kpi-row">
{render_kpi("Усього тривог", f"{k['total']:,}".replace(",", " "))}
{render_kpi("Останні 30 днів", str(k['last_30']), delta_str, delta_good=(delta < 0))}
{render_kpi("Останні 7 днів", str(k['last_7']))}
{render_kpi("Середнє/день", f"{k['mean_day']:.1f}")}
</div>

<h3>Історія</h3>
{render_chart_div(f"chart-daily-{idx}", section['charts']['daily'])}

<div class="row-2">
  <div>
    <h3>Сезонність по днях тижня</h3>
    {render_chart_div(f"chart-weekday-{idx}", section['charts']['weekday'])}
  </div>
  <div>
    <h3>Сезонність по годинах</h3>
    {render_chart_div(f"chart-hourly-{idx}", section['charts']['hourly'])}
  </div>
</div>

<h3>Прогноз на наступні 7 днів (Prophet)</h3>
{render_chart_div(f"chart-forecast-{idx}", section['charts']['forecast'])}
{render_forecast_table(section['forecast_rows'])}

<h3>Оптимальний 9-годинний робочий блок на кожен день</h3>
<p class="meta">8 годин роботи + 1 година обіду. Комендантська 00:00–05:00.
Алгоритм перебирає всі 16 кандидатних вікон і обирає те, що мінімізує очікувану кількість тривог у робочих годинах.</p>
{render_schedule_table(section['schedule_rows'])}
<div class="legend">
  <span><span class="swatch" style="background:#222"></span>комендантська</span>
  <span><span class="swatch" style="background:#2a7"></span>обраний робочий блок</span>
  <span><span class="swatch" style="background:#bbb"></span>дозволено, але не обрано</span>
</div>
{render_chart_div(f"chart-schedule-{idx}", section['charts']['schedule'])}
"""


def render_html(sections: list[dict], dataset_meta: dict) -> str:
    today = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    region_nav = " · ".join(
        f'<a href="#region-{i}">{escape(s["region"])}</a>' for i, s in enumerate(sections)
    )
    sections_html = "".join(render_region_section(s, i) for i, s in enumerate(sections))

    return f"""<!doctype html>
<html lang="uk">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Air Raid Workforce Planner — статичний звіт</title>
<style>{CSS}</style>
{VEGA_CDN}
</head>
<body>
<div class="container">

<h1>🛡️ Air Raid Workforce Planner — статичний звіт</h1>
<p class="meta">
  Згенеровано: {today} ·
  Джерело: Vadimkin/ukrainian-air-raid-sirens-dataset (volunteer feed) ·
  {dataset_meta['total_rows']:,} рядків, {dataset_meta['n_regions']} регіонів,
  період {dataset_meta['period_start']} → {dataset_meta['period_end']}
</p>
<p>Регіони: {region_nav}</p>

<details open>
<summary>Що тут і навіщо</summary>
<p>Цей звіт — статична версія дашборду <code>app.py</code>. Інтерактивності з выбором регіона / комендантської тут немає, але всі чарти живі (hover, zoom через Vega-Lite). Призначений для того, хто хоче побачити повну картину без встановлення Python.</p>
<p><strong>Модель:</strong> Prophet (Meta) з тижневою та річною сезонністю, <code>changepoint_prior_scale=0.2</code>. Оцінка точності — у README.md (backtest по 2 регіонах: ~30% покращення MAE vs seasonal-naive baseline на Києві, ~19% на Дніпропетровській).</p>
<p><strong>Алгоритм графіка роботи:</strong> для кожного дня прогнозу — добове очікування тривог × історичний погодинний розподіл = очікувана крива по годинах. Серед усіх 9-годинних вікон, що повністю лежать поза комендантською, обирається те, де сума очікуваних тривог найменша.</p>
<p><strong>Чого модель НЕ враховує:</strong> поїздки до/з роботи, узгодженість графіків між днями (кожен день оптимізується незалежно), різкі зміни режиму (нова кампанія атак мід-тижня).</p>
</details>

{sections_html}

<footer>
<p>Дані: <a href="https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset">Vadimkin/ukrainian-air-raid-sirens-dataset</a> (MIT). Моделювання: <a href="https://facebook.github.io/prophet/">Prophet</a> (BSD). Чарти: <a href="https://vega.github.io/vega-lite/">Vega-Lite</a>.</p>
<p>Цей файл згенерований скриптом <code>src/build_report.py</code>. Перерендерити: <code>build_report.bat</code> у корені проекту.</p>
</footer>

</div>
</body>
</html>
"""


def main() -> None:
    print("[1/3] Loading dataset...")
    df = load_raw()
    dataset_meta = {
        "total_rows": len(df),
        "n_regions": df["region"].nunique(),
        "period_start": df["started_at"].min().date().isoformat(),
        "period_end": df["started_at"].max().date().isoformat(),
    }

    sections = []
    for i, region in enumerate(REGIONS, 1):
        print(f"[2/3] Building section {i}/{len(REGIONS)}: {region}")
        sections.append(build_region_section(df, region))

    print("[3/3] Rendering HTML...")
    html = render_html(sections, dataset_meta)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"\nWrote {OUTPUT}  ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
