"""🛡️ Air Raid Workforce Planner — Streamlit dashboard.

Launch:
    streamlit run app.py
Or double-click start_dashboard.bat in the project root.

Purpose. Given the Vadimkin volunteer dataset of Ukraine air-raid alerts,
this dashboard helps a hypothetical critical-infrastructure / civil-defense
manager:
  1. See historical alert volume for any oblast.
  2. Get a 7-day forecast (Prophet, retrained per region).
  3. Get an OPTIMAL 9-hour work block for each of the next 7 days that
     minimises expected employee-exposure to alerts AND respects curfew.

Honest caveat. The forecast point-error is ±1.5–3 alerts/day; the value of
this product is mostly (a) the 9-hour-window risk-minimisation logic and
(b) regime-shift visibility, not pinpoint-accurate counts.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet

from src.load_data import load_raw, filter_region, to_daily_counts
from src.scheduler import CurfewWindow, build_schedule, hourly_distribution

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="🛡️ Air Raid Workforce Planner",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── data + model layer (cached) ──────────────────────────────────────────

@st.cache_data(show_spinner="Завантажую історичні тривоги (101k рядків)...")
def load_dataset() -> pd.DataFrame:
    return load_raw()


@st.cache_data(show_spinner=False)
def get_region_data(region: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_dataset()
    df_region = filter_region(df, region)
    daily = to_daily_counts(df_region, kyiv_local=True)
    return df_region, daily


@st.cache_resource(show_spinner="Навчаю Prophet для регіону...")
def fit_prophet_for(region: str) -> Prophet:
    _, daily = get_region_data(region)
    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.2,
    )
    m.fit(daily.rename(columns={"date": "ds", "alert_count": "y"}))
    return m


def forecast_next(region: str, horizon: int = 7) -> pd.DataFrame:
    m = fit_prophet_for(region)
    _, daily = get_region_data(region)
    last = daily["date"].max()
    future = pd.DataFrame({"ds": pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")})
    fc = m.predict(future)
    out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date"})
    out["yhat"] = out["yhat"].clip(lower=0)
    out["yhat_lower"] = out["yhat_lower"].clip(lower=0)
    return out


# ─── sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Параметри")
    df_all = load_dataset()
    regions = sorted(df_all["region"].unique())
    default_region = "Kyiv City" if "Kyiv City" in regions else regions[0]
    region = st.selectbox("Регіон", regions, index=regions.index(default_region))

    st.divider()
    st.subheader("🌙 Комендантська")
    curfew_start = st.number_input("Початок (год)", min_value=0, max_value=23, value=0)
    curfew_end = st.number_input("Кінець (год)", min_value=1, max_value=24, value=5)
    curfew = CurfewWindow(start_hour=int(curfew_start), end_hour=int(curfew_end))

    st.divider()
    st.subheader("👔 Робочий блок")
    window_hours = st.slider("Годин у блоці (8 робочих + 1 обід)", min_value=6, max_value=12, value=9)

    st.divider()
    st.caption(f"Дані: Vadimkin/ukrainian-air-raid-sirens-dataset · {len(df_all):,} рядків · {df_all['region'].nunique()} регіонів")


df_region, daily = get_region_data(region)

# ─── header + KPIs ────────────────────────────────────────────────────────

st.title("🛡️ Air Raid Workforce Planner")
st.caption(f"Регіон: **{region}** · Період: {daily['date'].min().date()} → {daily['date'].max().date()}")

col1, col2, col3, col4 = st.columns(4)
total = int(daily["alert_count"].sum())
last_30 = int(daily.tail(30)["alert_count"].sum())
last_7 = int(daily.tail(7)["alert_count"].sum())
mean_day = daily["alert_count"].mean()

col1.metric("Усього тривог", f"{total:,}")
col2.metric("Останні 30 днів", f"{last_30}", delta=f"{last_30 - int(daily.tail(60).head(30)['alert_count'].sum())} vs попередні 30")
col3.metric("Останні 7 днів", f"{last_7}")
col4.metric("Середнє/день", f"{mean_day:.1f}")

# ─── tabs ─────────────────────────────────────────────────────────────────

tab_overview, tab_forecast, tab_schedule = st.tabs([
    "📊 Огляд історії",
    "🔮 Прогноз +7 днів",
    "📅 Графік роботи на тиждень",
])


# ─── TAB 1 — overview ─────────────────────────────────────────────────────

with tab_overview:
    st.subheader("Денна історія")
    daily_chart_df = daily.assign(
        rolling_28=daily["alert_count"].rolling(28, min_periods=1).mean()
    )
    base = alt.Chart(daily_chart_df).encode(x=alt.X("date:T", title="Дата"))
    line_daily = base.mark_line(color="#aaa", opacity=0.6).encode(
        y=alt.Y("alert_count:Q", title="Тривог/день"),
        tooltip=["date:T", "alert_count:Q"],
    )
    line_smooth = base.mark_line(color="#c00", size=2).encode(y="rolling_28:Q")
    st.altair_chart(line_daily + line_smooth, use_container_width=True)
    st.caption("Сіра — щодня; червона — 28-денне ковзне середнє (тренд).")

    cA, cB = st.columns(2)
    with cA:
        st.subheader("По днях тижня")
        weekday_uk = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
        by_dow = (
            daily.assign(dow=daily["date"].dt.dayofweek)
            .groupby("dow")["alert_count"].mean()
            .reset_index()
            .assign(day=lambda d: d["dow"].map(lambda i: weekday_uk[i]))
        )
        st.altair_chart(
            alt.Chart(by_dow).mark_bar(color="#356").encode(
                x=alt.X("day:N", sort=weekday_uk, title=""),
                y=alt.Y("alert_count:Q", title="Середнє/день"),
                tooltip=["day:N", alt.Tooltip("alert_count:Q", format=".2f")],
            ),
            use_container_width=True,
        )

    with cB:
        st.subheader("По годинах доби (Київ-час)")
        dist = hourly_distribution(df_region)
        total_days = max((df_region["started_at"].max() - df_region["started_at"].min()).days, 1)
        rate = dist * len(df_region) / total_days
        hour_df = pd.DataFrame({"hour": np.arange(24), "rate": rate})
        st.altair_chart(
            alt.Chart(hour_df).mark_bar(color="#5b3").encode(
                x=alt.X("hour:O", title="Година (Київ-час)"),
                y=alt.Y("rate:Q", title="Тривог стартує / день"),
                tooltip=["hour:O", alt.Tooltip("rate:Q", format=".2f")],
            ),
            use_container_width=True,
        )

    st.subheader("Місячна теплова карта")
    heat_df = daily.assign(
        year=daily["date"].dt.year,
        month=daily["date"].dt.month,
    ).groupby(["year", "month"])["alert_count"].sum().reset_index()
    st.altair_chart(
        alt.Chart(heat_df).mark_rect().encode(
            x=alt.X("month:O", title="Місяць"),
            y=alt.Y("year:O", title="Рік", sort="descending"),
            color=alt.Color("alert_count:Q", scale=alt.Scale(scheme="reds"), title="Тривог за місяць"),
            tooltip=["year:O", "month:O", "alert_count:Q"],
        ),
        use_container_width=True,
    )


# ─── TAB 2 — forecast ─────────────────────────────────────────────────────

with tab_forecast:
    st.subheader(f"Прогноз на наступні 7 днів — {region}")
    fc = forecast_next(region, horizon=7)
    tail = daily.tail(60).rename(columns={"alert_count": "y"})

    hist_chart = alt.Chart(tail).mark_line(color="#222").encode(
        x=alt.X("date:T", title="Дата"),
        y=alt.Y("y:Q", title="Тривог/день"),
        tooltip=["date:T", "y:Q"],
    )
    fc_line = alt.Chart(fc).mark_line(color="#06c", size=2).encode(
        x="date:T",
        y="yhat:Q",
        tooltip=["date:T", alt.Tooltip("yhat:Q", format=".2f"),
                 alt.Tooltip("yhat_lower:Q", format=".2f"),
                 alt.Tooltip("yhat_upper:Q", format=".2f")],
    )
    fc_band = alt.Chart(fc).mark_area(color="#06c", opacity=0.2).encode(
        x="date:T",
        y=alt.Y("yhat_lower:Q", title=""),
        y2="yhat_upper:Q",
    )
    st.altair_chart(hist_chart + fc_band + fc_line, use_container_width=True)
    st.caption("Чорна — фактична історія (останні 60 днів). Синя — Prophet прогноз. Смуга — діапазон невизначеності 80%.")

    st.subheader("Таблиця прогнозу")
    weekday_uk = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
    fc_table = fc.assign(
        день=fc["date"].dt.dayofweek.map(lambda i: weekday_uk[i]),
        дата=fc["date"].dt.date,
        прогноз=fc["yhat"].round(1),
        мін=fc["yhat_lower"].round(1),
        макс=fc["yhat_upper"].round(1),
    )[["день", "дата", "прогноз", "мін", "макс"]]
    st.dataframe(fc_table, use_container_width=True, hide_index=True)


# ─── TAB 3 — schedule ─────────────────────────────────────────────────────

with tab_schedule:
    st.subheader(f"Оптимальний графік роботи на наступні 7 днів — {region}")
    fc = forecast_next(region, horizon=7)
    try:
        schedule = build_schedule(fc, df_region, window_hours=int(window_hours), curfew=curfew)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.markdown(
        f"Параметри: блок **{window_hours} год** (8 роб + 1 обід), "
        f"комендантська **{curfew.start_hour:02d}:00–{curfew.end_hour:02d}:00**."
    )

    sched_display = schedule.rename(columns={
        "weekday": "День",
        "date": "Дата",
        "yhat": "Прогноз тривог",
        "start": "Початок",
        "end": "Кінець",
        "alerts_in_block": "Очік. тривог у блоці",
        "alerts_outside_block": "Очік. тривог поза блоком",
        "vs_naive_pct": "Виграш vs середній графік, %",
    })
    st.dataframe(sched_display, use_container_width=True, hide_index=True)

    st.subheader("Чому ці години — погодинний ризик з виділеним блоком")
    dist = hourly_distribution(df_region)
    plots = []
    for _, row in schedule.iterrows():
        hourly = dist * row["yhat"]
        start_h = int(row["start"].split(":")[0])
        end_h = int(row["end"].split(":")[0])
        df_h = pd.DataFrame({
            "hour": np.arange(24),
            "expected_alerts": hourly,
            "in_block": [(start_h <= h < end_h) for h in range(24)],
            "in_curfew": [(curfew.start_hour <= h < curfew.end_hour) for h in range(24)],
        })
        df_h["bucket"] = np.where(
            df_h["in_curfew"], "🌙 комендантська",
            np.where(df_h["in_block"], "👔 робочий блок", "поза блоком"),
        )
        chart = alt.Chart(df_h).mark_bar().encode(
            x=alt.X("hour:O", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("expected_alerts:Q", title="Очікувано тривог"),
            color=alt.Color("bucket:N", scale=alt.Scale(
                domain=["🌙 комендантська", "👔 робочий блок", "поза блоком"],
                range=["#222", "#2a7", "#bbb"],
            ), legend=alt.Legend(orient="bottom")),
            tooltip=["hour:O", alt.Tooltip("expected_alerts:Q", format=".3f"), "bucket:N"],
        ).properties(
            height=140,
            title=f"{row['weekday']} {row['date']} → {row['start']}–{row['end']}  "
                  f"(очік. {row['alerts_in_block']:.2f} у блоці)",
        )
        plots.append(chart)

    st.altair_chart(alt.vconcat(*plots).resolve_scale(y="shared"), use_container_width=True)

    st.divider()
    csv_bytes = schedule.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Завантажити графік у CSV",
        data=csv_bytes,
        file_name=f"work_schedule__{region.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

    with st.expander("Як читати ці числа"):
        st.markdown(
            """
- **Прогноз тривог** — Prophet'ове очікування тривог на цілий день.
- **Очік. тривог у блоці** — скільки тривог потрапить у 9-год робочий блок, якщо обрати запропоновані години (помножений на історичний погодинний розподіл).
- **Виграш vs середній графік** — наскільки менше тривог у обраному вікні порівняно з якщо б ти **рівномірно** взяв 9/24 від денного прогнозу. Тобто це ефект оптимізації, не моделі.
- **Уважно:** це СЕРЕДНЄ очікування. У конкретний день може бути сильно більше або менше. Алгоритм мінімізує **очікувану** експозицію, не гарантує тиху зміну.
            """
        )
