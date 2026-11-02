# Impact of Temperature and Rainfall on Crop Yield - Streamlit UI
#
# Run in the VS Code terminal (keep the CSV in the same folder as this file):
#   python -m pip install -r requirements.txt
#   streamlit run app.py
#
# Modelling: ARIMA only (statsmodels). Everything else in the app is
# descriptive (averages, bands, charts) built with NumPy / Pandas / Altair.

import os
import warnings

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")  # hide statsmodels convergence messages

# =====================================================================
# SETTINGS
# =====================================================================
CSV_NAMES = [
    "revised climate_change_impact_on_agriculture_2024.csv",
    "revised_climate_change_impact_on_agriculture_2024.csv",
    "climate_change_impact_on_agriculture_2024.csv",
]

KEEP_COLUMNS = [
    "Year", "Country", "Crop_Type",
    "Average_Temperature_C", "Total_Precipitation_mm", "Crop_Yield_MT_per_HA",
]
NUMBER_COLUMNS = ["Average_Temperature_C", "Total_Precipitation_mm", "Crop_Yield_MT_per_HA"]

LABELS = {
    "Crop_Yield_MT_per_HA": ("Crop Yield", "MT/ha"),
    "Average_Temperature_C": ("Average Temperature", "°C"),
    "Total_Precipitation_mm": ("Total Rainfall", "mm"),
}

BAND_ORDER = ["Below 0", "0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35"]
ZONE_ORDER = ["Cold", "Optimal", "Hot"]
RAIN_ORDER = ["200-900", "900-1600", "1600-2300", "2300-3000"]
ZONE_COLORS = {"Cold": "#2a78d6", "Optimal": "#1baf7a", "Hot": "#eb6834"}

GREEN, ORANGE, GREY = "#1b8a5a", "#eb6834", "#8a8f98"

MIN_YEARS_FOR_ARIMA = 10

st.set_page_config(page_title="Crop Yield & Climate", page_icon="🌱", layout="wide")


# =====================================================================
# PART 1: LOAD AND CLEAN (NumPy / Pandas)
# =====================================================================
def clean(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw.columns = raw.columns.str.strip()

    missing = [c for c in KEEP_COLUMNS if c not in raw.columns]
    if missing:
        raise KeyError("Missing column(s) in the CSV: " + ", ".join(missing))

    data = raw.drop_duplicates()[KEEP_COLUMNS].copy()

    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    for col in NUMBER_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].mean())

    data = data.dropna(subset=["Year", "Country", "Crop_Type"])
    data["Year"] = data["Year"].astype(int)
    data["Country"] = data["Country"].astype(str).str.strip()
    data["Crop_Type"] = data["Crop_Type"].astype(str).str.strip()

    temp = data["Average_Temperature_C"]
    rain = data["Total_Precipitation_mm"]

    data["Temp_Band"] = np.select(
        [temp < 0, temp < 5, temp < 10, temp < 15, temp < 20, temp < 25, temp < 30],
        BAND_ORDER[:-1], default=BAND_ORDER[-1],
    )
    data["Temp_Zone"] = np.select([temp < 10, temp < 20], ["Cold", "Optimal"], default="Hot")
    data["Rain_Band"] = np.select(
        [rain < 900, rain < 1600, rain < 2300], RAIN_ORDER[:-1], default=RAIN_ORDER[-1]
    )
    return data.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_from_disk():
    here = os.path.dirname(os.path.abspath(__file__))
    for name in CSV_NAMES:
        for folder in (here, os.getcwd()):
            path = os.path.join(folder, name)
            if os.path.exists(path):
                raw = pd.read_csv(path)
                return raw, clean(raw), name
    return None, None, None


@st.cache_data(show_spinner=False)
def load_from_upload(file_bytes: bytes):
    import io
    raw = pd.read_csv(io.BytesIO(file_bytes))
    return raw, clean(raw)


# =====================================================================
# PART 3: ARIMA (statsmodels)
# =====================================================================
def to_yearly_series(years, values) -> pd.Series:
    """Build a regular yearly series; any missing years are filled by interpolation."""
    s = pd.Series(values, index=pd.Index(years, dtype=int)).astype(float).sort_index()
    full = range(s.index.min(), s.index.max() + 1)
    s = s.reindex(full).interpolate().bfill().ffill()
    s.index = pd.PeriodIndex(pd.Index(list(full), dtype=int), freq="Y")
    return s


def best_arima(series: pd.Series):
    """Try ARIMA(p,d,q) for p,q in 0..3 and d in 0..1; keep the lowest AIC."""
    best = (np.inf, None, None)
    for d in (0, 1):
        for p in range(4):
            for q in range(4):
                try:
                    fit = ARIMA(series, order=(p, d, q)).fit()
                except Exception:
                    continue
                if np.isfinite(fit.aic) and fit.aic < best[0]:
                    best = (fit.aic, fit, (p, d, q))
    return best[1], best[2]


@st.cache_data(show_spinner=False)
def run_arima(years: tuple, values: tuple, horizon: int, test_years: int):
    series = to_yearly_series(years, values)

    fit, order = best_arima(series)
    if fit is None:
        return None

    fc = fit.get_forecast(steps=horizon)
    ci = fc.conf_int(alpha=0.05)
    forecast = pd.DataFrame({
        "Year": fc.predicted_mean.index.year.astype(int),
        "Forecast": fc.predicted_mean.to_numpy(),
        "Lower": ci.iloc[:, 0].to_numpy(),
        "Upper": ci.iloc[:, 1].to_numpy(),
    })

    # In-sample one-step-ahead fit (skip the first point, which has no history)
    fitted = pd.DataFrame({
        "Year": series.index.year.astype(int),
        "Fitted": fit.fittedvalues.to_numpy(),
    }).iloc[1:]

    # Hold-out check: refit the same order on earlier years, predict the last few
    validation = None
    if test_years and len(series) - test_years >= MIN_YEARS_FOR_ARIMA - 2:
        train, test = series.iloc[:-test_years], series.iloc[-test_years:]
        try:
            vfit = ARIMA(train, order=order).fit()
            pred = vfit.get_forecast(steps=test_years).predicted_mean.to_numpy()
            actual = test.to_numpy()
            err = actual - pred
            validation = {
                "table": pd.DataFrame({
                    "Year": test.index.year.astype(int),
                    "Actual": actual,
                    "ARIMA prediction": pred,
                    "Error": err,
                }),
                "MAE": float(np.mean(np.abs(err))),
                "RMSE": float(np.sqrt(np.mean(err ** 2))),
                "MAPE": float(np.mean(np.abs(err / actual)) * 100) if np.all(actual != 0) else np.nan,
            }
        except Exception:
            validation = None

    return {
        "order": order,
        "aic": float(fit.aic),
        "history": pd.DataFrame({"Year": series.index.year.astype(int), "Actual": series.to_numpy()}),
        "fitted": fitted,
        "forecast": forecast,
        "validation": validation,
    }


# =====================================================================
# CHART HELPERS (Altair)
# =====================================================================
def trend_chart(yearly, col, color):
    label, unit = LABELS[col]
    return (
        alt.Chart(yearly)
        .mark_line(point=True, color=color)
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y(f"{col}:Q", title=unit, scale=alt.Scale(zero=False)),
            tooltip=["Year", alt.Tooltip(f"{col}:Q", title=label, format=".2f")],
        )
        .properties(title=f"Average {label} by Year" if not label.startswith("Average") else f"{label} by Year",
                    height=260)
    )


def bar_chart(table, x, order, title, x_title, highlight=None):
    color = alt.value("#e2a33a")
    if highlight is not None:
        color = alt.condition(alt.datum[x] == highlight, alt.value(GREEN), alt.value("#e2a33a"))
    bars = (
        alt.Chart(table)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:N", sort=order, title=x_title, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Avg_Yield:Q", title="Average yield (MT/ha)"),
            color=color,
            tooltip=[x, alt.Tooltip("Avg_Yield:Q", format=".2f"), "Records"],
        )
    )
    text = bars.mark_text(dy=-8).encode(text=alt.Text("Avg_Yield:Q", format=".2f"), color=alt.value("#444"))
    return (bars + text).properties(title=title, height=320)


def forecast_chart(res, col):
    label, unit = LABELS[col]
    hist, fitted, fc = res["history"], res["fitted"], res["forecast"]

    bridge = pd.concat([
        hist.iloc[[-1]].rename(columns={"Actual": "Forecast"}),
        fc[["Year", "Forecast"]],
    ])

    x = alt.X("Year:O", title="Year")
    y_scale = alt.Scale(zero=False)

    band = alt.Chart(fc).mark_area(opacity=0.18, color=ORANGE).encode(
        x=x, y=alt.Y("Lower:Q", title=unit, scale=y_scale), y2="Upper:Q",
        tooltip=["Year", alt.Tooltip("Lower:Q", format=".2f"), alt.Tooltip("Upper:Q", format=".2f")],
    )
    actual = alt.Chart(hist).mark_line(point=True, color=GREEN).encode(
        x=x, y=alt.Y("Actual:Q", scale=y_scale),
        tooltip=["Year", alt.Tooltip("Actual:Q", format=".2f")],
    )
    fit_line = alt.Chart(fitted).mark_line(color=GREY, strokeDash=[4, 3]).encode(
        x=x, y=alt.Y("Fitted:Q", scale=y_scale),
        tooltip=["Year", alt.Tooltip("Fitted:Q", format=".2f")],
    )
    fc_line = alt.Chart(bridge).mark_line(point=True, color=ORANGE, strokeDash=[6, 4]).encode(
        x=x, y=alt.Y("Forecast:Q", scale=y_scale),
        tooltip=["Year", alt.Tooltip("Forecast:Q", format=".2f")],
    )
    return (band + fit_line + actual + fc_line).properties(
        title=f"{label}: actual, ARIMA fit and forecast", height=340
    )


def yield_table(df, by, order):
    t = df.groupby(by)["Crop_Yield_MT_per_HA"].agg(Avg_Yield="mean", Records="count").reset_index()
    t = t.set_index(by).reindex([o for o in order if o in t[by].values]).reset_index()
    t["Avg_Yield"] = t["Avg_Yield"].round(2)
    return t


# =====================================================================
# DATA
# =====================================================================
st.title("🌱 Impact of Temperature and Rainfall on Crop Yield")
st.caption("10 countries · 10 crop types · 1990–2024 · forecasting with ARIMA")

try:
    raw, df, source = load_from_disk()
except KeyError as e:
    st.error(str(e))
    st.stop()

if df is None:
    st.info(
        "Couldn't find `revised climate_change_impact_on_agriculture_2024.csv` "
        "next to app.py. Upload it here instead."
    )
    upload = st.file_uploader("Upload the CSV", type="csv")
    if upload is None:
        st.stop()
    try:
        raw, df = load_from_upload(upload.getvalue())
        source = upload.name
    except KeyError as e:
        st.error(str(e))
        st.stop()

# =====================================================================
# SIDEBAR FILTERS
# =====================================================================
with st.sidebar:
    st.header("Filters")
    all_countries = sorted(df["Country"].unique())
    all_crops = sorted(df["Crop_Type"].unique())
    y_min, y_max = int(df["Year"].min()), int(df["Year"].max())

    sel_countries = st.multiselect("Countries", all_countries, placeholder="All countries")
    sel_crops = st.multiselect("Crop types", all_crops, placeholder="All crops")
    sel_years = st.slider("Years", y_min, y_max, (y_min, y_max))
    st.caption("Leave a list empty to include everything.")

view = df[df["Year"].between(*sel_years)]
if sel_countries:
    view = view[view["Country"].isin(sel_countries)]
if sel_crops:
    view = view[view["Crop_Type"].isin(sel_crops)]

scope = (
    f"{', '.join(sel_countries) if sel_countries else 'All countries'} · "
    f"{', '.join(sel_crops) if sel_crops else 'All crops'} · {sel_years[0]}–{sel_years[1]}"
)

if view.empty:
    st.warning("No records match these filters. Widen the selection in the sidebar.")
    st.stop()

yearly = view.groupby("Year", as_index=False)[NUMBER_COLUMNS].mean().sort_values("Year")

# =====================================================================
# TABS
# =====================================================================
tab_overview, tab_temp, tab_rain, tab_compare, tab_arima, tab_data = st.tabs([
    "📊 Overview", "🌡️ Temperature", "🌧️ Rainfall",
    "🌾 Crops & Countries", "📈 ARIMA Forecast", "🗂️ Data",
])

# ---------------- OVERVIEW ----------------
with tab_overview:
    st.caption(scope)
    m = st.columns(4)
    m[0].metric("Records", f"{len(view):,}")
    m[1].metric("Avg yield", f"{view['Crop_Yield_MT_per_HA'].mean():.2f} MT/ha")
    m[2].metric("Avg temperature", f"{view['Average_Temperature_C'].mean():.1f} °C")
    m[3].metric("Avg rainfall", f"{view['Total_Precipitation_mm'].mean():,.0f} mm")

    c1, c2 = st.columns(2)
    c1.altair_chart(trend_chart(yearly, "Average_Temperature_C", "#d64545"))
    c2.altair_chart(trend_chart(yearly, "Crop_Yield_MT_per_HA", GREEN))

    hist = (
        alt.Chart(view)
        .mark_bar(color="#2a78d6", opacity=0.8)
        .encode(
            x=alt.X("Crop_Yield_MT_per_HA:Q", bin=alt.Bin(maxbins=30), title="Crop yield (MT/ha)"),
            y=alt.Y("count():Q", title="Number of records"),
        )
        .properties(title="Distribution of Crop Yield", height=260)
    )
    st.altair_chart(hist)

# ---------------- TEMPERATURE ----------------
with tab_temp:
    st.caption(scope)
    band_t = yield_table(view, "Temp_Band", BAND_ORDER)
    zone_t = yield_table(view, "Temp_Zone", ZONE_ORDER)

    best = band_t.loc[band_t["Avg_Yield"].idxmax()]
    best_idx = BAND_ORDER.index(best["Temp_Band"])
    above = band_t[band_t["Temp_Band"].map(BAND_ORDER.index) > best_idx]

    if not above.empty:
        threshold = best["Temp_Band"].split("-")[-1] if "-" in best["Temp_Band"] else "0"
        drop = (1 - above["Avg_Yield"].iloc[0] / best["Avg_Yield"]) * 100
        st.success(
            f"**Yield peaks in the {best['Temp_Band']} °C band at {best['Avg_Yield']:.2f} MT/ha.** "
            f"Beyond **{threshold} °C** it declines — the next band "
            f"({above['Temp_Band'].iloc[0]} °C) is {drop:.0f}% lower."
        )
    else:
        st.info(f"Yield is highest in the warmest band in this selection ({best['Temp_Band']} °C).")

    c1, c2 = st.columns([3, 2])
    c1.altair_chart(bar_chart(band_t, "Temp_Band", BAND_ORDER,
                              "Average Yield by 5 °C Temperature Band", "Temperature band (°C)",
                              highlight=best["Temp_Band"]))
    zone_chart = (
        alt.Chart(zone_t)
        .mark_bar()
        .encode(
            x=alt.X("Temp_Zone:N", sort=ZONE_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Avg_Yield:Q", title="Average yield (MT/ha)"),
            color=alt.Color("Temp_Zone:N", scale=alt.Scale(domain=list(ZONE_COLORS),
                                                           range=list(ZONE_COLORS.values())), legend=None),
            tooltip=["Temp_Zone", alt.Tooltip("Avg_Yield:Q", format=".2f"), "Records"],
        )
    )
    c2.altair_chart(
        (zone_chart + zone_chart.mark_text(dy=-8, color="#444").encode(text=alt.Text("Avg_Yield:Q", format=".2f")))
        .properties(title="Cold (<10) · Optimal (10–20) · Hot (≥20 °C)", height=320)
    )

    sample = view.sample(min(len(view), 3000), random_state=1)
    scatter = (
        alt.Chart(sample)
        .mark_circle(size=14, opacity=0.45)
        .encode(
            x=alt.X("Average_Temperature_C:Q", title="Average temperature (°C)"),
            y=alt.Y("Crop_Yield_MT_per_HA:Q", title="Crop yield (MT/ha)"),
            color=alt.Color("Temp_Zone:N", sort=ZONE_ORDER, title="Zone",
                            scale=alt.Scale(domain=list(ZONE_COLORS), range=list(ZONE_COLORS.values()))),
            tooltip=["Country", "Crop_Type", "Year",
                     alt.Tooltip("Average_Temperature_C:Q", format=".1f"),
                     alt.Tooltip("Crop_Yield_MT_per_HA:Q", format=".2f")],
        )
        .properties(title=f"Temperature vs Crop Yield ({len(sample):,} records shown)", height=360)
    )
    st.altair_chart(scatter)

# ---------------- RAINFALL ----------------
with tab_rain:
    st.caption(scope)
    rain_t = yield_table(view, "Rain_Band", RAIN_ORDER)
    spread = rain_t["Avg_Yield"].max() - rain_t["Avg_Yield"].min()
    temp_spread = band_t["Avg_Yield"].max() - band_t["Avg_Yield"].min()
    st.info(
        f"Average yield varies by **{spread:.2f} MT/ha** across rainfall bands, "
        f"compared with **{temp_spread:.2f} MT/ha** across temperature bands."
    )
    st.altair_chart(bar_chart(rain_t, "Rain_Band", RAIN_ORDER,
                              "Average Yield by Rainfall Band", "Total precipitation (mm)"))

    sample = view.sample(min(len(view), 3000), random_state=2)
    st.altair_chart(
        alt.Chart(sample)
        .mark_circle(size=14, opacity=0.4, color="#2a78d6")
        .encode(
            x=alt.X("Total_Precipitation_mm:Q", title="Total precipitation (mm)"),
            y=alt.Y("Crop_Yield_MT_per_HA:Q", title="Crop yield (MT/ha)"),
            tooltip=["Country", "Crop_Type", "Year",
                     alt.Tooltip("Total_Precipitation_mm:Q", format=",.0f"),
                     alt.Tooltip("Crop_Yield_MT_per_HA:Q", format=".2f")],
        )
        .properties(title=f"Rainfall vs Crop Yield ({len(sample):,} records shown)", height=340)
    )

# ---------------- CROPS & COUNTRIES ----------------
with tab_compare:
    st.caption(scope)
    heat = (
        view.groupby(["Crop_Type", "Temp_Band"])["Crop_Yield_MT_per_HA"].mean().round(2).reset_index()
    )
    base = alt.Chart(heat).encode(
        x=alt.X("Temp_Band:N", sort=BAND_ORDER, title="Temperature band (°C)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Crop_Type:N", title=None),
    )
    st.altair_chart(
        (base.mark_rect().encode(
            color=alt.Color("Crop_Yield_MT_per_HA:Q", scale=alt.Scale(scheme="greens"), title="MT/ha"),
            tooltip=["Crop_Type", "Temp_Band", alt.Tooltip("Crop_Yield_MT_per_HA:Q", format=".2f")],
        ) + base.mark_text(fontSize=11).encode(
            text=alt.Text("Crop_Yield_MT_per_HA:Q", format=".2f"),
            color=alt.condition(alt.datum.Crop_Yield_MT_per_HA > heat["Crop_Yield_MT_per_HA"].median(),
                                alt.value("white"), alt.value("#222")),
        )).properties(title="Average Yield of Each Crop by Temperature Band", height=380)
    )

    country_t = (
        view.groupby("Country")["Crop_Yield_MT_per_HA"].mean().round(2)
        .sort_values(ascending=False).reset_index()
    )
    st.altair_chart(
        alt.Chart(country_t)
        .mark_bar(color="seagreen")
        .encode(
            x=alt.X("Crop_Yield_MT_per_HA:Q", title="Average yield (MT/ha)"),
            y=alt.Y("Country:N", sort="-x", title=None),
            tooltip=["Country", alt.Tooltip("Crop_Yield_MT_per_HA:Q", format=".2f")],
        )
        .properties(title="Average Crop Yield by Country", height=320)
    )

# ---------------- ARIMA ----------------
with tab_arima:
    st.caption(scope)
    st.write(
        "Each variable is averaged per year for the current filters, then an **ARIMA(p, d, q)** "
        "model is selected automatically (lowest AIC, p and q from 0–3, d from 0–1) and used to forecast."
    )

    c1, c2, c3 = st.columns(3)
    target = c1.selectbox(
        "Variable to forecast", list(LABELS),
        format_func=lambda c: f"{LABELS[c][0]} ({LABELS[c][1]})",
    )
    horizon = c2.slider("Years to forecast", 1, 10, 5)
    test_years = c3.slider("Hold-out years for validation", 0, 8, 5,
                           help="The model is refitted without these last years and scored on them.")

    n_years = yearly["Year"].nunique()
    if n_years < MIN_YEARS_FOR_ARIMA:
        st.warning(
            f"ARIMA needs at least {MIN_YEARS_FOR_ARIMA} years of data; the current filters give {n_years}. "
            "Widen the year range in the sidebar."
        )
    else:
        with st.spinner("Selecting and fitting the ARIMA model…"):
            res = run_arima(tuple(yearly["Year"]), tuple(yearly[target]), horizon, test_years)

        if res is None:
            st.error("No ARIMA model could be fitted to this series.")
        else:
            label, unit = LABELS[target]
            fc = res["forecast"]
            last_actual = res["history"]["Actual"].iloc[-1]

            k = st.columns(4)
            k[0].metric("Model", f"ARIMA{res['order']}")
            k[1].metric("AIC", f"{res['aic']:.1f}")
            k[2].metric(f"Last actual ({res['history']['Year'].iloc[-1]})", f"{last_actual:.2f} {unit}")
            k[3].metric(
                f"Forecast {fc['Year'].iloc[-1]}", f"{fc['Forecast'].iloc[-1]:.2f} {unit}",
                delta=f"{fc['Forecast'].iloc[-1] - last_actual:+.2f}",
            )

            st.altair_chart(forecast_chart(res, target))
            st.caption("Green = actual yearly average · grey dashed = ARIMA in-sample fit · "
                       "orange = forecast with 95% confidence band.")

            left, right = st.columns(2)
            with left:
                st.subheader("Forecast")
                st.dataframe(fc.round(2), hide_index=True, width="stretch")
                st.download_button(
                    "⬇️ Download forecast (CSV)",
                    fc.round(3).to_csv(index=False).encode("utf-8"),
                    file_name=f"arima_forecast_{target}.csv", mime="text/csv",
                )
            with right:
                st.subheader("Validation on hold-out years")
                v = res["validation"]
                if v is None:
                    st.info("Set at least one hold-out year (with enough data left to train on) to see validation.")
                else:
                    s = st.columns(3)
                    s[0].metric("MAE", f"{v['MAE']:.3f}")
                    s[1].metric("RMSE", f"{v['RMSE']:.3f}")
                    s[2].metric("MAPE", f"{v['MAPE']:.1f}%" if np.isfinite(v["MAPE"]) else "n/a")
                    st.dataframe(v["table"].round(3), hide_index=True, width="stretch")

            if res["order"][0] == 0 and res["order"][2] == 0:
                st.info(
                    f"The best model has no AR or MA terms, so past years don't help predict the next one: "
                    f"the forecast is essentially flat at the series {'mean' if res['order'][1] == 0 else 'last value'}."
                )

# ---------------- DATA ----------------
with tab_data:
    st.subheader("Data cleaning summary")
    d = st.columns(4)
    d[0].metric("Rows in file", f"{len(raw):,}")
    d[1].metric("Duplicate rows removed", f"{len(raw) - len(raw.drop_duplicates()):,}")
    d[2].metric("Missing values filled", f"{int(raw[NUMBER_COLUMNS].isnull().sum().sum()):,}")
    d[3].metric("Rows used", f"{len(df):,}")
    st.caption(f"Source: {source}. Unused columns (Region, CO₂, irrigation, fertiliser, etc.) are dropped "
               "as they are outside the problem statement.")

    st.subheader("Filtered records")
    st.dataframe(view[KEEP_COLUMNS + ["Temp_Band", "Temp_Zone", "Rain_Band"]],
                 hide_index=True, width="stretch")
    st.download_button(
        "⬇️ Download filtered data (CSV)",
        view.to_csv(index=False).encode("utf-8"),
        file_name="filtered_crop_data.csv", mime="text/csv",
    )

st.divider()
st.caption("Note: this dataset appears to be synthetic, so results describe patterns in this data, "
           "not real-world evidence. Forecasts are statistical projections, not guarantees.")
