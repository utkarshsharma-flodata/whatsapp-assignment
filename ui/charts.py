"""Altair charts for the admin dashboard.

Colours come from the validated reference palette (dataviz skill): single
series use categorical slot 1 (blue); the one two-series chart uses slots 1-2
(blue/orange), validated for colour-vision deficiency in light and dark mode.
Identity is never carried by colour alone: bars are labelled on the axis,
two-series charts have a legend, and every chart has a data table view.
"""

import altair as alt
import pandas as pd
import streamlit as st

_SERIES = {"light": ("#2a78d6", "#eb6834"), "dark": ("#3987e5", "#d95926")}
_BAR_SIZE = 18


def _palette() -> tuple[str, str]:
    theme = getattr(st.context, "theme", None)
    return _SERIES["dark" if theme is not None and theme.type == "dark" else "light"]


def _data_view(df: pd.DataFrame, key: str) -> None:
    with st.expander("View data"):
        st.dataframe(df, hide_index=True, width="stretch", key=f"data_{key}")


def _empty(message: str = "No data yet.") -> None:
    st.caption(f"📭 {message}")


def horizontal_bars(rows: list[dict], label: str, value: str, title: str, key: str,
                    sort: list[str] | None = None, value_title: str = "Leads") -> None:
    """Single-series horizontal bar chart (categories on the y axis, direct value labels)."""
    st.markdown(f"**{title}**")
    df = pd.DataFrame(rows)
    if df.empty or df[value].fillna(0).sum() == 0:
        _empty()
        return
    color, _ = _palette()
    order = sort or df.sort_values(value, ascending=False)[label].tolist()
    y = alt.Y(f"{label}:N", sort=order, title=None, axis=alt.Axis(labelLimit=180))
    base = alt.Chart(df).encode(
        y=y,
        x=alt.X(f"{value}:Q", title=value_title, axis=alt.Axis(tickMinStep=1, format="d")),
        tooltip=[alt.Tooltip(f"{label}:N", title=label.replace("_", " ").title()),
                 alt.Tooltip(f"{value}:Q", title=value_title)],
    )
    bars = base.mark_bar(color=color, cornerRadiusEnd=4)
    labels = base.mark_text(align="left", dx=4).encode(text=f"{value}:Q")
    # A fixed step per category keeps every bar and label readable however many categories there are.
    chart = (bars + labels).properties(height=alt.Step(28)).configure_scale(bandPaddingInner=0.35)
    st.altair_chart(chart, width="stretch")
    _data_view(df, key)


def leads_over_time(rows: list[dict], key: str) -> None:
    """New leads per day (per week when the range is longer than 60 days)."""
    st.markdown("**Leads by date**")
    df = pd.DataFrame(rows)
    if df.empty:
        _empty()
        return
    df["day"] = pd.to_datetime(df["day"])
    span = (df["day"].max() - df["day"].min()).days
    # Fill empty days/weeks with 0 so gaps are visible, then plot periods as ordered categories.
    freq, label, fmt = ("D", "Day", "%d %b") if span <= 60 else ("W-MON", "Week of", "%d %b %Y")
    df = df.set_index("day").resample(freq, label="left", closed="left")["leads"].sum().reset_index()
    df["period"] = df["day"].dt.strftime(fmt)
    color, _ = _palette()
    chart = alt.Chart(df).mark_bar(color=color, cornerRadiusEnd=4, size=_BAR_SIZE).encode(
        x=alt.X("period:O", sort=df["period"].tolist(), title=label, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("leads:Q", title="New leads", axis=alt.Axis(tickMinStep=1, format="d")),
        tooltip=[alt.Tooltip("period:O", title=label), alt.Tooltip("leads:Q", title="New leads")],
    )
    st.altair_chart(chart.properties(height=220), width="stretch")
    _data_view(df[["period", "leads"]].rename(columns={"period": label, "leads": "New leads"}), key)


def conversion_trend(rows: list[dict], key: str) -> None:
    """Converted vs lost leads per day they were closed (two series, legend + tooltip)."""
    st.markdown("**Conversion trend** (leads closed per day)")
    df = pd.DataFrame(rows)
    if df.empty:
        _empty("No leads have been marked Converted or Lost yet.")
        return
    long = df.melt(id_vars="day", value_vars=["converted", "lost"], var_name="outcome", value_name="leads")
    long["outcome"] = long["outcome"].str.capitalize()
    long["day"] = pd.to_datetime(long["day"])
    converted, lost = _palette()
    chart = alt.Chart(long).mark_bar(cornerRadiusEnd=4, size=12).encode(
        x=alt.X("yearmonthdate(day):T", title="Day closed"),
        xOffset=alt.XOffset("outcome:N", sort=["Converted", "Lost"]),
        y=alt.Y("leads:Q", title="Leads", axis=alt.Axis(tickMinStep=1, format="d")),
        color=alt.Color("outcome:N", sort=["Converted", "Lost"],
                        scale=alt.Scale(domain=["Converted", "Lost"], range=[converted, lost]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("day:T", title="Day", format="%d %b %Y"), alt.Tooltip("outcome:N", title="Outcome"),
                 alt.Tooltip("leads:Q", title="Leads")],
    )
    st.altair_chart(chart.properties(height=220), width="stretch")
    _data_view(df, key)


def score_history(rows: list[dict], key: str) -> None:
    """Total score over time for one lead (single series line, 2px, markers)."""
    df = pd.DataFrame(rows)
    if df.empty:
        _empty("No score recorded yet.")
        return
    df = df.assign(scored_at=pd.to_datetime(df["scored_at"]))
    color, _ = _palette()
    base = alt.Chart(df).encode(
        x=alt.X("scored_at:T", title="Scored at"),
        y=alt.Y("total_score:Q", title="Score", scale=alt.Scale(domain=[0, 100])),
        tooltip=[alt.Tooltip("scored_at:T", title="Scored at", format="%d %b %Y %H:%M"),
                 alt.Tooltip("total_score:Q", title="Score"), alt.Tooltip("category:N", title="Category")],
    )
    chart = base.mark_line(color=color, strokeWidth=2) + base.mark_point(color=color, filled=True, size=70)
    st.altair_chart(chart.properties(height=200), width="stretch")
