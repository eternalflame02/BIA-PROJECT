from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


st.set_page_config(
    page_title="Restaurant Sales & Operations Dashboard",
    page_icon="🍽️",
    layout="wide",
)


DATA_PATH = Path(__file__).with_name("restaurant_data.csv")

DATE_COLUMN = "Date"
TIMESTAMP_COLUMN = "Timestamp"
INVOICE_COLUMN = "Invoice No."
ITEM_COLUMN = "Item Name"
PRICE_COLUMN = "Price"
QTY_COLUMN = "Qty."
SUBTOTAL_COLUMN = "Sub Total"
DISCOUNT_COLUMN = "Discount"
TAX_COLUMN = "Tax"
FINAL_TOTAL_COLUMN = "Final Total"
TABLE_COLUMN = "Table No."
CATEGORY_COLUMN = "Category"

DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

EXCLUDED_COLUMNS = {
    "HSN": "excluded: no information",
    "Covers": "excluded: no information",
    "Variation": "excluded: no information",
    "Server Name": "excluded: no information",
}

PALETTE = ["#C96C3C", "#2E7D7A", "#D9A441", "#8A6A3E", "#6B7C93"]


def format_inr(value: float | int | None) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except TypeError:
        pass

    rounded = int(round(float(value)))
    sign = "-" if rounded < 0 else ""
    digits = str(abs(rounded))
    if len(digits) <= 3:
        return f"₹{sign}{digits}"

    last_three = digits[-3:]
    prefix = digits[:-3]
    groups: list[str] = []
    while prefix:
        groups.append(prefix[-2:])
        prefix = prefix[:-2]
    indian = ",".join(reversed(groups)) + "," + last_three
    return f"₹{sign}{indian}"


def format_percent(value: float | int | None) -> str:
    if value is None:
        return "0%"
    try:
        if pd.isna(value):
            return "0%"
    except TypeError:
        pass
    return f"{float(value):.1f}%"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    try:
        return pd.read_csv(DATA_PATH)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def normalize_category(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return " ".join(text.split()).title()


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        PRICE_COLUMN,
        QTY_COLUMN,
        SUBTOTAL_COLUMN,
        DISCOUNT_COLUMN,
        TAX_COLUMN,
        FINAL_TOTAL_COLUMN,
        TABLE_COLUMN,
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def clean_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    if raw_df is None or raw_df.empty:
        empty_report = {
            "row_count": 0,
            "duplicate_rows": 0,
            "unparseable_date_rows": 0,
            "unparseable_timestamp_rows": 0,
            "start_date": None,
            "end_date": None,
        }
        return raw_df.copy() if isinstance(raw_df, pd.DataFrame) else pd.DataFrame(), empty_report

    df = raw_df.copy()
    df = coerce_numeric_columns(df)

    if DATE_COLUMN in df.columns:
        df["Date Parsed"] = pd.to_datetime(df[DATE_COLUMN], format="%d-%m-%Y", errors="coerce")
    else:
        df["Date Parsed"] = pd.NaT

    if TIMESTAMP_COLUMN in df.columns:
        df["Timestamp Parsed"] = pd.to_datetime(
            df[TIMESTAMP_COLUMN],
            format="%d-%m-%Y %H:%M",
            errors="coerce",
        )
    else:
        df["Timestamp Parsed"] = pd.NaT

    if "Timestamp Parsed" in df.columns:
        df["Date Only"] = df["Timestamp Parsed"].dt.normalize()
        df["Hour"] = df["Timestamp Parsed"].dt.hour
        day_series = df["Timestamp Parsed"].dt.day_name()
        df["Day Number"] = df["Timestamp Parsed"].dt.dayofweek + 1
        df["Day"] = pd.Categorical(day_series, categories=DAY_ORDER, ordered=True)
        month_number = df["Timestamp Parsed"].dt.month
        df["Month"] = month_number
        month_name = df["Timestamp Parsed"].dt.month_name()
        df["Month Name"] = pd.Categorical(month_name, categories=MONTH_ORDER, ordered=True)
    else:
        df["Date Only"] = pd.NaT
        df["Hour"] = pd.NA
        df["Day Number"] = pd.NA
        df["Day"] = pd.Categorical([], categories=DAY_ORDER, ordered=True)
        df["Month"] = pd.NA
        df["Month Name"] = pd.Categorical([], categories=MONTH_ORDER, ordered=True)

    if CATEGORY_COLUMN in df.columns:
        df["Category Display"] = df[CATEGORY_COLUMN].map(normalize_category)
    else:
        df["Category Display"] = None

    if ITEM_COLUMN in df.columns:
        df["Item Display"] = df[ITEM_COLUMN].astype("string").str.strip()
    else:
        df["Item Display"] = None

    unparseable_date_rows = int(df["Date Parsed"].isna().sum())
    unparseable_timestamp_rows = int(df["Timestamp Parsed"].isna().sum())
    duplicate_rows = int(df.duplicated().sum())
    valid_dates = df["Date Parsed"].dropna()

    report = {
        "row_count": int(len(df)),
        "duplicate_rows": duplicate_rows,
        "unparseable_date_rows": unparseable_date_rows,
        "unparseable_timestamp_rows": unparseable_timestamp_rows,
        "start_date": valid_dates.min().date() if not valid_dates.empty else None,
        "end_date": valid_dates.max().date() if not valid_dates.empty else None,
    }
    return df, report


def normalize_date_selection(selection: object) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if selection is None:
        return None, None
    if isinstance(selection, tuple) or isinstance(selection, list):
        if len(selection) == 0:
            return None, None
        if len(selection) == 1:
            start = pd.Timestamp(selection[0])
            return start, start
        start = pd.Timestamp(min(selection[0], selection[1]))
        end = pd.Timestamp(max(selection[0], selection[1]))
        return start, end
    stamp = pd.Timestamp(selection)
    return stamp, stamp


def apply_filters(
    df: pd.DataFrame,
    date_selection: object,
    selected_categories: list[str] | None,
    selected_items: list[str] | None,
    selected_tables: list[float | int] | None,
    remove_duplicates: bool,
) -> pd.DataFrame:
    filtered = df.copy()

    if remove_duplicates:
        filtered = filtered.drop_duplicates().copy()

    start_date, end_date = normalize_date_selection(date_selection)
    if start_date is not None and end_date is not None and "Date Parsed" in filtered.columns:
        date_mask = filtered["Date Parsed"].between(start_date, end_date, inclusive="both")
        filtered = filtered.loc[date_mask].copy()

    if selected_categories is not None and "Category Display" in filtered.columns:
        if len(selected_categories) > 0:
            filtered = filtered[filtered["Category Display"].isin(selected_categories)].copy()

    if selected_items is not None and ITEM_COLUMN in filtered.columns:
        if len(selected_items) > 0:
            filtered = filtered[filtered[ITEM_COLUMN].isin(selected_items)].copy()

    if selected_tables is not None and TABLE_COLUMN in filtered.columns:
        if len(selected_tables) > 0:
            filtered = filtered[filtered[TABLE_COLUMN].isin(selected_tables)].copy()

    return filtered


def calculate_kpis(df: pd.DataFrame) -> dict[str, float]:
    revenue = float(df[FINAL_TOTAL_COLUMN].sum(skipna=True)) if FINAL_TOTAL_COLUMN in df.columns else 0.0
    orders = int(df[INVOICE_COLUMN].nunique(dropna=True)) if INVOICE_COLUMN in df.columns else 0
    items_sold = float(df[QTY_COLUMN].sum(skipna=True)) if QTY_COLUMN in df.columns else 0.0
    total_discount = float(df[DISCOUNT_COLUMN].sum(skipna=True)) if DISCOUNT_COLUMN in df.columns else 0.0
    total_subtotal = float(df[SUBTOTAL_COLUMN].sum(skipna=True)) if SUBTOTAL_COLUMN in df.columns else 0.0
    aov = revenue / orders if orders else 0.0
    discount_pct = (total_discount / total_subtotal * 100.0) if total_subtotal else 0.0
    return {
        "revenue": revenue,
        "orders": float(orders),
        "items_sold": items_sold,
        "aov": aov,
        "total_discount": total_discount,
        "total_subtotal": total_subtotal,
        "discount_pct": discount_pct,
    }


def empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.0, xanchor="left"),
        height=360,
        margin=dict(l=20, r=20, t=70, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color="#5f6b7a"),
    )
    return fig


def style_figure(fig: go.Figure, title: str, height: int = 360, x_title: str | None = None, y_title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.0, xanchor="left"),
        height=height,
        margin=dict(l=20, r=20, t=70, b=20),
        font=dict(family='"Segoe UI", "Trebuchet MS", sans-serif', size=13),
        hovermode="closest",
        showlegend=False,
    )
    if x_title is not None:
        fig.update_xaxes(title_text=x_title, gridcolor="rgba(120, 130, 140, 0.12)", zeroline=False)
    else:
        fig.update_xaxes(gridcolor="rgba(120, 130, 140, 0.12)", zeroline=False)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title, gridcolor="rgba(120, 130, 140, 0.12)", zeroline=False)
    else:
        fig.update_yaxes(gridcolor="rgba(120, 130, 140, 0.12)", zeroline=False)
    return fig


def create_daily_revenue_trend_chart(df: pd.DataFrame) -> go.Figure:
    title = "Daily Revenue Trend"
    if df.empty or DATE_COLUMN not in df.columns or FINAL_TOTAL_COLUMN not in df.columns:
        return empty_figure(title, "No data available for this selection.")

    trend = (
        df.dropna(subset=["Date Parsed"])
        .groupby("Date Parsed", as_index=False)[FINAL_TOTAL_COLUMN]
        .sum()
        .sort_values("Date Parsed")
    )
    if trend.empty:
        return empty_figure(title, "No dated rows are available for this selection.")

    fig = go.Figure(
        data=[
            go.Scatter(
                x=trend["Date Parsed"],
                y=trend[FINAL_TOTAL_COLUMN],
                mode="lines+markers",
                line=dict(color=PALETTE[0], width=3),
                marker=dict(size=7, color=PALETTE[0]),
                customdata=trend[FINAL_TOTAL_COLUMN].map(format_inr),
                hovertemplate="Date: %{x|%d-%b-%Y}<br>Revenue: %{customdata}<extra></extra>",
            )
        ]
    )
    fig.update_xaxes(title_text="Date", tickformat="%d-%b", dtick="D7")
    fig.update_yaxes(title_text="Revenue", tickprefix="₹")
    return style_figure(fig, title)


def create_revenue_by_category_chart(df: pd.DataFrame) -> go.Figure:
    title = "Revenue by Category"
    if df.empty or FINAL_TOTAL_COLUMN not in df.columns or "Category Display" not in df.columns:
        return empty_figure(title, "No data available for this selection.")

    grouped = (
        df.dropna(subset=["Category Display"])
        .groupby("Category Display", as_index=False)[FINAL_TOTAL_COLUMN]
        .sum()
        .sort_values(FINAL_TOTAL_COLUMN, ascending=True)
    )
    if grouped.empty:
        return empty_figure(title, "No category data is available for this selection.")

    fig = go.Figure(
        data=[
            go.Bar(
                x=grouped[FINAL_TOTAL_COLUMN],
                y=grouped["Category Display"],
                orientation="h",
                marker_color=PALETTE[1],
                customdata=grouped[FINAL_TOTAL_COLUMN].map(format_inr),
                hovertemplate="Category: %{y}<br>Revenue: %{customdata}<extra></extra>",
            )
        ]
    )
    fig.update_xaxes(title_text="Revenue", tickprefix="₹")
    fig.update_yaxes(title_text="Category")
    return style_figure(fig, title)


def create_top_items_by_quantity_chart(df: pd.DataFrame) -> go.Figure:
    title = "Top 10 Best-Selling Items"
    if df.empty or QTY_COLUMN not in df.columns or ITEM_COLUMN not in df.columns:
        return empty_figure(title, "No data available for this selection.")

    grouped = (
        df.dropna(subset=[ITEM_COLUMN])
        .groupby(ITEM_COLUMN, as_index=False)[QTY_COLUMN]
        .sum()
        .sort_values(QTY_COLUMN, ascending=False)
        .head(10)
        .sort_values(QTY_COLUMN, ascending=True)
    )
    if grouped.empty:
        return empty_figure(title, "No item-level quantity data is available for this selection.")

    fig = go.Figure(
        data=[
            go.Bar(
                x=grouped[QTY_COLUMN],
                y=grouped[ITEM_COLUMN],
                orientation="h",
                marker_color=PALETTE[2],
                customdata=grouped[QTY_COLUMN].map(lambda value: f"{int(round(value))}"),
                hovertemplate="Item: %{y}<br>Quantity Sold: %{customdata}<extra></extra>",
            )
        ]
    )
    fig.update_xaxes(title_text="Quantity Sold")
    fig.update_yaxes(title_text="Item")
    return style_figure(fig, title, height=380)


def create_top_items_by_qty_chart(df: pd.DataFrame) -> go.Figure:
    return create_top_items_by_quantity_chart(df)


def create_top_items_by_revenue_chart(df: pd.DataFrame) -> go.Figure:
    title = "Top 10 Items by Revenue"
    if df.empty or FINAL_TOTAL_COLUMN not in df.columns or ITEM_COLUMN not in df.columns:
        return empty_figure(title, "No data available for this selection.")

    grouped = (
        df.dropna(subset=[ITEM_COLUMN])
        .groupby(ITEM_COLUMN, as_index=False)[FINAL_TOTAL_COLUMN]
        .sum()
        .sort_values(FINAL_TOTAL_COLUMN, ascending=False)
        .head(10)
        .sort_values(FINAL_TOTAL_COLUMN, ascending=True)
    )
    if grouped.empty:
        return empty_figure(title, "No item-level revenue data is available for this selection.")

    fig = go.Figure(
        data=[
            go.Bar(
                x=grouped[FINAL_TOTAL_COLUMN],
                y=grouped[ITEM_COLUMN],
                orientation="h",
                marker_color=PALETTE[0],
                customdata=grouped[FINAL_TOTAL_COLUMN].map(format_inr),
                hovertemplate="Item: %{y}<br>Revenue: %{customdata}<extra></extra>",
            )
        ]
    )
    fig.update_xaxes(title_text="Revenue", tickprefix="₹")
    fig.update_yaxes(title_text="Item")
    return style_figure(fig, title, height=380)


def create_revenue_by_hour_chart(df: pd.DataFrame) -> go.Figure:
    title = "Revenue by Hour"
    if df.empty or FINAL_TOTAL_COLUMN not in df.columns or "Hour" not in df.columns:
        return empty_figure(title, "No data available for this selection.")

    grouped = (
        df.dropna(subset=["Hour"])
        .groupby("Hour", as_index=False)[FINAL_TOTAL_COLUMN]
        .sum()
        .sort_values("Hour")
    )
    if grouped.empty:
        return empty_figure(title, "No hour data is available for this selection.")

    fig = go.Figure(
        data=[
            go.Bar(
                x=grouped["Hour"].astype(int),
                y=grouped[FINAL_TOTAL_COLUMN],
                marker_color=PALETTE[3],
                customdata=grouped[FINAL_TOTAL_COLUMN].map(format_inr),
                hovertemplate="Hour: %{x}:00<br>Revenue: %{customdata}<extra></extra>",
            )
        ]
    )
    fig.update_xaxes(title_text="Hour of Day")
    fig.update_yaxes(title_text="Revenue", tickprefix="₹")
    return style_figure(fig, title)


def create_orders_by_day_of_week_chart(df: pd.DataFrame) -> go.Figure:
    title = "Orders by Day of Week"
    if df.empty or INVOICE_COLUMN not in df.columns or "Day" not in df.columns:
        return empty_figure(title, "No data available for this selection.")

    grouped = (
        df.dropna(subset=["Day"])
        .groupby("Day", observed=True)[INVOICE_COLUMN]
        .nunique()
        .reindex(DAY_ORDER)
        .dropna()
        .reset_index()
        .rename(columns={INVOICE_COLUMN: "Orders"})
    )
    if grouped.empty:
        return empty_figure(title, "No weekday data is available for this selection.")

    fig = go.Figure(
        data=[
            go.Bar(
                x=grouped["Day"],
                y=grouped["Orders"],
                marker_color=PALETTE[4],
                customdata=grouped["Orders"].map(lambda value: f"{int(value)}"),
                hovertemplate="Day: %{x}<br>Orders: %{customdata}<extra></extra>",
            )
        ]
    )
    fig.update_xaxes(title_text="Day of Week")
    fig.update_yaxes(title_text="Unique Orders")
    return style_figure(fig, title)


def create_orders_by_day_chart(df: pd.DataFrame) -> go.Figure:
    return create_orders_by_day_of_week_chart(df)


def create_quantity_by_category_chart(df: pd.DataFrame) -> go.Figure:
    title = "Quantity Sold by Category"
    if df.empty or QTY_COLUMN not in df.columns or "Category Display" not in df.columns:
        return empty_figure(title, "No data available for this selection.")

    grouped = (
        df.dropna(subset=["Category Display"])
        .groupby("Category Display", as_index=False)[QTY_COLUMN]
        .sum()
        .sort_values(QTY_COLUMN, ascending=True)
    )
    if grouped.empty:
        return empty_figure(title, "No category quantity data is available for this selection.")

    fig = go.Figure(
        data=[
            go.Bar(
                x=grouped[QTY_COLUMN],
                y=grouped["Category Display"],
                orientation="h",
                marker_color=PALETTE[1],
                customdata=grouped[QTY_COLUMN].map(lambda value: f"{int(round(value))}"),
                hovertemplate="Category: %{y}<br>Quantity Sold: %{customdata}<extra></extra>",
            )
        ]
    )
    fig.update_xaxes(title_text="Quantity Sold")
    fig.update_yaxes(title_text="Category")
    return style_figure(fig, title)


def create_sales_mix_chart(df: pd.DataFrame) -> go.Figure:
    title = "Sales Mix by Category"
    if df.empty or FINAL_TOTAL_COLUMN not in df.columns or "Category Display" not in df.columns:
        return empty_figure(title, "No data available for this selection.")

    grouped = (
        df.dropna(subset=["Category Display"])
        .groupby("Category Display", as_index=False)[FINAL_TOTAL_COLUMN]
        .sum()
    )
    if grouped.empty:
        return empty_figure(title, "No category revenue data is available for this selection.")

    total_revenue = grouped[FINAL_TOTAL_COLUMN].sum()
    if total_revenue == 0:
        return empty_figure(title, "Revenue is zero for the current selection.")

    grouped["Share"] = grouped[FINAL_TOTAL_COLUMN] / total_revenue * 100.0
    grouped = grouped.sort_values("Share", ascending=True)

    fig = go.Figure(
        data=[
            go.Bar(
                x=grouped["Share"],
                y=grouped["Category Display"],
                orientation="h",
                marker_color=PALETTE[2],
                customdata=grouped["Share"].map(lambda value: f"{value:.1f}%"),
                hovertemplate="Category: %{y}<br>Share of Revenue: %{customdata}<extra></extra>",
            )
        ]
    )
    fig.update_xaxes(title_text="Share of Revenue (%)")
    fig.update_yaxes(title_text="Category")
    return style_figure(fig, title)


def create_price_vs_quantity_scatter(df: pd.DataFrame) -> go.Figure:
    title = "Price vs Quantity Sold"
    required = {ITEM_COLUMN, PRICE_COLUMN, QTY_COLUMN, FINAL_TOTAL_COLUMN}
    if df.empty or not required.issubset(df.columns):
        return empty_figure(title, "No data available for this selection.")

    grouped = (
        df.dropna(subset=[ITEM_COLUMN])
        .groupby([ITEM_COLUMN, "Category Display"], dropna=False, as_index=False)
        .agg(
            Avg_Price=(PRICE_COLUMN, "mean"),
            Quantity_Sold=(QTY_COLUMN, "sum"),
            Revenue=(FINAL_TOTAL_COLUMN, "sum"),
        )
    )
    grouped = grouped.dropna(subset=["Avg_Price", "Quantity_Sold", "Revenue"])
    grouped = grouped[grouped["Quantity_Sold"] > 0]
    if grouped.empty:
        return empty_figure(title, "No item-level pricing data is available for this selection.")

    revenue_max = grouped["Revenue"].max()
    if pd.isna(revenue_max) or revenue_max <= 0:
        sizes = [10 for _ in range(len(grouped))]
    else:
        sizes = [max(8, min(34, (float(value) / float(revenue_max)) * 34)) for value in grouped["Revenue"]]

    hover_text = [
        (
            f"Item: {item}<br>Category: {category or 'Unknown'}<br>"
            f"Avg Price: {format_inr(price)}<br>Qty: {int(round(qty))}<br>Revenue: {format_inr(revenue)}"
        )
        for item, category, price, qty, revenue in zip(
            grouped[ITEM_COLUMN],
            grouped["Category Display"],
            grouped["Avg_Price"],
            grouped["Quantity_Sold"],
            grouped["Revenue"],
        )
    ]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=grouped["Avg_Price"],
                y=grouped["Quantity_Sold"],
                mode="markers",
                marker=dict(size=sizes, color=PALETTE[1], opacity=0.72, line=dict(width=0.6, color="#ffffff")),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        ]
    )
    fig.update_xaxes(title_text="Average Price", tickprefix="₹")
    fig.update_yaxes(title_text="Quantity Sold", type="log")
    return style_figure(fig, title, height=420)

def create_price_vs_quantity_chart(df: pd.DataFrame) -> go.Figure:
    return create_price_vs_quantity_scatter(df)


def create_top_tables_chart(df: pd.DataFrame) -> go.Figure:
    title = "Top Tables by Revenue"
    if df.empty or TABLE_COLUMN not in df.columns or FINAL_TOTAL_COLUMN not in df.columns:
        return empty_figure(title, "No data available for this selection.")

    grouped = (
        df.dropna(subset=[TABLE_COLUMN])
        .groupby(TABLE_COLUMN, as_index=False)
        .agg(
            Revenue=(FINAL_TOTAL_COLUMN, "sum"),
            Orders=(INVOICE_COLUMN, "nunique"),
        )
        .sort_values("Revenue", ascending=False)
        .head(10)
        .sort_values("Revenue", ascending=True)
    )
    if grouped.empty:
        return empty_figure(title, "No table data is available for this selection.")

    def table_label(value: float | int) -> str:
        try:
            numeric = int(round(float(value)))
        except (TypeError, ValueError):
            return "Unknown Table"
        return f"Table {numeric}{' *' if numeric == 50 else ''}"

    grouped["Table Label"] = grouped[TABLE_COLUMN].map(table_label)

    fig = go.Figure(
        data=[
            go.Bar(
                x=grouped["Revenue"],
                y=grouped["Table Label"],
                orientation="h",
                marker_color=PALETTE[0],
                customdata=[
                    [format_inr(revenue), int(orders)]
                    for revenue, orders in zip(grouped["Revenue"], grouped["Orders"])
                ],
                hovertemplate="%{y}<br>Revenue: %{customdata[0]}<br>Unique Orders: %{customdata[1]}<extra></extra>",
            )
        ]
    )
    fig.update_xaxes(title_text="Revenue", tickprefix="₹")
    fig.update_yaxes(title_text="Table")
    fig.add_annotation(
        text="Table 50 is highlighted with an asterisk because it behaves like a high-volume outlier; the dashboard keeps it in the data.",
        xref="paper",
        yref="paper",
        x=0,
        y=-0.18,
        showarrow=False,
        align="left",
        font=dict(size=11, color="#5f6b7a"),
    )
    return style_figure(fig, title, height=420)


def create_revenue_orders_aov_chart(df: pd.DataFrame) -> go.Figure:
    title = "Revenue vs Orders vs Average Order Value"
    if df.empty or DATE_COLUMN not in df.columns:
        return empty_figure(title, "No data available for this selection.")

    base = df.dropna(subset=["Date Parsed"])
    if base.empty:
        return empty_figure(title, "No dated rows are available for this selection.")

    grouped = (
        base.groupby("Date Parsed", as_index=False)
        .agg(
            Revenue=(FINAL_TOTAL_COLUMN, "sum"),
            Orders=(INVOICE_COLUMN, "nunique"),
        )
        .sort_values("Date Parsed")
    )
    grouped["AOV"] = grouped.apply(lambda row: row["Revenue"] / row["Orders"] if row["Orders"] else 0, axis=1)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.42, 0.32, 0.26],
        subplot_titles=("Revenue", "Orders", "Average Order Value"),
    )

    fig.add_trace(
        go.Scatter(
            x=grouped["Date Parsed"],
            y=grouped["Revenue"],
            mode="lines+markers",
            line=dict(color=PALETTE[0], width=3),
            marker=dict(size=6, color=PALETTE[0]),
            customdata=grouped["Revenue"].map(format_inr),
            hovertemplate="Date: %{x|%d-%b-%Y}<br>Revenue: %{customdata}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Date Parsed"],
            y=grouped["Orders"],
            mode="lines+markers",
            line=dict(color=PALETTE[1], width=3),
            marker=dict(size=6, color=PALETTE[1]),
            customdata=grouped["Orders"].map(lambda value: f"{int(value)}"),
            hovertemplate="Date: %{x|%d-%b-%Y}<br>Orders: %{customdata}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Date Parsed"],
            y=grouped["AOV"],
            mode="lines+markers",
            line=dict(color=PALETTE[2], width=3),
            marker=dict(size=6, color=PALETTE[2]),
            customdata=grouped["AOV"].map(format_inr),
            hovertemplate="Date: %{x|%d-%b-%Y}<br>AOV: %{customdata}<extra></extra>",
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="Revenue", tickprefix="₹", row=1, col=1)
    fig.update_yaxes(title_text="Orders", row=2, col=1)
    fig.update_yaxes(title_text="AOV", tickprefix="₹", row=3, col=1)
    fig.update_xaxes(title_text="Date", tickformat="%d-%b", row=3, col=1)
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.0, xanchor="left"),
        height=760,
        margin=dict(l=20, r=20, t=80, b=30),
        font=dict(family='"Segoe UI", "Trebuchet MS", sans-serif', size=13),
    )
    return fig


def build_top_items_table(df: pd.DataFrame) -> pd.DataFrame:
    required = {ITEM_COLUMN, "Category Display", QTY_COLUMN, FINAL_TOTAL_COLUMN, PRICE_COLUMN}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=["Item Name", "Category", "Quantity Sold", "Revenue", "Average Price"])

    table = (
        df.dropna(subset=[ITEM_COLUMN, "Category Display"])
        .groupby([ITEM_COLUMN, "Category Display"], as_index=False)
        .agg(
            Quantity_Sold=(QTY_COLUMN, "sum"),
            Revenue=(FINAL_TOTAL_COLUMN, "sum"),
        )
    )
    if table.empty:
        return pd.DataFrame(columns=["Item Name", "Category", "Quantity Sold", "Revenue", "Average Price"])

    table["Average Price"] = table.apply(
        lambda row: row["Revenue"] / row["Quantity_Sold"] if row["Quantity_Sold"] else 0,
        axis=1,
    )
    table = table.rename(
        columns={
            ITEM_COLUMN: "Item Name",
            "Category Display": "Category",
            "Quantity_Sold": "Quantity Sold",
        }
    )
    table = table.sort_values("Revenue", ascending=False).head(20).reset_index(drop=True)
    return table[["Item Name", "Category", "Quantity Sold", "Revenue", "Average Price"]]


def render_metric(label: str, value: str, help_text: str | None = None) -> None:
    help_html = f'<div class="kpi-help">{help_text}</div>' if help_text else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(df: pd.DataFrame, raw_report: dict[str, object]) -> None:
    st.title("🍽️ Restaurant Sales & Operations Dashboard")
    st.caption("Interactive analysis of restaurant sales, products, revenue and operations")

    with st.sidebar:
        st.header("🔎 Filters")

        available_dates = df["Date Parsed"].dropna() if "Date Parsed" in df.columns else pd.Series(dtype="datetime64[ns]")
        if available_dates.empty:
            date_selection = None
            st.info("Date filtering is unavailable because no valid dates were parsed.")
        else:
            min_date = available_dates.min().date()
            max_date = available_dates.max().date()
            date_selection = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )

        category_options = (
            sorted([value for value in df["Category Display"].dropna().unique().tolist()])
            if "Category Display" in df.columns
            else []
        )
        selected_categories = st.multiselect(
            "Category",
            options=category_options,
            default=category_options,
        ) if category_options else []

        item_options = (
            sorted([value for value in df[ITEM_COLUMN].dropna().astype(str).unique().tolist()])
            if ITEM_COLUMN in df.columns
            else []
        )
        selected_items = st.multiselect(
            "Item Name",
            options=item_options,
            default=item_options,
        ) if item_options else []

        table_options = []
        if TABLE_COLUMN in df.columns:
            table_values = df[TABLE_COLUMN].dropna().tolist()
            table_options = sorted({int(round(float(value))) for value in table_values})
        selected_tables = st.multiselect(
            "Table",
            options=table_options,
            default=table_options,
        ) if table_options else []

        remove_duplicates = st.checkbox("Remove exact duplicate rows", value=False)

    filtered_df = apply_filters(
        df,
        date_selection=date_selection,
        selected_categories=selected_categories,
        selected_items=selected_items,
        selected_tables=selected_tables,
        remove_duplicates=remove_duplicates,
    )

    if filtered_df.empty:
        st.info("No data available for the selected filters.")
        st.stop()

    kpis = calculate_kpis(filtered_df)

    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #fffdf8 0%, #f5f1ea 100%);
                color: #111827;
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .kpi-card {
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid rgba(53, 65, 80, 0.08);
                border-radius: 16px;
                padding: 0.95rem 1rem 0.85rem 1rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
                min-height: 112px;
            }
            .kpi-label {
                color: #111827;
                font-size: 0.88rem;
                line-height: 1.2;
                margin-bottom: 0.6rem;
                font-weight: 500;
            }
            .kpi-value {
                color: #111827;
                font-size: 0.92rem;
                line-height: 1.05;
                font-weight: 700;
                overflow: hidden;
                text-overflow: clip;
                white-space: nowrap;
                font-variant-numeric: tabular-nums;
            }
            .kpi-help {
                color: #6b7280;
                font-size: 0.85rem;
                margin-top: 0.55rem;
                line-height: 1.35;
            }
            .main h1, .main h2, .main h3, .main p, .main span {
                color: #111827;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Key Performance Indicators")
    kpi_columns = st.columns(5)
    with kpi_columns[0]:
        render_metric("Total Revenue", format_inr(kpis["revenue"]))
    with kpi_columns[1]:
        render_metric("Total Orders", f"{int(kpis['orders']):,}")
    with kpi_columns[2]:
        render_metric("Items Sold", f"{int(round(kpis['items_sold'])):,}")
    with kpi_columns[3]:
        render_metric("Average Order Value", format_inr(kpis["aov"]))
    with kpi_columns[4]:
        render_metric("Total Discount", format_inr(kpis["total_discount"]))
        st.caption(f"Discount as % of subtotal: {format_percent(kpis['discount_pct'])}")

    st.divider()
    st.subheader("Revenue Overview")
    st.plotly_chart(create_daily_revenue_trend_chart(filtered_df), use_container_width=True)
    st.plotly_chart(create_revenue_by_category_chart(filtered_df), use_container_width=True)

    st.divider()
    st.subheader("Product Performance")
    st.plotly_chart(create_top_items_by_qty_chart(filtered_df), use_container_width=True)
    st.plotly_chart(create_top_items_by_revenue_chart(filtered_df), use_container_width=True)
    st.plotly_chart(create_price_vs_quantity_chart(filtered_df), use_container_width=True)

    st.divider()
    st.subheader("Operational Analysis")
    st.plotly_chart(create_revenue_by_hour_chart(filtered_df), use_container_width=True)
    st.plotly_chart(create_orders_by_day_chart(filtered_df), use_container_width=True)
    st.plotly_chart(create_top_tables_chart(filtered_df), use_container_width=True)
    st.plotly_chart(create_revenue_orders_aov_chart(filtered_df), use_container_width=True)

    st.divider()
    st.subheader("Category Analysis")
    st.plotly_chart(create_quantity_by_category_chart(filtered_df), use_container_width=True)
    st.plotly_chart(create_sales_mix_chart(filtered_df), use_container_width=True)

    st.divider()
    st.subheader("Detailed Data")
    top_items_table = build_top_items_table(filtered_df)
    if top_items_table.empty:
        st.info("No detailed item data is available for the selected filters.")
    else:
        display_table = top_items_table.copy()
        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Item Name": st.column_config.TextColumn("Item Name"),
                "Category": st.column_config.TextColumn("Category"),
                "Quantity Sold": st.column_config.NumberColumn("Quantity Sold", format="%d"),
                "Revenue": st.column_config.NumberColumn("Revenue", format="₹%.0f"),
                "Average Price": st.column_config.NumberColumn("Average Price", format="₹%.0f"),
            },
        )

    st.divider()
    with st.expander("Dataset Information", expanded=False):
        total_subtotal = float(df[SUBTOTAL_COLUMN].sum(skipna=True)) if SUBTOTAL_COLUMN in df.columns else 0.0
        total_tax = float(df[TAX_COLUMN].sum(skipna=True)) if TAX_COLUMN in df.columns else 0.0
        total_discount = float(df[DISCOUNT_COLUMN].sum(skipna=True)) if DISCOUNT_COLUMN in df.columns else 0.0
        total_final = float(df[FINAL_TOTAL_COLUMN].sum(skipna=True)) if FINAL_TOTAL_COLUMN in df.columns else 0.0
        unique_invoices = int(df[INVOICE_COLUMN].nunique(dropna=True)) if INVOICE_COLUMN in df.columns else 0
        unique_items = int(df[ITEM_COLUMN].nunique(dropna=True)) if ITEM_COLUMN in df.columns else 0
        categories = int(df["Category Display"].nunique(dropna=True)) if "Category Display" in df.columns else 0
        info_columns = st.columns(2)
        with info_columns[0]:
            st.markdown(
                f"""
                - Rows loaded: **{raw_report['row_count']:,}**
                - Unique invoices: **{unique_invoices:,}**
                - Unique items: **{unique_items:,}**
                - Categories: **{categories:,}**
                - Start date: **{raw_report['start_date']}**
                - End date: **{raw_report['end_date']}**
                - Total subtotal: **{format_inr(total_subtotal)}**
                - Total tax: **{format_inr(total_tax)}**
                - Total discount: **{format_inr(total_discount)}**
                - Total final revenue: **{format_inr(total_final)}**
                - Exact duplicate rows: **{raw_report['duplicate_rows']:,}**
                - Unparseable date rows: **{raw_report['unparseable_date_rows']:,}**
                - Unparseable timestamp rows: **{raw_report['unparseable_timestamp_rows']:,}**
                """
            )
        with info_columns[1]:
            st.markdown("Missing-value summary")
            missing_summary = (
                df.isna().sum()
                .rename("Missing")
                .to_frame()
                .assign(Percentage=lambda frame: (frame["Missing"] / len(df) * 100.0).round(2))
                .reset_index()
                .rename(columns={"index": "Column"})
            )
            missing_summary = missing_summary[missing_summary["Missing"] > 0].sort_values("Missing", ascending=False)
            if missing_summary.empty:
                st.write("No missing values in the cleaned dataset.")
            else:
                st.dataframe(missing_summary, use_container_width=True, hide_index=True)

        st.markdown("Excluded columns")
        st.write(
            ", ".join([f"{column} ({reason})" for column, reason in EXCLUDED_COLUMNS.items()])
        )
        st.markdown(
            f"""
            - Outlier note: invoice **23827** contains a single line worth **{format_inr(22638)}**, which is materially above the next-largest line and is retained in the analysis.
            - Duplicate handling: exact duplicate rows are kept by default and can be removed with the sidebar checkbox.
            - Time-based views use explicit DD-MM-YYYY parsing so values like **01-04-2023** remain 1 April, not 4 January.
            """
        )


def main() -> None:
    raw_df = load_data()
    if raw_df is None:
        st.error("restaurant_data.csv was not found. Please place it in the project directory.")
        st.stop()
    if raw_df.empty:
        st.error("restaurant_data.csv is empty. Please provide a non-empty dataset.")
        st.stop()

    cleaned_df, report = clean_data(raw_df)
    render_dashboard(cleaned_df, report)


if __name__ == "__main__":
    main()