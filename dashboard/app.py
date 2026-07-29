"""Fulfillment Control Tower — e-commerce fulfillment analytics dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import agent
from data import (
    fmt_currency,
    fmt_days,
    fmt_int,
    fmt_pct,
    load_categories as _load_categories,
    load_monthly as _load_monthly,
    load_orders as _load_orders,
    load_sellers as _load_sellers,
    titleize,
)

# ----------------------------------------------------------------------------
# Palette (neutral base + single accent, fixed status colors)
# ----------------------------------------------------------------------------
ACCENT = "#2a78d6"
SEQUENTIAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK_SECONDARY, size=13),
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    margin=dict(l=10, r=10, t=10, b=10),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="system-ui, sans-serif"),
    hovermode="x unified",
    showlegend=False,
)

st.set_page_config(
    page_title="Fulfillment Control Tower",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Global CSS — compact, consulting-style
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    }}
    .block-container {{
        padding-top: 1.3rem;
        padding-bottom: 1.7rem;
        max-width: 1400px;
    }}
    h1, h2, h3 {{ color: {INK}; font-weight: 600; font-family: "Sora", system-ui, -apple-system, "Segoe UI", sans-serif; }}
    .header-title {{
        font-size: 2.0rem;
        font-weight: 700;
        color: {INK};
        margin-bottom: 0.1rem;
        font-family: "Sora", system-ui, -apple-system, "Segoe UI", sans-serif;
    }}
    .header-question {{
        font-size: 0.98rem;
        color: {INK_SECONDARY};
        border-left: 3px solid {ACCENT};
        padding-left: 0.6rem;
        margin-top: 0.35rem;
        margin-bottom: 0.15rem;
    }}
    .header-desc {{
        font-size: 0.88rem;
        color: {INK_MUTED};
        margin-bottom: 0.45rem;
    }}
    .section-title {{
        font-size: 1.05rem;
        font-weight: 600;
        color: {INK};
        margin-top: 0.1rem;
        margin-bottom: 0.2rem;
        font-family: "Sora", system-ui, -apple-system, "Segoe UI", sans-serif;
    }}
    .chart-caption {{
        color: {INK_MUTED};
        font-size: 0.76rem;
        margin-top: -0.3rem;
    }}
    .row-spacer {{ height: 0.3rem; }}
    div[class*="st-key-kpi_"] {{
        padding: 0.55rem 0.9rem 0.4rem 0.9rem !important;
    }}
    div[class*="st-key-chart_"] {{
        padding: 0.5rem 0.7rem 0.3rem 0.7rem !important;
    }}
    .sidebar-subhead {{
        font-size: 0.8rem;
        font-weight: 600;
        color: {INK_SECONDARY};
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-top: 0.2rem;
        margin-bottom: 0.1rem;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.78rem;
        color: {INK_MUTED};
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 1.5rem;
        color: {INK};
    }}
    .rec-card {{
        background-color: {SURFACE};
        border: 1px solid {GRIDLINE};
        border-left: 3px solid {ACCENT};
        border-radius: 6px;
        padding: 0.55rem 1rem;
        margin-bottom: 0.4rem;
    }}
    .rec-title {{
        font-weight: 600;
        color: {INK};
        font-size: 0.95rem;
        margin-bottom: 0.15rem;
        font-family: "Sora", system-ui, -apple-system, "Segoe UI", sans-serif;
    }}
    .rec-body {{
        color: {INK_SECONDARY};
        font-size: 0.86rem;
        line-height: 1.35rem;
    }}
    .scope-note {{
        color: {INK_MUTED};
        font-size: 0.76rem;
        margin-top: -0.2rem;
        margin-bottom: 0.4rem;
    }}
    .agent-badge {{
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.1rem 0.55rem;
        border-radius: 999px;
        margin-bottom: 0.5rem;
    }}
    .agent-badge-demo {{ background-color: #fdf3d9; color: #7a5b00; }}
    .agent-badge-llm {{ background-color: #e3edfb; color: {ACCENT}; }}
    .agent-section-label {{
        font-weight: 600;
        color: {INK};
        font-size: 0.88rem;
        margin-top: 0.6rem;
        margin-bottom: 0.1rem;
        font-family: "Sora", system-ui, -apple-system, "Segoe UI", sans-serif;
    }}
    .agent-section-body {{
        color: {INK_SECONDARY};
        font-size: 0.86rem;
        line-height: 1.4rem;
    }}
    footer {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Data loading (logic lives in data.py; cache it here at the Streamlit layer)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading order data...")
def load_orders() -> pd.DataFrame:
    return _load_orders()


@st.cache_data(show_spinner=False)
def load_sellers() -> pd.DataFrame:
    return _load_sellers()


@st.cache_data(show_spinner=False)
def load_categories() -> pd.DataFrame:
    return _load_categories()


@st.cache_data(show_spinner=False)
def load_monthly() -> pd.DataFrame:
    return _load_monthly()


orders = load_orders()
sellers = load_sellers()
categories = load_categories()
monthly = load_monthly()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown('<div class="header-title">Fulfillment Control Tower</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="header-question">Business question: How can the company reduce late deliveries '
    "and excessive freight costs while protecting customer satisfaction?</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="header-desc">An operational analytics view of order fulfillment — delivery '
    "timeliness, seller handling time, shipping distance, freight cost, and customer review "
    "outcomes — built to surface where the fulfillment network is losing time and money.</div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------
st.sidebar.header("Filters")

min_date, max_date = orders["purchase_date"].min(), orders["purchase_date"].max()
date_range = st.sidebar.date_input(
    "Purchase date",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

selected_outcome = st.sidebar.pills(
    "Delivery outcome",
    options=["All", "On-Time", "Late"],
    default="All",
    required=True,
    help="On-Time / Late applies to delivered orders only.",
)

order_status_options = sorted(orders["order_status_label"].unique())
order_status_key = "order_status_filter"
prior_order_statuses = st.session_state.get(order_status_key, order_status_options)
with st.sidebar.expander(
    f"Order status ({len(prior_order_statuses)}/{len(order_status_options)} selected)",
    expanded=False,
):
    selected_order_statuses = st.multiselect(
        "Order status",
        options=order_status_options,
        default=order_status_options,
        key=order_status_key,
        label_visibility="collapsed",
        help="Order lifecycle stage (e.g. Delivered, Shipped, Canceled) — independent of delivery outcome.",
    )

st.sidebar.markdown(
    '<div class="scope-note">Purchase date, delivery outcome, and order status apply to all '
    "order-level charts and KPIs.</div>",
    unsafe_allow_html=True,
)

st.sidebar.divider()
st.sidebar.markdown('<div class="sidebar-subhead">Chart-specific filters</div>', unsafe_allow_html=True)
st.sidebar.caption("Affect only the matching chart below — not KPIs or other charts.")

category_options = sorted(categories["product_category_name_english"].unique())
selected_categories = st.sidebar.multiselect(
    "Category — Top Categories chart",
    options=category_options,
    default=[],
    format_func=titleize,
    help="Filters only the Top Categories by Late Orders chart. Leave empty to show all categories.",
)

seller_options = sellers.sort_values("order_count", ascending=False)["seller_id"].tolist()
selected_sellers = st.sidebar.multiselect(
    "Seller — Top Sellers chart",
    options=seller_options,
    default=[],
    format_func=lambda sid: f"{sid[:8]}… ({int(sellers.loc[sellers.seller_id == sid, 'order_count'].iloc[0])} orders)",
    help="Filters only the Top Sellers by Late Orders chart. Leave empty to show the overall top 10.",
)

# ----------------------------------------------------------------------------
# Apply filters
# ----------------------------------------------------------------------------
mask = (
    (orders["purchase_date"] >= start_date)
    & (orders["purchase_date"] <= end_date)
    & (orders["order_status_label"].isin(selected_order_statuses))
)
filtered = orders.loc[mask].copy()

if selected_outcome != "All":
    is_late_target = selected_outcome == "Late"
    filtered = filtered[(filtered["order_status"] == "delivered") & (filtered["is_late"] == is_late_target)]

delivered = filtered[filtered["order_status"] == "delivered"].copy()

monthly_filtered = monthly[
    (monthly["purchase_month"] >= start_date.strftime("%Y-%m"))
    & (monthly["purchase_month"] <= end_date.strftime("%Y-%m"))
].copy()

sellers_filtered = sellers[sellers["seller_id"].isin(selected_sellers)] if selected_sellers else sellers
categories_filtered = (
    categories[categories["product_category_name_english"].isin(selected_categories)]
    if selected_categories
    else categories
)

st.caption(f"Showing {fmt_int(len(filtered))} of {fmt_int(len(orders))} total orders based on current filters.")

# ----------------------------------------------------------------------------
# KPI row
# ----------------------------------------------------------------------------
total_orders = len(filtered)
late_rate = delivered["is_late"].mean() * 100 if len(delivered) else np.nan
avg_review = filtered["review_score"].mean()
median_delivery_days = filtered["delivery_days"].median()
median_freight = filtered["total_freight_value"].median()

k1, k2, k3, k4, k5 = st.columns(5)
with k1.container(border=True, key="kpi_total"):
    st.metric("Total Orders", fmt_int(total_orders))
with k2.container(border=True, key="kpi_late_rate"):
    st.metric("Late Delivery Rate", fmt_pct(late_rate), help="Share of delivered orders that arrived after the estimated delivery date.")
with k3.container(border=True, key="kpi_review"):
    st.metric("Avg Review Score", "N/A" if pd.isna(avg_review) else f"{avg_review:.2f} / 5")
with k4.container(border=True, key="kpi_delivery_days"):
    st.metric("Median Delivery Days", fmt_days(median_delivery_days))
with k5.container(border=True, key="kpi_freight"):
    st.metric("Median Freight Value", fmt_currency(median_freight))

st.markdown("---")

# ----------------------------------------------------------------------------
# Chart helpers
# ----------------------------------------------------------------------------
def empty_chart_message(message="No data available for the current filter selection."):
    st.info(message)


def row_spacer():
    st.markdown('<div class="row-spacer"></div>', unsafe_allow_html=True)


def style_fig(fig, height=250, yaxis_title=None, xaxis_title=None):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(title=xaxis_title, showgrid=False, linecolor=GRIDLINE, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(title=yaxis_title, showgrid=True, gridcolor=GRIDLINE, zeroline=False, tickfont=dict(color=INK_MUTED))
    return fig


# ----------------------------------------------------------------------------
# 1. Monthly late-delivery rate trend
# ----------------------------------------------------------------------------
with st.container(border=True, key="chart_monthly"):
    st.markdown('<div class="section-title">Monthly Late-Delivery Rate Trend</div>', unsafe_allow_html=True)
    if monthly_filtered.empty:
        empty_chart_message()
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=monthly_filtered["month_label"],
                y=monthly_filtered["late_delivery_rate"],
                mode="lines+markers",
                line=dict(color=ACCENT, width=2.5),
                marker=dict(size=6, color=ACCENT),
                fill="tozeroy",
                fillcolor="rgba(42, 120, 214, 0.08)",
                hovertemplate="<b>%{x}</b><br>Late rate: %{y:.2f}%<br>Total orders: %{customdata:,}<extra></extra>",
                customdata=monthly_filtered["total_orders"],
            )
        )
        fig = style_fig(fig, height=255, yaxis_title="Late delivery rate (%)")
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Based on delivered orders per purchase month (months with at least 100 delivered orders). Reflects the selected date range; not affected by chart-specific or order-status filters.")

row_spacer()

# ----------------------------------------------------------------------------
# Row: handling-time group / distance group
# ----------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1, st.container(border=True, key="chart_handling"):
    st.markdown('<div class="section-title">Late-Delivery Rate by Handling-Time Group</div>', unsafe_allow_html=True)
    if delivered.empty:
        empty_chart_message()
    else:
        grp = (
            delivered.groupby("handling_group", observed=True)
            .agg(late_rate=("is_late", "mean"), orders=("order_id", "count"))
            .reset_index()
        )
        grp["late_rate_pct"] = grp["late_rate"] * 100
        fig = go.Figure(
            go.Bar(
                x=grp["handling_group"].astype(str),
                y=grp["late_rate_pct"],
                marker_color=SEQUENTIAL[: len(grp)],
                text=[f"{v:.1f}%" for v in grp["late_rate_pct"]],
                textposition="outside",
                textfont=dict(color=INK_SECONDARY, size=12),
                hovertemplate="<b>%{x} to ship</b><br>Late rate: %{y:.1f}%<br>Orders: %{customdata:,}<extra></extra>",
                customdata=grp["orders"],
            )
        )
        fig = style_fig(fig, height=250, yaxis_title="Late delivery rate (%)")
        fig.update_yaxes(range=[0, grp["late_rate_pct"].max() * 1.18])
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

with col2, st.container(border=True, key="chart_distance"):
    st.markdown('<div class="section-title">Late-Delivery Rate by Distance Group</div>', unsafe_allow_html=True)
    if delivered.empty:
        empty_chart_message()
    else:
        grp = (
            delivered.groupby("distance_group", observed=True)
            .agg(late_rate=("is_late", "mean"), orders=("order_id", "count"))
            .reset_index()
        )
        grp["late_rate_pct"] = grp["late_rate"] * 100
        fig = go.Figure(
            go.Bar(
                x=grp["distance_group"].astype(str),
                y=grp["late_rate_pct"],
                marker_color=SEQUENTIAL[: len(grp)],
                text=[f"{v:.1f}%" for v in grp["late_rate_pct"]],
                textposition="outside",
                textfont=dict(color=INK_SECONDARY, size=12),
                hovertemplate="<b>%{x}</b><br>Late rate: %{y:.1f}%<br>Orders: %{customdata:,}<extra></extra>",
                customdata=grp["orders"],
            )
        )
        fig = style_fig(fig, height=250, yaxis_title="Late delivery rate (%)")
        fig.update_yaxes(range=[0, grp["late_rate_pct"].max() * 1.18])
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

row_spacer()

# ----------------------------------------------------------------------------
# Row: top sellers / top categories by late orders
# ----------------------------------------------------------------------------
col3, col4 = st.columns(2)

with col3, st.container(border=True, key="chart_sellers"):
    st.markdown('<div class="section-title">Top 10 Sellers by Late-Order Count</div>', unsafe_allow_html=True)
    if sellers_filtered.empty:
        empty_chart_message()
    else:
        top_sellers = sellers_filtered.sort_values("late_orders", ascending=False).head(10).iloc[::-1]
        hover_text = [
            f"Seller {sid[:8]}…<br>"
            f"Total orders: {fmt_int(oc)}<br>"
            f"Late orders: {fmt_int(lo)} ({lr * 100:.1f}% late rate)<br>"
            f"Median handling time: {fmt_days(mh) if pd.notna(mh) else 'N/A'}<br>"
            f"State: {st_ if pd.notna(st_) else 'N/A'}"
            for sid, oc, lo, lr, mh, st_ in zip(
                top_sellers["seller_id"],
                top_sellers["order_count"],
                top_sellers["late_orders"],
                top_sellers["late_rate"],
                top_sellers["median_handling_days"],
                top_sellers["seller_state"],
            )
        ]
        fig = go.Figure(
            go.Bar(
                x=top_sellers["late_orders"],
                y=[f"{sid[:8]}…" for sid in top_sellers["seller_id"]],
                orientation="h",
                marker_color=ACCENT,
                hovertext=hover_text,
                hoverinfo="text",
            )
        )
        fig = style_fig(fig, xaxis_title="Number of late orders")
        fig.update_xaxes(range=[0, top_sellers["late_orders"].max() * 1.12])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown(
            '<div class="chart-caption">Ranked by number of late orders, not late rate — hover for '
            "order volume, late rate, handling time, and state.</div>",
            unsafe_allow_html=True,
        )

with col4, st.container(border=True, key="chart_categories"):
    st.markdown('<div class="section-title">Top Categories by Late Orders</div>', unsafe_allow_html=True)
    if categories_filtered.empty:
        empty_chart_message()
    else:
        top_categories = categories_filtered.sort_values("late_orders", ascending=False).head(10).iloc[::-1]
        fig = go.Figure(
            go.Bar(
                x=top_categories["late_orders"],
                y=top_categories["category_label"],
                orientation="h",
                marker_color=ACCENT,
                text=[fmt_int(v) for v in top_categories["late_orders"]],
                textposition="outside",
                textfont=dict(color=INK_SECONDARY, size=12),
                hovertemplate="%{y}<br>Late orders: %{x}<br>Late rate: %{customdata:.1%}<extra></extra>",
                customdata=top_categories["late_rate"],
            )
        )
        fig = style_fig(fig, xaxis_title="Number of late orders")
        fig.update_xaxes(range=[0, top_categories["late_orders"].max() * 1.15])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

row_spacer()

# ----------------------------------------------------------------------------
# Row: freight by weight group / review score on-time vs late
# ----------------------------------------------------------------------------
col5, col6 = st.columns(2)

with col5, st.container(border=True, key="chart_freight"):
    st.markdown('<div class="section-title">Freight Value by Weight Group</div>', unsafe_allow_html=True)
    weight_df = filtered.dropna(subset=["weight_group", "total_freight_value"])
    if weight_df.empty:
        empty_chart_message()
    else:
        grp = (
            weight_df.groupby("weight_group", observed=True)
            .agg(median_freight=("total_freight_value", "median"), orders=("order_id", "count"))
            .reset_index()
        )
        fig = go.Figure(
            go.Bar(
                x=grp["weight_group"].astype(str),
                y=grp["median_freight"],
                marker_color=SEQUENTIAL[: len(grp)],
                text=[fmt_currency(v) for v in grp["median_freight"]],
                textposition="outside",
                textfont=dict(color=INK_SECONDARY, size=12),
                hovertemplate="<b>%{x}</b><br>Median freight: R$ %{y:.2f}<br>Orders: %{customdata:,}<extra></extra>",
                customdata=grp["orders"],
            )
        )
        fig = style_fig(fig, yaxis_title="Median freight value (R$)")
        fig.update_yaxes(range=[0, grp["median_freight"].max() * 1.18])
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

with col6, st.container(border=True, key="chart_review"):
    st.markdown('<div class="section-title">Avg Review Score: On-Time vs Late</div>', unsafe_allow_html=True)
    review_df = delivered.dropna(subset=["review_score"])
    if review_df.empty:
        empty_chart_message()
    else:
        grp = review_df.groupby("is_late")["review_score"].mean()
        labels = ["On-Time", "Late"]
        values = [grp.get(False, np.nan), grp.get(True, np.nan)]
        colors = [GOOD, CRITICAL]
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f"{v:.2f}" if not pd.isna(v) else "N/A" for v in values],
                textposition="outside",
                hovertemplate="<b>%{x} deliveries</b><br>Avg review score: %{y:.2f} / 5<extra></extra>",
            )
        )
        fig = style_fig(fig, yaxis_title="Average review score (1-5)")
        fig.update_yaxes(range=[0, 5.5])
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

st.markdown("---")

# ----------------------------------------------------------------------------
# Operational Recommendations (grounded in filtered data)
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title" style="font-size:1.2rem;">Operational Recommendations</div>', unsafe_allow_html=True)


def rate_for(df, group_col, label):
    row = df[df[group_col].astype(str) == label]
    if row.empty or row["orders"].sum() == 0:
        return np.nan
    return (row["late_rate"] * row["orders"]).sum() / row["orders"].sum() * 100


handling_grp = (
    delivered.groupby("handling_group", observed=True)
    .agg(late_rate=("is_late", "mean"), orders=("order_id", "count"))
    .reset_index()
    if not delivered.empty
    else pd.DataFrame(columns=["handling_group", "late_rate", "orders"])
)
distance_grp = (
    delivered.groupby("distance_group", observed=True)
    .agg(late_rate=("is_late", "mean"), orders=("order_id", "count"))
    .reset_index()
    if not delivered.empty
    else pd.DataFrame(columns=["distance_group", "late_rate", "orders"])
)
weight_grp = (
    weight_df.groupby("weight_group", observed=True)
    .agg(median_freight=("total_freight_value", "median"))
    .reset_index()
    if not weight_df.empty
    else pd.DataFrame(columns=["weight_group", "median_freight"])
)

fast_rate = rate_for(handling_grp, "handling_group", "0-1 days")
slow_rate = rate_for(handling_grp, "handling_group", "8+ days")
near_rate = rate_for(distance_grp, "distance_group", "≤250 km")
far_rate = rate_for(distance_grp, "distance_group", "1,000+ km")
light_freight = weight_grp.loc[weight_grp["weight_group"].astype(str) == "<0.5 kg", "median_freight"]
heavy_freight = weight_grp.loc[weight_grp["weight_group"].astype(str) == "10+ kg", "median_freight"]
light_freight = light_freight.iloc[0] if len(light_freight) else np.nan
heavy_freight = heavy_freight.iloc[0] if len(heavy_freight) else np.nan

alert_sellers = sellers_filtered[(sellers_filtered["late_rate"] >= 0.15) & (sellers_filtered["order_count"] >= 5)]
top_late_categories = categories_filtered.sort_values("late_orders", ascending=False).head(2)
peak_month = monthly_filtered.sort_values("late_delivery_rate", ascending=False).head(1)

if not top_late_categories.empty:
    category_evidence = ", ".join(
        f"{r.category_label} ({fmt_int(r.late_orders)} late orders)" for r in top_late_categories.itertuples()
    ) + " lead in late-order volume."
else:
    category_evidence = "No category data available for the current selection."

if not peak_month.empty:
    month_evidence = (
        f"{peak_month.iloc[0]['month_label']} recorded the highest late rate in the selected range, at "
        f"{fmt_pct(peak_month.iloc[0]['late_delivery_rate'])}."
    )
else:
    month_evidence = "No monthly data available for the current selection."

recommendations = [
    (
        "Create seller handling-time alerts",
        f"Orders handled within 1 day show a {fmt_pct(fast_rate)} late rate versus {fmt_pct(slow_rate)} for orders "
        f"taking 8+ days, and {fmt_int(len(alert_sellers))} sellers (5+ orders) run at a 15%+ late rate. "
        "Flag orders unshipped after 3 days and escalate sellers above a 7-day median handling time, weighted by order volume.",
    ),
    (
        "Prioritize high-volume categories with high late-order counts",
        f"{category_evidence} Run targeted seller, route, and handling-time reviews in these categories before any marketplace-wide change.",
    ),
    (
        "Improve planning for long-distance routes",
        f"Routes of 1,000+ km show a {fmt_pct(far_rate)} late rate versus {fmt_pct(near_rate)} for routes up to 250 km. "
        "Extend promised delivery windows on long routes and prefer closer sellers when equivalent options exist.",
    ),
    (
        "Review freight costs for heavy and low-value orders",
        f"Orders over 10 kg carry a median freight value of {fmt_currency(heavy_freight)} versus {fmt_currency(light_freight)} for orders under 0.5 kg. "
        "Review packaging efficiency and freight-subsidy rules for heavy or low-value products.",
    ),
    (
        "Monitor monthly delivery spikes",
        f"{month_evidence} Track late rate, handling time, and review scores monthly, and investigate spikes as soon as they appear.",
    ),
]

for title, body in recommendations:
    st.markdown(
        f'<div class="rec-card"><div class="rec-title">{title}</div><div class="rec-body">{body}</div></div>',
        unsafe_allow_html=True,
    )

st.caption(
    "Recommendation figures recompute from the order-level, seller, and category filters currently applied above."
)

st.markdown("---")

# ----------------------------------------------------------------------------
# AI Operations Investigator
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title" style="font-size:1.2rem;">AI Operations Investigator</div>', unsafe_allow_html=True)
st.caption(
    "Ask an operational question. The agent picks which analytical tools to run, executes them "
    "against this dashboard's data, and returns an evidence-based diagnosis — it never invents a "
    "number that a tool didn't return. Uses the current Purchase date and Delivery outcome filters; "
    "category, seller, and order-status filters are not passed to the agent, for the same reason "
    "they only scope their own charts above."
)

SUGGESTED_QUESTIONS = [
    "Why did late deliveries increase?",
    "Which sellers should operations review first?",
    "Where can freight costs be reduced?",
    "What is hurting customer satisfaction?",
    "Generate an executive operations brief.",
]

suggestion_cols = st.columns(len(SUGGESTED_QUESTIONS))
clicked_question = None
for col, q in zip(suggestion_cols, SUGGESTED_QUESTIONS):
    if col.button(q, key=f"suggested_{q}", width="stretch"):
        clicked_question = q

input_col, ask_col, demo_col = st.columns([3, 1, 1.4])
custom_question = input_col.text_input(
    "Ask your own question",
    key="investigator_input",
    placeholder="e.g. Why are freight costs high for electronics?",
    label_visibility="collapsed",
)
ask_clicked = ask_col.button("Investigate", type="primary", width="stretch")
force_demo = demo_col.checkbox(
    "Demo mode", value=False, help="Skip the API call and use the deterministic rule-based report instead."
)

question_to_run = clicked_question or (custom_question.strip() if ask_clicked and custom_question.strip() else None)

if question_to_run:
    with st.spinner("Investigating..."):
        investigation = agent.investigate(
            question_to_run,
            filtered,
            sellers,
            categories,
            monthly_filtered,
            {"start_date": start_date, "end_date": end_date, "outcome": selected_outcome},
            force_demo=force_demo,
        )
    st.session_state["investigator_result"] = investigation
    st.session_state["investigator_question_asked"] = question_to_run

def _escape_markdown_dollars(text: str) -> str:
    # Report text cites currency as "R$ 12.34" — an unescaped "$" opens
    # Streamlit's LaTeX math mode mid-sentence and mangles the rendering.
    return str(text).replace("$", "\\$")


result = st.session_state.get("investigator_result")
asked = st.session_state.get("investigator_question_asked")

if result:
    st.markdown(f'<div class="chart-caption">Question: “{_escape_markdown_dollars(asked)}”</div>', unsafe_allow_html=True)

    with st.expander(f"Tools the agent used ({len(result.tool_calls)})"):
        if not result.tool_calls:
            st.caption("No tools were called for this question.")
        for call in result.tool_calls:
            label = call.input if call.input else "no parameters"
            st.markdown(f"**`{call.name}`** — {_escape_markdown_dollars(label)}")
            st.json(call.output, expanded=False)

    with st.container(border=True, key="chart_agent_report"):
        badge_class = "agent-badge-demo" if result.mode == "demo" else "agent-badge-llm"
        badge_text = (
            "Demo mode — no ANTHROPIC_API_KEY configured, showing a deterministic rule-based report"
            if result.mode == "demo"
            else f"Generated by {agent.MODEL}"
        )
        st.markdown(f'<div class="agent-badge {badge_class}">{badge_text}</div>', unsafe_allow_html=True)

        if result.mode == "error":
            st.error(result.answer)
        elif "Report" in result.sections:
            st.markdown(
                f'<div class="agent-section-body">{_escape_markdown_dollars(result.sections["Report"])}</div>',
                unsafe_allow_html=True,
            )
        else:
            for header in agent.SECTION_HEADERS:
                body = result.sections.get(header)
                if not body:
                    continue
                st.markdown(f'<div class="agent-section-label">{header}</div>', unsafe_allow_html=True)
                st.markdown(_escape_markdown_dollars(body))

        if result.warning and result.mode != "error":
            st.caption(f"Note: {result.warning}")
else:
    st.caption("Pick a suggested question above or ask your own to run an investigation.")
