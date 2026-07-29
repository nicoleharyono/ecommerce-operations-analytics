"""Deterministic analytical tools for the AI Operations Investigator.

Every function here is a pure function over pandas DataFrames that are
already loaded (via data.py) and, for order-level data, already filtered to
the dashboard's current purchase-date range and delivery-outcome selection.
No network calls, no LLM calls, no Streamlit — this module is the boundary
between "the agent can call this" and "the numbers are computed here, once,
deterministically." The agent (agent.py) only ever reads what these
functions return; it never computes a statistic itself.

Seller and category data (seller_performance.csv / category_summary.csv)
are pre-aggregated across the full order history and are NOT scoped by the
dashboard's date or delivery-outcome filters, matching the same caveat
already surfaced on the dashboard's own seller/category charts.
"""

from __future__ import annotations

import pandas as pd


# ----------------------------------------------------------------------------
# Tool schemas (Claude tool-calling format)
# ----------------------------------------------------------------------------
TOOL_DEFINITIONS = {
    "get_kpi_summary": {
        "description": (
            "Return current headline KPIs — total orders, late delivery rate, "
            "average review score, median delivery days, median freight value — "
            "for the dashboard's current purchase-date range and delivery-outcome filter."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    "get_monthly_late_rate_trend": {
        "description": (
            "Return the month-by-month late-delivery rate within the current date range, "
            "the peak month, and a comparison of the most recent months against the prior period."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "recent_window_months": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of most-recent months to average for the recent-vs-prior comparison. Default 3.",
                }
            },
            "required": [],
        },
    },
    "get_handling_time_analysis": {
        "description": (
            "Return late-delivery rate broken down by seller handling-time group "
            "(0-1, 2-3, 4-7, 8+ days to ship), including the ratio between the slowest and fastest group."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    "get_distance_analysis": {
        "description": (
            "Return late-delivery rate broken down by shipping-distance group, "
            "including the ratio between the farthest and nearest group."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    "get_seller_performance_analysis": {
        "description": (
            "Return the top sellers by number of late orders, plus a count of high-risk "
            "sellers (late rate and order volume above threshold). Not scoped by the "
            "dashboard's date or delivery-outcome filters — covers all-time seller aggregates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_n": {"type": "integer", "minimum": 1, "maximum": 50, "description": "How many top sellers to return. Default 10."},
                "late_rate_threshold": {"type": "number", "description": "Late-rate fraction (0-1) above which a seller counts as high-risk. Default 0.15."},
                "min_orders": {"type": "integer", "minimum": 1, "description": "Minimum order count for a seller to count toward high-risk flagging. Default 5."},
            },
            "required": [],
        },
    },
    "get_category_performance_analysis": {
        "description": (
            "Return the top product categories by number of late orders, including "
            "pre-flagged high-late-risk, high-freight-burden, and low-review-score categories. "
            "Not scoped by the dashboard's date or delivery-outcome filters — covers all-time category aggregates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_n": {"type": "integer", "minimum": 1, "maximum": 50, "description": "How many top categories to return. Default 10."},
            },
            "required": [],
        },
    },
    "get_freight_cost_analysis": {
        "description": (
            "Return median freight value by order-weight group, the ratio between the "
            "heaviest and lightest group, and categories flagged with a high freight "
            "burden relative to item value."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    "get_review_score_impact_analysis": {
        "description": (
            "Return the average customer review score for on-time versus late deliveries "
            "and the size of the gap, to quantify the customer-satisfaction impact of lateness."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
}


# ----------------------------------------------------------------------------
# Tool implementations
# ----------------------------------------------------------------------------
def _round(x, ndigits=2):
    return None if pd.isna(x) else round(float(x), ndigits)


def kpi_summary(orders_df: pd.DataFrame, filters: dict) -> dict:
    delivered = orders_df[orders_df["order_status"] == "delivered"]
    late_rate = delivered["is_late"].mean() * 100 if len(delivered) else None
    return {
        "date_range": {"start": str(filters["start_date"]), "end": str(filters["end_date"])},
        "delivery_outcome_filter": filters["outcome"],
        "total_orders": int(len(orders_df)),
        "delivered_orders": int(len(delivered)),
        "late_orders": int(delivered["is_late"].sum()) if len(delivered) else 0,
        "late_delivery_rate_pct": _round(late_rate),
        "avg_review_score": _round(orders_df["review_score"].mean()),
        "median_delivery_days": _round(orders_df["delivery_days"].median(), 1),
        "median_freight_value_brl": _round(orders_df["total_freight_value"].median()),
    }


def monthly_late_rate_trend(monthly_df: pd.DataFrame, recent_window_months: int = 3) -> dict:
    if monthly_df.empty:
        return {"error": "No monthly data available for the current date range."}

    df = monthly_df.sort_values("purchase_month").reset_index(drop=True)
    months = [
        {
            "month": row["month_label"],
            "total_orders": int(row["total_orders"]),
            "late_delivery_rate_pct": _round(row["late_delivery_rate"]),
        }
        for _, row in df.iterrows()
    ]
    peak_row = df.loc[df["late_delivery_rate"].idxmax()]
    result = {
        "months": months,
        "peak_month": {"month": peak_row["month_label"], "late_delivery_rate_pct": _round(peak_row["late_delivery_rate"])},
    }

    n = min(recent_window_months, len(df) // 2) if len(df) >= 2 else 0
    if n >= 1:
        recent = df.tail(n)
        prior = df.iloc[-2 * n : -n]
        recent_avg = recent["late_delivery_rate"].mean()
        prior_avg = prior["late_delivery_rate"].mean()
        result["recent_vs_prior"] = {
            "recent_window_months": n,
            "recent_avg_late_rate_pct": _round(recent_avg),
            "prior_avg_late_rate_pct": _round(prior_avg),
            "change_pct_points": _round(recent_avg - prior_avg),
            "recent_months": list(recent["month_label"]),
            "prior_months": list(prior["month_label"]),
        }
    else:
        result["recent_vs_prior"] = {"error": "Not enough months in range for a recent-vs-prior comparison."}

    return result


def _group_late_rate(delivered: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        delivered.groupby(group_col, observed=True)
        .agg(late_rate=("is_late", "mean"), orders=("order_id", "count"))
        .reset_index()
    )


def handling_time_analysis(orders_df: pd.DataFrame) -> dict:
    delivered = orders_df[orders_df["order_status"] == "delivered"].dropna(subset=["handling_group"])
    if delivered.empty:
        return {"error": "No delivered orders with handling-time data for the current filters."}
    grp = _group_late_rate(delivered, "handling_group")
    grp["late_rate_pct"] = grp["late_rate"] * 100
    groups = [
        {"group": str(r.handling_group), "late_rate_pct": _round(r.late_rate_pct), "orders": int(r.orders)}
        for r in grp.itertuples()
    ]
    fastest = grp.iloc[0]
    slowest = grp.iloc[-1]
    multiplier = None if fastest.late_rate_pct == 0 else _round(slowest.late_rate_pct / fastest.late_rate_pct)
    return {
        "groups": groups,
        "fastest_group": {"group": str(fastest.handling_group), "late_rate_pct": _round(fastest.late_rate_pct)},
        "slowest_group": {"group": str(slowest.handling_group), "late_rate_pct": _round(slowest.late_rate_pct)},
        "slowest_to_fastest_rate_multiplier": multiplier,
    }


def distance_analysis(orders_df: pd.DataFrame) -> dict:
    delivered = orders_df[orders_df["order_status"] == "delivered"].dropna(subset=["distance_group"])
    if delivered.empty:
        return {"error": "No delivered orders with distance data for the current filters."}
    grp = _group_late_rate(delivered, "distance_group")
    grp["late_rate_pct"] = grp["late_rate"] * 100
    groups = [
        {"group": str(r.distance_group), "late_rate_pct": _round(r.late_rate_pct), "orders": int(r.orders)}
        for r in grp.itertuples()
    ]
    nearest = grp.iloc[0]
    farthest = grp.iloc[-1]
    multiplier = None if nearest.late_rate_pct == 0 else _round(farthest.late_rate_pct / nearest.late_rate_pct)
    return {
        "groups": groups,
        "nearest_group": {"group": str(nearest.distance_group), "late_rate_pct": _round(nearest.late_rate_pct)},
        "farthest_group": {"group": str(farthest.distance_group), "late_rate_pct": _round(farthest.late_rate_pct)},
        "farthest_to_nearest_rate_multiplier": multiplier,
    }


def seller_performance_analysis(
    sellers_df: pd.DataFrame, top_n: int = 10, late_rate_threshold: float = 0.15, min_orders: int = 5
) -> dict:
    top_n = max(1, min(int(top_n), 50))
    top = sellers_df.sort_values("late_orders", ascending=False).head(top_n)
    top_sellers = [
        {
            "seller_id": f"{r.seller_id[:8]}…",
            "order_count": int(r.order_count),
            "late_orders": int(r.late_orders),
            "late_rate_pct": _round(r.late_rate * 100),
            "median_handling_days": _round(r.median_handling_days, 1) if pd.notna(r.median_handling_days) else None,
            "state": r.seller_state if pd.notna(r.seller_state) else None,
        }
        for r in top.itertuples()
    ]
    high_risk = sellers_df[(sellers_df["late_rate"] >= late_rate_threshold) & (sellers_df["order_count"] >= min_orders)]
    return {
        "top_sellers_by_late_orders": top_sellers,
        "high_risk_threshold": {"late_rate_pct": _round(late_rate_threshold * 100), "min_orders": min_orders},
        "high_risk_seller_count": int(len(high_risk)),
        "total_sellers": int(len(sellers_df)),
        "seller_late_rate_median_pct": _round(sellers_df["late_rate"].median() * 100),
    }


def category_performance_analysis(categories_df: pd.DataFrame, top_n: int = 10) -> dict:
    top_n = max(1, min(int(top_n), 50))
    top = categories_df.sort_values("late_orders", ascending=False).head(top_n)
    top_categories = [
        {
            "category": r.category_label,
            "order_count": int(r.order_count),
            "late_orders": int(r.late_orders),
            "late_rate_pct": _round(r.late_rate * 100),
            "avg_review_score": _round(r.average_review_score),
            "high_late_risk": bool(r.high_late_risk),
            "high_freight_burden": bool(r.high_freight_burden),
            "low_review_score": bool(r.low_review_score),
        }
        for r in top.itertuples()
    ]
    return {
        "top_categories_by_late_orders": top_categories,
        "high_late_risk_category_count": int(categories_df["high_late_risk"].sum()),
        "high_freight_burden_category_count": int(categories_df["high_freight_burden"].sum()),
        "low_review_score_category_count": int(categories_df["low_review_score"].sum()),
        "total_categories": int(len(categories_df)),
    }


def freight_cost_analysis(orders_df: pd.DataFrame, categories_df: pd.DataFrame) -> dict:
    weight_df = orders_df.dropna(subset=["weight_group", "total_freight_value"])
    if weight_df.empty:
        return {"error": "No orders with weight and freight data for the current filters."}
    grp = (
        weight_df.groupby("weight_group", observed=True)
        .agg(median_freight=("total_freight_value", "median"), orders=("order_id", "count"))
        .reset_index()
    )
    groups = [
        {"group": str(r.weight_group), "median_freight_brl": _round(r.median_freight), "orders": int(r.orders)}
        for r in grp.itertuples()
    ]
    lightest = grp.iloc[0]
    heaviest = grp.iloc[-1]
    multiplier = None if lightest.median_freight == 0 else _round(heaviest.median_freight / lightest.median_freight)

    high_burden = categories_df[categories_df["high_freight_burden"]].sort_values("order_count", ascending=False)
    high_burden_categories = [
        {"category": r.category_label, "median_freight_ratio": _round(r.median_freight_ratio, 3), "order_count": int(r.order_count)}
        for r in high_burden.head(10).itertuples()
    ]

    return {
        "weight_groups": groups,
        "lightest_group": {"group": str(lightest.weight_group), "median_freight_brl": _round(lightest.median_freight)},
        "heaviest_group": {"group": str(heaviest.weight_group), "median_freight_brl": _round(heaviest.median_freight)},
        "heaviest_to_lightest_freight_multiplier": multiplier,
        "high_freight_burden_categories": high_burden_categories,
    }


def review_score_impact_analysis(orders_df: pd.DataFrame) -> dict:
    delivered = orders_df[orders_df["order_status"] == "delivered"].dropna(subset=["review_score"])
    if delivered.empty:
        return {"error": "No delivered orders with review scores for the current filters."}
    grp = delivered.groupby("is_late")["review_score"].agg(["mean", "count"])
    on_time_mean = grp["mean"].get(False)
    late_mean = grp["mean"].get(True)
    on_time_n = int(grp["count"].get(False, 0))
    late_n = int(grp["count"].get(True, 0))
    gap = None if pd.isna(on_time_mean) or pd.isna(late_mean) else _round(on_time_mean - late_mean)
    return {
        "on_time_avg_review_score": _round(on_time_mean) if pd.notna(on_time_mean) else None,
        "on_time_sample_size": on_time_n,
        "late_avg_review_score": _round(late_mean) if pd.notna(late_mean) else None,
        "late_sample_size": late_n,
        "review_score_gap": gap,
    }


# ----------------------------------------------------------------------------
# Tool registry factory
# ----------------------------------------------------------------------------
def build_tools(
    orders_df: pd.DataFrame,
    sellers_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    filters: dict,
) -> dict:
    """Bind the tool implementations to the dashboard's current data/filters.

    Returns {tool_name: {"spec": <claude tool schema>, "fn": <callable(**kwargs) -> dict>}}.
    """
    fns = {
        "get_kpi_summary": lambda: kpi_summary(orders_df, filters),
        "get_monthly_late_rate_trend": lambda recent_window_months=3: monthly_late_rate_trend(
            monthly_df, recent_window_months
        ),
        "get_handling_time_analysis": lambda: handling_time_analysis(orders_df),
        "get_distance_analysis": lambda: distance_analysis(orders_df),
        "get_seller_performance_analysis": lambda top_n=10, late_rate_threshold=0.15, min_orders=5: seller_performance_analysis(
            sellers_df, top_n, late_rate_threshold, min_orders
        ),
        "get_category_performance_analysis": lambda top_n=10: category_performance_analysis(categories_df, top_n),
        "get_freight_cost_analysis": lambda: freight_cost_analysis(orders_df, categories_df),
        "get_review_score_impact_analysis": lambda: review_score_impact_analysis(orders_df),
    }
    return {
        name: {"spec": {"name": name, **TOOL_DEFINITIONS[name]}, "fn": fn}
        for name, fn in fns.items()
    }
