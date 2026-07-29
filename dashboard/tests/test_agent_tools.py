"""Tests for the deterministic analytical tools.

These check tool outputs against known ground-truth figures already
validated against the source notebook (notebooks/02_dashboard_and_recommendations.ipynb)
and against the live dashboard's own charts, so a future data or logic
change that silently shifts these numbers gets caught here.
"""

import datetime

import pytest

import agent_tools
import data

FILTERS = {"start_date": datetime.date(2016, 1, 1), "end_date": datetime.date(2018, 12, 31), "outcome": "All"}


@pytest.fixture(scope="module")
def orders_df():
    return data.load_orders()


@pytest.fixture(scope="module")
def sellers_df():
    return data.load_sellers()


@pytest.fixture(scope="module")
def categories_df():
    return data.load_categories()


@pytest.fixture(scope="module")
def monthly_df():
    return data.load_monthly()


def test_kpi_summary_matches_known_totals(orders_df):
    result = agent_tools.kpi_summary(orders_df, FILTERS)
    assert result["total_orders"] == 99441
    assert result["delivered_orders"] == 96478
    assert result["late_delivery_rate_pct"] == pytest.approx(6.77, abs=0.05)
    assert result["median_delivery_days"] == pytest.approx(10.0, abs=0.5)
    assert result["median_freight_value_brl"] == pytest.approx(17.17, abs=0.5)


def test_kpi_summary_reflects_date_range_filter(orders_df):
    narrow_filters = {"start_date": datetime.date(2018, 1, 1), "end_date": datetime.date(2018, 1, 31), "outcome": "All"}
    narrow_df = orders_df[
        (orders_df["purchase_date"] >= narrow_filters["start_date"]) & (orders_df["purchase_date"] <= narrow_filters["end_date"])
    ]
    result = agent_tools.kpi_summary(narrow_df, narrow_filters)
    assert result["total_orders"] < 99441
    assert result["date_range"] == {"start": "2018-01-01", "end": "2018-01-31"}


def test_handling_time_analysis_matches_notebook_figures(orders_df):
    result = agent_tools.handling_time_analysis(orders_df)
    assert result["fastest_group"]["group"] == "0-1 days"
    assert result["fastest_group"]["late_rate_pct"] == pytest.approx(4.6, abs=0.2)
    assert result["slowest_group"]["group"] == "8+ days"
    assert result["slowest_group"]["late_rate_pct"] == pytest.approx(24.7, abs=0.5)
    assert result["slowest_to_fastest_rate_multiplier"] > 1


def test_distance_analysis_matches_notebook_figures(orders_df):
    result = agent_tools.distance_analysis(orders_df)
    assert result["nearest_group"]["group"] == "≤250 km"
    assert result["nearest_group"]["late_rate_pct"] == pytest.approx(4.6, abs=0.2)
    assert result["farthest_group"]["group"] == "1,000+ km"
    assert result["farthest_group"]["late_rate_pct"] == pytest.approx(10.4, abs=0.3)


def test_freight_cost_analysis_matches_notebook_figures(orders_df, categories_df):
    result = agent_tools.freight_cost_analysis(orders_df, categories_df)
    assert result["lightest_group"]["group"] == "<0.5 kg"
    assert result["lightest_group"]["median_freight_brl"] == pytest.approx(15.10, abs=0.5)
    assert result["heaviest_group"]["group"] == "10+ kg"
    assert result["heaviest_group"]["median_freight_brl"] == pytest.approx(49.97, abs=1.0)
    assert result["heaviest_to_lightest_freight_multiplier"] > 3


def test_review_score_impact_matches_notebook_figures(orders_df):
    result = agent_tools.review_score_impact_analysis(orders_df)
    assert result["on_time_avg_review_score"] == pytest.approx(4.29, abs=0.1)
    assert result["late_avg_review_score"] == pytest.approx(2.27, abs=0.1)
    assert result["review_score_gap"] > 1.5
    assert result["on_time_sample_size"] > result["late_sample_size"]


def test_category_performance_flags_known_top_categories(categories_df):
    result = agent_tools.category_performance_analysis(categories_df, top_n=5)
    top_names = {c["category"] for c in result["top_categories_by_late_orders"]}
    assert "Bed Bath Table" in top_names
    assert "Health Beauty" in top_names
    assert result["total_categories"] == len(categories_df)


def test_category_performance_respects_top_n(categories_df):
    result = agent_tools.category_performance_analysis(categories_df, top_n=3)
    assert len(result["top_categories_by_late_orders"]) == 3


def test_seller_performance_analysis_structure(sellers_df):
    result = agent_tools.seller_performance_analysis(sellers_df, top_n=10, late_rate_threshold=0.15, min_orders=5)
    assert len(result["top_sellers_by_late_orders"]) == 10
    assert result["total_sellers"] == len(sellers_df)
    assert result["high_risk_seller_count"] >= 0
    # sorted descending by late order count
    late_orders = [s["late_orders"] for s in result["top_sellers_by_late_orders"]]
    assert late_orders == sorted(late_orders, reverse=True)


def test_monthly_late_rate_trend_finds_march_2018_peak(monthly_df):
    result = agent_tools.monthly_late_rate_trend(monthly_df)
    assert result["peak_month"]["month"] == "Mar 2018"
    assert result["peak_month"]["late_delivery_rate_pct"] == pytest.approx(19.0, abs=0.5)
    assert "recent_vs_prior" in result


def test_monthly_late_rate_trend_handles_empty_range():
    import pandas as pd

    empty = pd.DataFrame(columns=["purchase_month", "month_label", "total_orders", "late_delivery_rate"])
    result = agent_tools.monthly_late_rate_trend(empty)
    assert "error" in result


def test_build_tools_returns_all_eight(orders_df, sellers_df, categories_df, monthly_df):
    tools = agent_tools.build_tools(orders_df, sellers_df, categories_df, monthly_df, FILTERS)
    assert set(tools.keys()) == set(agent_tools.TOOL_DEFINITIONS.keys())
    assert len(tools) == 8
    for name, impl in tools.items():
        assert impl["spec"]["name"] == name
        assert "description" in impl["spec"]
        assert "input_schema" in impl["spec"]
        output = impl["fn"]()
        assert isinstance(output, dict)
