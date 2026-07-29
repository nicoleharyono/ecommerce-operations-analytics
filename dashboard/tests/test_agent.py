"""Tests for the agent orchestration layer: demo mode and the
numeric-support guard that catches unsupported/fabricated statistics.
"""

import datetime

import pytest

import agent
import agent_tools
import data

FILTERS = {"start_date": datetime.date(2016, 1, 1), "end_date": datetime.date(2018, 12, 31), "outcome": "All"}


@pytest.fixture(scope="module")
def dfs():
    return {
        "orders": data.load_orders(),
        "sellers": data.load_sellers(),
        "categories": data.load_categories(),
        "monthly": data.load_monthly(),
    }


def test_get_api_key_returns_none_without_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert agent.get_api_key() is None


def test_get_api_key_reads_env_var(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    assert agent.get_api_key() == "sk-test-123"


def test_investigate_falls_back_to_demo_mode_without_key(dfs, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = agent.investigate(
        "Why did late deliveries increase?", dfs["orders"], dfs["sellers"], dfs["categories"], dfs["monthly"], FILTERS
    )
    assert result.mode == "demo"
    assert result.tool_calls  # at least one tool was actually run
    assert "Executive Diagnosis" in result.sections
    assert "Evidence" in result.sections


def test_investigate_force_demo_ignores_env_key(dfs, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-be-ignored")
    result = agent.investigate(
        "Generate an executive operations brief.",
        dfs["orders"],
        dfs["sellers"],
        dfs["categories"],
        dfs["monthly"],
        FILTERS,
        force_demo=True,
    )
    assert result.mode == "demo"


@pytest.mark.parametrize(
    "question,expected_tool",
    [
        ("Why did late deliveries increase?", "get_monthly_late_rate_trend"),
        ("Which sellers should operations review first?", "get_seller_performance_analysis"),
        ("Where can freight costs be reduced?", "get_freight_cost_analysis"),
        ("What is hurting customer satisfaction?", "get_review_score_impact_analysis"),
        ("Generate an executive operations brief.", "get_kpi_summary"),
    ],
)
def test_demo_routing_selects_relevant_tools(dfs, question, expected_tool, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = agent.investigate(question, dfs["orders"], dfs["sellers"], dfs["categories"], dfs["monthly"], FILTERS)
    called = {tc.name for tc in result.tool_calls}
    assert expected_tool in called


def test_demo_mode_never_calls_the_network(dfs, monkeypatch):
    # If demo mode ever imported/called anthropic, this would fail loudly
    # since no API key is configured and no network access is mocked.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = agent.investigate(
        "Generate an executive operations brief.",
        dfs["orders"],
        dfs["sellers"],
        dfs["categories"],
        dfs["monthly"],
        FILTERS,
    )
    assert result.mode == "demo"
    assert len(result.tool_calls) == 8  # the "brief" route calls every tool


def test_off_topic_question_falls_back_to_full_brief(dfs, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = agent.investigate(
        "What's the weather like today?", dfs["orders"], dfs["sellers"], dfs["categories"], dfs["monthly"], FILTERS
    )
    # no keyword matched -> falls back to the comprehensive route (all 8 tools)
    assert len(result.tool_calls) == 8


# ----------------------------------------------------------------------------
# validate_no_unsupported_numbers — the guard against fabricated statistics
# ----------------------------------------------------------------------------
def test_validate_passes_when_every_number_is_grounded():
    tool_output = {"late_delivery_rate_pct": 24.7, "orders": 1532, "group": "8+ days"}
    answer = "Orders taking 8+ days show a 24.7% late rate across 1532 orders."
    assert agent.validate_no_unsupported_numbers(answer, [tool_output]) == []


def test_validate_catches_a_fabricated_number():
    tool_output = {"late_delivery_rate_pct": 24.7, "orders": 1532}
    answer = "Orders taking 8+ days show a 41.2% late rate, dramatically higher than normal."
    unsupported = agent.validate_no_unsupported_numbers(answer, [tool_output])
    assert "41.2" in unsupported


def test_validate_ignores_small_counting_numbers():
    tool_output = {"top_sellers_by_late_orders": [{"late_orders": 42}]}
    answer = "There are 3 sellers worth reviewing first."
    # "3" is a small derived count, not expected to trace to a literal tool value
    assert agent.validate_no_unsupported_numbers(answer, [tool_output]) == []


def test_validate_across_multiple_tool_outputs():
    outputs = [{"a": 15.1}, {"b": 49.97}]
    answer = "Freight rises from R$ 15.1 to R$ 49.97 for heavier orders."
    assert agent.validate_no_unsupported_numbers(answer, outputs) == []


def test_demo_mode_report_never_cites_unsupported_numbers(dfs, monkeypatch):
    """End-to-end guard: every number the demo-mode report cites must trace
    back to the tool outputs it actually gathered."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for question in [
        "Why did late deliveries increase?",
        "Which sellers should operations review first?",
        "Where can freight costs be reduced?",
        "What is hurting customer satisfaction?",
        "Generate an executive operations brief.",
    ]:
        result = agent.investigate(
            question, dfs["orders"], dfs["sellers"], dfs["categories"], dfs["monthly"], FILTERS
        )
        tool_outputs = [tc.output for tc in result.tool_calls]
        unsupported = agent.validate_no_unsupported_numbers(result.answer, tool_outputs)
        assert unsupported == [], f"Unsupported numbers in demo report for {question!r}: {unsupported}"


def test_parse_sections_splits_headers():
    text = (
        "Executive Diagnosis:\nHandling time is the main driver.\n\n"
        "Evidence:\n- 24.7% vs 4.6%\n\n"
        "Recommended Actions:\n- Escalate slow sellers\n\n"
        "Metrics to Monitor:\n- Late rate by handling group"
    )
    sections = agent._parse_sections(text)
    assert sections["Executive Diagnosis"] == "Handling time is the main driver."
    assert "24.7% vs 4.6%" in sections["Evidence"]
    assert set(sections.keys()) == {"Executive Diagnosis", "Evidence", "Recommended Actions", "Metrics to Monitor"}


def test_parse_sections_falls_back_when_headers_missing():
    text = "Just a plain answer with no section headers."
    sections = agent._parse_sections(text)
    assert sections == {"Report": text}


# ----------------------------------------------------------------------------
# "Why did late deliveries increase?" — dedicated increase/decrease logic
# ----------------------------------------------------------------------------
def test_late_increase_report_states_decrease_on_real_data(dfs):
    # Over the dashboard's full default date range, the last 3 months (Jun-Aug 2018,
    # ~3.6% avg) are well below the prior 3 months (Mar-May 2018, ~10.0% avg, which
    # includes the Mar 2018 peak) — the report must say so plainly, not claim an increase.
    tool_impls = agent_tools.build_tools(dfs["orders"], dfs["sellers"], dfs["categories"], dfs["monthly"], FILTERS)
    result = agent._build_late_increase_report(tool_impls)
    diagnosis = result.sections["Executive Diagnosis"]
    assert "did NOT increase" in diagnosis
    assert "have increased" not in diagnosis
    assert "Mar 2018" in result.answer  # falls back to the largest historical spike


def test_late_increase_report_detects_a_synthetic_increase(dfs):
    import pandas as pd

    rising_monthly = pd.DataFrame(
        {
            "purchase_month": ["2018-01", "2018-02", "2018-03", "2018-04", "2018-05", "2018-06"],
            "month_label": ["Jan 2018", "Feb 2018", "Mar 2018", "Apr 2018", "May 2018", "Jun 2018"],
            "total_orders": [500] * 6,
            "late_delivery_rate": [2.0, 2.0, 2.0, 10.0, 12.0, 14.0],
        }
    )
    tool_impls = agent_tools.build_tools(dfs["orders"], dfs["sellers"], dfs["categories"], rising_monthly, FILTERS)
    result = agent._build_late_increase_report(tool_impls)
    diagnosis = result.sections["Executive Diagnosis"]
    assert "have increased" in diagnosis
    assert "did NOT increase" not in diagnosis

    # Still fully grounded: every number traces back to a tool output.
    tool_outputs = [tc.output for tc in result.tool_calls]
    assert agent.validate_no_unsupported_numbers(result.answer, tool_outputs) == []


def test_late_increase_report_labels_hypotheses_and_correlation(dfs):
    tool_impls = agent_tools.build_tools(dfs["orders"], dfs["sellers"], dfs["categories"], dfs["monthly"], FILTERS)
    result = agent._build_late_increase_report(tool_impls)
    text = result.answer.lower()
    # Seasonality/carrier/capacity must appear only as hypotheses, never as established causes.
    assert "hypothes" in text
    assert "not confirmed" in text or "untested hypotheses" in text
    # Handling time / distance must be framed as correlation, not causation.
    assert "correlat" in text or "associat" in text
    assert "not a proven cause" in text or "does not prove" in text


def test_late_increase_report_handles_insufficient_monthly_data(dfs):
    import pandas as pd

    empty_monthly = pd.DataFrame(columns=["purchase_month", "month_label", "total_orders", "late_delivery_rate"])
    tool_impls = agent_tools.build_tools(dfs["orders"], dfs["sellers"], dfs["categories"], empty_monthly, FILTERS)
    result = agent._build_late_increase_report(tool_impls)
    assert "isn't enough monthly data" in result.sections["Executive Diagnosis"].lower()
    assert "have increased" not in result.sections["Executive Diagnosis"]
    assert "did NOT increase" not in result.sections["Executive Diagnosis"]


def test_investigate_routes_late_question_to_dedicated_builder(dfs, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = agent.investigate(
        "Why did late deliveries increase?", dfs["orders"], dfs["sellers"], dfs["categories"], dfs["monthly"], FILTERS
    )
    called = [tc.name for tc in result.tool_calls]
    assert called == ["get_monthly_late_rate_trend", "get_handling_time_analysis", "get_distance_analysis"]
