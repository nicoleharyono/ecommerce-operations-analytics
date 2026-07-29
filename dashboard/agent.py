"""AI Operations Investigator — agent orchestration.

Two modes:
- LLM mode: a manual Claude tool-use loop over the deterministic tools in
  agent_tools.py. Used whenever an Anthropic API key is available.
- Demo mode: a deterministic, keyword-routed report built from the exact
  same tool functions, with zero network calls. Used whenever no API key is
  configured, so the portfolio app works without one.

Either way, every number in the final report traces back to a tool output —
the agent (LLM or rule-based) never computes a statistic itself.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import agent_tools

MODEL = "claude-opus-4-8"
MAX_ITERATIONS = 4

SECTION_HEADERS = ["Executive Diagnosis", "Evidence", "Recommended Actions", "Metrics to Monitor"]

SYSTEM_PROMPT_TEMPLATE = """You are the AI Operations Investigator embedded in the Fulfillment Control Tower \
dashboard, an e-commerce fulfillment analytics tool.

Current dashboard scope: purchase dates {start} to {end}; delivery outcome filter = {outcome}. \
Tools scoped to order-level data automatically respect this range; seller and category tools \
cover all-time aggregates regardless of this filter (the same scoping the dashboard's own \
seller/category charts use).

You have eight deterministic analytical tools backed by real operational data (KPI summary, \
monthly late-rate trend, handling-time analysis, distance analysis, seller performance, category \
performance, freight-cost analysis, review-score impact). You have no access to raw data any other way.

Your job, for every question:
1. Interpret what the user is actually asking.
2. Call only the tools relevant to the question — not all of them by default.
3. When useful, call multiple tools, or the same tool with different parameters, to compare \
segments or time periods (e.g. peak month vs. baseline, heavy vs. light orders, on-time vs. late).
4. Identify and rank the most likely drivers based on what the tool data actually shows — the \
size of a late-rate gap or a freight multiplier is your evidence for ranking, not intuition.
5. If the available tools cannot answer the question (it asks about something outside \
fulfillment/freight/review data), say so plainly instead of guessing.

Special procedure for "did X increase / get worse" questions (e.g. "why did late deliveries \
increase?"): before diagnosing anything, call get_monthly_late_rate_trend and read its \
recent_vs_prior comparison first.
- If the recent average is higher than the prior average, state the increase explicitly (with \
both numbers and the change) and then investigate likely drivers for that recent period.
- If the recent average is equal to or lower than the prior average, say explicitly that late \
deliveries did NOT increase in the most recent period — do not describe a downward or flat trend \
as an increase. Then identify the largest historical spike (the peak month) in the selected range \
and analyze the strongest associated drivers for that instead.
- If there isn't enough monthly data to compare a recent period against a prior one, say so, and \
fall back to reporting the peak month alone.

When you have gathered enough evidence, write your final answer as plain text with exactly \
these four sections, in this order, each heading on its own line with nothing else on that line:

Executive Diagnosis:
2-4 sentences stating the most likely cause(s), in plain business language.

Evidence:
A bulleted list. Every bullet must cite a specific number that came from a tool result, e.g. \
"24.7% late rate for 8+ day handling vs 4.6% for 0-1 day (5.4x)".

Recommended Actions:
A bulleted list of concrete operational actions, grounded in the evidence above.

Metrics to Monitor:
A bulleted list of specific metrics or segments to track going forward.

Rules:
- Every number in your answer must come from a tool result. Never estimate, round differently \
than the tool output, or invent a statistic.
- Never attribute a cause the tool data doesn't support.
- If a tool returns no data for the current filter scope, say so in Evidence rather than \
omitting it silently.
- No tool measures seasonality, carrier performance, or fulfillment-capacity constraints. Never \
state any of those as an established cause of a trend or spike. If you raise them at all, label \
them explicitly as hypotheses to investigate, not conclusions.
- Handling time and shipping distance are correlated with late-delivery rate in this data, not \
proven causes of it. Describe them as "associated with" or "correlated with" lateness — never as \
"causing" or "driving" it — and note that correlation, not causation, is what the tools establish.
"""


@dataclass
class ToolCallRecord:
    name: str
    input: dict
    output: dict


@dataclass
class InvestigationResult:
    answer: str
    sections: dict
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    mode: str = "llm"  # "llm", "demo", or "error"
    warning: str | None = None


def get_api_key() -> str | None:
    """Streamlit secrets first, then the ANTHROPIC_API_KEY environment variable."""
    try:
        import streamlit as st

        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


def _parse_sections(text: str) -> dict:
    pattern = r"^\s*#{0,3}\s*(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*:?\s*$"
    lines = text.splitlines()
    matches = [(i, line) for i, line in enumerate(lines) if re.match(pattern, line, flags=re.IGNORECASE)]
    if not matches:
        return {"Report": text.strip()}

    sections = {}
    for idx, (line_no, raw_header) in enumerate(matches):
        header = next(h for h in SECTION_HEADERS if h.lower() == re.sub(r"[#:\s]+$", "", raw_header).strip("# ").lower())
        end = matches[idx + 1][0] if idx + 1 < len(matches) else len(lines)
        sections[header] = "\n".join(lines[line_no + 1 : end]).strip()
    return sections


def _flatten_numbers(obj) -> set[str]:
    """Collect every number that appears anywhere in a tool output (values or embedded in strings)."""
    found: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, bool):
            return
        elif isinstance(o, (int, float)):
            found.add(_norm_number(o))
        elif isinstance(o, str):
            for m in re.findall(r"-?\d[\d,]*\.?\d*", o):
                found.add(_norm_number(m))

    walk(obj)
    return found


def _norm_number(x) -> str:
    s = str(x).replace(",", "")
    try:
        f = float(s)
    except ValueError:
        return s
    if f == int(f):
        return str(int(f))
    return f"{f:.1f}".rstrip("0").rstrip(".") if f"{f:.1f}" != f"{f}" else str(f)


def validate_no_unsupported_numbers(answer_text: str, tool_outputs: list[dict]) -> list[str]:
    """Return numeric tokens in `answer_text` that don't trace back to any tool output.

    Small counting numbers (0-12) are always considered safe, since they commonly
    arise from list lengths / bullet counts rather than being cited statistics.
    """
    supported: set[str] = set()
    for out in tool_outputs:
        supported |= _flatten_numbers(out)

    unsupported = []
    for raw in re.findall(r"-?\d[\d,]*\.?\d*", answer_text):
        norm = _norm_number(raw)
        try:
            if 0 <= float(norm) <= 12:
                continue
        except ValueError:
            pass
        if norm not in supported:
            unsupported.append(raw)
    return unsupported


# ----------------------------------------------------------------------------
# Demo mode (no API key) — deterministic, keyword-routed, zero network calls
# ----------------------------------------------------------------------------
_LATE_INCREASE_KEYWORDS = ("late", "delay", "increas", "rose", "spike", "worse")

_HYPOTHESIS_CAVEAT = (
    "Seasonal demand, carrier disruptions, and capacity constraints are plausible hypotheses for "
    "month-to-month swings, but no tool in this dashboard measures them directly — treat them as "
    "hypotheses to investigate, not established causes."
)

_KEYWORD_ROUTES = {
    ("seller",): ["get_seller_performance_analysis"],
    ("freight", "shipping cost", "expensive"): ["get_freight_cost_analysis"],
    ("satisfaction", "review score", "customer rating", "complain", "happy"): [
        "get_review_score_impact_analysis",
        "get_category_performance_analysis",
    ],
    ("brief", "executive", "overview", "summary"): list(agent_tools.TOOL_DEFINITIONS.keys()),
}

_EVIDENCE_BUILDERS = {
    "get_kpi_summary": lambda o: [
        f"Current scope ({o['date_range']['start']} to {o['date_range']['end']}, outcome={o['delivery_outcome_filter']}): "
        f"{o['total_orders']} orders, {o['late_delivery_rate_pct']}% late delivery rate, "
        f"{o['avg_review_score']} average review score, {o['median_delivery_days']}-day median delivery, "
        f"R$ {o['median_freight_value_brl']} median freight."
    ],
    "get_monthly_late_rate_trend": lambda o: (
        [f"Peak month: {o['peak_month']['month']} at {o['peak_month']['late_delivery_rate_pct']}% late rate."]
        + (
            [
                f"Last {o['recent_vs_prior']['recent_window_months']} months average "
                f"{o['recent_vs_prior']['recent_avg_late_rate_pct']}% late rate vs "
                f"{o['recent_vs_prior']['prior_avg_late_rate_pct']}% in the prior period "
                f"({o['recent_vs_prior']['change_pct_points']:+} pct points)."
            ]
            if "change_pct_points" in o.get("recent_vs_prior", {})
            else [o.get("recent_vs_prior", {}).get("error", "")]
        )
    ),
    "get_handling_time_analysis": lambda o: [
        f"Handling time: {o['slowest_group']['group']} orders show a {o['slowest_group']['late_rate_pct']}% late rate "
        f"vs {o['fastest_group']['late_rate_pct']}% for {o['fastest_group']['group']} "
        f"({o['slowest_to_fastest_rate_multiplier']}x)."
    ],
    "get_distance_analysis": lambda o: [
        f"Distance: {o['farthest_group']['group']} routes show a {o['farthest_group']['late_rate_pct']}% late rate "
        f"vs {o['nearest_group']['late_rate_pct']}% for {o['nearest_group']['group']} "
        f"({o['farthest_to_nearest_rate_multiplier']}x)."
    ],
    "get_seller_performance_analysis": lambda o: [
        f"{o['high_risk_seller_count']} of {o['total_sellers']} sellers are high-risk "
        f"(late rate ≥ {o['high_risk_threshold']['late_rate_pct']}%, {o['high_risk_threshold']['min_orders']}+ orders).",
    ]
    + [
        f"Top late-order seller: {s['seller_id']} — {s['late_orders']} late orders "
        f"({s['late_rate_pct']}% late rate, {s['order_count']} total orders)."
        for s in o["top_sellers_by_late_orders"][:1]
    ],
    "get_category_performance_analysis": lambda o: [
        f"{o['high_late_risk_category_count']} of {o['total_categories']} categories are flagged high-late-risk; "
        f"{o['low_review_score_category_count']} are flagged low-review-score."
    ]
    + [
        f"Top late-order category: {c['category']} — {c['late_orders']} late orders "
        f"({c['late_rate_pct']}% late rate, avg review {c['avg_review_score']})."
        for c in o["top_categories_by_late_orders"][:1]
    ],
    "get_freight_cost_analysis": lambda o: [
        f"Freight: {o['heaviest_group']['group']} orders have a median freight of "
        f"R$ {o['heaviest_group']['median_freight_brl']} vs R$ {o['lightest_group']['median_freight_brl']} "
        f"for {o['lightest_group']['group']} ({o['heaviest_to_lightest_freight_multiplier']}x)."
    ],
    "get_review_score_impact_analysis": lambda o: [
        f"On-time deliveries average a {o['on_time_avg_review_score']} review score (n={o['on_time_sample_size']}) "
        f"vs {o['late_avg_review_score']} for late deliveries (n={o['late_sample_size']}) — "
        f"a gap of {o['review_score_gap']} points."
    ],
}

_ACTION_BUILDERS = {
    "get_monthly_late_rate_trend": ["Investigate the peak month for seasonal demand, capacity, or carrier disruptions."],
    "get_handling_time_analysis": ["Flag orders unshipped after 3 days and escalate sellers with slow median handling time."],
    "get_distance_analysis": ["Extend promised delivery windows on long routes and prefer closer sellers when available."],
    "get_seller_performance_analysis": ["Prioritize operational review of the highest-late-order-count sellers, weighted by volume."],
    "get_category_performance_analysis": ["Run targeted seller and handling-time reviews in the flagged high-risk categories."],
    "get_freight_cost_analysis": ["Review packaging efficiency and freight-subsidy rules for heavy or low-value products."],
    "get_review_score_impact_analysis": ["Treat on-time delivery as a customer-satisfaction lever, not just an ops metric."],
}

_METRIC_BUILDERS = {
    "get_monthly_late_rate_trend": ["Monthly late-delivery rate"],
    "get_handling_time_analysis": ["Late rate by handling-time group"],
    "get_distance_analysis": ["Late rate by distance group"],
    "get_seller_performance_analysis": ["High-risk seller count", "Top sellers by late orders"],
    "get_category_performance_analysis": ["High-late-risk category count", "Top categories by late orders"],
    "get_freight_cost_analysis": ["Median freight value by weight group"],
    "get_review_score_impact_analysis": ["Review-score gap between on-time and late deliveries"],
}


def _route_demo_tools(question: str) -> list[str]:
    q = question.lower()
    matched: list[str] = []
    for keywords, tools in _KEYWORD_ROUTES.items():
        if any(kw in q for kw in keywords):
            for t in tools:
                if t not in matched:
                    matched.append(t)
    if not matched:
        matched = list(agent_tools.TOOL_DEFINITIONS.keys())
    return matched


def _build_late_increase_report(tool_impls: dict) -> InvestigationResult:
    """Dedicated logic for "did late deliveries increase" style questions.

    Always checks the recent-vs-prior comparison first and states plainly whether
    an increase actually happened before looking for drivers, rather than assuming
    an increase because the question asked about one.
    """
    tool_calls: list[ToolCallRecord] = []

    def call(name: str) -> dict:
        output = tool_impls[name]["fn"]()
        tool_calls.append(ToolCallRecord(name=name, input={}, output=output))
        return output

    trend = call("get_monthly_late_rate_trend")
    handling = call("get_handling_time_analysis")
    distance = call("get_distance_analysis")

    evidence: list[str] = []
    rvp = trend.get("recent_vs_prior", {})
    increased = None  # True / False / None (insufficient data)

    if "change_pct_points" in rvp:
        increased = rvp["change_pct_points"] > 0
        evidence.append(
            f"Last {rvp['recent_window_months']} months averaged a {rvp['recent_avg_late_rate_pct']}% late rate "
            f"vs {rvp['prior_avg_late_rate_pct']}% in the prior {rvp['recent_window_months']} months "
            f"({rvp['change_pct_points']:+} pct points)."
        )
    else:
        evidence.append(rvp.get("error", "Not enough monthly data to compare a recent period against a prior one."))

    if "peak_month" in trend:
        evidence.append(
            f"Largest late-rate spike in the selected range: {trend['peak_month']['month']} at "
            f"{trend['peak_month']['late_delivery_rate_pct']}%."
        )

    if "slowest_group" in handling:
        evidence.append(
            f"Handling time is associated with lateness: {handling['slowest_group']['group']} orders show a "
            f"{handling['slowest_group']['late_rate_pct']}% late rate vs {handling['fastest_group']['late_rate_pct']}% "
            f"for {handling['fastest_group']['group']} ({handling['slowest_to_fastest_rate_multiplier']}x) — "
            "a correlation, not a proven cause."
        )
    elif "error" in handling:
        evidence.append(handling["error"])

    if "farthest_group" in distance:
        evidence.append(
            f"Distance is associated with lateness: {distance['farthest_group']['group']} routes show a "
            f"{distance['farthest_group']['late_rate_pct']}% late rate vs {distance['nearest_group']['late_rate_pct']}% "
            f"for {distance['nearest_group']['group']} ({distance['farthest_to_nearest_rate_multiplier']}x) — "
            "a correlation, not a proven cause."
        )
    elif "error" in distance:
        evidence.append(distance["error"])

    evidence.append(_HYPOTHESIS_CAVEAT)

    hypothesis_action = (
        "Directly investigate seasonal demand, carrier performance, and capacity as hypotheses — "
        "they are not confirmed by this data."
    )
    handling_action = "Flag orders unshipped after 3 days and escalate sellers with slow median handling time."
    distance_action = "Extend promised delivery windows on long routes and prefer closer sellers when available."

    if increased is True:
        diagnosis = (
            f"Late deliveries have increased in the most recent period: the last {rvp['recent_window_months']} months "
            f"averaged {rvp['recent_avg_late_rate_pct']}% vs {rvp['prior_avg_late_rate_pct']}% before "
            f"({rvp['change_pct_points']:+} pct points). Handling time and shipping distance are the strongest "
            "measured correlates of lateness in this data, though that does not prove they caused the recent "
            "increase; seasonal demand, carrier disruptions, and capacity constraints remain untested hypotheses."
        )
        actions = [
            "Prioritize root-cause investigation now — the trend is currently worsening.",
            handling_action,
            distance_action,
            hypothesis_action,
        ]
    elif increased is False:
        peak_note = (
            f" the largest historical spike in the selected range was {trend['peak_month']['month']} at "
            f"{trend['peak_month']['late_delivery_rate_pct']}%."
            if "peak_month" in trend
            else ""
        )
        diagnosis = (
            f"Late deliveries did NOT increase in the most recent period — the last {rvp['recent_window_months']} "
            f"months averaged {rvp['recent_avg_late_rate_pct']}% vs {rvp['prior_avg_late_rate_pct']}% before "
            f"({rvp['change_pct_points']:+} pct points). Instead,{peak_note} Handling time and shipping distance "
            "are the strongest measured correlates of lateness overall, though that does not prove they caused "
            "that spike; seasonal demand, carrier disruptions, and capacity constraints remain untested hypotheses."
        )
        actions = [
            "No active worsening trend — treat this as ongoing monitoring, not incident response.",
        ] + (
            [f"Review what happened around {trend['peak_month']['month']} specifically, since it remains the largest recorded spike."]
            if "peak_month" in trend
            else []
        ) + [handling_action, distance_action, hypothesis_action]
    else:
        diagnosis = (
            "There isn't enough monthly data in the current filter range to compare a recent period against a "
            "prior one. "
            + (
                f"The largest late-rate month on record here is {trend['peak_month']['month']} at "
                f"{trend['peak_month']['late_delivery_rate_pct']}%. "
                if "peak_month" in trend
                else ""
            )
            + "Handling time and shipping distance are the strongest measured correlates of lateness in this data, "
            "though that does not prove causation; seasonal demand, carrier disruptions, and capacity constraints "
            "remain untested hypotheses."
        )
        actions = [
            "Widen the purchase-date filter to get enough months for a recent-vs-prior comparison.",
            handling_action,
            hypothesis_action,
        ]

    metrics = [
        "Monthly late-delivery rate (recent vs. prior period)",
        "Late rate by handling-time group",
        "Late rate by distance group",
    ]

    sections = {
        "Executive Diagnosis": diagnosis,
        "Evidence": "\n".join(f"- {e}" for e in evidence),
        "Recommended Actions": "\n".join(f"- {a}" for a in dict.fromkeys(actions)),
        "Metrics to Monitor": "\n".join(f"- {m}" for m in metrics),
    }
    answer = "\n\n".join(f"{h}:\n{sections[h]}" for h in SECTION_HEADERS)
    return InvestigationResult(answer=answer, sections=sections, tool_calls=tool_calls, mode="demo")


def _run_demo_mode(question: str, tool_impls: dict) -> InvestigationResult:
    if any(kw in question.lower() for kw in _LATE_INCREASE_KEYWORDS):
        return _build_late_increase_report(tool_impls)

    tool_names = _route_demo_tools(question)
    tool_calls: list[ToolCallRecord] = []
    evidence: list[str] = []
    actions: list[str] = []
    metrics: list[str] = []

    for name in tool_names:
        impl = tool_impls[name]
        output = impl["fn"]()
        tool_calls.append(ToolCallRecord(name=name, input={}, output=output))
        if "error" in output:
            evidence.append(f"{name}: {output['error']}")
            continue
        builder = _EVIDENCE_BUILDERS.get(name)
        if builder:
            evidence.extend(b for b in builder(output) if b)
        actions.extend(_ACTION_BUILDERS.get(name, []))
        metrics.extend(_METRIC_BUILDERS.get(name, []))

    diagnosis = (
        "Demo mode (no Anthropic API key configured) — this report is generated deterministically "
        "from the same tools the AI investigator would use, without calling an LLM. "
        + (evidence[0] if evidence else "No relevant data was found for this question.")
    )
    sections = {
        "Executive Diagnosis": diagnosis,
        "Evidence": "\n".join(f"- {e}" for e in evidence) if evidence else "- No data available for this question.",
        "Recommended Actions": "\n".join(f"- {a}" for a in dict.fromkeys(actions)) if actions else "- N/A",
        "Metrics to Monitor": "\n".join(f"- {m}" for m in dict.fromkeys(metrics)) if metrics else "- N/A",
    }
    answer = "\n\n".join(f"{h}:\n{sections[h]}" for h in SECTION_HEADERS)
    return InvestigationResult(answer=answer, sections=sections, tool_calls=tool_calls, mode="demo")


# ----------------------------------------------------------------------------
# LLM mode
# ----------------------------------------------------------------------------
def _run_llm_mode(question: str, tool_impls: dict, filters: dict, api_key: str) -> InvestigationResult:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    tools = [impl["spec"] for impl in tool_impls.values()]
    system = SYSTEM_PROMPT_TEMPLATE.format(start=filters["start_date"], end=filters["end_date"], outcome=filters["outcome"])
    messages = [{"role": "user", "content": question}]
    tool_calls: list[ToolCallRecord] = []

    try:
        for _ in range(MAX_ITERATIONS):
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=system,
                tools=tools,
                messages=messages,
                output_config={"effort": "medium"},
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                answer = "".join(b.text for b in response.content if b.type == "text")
                return InvestigationResult(
                    answer=answer, sections=_parse_sections(answer), tool_calls=tool_calls, mode="llm"
                )

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                impl = tool_impls.get(block.name)
                if impl is None:
                    output = {"error": f"Unknown tool: {block.name}"}
                else:
                    try:
                        output = impl["fn"](**block.input)
                    except Exception as exc:  # defensive: bad LLM-supplied args
                        output = {"error": f"Tool call failed: {exc}"}
                tool_calls.append(ToolCallRecord(name=block.name, input=block.input, output=output))
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(output, default=str)}
                )
            messages.append({"role": "user", "content": tool_results})

        return InvestigationResult(
            answer="The investigation needed more tool calls than allowed. Try a narrower question.",
            sections={},
            tool_calls=tool_calls,
            mode="llm",
            warning="max_iterations_exceeded",
        )
    except Exception as exc:
        return InvestigationResult(
            answer=f"Could not reach the Anthropic API ({exc}). Falling back is not automatic — "
            "retry, or clear the API key to use demo mode.",
            sections={},
            tool_calls=tool_calls,
            mode="error",
            warning=str(exc),
        )


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def investigate(
    question: str,
    orders_df,
    sellers_df,
    categories_df,
    monthly_df,
    filters: dict,
    force_demo: bool = False,
) -> InvestigationResult:
    tool_impls = agent_tools.build_tools(orders_df, sellers_df, categories_df, monthly_df, filters)
    api_key = None if force_demo else get_api_key()
    if not api_key:
        return _run_demo_mode(question, tool_impls)
    return _run_llm_mode(question, tool_impls, filters, api_key)
