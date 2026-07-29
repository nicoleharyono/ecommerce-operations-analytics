# Fulfillment Control Tower

An interactive Streamlit dashboard for the e-commerce fulfillment analytics
portfolio project. It surfaces where late deliveries and freight costs come
from — by handling time, distance, seller, category, and month — so the
business can act on it without hurting customer satisfaction.

## Data

The app reads four pre-processed files from `../data/processed/` (relative to
this folder), produced by `notebooks/02_dashboard_and_recommendations.ipynb`:

- `orders_analysis.csv` — order-level fulfillment data (dates, handling time,
  distance, weight, freight, review score, delivery outcome)
- `seller_performance.csv` — per-seller order volume, handling time, late rate
- `category_summary.csv` — per-category order volume, late rate, review score,
  freight metrics
- `monthly_performance.csv` — monthly late-delivery rate and related metrics

No columns are invented: every field the dashboard displays exists in one of
these files, and grouping bins (handling time, distance, weight) are derived
from raw columns already present in `orders_analysis.csv`.

## Running locally

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r dashboard/requirements.txt
cd dashboard
streamlit run app.py
```

The app opens at `http://localhost:8501`. Run it from inside the `dashboard/`
folder so Streamlit picks up the color theme in `dashboard/.streamlit/config.toml`
(data paths in `app.py` are resolved relative to the script, so they work
regardless of the working directory).

## Notes on filter scope

- **Purchase date**, **Delivery outcome**, and **Order status** filter the
  order-level charts and KPI cards (they operate on `orders_analysis.csv`).
- **Product category** and **seller** filter only their respective charts
  (Top Categories / Top Sellers by Late Orders), because `seller_performance.csv`
  and `category_summary.csv` are pre-aggregated across the full order history
  and don't carry a per-order date or status to join against.
- The "Operational Recommendations" section recomputes its figures live from
  whatever filters are currently applied.

## AI Operations Investigator

An agentic feature below the static charts: ask an operational question, and
an LLM tool-calling agent decides which analyses to run, executes them
against the real dashboard data, and returns an evidence-based diagnosis —
executive summary, evidence, recommended actions, and metrics to monitor.

### Architecture

```
dashboard/
├── data.py          # pure data loading (no Streamlit) — shared by app.py, agent_tools.py, tests
├── agent_tools.py   # 8 deterministic analysis functions + their Claude tool schemas
├── agent.py         # orchestration: LLM tool-use loop, demo-mode fallback, numeric-support guard
└── app.py           # UI — suggested questions, tool-call log, report card
```

This is a **tool-calling architecture, not a single prompt over the raw CSVs**.
The LLM never sees or touches a dataframe — it only sees the JSON a tool
returns, and every number in the final report has to trace back to one of
those tool calls. The eight tools are plain, testable Python functions:

| Tool | What it computes |
|---|---|
| `get_kpi_summary` | Current headline KPIs for the active date/outcome filter |
| `get_monthly_late_rate_trend` | Month-by-month late rate, peak month, recent-vs-prior comparison |
| `get_handling_time_analysis` | Late rate by seller handling-time group |
| `get_distance_analysis` | Late rate by shipping-distance group |
| `get_seller_performance_analysis` | Top late-order sellers, high-risk seller count |
| `get_category_performance_analysis` | Top late-order categories, pre-flagged risk categories |
| `get_freight_cost_analysis` | Median freight by weight group, high-freight-burden categories |
| `get_review_score_impact_analysis` | Review-score gap between on-time and late deliveries |

The agent loop (`agent._run_llm_mode`) is a manual `while` loop over
`client.messages.create(..., tools=...)`: on each turn the model can call one
or more tools, we execute them locally and return the results, and this
repeats (capped at 4 iterations) until the model responds with plain text
instead of a tool call. Every tool call — name, input, and output — is kept
and shown in the "Tools the agent used" expander in the UI, so nothing is
hidden. The final answer is required (via the system prompt) to use four
labeled sections — Executive Diagnosis, Evidence, Recommended Actions,
Metrics to Monitor — which `agent._parse_sections` splits out for the report
card; if the model ever doesn't follow that format, the raw text is shown
instead of failing.

**Filter scope passed to the agent:** only Purchase date and Delivery
outcome — the same two filters that scope the order-level charts and KPIs
above. Category, seller, and order-status filters are intentionally *not*
passed through, for the same reason they only scope their own charts
elsewhere in the dashboard (`seller_performance.csv` / `category_summary.csv`
are pre-aggregated across the full order history).

### Setup — Anthropic API key

The agent looks for a key in this order:

1. `st.secrets["ANTHROPIC_API_KEY"]` — create `dashboard/.streamlit/secrets.toml`:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   (this file is untracked by git in a typical setup — don't commit a real key)
2. the `ANTHROPIC_API_KEY` environment variable, e.g. `export ANTHROPIC_API_KEY=sk-ant-...`

### Demo mode (no API key required)

If neither of the above is set, the app **does not crash or ask for a key** —
it automatically falls back to a deterministic, rule-based report generator
(`agent._run_demo_mode`) that routes each question to the same relevant
tools via keyword matching and templates a report from the real tool output,
with zero network calls. A "Demo mode" checkbox next to the Investigate
button also lets you force this path even with a key configured, e.g. to
demo the app without spending API credits. A demo-mode badge always makes
it clear which path produced a given report.

### Example questions

- "Why did late deliveries increase?"
- "Which sellers should operations review first?"
- "Where can freight costs be reduced?"
- "What is hurting customer satisfaction?"
- "Generate an executive operations brief."
- Or type your own — e.g. "Why are freight costs high for electronics?"

### Limitations

- Late-order *counts* for sellers and categories are estimated as
  `order_count × late_rate` from the pre-aggregated CSVs (rounded to the
  nearest integer) — the same estimation the dashboard's own charts use,
  since order-item-level seller/category joins aren't in the processed data.
- The agent has no memory across questions — each investigation starts a
  fresh tool-use loop with no conversation history from a prior question.
- If a question is outside what the eight tools can answer (e.g. it asks
  about something unrelated to fulfillment/freight/reviews), the agent is
  instructed to say so rather than speculate — it doesn't have a "general
  knowledge" fallback.

### Tests

```bash
cd dashboard
pytest tests/ -v
```

`tests/test_agent_tools.py` checks each tool's numbers against known
ground-truth figures (validated against the source notebook). `tests/test_agent.py`
covers demo-mode routing and `validate_no_unsupported_numbers` — a guard
that scans a candidate answer for numeric tokens and confirms every one
traces back to an actual tool output, including an end-to-end check that
the demo-mode report for every suggested question never cites a number it
didn't compute.
