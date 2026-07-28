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

- **Purchase date** and **delivery status** filter the order-level charts and
  KPI cards (they operate on `orders_analysis.csv`).
- **Product category** and **seller** filter only their respective charts
  (Top Categories / Top Sellers by Late Orders), because `seller_performance.csv`
  and `category_summary.csv` are pre-aggregated across the full order history
  and don't carry a per-order date or status to join against.
- The "Operational Recommendations" section recomputes its figures live from
  whatever filters are currently applied.
