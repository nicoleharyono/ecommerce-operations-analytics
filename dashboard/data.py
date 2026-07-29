"""Pure data loading and formatting for the Fulfillment Control Tower.

No Streamlit dependency — importable from the dashboard app, the AI agent
tools, and tests alike. `app.py` wraps these loaders with `st.cache_data`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def fmt_pct(x, decimals=1):
    return "N/A" if pd.isna(x) else f"{x:.{decimals}f}%"


def fmt_currency(x, decimals=2):
    return "N/A" if pd.isna(x) else f"R$ {x:,.{decimals}f}"


def fmt_days(x, decimals=1):
    return "N/A" if pd.isna(x) else f"{x:.{decimals}f} days"


def fmt_int(x):
    return "N/A" if pd.isna(x) else f"{x:,.0f}"


def titleize(s: str) -> str:
    return str(s).replace("_", " ").title()


def load_orders() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / "orders_analysis.csv",
        parse_dates=["order_purchase_timestamp"],
    )

    df["purchase_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    df["purchase_date"] = df["order_purchase_timestamp"].dt.date
    df["order_status_label"] = df["order_status"].map(titleize)

    handling_bins = [-0.01, 1, 3, 7, np.inf]
    handling_labels = ["0-1 days", "2-3 days", "4-7 days", "8+ days"]
    df["handling_group"] = pd.cut(df["handling_days"], bins=handling_bins, labels=handling_labels)

    distance_bins = [-0.01, 250, 500, 1000, np.inf]
    distance_labels = ["≤250 km", "250-500 km", "500-1,000 km", "1,000+ km"]
    df["distance_group"] = pd.cut(df["max_distance_km"], bins=distance_bins, labels=distance_labels)

    weight_bins = [-0.01, 500, 2000, 5000, 10000, np.inf]
    weight_labels = ["<0.5 kg", "0.5-2 kg", "2-5 kg", "5-10 kg", "10+ kg"]
    df["weight_group"] = pd.cut(df["total_weight_g"], bins=weight_bins, labels=weight_labels)

    return df


def load_sellers() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "seller_performance.csv")
    df["late_orders"] = (df["order_count"] * df["late_rate"]).round().astype(int)

    raw_sellers_path = DATA_DIR.parent / "raw" / "olist_sellers_dataset.csv"
    if raw_sellers_path.exists():
        states = pd.read_csv(raw_sellers_path, usecols=["seller_id", "seller_state"])
        df = df.merge(states, on="seller_id", how="left")
    else:
        df["seller_state"] = np.nan

    return df


def load_categories() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "category_summary.csv")
    df["late_orders"] = (df["order_count"] * df["late_rate"]).round().astype(int)
    df["category_label"] = df["product_category_name_english"].map(titleize)
    return df


def load_monthly() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "monthly_performance.csv")
    df["month_label"] = pd.to_datetime(df["purchase_month"] + "-01").dt.strftime("%b %Y")
    return df
