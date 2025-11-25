"""
Preprocess enriched restaurant data to create model-ready datasets.

Steps Per Restaurant
--------------------
1. Load each restaurant CSV from `input_data/enriched_data`, skipping `_debug` folders.
2. Drop leftover holiday metadata columns and remove beverage/MRP menu items.
3. Filter out items without sales in the past 30 days or <= 5 selling days; compute last-month totals.
4. Add `predict_model` (1 if last-month count > 50, else 0).
5. Generate features per item:
   - 3-day & 7-day averages (based on prior days only).
   - 7/14/21/28-day lag counts.
   - 1/2/3-month averages using only days with sales.
6. Save the processed CSV to `input_data/preprocessed_data/{foodcourt}/{restaurant}.csv`.
"""

import logging
import os
import re
from datetime import timedelta
from typing import Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_data")
ENRICHED_DIR = os.path.join(INPUT_DIR, "enriched_data")
OUTPUT_DIR = os.path.join(INPUT_DIR, "preprocessed_data")
SKIP_DIR_NAME = "_debug"

PAST_MONTH_DAYS = 30
PREDICT_THRESHOLD = 50

EXCLUDE_KEYWORDS = [
    "coffee",
    "coffe",
    "chai",
    "tea",
    "drink",
    "juice",
    "beverage",
    "shake",
    "mocktail",
    "latte",
    "espresso",
    "mocha",
    "cappuccino",
    "frappe",
    "mrp",
]
EXCLUDE_PATTERN = re.compile("|".join(EXCLUDE_KEYWORDS), re.IGNORECASE)

LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

LAG_WINDOWS = [7, 14, 21, 28]
MONTH_WINDOWS = [30, 60, 90]
MONTH_AVG_COLS = ["avg_1_month", "avg_2_month", "avg_3_month"]
ROLLING_WINDOWS: Tuple[int, int] = (3, 7)

# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #


def is_excluded_item(name) -> bool:
    """Return True if the item name matches beverage or MRP keywords."""
    if not isinstance(name, str):
        return False
    return bool(EXCLUDE_PATTERN.search(name))


def filter_excluded_items(df: pd.DataFrame):
    """Remove rows where itemname matches beverage/MRP keywords."""
    if "itemname" not in df.columns:
        return df, pd.DataFrame()
    mask = df["itemname"].astype(str).apply(lambda x: not is_excluded_item(x))
    removed = df[~mask][["menuitemid", "itemname"]].copy()
    removed = removed.drop_duplicates()
    return df[mask], removed


def drop_inactive_items(df: pd.DataFrame):
    """Keep only items that sold at least once in the last 30 days."""
    if df.empty:
        return df, pd.DataFrame(), None

    latest_date = df["date"].max()
    if pd.isna(latest_date):
        return pd.DataFrame(), pd.DataFrame(), None

    cutoff = latest_date - timedelta(days=PAST_MONTH_DAYS)
    recent_sales = df[df["date"] >= cutoff]
    active_items = set(recent_sales["menuitemid"])
    inactive = df[~df["menuitemid"].isin(active_items)][["menuitemid", "itemname"]].drop_duplicates()
    return df[df["menuitemid"].isin(active_items)], inactive, cutoff


def add_predict_flag(df: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Add predict_model column based on last-month totals."""
    recent_counts = (
        df[df["date"] >= cutoff].groupby("menuitemid")["count"].sum().to_dict()
    )
    df["predict_model"] = df["menuitemid"].map(
        lambda mid: int(recent_counts.get(mid, 0) > PREDICT_THRESHOLD)
    )
    return df


def filter_recent_activity(df: pd.DataFrame, cutoff):
    """Remove items with zero sales or <=5 non-zero sale days in past month."""
    recent = df[df["date"] >= cutoff]
    agg = recent.groupby("menuitemid").agg(
        total_count=("count", "sum"),
        sale_days=("count", lambda x: (x > 0).sum()),
    )
    eligible_ids = agg[
        (agg["total_count"] > 0) & (agg["sale_days"] > 5)
    ].index
    removed = df[~df["menuitemid"].isin(eligible_ids)][["menuitemid", "itemname"]].drop_duplicates()
    return df[df["menuitemid"].isin(eligible_ids)], removed


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling averages and lag features per menu item."""
    if df.empty:
        return df

    df = df.sort_values(["menuitemid", "date"]).copy()
    original_ids = df["menuitemid"].copy()
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0.0)

    def apply_item(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("date").copy()
        shifted = group["count"].shift(1)
        group["avg_3_day"] = shifted.rolling(window=ROLLING_WINDOWS[0], min_periods=1).mean()
        group["avg_7_day"] = shifted.rolling(window=ROLLING_WINDOWS[1], min_periods=1).mean()

        for lag in LAG_WINDOWS:
            group[f"lag_{lag}_day"] = group["count"].shift(lag)

        counts = group["count"]
        positive_counts = counts.where(counts > 0, 0.0)
        sold_indicator = counts.gt(0).astype(float)

        for window, col_name in zip(MONTH_WINDOWS, MONTH_AVG_COLS):
            sum_counts = positive_counts.rolling(window=window, min_periods=1).sum()
            days_with_sales = sold_indicator.rolling(window=window, min_periods=1).sum()
            avg = sum_counts / days_with_sales.replace(0, np.nan)
            group[col_name] = avg.fillna(0.0)

        return group

    groupby_obj = df.groupby("menuitemid", group_keys=False)
    try:
        df = groupby_obj.apply(apply_item, include_groups=False)
    except TypeError:
        df = groupby_obj.apply(apply_item)

    if "menuitemid" not in df.columns:
        df["menuitemid"] = original_ids.values

    feature_cols = (
        ["avg_3_day", "avg_7_day"]
        + [f"lag_{lag}_day" for lag in LAG_WINDOWS]
        + MONTH_AVG_COLS
    )
    for col in feature_cols:
        df[col] = df[col].fillna(0.0)

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure final column ordering and drop unwanted holiday columns."""
    drop_cols = [col for col in df.columns if col.startswith("holiday")]
    if "weather_description" in df.columns:
        drop_cols.append("weather_description")
    if drop_cols:
        df = df.drop(columns=drop_cols)

    desired_order = [
        "foodcourtid",
        "foodcourtname",
        "restaurant",
        "restaurantname",
        "menuitemid",
        "itemname",
        "price",
        "isVeg",
        "isSpicy",
        "date",
        "count",
        "is_mon",
        "is_tue",
        "is_wed",
        "is_thu",
        "is_fri",
        "is_sat",
        "is_sun",
        "predict_model",
        "avg_3_day",
        "avg_7_day",
        "lag_7_day",
        "lag_14_day",
        "lag_21_day",
        "lag_28_day",
        "avg_1_month",
        "avg_2_month",
        "avg_3_month",
    ]
    existing_order = [col for col in desired_order if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_order]
    return df[existing_order + remaining_cols]


def process_restaurant_df(df: pd.DataFrame):
    """Apply all preprocessing steps to a single restaurant dataframe."""
    log = {
        "total_items": df["menuitemid"].nunique() if df.size else 0,
        "removed_beverage": pd.DataFrame(columns=["menuitemid", "itemname"]),
        "removed_inactive": pd.DataFrame(columns=["menuitemid", "itemname"]),
        "removed_recent": pd.DataFrame(columns=["menuitemid", "itemname"]),
        "kept_items": pd.DataFrame(columns=["menuitemid", "itemname"]),
    }
    if df.empty:
        return df, log

    df = df.copy()
    for col in ["date", "menuitemid", "itemname"]:
        if col not in df.columns:
            logging.warning("Missing column %s; skipping dataset.", col)
            return pd.DataFrame(), log

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "menuitemid", "itemname"])
    df = df.sort_values("date").drop_duplicates(subset=["menuitemid", "date"], keep="last")
    df["menuitemid"] = df["menuitemid"].astype(str)
    df["count"] = pd.to_numeric(df.get("count", 0), errors="coerce").fillna(0.0)

    df, removed_beverage = filter_excluded_items(df)
    log["removed_beverage"] = removed_beverage
    if df.empty:
        return df, log

    df, inactive, cutoff = drop_inactive_items(df)
    log["removed_inactive"] = inactive
    if df.empty or cutoff is None:
        return df, log

    df, removed_recent = filter_recent_activity(df, cutoff)
    log["removed_recent"] = removed_recent
    if df.empty:
        return df, log

    df = add_predict_flag(df, cutoff)
    df = add_temporal_features(df)
    df = reorder_columns(df)
    log["kept_items"] = df[["menuitemid", "itemname"]].drop_duplicates()
    return df, log


# --------------------------------------------------------------------------- #
# Main Execution
# --------------------------------------------------------------------------- #


def write_log(foodcourt_id, restaurant_id, log_data):
    if not log_data:
        return
    fc_dir = os.path.join(LOG_DIR, foodcourt_id)
    os.makedirs(fc_dir, exist_ok=True)
    path = os.path.join(fc_dir, f"{restaurant_id}.txt")

    def format_section(title, df):
        lines = [f"{title}: {len(df)}"]
        if not df.empty:
            for _, row in df.iterrows():
                lines.append(f"  - {row['menuitemid']} | {row.get('itemname', '')}")
        return "\n".join(lines)

    sections = [
        f"Foodcourt: {foodcourt_id}",
        f"Restaurant: {restaurant_id}",
        f"Total items before filters: {log_data.get('total_items', 0)}",
        format_section("Removed - Beverage/MRP", log_data.get("removed_beverage", pd.DataFrame())),
        format_section("Removed - No sales in last 30 days", log_data.get("removed_inactive", pd.DataFrame())),
        format_section("Removed - <=5 sale days or zero sales", log_data.get("removed_recent", pd.DataFrame())),
        format_section("Kept items (model runs on)", log_data.get("kept_items", pd.DataFrame())),
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))


def main():
    if not os.path.exists(ENRICHED_DIR):
        logging.error("Enriched data directory not found: %s", ENRICHED_DIR)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    processed_files = 0
    skipped_files = 0

    foodcourts = sorted(
        [
            fc
            for fc in os.listdir(ENRICHED_DIR)
            if os.path.isdir(os.path.join(ENRICHED_DIR, fc)) and fc != SKIP_DIR_NAME
        ]
    )

    for foodcourt_id in foodcourts:
        fc_path = os.path.join(ENRICHED_DIR, foodcourt_id)
        filenames = sorted(
            [
                fn
                for fn in os.listdir(fc_path)
                if fn.lower().endswith(".csv") and not fn.startswith(SKIP_DIR_NAME)
            ]
        )

        for filename in filenames:
            input_path = os.path.join(fc_path, filename)
            try:
                df = pd.read_csv(input_path)
            except Exception as exc:
                logging.error("Failed to read %s: %s", input_path, exc)
                skipped_files += 1
                continue

            processed_df, log_data = process_restaurant_df(df)
            if processed_df.empty:
                logging.info("Skipping %s (no usable rows after preprocessing)", input_path)
                skipped_files += 1
                write_log(foodcourt_id, filename.replace(".csv", ""), log_data)
                continue

            output_fc_dir = os.path.join(OUTPUT_DIR, foodcourt_id)
            os.makedirs(output_fc_dir, exist_ok=True)
            output_path = os.path.join(output_fc_dir, filename)

            processed_df.to_csv(output_path, index=False, encoding="utf-8")
            processed_files += 1
            logging.info("Preprocessed %s -> %s (%d rows)", input_path, output_path, len(processed_df))
            write_log(foodcourt_id, filename.replace(".csv", ""), log_data)

    logging.info("Preprocessing complete. Files written: %d | Skipped: %d", processed_files, skipped_files)


if __name__ == "__main__":
    main()

