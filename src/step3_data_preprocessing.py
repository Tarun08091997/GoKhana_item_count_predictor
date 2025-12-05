"""
Preprocess enriched restaurant data to create model-ready datasets.

Steps Per Restaurant
--------------------
1. Load each restaurant CSV from `input_data/Model Training/enriched_data`, skipping `_debug` folders.
2. Drop leftover holiday metadata columns (beverage/MRP items are filtered in enrichment step).
3. Flag items without sales in the past 30 days or <= 5 selling days for moving-average models; compute last-month totals.
4. Set `predict_model`:
   - 1 = enough history for XGBoost (>= 6 months span and last-month total > 50).
   - 2 = between 3 and < 6 months of history (weekly moving average).
   - 3 = everything else (forced to moving average).
5. Generate features per item:
   - 3-day & 7-day averages (based on prior days only).
   - 7/14/21/28-day lag counts.
   - 1/2/3-month averages using only days with sales.
6. Skip restaurants whose processed data is empty or spans < 6 months;
   record discard reasons in `discard_report.txt` under the output folder.
7. Save the processed CSV to `input_data/Model Training/preprocessed_data/{foodcourt}/{restaurant}.csv`.
"""

import logging
import os
import re
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Tuple, List, Optional, Set

import numpy as np
import pandas as pd
from src.util.progress_bar import ProgressBar

# Filter utils removed - we only use FR_data.json and fetched_data/ now

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "input_data")
# Read from output_data/enrich_data/ (Excel files, one per item)
# Write to output_data/preprocessing/ (Excel files, one per item)
# Import will be done in main() to avoid circular imports
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


# LOG_DIR and DISCARD_REPORT_FILENAME will be set in main()
DISCARD_REPORT_FILENAME = "discard_report.xlsx"
MIN_DATA_SPAN_DAYS = 180

LAG_WINDOWS = [7, 14, 21, 28]
MONTH_WINDOWS = [30, 60, 90]
MONTH_AVG_COLS = ["avg_1_month", "avg_2_month", "avg_3_month"]
ROLLING_WINDOWS: Tuple[int, int] = (3, 7)
THREE_MONTH_SPAN_DAYS = 90

# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #


# Progress bar functionality moved to pipeline_utils.ProgressBar


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


def remove_leading_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove leading zero-count rows for each item.
    For each item, find the first date with count > 0 and remove all rows before that date.
    """
    if df.empty:
        return df
    
    df = df.sort_values(["menuitemid", "date"]).copy()
    
    # For each item, find first non-zero date
    first_sale_dates = df[df["count"] > 0].groupby("menuitemid")["date"].min()
    
    # Filter each item's data to start from its first sale date
    filtered_rows = []
    for item_id, item_df in df.groupby("menuitemid"):
        first_sale_date = first_sale_dates.get(item_id)
        if first_sale_date is not None:
            # Keep only rows from first sale date onwards
            item_df = item_df[item_df["date"] >= first_sale_date]
        filtered_rows.append(item_df)
    
    if filtered_rows:
        return pd.concat(filtered_rows, ignore_index=True)
    return pd.DataFrame()


def compute_item_stats(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    stats = df.groupby("menuitemid").agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
    )
    stats["span_days"] = (stats["last_date"] - stats["first_date"]).dt.days + 1

    recent = df[df["date"] >= cutoff]
    recent_stats = recent.groupby("menuitemid").agg(
        recent_total=("count", "sum"),
        recent_sale_days=("count", lambda x: (x > 0).sum()),
    )
    stats = stats.join(recent_stats, how="left").fillna({"recent_total": 0.0, "recent_sale_days": 0})
    return stats


def assign_predict_model(stats: pd.DataFrame) -> pd.Series:
    """
    Assign predict_model value to each item:
    1 = XGBoost (needs >= 6 months span and recent_total > 50)
    2 = Weekly Moving Average with 3+ months span
    3 = Weekly Moving Average (everything else, including low activity items)
    
    All items get assigned a model - nothing is discarded based on data span.
    """
    def classify(row):
        span = row["span_days"]
        recent_total = row["recent_total"]
        sale_days = row["recent_sale_days"]
        has_recent_activity = (recent_total > 0) and (sale_days > 5)
        
        # Model 1: XGBoost - needs >= 6 months and good recent activity
        if span >= MIN_DATA_SPAN_DAYS and recent_total > PREDICT_THRESHOLD and has_recent_activity:
            return 1
        
        # Model 2: Weekly Moving Average with 3+ months of data
        if span >= THREE_MONTH_SPAN_DAYS:
            return 2
        
        # Model 3: Weekly Moving Average for everything else
        # This includes items with less data, low activity, etc.
        # Everything gets a model - nothing discarded
        return 3

    return stats.apply(classify, axis=1).astype(int)


def extract_names(df: pd.DataFrame | None):
    """Return (foodcourt_name, restaurant_name) from dataframe when available."""
    if df is None or df.empty:
        return "", ""
    fc_name = ""
    rest_name = ""
    if "foodcourtname" in df.columns and not df["foodcourtname"].empty:
        fc_name = str(df["foodcourtname"].iloc[0])
    if "restaurantname" in df.columns and not df["restaurantname"].empty:
        rest_name = str(df["restaurantname"].iloc[0])
    return fc_name, rest_name


def append_discard_entry(
    entries,
    foodcourt_id: str,
    restaurant_id: str,
    df_context: pd.DataFrame | None,
    reason: str,
):
    fc_name, rest_name = extract_names(df_context)
    entries.append(
        (
            foodcourt_id or "",
            fc_name,
            restaurant_id or "",
            rest_name,
            reason,
        )
    )


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
        shifted_nonzero = shifted.where(shifted > 0, np.nan)
        group["avg_3_day"] = shifted_nonzero.rolling(window=ROLLING_WINDOWS[0], min_periods=1).mean()
        group["avg_7_day"] = shifted_nonzero.rolling(window=ROLLING_WINDOWS[1], min_periods=1).mean()

        lag_cols = []
        for lag in LAG_WINDOWS:
            col_name = f"lag_{lag}_day"
            group[col_name] = group["count"].shift(lag)
            lag_cols.append(col_name)

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
            warning_msg = f"Missing column {col}; skipping dataset."
            from src.util.pipeline_utils import get_pipeline_logger
            get_pipeline_logger().log_warning("preprocessing", warning_msg)
            # Don't print to console - only log to pipeline logs
            return pd.DataFrame(), log

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "menuitemid", "itemname"])
    df = df.sort_values("date").drop_duplicates(subset=["menuitemid", "date"], keep="last")
    df["menuitemid"] = df["menuitemid"].astype(str)
    df["count"] = pd.to_numeric(df.get("count", 0), errors="coerce").fillna(0.0)

    # Note: Beverage/MRP items are now filtered in enrichment step (step2)
    # No need to filter again here - they should already be excluded
    removed_beverage = pd.DataFrame()  # Empty - already handled in enrichment
    log["removed_beverage"] = removed_beverage
    # All remaining items (even with zero counts or minimal data) will be processed

    latest_date = df["date"].max() if not df.empty else None
    if pd.isna(latest_date) and not df.empty:
        # If we have data but invalid latest_date, try to fix
        valid_dates = df["date"].dropna()
        if not valid_dates.empty:
            latest_date = valid_dates.max()
    
    if df.empty or pd.isna(latest_date):
        # Return empty dataframe with predict_model=3 for edge cases
        # This will still be saved and can be assigned model 3
        if not df.empty:
            df["predict_model"] = 3
        return df, log
    
    cutoff = latest_date - timedelta(days=PAST_MONTH_DAYS)

    stats = compute_item_stats(df, cutoff)

    # Remove leading zeros for each item (set starting date to first non-zero sale)
    # But keep items even if they have all zeros - they'll get predict_model=3
    df = remove_leading_zeros(df)
    # Don't check if empty - continue processing

    # Assign predict_model to all items (1, 2, or 3)
    # Items with less data will get model 3, but they will still be processed
    predict_map = assign_predict_model(stats)
    df["predict_model"] = df["menuitemid"].map(predict_map).fillna(3).astype(int)
    
    # DON'T filter out zero counts - keep all items, assign model 3 if needed
    # Zero count items can still be used for training (will predict 0 or use moving average)
    
    df = add_temporal_features(df)
    df = reorder_columns(df)
    log["kept_items"] = df[["menuitemid", "itemname"]].drop_duplicates()
    return df, log


# --------------------------------------------------------------------------- #
# Main Execution
# --------------------------------------------------------------------------- #


def write_log(foodcourt_id, restaurant_id, log_data, log_dir):
    """
    No-op function - log writing removed.
    Only pipeline_logs.xlsx is used for logging, not individual text files.
    """
    pass


def main(retrain_config: Optional[dict] = None, file_saver=None, restaurant_tracker=None, checkpoint_manager=None):
    """
    Main function for data preprocessing.
    
    Args:
        retrain_config: Optional retrain configuration dict from retrain.json
        file_saver: Optional FileSaver instance for saving files
        restaurant_tracker: Optional RestaurantTracker instance for tracking item status
        checkpoint_manager: Optional CheckpointManager instance for checkpoint/resume functionality
    """
    # Set up paths (import here to avoid circular imports)
    from src.util.pipeline_utils import get_output_base_dir, get_pipeline_logger, get_pipeline_log_dir, get_pipeline_type
    pipeline_logger = get_pipeline_logger()
    OUTPUT_BASE = get_output_base_dir()
    pipeline_type = get_pipeline_type()
    ENRICHED_DIR = str(OUTPUT_BASE / "enrich_data" / pipeline_type)
    OUTPUT_DIR = str(OUTPUT_BASE / "preprocessing" / pipeline_type)
    # No separate preprocessing log directory needed - only pipeline_logs.xlsx is used
    LOG_DIR = None
    
    # Track saved files for validation (per foodcourt)
    saved_files_by_fc = {}
    
    if not os.path.exists(ENRICHED_DIR):
        error_msg = f"Enriched data directory not found: {ENRICHED_DIR}"
        pipeline_logger.log_general_error("preprocessing", error_msg)
        logging.error(error_msg)
        return
    
    # Count enriched data files
    enriched_files = 0
    enriched_foodcourts = set()
    enriched_restaurants = set()
    for fc_dir in os.listdir(ENRICHED_DIR):
        if fc_dir == SKIP_DIR_NAME:
            continue
        fc_path = os.path.join(ENRICHED_DIR, fc_dir)
        if os.path.isdir(fc_path):
            enriched_foodcourts.add(fc_dir)
            for filename in os.listdir(fc_path):
                if filename.endswith('.csv'):
                    enriched_files += 1
                    # Extract restaurant ID from filename: {fc_id}_{rest_id}_{item_id}_{item_name}_enrich_data.csv
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        enriched_restaurants.add((fc_dir, parts[1]))
    
    # Count preprocessed data files
    preprocessed_files = 0
    preprocessed_foodcourts = set()
    preprocessed_restaurants = set()
    if os.path.exists(OUTPUT_DIR):
        for fc_dir in os.listdir(OUTPUT_DIR):
            if fc_dir == SKIP_DIR_NAME:
                continue
            fc_path = os.path.join(OUTPUT_DIR, fc_dir)
            if os.path.isdir(fc_path):
                preprocessed_foodcourts.add(fc_dir)
                for filename in os.listdir(fc_path):
                    if filename.endswith('.csv'):
                        preprocessed_files += 1
                        parts = filename.split('_')
                        if len(parts) >= 2:
                            preprocessed_restaurants.add((fc_dir, parts[1]))
    
    # Log process start with summary
    summary = {
        "enrich_data/ (input)": f"{len(enriched_foodcourts)} foodcourts, {len(enriched_restaurants)} restaurants, {enriched_files} files",
        "preprocessing/ (existing)": f"{len(preprocessed_foodcourts)} foodcourts, {len(preprocessed_restaurants)} restaurants, {preprocessed_files} files",
        "Needs processing": f"{enriched_files - preprocessed_files} files (difference + retrain config)"
    }
    pipeline_logger.log_process_start("preprocessing", summary)
    pipeline_logger.log_info("preprocessing", f"✅ Enriched data directory found: {ENRICHED_DIR}")
    logging.info("✅ Enriched data directory found: %s", ENRICHED_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # LOG_DIR removed - no separate preprocessing log directory needed, only pipeline_logs.xlsx

    processed_files = 0
    skipped_files = 0
    discard_entries: List[Tuple[str, str, str, str, str]] = []
    
    # Get global pipeline start time for progress reporting
    from src.util.pipeline_utils import get_pipeline_start_time
    pipeline_start_time = get_pipeline_start_time()
    
    # Check if preprocessing step should be skipped (empty config means skip)
    # Get foodcourt_ids and restaurant_ids to filter by (if retrain_config provided)
    allowed_foodcourt_ids = None
    allowed_restaurant_ids = None
    item_ids_filter = None
    if retrain_config:
        from src.util.pipeline_utils import get_retrain_config_for_step
        step_config = get_retrain_config_for_step("preprocessing")
        foodcourt_ids = step_config.get("foodcourt_ids", [])
        restaurant_ids = step_config.get("restaurant_ids", [])
        item_ids = step_config.get("item_ids", [])
        item_names = step_config.get("item_names", [])
        
        # If config is empty (no foodcourt_ids, restaurant_ids, item_ids, or item_names), skip this step
        if not foodcourt_ids and not restaurant_ids and not item_ids and not item_names:
            logging.info("Preprocessing config is empty in retrain.json. Skipping preprocessing step (using existing preprocessed data).")
            pipeline_logger.log_info("preprocessing", "Step skipped - config is empty in retrain.json")
            pipeline_logger.log_process_results("preprocessing", {"Status": "Skipped", "Reason": "Empty config in retrain.json"})
            return
        
        if foodcourt_ids:  # If list is not empty, filter by these foodcourt_ids
            allowed_foodcourt_ids = set(str(fc_id).strip() for fc_id in foodcourt_ids)
            logging.info("Filtering by foodcourt_ids from retrain.json: %s", allowed_foodcourt_ids)
        if restaurant_ids:
            allowed_restaurant_ids = set(str(r_id).strip() for r_id in restaurant_ids)
            logging.info("Filtering by restaurant_ids from retrain.json: %s", allowed_restaurant_ids)
        if item_ids:
            item_ids_filter = item_ids
            logging.info("Filtering by item_ids from retrain.json: %d item filters", len(item_ids))
        if item_names:
            item_names_filter = item_names
            logging.info("Filtering by item_names from retrain.json: %d item names", len(item_names))
        else:
            item_names_filter = None
    
    # Group files by foodcourt and restaurant for progress reporting
    foodcourts = sorted(
        [
            fc
            for fc in os.listdir(ENRICHED_DIR)
            if os.path.isdir(os.path.join(ENRICHED_DIR, fc)) and fc != SKIP_DIR_NAME
        ]
    )
    
    # Build structure: {foodcourt_id: {restaurant_id: [filenames]}}
    files_by_fc_rest = {}
    for foodcourt_id in foodcourts:
        # Filter by foodcourt_id if retrain_config specifies it
        if allowed_foodcourt_ids is not None and foodcourt_id not in allowed_foodcourt_ids:
            continue
        
        fc_path = os.path.join(ENRICHED_DIR, foodcourt_id)
        if not os.path.exists(fc_path):
            continue
        
        files_by_fc_rest[foodcourt_id] = {}
        filenames = sorted(
            [
                fn
                for fn in os.listdir(fc_path)
                if fn.lower().endswith((".csv")) and not fn.startswith(SKIP_DIR_NAME)
            ]
        )
        
        for filename in filenames:
            # Parse filename to get restaurant_id and item_id
            # Remove all extensions (.csv, .xlsx, .xls)
            base_name = filename.replace(".csv", "").replace(".xlsx", "").replace(".xls", "")
            if not base_name.endswith("_enrich_data"):
                continue
            prefix = base_name[:-12]  # Remove "_enrich_data"
            parts = prefix.split("_")
            if len(parts) < 2:
                continue
            restaurant_id = parts[1]
            
            # Filter by restaurant_id if retrain_config specifies it
            if allowed_restaurant_ids is not None and restaurant_id not in allowed_restaurant_ids:
                continue
            
            # Extract item_id from filename
            # Format: {fc_id}_{rest_id}_{item_id}_{item_name}_enrich_data.csv
            item_id = parts[2] if len(parts) > 2 else ""
            
            # Filter by item_ids or item_names if specified
            # Read item_name from CSV file (source of truth) instead of parsing from filename
            # This ensures proper matching with retrain.json which uses original item names
            if item_ids_filter or item_names_filter:
                from src.util.pipeline_utils import matches_item_filter, get_item_name_from_file
                # Read actual item name from CSV file
                file_path = Path(fc_path) / filename
                item_name = None
                if file_path.exists():
                    try:
                        item_name = get_item_name_from_file(file_path, preferred_columns=["itemname", "item_name"])
                    except Exception as e:
                        logging.debug(f"Failed to read item name from {filename}: {e}")
                        # Fallback to parsing from filename if reading fails
                        item_name = "_".join(parts[3:]) if len(parts) > 3 else ""
                else:
                    # Fallback to parsing from filename if file doesn't exist
                    item_name = "_".join(parts[3:]) if len(parts) > 3 else ""
                
                if not matches_item_filter(foodcourt_id, restaurant_id, item_name, item_id, 
                                          item_ids_filter, item_names_filter):
                    # Item not in filter, skip this file
                    continue
            
            if restaurant_id not in files_by_fc_rest[foodcourt_id]:
                files_by_fc_rest[foodcourt_id][restaurant_id] = []
            files_by_fc_rest[foodcourt_id][restaurant_id].append(filename)
    
    # Count total foodcourts and restaurants for progress display
    total_foodcourts = len(files_by_fc_rest)
    total_restaurants = sum(len(rests) for rests in files_by_fc_rest.values())
    total_files = sum(sum(len(files) for files in rests.values()) for rests in files_by_fc_rest.values())
    
    if total_files == 0 and total_foodcourts > 0:
        # Log warning if files_by_fc_rest has foodcourts but no files
        logging.warning(f"No files found in files_by_fc_rest structure. Total foodcourts: {total_foodcourts}, but no files discovered.")
        pipeline_logger.log_warning("preprocessing", f"No enriched data files found after filtering. Check filename format and filters.")
    
    logging.info(f"Discovered {total_foodcourts} foodcourts, {total_restaurants} restaurants, {total_files} files to process")
    
    # Create mapping for progress display
    fc_to_index = {fc_id: idx for idx, fc_id in enumerate(files_by_fc_rest.keys(), 1)}
    
    fc_idx = 0
    for foodcourt_id in sorted(files_by_fc_rest.keys()):
        fc_idx += 1
        fc_path = os.path.join(ENRICHED_DIR, foodcourt_id)
        if not os.path.exists(fc_path):
            continue
        
        restaurants = files_by_fc_rest[foodcourt_id]
        total_restaurants_in_fc = len(restaurants)
        rest_idx = 0
        
        for restaurant_id in sorted(restaurants.keys()):
            rest_idx += 1
            filenames = restaurants[restaurant_id]
            total_items = len(filenames)
            
            # Display progress: FC: 1/n | Rest: x/m | Time: HH:mm:SS
            if pipeline_start_time:
                elapsed = time.time() - pipeline_start_time
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                logging.info("FC: %d/%d | Rest: %d/%d | Time: %s", 
                           fc_idx, total_foodcourts, rest_idx, total_restaurants_in_fc, time_str)
            
            # Track item processing progress
            item_start_time = time.time()
            items_processed = 0
            
            # Initialize progress bar
            progress = None
            if total_items > 0:
                progress_prefix = (
                    f"FC {fc_idx}/{total_foodcourts} | "
                    f"Rest {rest_idx}/{total_restaurants_in_fc} | Preprocessing"
                )
                progress = ProgressBar(
                    total=total_items,
                    prefix=progress_prefix,
                    suffix="items",
                    length=40,
                    show_elapsed=True
                )
            
            for item_idx, filename in enumerate(filenames, 1):
                input_path = os.path.join(fc_path, filename)
                
                # Parse filename: {F_id}_{R_id}_{item_id}_{item_name}_enrich_data.csv
                # Extract restaurant_id and item_name from filename
                # Remove extension first (CSV files)
                base_name = filename.replace(".csv", "").replace(".xlsx", "").replace(".xls", "")
                
                # Check if filename ends with _enrich_data
                if not base_name.endswith("_enrich_data"):
                    warning_msg = f"Unexpected filename format: {filename} (expected to end with _enrich_data)"
                    error_msg = "Unexpected filename format"
                    pipeline_logger.log_warning("preprocessing", warning_msg)
                    # Add discard reason to file_locator
                    from src.util.pipeline_utils import get_file_locator, get_mongo_names
                    fc_name, rest_name = get_mongo_names(foodcourt_id, restaurant_id)
                    get_file_locator().add_discard_reason(
                        foodcourt_id, fc_name or foodcourt_id,
                        restaurant_id, rest_name or restaurant_id,
                        None, item_name,
                        "preprocessing",
                        error_msg
                    )
                    # Track error in restaurant tracker
                    if restaurant_tracker:
                        from src.util.pipeline_utils import get_all_names
                        fc_name, rest_name, item_name_val = get_all_names(
                            foodcourt_id, restaurant_id, None, filename
                        )
                        restaurant_tracker.add_error(
                            foodcourt_id, restaurant_id, item_name,
                            error_msg, "preprocessing",
                            foodcourt_name=fc_name, restaurant_name=rest_name, item_name=item_name or item_name_val
                        )
                    # Don't print to console - only log to pipeline logs
                    items_processed += 1
                    # Update progress bar for skipped items
                    if progress:
                        progress.set_current(item_idx)
                    continue
                
                # Remove "_enrich_data" suffix to get the prefix
                prefix = base_name[:-12]  # Remove "_enrich_data" (12 characters)
                
                # Split the prefix to get foodcourt_id, restaurant_id, and item_name
                parts = prefix.split("_")
                if len(parts) < 2:
                    warning_msg = f"Unexpected filename format: {filename} (not enough parts after removing _enrich_data)"
                    error_msg = "Unexpected filename format (not enough parts)"
                    pipeline_logger.log_warning("preprocessing", warning_msg)
                    # Add discard reason to file_locator
                    from src.util.pipeline_utils import get_file_locator, get_mongo_names
                    fc_name, rest_name = get_mongo_names(foodcourt_id, restaurant_id)
                    get_file_locator().add_discard_reason(
                        foodcourt_id, fc_name or foodcourt_id,
                        restaurant_id, rest_name or restaurant_id,
                        None, item_name,
                        "preprocessing",
                        error_msg
                    )
                    # Track error in restaurant tracker
                    if restaurant_tracker:
                        from src.util.pipeline_utils import get_all_names
                        fc_name, rest_name, item_name_val = get_all_names(
                            foodcourt_id, restaurant_id, None, filename
                        )
                        restaurant_tracker.add_error(
                            foodcourt_id, restaurant_id, item_name,
                            error_msg, "preprocessing",
                            foodcourt_name=fc_name, restaurant_name=rest_name, item_name=item_name or item_name_val
                        )
                    # Don't print to console - only log to pipeline logs
                    items_processed += 1
                    # Update progress bar for skipped items
                    if progress:
                        progress.set_current(item_idx)
                    continue
                
                # Parse filename: {fc_id}_{rest_id}_{item_id}_{item_name}_enrich_data.csv
                # Format: fc_id, rest_id, item_id (optional), item_name (can contain underscores)
                foodcourt_id_from_file = parts[0]
                restaurant_id_from_file = parts[1]
                
                # Determine if item_id is present (parts[2] is likely item_id if it's a valid MongoDB ObjectId format)
                # If parts has 3+ elements, parts[2] might be item_id, rest is item_name
                # If parts has 2 elements, no item_id, parts[2:] is item_name (but this case shouldn't happen)
                # We'll handle both formats: with and without item_id
                if len(parts) >= 3:
                    # Check if parts[2] looks like an ObjectId (24 hex chars) - if so, it's item_id
                    potential_item_id = parts[2]
                    if len(potential_item_id) == 24 and all(c in '0123456789abcdef' for c in potential_item_id.lower()):
                        # parts[2] is item_id, parts[3:] is item_name
                        item_id_from_file = potential_item_id
                        item_name = "_".join(parts[3:]) if len(parts) > 3 else ""
                    else:
                        # parts[2] is part of item_name (old format without item_id)
                        item_id_from_file = ""
                        item_name = "_".join(parts[2:]) if len(parts) > 2 else ""
                else:
                    # Fallback: no item_id
                    item_id_from_file = ""
                    item_name = "_".join(parts[2:]) if len(parts) > 2 else ""
                
                # Verify foodcourt_id and restaurant_id match
                if foodcourt_id_from_file != foodcourt_id or restaurant_id_from_file != restaurant_id:
                    warning_msg = f"ID mismatch in filename: {filename} (expected {foodcourt_id}/{restaurant_id}, got {foodcourt_id_from_file}/{restaurant_id_from_file})"
                    error_msg = "ID mismatch in filename"
                    pipeline_logger.log_warning("preprocessing", warning_msg)
                    # Add discard reason to file_locator
                    from src.util.pipeline_utils import get_file_locator, get_mongo_names
                    fc_name, rest_name = get_mongo_names(foodcourt_id, restaurant_id)
                    item_id_for_tracking = item_id_from_file if 'item_id_from_file' in locals() else None
                    get_file_locator().add_discard_reason(
                        foodcourt_id, fc_name or foodcourt_id,
                        restaurant_id, rest_name or restaurant_id,
                        item_id_for_tracking, item_name,
                        "preprocessing",
                        error_msg
                    )
                    # Track error in restaurant tracker
                    if restaurant_tracker:
                        from src.util.pipeline_utils import get_all_names
                        fc_name, rest_name, item_name_val = get_all_names(
                            foodcourt_id, restaurant_id, item_id_for_tracking, filename
                        )
                        restaurant_tracker.add_error(
                            foodcourt_id, restaurant_id, item_id_for_tracking or item_name,
                            error_msg, "preprocessing",
                            foodcourt_name=fc_name, restaurant_name=rest_name, item_name=item_name or item_name_val
                        )
                    # Don't print to console - only log to pipeline logs
                    items_processed += 1
                    # Update progress bar for skipped items
                    if progress:
                        progress.set_current(item_idx)
                    continue
                
                # Get item_id and actual item_name from the dataframe (source of truth)
                try:
                    # Read CSV file
                    df = pd.read_csv(input_path)
                except Exception as exc:
                    logging.error("Failed to read %s: %s", input_path, exc)
                    skipped_files += 1
                    items_processed += 1
                    from src.util.pipeline_utils import get_pipeline_logger, get_mongo_names, get_file_locator
                    fc_name, rest_name = get_mongo_names(foodcourt_id, restaurant_id)
                    error_msg = f"Failed to read input Excel file: {exc}"
                    get_pipeline_logger().log_preprocessing_error(
                        foodcourt_id, fc_name or foodcourt_id,
                        restaurant_id, rest_name or restaurant_id,
                        None, item_name,
                        error_msg
                    )
                    # Track error in restaurant tracker
                    if restaurant_tracker:
                        restaurant_tracker.add_error(
                            foodcourt_id, restaurant_id, item_name,
                            error_msg, "preprocessing"
                        )
                    # Add discard reason to file_locator
                    get_file_locator().add_discard_reason(
                        foodcourt_id, fc_name or foodcourt_id,
                        restaurant_id, rest_name or restaurant_id,
                        None, item_name,
                        "preprocessing",
                        f"Failed to read input Excel file: {str(exc)[:50]}"
                    )
                    # Update progress bar for failed items
                    if progress:
                        progress.set_current(item_idx)
                    continue
                
                if df.empty:
                    warning_msg = f"Empty dataframe in {input_path}"
                    error_msg = "Empty dataframe"
                    pipeline_logger.log_warning("preprocessing", warning_msg)
                    # Add discard reason to file_locator
                    from src.util.pipeline_utils import get_file_locator, get_mongo_names
                    fc_name, rest_name = get_mongo_names(foodcourt_id, restaurant_id)
                    get_file_locator().add_discard_reason(
                        foodcourt_id, fc_name or foodcourt_id,
                        restaurant_id, rest_name or restaurant_id,
                        None, item_name,
                        "preprocessing",
                        error_msg
                    )
                    # Track error in restaurant tracker
                    if restaurant_tracker:
                        restaurant_tracker.add_error(
                            foodcourt_id, restaurant_id, item_name,
                            error_msg, "preprocessing"
                        )
                    # Don't print to console - only log to pipeline logs
                    items_processed += 1
                    # Update progress bar for skipped items
                    if progress:
                        progress.set_current(item_idx)
                    continue
                
                # Get actual item_name from CSV file (source of truth) - this ensures consistency
                from src.util.pipeline_utils import get_item_name_from_file
                actual_item_name = get_item_name_from_file(Path(input_path))
                if actual_item_name:
                    item_name = actual_item_name  # Use the actual item_name from CSV
                
                # Get item_id from dataframe
                item_id = None
                for col in ["menuitemid", "item_identifier"]:
                    if col in df.columns and len(df[col]) > 0:
                        item_id = str(df[col].iloc[0])
                        break
                if not item_id:
                    item_id = item_name
                
                # Filter by item_ids if retrain_config specifies it
                if item_ids_filter is not None:
                    from src.util.pipeline_utils import matches_item_filter
                    if not matches_item_filter(foodcourt_id, restaurant_id, item_name, item_id, item_ids_filter):
                        # Item not in item_ids filter, skip it
                        items_processed += 1
                        if progress:
                            progress.set_current(item_idx)
                        continue
                
                # Check retrain logic
                from src.util.pipeline_utils import should_force_retrain, get_retrain_config_for_step
                force_retrain = should_force_retrain("preprocessing", foodcourt_id, restaurant_id, item_name, item_id)
                
                # Also check if downstream steps (model_generation) need this item
                # If model_generation has filters, we need to ensure those items are preprocessed
                if not force_retrain and retrain_config:
                    model_gen_config = get_retrain_config_for_step("model_generation")
                    model_gen_fc_ids = model_gen_config.get("foodcourt_ids", [])
                    model_gen_rest_ids = model_gen_config.get("restaurant_ids", [])
                    
                    # If model_generation has foodcourt_ids filter, check if this item matches
                    if model_gen_fc_ids:
                        if foodcourt_id in [str(fc_id).strip() for fc_id in model_gen_fc_ids]:
                            # If restaurant_ids is empty, process all restaurants in this foodcourt
                            if not model_gen_rest_ids:
                                force_retrain = True
                            # If restaurant_ids is specified, check if this restaurant matches
                            elif restaurant_id in [str(r_id).strip() for r_id in model_gen_rest_ids]:
                                force_retrain = True
                
                # Create item dict for checkpoint
                item_dict = {
                    "foodcourt_id": foodcourt_id,
                    "restaurant_id": restaurant_id,
                    "item_id": item_id if item_id else None,
                    "item_name": item_name if item_name else None
                }
                
                # Check checkpoint: skip if already completed (unless force retrain)
                if checkpoint_manager:
                    if checkpoint_manager.is_item_completed("preprocessing", item_dict) and not force_retrain:
                        items_processed += 1
                        if progress:
                            progress.set_current(item_idx)
                        continue
                
                # Generate output filename
                from src.util.pipeline_utils import get_file_name
                output_filename = get_file_name(foodcourt_id, restaurant_id, item_name, "preprocessing", item_id)
                output_fc_dir = os.path.join(OUTPUT_DIR, foodcourt_id)
                output_path = os.path.join(output_fc_dir, output_filename)
                
                # Skip if preprocessed data already exists and not forcing retrain
                if not force_retrain and os.path.exists(output_path):
                    # Item skipped (already processed) - no error, just skip tracking
                    items_processed += 1
                    # Update progress bar for skipped items too
                    if progress:
                        progress.set_current(item_idx)
                    continue
                
                # Mark as in progress before processing
                if checkpoint_manager:
                    checkpoint_manager.mark_item_in_progress("preprocessing", item_dict)

                processed_df, log_data = process_restaurant_df(df)
                
                # Note: Beverage/MRP items are now filtered in enrichment step (step2)
                # No need to handle them here - they should already be excluded
                
                # Only check for critical errors (invalid data structure)
                # Do NOT discard based on empty data, low span, etc. - assign to model 3 instead
                date_series = pd.to_datetime(processed_df["date"], errors="coerce") if not processed_df.empty else pd.Series(dtype='datetime64[ns]')
                
                if processed_df.empty:
                    # If completely empty after processing, log but still try to create file
                    warning_msg = f"Processed data is empty for {input_path} but will still save (assigned to model 3)"
                    pipeline_logger.log_warning("preprocessing", warning_msg)
                    # Don't print to console - only log to pipeline logs
                elif date_series.isna().all() and not processed_df.empty:
                    # If dates are invalid but we have data, try to fix or use model 3
                    warning_msg = f"Invalid dates for {input_path} but will still process (assigned to model 3)"
                    pipeline_logger.log_warning("preprocessing", warning_msg)
                    # Don't print to console - only log to pipeline logs
                elif not processed_df.empty and "count" in processed_df.columns and processed_df["count"].sum() == 0:
                    # All zeros - still process, will use model 3
                    pass  # Silent - no need to log

                # Save all items - no discarding based on data quality
                # Items will be assigned predict_model = 1, 2, or 3 based on their characteristics
                # Note: Beverage/MRP items are filtered in enrichment step (step2), so they won't reach preprocessing
                output_fc_dir = os.path.join(OUTPUT_DIR, foodcourt_id)
                os.makedirs(output_fc_dir, exist_ok=True)
                
                try:
                    # Keep date column as datetime - Excel can handle datetime objects
                    # No need to convert to string, as this causes comparison errors in downstream steps
                    if not processed_df.empty and "date" in processed_df.columns:
                        processed_df = processed_df.copy()
                        # Ensure date is datetime type (already converted earlier in process_restaurant_df)
                        processed_df["date"] = pd.to_datetime(processed_df["date"], errors="coerce")
                    
                    # Save as CSV
                    from src.util.pipeline_utils import save_dataframe_to_excel, get_file_locator, get_mongo_names
                    save_dataframe_to_excel(processed_df, Path(output_path), sheet_name="Preprocessed Data")  # Function now saves CSV
                    
                    # Validate file exists (for first 2 files per foodcourt)
                    # Track saved files for validation
                    if foodcourt_id not in saved_files_by_fc:
                        saved_files_by_fc[foodcourt_id] = []
                    saved_files_by_fc[foodcourt_id].append(Path(output_path))
                    
                    # Validate after first 2 files
                    if len(saved_files_by_fc[foodcourt_id]) == 2:
                        from src.util.pipeline_utils import validate_output_files
                        output_dir_path = Path(OUTPUT_DIR)
                        validate_output_files("preprocessing", output_dir_path, foodcourt_id, 
                                            saved_files_by_fc[foodcourt_id])
                    
                    # Mark as completed after successful save
                    if checkpoint_manager:
                        checkpoint_manager.mark_item_completed("preprocessing", item_dict)
                    
                    # Add to file locator
                    fc_name, rest_name = get_mongo_names(foodcourt_id, restaurant_id)
                    get_file_locator().add_file(
                        foodcourt_id, fc_name or foodcourt_id,
                        restaurant_id, rest_name or restaurant_id,
                        item_id, item_name,
                        "preprocessing",
                        Path(output_path)
                    )
                    
                    # Track success in restaurant tracker
                    if restaurant_tracker:
                        from src.util.pipeline_utils import get_all_names
                        fc_name, rest_name, item_name_val = get_all_names(
                            foodcourt_id, restaurant_id, item_id, output_filename
                        )
                        restaurant_tracker.add_success(
                            foodcourt_id, restaurant_id, item_id or item_name,
                            output_filename, "preprocessing",
                            foodcourt_name=fc_name, restaurant_name=rest_name, item_name=item_name or item_name_val
                        )
                    
                    processed_files += 1
                    items_processed += 1
                    
                    # Update progress bar
                    if progress:
                        progress.set_current(item_idx)
                    
                    # Log writing removed - only pipeline_logs.xlsx is used
                except Exception as exc:
                    # Mark as failed in checkpoint
                    if checkpoint_manager:
                        error_msg = f"Failed to write Excel file: {exc}"
                        checkpoint_manager.mark_item_failed("preprocessing", item_dict, error_msg)
                    error_msg = f"Failed to write preprocessed Excel for {foodcourt_id}/{restaurant_id}/{item_name}: {exc}"
                    logging.error(error_msg)
                    from src.util.pipeline_utils import get_pipeline_logger, get_mongo_names, get_file_locator
                    fc_name, rest_name = get_mongo_names(foodcourt_id, restaurant_id)
                    error_msg = f"Failed to write Excel file: {exc}"
                    get_pipeline_logger().log_preprocessing_error(
                        foodcourt_id, fc_name or foodcourt_id,
                        restaurant_id, rest_name or restaurant_id,
                        item_id, item_name,
                        error_msg
                    )
                    # Track error in restaurant tracker
                    if restaurant_tracker:
                        from src.util.pipeline_utils import get_all_names
                        fc_name, rest_name, item_name_val = get_all_names(
                            foodcourt_id, restaurant_id, item_id, output_filename if 'output_filename' in locals() else filename
                        )
                        restaurant_tracker.add_error(
                            foodcourt_id, restaurant_id, item_id or item_name,
                            error_msg, "preprocessing",
                            foodcourt_name=fc_name, restaurant_name=rest_name, item_name=item_name or item_name_val
                        )
                    # Add discard reason to file_locator
                    get_file_locator().add_discard_reason(
                        foodcourt_id, fc_name or foodcourt_id,
                        restaurant_id, rest_name or restaurant_id,
                        item_id, item_name,
                        "preprocessing",
                        f"Failed to write Excel file: {str(exc)[:50]}"
                    )
                    skipped_files += 1
                    items_processed += 1
                    # Update progress bar for failed items
                    if progress:
                        progress.set_current(item_idx)

    logging.info("Preprocessing complete. Files written: %d | Skipped: %d", processed_files, skipped_files)
    
    # Log process results
    results = {
        "Files processed": processed_files,
        "Files skipped": skipped_files,
        "Total files": processed_files + skipped_files
    }
    pipeline_logger.log_process_results("preprocessing", results)

    # Log discard report to preprocessing_logs sheet in pipeline_logs.xlsx
    if discard_entries:
        from src.util.pipeline_utils import get_pipeline_logger, get_mongo_names
        pipeline_logger = get_pipeline_logger()
        
        for entry in discard_entries:
            fc_id = entry[0]  # foodCourtId
            fc_name = entry[1]  # foodCourtName
            rest_id = entry[2]  # restaurantId
            rest_name = entry[3]  # restaurantName
            reason = entry[4]  # reason
            
            # Try to get names from MongoDB if not provided
            if not fc_name or not rest_name:
                fc_name_mongo, rest_name_mongo = get_mongo_names(fc_id, rest_id)
                if not fc_name:
                    fc_name = fc_name_mongo or fc_id
                if not rest_name:
                    rest_name = rest_name_mongo or rest_id
            
            # Extract item_id and item_name from reason if possible
            # Reason format: "beverage/MRP item: {item_name}"
            item_id = None
            item_name = None
            if "item:" in reason.lower():
                try:
                    item_name = reason.split("item:")[-1].strip()
                except:
                    pass
            
            # Log to preprocessing_logs
            pipeline_logger.log_preprocessing_log(
                fc_id, fc_name,
                rest_id, rest_name,
                item_id, item_name,
                reason
            )
        
        logging.info("Discard report logged to preprocessing_logs sheet in pipeline_logs.xlsx (%d entries)", len(discard_entries))


if __name__ == "__main__":
    main()

