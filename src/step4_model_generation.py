"""
Model training orchestrator.
Loads preprocessed data, performs dynamic date splits, delegates to model strategies,
and writes the resulting metrics/artifacts.
"""

import json
import logging
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from src.util.progress_bar import ProgressBar

from src.util.models import (
    AVAILABLE_MODELS,
    ItemResult,
    ModelTrainingResult,
    TrainingBundle,
    sanitize_name,
)
import src.util.models as models
# No longer using filter_utils - scanning preprocessing directory instead


LOG_FORMAT = "%(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


# Progress bar functionality moved to pipeline_utils.ProgressBar


BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "input_data"
# Read from output_data/preprocessing/{pipeline_type}/ (Excel files, one per item)
# Write models to output_data/trainedModel/{pipeline_type}/models/{model_type}/
# Write results to output_data/trainedModel/{pipeline_type}/results/
from src.util.pipeline_utils import get_output_base_dir, get_pipeline_type
OUTPUT_BASE = get_output_base_dir()
PIPELINE_TYPE = get_pipeline_type()
PREPROCESSED_DIR = OUTPUT_BASE / "preprocessing" / PIPELINE_TYPE
TRAINED_MODEL_DIR = OUTPUT_BASE / "trainedModel" / PIPELINE_TYPE
TRAINED_MODEL_MODELS_DIR = TRAINED_MODEL_DIR / "models"
TRAINED_MODEL_RESULTS_DIR = TRAINED_MODEL_DIR / "results"
# Create model type subdirectories
XGBOOST_MODELS_DIR = TRAINED_MODEL_MODELS_DIR / "XGBoost"
PROPHET_MODELS_DIR = TRAINED_MODEL_MODELS_DIR / "Prophet"
MOVING_AVG_MODELS_DIR = TRAINED_MODEL_MODELS_DIR / "MovingAverage"
ANALYSIS_PATH = TRAINED_MODEL_RESULTS_DIR / "analysis.xlsx"
VALIDATION_SUMMARY_PATH = TRAINED_MODEL_RESULTS_DIR / "validation_summary.xlsx"
FEATURE_COLUMNS = [
    "price",
    "isVeg",
    "isSpicy",
    "is_mon",
    "is_tue",
    "is_wed",
    "is_thu",
    "is_fri",
    "is_sat",
    "is_sun",
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

ITEM_IDENTIFIER_COL = "item_identifier"
MODEL_DATE = pd.Timestamp("2025-11-02")
MIN_TRAIN_ROWS = 20
MIN_FEATURES_REQUIRED = 8
MAX_PRE_WINDOW = pd.DateOffset(years=2)
MIN_PRE_WINDOW_DAYS = 180
VALIDATION_WINDOW_DAYS = 7

BEVERAGE_PATTERN = re.compile(r"(coffee|coffe|chai|tea|drink|juice|shake|mocktail)", re.IGNORECASE)

# Development environment flag: if True, reuse existing models instead of retraining
Development_Environment = True


def load_items_from_preprocessing(retrain_config: Optional[dict] = None) -> Optional[pd.DataFrame]:
    """
    Scan the preprocessing directory to find all items that have preprocessed data.
    Filter by foodcourt_ids from retrain.json if provided.
    
    File naming pattern: {foodcourt_id}_{restaurant_id}_{item_id}_{item_name}_preprocessing.csv (CSV format)
    
    Args:
        retrain_config: Optional retrain configuration dict from retrain.json
                       If provided and model_generation has foodcourt_ids, only process those.
                       If model_generation is empty, process all foodcourts.
    
    Returns:
        DataFrame with columns: foodcourt_id, restaurant_id, item_name
        Returns None if preprocessing directory doesn't exist or is empty.
    """
    if not PREPROCESSED_DIR.exists():
        logging.error("Preprocessed directory not found at %s", PREPROCESSED_DIR)
        return None
    
    # Get foodcourt_ids, restaurant_ids, and item_ids to filter by (if retrain_config provided)
    allowed_foodcourt_ids = None
    allowed_restaurant_ids = None
    item_ids_filter = None
    if retrain_config:
        from src.util.pipeline_utils import get_retrain_config_for_step
        step_config = get_retrain_config_for_step("model_generation")
        foodcourt_ids = step_config.get("foodcourt_ids", [])
        restaurant_ids = step_config.get("restaurant_ids", [])
        item_ids = step_config.get("item_ids", [])
        item_names = step_config.get("item_names", [])
        
        if foodcourt_ids:  # If list is not empty, filter by these foodcourt_ids
            allowed_foodcourt_ids = set(str(fc_id).strip() for fc_id in foodcourt_ids)
            logging.debug("Filtering by foodcourt_ids from retrain.json: %s", allowed_foodcourt_ids)
        if restaurant_ids:
            allowed_restaurant_ids = set(str(r_id).strip() for r_id in restaurant_ids)
            logging.debug("Filtering by restaurant_ids from retrain.json: %s", allowed_restaurant_ids)
        if item_ids:
            item_ids_filter = item_ids
            logging.debug("Filtering by item_ids from retrain.json: %d item filters", len(item_ids))
        if item_names:
            item_names_filter = item_names
            logging.debug("Filtering by item_names from retrain.json: %d item names", len(item_names))
        else:
            item_names_filter = None
        # If model_generation is empty, process all foodcourts (allowed_foodcourt_ids stays None)
    
    items_list = []
    
    # Scan preprocessing directory structure: preprocessing/FRI_LEVEL/{foodcourt_id}/*.csv
    for fc_dir in os.listdir(PREPROCESSED_DIR):
        fc_path = PREPROCESSED_DIR / fc_dir
        if not fc_path.is_dir():
            continue
        
        foodcourt_id = fc_dir.strip()
        
        # Filter by foodcourt_id if retrain_config specifies it
        if allowed_foodcourt_ids is not None and foodcourt_id not in allowed_foodcourt_ids:
            continue
        
        # Filter by restaurant_id if retrain_config specifies it
        # Note: restaurant_id is extracted from filename below, so we'll check it after parsing
        
        # Scan all CSV files in this foodcourt directory
        for filename in os.listdir(fc_path):
            if not filename.endswith('.csv'):
                continue
            
            # Parse filename: {foodcourt_id}_{restaurant_id}_{item_id}_{item_name}_preprocessing.csv
            # Remove .csv extension
            base_name = filename.replace("_preprocessing.csv", "").replace(".csv", "")
            
            # Split by underscore - but item_name might contain underscores
            # Pattern: {fc_id}_{rest_id}_{item_id}_{item_name_with_underscores}
            parts = base_name.split('_')
            
            if len(parts) < 3:
                logging.debug("Skipping file with unexpected format: %s", filename)
                continue
            
            # First part should be foodcourt_id (should match directory name)
            file_fc_id = parts[0]
            if file_fc_id != foodcourt_id:
                logging.debug("Foodcourt ID mismatch in filename %s: dir=%s, file=%s", 
                             filename, foodcourt_id, file_fc_id)
            
            # Second part is restaurant_id
            restaurant_id = parts[1]
            
            # Filter by restaurant_id if retrain_config specifies it
            if allowed_restaurant_ids is not None and restaurant_id not in allowed_restaurant_ids:
                continue
            
            # Determine if item_id is present (3rd part might be item_id or start of item_name)
            # If we have 4+ parts, assume 3rd part is item_id, rest is item_name
            # If we have 3 parts, assume no item_id, 3rd part is start of item_name
            item_id = None
            if len(parts) >= 4:
                # New format with item_id: {fc_id}_{rest_id}_{item_id}_{item_name}
                item_id = parts[2]
                item_name_from_filename = '_'.join(parts[3:])
            else:
                # Old format without item_id: {fc_id}_{rest_id}_{item_name}
                item_name_from_filename = '_'.join(parts[2:])
            
            # Try to read the original item_name from the CSV file (source of truth)
            # This ensures proper matching with retrain.json which uses original item names
            from src.util.pipeline_utils import get_item_name_from_excel, has_upstream_error
            item_name = None
            file_path = fc_path / filename
            # Removed exists() check - just try to read and catch exception
            try:
                item_name = get_item_name_from_excel(file_path)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                pass
            except Exception as e:
                logging.debug(f"Failed to read item name from {filename}: {e}")
            
            # Fallback: extract from filename (sanitized version) - only if CSV read failed
            if not item_name:
                item_name = item_name_from_filename
                logging.debug("Using item_name from filename for %s: %s (note: this is sanitized)", filename, item_name)
            
            # Check if upstream steps (enrich_data, preprocessing) have errors
            # If so, skip this item (no point processing if upstream failed)
            if has_upstream_error(foodcourt_id, restaurant_id, item_id, item_name, "model_generation"):
                logging.debug("Skipping %s/%s/%s - upstream step has error", foodcourt_id, restaurant_id, item_name)
                continue
            
            # Filter by item_ids or item_names if retrain_config specifies it
            # Always use item_name read from CSV (source of truth) for filtering
            if item_ids_filter is not None or item_names_filter is not None:
                from src.util.pipeline_utils import matches_item_filter
                if not matches_item_filter(foodcourt_id, restaurant_id, item_name, item_id, 
                                          item_ids_filter, item_names_filter):
                    logging.debug("Skipping %s/%s/%s - not in item_ids/item_names filter", foodcourt_id, restaurant_id, item_name)
                    continue
            
            items_list.append({
                "foodcourt_id": foodcourt_id,
                "restaurant_id": restaurant_id,
                "item_id": item_id if item_id else "",
                "item_name": item_name
            })
    
    if not items_list:
        logging.warning("No preprocessed files found in %s", PREPROCESSED_DIR)
        return None
    
    items_df = pd.DataFrame(items_list)
    logging.debug("Found %d items with preprocessed data in %s", len(items_df), PREPROCESSED_DIR)
    
    if allowed_foodcourt_ids or allowed_restaurant_ids:
        logging.debug("Filtered to %d items after applying retrain.json filter", len(items_df))
    
    return items_df


def find_item_in_preprocessed_data(
    foodcourt_id: str,
    restaurant_id: str,
    item_name: str,
    item_id: str = "",
) -> Optional[pd.DataFrame]:
    """
    Check if item data exists in preprocessed files.
    Returns the item DataFrame if found, None otherwise.
    Now reads from item-level CSV files: {F_id}_{R_id}_{item_id}_{item_name}_preprocessing.csv
    
    Args:
        foodcourt_id: Foodcourt ID
        restaurant_id: Restaurant ID
        item_name: Item name
        item_id: Optional item ID (if provided, will try filename with item_id first)
    """
    from src.util.pipeline_utils import get_file_name
    
    # Try with item_id first if provided
    if item_id:
        filename_with_id = get_file_name(foodcourt_id, restaurant_id, item_name, "preprocessing", item_id)
        item_csv_path = PREPROCESSED_DIR / foodcourt_id / filename_with_id
        # Removed exists() check - just try to read and catch exception
        try:
            df = pd.read_csv(item_csv_path)
            # Ensure date column is converted to datetime
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                # Drop rows with invalid dates
                df = df.dropna(subset=["date"])
            return df
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass  # Try next option
        except Exception as exc:
            logging.debug(f"Failed to load {item_csv_path}: {exc}")
    
    # Fallback: try without item_id
    filename = get_file_name(foodcourt_id, restaurant_id, item_name, "preprocessing")
    item_csv_path = PREPROCESSED_DIR / foodcourt_id / filename
    
    # Removed exists() check - just try to read and catch exception
    try:
        df = pd.read_csv(item_csv_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # Also try scanning the directory for files matching the pattern
        fc_path = PREPROCESSED_DIR / foodcourt_id
        # Removed fc_path.exists() check - just try to access
        try:
            # Look for files matching: {fc_id}_{rest_id}_{item_id}_{item_name}_preprocessing.csv
            # or {fc_id}_{rest_id}_{item_name}_preprocessing.csv
            from src.util.pipeline_utils import sanitize_name
            sanitized_item = sanitize_name(item_name)
            prefix = f"{foodcourt_id}_{restaurant_id}_"
            suffix = f"_{sanitized_item}_preprocessing.csv"
            
            matching_files = []
            # Scan all CSV files in the directory
            for file_path in fc_path.glob("*.csv"):
                filename = file_path.name
                # Check if it matches the pattern (with or without item_id)
                if filename.startswith(prefix) and filename.endswith(suffix):
                    matching_files.append(file_path)
                # Also try exact match without sanitization (in case item_name has special chars)
                elif filename.startswith(prefix) and "_preprocessing.csv" in filename:
                    # Extract potential item_name from filename
                    remaining = filename[len(prefix):].replace("_preprocessing.csv", "")
                    # Check if it ends with item_name (case-insensitive, flexible matching)
                    if remaining.endswith(sanitized_item) or sanitized_item in remaining:
                        matching_files.append(file_path)
            
            if matching_files:
                # Try the first matching file
                item_csv_path = matching_files[0]
                logging.debug(f"Found preprocessing file by pattern matching: {item_csv_path.name}")
            else:
                return None
        except Exception:
            # Directory scan failed, return None
            return None
    
    # Load the item's preprocessed data from CSV (if we found a file)
    try:
        df = pd.read_csv(item_csv_path)
        
        # Ensure date column is converted to datetime (same as load_preprocessed_file)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            # Drop rows with invalid dates
            df = df.dropna(subset=["date"])
        
        return df
    
    except Exception as exc:
        warning_msg = f"Error checking data for {foodcourt_id}/{restaurant_id}/{item_name}: {exc}"
        from src.util.pipeline_utils import get_pipeline_logger
        get_pipeline_logger().log_warning("model_generation", warning_msg)
        # Don't print to console - only log to pipeline logs
        return None


def load_preprocessed_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Missing 'date' column in preprocessed file")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "menuitemid" in df.columns:
        df[ITEM_IDENTIFIER_COL] = df["menuitemid"].astype(str)
    else:
        warning_msg = f"Column 'menuitemid' missing in {path}; using itemname as identifier"
        from src.util.pipeline_utils import get_pipeline_logger
        get_pipeline_logger().log_warning("model_generation", warning_msg)
        # Don't print to console - only log to pipeline logs
        if "itemname" not in df.columns:
            raise ValueError("Missing both 'menuitemid' and 'itemname' columns")
        df[ITEM_IDENTIFIER_COL] = df["itemname"].astype(str)

    df = df.dropna(subset=["date", ITEM_IDENTIFIER_COL])
    return df.sort_values("date").reset_index(drop=True)


def subset_features(df: pd.DataFrame) -> List[str]:
    cols = [col for col in FEATURE_COLUMNS if col in df.columns]
    if "available_for_model" in cols:
        cols.remove("available_for_model")
    return cols


def clamp_pre_window(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    min_allowed_date = MODEL_DATE - MAX_PRE_WINDOW
    return df[df["date"] >= min_allowed_date].copy()


def has_minimum_history(df: pd.DataFrame) -> bool:
    """
    Check if data has minimum history.
    Note: We no longer discard items - all items get processed.
    This function is kept for compatibility but returns True for any non-empty data.
    """
    if df.empty:
        return False
    # Accept any data - items will be assigned to appropriate model (1, 2, or 3)
    return True


def split_train_validation(df: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Split data into training and validation sets.
    No longer discards items - all items get processed with appropriate model assignment.
    """
    # Ensure date column is datetime
    if "date" not in df.columns:
        return None
    
    # Convert date column to datetime if it's not already
    if df["date"].dtype == 'object' or not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        if df.empty:
            return None
    
    try:
        pre_df = df[df["date"] < MODEL_DATE].copy()
    except TypeError as e:
        # Handle case where date comparison fails (e.g., mixed types)
        from src.util.pipeline_utils import get_pipeline_logger
        get_pipeline_logger().log_warning("model_generation", 
            f"Date comparison error in split_train_validation: {e}. Converting dates.")
        # Force conversion and retry
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        if df.empty:
            return None
        pre_df = df[df["date"] < MODEL_DATE].copy()
    
    if pre_df.empty:
        # If no pre-MODEL_DATE data, use all available data
        pre_df = df.copy()
        if pre_df.empty:
            return None

    pre_df = clamp_pre_window(pre_df)
    
    # Don't check minimum history - process all items
    # Items will be assigned to model 1, 2, or 3 based on predict_model value

    validation_end = MODEL_DATE - pd.Timedelta(days=1)
    validation_start = validation_end - pd.Timedelta(days=VALIDATION_WINDOW_DAYS - 1)
    train_end = validation_start - pd.Timedelta(days=1)

    train_df = pre_df[pre_df["date"] <= train_end].copy()
    validation_df = pre_df[(pre_df["date"] >= validation_start) & (pre_df["date"] <= validation_end)].copy()

    # If validation is empty, create from available data (even if less than 7 days)
    if validation_df.empty and not pre_df.empty:
        # Use last available days for validation (up to 7 days or whatever is available)
        validation_df = pre_df.tail(min(VALIDATION_WINDOW_DAYS, len(pre_df))).copy()
        train_df = pre_df.iloc[:-len(validation_df)].copy() if len(pre_df) > len(validation_df) else pd.DataFrame()
    
    # If train is empty but we have validation, use validation data for training too
    if train_df.empty and not validation_df.empty:
        train_df = validation_df.copy()
    
    # Process all items - even with minimal data
    if train_df.empty and validation_df.empty:
        # Last resort: use all data for both
        train_df = pre_df.copy()
        validation_df = pre_df.tail(min(VALIDATION_WINDOW_DAYS, len(pre_df))).copy()
    
    return train_df, validation_df


def append_analysis(summary_rows: List[Dict[str, object]], analysis_path: Path):
    if not summary_rows:
        return
    df = pd.DataFrame(summary_rows)
    if df.empty:
        return

    # ensure numeric columns
    for col in ("train_rows", "train_rmspe"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["train_rmspe"] = df["train_rmspe"].round(3)

    # build hyperlink column for high-error items
    def build_link(row: pd.Series) -> str:
        path = row.get("output_path", "")
        if not path or pd.isna(row["train_rmspe"]) or row["train_rmspe"] <= 10:
            return ""
        target = Path(path).resolve()
        return f'=HYPERLINK("{target.as_uri()}", "Open Folder")'

    df["data_path"] = df.apply(build_link, axis=1)

    ordered_cols = [
        "foodcourtid",
        "restaurant",
        "item_id",
        "item_name",
        "train_rows",
        "train_rmspe",
        "data_path",
    ]
    df = df[ordered_cols]
    df = df.sort_values("train_rmspe", ascending=False)

    try:
        # Sanitize string columns to handle Unicode encoding issues
        df_to_write = df.copy()
        for col in df_to_write.columns:
            if df_to_write[col].dtype == 'object':
                df_to_write[col] = df_to_write[col].astype(str).apply(
                    lambda x: x.encode('utf-8', errors='replace').decode('utf-8') if pd.notna(x) else x
                )
        df_to_write.to_excel(analysis_path, index=False, engine='openpyxl')
    except PermissionError:
        logging.error("Permission denied writing analysis to %s", analysis_path)
    except Exception as exc:
        logging.error("Error writing analysis to %s: %s", analysis_path, exc)


def load_postprocessing_results(item_dir: Path, item_slug: str) -> Optional[pd.DataFrame]:
    """Load postprocessing results if available."""
    postprocess_path = item_dir / f"{item_slug}_prediction_redistributed.csv"
    if postprocess_path.exists():
        try:
            return pd.read_csv(postprocess_path)
        except Exception as exc:
            warning_msg = f"Failed to load postprocessing results for {item_slug}: {exc}"
            from src.util.pipeline_utils import get_pipeline_logger
            get_pipeline_logger().log_warning("model_generation", warning_msg)
            # Don't print to console - only log to pipeline logs
    return None


def calculate_avg_percent_error(pct_errors: np.ndarray) -> float:
    """Calculate average percent error (as percentage, not decimal)."""
    if len(pct_errors) == 0:
        return float("nan")
    valid = ~np.isnan(pct_errors)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(pct_errors[valid]) * 100.0)


def calculate_avg_percent_error_capped(pct_errors: np.ndarray, cap: float = 100.0) -> float:
    """Calculate average percent error capped at specified value (as percentage)."""
    if len(pct_errors) == 0:
        return float("nan")
    valid = ~np.isnan(pct_errors)
    if not np.any(valid):
        return float("nan")
    capped = np.clip(np.abs(pct_errors[valid]) * 100.0, 0, cap)
    return float(np.mean(capped))


def build_comprehensive_validation_summary(
    validation_rows: List[Dict[str, object]],
    output_base_dir: Path,
) -> pd.DataFrame:
    """
    Build a comprehensive validation summary with all models side by side.
    One row per item showing XGBOOST, decay_based, and weekday_aware results.
    """
    if not validation_rows:
        return pd.DataFrame()
    
    # Group validation rows by item (foodcourtid, restaurant, menuitemname)
    items_dict: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    
    for row in validation_rows:
        key = (
            str(row.get("foodcourtid", "")),
            str(row.get("restaurant", "")),
            str(row.get("menuitemname", "")),
        )
        
        if key not in items_dict:
            items_dict[key] = {
                "foodcourtid": row.get("foodcourtid", ""),
                "foodcourtname": row.get("foodcourtname", ""),
                "restaurant": row.get("restaurant", ""),
                "restaurantname": row.get("restaurantname", ""),
                "menuitemname": row.get("menuitemname", ""),
                "active_days": row.get("active_days", 0),
                "total_count": row.get("total_count", 0.0),
            }
        
        model_name = str(row.get("model_name", ""))
        
        # Extract model results
        if "xgboost" in model_name.lower():
            items_dict[key]["XGBOOST_predicted_count"] = row.get("predicted_count", 0.0)
        elif "prophet" in model_name.lower():
            items_dict[key]["PROPHET_predicted_count"] = row.get("predicted_count", 0.0)
        elif "decay" in model_name.lower():
            items_dict[key]["decay_based_predicted_count"] = row.get("predicted_count", 0.0)
        elif "weekday" in model_name.lower():
            items_dict[key]["weekday_aware_predicted_count"] = row.get("predicted_count", 0.0)
    
    # Now load detailed validation results and postprocessing for each item
    comprehensive_rows = []
    
    for key, item_data in items_dict.items():
        foodcourt_id, restaurant_id, item_name = key
        
        # Find item directory
        item_slug = sanitize_name(item_name)
        item_dir = output_base_dir / foodcourt_id / restaurant_id / item_slug
        
        # Extract item_id from validation_rows if available
        item_id = ""
        for row in validation_rows:
            if (str(row.get("foodcourtid", "")) == foodcourt_id and 
                str(row.get("restaurant", "")) == restaurant_id and
                str(row.get("menuitemname", "")) == item_name):
                item_id = str(row.get("item_id", ""))
                break
        
        if not item_dir.exists():
            continue
        
        # Initialize row with basic info
        row = item_data.copy()
        
        # Load validation results for each model type
        # XGBOOST
        xgb_val_path = item_dir / f"{item_slug}_validation_results.csv"
        if xgb_val_path.exists():
            try:
                xgb_df = pd.read_csv(xgb_val_path)
                if "predicted_count" in xgb_df.columns and "actual_count" in xgb_df.columns:
                    row["XGBOOST_predicted_count"] = float(xgb_df["predicted_count"].sum())
                    if "pct_error" in xgb_df.columns:
                        pct_errors = xgb_df["pct_error"].values / 100.0  # Convert from percentage to decimal
                        row["XGBOOST_avg_percent_error"] = calculate_avg_percent_error(pct_errors)
                        row["XGBOOST_avg_percent_error_capped"] = calculate_avg_percent_error_capped(pct_errors)
            except Exception as exc:
                warning_msg = f"Failed to load XGBOOST validation for {item_slug}: {exc}"
                from src.util.pipeline_utils import get_pipeline_logger
                get_pipeline_logger().log_warning("model_generation", warning_msg)
                # Don't print to console - only log to pipeline logs
        
        # Prophet - load from results directory
        from src.util.pipeline_utils import get_result_file_name, get_output_base_dir, get_pipeline_type
        OUTPUT_BASE = get_output_base_dir()
        PIPELINE_TYPE = get_pipeline_type()
        TRAINED_MODEL_RESULTS_DIR = OUTPUT_BASE / "trainedModel" / PIPELINE_TYPE / "results"
        prophet_val_filename = get_result_file_name(
            foodcourt_id, restaurant_id, item_name, "model_generation", "Prophet", item_id, "validation"
        )
        prophet_val_path = TRAINED_MODEL_RESULTS_DIR / prophet_val_filename
        if prophet_val_path.exists():
            try:
                prophet_df = pd.read_csv(prophet_val_path)
                if "predicted_count" in prophet_df.columns and "actual_count" in prophet_df.columns:
                    row["PROPHET_predicted_count"] = float(prophet_df["predicted_count"].sum())
                    if "pct_error" in prophet_df.columns:
                        pct_errors = prophet_df["pct_error"].values / 100.0
                        row["PROPHET_avg_percent_error"] = calculate_avg_percent_error(pct_errors)
                        row["PROPHET_avg_percent_error_capped"] = calculate_avg_percent_error_capped(pct_errors)
            except Exception as exc:
                warning_msg = f"Failed to load Prophet validation for {item_slug}: {exc}"
                from src.util.pipeline_utils import get_pipeline_logger
                get_pipeline_logger().log_warning("model_generation", warning_msg)
        
        # Decay-based
        decay_val_path = item_dir / f"{item_slug}_validation_decay_results.csv"
        if decay_val_path.exists():
            try:
                decay_df = pd.read_csv(decay_val_path)
                if "predicted_count" in decay_df.columns and "actual_count" in decay_df.columns:
                    row["decay_based_predicted_count"] = float(decay_df["predicted_count"].sum())
                    if "pct_error" in decay_df.columns:
                        pct_errors = decay_df["pct_error"].values / 100.0
                        row["decay_based_avg_percent_error"] = calculate_avg_percent_error(pct_errors)
                        row["decay_based_avg_percent_error_capped"] = calculate_avg_percent_error_capped(pct_errors)
            except Exception as exc:
                warning_msg = f"Failed to load decay validation for {item_slug}: {exc}"
                from src.util.pipeline_utils import get_pipeline_logger
                get_pipeline_logger().log_warning("model_generation", warning_msg)
                # Don't print to console - only log to pipeline logs
        
        # Weekday-aware
        weekday_val_path = item_dir / f"{item_slug}_validation_weekday_results.csv"
        if weekday_val_path.exists():
            try:
                weekday_df = pd.read_csv(weekday_val_path)
                if "predicted_count" in weekday_df.columns and "actual_count" in weekday_df.columns:
                    row["weekday_aware_predicted_count"] = float(weekday_df["predicted_count"].sum())
                    if "pct_error" in weekday_df.columns:
                        pct_errors = weekday_df["pct_error"].values / 100.0
                        row["weekday_aware_avg_percent_error"] = calculate_avg_percent_error(pct_errors)
                        row["weekday_aware_avg_percent_error_capped"] = calculate_avg_percent_error_capped(pct_errors)
            except Exception as exc:
                warning_msg = f"Failed to load weekday validation for {item_slug}: {exc}"
                from src.util.pipeline_utils import get_pipeline_logger
                get_pipeline_logger().log_warning("model_generation", warning_msg)
                # Don't print to console - only log to pipeline logs
        
        # Load postprocessing results if available
        postprocess_df = load_postprocessing_results(item_dir, item_slug)
        if postprocess_df is not None:
            # Check which model was used (check for XGBOOST first, then others)
            if "predicted_count_postprocessing" in postprocess_df.columns and "error_pct_postprocessing" in postprocess_df.columns:
                pct_errors_post = postprocess_df["error_pct_postprocessing"].values / 100.0
                
                # Determine which model this postprocessing is for
                # Check if XGBOOST validation exists
                if xgb_val_path.exists():
                    row["XGBOOST_avg_percent_error_postprocessing"] = calculate_avg_percent_error(pct_errors_post)
                    row["XGBOOST_avg_postprocessing_capped_error"] = calculate_avg_percent_error_capped(pct_errors_post)
                # Check if Prophet validation exists
                if prophet_val_path.exists():
                    row["PROPHET_avg_percent_error_postprocessing"] = calculate_avg_percent_error(pct_errors_post)
                    row["PROPHET_avg_postprocessing_capped_error"] = calculate_avg_percent_error_capped(pct_errors_post)
                # Check if decay validation exists
                if decay_val_path.exists():
                    row["decay_based_avg_percent_error_postprocessing"] = calculate_avg_percent_error(pct_errors_post)
                    row["decay_based_avg_postprocessing_capped_error"] = calculate_avg_percent_error_capped(pct_errors_post)
                # Check if weekday validation exists
                if weekday_val_path.exists():
                    row["weekday_aware_avg_percent_error_postprocessing"] = calculate_avg_percent_error(pct_errors_post)
                    row["weekday_aware_avg_postprocessing_capped_error"] = calculate_avg_percent_error_capped(pct_errors_post)
        
        # Add folder link
        folder_path = item_dir.resolve()
        row["item_results_folder"] = str(folder_path)
        row["item_results_folder_link"] = f'=HYPERLINK("{folder_path.as_uri()}", "Open Folder")'
        
        comprehensive_rows.append(row)
    
    # Create DataFrame with ordered columns
    if not comprehensive_rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(comprehensive_rows)
    
    # Define column order
    base_cols = [
        "foodcourtid", "foodcourtname", "restaurant", "restaurantname", "menuitemname",
        "active_days", "total_count"
    ]
    
    xgb_cols = [
        "XGBOOST_predicted_count",
        "XGBOOST_avg_percent_error",
        "XGBOOST_avg_percent_error_capped",
        "XGBOOST_avg_percent_error_postprocessing",
        "XGBOOST_avg_postprocessing_capped_error",
    ]
    
    prophet_cols = [
        "PROPHET_predicted_count",
        "PROPHET_avg_percent_error",
        "PROPHET_avg_percent_error_capped",
        "PROPHET_avg_percent_error_postprocessing",
        "PROPHET_avg_postprocessing_capped_error",
    ]
    
    decay_cols = [
        "decay_based_predicted_count",
        "decay_based_avg_percent_error",
        "decay_based_avg_percent_error_capped",
        "decay_based_avg_percent_error_postprocessing",
        "decay_based_avg_postprocessing_capped_error",
    ]
    
    weekday_cols = [
        "weekday_aware_predicted_count",
        "weekday_aware_avg_percent_error",
        "weekday_aware_avg_percent_error_capped",
        "weekday_aware_avg_percent_error_postprocessing",
        "weekday_aware_avg_postprocessing_capped_error",
    ]
    
    link_cols = ["item_results_folder", "item_results_folder_link"]
    
    # Build final column order (only include columns that exist)
    final_cols = base_cols.copy()
    for col_list in [xgb_cols, prophet_cols, decay_cols, weekday_cols, link_cols]:
        for col in col_list:
            if col in df.columns:
                final_cols.append(col)
    
    # Add any remaining columns
    for col in df.columns:
        if col not in final_cols:
            final_cols.append(col)
    
    return df[final_cols]


def write_validation_summary(validation_rows: List[Dict[str, object]], summary_path: Path, output_base_dir: Path):
    """Write comprehensive validation summary with all models side by side."""
    if not validation_rows:
        return
    
    # Build comprehensive summary
    summary_df = build_comprehensive_validation_summary(validation_rows, output_base_dir)
    
    if summary_df.empty:
        warning_msg = "No validation summary data to write"
        from src.util.pipeline_utils import get_pipeline_logger
        get_pipeline_logger().log_warning("model_generation", warning_msg)
        # Don't print to console - only log to pipeline logs
        return
    
    try:
        # Save as Excel instead of CSV
        from src.util.pipeline_utils import save_dataframe_to_excel
        # Ensure CSV extension
        if summary_path.suffix != ".csv":
            summary_path = summary_path.with_suffix(".csv")
        save_dataframe_to_excel(summary_df, summary_path, sheet_name="Validation Summary")  # Function now saves CSV
        logging.debug("Saved comprehensive validation summary to %s", summary_path)
    except PermissionError:
        logging.error("Permission denied writing validation summary to %s", summary_path)


def load_existing_model_results(
    item_slug: str,
    output_dir: Path,
    bundle: TrainingBundle,
) -> Optional[ModelTrainingResult]:
    """
    Check if model and validation results already exist.
    If Development_Environment is True, load and reuse them.
    """
    if not Development_Environment:
        return None
    
    # Check for XGBoost model (.pkl file)
    xgb_model_path = output_dir / f"{item_slug}.pkl"
    validation_csv_path = output_dir / f"{item_slug}_validation_results.csv"
    
    # Check for WeeklyMovingAverage model config
    ma_model_config_path = output_dir / f"{item_slug}_model_config.pkl"
    validation_decay_csv_path = output_dir / f"{item_slug}_validation_decay_results.csv"
    
    # Try to load XGBoost results
    if xgb_model_path.exists() and validation_csv_path.exists():
        try:
            logging.debug("Reusing existing XGBoost model for %s", item_slug)
            # Load validation results
            val_df = pd.read_csv(validation_csv_path)
            if "actual_count" in val_df.columns and "predicted_count" in val_df.columns:
                y_val = val_df["actual_count"].values
                y_val_pred = val_df["predicted_count"].values
                val_pct = models._compute_percentage_errors(y_val, y_val_pred)
                
                # Load train results if available
                train_csv_path = output_dir / f"{item_slug}_train_results.csv"
                train_rmse = 0.0
                train_rmspe = 0.0
                if train_csv_path.exists():
                    train_df = pd.read_csv(train_csv_path)
                    if "actual_count" in train_df.columns and "predicted_count" in train_df.columns:
                        y_train = train_df["actual_count"].values
                        y_train_pred = train_df["predicted_count"].values
                        train_pct = models._compute_percentage_errors(y_train, y_train_pred)
                        train_rmse = float(np.sqrt(np.mean((y_train_pred - y_train) ** 2)))
                        train_rmspe = models._compute_rmspe(train_pct)
                
                validation_summary = models._build_validation_summary(bundle, y_val, y_val_pred, val_pct)
                validation_summary["model_name"] = "xgboost"
                
                item_result = ItemResult(
                    foodcourt_id=bundle.metadata.get("foodcourtid", ""),
                    restaurant_id=bundle.metadata.get("restaurant", ""),
                    item_id=bundle.metadata.get("item_slug", ""),
                    item_name=bundle.metadata.get("item_name", item_slug),
                    train_rows=len(bundle.train_df),
                    train_rmse=train_rmse,
                    train_rmspe=train_rmspe,
                )
                
                return ModelTrainingResult(
                    item_result=item_result,
                    validation_summary=validation_summary,
                    model_name="xgboost",
                    output_path=str(output_dir.resolve()),
                )
        except Exception as exc:
            warning_msg = f"Failed to load existing XGBoost results for {item_slug}: {exc}"
            from src.util.pipeline_utils import get_pipeline_logger
            get_pipeline_logger().log_warning("model_generation", warning_msg)
            # Don't print to console - only log to pipeline logs
    
    # Try to load WeeklyMovingAverage results
    if ma_model_config_path.exists() and validation_decay_csv_path.exists():
        try:
            logging.debug("Reusing existing WeeklyMovingAverage model for %s", item_slug)
            # Load decay validation results
            val_decay_df = pd.read_csv(validation_decay_csv_path)
            y_val = val_decay_df["actual_count"].values if "actual_count" in val_decay_df.columns else np.array([])
            y_val_pred_decay = val_decay_df["predicted_count"].values if "predicted_count" in val_decay_df.columns else np.array([])
            
            # Load weekday validation results if available
            validation_weekday_csv_path = output_dir / f"{item_slug}_validation_weekday_results.csv"
            y_val_pred_weekday = y_val_pred_decay.copy()
            if validation_weekday_csv_path.exists():
                val_weekday_df = pd.read_csv(validation_weekday_csv_path)
                if "predicted_count" in val_weekday_df.columns:
                    y_val_pred_weekday = val_weekday_df["predicted_count"].values
            
            val_pct_decay = models._compute_percentage_errors(y_val, y_val_pred_decay) if len(y_val) > 0 else np.array([])
            val_pct_weekday = models._compute_percentage_errors(y_val, y_val_pred_weekday) if len(y_val) > 0 else np.array([])
            
            validation_summary_decay = models._build_validation_summary(bundle, y_val, y_val_pred_decay, val_pct_decay) if len(y_val) > 0 else {}
            validation_summary_weekday = models._build_validation_summary(bundle, y_val, y_val_pred_weekday, val_pct_weekday) if len(y_val) > 0 else {}
            
            # Load train results if available
            train_decay_csv_path = output_dir / f"{item_slug}_train_decay_results.csv"
            train_rmse = 0.0
            train_rmspe = 0.0
            if train_decay_csv_path.exists():
                train_df = pd.read_csv(train_decay_csv_path)
                if "actual_count" in train_df.columns and "predicted_count" in train_df.columns:
                    y_train = train_df["actual_count"].values
                    y_train_pred = train_df["predicted_count"].values
                    train_pct = models._compute_percentage_errors(y_train, y_train_pred)
                    train_rmse = float(np.sqrt(np.mean((y_train_pred - y_train) ** 2)))
                    train_rmspe = models._compute_rmspe(train_pct)
            
            item_result = ItemResult(
                foodcourt_id=bundle.metadata.get("foodcourtid", ""),
                restaurant_id=bundle.metadata.get("restaurant", ""),
                item_id=bundle.metadata.get("item_slug", ""),
                item_name=bundle.metadata.get("item_name", item_slug),
                train_rows=len(bundle.train_df),
                train_rmse=train_rmse,
                train_rmspe=train_rmspe,
            )
            
            # Return combined summary (similar to WeeklyMovingAverageModel.train)
            combined_summary = {
                **validation_summary_decay,
                "decay_rmse": float(np.sqrt(np.mean((y_val_pred_decay - y_val) ** 2))) if len(y_val) else float("nan"),
                "weekday_rmse": float(np.sqrt(np.mean((y_val_pred_weekday - y_val) ** 2))) if len(y_val) else float("nan"),
                "decay_val_rmspe": models._compute_rmspe(val_pct_decay) if len(val_pct_decay) > 0 else float("nan"),
                "weekday_val_rmspe": models._compute_rmspe(val_pct_weekday) if len(val_pct_weekday) > 0 else float("nan"),
            }
            
            # Create separate summaries for both methods
            validation_summary = {
                "_decay_summary": {**validation_summary_decay, "model_name": "weekly_moving_average_decay"},
                "_weekday_summary": {**validation_summary_weekday, "model_name": "weekly_moving_average_weekday"},
            }
            
            return ModelTrainingResult(
                item_result=item_result,
                validation_summary=validation_summary,
                model_name="weekly_moving_average",
                output_path=str(output_dir.resolve()),
            )
        except Exception as exc:
            warning_msg = f"Failed to load existing WeeklyMovingAverage results for {item_slug}: {exc}"
            from src.util.pipeline_utils import get_pipeline_logger
            get_pipeline_logger().log_warning("model_generation", warning_msg)
            # Don't print to console - only log to pipeline logs
    
    return None


def process_item(
    df_item: pd.DataFrame,
    feature_cols: List[str],
    item_slug: str,
    output_dir: Path,
) -> List[ModelTrainingResult]:
    """
    Train all applicable models for a single item simultaneously.
    Returns a list of ModelTrainingResult objects (one per successfully trained model).
    """
    window_split = split_train_validation(df_item)
    if window_split is None:
        # This should rarely happen now, but if it does, create minimal splits
        warning_msg = f"Unable to create train/validation windows for {item_slug}; creating minimal splits"
        from src.util.pipeline_utils import get_pipeline_logger
        get_pipeline_logger().log_warning("model_generation", warning_msg)
        # Don't print to console - only log to pipeline logs
        if df_item.empty:
            return []
        # Use all data for both train and validation if split fails
        train_df = df_item.copy()
        validation_df = df_item.tail(min(VALIDATION_WINDOW_DAYS, len(df_item))).copy()
    else:
        train_df, validation_df = window_split
    predict_value = 1
    if "predict_model" in df_item.columns and not df_item["predict_model"].empty:
        try:
            predict_value = int(df_item["predict_model"].iloc[0])
        except Exception:
            predict_value = 1

    bundle = TrainingBundle(
        train_df=train_df,
        validation_df=validation_df,
        full_item_df=df_item,
        feature_cols=feature_cols,
        output_dir=output_dir,
        metadata={
            "item_slug": item_slug,
            "item_name": str(df_item["itemname"].iloc[0]) if "itemname" in df_item.columns else item_slug,
            "foodcourtid": str(df_item["foodcourtid"].iloc[0]) if "foodcourtid" in df_item.columns else "",
            "foodcourtname": str(df_item["foodcourtname"].iloc[0]) if "foodcourtname" in df_item.columns else "",
            "restaurant": str(df_item["restaurant"].iloc[0]),
            "restaurantname": str(df_item["restaurantname"].iloc[0]) if "restaurantname" in df_item.columns else "",
            "model_date": MODEL_DATE.isoformat(),
            "predict_model": str(predict_value),
        },
    )

    # Train all applicable models simultaneously
    results = []
    
    # Determine which models to train based on predict_model value
    # predict_model = 1: Try XGBoost, Prophet, then MovingAverage as fallback
    # predict_model = 2 or 3: Only MovingAverage
    models_to_try = []
    if predict_value == 1:
        # Try all advanced models first
        models_to_try = [m for m in AVAILABLE_MODELS if m.name in ["xgboost", "prophet"]]
        # Add MovingAverage as fallback
        ma_model = next((m for m in AVAILABLE_MODELS if m.name == "weekly_moving_average"), None)
        if ma_model:
            models_to_try.append(ma_model)
    else:
        # Only MovingAverage for predict_model 2 or 3
        ma_model = next((m for m in AVAILABLE_MODELS if m.name == "weekly_moving_average"), None)
        if ma_model:
            models_to_try = [ma_model]
    
    # Train all applicable models
    for model in models_to_try:
        if model.can_train(bundle):
            try:
                result = model.train(bundle)
                if result:
                    results.append(result)
            except Exception as exc:
                warning_msg = f"Failed to train {model.name} for {item_slug}: {exc}"
                from src.util.pipeline_utils import get_pipeline_logger
                get_pipeline_logger().log_warning("model_generation", warning_msg)
                logging.warning(warning_msg)
    
    # If no models trained and predict_model = 1, try MovingAverage as last resort
    if not results and predict_value == 1:
        ma_model = next((m for m in AVAILABLE_MODELS if m.name == "weekly_moving_average"), None)
        if ma_model and ma_model.can_train(bundle):
            warning_msg = f"Using WeeklyMovingAverage as fallback for {item_slug}"
            from src.util.pipeline_utils import get_pipeline_logger
            get_pipeline_logger().log_warning("model_generation", warning_msg)
            try:
                result = ma_model.train(bundle)
                if result:
                    results.append(result)
            except Exception as exc:
                logging.warning(f"Failed to train MovingAverage fallback: {exc}")
    
    if not results:
        logging.warning(f"No models could be trained for {item_slug}")
    
    return results


def process_single_item(
    item_df: pd.DataFrame,
    feature_cols: List[str],
    foodcourt_id: str,
    restaurant_id: str,
    item_name: str,
    output_base: Path,
    item_id: str = "",
    restaurant_tracker=None,
) -> Tuple[Optional[Dict[str, object]], List[Dict[str, object]]]:
    """
    Process a single item for training - trains all applicable models simultaneously.
    Returns (restaurant_row, list of validation_summaries) if successful, (None, []) otherwise.
    Now uses new structure: models go to trainedModel/models/{model_type}/, results to trainedModel/results/
    
    Args:
        restaurant_tracker: Optional RestaurantTracker instance for saving metrics
    """
    item_slug = sanitize_name(item_name)
    # Create a temporary directory for this item's processing
    # Models will be saved to model type directories, results to results directory
    item_out_dir = output_base / "temp" / foodcourt_id / restaurant_id / item_slug
    item_out_dir.mkdir(parents=True, exist_ok=True)

    results = process_item(item_df, feature_cols, item_slug, item_out_dir)
    if results and len(results) > 0:
        # Use the first result for restaurant_row (for backward compatibility)
        first_result = results[0]
        item_result: ItemResult = first_result.item_result
        
        restaurant_row = {
            "foodcourtid": item_result.foodcourt_id,
            "restaurant": item_result.restaurant_id,
            "item_id": item_result.item_id,
            "item_name": item_result.item_name,
            "train_rows": item_result.train_rows,
            "train_rmspe": item_result.train_rmspe,
            "model_name": first_result.model_name,  # Keep first model name for compatibility
            "output_path": first_result.output_path,
        }
        
        # Collect all validation summaries
        validation_summaries = []
        for result in results:
            if result.validation_summary:
                validation_summaries.append(result.validation_summary)
        
        # Calculate and save metrics to restaurant tracker for all models
        if restaurant_tracker:
            try:
                    from src.util.restaurant_tracker import calculate_metrics_from_df
                    from src.util.pipeline_utils import get_result_file_name, get_pipeline_type, get_file_name
                    from src.util.path_utils import get_output_base_dir
                    import pandas as pd
                    import os
                    
                    PIPELINE_TYPE = get_pipeline_type()
                    OUTPUT_BASE = get_output_base_dir()
                    TRAINED_MODEL_RESULTS_DIR = OUTPUT_BASE / "trainedModel" / PIPELINE_TYPE / "results"
                    
                    # Get predict_model value to determine which models should have been tried
                    predict_model = 1  # Default
                    if "predict_model" in item_df.columns and not item_df["predict_model"].empty:
                        try:
                            predict_model = int(item_df["predict_model"].iloc[0])
                        except Exception:
                            predict_model = 1
                    
                    # Determine which models should have been tried
                    models_to_check = []
                    if predict_model == 1:
                        # Try all advanced models: XGBoost, Prophet, then MovingAverage as fallback
                        models_to_check = ["XGBoost", "Prophet", "MovingAverage"]
                    else:
                        # predict_model = 2 or 3: Only MovingAverage
                        models_to_check = ["MovingAverage"]
                    
                    # Map model names from results to display names
                    trained_model_names = {}
                    for result in results:
                        model_name = result.model_name
                        if model_name == "xgboost":
                            trained_model_names["XGBoost"] = result
                        elif model_name == "prophet":
                            trained_model_names["Prophet"] = result
                        elif model_name == "weekly_moving_average":
                            trained_model_names["MovingAverage"] = result
                    
                    # Track all models
                    for model_display in models_to_check:
                        # Get validation CSV file path (models save CSV files to results directory)
                        validation_filename = get_result_file_name(
                            foodcourt_id, restaurant_id, item_name, "model_generation",
                            model_display, item_id, "validation"
                        )
                        validation_path = TRAINED_MODEL_RESULTS_DIR / validation_filename
                        
                        # Also check training CSV file path
                        training_filename = get_result_file_name(
                            foodcourt_id, restaurant_id, item_name, "model_generation",
                            model_display, item_id, "training"
                        )
                        training_path = TRAINED_MODEL_RESULTS_DIR / training_filename
                        
                        # Excel file path (if it exists, both training and validation are in same Excel)
                        # Note: get_file_name doesn't support model_name, so we construct it manually
                        # This is for backward compatibility only - we now use CSV files
                        item_slug = sanitize_name(item_name)
                        model_slug = sanitize_name(model_display)
                        if item_id:
                            excel_filename = f"{foodcourt_id}_{restaurant_id}_{item_id}_{item_slug}_model_generation_{model_slug}.xlsx"
                        else:
                            excel_filename = f"{foodcourt_id}_{restaurant_id}_{item_slug}_model_generation_{model_slug}.xlsx"
                        excel_path = TRAINED_MODEL_RESULTS_DIR / excel_filename
                        
                        # Check if model was actually generated (CSV files or Excel file exists)
                        model_generated = (validation_path.exists() or training_path.exists() or excel_path.exists()) if (validation_path and training_path and excel_path) else False
                        
                        # Determine if this model was trained
                        used = model_display in trained_model_names
                        
                        # Determine reason
                        if used:
                            if predict_model == 1:
                                if model_display in ["XGBoost", "Prophet"]:
                                    reason = f"Trained successfully - sufficient data for {model_display}"
                                else:
                                    reason = f"Trained as fallback - {model_display}"
                            else:
                                reason = f"Trained - predict_model={predict_model}"
                        else:
                            if not model_generated:
                                if model_display == "XGBoost":
                                    reason = f"Not generated - insufficient non-zero data (need at least {20} rows with count > 0)"
                                elif model_display == "Prophet":
                                    reason = "Not generated - insufficient data (need at least 20 rows)"
                                else:
                                    reason = "Not generated - model could not train"
                            else:
                                reason = f"Not trained - model training failed or insufficient data"
                        
                        # Get training and validation metrics
                        training_metrics = {}
                        validation_metrics = {}
                        training_file_path = None
                        validation_file_path = None
                        
                        if model_generated:
                            # Try to read from CSV files first (preferred - models.py saves these)
                            if training_path.exists():
                                try:
                                    train_df = pd.read_csv(training_path)
                                    training_metrics = calculate_metrics_from_df(
                                        train_df, actual_col="actual_count", 
                                        predicted_col="predicted_count", error_pct_col="pct_error"
                                    )
                                    training_file_path = str(training_path)
                                except Exception:
                                    pass
                            
                            if validation_path.exists():
                                try:
                                    val_df = pd.read_csv(validation_path)
                                    validation_metrics = calculate_metrics_from_df(
                                        val_df, actual_col="actual_count", 
                                        predicted_col="predicted_count", error_pct_col="pct_error"
                                    )
                                    validation_file_path = str(validation_path)
                                except Exception:
                                    pass
                            
                            # Fallback: Try Excel file if CSV files don't exist
                            if not training_file_path and excel_path.exists():
                                try:
                                    # Read Training Data sheet
                                    train_df = pd.read_excel(excel_path, sheet_name="Training Data")
                                    training_metrics = calculate_metrics_from_df(
                                        train_df, actual_col="actual_count", 
                                        predicted_col="predicted_count", error_pct_col="pct_error"
                                    )
                                    training_file_path = str(excel_path)
                                except Exception:
                                    pass
                            
                            if not validation_file_path and excel_path.exists():
                                try:
                                    # Read Validation Data sheet
                                    val_df = pd.read_excel(excel_path, sheet_name="Validation Data")
                                    validation_metrics = calculate_metrics_from_df(
                                        val_df, actual_col="actual_count", 
                                        predicted_col="predicted_count", error_pct_col="pct_error"
                                    )
                                    validation_file_path = str(excel_path)
                                except Exception:
                                    pass
                            
                            # If still no metrics, set defaults
                            if not training_metrics:
                                training_metrics = {
                                    "abs_avg_deviation": 0.0, "avg_abs_accuracy_pct": 0.0,
                                    "avg_abs_accuracy_pct_capped": 0.0,
                                    "total_days": 0, "active_days": 0
                                }
                            
                            if not validation_metrics:
                                validation_metrics = {
                                    "abs_avg_deviation": 0.0, "avg_abs_accuracy_pct": 0.0,
                                    "avg_abs_accuracy_pct_capped": 0.0,
                                    "total_days": 0, "active_days": 0
                                }
                            
                            # Analyze accuracy if validation accuracy is below 75%
                            accuracy_reasons = []
                            validation_accuracy = validation_metrics.get("avg_abs_accuracy_pct", 0.0)
                            
                            if validation_accuracy < 75.0:
                                # Get training and validation DataFrames for analysis
                                try:
                                    train_df_for_analysis = None
                                    val_df_for_analysis = None
                                    
                                    # Load training data (use the train_df already loaded above if available)
                                    if training_path.exists():
                                        train_df_for_analysis = pd.read_csv(training_path)
                                        if "actual_count" in train_df_for_analysis.columns:
                                            train_df_for_analysis["item_count"] = train_df_for_analysis["actual_count"]
                                        elif "count" in train_df_for_analysis.columns:
                                            train_df_for_analysis["item_count"] = train_df_for_analysis["count"]
                                        else:
                                            train_df_for_analysis["item_count"] = 0
                                    elif excel_path.exists():
                                        # Try Excel file
                                        train_df_for_analysis = pd.read_excel(excel_path, sheet_name="Training Data")
                                        if "actual_count" in train_df_for_analysis.columns:
                                            train_df_for_analysis["item_count"] = train_df_for_analysis["actual_count"]
                                        elif "count" in train_df_for_analysis.columns:
                                            train_df_for_analysis["item_count"] = train_df_for_analysis["count"]
                                        else:
                                            train_df_for_analysis["item_count"] = 0
                                    
                                    # Load validation data
                                    if validation_path.exists():
                                        val_df_for_analysis = pd.read_csv(validation_path)
                                        if "actual_count" in val_df_for_analysis.columns:
                                            val_df_for_analysis["item_count"] = val_df_for_analysis["actual_count"]
                                        elif "count" in val_df_for_analysis.columns:
                                            val_df_for_analysis["item_count"] = val_df_for_analysis["count"]
                                        else:
                                            val_df_for_analysis["item_count"] = 0
                                    elif excel_path.exists():
                                        # Try Excel file
                                        val_df_for_analysis = pd.read_excel(excel_path, sheet_name="Validation Data")
                                        if "actual_count" in val_df_for_analysis.columns:
                                            val_df_for_analysis["item_count"] = val_df_for_analysis["actual_count"]
                                        elif "count" in val_df_for_analysis.columns:
                                            val_df_for_analysis["item_count"] = val_df_for_analysis["count"]
                                        else:
                                            val_df_for_analysis["item_count"] = 0
                                    
                                    if train_df_for_analysis is not None and not train_df_for_analysis.empty and \
                                       val_df_for_analysis is not None and not val_df_for_analysis.empty:
                                        from src.step5_postprocessing import analyze_accuracy_reasons
                                        accuracy_reasons = analyze_accuracy_reasons(
                                            train_df_for_analysis,
                                            val_df_for_analysis,
                                            validation_accuracy,
                                            actual_col="item_count",
                                            predicted_col="predicted_count"
                                        )
                                    elif train_df_for_analysis is None or train_df_for_analysis.empty:
                                        accuracy_reasons = ["Training data not available for analysis"]
                                except Exception as analysis_exc:
                                    logging.debug(f"Could not analyze accuracy reasons: {analysis_exc}")
                            
                            # Add accuracy reasons to validation metrics
                            if accuracy_reasons:
                                validation_metrics["accuracy_reasons"] = accuracy_reasons
                        else:
                            # Model not generated
                            training_metrics = {
                                "abs_avg_deviation": 0.0, "avg_abs_accuracy_pct": 0.0,
                                "avg_abs_accuracy_pct_capped": 0.0,
                                "total_days": 0, "active_days": 0
                            }
                            validation_metrics = {
                                "abs_avg_deviation": 0.0, "avg_abs_accuracy_pct": 0.0,
                                "avg_abs_accuracy_pct_capped": 0.0,
                                "total_days": 0, "active_days": 0
                            }
                        
                        # Save to restaurant tracker
                        restaurant_tracker.add_model_results(
                            foodcourt_id, restaurant_id, item_id or item_name,
                            model_display, training_metrics, validation_metrics,
                            training_file_path=training_file_path,
                            validation_file_path=validation_file_path,
                            used=used,
                            reason=reason,
                            step_name="model_generation"
                        )
            except Exception as exc:
                logging.warning(f"Failed to save model metrics to restaurant tracker: {exc}")
        
        return restaurant_row, validation_summaries
    
    return None, []


def main(retrain_config: Optional[dict] = None, file_saver=None, restaurant_tracker=None, checkpoint_manager=None):
    """
    Main function for model generation.
    
    Args:
        retrain_config: Optional retrain configuration dict from retrain.json
        file_saver: Optional FileSaver instance for saving files
        restaurant_tracker: Optional RestaurantTracker instance for tracking item status
    
    Args:
        retrain_config: Optional retrain configuration dict from retrain.json
    """
    from src.util.pipeline_utils import get_pipeline_logger
    pipeline_logger = get_pipeline_logger()
    
    if not PREPROCESSED_DIR.exists():
        error_msg = f"Preprocessed directory not found: {PREPROCESSED_DIR}"
        pipeline_logger.log_general_error("model_generation", error_msg)
        logging.error(error_msg)
        return

    # Create output directories
    TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TRAINED_MODEL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINED_MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    XGBOOST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROPHET_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MOVING_AVG_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Count preprocessed files
    preprocessed_files = 0
    preprocessed_foodcourts = set()
    for fc_dir in os.listdir(PREPROCESSED_DIR):
        fc_path = PREPROCESSED_DIR / fc_dir
        if fc_path.is_dir():
            preprocessed_foodcourts.add(fc_dir)
            for filename in os.listdir(fc_path):
                if filename.endswith('.csv'):
                    preprocessed_files += 1
    
    # Count trained models
    trained_models = 0
    if XGBOOST_MODELS_DIR.exists():
        trained_models += len(list(XGBOOST_MODELS_DIR.glob("*.pkl")))
    if PROPHET_MODELS_DIR.exists():
        trained_models += len(list(PROPHET_MODELS_DIR.glob("*.pkl")))
    if MOVING_AVG_MODELS_DIR.exists():
        trained_models += len(list(MOVING_AVG_MODELS_DIR.glob("*.pkl")))
    
    pipeline_logger.log_info("model_generation", f"✅ Preprocessed data directory found: {PREPROCESSED_DIR}")
    logging.info("✅ Preprocessed data directory found: %s", PREPROCESSED_DIR)

    logging.info("=" * 80)
    logging.info("Starting model training run")
    logging.info("=" * 80)

    # Load items from preprocessing directory (filtered by retrain.json if provided)
    filter_df = load_items_from_preprocessing(retrain_config=retrain_config)
    if filter_df is None or len(filter_df) == 0:
        error_msg = f"No preprocessed items found in {PREPROCESSED_DIR}"
        pipeline_logger.log_general_error("model_generation", error_msg)
        logging.error("=" * 80)
        logging.error("ERROR: %s", error_msg)
        if retrain_config and retrain_config.get("model_generation"):
            logging.error("Retrain.json specifies foodcourt_ids: %s", retrain_config.get("model_generation"))
            logging.error("No preprocessed files found for these foodcourt_ids.")
        logging.error("=" * 80)
        return
    
    # Filter out completed items if checkpoint manager is provided
    if checkpoint_manager:
        original_count = len(filter_df)
        items_to_process = []
        for idx, row in filter_df.iterrows():
            item_dict = {
                "foodcourt_id": str(row["foodcourt_id"]).strip(),
                "restaurant_id": str(row["restaurant_id"]).strip(),
                "item_name": str(row["item_name"]).strip()
            }
            # Add item_id if available
            if "item_id" in row and pd.notna(row["item_id"]):
                item_dict["item_id"] = str(row["item_id"]).strip()
            
            # Skip if already completed
            if not checkpoint_manager.is_item_completed("model_generation", item_dict):
                items_to_process.append(idx)
        
        filter_df = filter_df.loc[items_to_process].reset_index(drop=True)
        skipped_count = original_count - len(filter_df)
        if skipped_count > 0:
            logging.info(f"Checkpoint: Skipping {skipped_count} already completed items. "
                        f"Remaining: {len(filter_df)} items to process.")
    
    # Log process start with summary
    summary = {
        "preprocessing/ (input)": f"{len(preprocessed_foodcourts)} foodcourts, {preprocessed_files} files",
        "trainedModel/ (existing)": f"{trained_models} models",
        "Items to process": f"{len(filter_df)} items from preprocessing directory"
    }
    pipeline_logger.log_process_start("model_generation", summary)
    
    logging.info("=" * 80)
    logging.info("Processing %d items from preprocessing directory", len(filter_df))
    logging.info("=" * 80)

    summary_rows: List[Dict[str, object]] = []
    validation_rows: List[Dict[str, object]] = []
    missing_items: List[Dict[str, str]] = []
    
    # Get global pipeline start time for progress reporting
    from src.util.pipeline_utils import get_pipeline_start_time
    pipeline_start_time = get_pipeline_start_time()
    
    # Start time tracking for this step
    step_start_time = time.time()
    total_items = len(filter_df)
    items_processed = 0
    
    # Initialize progress bar
    progress = None
    if total_items > 0:
        progress = ProgressBar(
            total=total_items,
            prefix="Model Generation",
            suffix="",
            length=40,
            show_elapsed=False
        )

    # Iterate through each item in the CSV
    for idx, row in filter_df.iterrows():
        foodcourt_id = str(row["foodcourt_id"]).strip()
        restaurant_id = str(row["restaurant_id"]).strip()
        item_name = str(row["item_name"]).strip()
        
        item_idx = idx + 1
        
        # Prepare item info string for progress bar (using global pipeline time)
        item_info = ""
        if pipeline_start_time:
            elapsed = time.time() - pipeline_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            item_info = f"Item: {item_idx}/{total_items} | Time: {time_str}"
        
        # Track item processing progress
        item_start_time = time.time()
        
        # Get item_id from the row if available
        item_id_from_row = str(row.get("item_id", "")).strip() if "item_id" in row else ""
        
        # Create item dict for checkpoint tracking
        item_dict = {
            "foodcourt_id": foodcourt_id,
            "restaurant_id": restaurant_id,
            "item_name": item_name
        }
        if item_id_from_row:
            item_dict["item_id"] = item_id_from_row
        
        # Mark item as in progress if checkpoint manager is available
        if checkpoint_manager:
            checkpoint_manager.mark_item_in_progress("model_generation", item_dict)
        
        # Check if data exists for this item
        item_df = find_item_in_preprocessed_data(foodcourt_id, restaurant_id, item_name, item_id_from_row)
        
        # Ensure date column is datetime if DataFrame exists (additional safety check)
        if item_df is not None and not item_df.empty and "date" in item_df.columns:
            if item_df["date"].dtype == 'object':
                item_df["date"] = pd.to_datetime(item_df["date"], errors="coerce")
                item_df = item_df.dropna(subset=["date"])
        
        if item_df is None or item_df.empty:
            warning_msg = f"Data NOT FOUND for {foodcourt_id} / {restaurant_id} / {item_name}"
            error_msg = "Item not found in preprocessed data"
            pipeline_logger.log_warning("model_generation", warning_msg)  # pipeline_logger is defined in main()
            missing_items.append({
                "foodcourt_id": foodcourt_id,
                "restaurant_id": restaurant_id,
                "item_name": item_name,
                "status": "Data not found",
                "reason": error_msg
            })
            # Track error in restaurant tracker (item_id not extracted yet, use item_name)
            if restaurant_tracker:
                restaurant_tracker.add_error(
                    foodcourt_id, restaurant_id, item_name,
                    error_msg, "model_generation"
                )
            items_processed += 1
            # Update progress bar
            if progress:
                progress.set_current(item_idx, item_info)
            continue

        # Extract item_id from dataframe
        item_id = ""
        for col in ["menuitemid", "item_identifier", "item_id"]:
            if col in item_df.columns and len(item_df[col]) > 0:
                item_id = str(item_df[col].iloc[0])
                break
        if not item_id:
            item_id = item_name  # Fallback to item_name if item_id not found

        # Check if we have enough features
        try:
            feature_cols = subset_features(item_df)
            if len(feature_cols) < MIN_FEATURES_REQUIRED:
                warning_msg = f"Insufficient features for {foodcourt_id} / {restaurant_id} / {item_name}"
                error_msg = f"Only {len(feature_cols)} features available, need {MIN_FEATURES_REQUIRED}"
                pipeline_logger.log_warning("model_generation", warning_msg)
                missing_items.append({
                    "foodcourt_id": foodcourt_id,
                    "restaurant_id": restaurant_id,
                    "item_name": item_name,
                    "status": "Insufficient features",
                    "reason": error_msg
                })
                # Track error in restaurant tracker
                if restaurant_tracker:
                    restaurant_tracker.add_error(
                        foodcourt_id, restaurant_id, item_id or item_name,
                        error_msg, "model_generation"
                    )
                items_processed += 1
                # Update progress bar
                if progress:
                    progress.set_current(item_idx, item_info)
                continue

            # Check retrain logic
            from src.util.pipeline_utils import should_force_retrain
            force_retrain = should_force_retrain("model_generation", foodcourt_id, restaurant_id, item_name, item_id)
            
            # Check if model already exists
            from src.util.pipeline_utils import get_model_file_name, get_result_file_name
            item_slug = sanitize_name(item_name)
            
            # Check for existing models (check all model types)
            # For now, skip if any model exists (we'll improve this later to check which models need retraining)
            has_existing_model = False
            if not force_retrain:
                for model_type in ["XGBoost", "Prophet", "MovingAverage"]:
                    model_file = get_model_file_name(foodcourt_id, restaurant_id, item_name, model_type, item_id)
                    if model_type == "XGBoost":
                        model_path = XGBOOST_MODELS_DIR / model_file
                    elif model_type == "Prophet":
                        model_path = PROPHET_MODELS_DIR / model_file
                    else:
                        model_path = MOVING_AVG_MODELS_DIR / model_file
                    
                    if model_path.exists():
                        has_existing_model = True
                        break
            
            if has_existing_model and not force_retrain:
                # Skip if models already exist (unless force retrain)
                items_processed += 1
                # Update progress bar
                if progress:
                    progress.set_current(item_idx, item_info)
                continue
            
            # Process the item (trains all models simultaneously)
            restaurant_row, validation_summaries = process_single_item(
                item_df,
                feature_cols,
                foodcourt_id,
                restaurant_id,
                item_name,
                TRAINED_MODEL_DIR,
                item_id,
                restaurant_tracker=restaurant_tracker,
            )

            if restaurant_row:
                summary_rows.append(restaurant_row)
                
                # Add all validation summaries from all trained models
                for validation_summary in validation_summaries:
                    if validation_summary:
                        # For WeeklyMovingAverageModel, we have both decay and weekday summaries
                        if "_decay_summary" in validation_summary and "_weekday_summary" in validation_summary:
                            validation_rows.append(validation_summary["_decay_summary"])
                            validation_rows.append(validation_summary["_weekday_summary"])
                        else:
                            validation_rows.append(validation_summary)
                
                # Mark item as completed in checkpoint
                if checkpoint_manager:
                    checkpoint_manager.mark_item_completed("model_generation", item_dict)
                
                items_processed += 1
                # Update progress bar
                if progress:
                    progress.set_current(item_idx, item_info)
            else:
                warning_msg = f"Training failed for {foodcourt_id} / {restaurant_id} / {item_name}"
                error_msg = "Model training returned no result"
                pipeline_logger.log_warning("model_generation", warning_msg)
                missing_items.append({
                    "foodcourt_id": foodcourt_id,
                    "restaurant_id": restaurant_id,
                    "item_name": item_name,
                    "status": "Training failed",
                    "reason": error_msg
                })
                # Mark item as failed in checkpoint
                if checkpoint_manager:
                    checkpoint_manager.mark_item_failed("model_generation", item_dict, error_msg)
                # Track error in restaurant tracker
                if restaurant_tracker:
                    restaurant_tracker.add_error(
                        foodcourt_id, restaurant_id, item_id or item_name,
                        error_msg, "model_generation"
                    )
                items_processed += 1
                # Update progress bar
                if progress:
                    progress.set_current(item_idx, item_info)
        
        except Exception as exc:
            error_msg = str(exc)
            missing_items.append({
                "foodcourt_id": foodcourt_id,
                "restaurant_id": restaurant_id,
                "item_name": item_name,
                "status": "Error",
                "reason": error_msg
            })
            # Mark item as failed in checkpoint
            if checkpoint_manager:
                checkpoint_manager.mark_item_failed("model_generation", item_dict, error_msg)
            # Track error in restaurant tracker
            if restaurant_tracker:
                restaurant_tracker.add_error(
                    foodcourt_id, restaurant_id, item_id or item_name,
                    error_msg, "model_generation"
                )
            items_processed += 1
            # Update progress bar for errors
            if progress:
                progress.set_current(item_idx, item_info)
    
    # Save report of items without data to logs directory with hyperlinks
    if missing_items:
        from src.util.pipeline_utils import get_pipeline_log_dir, create_excel_hyperlink, get_file_name
        log_dir = get_pipeline_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Add hyperlinks to input files (preprocessing files)
        missing_with_links = []
        for item in missing_items:
            item_copy = item.copy()
            # Try to create hyperlink to preprocessing file
            fc_id = item.get("foodcourt_id", "")
            rest_id = item.get("restaurant_id", "")
            item_name = item.get("item_name", "")
            
            if fc_id and rest_id and item_name:
                preprocessing_filename = get_file_name(fc_id, rest_id, item_name, "preprocessing")
                preprocessing_path = PREPROCESSED_DIR / fc_id / preprocessing_filename
                if preprocessing_path.exists():
                    item_copy["input_file_link"] = create_excel_hyperlink(
                        preprocessing_path, preprocessing_filename
                    )
                else:
                    item_copy["input_file_link"] = "File not found"
            else:
                item_copy["input_file_link"] = "N/A"
            
            missing_with_links.append(item_copy)
        
        missing_df = pd.DataFrame(missing_with_links)
        
        # Log to model_generation_logs sheet in pipeline_logs.xlsx instead of separate report
        from src.util.pipeline_utils import get_pipeline_logger, get_mongo_names
        pipeline_logger = get_pipeline_logger()
        
        for item in missing_items:
            fc_id = item.get("foodcourt_id", "")
            rest_id = item.get("restaurant_id", "")
            item_id = item.get("item_id", "")
            item_name = item.get("item_name", "")
            reason = item.get("reason", "Unknown")
            
            # Get names
            fc_name, rest_name = get_mongo_names(fc_id, rest_id)
            
            # Get input file link (preprocessing file)
            input_file_link = "N/A"
            if "input_file_link" in item:
                input_file_link = item["input_file_link"]
            else:
                # Try to find preprocessing file
                from src.util.pipeline_utils import get_file_name, get_output_base_dir, get_pipeline_type
                output_base = get_output_base_dir()
                pipeline_type = get_pipeline_type()
                preprocessing_dir = output_base / "preprocessing" / pipeline_type / fc_id
                preprocessing_filename = get_file_name(fc_id, rest_id, item_name, "preprocessing")
                preprocessing_path = preprocessing_dir / preprocessing_filename
                if preprocessing_path.exists():
                    from src.util.pipeline_utils import create_excel_hyperlink
                    input_file_link = create_excel_hyperlink(preprocessing_path, preprocessing_filename)
                else:
                    input_file_link = "Preprocessing file not found"
            
            # Log to model_generation_logs
            pipeline_logger.log_model_generation_log(
                fc_id, fc_name or fc_id,
                rest_id, rest_name or rest_id,
                item_id, item_name,
                input_file_link,
                reason
            )
        
        logging.info("=" * 80)
        logging.info("Logged %d items without data to model_generation_logs sheet in pipeline_logs.xlsx", len(missing_items))
        logging.info("=" * 80)
    else:
        logging.info("=" * 80)
        logging.info("All items from CSV were successfully processed!")
        logging.info("=" * 80)

    append_analysis(summary_rows, ANALYSIS_PATH)
    # Use TRAINED_MODEL_DIR / "temp" where validation CSV files are actually saved
    validation_base_dir = TRAINED_MODEL_DIR / "temp"
    write_validation_summary(validation_rows, VALIDATION_SUMMARY_PATH, validation_base_dir)
    logging.info("Modeling complete. Analysis stored in %s", ANALYSIS_PATH)
    logging.info("=" * 80)
    
    # Log process results
    successful_items = len(summary_rows)
    failed_items = len(missing_items)
    results = {
        "Items processed": len(filter_df),
        "Successfully trained": successful_items,
        "Failed/Missing": failed_items,
        "Validation summaries": len(validation_rows)
    }
    pipeline_logger.log_process_results("model_generation", results)


if __name__ == "__main__":
    main()


