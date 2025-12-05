"""
Postprocessing stage to redistribute 7-day predictions based on day-of-week patterns.

This script:
1. Loads validation predictions (7 days) for each item
2. Loads training data for the same item
3. Calculates day-of-week percentages from last 3 months of training data
4. Redistributes the 7-day predicted totals based on historical day-of-week patterns
5. Saves redistributed predictions
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# Filter loading is done directly in main() to get DataFrame instead of set

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input_data"
# Read from output_data/trainedModel/{pipeline_type}/results/ and output_data/preprocessing/{pipeline_type}/
# Write to output_data/postprocessing/{pipeline_type}/
from src.util.progress_bar import ProgressBar
from src.util.path_utils import get_output_base_dir, get_pipeline_type
OUTPUT_BASE = get_output_base_dir()
PIPELINE_TYPE = get_pipeline_type()
PREPROCESSED_DIR = OUTPUT_BASE / "preprocessing" / PIPELINE_TYPE
TRAINED_MODEL_RESULTS_DIR = OUTPUT_BASE / "trainedModel" / PIPELINE_TYPE / "results"
OUTPUT_BASE_DIR = OUTPUT_BASE / "postprocessing" / PIPELINE_TYPE
MODEL_DATE = pd.Timestamp("2025-11-02")
LOOKBACK_MONTHS = 3  # Use last 3 months for day-of-week pattern calculation
# No longer using CSV - scanning results directory instead


# Progress bar functionality moved to pipeline_utils.ProgressBar


def load_all_validation_predictions(foodcourt_id: str, restaurant_id: str, item_name: str, item_id: str = "") -> Dict[str, pd.DataFrame]:
    """
    Load validation predictions for ALL models (XGBoost, MovingAverage decay, MovingAverage weekday).
    Returns a dictionary mapping model_name to DataFrame.
    """
    from src.util.pipeline_utils import get_result_file_name, sanitize_name, get_output_base_dir
    item_slug = sanitize_name(item_name)
    OUTPUT_BASE = get_output_base_dir()
    TEMP_DIR = OUTPUT_BASE / "trainedModel" / PIPELINE_TYPE / "temp"
    
    results = {}
    
    # Try to load XGBoost validation CSV (try both with and without item_id)
    xgb_validation_csv = None
    # Removed exists() checks - just try to read and catch exceptions
    for item_id_to_try in [item_id, ""]:
        csv_path = TRAINED_MODEL_RESULTS_DIR / get_result_file_name(foodcourt_id, restaurant_id, item_name, "model_generation", "XGBoost", item_id_to_try, "validation")
        try:
            df = pd.read_csv(csv_path)
            if "date" in df.columns and "predicted_count" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df["model_name"] = "XGBoost"
                results["XGBoost"] = df
                xgb_validation_csv = csv_path
                break
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
        except Exception as exc:
            LOGGER.warning("Failed to load XGBoost validation CSV: %s", exc)
    
    xgb_csv = TEMP_DIR / foodcourt_id / restaurant_id / item_slug / f"{item_slug}_validation_results.csv"
    
    if "XGBoost" not in results:
        try:
            df = pd.read_csv(xgb_csv)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df["model_name"] = "XGBoost"
                results["XGBoost"] = df
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass
        except Exception as exc:
            LOGGER.warning("Failed to load XGBoost CSV: %s", exc)
    
    # Try to load MovingAverage validation CSV (try both with and without item_id)
    ma_validation_csv = None
    for item_id_to_try in [item_id, ""]:
        csv_path = TRAINED_MODEL_RESULTS_DIR / get_result_file_name(foodcourt_id, restaurant_id, item_name, "model_generation", "MovingAverage", item_id_to_try, "validation")
        try:
            df = pd.read_csv(csv_path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df["model_name"] = "MovingAverage"
                results["MovingAverage"] = df
                ma_validation_csv = csv_path
                break
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
        except Exception as exc:
            LOGGER.warning("Failed to load MovingAverage validation CSV: %s", exc)
    
    ma_decay_csv = TEMP_DIR / foodcourt_id / restaurant_id / item_slug / f"{item_slug}_validation_decay_results.csv"
    
    try:
        df = pd.read_csv(ma_decay_csv)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["model_name"] = "MovingAverage_decay"
            results["MovingAverage_decay"] = df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pass
    except Exception as exc:
        LOGGER.warning("Failed to load MovingAverage decay CSV: %s", exc)
    
    # Try to load MovingAverage weekday
    ma_weekday_csv = TEMP_DIR / foodcourt_id / restaurant_id / item_slug / f"{item_slug}_validation_weekday_results.csv"
    try:
        df = pd.read_csv(ma_weekday_csv)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["model_name"] = "MovingAverage_weekday"
            results["MovingAverage_weekday"] = df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pass
    except Exception as exc:
        LOGGER.warning("Failed to load MovingAverage weekday CSV: %s", exc)
    
    return results


def load_validation_predictions(foodcourt_id: str, restaurant_id: str, item_name: str) -> Optional[pd.DataFrame]:
    """
    Load validation predictions from trainedModel/results/ CSV files.
    Falls back to CSV files in temp/ directory if results CSV files don't exist.
    Reads from validation CSV files (format: {base}_validation.csv).
    """
    from src.util.pipeline_utils import get_result_file_name, sanitize_name
    item_slug = sanitize_name(item_name)
    
    # First, try validation CSV files in results/ directory
    # Try both with and without item_id (models may save with item_id)
    possible_files = []
    for item_id_to_try in [item_id, ""]:
        possible_files.extend([
            get_result_file_name(foodcourt_id, restaurant_id, item_name, "model_generation", "XGBoost", item_id_to_try, "validation"),
            get_result_file_name(foodcourt_id, restaurant_id, item_name, "model_generation", "MovingAverage", item_id_to_try, "validation"),
        ])
    
    for filename in possible_files:
        file_path = TRAINED_MODEL_RESULTS_DIR / filename
        # Removed exists() check - just try to read and catch exception
        try:
            # Read from validation CSV file
            df = pd.read_csv(file_path)
            if "date" in df.columns and "predicted_count" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                LOGGER.debug("Loaded validation predictions from %s", file_path.name)
                return df
        except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
            # File doesn't exist or is empty - continue to next option
            pass
        except Exception as exc:
            LOGGER.warning("Failed to load %s: %s", file_path, exc)
    
    # Fallback: Try CSV files in temp/ directory
    # Structure: temp/{foodcourt_id}/{restaurant_id}/{item_slug}/{item_slug}_validation_results.csv
    from src.util.pipeline_utils import get_output_base_dir
    OUTPUT_BASE = get_output_base_dir()
    TEMP_DIR = OUTPUT_BASE / "trainedModel" / PIPELINE_TYPE / "temp"
    
    csv_paths = [
        TEMP_DIR / foodcourt_id / restaurant_id / item_slug / f"{item_slug}_validation_results.csv",
        TEMP_DIR / foodcourt_id / restaurant_id / item_slug / f"{item_slug}_validation_decay_results.csv",
        TEMP_DIR / foodcourt_id / restaurant_id / item_slug / f"{item_slug}_validation_weekday_results.csv",
    ]
    
    for csv_path in csv_paths:
        # Removed exists() check - just try to read and catch exception
        try:
            df = pd.read_csv(csv_path)
            # Ensure date column exists and is datetime
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            elif "Date" in df.columns:
                df["date"] = pd.to_datetime(df["Date"])
                df = df.rename(columns={"Date": "date"})
            
            # Add metadata columns from parameters (CSV files don't include these)
            df["foodcourt_id"] = foodcourt_id
            df["restaurant_id"] = restaurant_id
            
            # Check for predicted_count column (may be named differently)
            if "predicted_count" in df.columns:
                LOGGER.debug("Loaded validation predictions from CSV: %s", csv_path.name)
                return df
            elif "Predicted_Count" in df.columns:
                df = df.rename(columns={"Predicted_Count": "predicted_count"})
                LOGGER.debug("Loaded validation predictions from CSV: %s", csv_path.name)
                return df
            # If neither predicted_count nor Predicted_Count exists, try to find any prediction column
            elif len(df.columns) > 0 and "count" in " ".join(df.columns).lower():
                # Try to infer which column is predicted_count
                for col in df.columns:
                    if "predicted" in col.lower() and "count" in col.lower():
                        df = df.rename(columns={col: "predicted_count"})
                        LOGGER.debug("Loaded validation predictions from CSV: %s (renamed %s to predicted_count)", csv_path.name, col)
                        return df
        except Exception as exc:
            LOGGER.warning("Failed to load CSV %s: %s", csv_path, exc)
    
    warning_msg = f"No validation predictions found for {foodcourt_id}/{restaurant_id}/{item_name}"
    from src.util.pipeline_utils import get_pipeline_logger
    get_pipeline_logger().log_warning("postprocessing", warning_msg)
    LOGGER.warning(warning_msg)
    return None




def load_training_data(foodcourt_id: str, restaurant_id: str, item_name: str, item_id: str = "") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load preprocessed training data for an item from Excel file.
    
    Returns:
        Tuple of (DataFrame or None, error_reason or None)
        error_reason can be: "file_not_found", "item_not_found", "no_data", "load_error"
    """
    from src.util.pipeline_utils import get_file_name, get_pipeline_logger, sanitize_name
    # Try with item_id first if provided, then fallback to item_name only
    csv_path = None
    # Removed exists() checks - just try to read and catch exceptions
    if item_id:
        filename_with_id = get_file_name(foodcourt_id, restaurant_id, item_name, "preprocessing", item_id)
        csv_path = PREPROCESSED_DIR / foodcourt_id / filename_with_id
        try:
            df = pd.read_csv(csv_path)
            return df, None
        except (FileNotFoundError, pd.errors.EmptyDataError):
            csv_path = None  # Try without item_id next
        except Exception as e:
            return None, f"load_error: {e}"
    
    if not csv_path:
        filename = get_file_name(foodcourt_id, restaurant_id, item_name, "preprocessing")
        csv_path = PREPROCESSED_DIR / foodcourt_id / filename
    
    # Debug: Log the expected path
    LOGGER.debug("Looking for preprocessing file at: %s", csv_path)
    LOGGER.debug("Item name used: '%s'", item_name)
    
    # Try to read primary path
    try:
        df = pd.read_csv(csv_path)
        return df, None
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # Try alternative filename variations (spaces vs underscores)
        fc_dir = PREPROCESSED_DIR / foodcourt_id
        # Removed fc_dir.exists() check - just try to access
        try:
            # Try with spaces replaced by underscores
            item_name_alt1 = item_name.replace(" ", "_")
            filename_alt1 = get_file_name(foodcourt_id, restaurant_id, item_name_alt1, "preprocessing")
            csv_path_alt1 = fc_dir / filename_alt1
            
            # Try with underscores replaced by spaces
            item_name_alt2 = item_name.replace("_", " ")
            filename_alt2 = get_file_name(foodcourt_id, restaurant_id, item_name_alt2, "preprocessing")
            csv_path_alt2 = fc_dir / filename_alt2
            
            # Try with sanitized name (in case original had special chars)
            item_name_alt3 = sanitize_name(item_name)
            filename_alt3 = get_file_name(foodcourt_id, restaurant_id, item_name_alt3, "preprocessing")
            csv_path_alt3 = fc_dir / filename_alt3
            
            # Try each alternative - removed exists() checks
            for alt_path in [csv_path_alt1, csv_path_alt2, csv_path_alt3]:
                try:
                    df = pd.read_csv(alt_path)
                    LOGGER.debug("Found preprocessing file with alternative name: %s", alt_path)
                    return df, None
                except (FileNotFoundError, pd.errors.EmptyDataError):
                    continue
                except Exception as e:
                    return None, f"load_error: {e}"
        except Exception:
            pass
        
        # If we get here, file not found
        return None, "file_not_found"
    except Exception as e:
        return None, f"load_error: {e}"


def calculate_day_of_week_percentages(df: pd.DataFrame, model_date: pd.Timestamp) -> Dict[int, float]:
    """
    Calculate percentage of weekly count contributed by each day of week
    from the last 3 months of training data.
    
    Returns a dictionary: {weekday: percentage} where weekday is 0=Monday, 6=Sunday
    """
    if df.empty or "date" not in df.columns or "count" not in df.columns:
        return {}
    
    # Ensure date column is datetime
    df = df.copy()
    if df["date"].dtype == 'object' or not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        if df.empty:
            return {}
    
    # Get last 3 months of data before model_date
    lookback_start = model_date - pd.DateOffset(months=LOOKBACK_MONTHS)
    recent_df = df[(df["date"] >= lookback_start) & (df["date"] < model_date)].copy()
    
    if recent_df.empty:
        warning_msg = f"No training data in last {LOOKBACK_MONTHS} months before model_date"
        from src.util.pipeline_utils import get_pipeline_logger
        get_pipeline_logger().log_warning("postprocessing", warning_msg)
        LOGGER.warning(warning_msg)
        return {}
    
    # Group by week and day of week
    recent_df["week"] = recent_df["date"].dt.to_period("W")
    recent_df["weekday"] = recent_df["date"].dt.weekday  # 0=Monday, 6=Sunday
    
    # Calculate total count per day of week
    weekday_totals = recent_df.groupby("weekday")["count"].sum()
    
    # Calculate percentage of weekly count for each day
    # For each week, calculate what % each day contributes
    weekly_patterns = []
    
    for week_period in recent_df["week"].unique():
        week_df = recent_df[recent_df["week"] == week_period]
        week_total = week_df["count"].sum()
        
        if week_total > 0:
            week_pattern = week_df.groupby("weekday")["count"].sum() / week_total
            weekly_patterns.append(week_pattern.to_dict())
    
    if not weekly_patterns:
        # Fallback: use overall weekday totals
        total_count = weekday_totals.sum()
        if total_count > 0:
            return {int(day): float(count / total_count) for day, count in weekday_totals.items()}
        return {}
    
    # Average the percentages across all weeks
    weekday_percentages = {}
    for day in range(7):  # Monday to Sunday
        percentages = [pattern.get(day, 0.0) for pattern in weekly_patterns if day in pattern]
        if percentages:
            weekday_percentages[day] = float(np.mean(percentages))
        else:
            weekday_percentages[day] = 0.0
    
    # Normalize to ensure they sum to 1.0
    total_pct = sum(weekday_percentages.values())
    if total_pct > 0:
        weekday_percentages = {day: pct / total_pct for day, pct in weekday_percentages.items()}
    
    return weekday_percentages


def redistribute_predictions(
    predictions_df: pd.DataFrame, 
    weekday_percentages: Dict[int, float]
) -> pd.DataFrame:
    """
    Redistribute 7-day predictions based on day-of-week percentages.
    
    Args:
        predictions_df: DataFrame with 'date' and 'predicted_count' columns
        weekday_percentages: Dictionary mapping weekday (0-6) to percentage
        
    Returns:
        DataFrame with redistributed predictions (simplified columns only)
    """
    if predictions_df.empty:
        return predictions_df
    
    result_df = pd.DataFrame()
    
    # Add foodcourt and restaurant IDs and names
    for col in ["foodcourt_id", "foodcourtid"]:
        if col in predictions_df.columns:
            result_df["foodcourt_id"] = predictions_df[col].values
            break
    for col in ["foodcourt_name", "foodcourtname"]:
        if col in predictions_df.columns:
            result_df["foodcourt_name"] = predictions_df[col].values
            break
    for col in ["restaurant_id", "restaurant"]:
        if col in predictions_df.columns:
            result_df["restaurant_id"] = predictions_df[col].values
            break
    for col in ["restaurant_name", "restaurantname"]:
        if col in predictions_df.columns:
            result_df["restaurant_name"] = predictions_df[col].values
            break
    
    # Add date and actual count
    if "date" in predictions_df.columns:
        result_df["date"] = pd.to_datetime(predictions_df["date"]).dt.strftime("%Y-%m-%d")
    
    # Get actual_count - handle NaN/inf
    if "actual_count" in predictions_df.columns:
        actual_count_vals = predictions_df["actual_count"].values
    elif "count" in predictions_df.columns:
        actual_count_vals = predictions_df["count"].values
    else:
        actual_count_vals = np.zeros(len(predictions_df))
    actual_count_vals = np.nan_to_num(actual_count_vals, nan=0.0, posinf=0.0, neginf=0.0)
    result_df["actual_count"] = actual_count_vals
    
    # Get original predictions - handle NaN/inf
    if "predicted_count" in predictions_df.columns:
        original_pred = predictions_df["predicted_count"].values
    else:
        original_pred = np.zeros(len(predictions_df))
    original_pred = np.nan_to_num(original_pred, nan=0.0, posinf=0.0, neginf=0.0)
    result_df["predicted_count_original"] = original_pred
    
    # ROUND predicted counts BEFORE redistribution (as per requirements)
    original_pred_rounded = np.round(original_pred).astype(int)
    
    # Calculate redistributed predictions
    result_df["weekday"] = pd.to_datetime(result_df["date"]).dt.weekday
    # Use ROUNDED predicted counts for total (as per requirements)
    total_predicted = original_pred_rounded.sum()
    
    redistributed = []
    for weekday in result_df["weekday"]:
        percentage = weekday_percentages.get(int(weekday), 1.0 / 7.0)
        redistributed_count = total_predicted * percentage
        redistributed.append(redistributed_count)
    
    # Round postprocessing predictions before calculating errors
    redistributed_rounded = np.round(redistributed).astype(int)
    result_df["predicted_count_postprocessing"] = redistributed_rounded
    result_df["error_postprocessing"] = redistributed_rounded - result_df["actual_count"]
    
    # Calculate error percentage using rounded predicted values
    error_pct = np.where(
        result_df["actual_count"] != 0,
        (result_df["error_postprocessing"] / result_df["actual_count"]) * 100.0,
        0.0
    )
    result_df["error_pct_postprocessing"] = np.nan_to_num(error_pct, nan=0.0)
    
    # Drop weekday column (not needed in output)
    result_df = result_df.drop(columns=["weekday"], errors="ignore")
    
    return result_df


def analyze_accuracy_reasons(train_df: pd.DataFrame, validation_df: pd.DataFrame, 
                             accuracy_pct: float,
                             actual_col: str = "item_count", 
                             predicted_col: str = "predicted_count") -> List[str]:
    """
    Analyze reasons for low accuracy (<75%) using training data as basis.
    
    Args:
        train_df: Training DataFrame (used as basis for all analysis)
        validation_df: Validation DataFrame (only used for validation day count)
        actual_col: Column name for actual counts
        predicted_col: Column name for predicted counts (not used, kept for compatibility)
        accuracy_pct: Validation accuracy percentage
    
    Returns list of reason strings like:
    - "Sudden spike of item sold" (based on training data)
    - "Fresh start / too less data" (based on validation days)
    - "Active days are very small" (based on training data)
    - "No particular pattern can be sold any day" (based on training data)
    - "Does not depend on anything random selling counts" (based on training data)
    """
    reasons = []
    
    if train_df.empty:
        reasons.append("No training data available for analysis")
        return reasons
    
    # Filter training data to active days only
    train_active_df = train_df[train_df[actual_col] > 0].copy()
    
    if train_active_df.empty:
        reasons.append("No active days in training data")
        return reasons
    
    # Check: Too less validation data
    if validation_df.empty:
        reasons.append("No validation data available")
    else:
        validation_total_days = len(validation_df)
        if validation_total_days < 5:
            reasons.append(f"Fresh start / too less data (only {validation_total_days} validation days)")
    
    # Calculate training statistics
    train_total_days = len(train_df)
    train_active_days = len(train_active_df)
    train_active_day_pct = (train_active_days / train_total_days * 100) if train_total_days > 0 else 0
    
    # Check: Active days are very small (based on training data)
    if train_active_day_pct < 30:
        reasons.append(f"Active days are very small ({train_active_days}/{train_total_days} days, {train_active_day_pct:.1f}%)")
    
    # Check for sudden spikes in training data
    if "date" in train_active_df.columns and len(train_active_df) > 1:
        train_active_df = train_active_df.sort_values("date")
        train_actual_vals = train_active_df[actual_col].values
        
        # Calculate coefficient of variation (std/mean) to detect spikes
        if len(train_actual_vals) > 1 and np.mean(train_actual_vals) > 0:
            cv = np.std(train_actual_vals) / np.mean(train_actual_vals)
            if cv > 1.5:  # High variation indicates spikes
                # Check for individual spikes (>3x mean)
                mean_val = np.mean(train_actual_vals)
                spikes = np.sum(train_actual_vals > mean_val * 3)
                if spikes > 0:
                    reasons.append(f"Sudden spike of item sold ({spikes} days with >3x average in training data)")
    
    # Removed: "No particular pattern can be sold any day" reason check
    
    # Check: Random selling counts (high variance relative to mean in training data)
    if len(train_active_df) > 1:
        train_actual_vals = train_active_df[actual_col].values
        if np.mean(train_actual_vals) > 0:
            cv = np.std(train_actual_vals) / np.mean(train_actual_vals)
            if cv > 1.0 and cv < 1.5:  # High but not extreme variation
                reasons.append("Does not depend on anything random selling counts (high variance in training data)")
    
    # If no specific reasons found, add generic reason
    if not reasons:
        reasons.append("Low accuracy due to model limitations")
    
    return reasons


def redistribute_moving_average_predictions(
    predictions_df: pd.DataFrame,
    train_df: pd.DataFrame,
    foodcourt_id: str,
    foodcourt_name: str,
    restaurant_id: str,
    restaurant_name: str,
    item_id: str,
    item_name: str
) -> Tuple[pd.DataFrame, str]:
    """
    Special postprocessing for MovingAverage models with different scenarios.
    
    Scenarios:
    1. Data exists and continuous for last 2-3 weeks: total over non-zero days and divide according to previous 2 weeks per day pattern
    2. Not sold for last 2-3 weeks: check for a week where previous to it value was 0 and then it was selling
    3. Sudden spikes: can't do anything (return original predictions)
    
    Returns:
        Tuple of (redistributed DataFrame, scenario_description)
    """
    if predictions_df.empty:
        return predictions_df, "No predictions"
    
    result_df = pd.DataFrame()
    
    # Copy base columns
    for col in ["foodcourt_id", "foodcourtid", "foodcourt_name", "foodcourtname",
                "restaurant_id", "restaurant", "restaurant_name", "restaurantname"]:
        if col in predictions_df.columns:
            result_df[col.replace("foodcourtid", "foodcourt_id").replace("foodcourtname", "foodcourt_name")
                     .replace("restaurant", "restaurant_id").replace("restaurantname", "restaurant_name")] = predictions_df[col].values
    
    if "date" in predictions_df.columns:
        result_df["date"] = pd.to_datetime(predictions_df["date"]).dt.strftime("%Y-%m-%d")
    
    # Get actual and predicted counts
    if "actual_count" in predictions_df.columns:
        actual_count_vals = predictions_df["actual_count"].values
    elif "count" in predictions_df.columns:
        actual_count_vals = predictions_df["count"].values
    else:
        actual_count_vals = np.zeros(len(predictions_df))
    
    if "predicted_count" in predictions_df.columns:
        predicted_vals = predictions_df["predicted_count"].values
    else:
        predicted_vals = np.zeros(len(predictions_df))
    
    # Round predicted values
    predicted_rounded = np.round(predicted_vals).astype(int)
    total_predicted = predicted_rounded.sum()
    
    result_df["actual_count"] = actual_count_vals
    result_df["predicted_count_original"] = predicted_rounded
    
    # Determine scenario based on training data
    scenario = "default"
    
    if train_df.empty or "date" not in train_df.columns or "count" not in train_df.columns:
        # No training data - use uniform distribution
        result_df["weekday"] = pd.to_datetime(result_df["date"]).dt.weekday
        redistributed = [total_predicted / 7.0] * len(result_df)
        scenario = "No training data - uniform distribution"
    else:
        train_df["date"] = pd.to_datetime(train_df["date"])
        train_df = train_df.sort_values("date")
        train_df["weekday"] = train_df["date"].dt.weekday
        
        # Get last 2-3 weeks of training data
        last_date = train_df["date"].max()
        three_weeks_ago = last_date - pd.Timedelta(days=21)
        recent_train = train_df[train_df["date"] >= three_weeks_ago].copy()
        
        # Scenario 1: Data exists and continuous for last 2-3 weeks
        if len(recent_train) >= 14:  # At least 2 weeks
            active_recent = recent_train[recent_train["count"] > 0]
            if len(active_recent) >= 10:  # At least 10 active days in 3 weeks
                # Calculate weekday percentages from recent 2-3 weeks
                weekday_totals = active_recent.groupby("weekday")["count"].sum()
                total_recent = weekday_totals.sum()
                
                if total_recent > 0:
                    weekday_percentages = {int(day): float(count / total_recent) 
                                         for day, count in weekday_totals.items()}
                    # Normalize
                    total_pct = sum(weekday_percentages.values())
                    if total_pct > 0:
                        weekday_percentages = {day: pct / total_pct for day, pct in weekday_percentages.items()}
                    
                    result_df["weekday"] = pd.to_datetime(result_df["date"]).dt.weekday
                    redistributed = [total_predicted * weekday_percentages.get(int(day), 1.0/7.0) 
                                   for day in result_df["weekday"]]
                    scenario = "Continuous data - redistributed based on last 2-3 weeks pattern"
                else:
                    # Uniform distribution
                    result_df["weekday"] = pd.to_datetime(result_df["date"]).dt.weekday
                    redistributed = [total_predicted / 7.0] * len(result_df)
                    scenario = "Recent data has no sales - uniform distribution"
            else:
                # Scenario 2: Not sold for last 2-3 weeks - look for similar pattern
                # Find weeks where previous week had 0 sales then started selling
                scenario = "Not sold recently - checking historical patterns"
                
                # Group by week
                train_df["year_week"] = train_df["date"].dt.to_period("W")
                weekly_totals = train_df.groupby("year_week")["count"].sum()
                
                # Find pattern: week with sales after zero weeks
                found_pattern = False
                for i in range(1, len(weekly_totals)):
                    if weekly_totals.iloc[i-1] == 0 and weekly_totals.iloc[i] > 0:
                        # Found pattern - use that week's distribution
                        pattern_week = weekly_totals.index[i]
                        pattern_week_df = train_df[train_df["year_week"] == pattern_week]
                        pattern_week_df = pattern_week_df[pattern_week_df["count"] > 0]
                        
                        if len(pattern_week_df) > 0:
                            weekday_totals = pattern_week_df.groupby("weekday")["count"].sum()
                            total_pattern = weekday_totals.sum()
                            
                            if total_pattern > 0:
                                weekday_percentages = {int(day): float(count / total_pattern) 
                                                     for day, count in weekday_totals.items()}
                                total_pct = sum(weekday_percentages.values())
                                if total_pct > 0:
                                    weekday_percentages = {day: pct / total_pct for day, pct in weekday_percentages.items()}
                                
                                result_df["weekday"] = pd.to_datetime(result_df["date"]).dt.weekday
                                redistributed = [total_predicted * weekday_percentages.get(int(day), 1.0/7.0) 
                                               for day in result_df["weekday"]]
                                scenario = f"Using historical pattern from week {pattern_week} (after zero sales)"
                                found_pattern = True
                                break
                
                if not found_pattern:
                    # Uniform distribution as fallback
                    result_df["weekday"] = pd.to_datetime(result_df["date"]).dt.weekday
                    redistributed = [total_predicted / 7.0] * len(result_df)
                    scenario = "No historical pattern found - uniform distribution"
        else:
            # Not enough recent data - use overall pattern
            active_train = train_df[train_df["count"] > 0]
            if len(active_train) > 0:
                weekday_totals = active_train.groupby("weekday")["count"].sum()
                total_train = weekday_totals.sum()
                
                if total_train > 0:
                    weekday_percentages = {int(day): float(count / total_train) 
                                         for day, count in weekday_totals.items()}
                    total_pct = sum(weekday_percentages.values())
                    if total_pct > 0:
                        weekday_percentages = {day: pct / total_pct for day, pct in weekday_percentages.items()}
                    
                    result_df["weekday"] = pd.to_datetime(result_df["date"]).dt.weekday
                    redistributed = [total_predicted * weekday_percentages.get(int(day), 1.0/7.0) 
                                   for day in result_df["weekday"]]
                    scenario = "Using overall training pattern (insufficient recent data)"
                else:
                    result_df["weekday"] = pd.to_datetime(result_df["date"]).dt.weekday
                    redistributed = [total_predicted / 7.0] * len(result_df)
                    scenario = "No sales in training data - uniform distribution"
            else:
                result_df["weekday"] = pd.to_datetime(result_df["date"]).dt.weekday
                redistributed = [total_predicted / 7.0] * len(result_df)
                scenario = "No active days in training - uniform distribution"
    
    # Check for sudden spikes in validation - if detected, don't redistribute
    if len(result_df) > 1:
        actual_vals = result_df["actual_count"].values
        if np.mean(actual_vals) > 0:
            cv = np.std(actual_vals) / np.mean(actual_vals)
            if cv > 2.0:  # Very high variation indicates sudden spikes
                # Don't redistribute - use original predictions
                redistributed = predicted_rounded.tolist()
                scenario = "Sudden spikes detected - using original predictions"
    
    # Round redistributed values
    redistributed_rounded = np.round(redistributed).astype(int)
    result_df["predicted_count_postprocessing"] = redistributed_rounded
    
    # Calculate metrics
    result_df["abs_deviation"] = np.abs(redistributed_rounded - result_df["actual_count"])
    with np.errstate(divide='ignore', invalid='ignore'):
        error_pct = np.where(
            result_df["actual_count"] != 0,
            np.abs((redistributed_rounded - result_df["actual_count"]) / result_df["actual_count"]) * 100.0,
            0.0
        )
    result_df["abs_error_pct"] = np.nan_to_num(error_pct, nan=0.0)
    
    # Drop weekday column
    result_df = result_df.drop(columns=["weekday"], errors="ignore")
    
    return result_df, scenario


def process_model_predictions(
    pred_df: pd.DataFrame,
    model_name: str,
    weekday_percentages: Dict[int, float],
    foodcourt_id: str,
    foodcourt_name: str,
    restaurant_id: str,
    restaurant_name: str,
    item_id: str,
    item_name: str
) -> pd.DataFrame:
    """
    Process predictions for a single model and return DataFrame with model-specific columns.
    
    Returns DataFrame with columns:
    - Base: foodcourt_id, foodcourt_name, restaurant_id, restaurant_name, item_id, item_name, date, item_count
    - Model-specific: {model}_predicted_count, {model}_abs_deviation, {model}_abs_error_pct,
                      {model}_predicted_count_postprocessing, {model}_postProcess_abs_dev, {model}_postProcess_abs_pct
    """
    if pred_df.empty or "predicted_count" not in pred_df.columns:
        return pd.DataFrame()
    
    # Ensure we have date and actual_count
    if "date" not in pred_df.columns:
        return pd.DataFrame()
    
    # Get actual_count - handle NaN/inf from the start
    if "actual_count" in pred_df.columns:
        actual_count = pred_df["actual_count"].values
    elif "count" in pred_df.columns:
        actual_count = pred_df["count"].values
    else:
        actual_count = np.zeros(len(pred_df))
    
    # Clean actual_count - handle NaN and inf values
    actual_count = np.nan_to_num(actual_count, nan=0.0, posinf=0.0, neginf=0.0)
    actual_count = np.where(np.isfinite(actual_count), actual_count, 0.0)
    
    # Get original predictions - handle NaN and inf values
    original_pred = pred_df["predicted_count"].values
    original_pred = np.nan_to_num(original_pred, nan=0.0, posinf=0.0, neginf=0.0)
    # Ensure all values are finite before rounding
    original_pred = np.where(np.isfinite(original_pred), original_pred, 0.0)
    
    # Round predicted values first, then calculate errors
    original_pred_rounded = np.round(original_pred).astype(int)
    
    # Calculate absolute deviation and error percentage for original (using rounded predicted values)
    abs_deviation = np.abs(original_pred_rounded - actual_count)
    # Suppress divide by zero warning - we handle it with np.where
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_error_pct = np.abs(np.where(
            actual_count != 0,
            ((original_pred_rounded - actual_count) / actual_count) * 100.0,
            0.0
        ))
    # Ensure error_pct is finite
    abs_error_pct = np.nan_to_num(abs_error_pct, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Redistribute predictions
    redistributed_df = redistribute_predictions(pred_df, weekday_percentages)
    
    if redistributed_df.empty:
        return pd.DataFrame()
    
    # Get postprocessing predictions - handle NaN and inf values
    post_pred = redistributed_df["predicted_count_postprocessing"].values
    post_pred = np.nan_to_num(post_pred, nan=0.0, posinf=0.0, neginf=0.0)
    post_pred = np.where(np.isfinite(post_pred), post_pred, 0.0)
    # Round postprocessing predictions before calculating errors
    post_pred_rounded = np.round(post_pred).astype(int)
    post_abs_dev = np.abs(post_pred_rounded - actual_count)
    # Recalculate error percentage using rounded postprocessing predictions
    # Suppress divide by zero warning - we handle it with np.where
    with np.errstate(divide='ignore', invalid='ignore'):
        post_abs_error_pct = np.abs(np.where(
            actual_count != 0,
            ((post_pred_rounded - actual_count) / actual_count) * 100.0,
            0.0
        ))
    post_abs_error_pct = np.nan_to_num(post_abs_error_pct, nan=0.0, posinf=0.0, neginf=0.0)
    post_abs_error_pct = np.where(np.isfinite(post_abs_error_pct), post_abs_error_pct, 0.0)
    
    # Create result DataFrame - ensure safe integer conversion
    try:
        predicted_count_int = original_pred_rounded  # Already rounded above
        predicted_count_post_int = post_pred_rounded  # Already rounded above
    except (ValueError, OverflowError):
        # Fallback: convert NaN/inf to 0 first, then round
        predicted_count_int = np.round(np.nan_to_num(original_pred, nan=0.0, posinf=0.0, neginf=0.0)).astype(int)
        predicted_count_post_int = np.round(np.nan_to_num(post_pred, nan=0.0, posinf=0.0, neginf=0.0)).astype(int)
    
    result_df = pd.DataFrame({
        "foodcourt_id": foodcourt_id,
        "foodcourt_name": foodcourt_name,
        "restaurant_id": restaurant_id,
        "restaurant_name": restaurant_name,
        "item_id": item_id,
        "item_name": item_name,
        "date": pd.to_datetime(pred_df["date"]).dt.strftime("%Y-%m-%d"),
        "item_count": actual_count,
        f"{model_name}_predicted_count": predicted_count_int,
        f"{model_name}_abs_deviation": abs_deviation,
        f"{model_name}_abs_error_pct": abs_error_pct,
        f"{model_name}_predicted_count_postprocessing": predicted_count_post_int,
        f"{model_name}_postProcess_abs_dev": post_abs_dev,
        f"{model_name}_postProcess_abs_pct": post_abs_error_pct,
    })
    
    return result_df


def postprocess_item(
    foodcourt_id: str,
    restaurant_id: str,
    item_name: str,
    restaurant_tracker=None,
    item_id: str = "",
) -> bool:
    """Postprocess predictions for a single item - processes all models and saves to master CSV."""
    try:
        from src.util.pipeline_utils import sanitize_name, get_mongo_names, get_pipeline_logger
        item_slug = sanitize_name(item_name)
        pipeline_logger = get_pipeline_logger()
        
        # Load all models' validation predictions
        all_predictions = load_all_validation_predictions(foodcourt_id, restaurant_id, item_name, item_id)
        if not all_predictions:
            # Log to postprocessing_logs when model results exist but postprocessing fails
            from src.util.pipeline_utils import get_result_file_name, get_output_base_dir, get_pipeline_type, create_excel_hyperlink
            output_base = get_output_base_dir()
            pipeline_type = get_pipeline_type()
            results_dir = output_base / "trainedModel" / pipeline_type / "results"
            
            # Check if model results exist
            model_result_exists = False
            input_file_link = "N/A"
            result_path = None
            result_filename = ""
            item_id_val = ""
            
            for model_type in ["XGBoost", "MovingAverage"]:
                # Try validation CSV file
                result_filename = get_result_file_name(foodcourt_id, restaurant_id, item_name, "model_generation", model_type, "", "validation")
                result_path = results_dir / result_filename
                if result_path.exists():
                    model_result_exists = True
                    input_file_link = create_excel_hyperlink(result_path, result_filename)
                    # Try to get item_id from model results
                    try:
                        df_check = pd.read_csv(result_path, nrows=1)
                        for col in ["item_id", "menuitemid", "item_identifier"]:
                            if col in df_check.columns:
                                item_id_val = str(df_check[col].iloc[0])
                                break
                    except:
                        pass
                    break
            
            if model_result_exists:
                # Log to postprocessing_logs
                fc_name, rest_name = get_mongo_names(foodcourt_id, restaurant_id)
                if not item_id_val:
                    item_id_val = item_name
                
                # Log to postprocessing_logs with input file link
                pipeline_logger.log_postprocessing_log(
                    foodcourt_id, fc_name or foodcourt_id,
                    restaurant_id, rest_name or restaurant_id,
                    item_id_val, item_name,
                    input_file_link,
                    f"No predictions found for any model (model results exist but validation data not found)"
                )
                
                # Add discard reason to file_locator
                from src.util.pipeline_utils import get_file_locator
                get_file_locator().add_discard_reason(
                    foodcourt_id, fc_name or foodcourt_id,
                    restaurant_id, rest_name or restaurant_id,
                    item_id_val, item_name,
                    "postprocessing",
                    "No predictions found for any model"
                )
            
            warning_msg = f"No predictions found for any model for {foodcourt_id}/{restaurant_id}/{item_name}"
            error_msg = "No predictions found for any model"
            pipeline_logger.log_warning("postprocessing", warning_msg)
            # Track error in restaurant tracker
            if restaurant_tracker:
                restaurant_tracker.add_error(
                    foodcourt_id, restaurant_id, item_id_val or item_name,
                    error_msg, "postprocessing"
                )
            # Don't print to console - only log to pipeline logs
            return False
        
        # Check if all validation dates have zero actual counts BEFORE loading training data
        all_zero_count = True
        for model_name, pred_df in all_predictions.items():
            if "actual_count" in pred_df.columns:
                actual_counts = pred_df["actual_count"].values
            elif "count" in pred_df.columns:
                actual_counts = pred_df["count"].values
            else:
                continue
            
            actual_counts = np.nan_to_num(actual_counts, nan=0.0, posinf=0.0, neginf=0.0)
            if np.any(actual_counts > 0):
                all_zero_count = False
                break
        
        if all_zero_count:
            # Log to postprocessing_logs
            from src.util.pipeline_utils import get_result_file_name, get_output_base_dir, get_pipeline_type, create_excel_hyperlink
            output_base = get_output_base_dir()
            pipeline_type = get_pipeline_type()
            results_dir = output_base / "trainedModel" / pipeline_type / "results"
            
            input_file_link = "N/A"
            item_id_val = ""
            for model_type in ["XGBoost", "MovingAverage"]:
                # Try validation CSV file
                result_filename = get_result_file_name(foodcourt_id, restaurant_id, item_name, "model_generation", model_type, "", "validation")
                result_path = results_dir / result_filename
                if result_path.exists():
                    input_file_link = create_excel_hyperlink(result_path, result_filename)
                    # Try to get item_id from model results
                    try:
                        df_check = pd.read_csv(result_path, nrows=1)
                        for col in ["item_id", "menuitemid", "item_identifier"]:
                            if col in df_check.columns:
                                item_id_val = str(df_check[col].iloc[0])
                                break
                    except:
                        pass
                    break
            
            fc_name, rest_name = get_mongo_names(foodcourt_id, restaurant_id)
            if not item_id_val:
                item_id_val = item_name
            error_msg = "Not sold in val dates"
            pipeline_logger.log_postprocessing_log(
                foodcourt_id, fc_name or foodcourt_id,
                restaurant_id, rest_name or restaurant_id,
                item_id_val, item_name,
                input_file_link,
                error_msg
            )
            
            # Add discard reason to file_locator
            from src.util.pipeline_utils import get_file_locator
            get_file_locator().add_discard_reason(
                foodcourt_id, fc_name or foodcourt_id,
                restaurant_id, rest_name or restaurant_id,
                item_id_val, item_name,
                "postprocessing",
                error_msg
            )
            # Track error in restaurant tracker
            if restaurant_tracker:
                restaurant_tracker.add_error(
                    foodcourt_id, restaurant_id, item_id_val,
                    error_msg, "postprocessing"
                )
            
            warning_msg = f"Not sold in val dates for {foodcourt_id}/{restaurant_id}/{item_name}"
            pipeline_logger.log_warning("postprocessing", warning_msg)
            # Don't print to console - only log to pipeline logs
            return False
        
        # Load training data from preprocessing (pass item_id if available)
        train_df, error_reason = load_training_data(foodcourt_id, restaurant_id, item_name, item_id)
        if train_df is None or train_df.empty:
            # Get preprocessing file path
            from src.util.pipeline_utils import get_file_name
            preprocessing_filename = get_file_name(foodcourt_id, restaurant_id, item_name, "preprocessing")
            preprocessing_path = PREPROCESSED_DIR / foodcourt_id / preprocessing_filename
            
            # Log to postprocessing_logs
            from src.util.pipeline_utils import get_result_file_name, get_output_base_dir, get_pipeline_type, create_excel_hyperlink
            output_base = get_output_base_dir()
            pipeline_type = get_pipeline_type()
            results_dir = output_base / "trainedModel" / pipeline_type / "results"
            
            # Check if model results exist
            model_result_exists = False
            preprocessing_exists = preprocessing_path.exists()
            input_file_link = "N/A"
            item_id_val = ""
            
            for model_type in ["XGBoost", "MovingAverage"]:
                # Try validation CSV file (try with item_id first, then without)
                result_filename = None
                result_path = None
                for item_id_to_try in [item_id, ""]:
                    result_filename = get_result_file_name(foodcourt_id, restaurant_id, item_name, "model_generation", model_type, item_id_to_try, "validation")
                    result_path = results_dir / result_filename
                    if result_path.exists():
                        break
                
                if result_path and result_path.exists():
                    model_result_exists = True
                    input_file_link = create_excel_hyperlink(result_path, result_filename)
                    # Try to get item_id from model results
                    try:
                        df_check = pd.read_csv(result_path, nrows=1)
                        for col in ["item_id", "menuitemid", "item_identifier"]:
                            if col in df_check.columns:
                                item_id_val = str(df_check[col].iloc[0])
                                break
                    except:
                        pass
                    break
            
            # Determine the specific error reason
            if model_result_exists:
                fc_name, rest_name = get_mongo_names(foodcourt_id, restaurant_id)
                if not item_id_val:
                    item_id_val = item_name
                
                # Create specific error message based on error_reason
                if error_reason == "file_not_found":
                    error_msg = "Preprocessing file not found"
                elif error_reason == "item_not_found":
                    error_msg = f"Item '{item_name}' not found in preprocessing file"
                elif error_reason == "no_data":
                    error_msg = "Preprocessing file exists but contains no data for this item"
                elif error_reason == "load_error":
                    error_msg = "Preprocessing file exists but could not be loaded (error reading file)"
                else:
                    error_msg = "Preprocessing file exists but could not be loaded"
                
                pipeline_logger.log_postprocessing_log(
                    foodcourt_id, fc_name or foodcourt_id,
                    restaurant_id, rest_name or restaurant_id,
                    item_id_val, item_name,
                    input_file_link,
                    error_msg
                )
                
                # Add discard reason to file_locator
                from src.util.pipeline_utils import get_file_locator
                get_file_locator().add_discard_reason(
                    foodcourt_id, fc_name or foodcourt_id,
                    restaurant_id, rest_name or restaurant_id,
                    item_id_val, item_name,
                    "postprocessing",
                    error_msg
                )
                # Track error in restaurant tracker
                if restaurant_tracker:
                    restaurant_tracker.add_error(
                        foodcourt_id, restaurant_id, item_id_val,
                        error_msg, "postprocessing"
                    )
            
            warning_msg = f"No training data available for {foodcourt_id}/{restaurant_id}/{item_name}, skipping redistribution"
            error_msg = error_reason if error_reason else "No training data available"
            pipeline_logger.log_warning("postprocessing", warning_msg)
            # Track error in restaurant tracker
            if restaurant_tracker:
                item_id_val = item_id if item_id else item_name
                restaurant_tracker.add_error(
                    foodcourt_id, restaurant_id, item_id_val,
                    error_msg, "postprocessing"
                )
            # Don't print to console - only log to pipeline logs
            return False
        
        # Calculate day-of-week percentages
        weekday_percentages = calculate_day_of_week_percentages(train_df, MODEL_DATE)
        if not weekday_percentages:
            warning_msg = f"Could not calculate day-of-week patterns for {foodcourt_id}/{restaurant_id}/{item_name}"
            error_msg = "Could not calculate day-of-week patterns"
            pipeline_logger.log_warning("postprocessing", warning_msg)
            # Track error in restaurant tracker
            if restaurant_tracker:
                item_id_val = item_id if item_id else item_name
                restaurant_tracker.add_error(
                    foodcourt_id, restaurant_id, item_id_val,
                    error_msg, "postprocessing"
                )
            LOGGER.warning(warning_msg)
            return False
        
        # Get names from MongoDB
        foodcourt_name, restaurant_name = get_mongo_names(foodcourt_id, restaurant_id)
        if not foodcourt_name:
            foodcourt_name = foodcourt_id
        if not restaurant_name:
            restaurant_name = restaurant_id
        
        # Get item_id from training data
        item_id = ""
        for col in ["menuitemid", "item_identifier"]:
            if col in train_df.columns and len(train_df) > 0:
                item_id = str(train_df[col].iloc[0])
                break
        if not item_id:
            item_id = item_name
        
        # Process each model's predictions - XGBoost, Prophet, and MovingAverage
        model_results = []
        for model_name, pred_df in all_predictions.items():
            # Normalize model name
            normalized_model_name = None
            is_moving_average = False
            
            if model_name.lower() in ["xgboost", "xgb"]:
                normalized_model_name = "XGBoost"
            elif model_name.lower() in ["prophet"]:
                normalized_model_name = "Prophet"
            elif "MovingAverage" in model_name or "moving_average" in model_name.lower():
                normalized_model_name = "MovingAverage"
                is_moving_average = True
            else:
                LOGGER.debug(f"Skipping postprocessing for {model_name} (unknown model type)")
                continue
            
            # Process MovingAverage with special logic
            if is_moving_average:
                if train_df is None or train_df.empty:
                    LOGGER.warning(f"Cannot process MovingAverage for {item_name}: no training data available")
                    continue
                
                ma_result, scenario = redistribute_moving_average_predictions(
                    pred_df, train_df,
                    foodcourt_id, foodcourt_name, restaurant_id, restaurant_name,
                    item_id, item_name
                )
                
                if not ma_result.empty:
                    # Store postprocessing metrics before renaming
                    postprocess_dev = ma_result["abs_deviation"].copy()
                    postprocess_error_pct = ma_result["abs_error_pct"].copy()
                    
                    # Rename columns to match expected format
                    ma_result = ma_result.rename(columns={
                        "predicted_count_original": f"{normalized_model_name}_predicted_count",
                        "abs_deviation": f"{normalized_model_name}_abs_deviation",
                        "abs_error_pct": f"{normalized_model_name}_abs_error_pct",
                        "predicted_count_postprocessing": f"{normalized_model_name}_predicted_count_postprocessing"
                    })
                    # Add postprocessing deviation and error
                    ma_result[f"{normalized_model_name}_postProcess_abs_dev"] = postprocess_dev
                    ma_result[f"{normalized_model_name}_postProcess_abs_pct"] = postprocess_error_pct
                    ma_result["item_count"] = ma_result["actual_count"]
                    # Add base columns if missing
                    if "foodcourt_id" not in ma_result.columns:
                        ma_result["foodcourt_id"] = foodcourt_id
                    if "foodcourt_name" not in ma_result.columns:
                        ma_result["foodcourt_name"] = foodcourt_name
                    if "restaurant_id" not in ma_result.columns:
                        ma_result["restaurant_id"] = restaurant_id
                    if "restaurant_name" not in ma_result.columns:
                        ma_result["restaurant_name"] = restaurant_name
                    if "item_id" not in ma_result.columns:
                        ma_result["item_id"] = item_id
                    if "item_name" not in ma_result.columns:
                        ma_result["item_name"] = item_name
                    ma_result["postprocessing_scenario"] = scenario
                    model_results.append(ma_result)
            else:
                # Process XGBoost and Prophet with standard logic
                model_result = process_model_predictions(
                    pred_df, normalized_model_name, weekday_percentages,
                    foodcourt_id, foodcourt_name, restaurant_id, restaurant_name,
                    item_id, item_name
                )
                if not model_result.empty:
                    model_results.append(model_result)
        
        if not model_results:
            warning_msg = f"No valid predictions to process for {foodcourt_id}/{restaurant_id}/{item_name}"
            error_msg = "No valid predictions to process"
            pipeline_logger.log_warning("postprocessing", warning_msg)
            # Track error in restaurant tracker
            if restaurant_tracker:
                item_id_val = item_id if item_id else item_name
                restaurant_tracker.add_error(
                    foodcourt_id, restaurant_id, item_id_val,
                    error_msg, "postprocessing"
                )
            LOGGER.warning(warning_msg)
            return False
        
        # Merge all models' results on date (outer join to include all dates)
        # Start with first model
        combined_df = model_results[0].copy()
        
        # Get base columns (same for all models)
        base_cols = ["foodcourt_id", "foodcourt_name", "restaurant_id", "restaurant_name", 
                     "item_id", "item_name", "date", "item_count"]
        
        # Merge remaining models
        for model_result in model_results[1:]:
            # Merge on date only, keeping all dates from both DataFrames
            combined_df = combined_df.merge(
                model_result,
                on=["date"],
                how="outer",
                suffixes=("", "_new")
            )
            
            # Update base columns from new model if they're missing (for new dates)
            for col in base_cols:
                if col in combined_df.columns and f"{col}_new" in combined_df.columns:
                    combined_df[col] = combined_df[col].fillna(combined_df[f"{col}_new"])
                    combined_df = combined_df.drop(columns=[f"{col}_new"])
            
            # Remove any other duplicate columns
            for col in combined_df.columns:
                if col.endswith("_new") and col.replace("_new", "") in combined_df.columns:
                    combined_df = combined_df.drop(columns=[col])
        
        # Ensure base columns are filled consistently
        if len(combined_df) > 0:
            for col in base_cols:
                if col in combined_df.columns:
                    # Fill missing values with first non-null value
                    first_val = combined_df[col].dropna().iloc[0] if not combined_df[col].dropna().empty else ""
                    combined_df[col] = combined_df[col].fillna(first_val)
        
        # Sort by date
        combined_df = combined_df.sort_values("date").reset_index(drop=True)
        
        # Save individual CSV file per item (FRI level)
        # Structure: postprocessing/FRI_LEVEL/{foodcourt_id}/{FC_id}_{R_id}_{item_name}_postprocessing.csv
        from src.util.pipeline_utils import get_file_name, get_file_locator, create_excel_hyperlink
        output_fc_dir = OUTPUT_BASE_DIR / foodcourt_id
        output_fc_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename (item_id is already extracted above)
        output_filename = get_file_name(foodcourt_id, restaurant_id, item_name, "postprocessing", item_id)
        output_filename = output_filename.replace(".xlsx", ".csv")  # Use CSV instead of Excel
        output_path = output_fc_dir / output_filename
        
        # Save to CSV
        combined_df.to_csv(output_path, index=False, encoding="utf-8")
        
        # Read item_id from CSV file (source of truth) to ensure we have the correct one
        csv_item_id = item_id  # Default to the one we extracted
        if "item_id" in combined_df.columns and len(combined_df) > 0:
            csv_item_id_val = combined_df["item_id"].dropna().iloc[0] if not combined_df["item_id"].dropna().empty else ""
            if csv_item_id_val:
                csv_item_id = str(csv_item_id_val)
        
        # Add to file locator with hyperlink
        file_locator = get_file_locator()
        file_locator.add_file(
            foodcourt_id, foodcourt_name,
            restaurant_id, restaurant_name,
            csv_item_id, item_name,
            "postprocessing",
            output_path
        )
        
        # Calculate and save postprocessing metrics to restaurant tracker
        if restaurant_tracker:
            try:
                from src.util.restaurant_tracker import calculate_metrics_from_df
                
                # Extract metrics for each model
                model_metrics = {}
                
                # Get all model names from combined_df columns
                model_names = set()
                for col in combined_df.columns:
                    if "_predicted_count_postprocessing" in col:
                        model_name = col.replace("_predicted_count_postprocessing", "")
                        model_names.add(model_name)
                
                # Calculate metrics for each model
                for model_name in model_names:
                    # Filter out rows where actual_count is 0 for validation metrics (as per requirements)
                    # Only calculate metrics on days where item was actually sold
                    validation_df = combined_df[combined_df.get("item_count", 0) > 0].copy()
                    
                    if not validation_df.empty:
                        # Calculate postprocessing metrics (after redistribution)
                        postprocessing_validation_metrics = calculate_metrics_from_df(
                            validation_df,
                            actual_col="item_count",
                            predicted_col=f"{model_name}_predicted_count_postprocessing",
                            error_pct_col=f"{model_name}_postProcess_abs_pct"
                        )
                        
                        # Calculate original metrics (before redistribution) - also ignoring days with actual_count = 0
                        original_validation_metrics = calculate_metrics_from_df(
                            validation_df,
                            actual_col="item_count",
                            predicted_col=f"{model_name}_predicted_count",
                            error_pct_col=f"{model_name}_abs_error_pct"
                        )
                        
                        # Use postprocessing metrics as validation_metrics (main metrics)
                        validation_metrics = postprocessing_validation_metrics
                        
                        # Store original metrics separately for reference
                        validation_metrics["original_abs_avg_deviation"] = original_validation_metrics.get("abs_avg_deviation", 0.0)
                        validation_metrics["original_avg_abs_accuracy_pct"] = original_validation_metrics.get("avg_abs_accuracy_pct", 0.0)
                    else:
                        validation_metrics = {
                            "abs_avg_deviation": 0.0, "avg_abs_accuracy_pct": 0.0,
                            "avg_abs_accuracy_pct_capped": 0.0,
                            "total_days": 0, "active_days": 0,
                            "original_abs_avg_deviation": 0.0,
                            "original_avg_abs_accuracy_pct": 0.0
                        }
                    
                    # For training metrics, we need to load from model results
                    # Use the same metrics structure but with training data
                    training_metrics = {
                        "abs_avg_deviation": 0.0, "avg_abs_accuracy_pct": 0.0,
                        "avg_abs_accuracy_pct_capped": 0.0,
                        "total_days": 0, "active_days": 0
                    }
                    
                    # Try to load training data from model results
                    try:
                        from src.util.pipeline_utils import get_result_file_name, get_output_base_dir, get_pipeline_type
                        output_base = get_output_base_dir()
                        pipeline_type = get_pipeline_type()
                        results_dir = output_base / "trainedModel" / pipeline_type / "results"
                        
                        training_filename = get_result_file_name(
                            foodcourt_id, restaurant_id, item_name, "model_generation",
                            model_name, csv_item_id, "training"
                        )
                        training_path = results_dir / training_filename
                        
                        if training_path.exists():
                            train_df = pd.read_csv(training_path)
                            training_metrics = calculate_metrics_from_df(
                                train_df,
                                actual_col="actual_count",
                                predicted_col="predicted_count",
                                error_pct_col="pct_error"
                            )
                    except Exception as train_exc:
                        LOGGER.debug(f"Could not load training metrics for {model_name}: {train_exc}")
                    
                    # Analyze accuracy if validation accuracy is below 75%
                    accuracy_reasons = []
                    validation_accuracy = validation_metrics.get("avg_abs_accuracy_pct", 0.0)
                    
                    if validation_accuracy < 75.0:
                        # Get validation DataFrame for analysis
                        validation_df_for_analysis = validation_df.copy()
                        if "item_count" not in validation_df_for_analysis.columns:
                            validation_df_for_analysis["item_count"] = validation_df_for_analysis.get("actual_count", 0)
                        
                        # Prepare training DataFrame for analysis
                        train_df_for_analysis = None
                        if train_df is not None and not train_df.empty:
                            train_df_for_analysis = train_df.copy()
                            # Ensure we have the right column name
                            if "count" in train_df_for_analysis.columns:
                                train_df_for_analysis["item_count"] = train_df_for_analysis["count"]
                            elif "actual_count" in train_df_for_analysis.columns:
                                train_df_for_analysis["item_count"] = train_df_for_analysis["actual_count"]
                            elif "item_count" not in train_df_for_analysis.columns:
                                train_df_for_analysis["item_count"] = 0
                        else:
                            # Try to load training data if not available
                            train_df_for_analysis, _ = load_training_data(foodcourt_id, restaurant_id, item_name, csv_item_id)
                            if train_df_for_analysis is not None and not train_df_for_analysis.empty:
                                if "count" in train_df_for_analysis.columns:
                                    train_df_for_analysis["item_count"] = train_df_for_analysis["count"]
                                elif "actual_count" in train_df_for_analysis.columns:
                                    train_df_for_analysis["item_count"] = train_df_for_analysis["actual_count"]
                                elif "item_count" not in train_df_for_analysis.columns:
                                    train_df_for_analysis["item_count"] = 0
                        
                        if train_df_for_analysis is not None and not train_df_for_analysis.empty:
                            accuracy_reasons = analyze_accuracy_reasons(
                                train_df_for_analysis,
                                validation_df_for_analysis,
                                validation_accuracy,
                                actual_col="item_count",
                                predicted_col=f"{model_name}_predicted_count_postprocessing"
                            )
                        else:
                            accuracy_reasons = ["Training data not available for analysis"]
                    
                    model_metrics[model_name] = {
                        "training": training_metrics,
                        "validation": validation_metrics,
                        "accuracy_reasons": accuracy_reasons if accuracy_reasons else None
                    }
                
                # Save to restaurant tracker
                restaurant_tracker.add_postprocessing_results(
                    foodcourt_id, restaurant_id, csv_item_id or item_name,
                    output_filename, model_metrics
                )
            except Exception as exc:
                LOGGER.warning(f"Failed to save postprocessing metrics to restaurant tracker: {exc}")
        
        # Removed INFO log: "Postprocessed predictions for..." to reduce log noise
        
        return True
        
    except Exception as exc:
        LOGGER.error("Error postprocessing %s/%s/%s: %s", foodcourt_id, restaurant_id, item_name, exc)
        from src.util.pipeline_utils import get_pipeline_logger, get_mongo_names
        pipeline_logger = get_pipeline_logger()
        fc_name, rest_name = get_mongo_names(foodcourt_id, restaurant_id)
        # Get item_id if available (it should be initialized at function start)
        item_id_val = item_id if item_id else item_name
        error_msg = f"Error: {exc}"
        pipeline_logger.log_postprocessing_error(
            foodcourt_id, fc_name or foodcourt_id,
            restaurant_id, rest_name or restaurant_id,
            item_id_val, item_name,
            error_msg
        )
        # Track error in restaurant tracker
        if restaurant_tracker:
            restaurant_tracker.add_error(
                foodcourt_id, restaurant_id, item_id_val,
                error_msg, "postprocessing"
            )
        return False


def load_items_from_results(retrain_config: Optional[dict] = None, restaurant_tracker=None) -> Optional[pd.DataFrame]:
    """
    Scan the trainedModel/results/ directory to find all items that have validation results.
    Filter by foodcourt_ids and restaurant_ids from retrain.json if provided.
    
    File naming pattern: {foodcourt_id}_{restaurant_id}_{item_id}_{item_name}_model_generation_{model_name}_{data_type}.csv
    
    Args:
        retrain_config: Optional retrain configuration dict from retrain.json
                       If provided and postprocessing has foodcourt_ids/restaurant_ids, only process those.
                       If postprocessing is empty, process all foodcourts.
        restaurant_tracker: Optional RestaurantTracker instance to get item_id from tracking files
    
    Returns:
        DataFrame with columns: foodcourt_id, restaurant_id, item_name, item_id
        Returns None if results directory doesn't exist or is empty.
    """
    if not TRAINED_MODEL_RESULTS_DIR.exists():
        logging.error("Results directory not found at %s", TRAINED_MODEL_RESULTS_DIR)
        return None
    
    # Get foodcourt_ids, restaurant_ids, and item_ids to filter by (if retrain_config provided)
    allowed_foodcourt_ids = None
    allowed_restaurant_ids = None
    item_ids_filter = None
    if retrain_config:
        from src.util.pipeline_utils import get_retrain_config_for_step
        step_config = get_retrain_config_for_step("postprocessing")
        foodcourt_ids = step_config.get("foodcourt_ids", [])
        restaurant_ids = step_config.get("restaurant_ids", [])
        item_ids = step_config.get("item_ids", [])
        item_names = step_config.get("item_names", [])
        
        if foodcourt_ids:
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
    
    items_set = set()  # Use set to avoid duplicates
    
    # Scan all CSV files in results directory (look for validation files)
    for filename in os.listdir(TRAINED_MODEL_RESULTS_DIR):
        if not filename.endswith('.csv'):
            continue
        
        # Parse filename: {foodcourt_id}_{restaurant_id}_{item_id}_{item_name}_model_generation_{model_name}_{data_type}.csv
        # Only process validation files (skip training files)
        if not filename.endswith('_validation.csv'):
            continue
        
        # Remove "_validation.csv" suffix, then check for model_generation
        base_name = filename.replace("_validation.csv", "").replace(".csv", "")
        
        # Check if it's a model_generation result file
        if "_model_generation_" not in base_name:
            continue
        
        # Remove "_model_generation_{model_name}" suffix
        prefix = base_name.split("_model_generation_")[0]
        
        # Split by underscore - but item_name might contain underscores
        # Format can be: {fc_id}_{rest_id}_{item_id}_{item_name} or {fc_id}_{rest_id}_{item_name}
        parts = prefix.split('_')
        
        if len(parts) < 3:
            logging.warning("Skipping file with unexpected format: %s", filename)
            continue
        
        # First part is foodcourt_id
        foodcourt_id = parts[0]
        
        # Filter by foodcourt_id if retrain_config specifies it
        if allowed_foodcourt_ids is not None and foodcourt_id not in allowed_foodcourt_ids:
            continue
        
        # Second part is restaurant_id
        restaurant_id = parts[1]
        
        # Filter by restaurant_id if retrain_config specifies it
        if allowed_restaurant_ids is not None and restaurant_id not in allowed_restaurant_ids:
            continue
        
        # Determine if item_id is present (3rd part might be item_id or start of item_name)
        # If we have 4+ parts, assume 3rd part is item_id, rest is item_name
        # If we have 3 parts, assume no item_id, 3rd part is start of item_name
        item_id_from_filename = None
        if len(parts) >= 4:
            # New format with item_id: {fc_id}_{rest_id}_{item_id}_{item_name}
            # Check if parts[2] looks like an ObjectId (24 hex chars)
            potential_item_id = parts[2]
            if len(potential_item_id) == 24 and all(c in '0123456789abcdef' for c in potential_item_id.lower()):
                # parts[2] is item_id, parts[3:] is item_name
                item_id_from_filename = potential_item_id
                item_name = '_'.join(parts[3:])
            else:
                # parts[2] is part of item_name (old format without item_id)
                item_name = '_'.join(parts[2:])
        else:
            # Old format without item_id: {fc_id}_{rest_id}_{item_name}
            # item_name starts at parts[2]
            item_name = '_'.join(parts[2:])
        
        # Try to read the original item_name from preprocessed data file (source of truth)
        # Validation CSV files don't have item_name column, so we need to get it from preprocessed data
        # This ensures proper matching with retrain.json which uses original item names
        from src.util.pipeline_utils import get_item_name_from_excel, has_upstream_error, sanitize_name
        item_name_from_file = None
        
        # Try to get original item name from preprocessed data file by scanning for files with matching item_id
        if item_id_from_filename and (PREPROCESSED_DIR / foodcourt_id).exists():
            fc_path = PREPROCESSED_DIR / foodcourt_id
            # Look for preprocessed file that contains this item_id in the filename
            for preprocessed_file in fc_path.glob(f"*_{item_id_from_filename}_*_preprocessing.csv"):
                try:
                    item_name_from_file = get_item_name_from_excel(preprocessed_file)
                    if item_name_from_file:
                        break
                except Exception as e:
                    logging.debug(f"Failed to read item name from preprocessed file {preprocessed_file.name}: {e}")
                    continue
        
        # Fallback: try reading from validation CSV (though it likely won't have item_name column)
        # Removed exists() check - just try to read and catch exception
        if not item_name_from_file:
            file_path = TRAINED_MODEL_RESULTS_DIR / filename
            try:
                item_name_from_file = get_item_name_from_excel(file_path)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                pass
            except Exception as e:
                logging.debug(f"Failed to read item name from {filename}: {e}")
        
        # Final fallback: use sanitized name from filename
        if not item_name_from_file:
            item_name_from_file = item_name
            logging.debug("Using item_name from filename for %s: %s (note: this is sanitized)", filename, item_name)
        
        # Try to get item_id from restaurant_tracker if available
        item_id_to_use = item_id_from_filename
        if restaurant_tracker and not item_id_to_use:
            # Try to get item_id from tracking file by matching item_name
            try:
                tracking_data = restaurant_tracker._load_tracking_file(foodcourt_id, restaurant_id)
                for item_id_key, item_data in tracking_data.items():
                    if item_id_key == "_metadata":
                        continue
                    stored_item_name = item_data.get("item_name", "")
                    if stored_item_name and stored_item_name.lower() == item_name_from_file.lower():
                        item_id_to_use = item_id_key
                        break
            except:
                pass
        
        # Check if upstream steps (enrich_data, preprocessing, model_generation) have errors
        # If so, skip this item (no point processing if upstream failed)
        if has_upstream_error(foodcourt_id, restaurant_id, item_id_to_use, item_name_from_file, "postprocessing"):
            logging.debug("Skipping %s/%s/%s - upstream step has error", foodcourt_id, restaurant_id, item_name_from_file)
            continue
        
        # Filter by item_ids or item_names if retrain_config specifies it
        # Always use item_name read from CSV (source of truth) for filtering
        if item_ids_filter is not None or item_names_filter is not None:
            from src.util.pipeline_utils import matches_item_filter
            if not matches_item_filter(foodcourt_id, restaurant_id, item_name_from_file, item_id_to_use, 
                                      item_ids_filter, item_names_filter):
                logging.debug("Skipping %s/%s/%s - not in item_ids/item_names filter", foodcourt_id, restaurant_id, item_name_from_file)
                continue
        
        # Add to set (will automatically handle duplicates)
        # Include item_id in the tuple so we can use it later
        items_set.add((foodcourt_id, restaurant_id, item_name_from_file, item_id_to_use or ""))
    
    if not items_set:
        logging.warning("No validation result files found in %s", TRAINED_MODEL_RESULTS_DIR)
        return None
    
    items_list = [{"foodcourt_id": fc_id, "restaurant_id": r_id, "item_name": item_name, "item_id": item_id} 
                  for fc_id, r_id, item_name, item_id in items_set]
    items_df = pd.DataFrame(items_list)
    logging.info("Found %d items with validation results in %s", len(items_df), TRAINED_MODEL_RESULTS_DIR)
    
    if allowed_foodcourt_ids or allowed_restaurant_ids:
        logging.info("Filtered to %d items after applying retrain.json filter", len(items_df))
    
    return items_df


def main(retrain_config: Optional[dict] = None, file_saver=None, restaurant_tracker=None):
    """
    Main entry point for postprocessing.
    
    Args:
        retrain_config: Optional retrain configuration dict from retrain.json
        file_saver: Optional FileSaver instance for saving files
        restaurant_tracker: Optional RestaurantTracker instance for tracking item status
    """
    from src.util.pipeline_utils import get_pipeline_logger
    pipeline_logger = get_pipeline_logger()
    
    LOGGER.info("=" * 80)
    LOGGER.info("Starting postprocessing stage: Day-of-week redistribution")
    LOGGER.info("=" * 80)
    
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Count validation results files (try multiple patterns)
    validation_files = 0
    if TRAINED_MODEL_RESULTS_DIR.exists():
        # Try different patterns to match files with/without item_id
        pattern1 = list(TRAINED_MODEL_RESULTS_DIR.glob("*_model_generation_*_validation.csv"))
        pattern2 = list(TRAINED_MODEL_RESULTS_DIR.glob("*_*_model_generation_*_validation.csv"))  # With item_id
        validation_files = len(set(pattern1 + pattern2))  # Remove duplicates
    
    # Count postprocessing output files
    postprocessed_files = 0
    if OUTPUT_BASE_DIR.exists():
        for fc_dir in os.listdir(OUTPUT_BASE_DIR):
            fc_path = OUTPUT_BASE_DIR / fc_dir
            if fc_path.is_dir():
                postprocessed_files += len(list(fc_path.glob("*.csv")))
    
    # Load items from results directory (filtered by retrain.json if provided)
    filter_df = load_items_from_results(retrain_config=retrain_config, restaurant_tracker=restaurant_tracker)
    if filter_df is None or len(filter_df) == 0:
        error_msg = f"No validation result files found in {TRAINED_MODEL_RESULTS_DIR}"
        pipeline_logger.log_general_error("postprocessing", error_msg)
        LOGGER.error("=" * 80)
        LOGGER.error("ERROR: %s", error_msg)
        if retrain_config:
            from src.util.pipeline_utils import get_retrain_config_for_step
            step_config = get_retrain_config_for_step("postprocessing")
            if step_config.get("foodcourt_ids") or step_config.get("restaurant_ids"):
                LOGGER.error("Retrain.json specifies filters: %s", step_config)
                LOGGER.error("No validation result files found for these filters.")
        LOGGER.error("=" * 80)
        return
    
    # Log process start with summary
    summary = {
        "trainedModel/results/ (input)": f"{validation_files} validation result files",
        "postprocessing/ (existing)": f"{postprocessed_files} files",
        "Items to process": f"{len(filter_df)} items from results directory"
    }
    pipeline_logger.log_process_start("postprocessing", summary)
    pipeline_logger.log_info("postprocessing", f"✅ Results directory found: {TRAINED_MODEL_RESULTS_DIR}")
    LOGGER.info("✅ Results directory found: %s", TRAINED_MODEL_RESULTS_DIR)
    LOGGER.info("Loaded %d items with validation results to process", len(filter_df))
    
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    
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
            prefix="Postprocessing",
            suffix="items",
            length=40,
            show_elapsed=True
        )
    
    # Iterate through items from results directory
    for idx, row in filter_df.iterrows():
        foodcourt_id = str(row["foodcourt_id"]).strip()
        restaurant_id = str(row["restaurant_id"]).strip()
        item_name = str(row["item_name"]).strip()
        item_id = str(row.get("item_id", "")).strip() if "item_id" in row else ""
        if not item_id:
            item_id = ""
        
        item_idx = idx + 1
        
        # Display progress: Item: x/y | Time: HH:mm:SS (using global pipeline time)
        if pipeline_start_time:
            elapsed = time.time() - pipeline_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            logging.info("Item: %d/%d | Time: %s", item_idx, total_items, time_str)
        
        # Track item processing progress
        item_start_time = time.time()
        
        # Check retrain logic
        from src.util.pipeline_utils import should_force_retrain
        force_retrain = should_force_retrain("postprocessing", foodcourt_id, restaurant_id, item_name, item_id)
        
        # Check if postprocessed data already exists
        from src.util.pipeline_utils import get_file_name
        output_filename = get_file_name(foodcourt_id, restaurant_id, item_name, "postprocessing")
        output_filename = output_filename.replace(".xlsx", ".csv")  # Use CSV
        output_path = OUTPUT_BASE_DIR / foodcourt_id / output_filename
        
        if not force_retrain and output_path.exists():
            skipped_count += 1
            items_processed += 1
            # Update progress bar for skipped items
            if progress:
                progress.set_current(item_idx)
            continue
        
        # Process the item (pass item_id if available)
        if postprocess_item(foodcourt_id, restaurant_id, item_name, restaurant_tracker=restaurant_tracker, item_id=item_id):
            processed_count += 1
            items_processed += 1
            # Update progress bar
            if progress:
                progress.set_current(item_idx)
        else:
            failed_count += 1
            items_processed += 1
            # Update progress bar for failed items
            if progress:
                progress.set_current(item_idx)
    
    LOGGER.info("=" * 80)
    LOGGER.info("Postprocessing complete: Processed=%d, Failed=%d, Skipped=%d", 
               processed_count, failed_count, skipped_count)
    LOGGER.info("=" * 80)
    
    # Log process results
    results = {
        "Items processed": len(filter_df),
        "Successfully processed": processed_count,
        "Failed": failed_count,
        "Skipped": skipped_count
    }
    pipeline_logger.log_process_results("postprocessing", results)