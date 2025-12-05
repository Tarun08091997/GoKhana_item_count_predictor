"""
Step 6: Compiled Result Generation

This script creates final summary metrics for each item by analyzing postprocessing results:

1. **CAPPED Summary** (per item):
   - Selects best model per day (lowest error)
   - Caps errors at 100%
   - Calculates average accuracy and deviation
   - Includes: model name, postProcessing_used flag, abs_avg_deviation, abs_avg_accuracy

2. **ORIGINAL Summary** (per item):
   - Shows all models tried
   - Includes: total_days, active_days, models_count, models_tried, model_selected

**Output**: All summaries are saved to restaurant_tracker JSON files under each item's "compiled_results" key.
No CSV files are created - everything is in the tracking JSON files.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input_data"
FILTER_CSV_PATH = INPUT_DIR / "train_model_for_items.csv"
# Read from output_data/trainedModel/{pipeline_type}/results/ and output_data/postprocessing/{pipeline_type}/
# Write to output_data/compiled_results/{pipeline_type}/
from src.util.pipeline_utils import get_output_base_dir, get_pipeline_type
OUTPUT_BASE = get_output_base_dir()
PIPELINE_TYPE = get_pipeline_type()
TRAINED_MODEL_RESULTS_DIR = OUTPUT_BASE / "trainedModel" / PIPELINE_TYPE / "results"
POSTPROCESSING_DIR = OUTPUT_BASE / "postprocessing" / PIPELINE_TYPE
TRAINED_MODEL_RESULTS_DIR_FULL = OUTPUT_BASE / "trainedModel" / PIPELINE_TYPE / "results"
    # Removed OUTPUT_BASE_DIR for compiled_results - using restaurant_tracking instead


def create_capped_compiled_results(postprocessing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create CAPPED compiled results:
    - Cap abs_error_pct at 100 for each day
    - Select best model per day (lowest error)
    - Include: model name, postProcessing_used, abs_avg_deviation, abs_avg_accuracy
    - Average across all validation dates (excluding 0 count days)
    """
    if postprocessing_df.empty:
        return pd.DataFrame()
    
    # Get all model names from column names
    model_names = set()
    for col in postprocessing_df.columns:
        if "_predicted_count" in col and not col.endswith("_postprocessing"):
            model_name = col.replace("_predicted_count", "")
            model_names.add(model_name)
    
    capped_rows = []
    
    # Group by item and date
    for (fc_id, rest_id, item_id, date), group in postprocessing_df.groupby(
        ["foodcourt_id", "restaurant_id", "item_id", "date"]
    ):
        # Skip if item_count is 0
        if "item_count" in group.columns and group["item_count"].iloc[0] == 0:
            continue
        
        # Get base info
        row = {
            "foodcourt_id": fc_id,
            "foodcourt_name": group["foodcourt_name"].iloc[0] if "foodcourt_name" in group.columns else "",
            "restaurant_id": rest_id,
            "restaurant_name": group["restaurant_name"].iloc[0] if "restaurant_name" in group.columns else "",
            "item_id": item_id,
            "item_name": group["item_name"].iloc[0] if "item_name" in group.columns else "",
            "date": date,
            "item_count": group["item_count"].iloc[0] if "item_count" in group.columns else 0,
        }
        
        # Find best model (lowest abs_error_pct, preferring postprocessing if available)
        best_model = None
        best_error = float('inf')
        best_is_postprocessing = False
        
        for model_name in model_names:
            # Try postprocessing first
            post_error_col = f"{model_name}_postProcess_abs_pct"
            if post_error_col in group.columns:
                error_val = group[post_error_col].iloc[0]
                if pd.notna(error_val):
                    capped_error = min(abs(float(error_val)), 100.0)
                    if capped_error < best_error:
                        best_error = capped_error
                        best_model = model_name
                        best_is_postprocessing = True
            
            # Try original if no postprocessing
            if best_model != model_name:
                orig_error_col = f"{model_name}_abs_error_pct"
                if orig_error_col in group.columns:
                    error_val = group[orig_error_col].iloc[0]
                    if pd.notna(error_val):
                        capped_error = min(abs(float(error_val)), 100.0)
                        if capped_error < best_error:
                            best_error = capped_error
                            best_model = model_name
                            best_is_postprocessing = False
        
        if best_model:
            row["model"] = best_model
            row["postProcessing_used"] = "Yes" if best_is_postprocessing else "No"
            
            # Get values from best model
            if best_is_postprocessing:
                pred_val = group[f"{best_model}_predicted_count_postprocessing"].iloc[0] if f"{best_model}_predicted_count_postprocessing" in group.columns else 0
                row["predicted_count"] = int(np.nan_to_num(pred_val, nan=0.0))
                row["abs_deviation"] = group[f"{best_model}_postProcess_abs_dev"].iloc[0] if f"{best_model}_postProcess_abs_dev" in group.columns else 0
                error_val = group[f"{best_model}_postProcess_abs_pct"].iloc[0] if f"{best_model}_postProcess_abs_pct" in group.columns else 0
                row["abs_error_pct"] = min(abs(float(np.nan_to_num(error_val, nan=0.0))), 100.0)
            else:
                pred_val = group[f"{best_model}_predicted_count"].iloc[0] if f"{best_model}_predicted_count" in group.columns else 0
                row["predicted_count"] = int(np.nan_to_num(pred_val, nan=0.0))
                row["abs_deviation"] = group[f"{best_model}_abs_deviation"].iloc[0] if f"{best_model}_abs_deviation" in group.columns else 0
                error_val = group[f"{best_model}_abs_error_pct"].iloc[0] if f"{best_model}_abs_error_pct" in group.columns else 0
                row["abs_error_pct"] = min(abs(float(np.nan_to_num(error_val, nan=0.0))), 100.0)
            
            capped_rows.append(row)
    
    if not capped_rows:
        return pd.DataFrame()
    
    capped_df = pd.DataFrame(capped_rows)
    
    # Calculate averages per item (excluding 0 count days)
    summary_rows = []
    for (fc_id, rest_id, item_id), group in capped_df.groupby(["foodcourt_id", "restaurant_id", "item_id"]):
        # Exclude 0 count days
        group = group[group["item_count"] != 0]
        if group.empty:
            continue
        
        summary_row = {
            "foodcourt_id": fc_id,
            "foodcourt_name": group["foodcourt_name"].iloc[0],
            "restaurant_id": rest_id,
            "restaurant_name": group["restaurant_name"].iloc[0],
            "item_id": item_id,
            "item_name": group["item_name"].iloc[0],
            "model": group["model"].mode().iloc[0] if not group["model"].empty else "",  # Most common model
            "postProcessing_used": group["postProcessing_used"].mode().iloc[0] if not group["postProcessing_used"].empty else "No",
            "abs_avg_deviation": group["abs_deviation"].mean(),
            "abs_avg_accuracy": 100.0 - group["abs_error_pct"].mean(),  # 100 - avg_error
        }
        summary_rows.append(summary_row)
    
    return pd.DataFrame(summary_rows)


def create_original_compiled_results(postprocessing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ORIGINAL compiled results:
    - No capping
    - Exclude days where item_count = 0
    - Show all models
    - Include training metrics (active training count, etc.)
    """
    if postprocessing_df.empty:
        return pd.DataFrame()
    
    # Exclude 0 count days
    df = postprocessing_df[postprocessing_df["item_count"] != 0].copy()
    
    if df.empty:
        return pd.DataFrame()
    
    # Get all model names
    model_names = set()
    for col in df.columns:
        if "_predicted_count" in col and not col.endswith("_postprocessing"):
            model_name = col.replace("_predicted_count", "")
            model_names.add(model_name)
    
    # Create rows with all models' data
    result_rows = []
    
    for _, row in df.iterrows():
        base_row = {
            "foodcourt_id": row["foodcourt_id"],
            "foodcourt_name": row.get("foodcourt_name", ""),
            "restaurant_id": row["restaurant_id"],
            "restaurant_name": row.get("restaurant_name", ""),
            "item_id": row["item_id"],
            "item_name": row.get("item_name", ""),
            "date": row["date"],
            "item_count": row["item_count"],
        }
        
        # Add data for each model
        for model_name in model_names:
            # Original predictions
            if f"{model_name}_predicted_count" in row:
                val = row[f"{model_name}_predicted_count"]
                base_row[f"{model_name}_predicted_count"] = int(np.nan_to_num(val, nan=0.0)) if pd.notna(val) else 0
            if f"{model_name}_abs_deviation" in row:
                base_row[f"{model_name}_abs_deviation"] = np.nan_to_num(row[f"{model_name}_abs_deviation"], nan=0.0)
            if f"{model_name}_abs_error_pct" in row:
                base_row[f"{model_name}_abs_error_pct"] = np.nan_to_num(row[f"{model_name}_abs_error_pct"], nan=0.0)
            
            # Postprocessing predictions
            if f"{model_name}_predicted_count_postprocessing" in row:
                val = row[f"{model_name}_predicted_count_postprocessing"]
                base_row[f"{model_name}_predicted_count_postprocessing"] = int(np.nan_to_num(val, nan=0.0)) if pd.notna(val) else 0
            if f"{model_name}_postProcess_abs_dev" in row:
                base_row[f"{model_name}_postProcess_abs_dev"] = np.nan_to_num(row[f"{model_name}_postProcess_abs_dev"], nan=0.0)
            if f"{model_name}_postProcess_abs_pct" in row:
                base_row[f"{model_name}_postProcess_abs_pct"] = np.nan_to_num(row[f"{model_name}_postProcess_abs_pct"], nan=0.0)
        
        result_rows.append(base_row)
    
    result_df = pd.DataFrame(result_rows)
    
    # TODO: Add training metrics (active training count, etc.) - need to load from model results
    # For now, return the basic structure
    
    return result_df


def load_model_results_to_postprocessing_format() -> pd.DataFrame:
    """
    Load model results from trainedModel/results/ and convert to postprocessing format.
    This allows compiled results to be created even when postprocessing doesn't exist.
    """
    all_results = []
    
    if not TRAINED_MODEL_RESULTS_DIR.exists():
        return pd.DataFrame()
    
    # Scan all Excel files in results directory
    for filename in os.listdir(TRAINED_MODEL_RESULTS_DIR):
        if not filename.endswith("_model_generation_*.xlsx"):
            # Check if it matches the pattern
            if "_model_generation_" not in filename or not filename.endswith(".xlsx"):
                continue
        
        file_path = TRAINED_MODEL_RESULTS_DIR / filename
        try:
            # Parse filename: {fc_id}_{rest_id}_{item_name}_model_generation_{model_name}.xlsx
            base_name = filename.replace(".xlsx", "")
            if "_model_generation_" not in base_name:
                continue
            
            prefix = base_name.split("_model_generation_")[0]
            model_name = base_name.split("_model_generation_")[1]
            parts = prefix.split('_')
            
            if len(parts) < 3:
                continue
            
            foodcourt_id = parts[0]
            restaurant_id = parts[1]
            item_name = '_'.join(parts[2:])
            
            # Read Validation Data sheet
            try:
                val_df = pd.read_excel(file_path, sheet_name="Validation Data")
            except Exception as exc:
                LOGGER.warning("Failed to read Validation Data sheet from %s: %s", filename, exc)
                continue
            
            if val_df.empty:
                continue
            
            # Ensure date column exists and is in string format (YYYY-MM-DD)
            if "date" not in val_df.columns:
                # Try to find date column with different names
                for col in ["Date", "DATE", "date"]:
                    if col in val_df.columns:
                        val_df = val_df.rename(columns={col: "date"})
                        break
                else:
                    LOGGER.warning("No date column found in %s", filename)
                    continue
            
            # Convert date to string format (YYYY-MM-DD)
            val_df["date"] = pd.to_datetime(val_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            # Remove rows with invalid dates
            val_df = val_df[val_df["date"].notna()]
            
            # Get metadata from first row or filename
            foodcourt_name = val_df.get("foodcourt_name", pd.Series([""])).iloc[0] if "foodcourt_name" in val_df.columns else ""
            restaurant_name = val_df.get("restaurant_name", pd.Series([""])).iloc[0] if "restaurant_name" in val_df.columns else ""
            item_id = val_df.get("item_id", pd.Series([""])).iloc[0] if "item_id" in val_df.columns else ""
            if not item_id:
                item_id = item_name
            
            # Convert to postprocessing format
            result_df = pd.DataFrame({
                "foodcourt_id": foodcourt_id,
                "foodcourt_name": foodcourt_name,
                "restaurant_id": restaurant_id,
                "restaurant_name": restaurant_name,
                "item_id": item_id,
                "item_name": item_name,
                "date": val_df["date"],
                "item_count": val_df.get("actual_count", val_df.get("count", 0)),
                f"{model_name}_predicted_count": val_df.get("predicted_count", 0),
                f"{model_name}_abs_deviation": np.abs(val_df.get("predicted_count", 0) - val_df.get("actual_count", val_df.get("count", 0))),
                f"{model_name}_abs_error_pct": np.abs(val_df.get("pct_error", val_df.get("error_pct", 0))),
            })
            
            all_results.append(result_df)
            
        except Exception as exc:
            LOGGER.warning("Failed to load model result %s: %s", filename, exc)
            continue
    
    if not all_results:
        return pd.DataFrame()
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df


def main(retrain_config: Optional[dict] = None, file_saver=None, restaurant_tracker=None):
    """
    Main entry point for error compilation.
    Now saves compiled results to restaurant tracker JSON files instead of CSV.
    
    Args:
        retrain_config: Optional retrain configuration dict from retrain.json
        file_saver: Optional FileSaver instance for saving files
        restaurant_tracker: Optional RestaurantTracker instance for tracking item status
    """
    from src.util.pipeline_utils import get_pipeline_logger
    pipeline_logger = get_pipeline_logger()
    
    LOGGER.info("=" * 80)
    LOGGER.info("Compiling all model errors into comprehensive table")
    LOGGER.info("=" * 80)
    
    # Removed OUTPUT_BASE_DIR.mkdir - compiled_results directory no longer needed
    
    # Count validation results files
    validation_files = 0
    if TRAINED_MODEL_RESULTS_DIR.exists():
        validation_files = len(list(TRAINED_MODEL_RESULTS_DIR.glob("*_model_generation_*.xlsx")))
    
    # Count postprocessing CSV files
    postprocessing_files = 0
    if POSTPROCESSING_DIR.exists():
        for fc_dir in os.listdir(POSTPROCESSING_DIR):
            fc_path = POSTPROCESSING_DIR / fc_dir
            if fc_path.is_dir():
                postprocessing_files += len(list(fc_path.glob("*_postprocessing.csv")))
    
    # Log process start with summary
    summary = {
        "trainedModel/results/ (input)": f"{validation_files} validation result files",
        "postprocessing/ (input)": f"{postprocessing_files} postprocessing CSV files",
        "Items to process": "From postprocessing directory and model results"
    }
    pipeline_logger.log_process_start("compiled_result_generation", summary)
    
    try:
        # Load all postprocessing CSV files from directory
        postprocessing_dfs = []
        postprocessing_loaded = False
        
        if POSTPROCESSING_DIR.exists():
            postprocessing_loaded = True
        
        # Get retrain config for filtering (if provided)
        allowed_foodcourt_ids = None
        allowed_restaurant_ids = None
        item_ids_filter = None
        if retrain_config:
            from src.util.pipeline_utils import get_retrain_config_for_step
            step_config = get_retrain_config_for_step("compiled_result_generation")
            foodcourt_ids = step_config.get("foodcourt_ids", [])
            restaurant_ids = step_config.get("restaurant_ids", [])
            item_ids = step_config.get("item_ids", [])
            item_names = step_config.get("item_names", [])
            
            if foodcourt_ids:
                allowed_foodcourt_ids = set(str(fc_id).strip() for fc_id in foodcourt_ids)
                LOGGER.info("Filtering by foodcourt_ids from retrain.json: %s", allowed_foodcourt_ids)
            if restaurant_ids:
                allowed_restaurant_ids = set(str(r_id).strip() for r_id in restaurant_ids)
                LOGGER.info("Filtering by restaurant_ids from retrain.json: %s", allowed_restaurant_ids)
            if item_ids:
                item_ids_filter = item_ids
                LOGGER.info("Filtering by item_ids from retrain.json: %d item filters", len(item_ids))
            if item_names:
                item_names_filter = item_names
                LOGGER.info("Filtering by item_names from retrain.json: %d item names", len(item_names))
            else:
                item_names_filter = None
        
        # Scan postprocessing directory: postprocessing/FRI_LEVEL/{foodcourt_id}/*_postprocessing.csv
        if POSTPROCESSING_DIR.exists():
            for fc_dir in os.listdir(POSTPROCESSING_DIR):
                fc_path = POSTPROCESSING_DIR / fc_dir
                if not fc_path.is_dir():
                    continue
                
                # Filter by foodcourt_id if retrain_config specifies it
                if allowed_foodcourt_ids is not None and fc_dir not in allowed_foodcourt_ids:
                    continue
                
                for filename in os.listdir(fc_path):
                    if filename.endswith("_postprocessing.csv"):
                        # Parse filename to get restaurant_id and item info for filtering
                        # Format: {fc_id}_{rest_id}_{item_id}_{item_name}_postprocessing.csv or {fc_id}_{rest_id}_{item_name}_postprocessing.csv
                        base_name = filename.replace("_postprocessing.csv", "")
                        parts = base_name.split('_')
                        if len(parts) >= 2:
                            file_fc_id = parts[0]
                            file_rest_id = parts[1]
                            
                            # Filter by restaurant_id if retrain_config specifies it
                            if allowed_restaurant_ids is not None and file_rest_id not in allowed_restaurant_ids:
                                continue
                        
                        csv_path = fc_path / filename
                        try:
                            df = pd.read_csv(csv_path)
                            if not df.empty:
                                # Filter by item_ids or item_names if specified (check after loading to get actual item_name/item_id from data)
                                if item_ids_filter is not None or item_names_filter is not None:
                                    # Get item_id and item_name from first row of dataframe (more reliable than filename)
                                    item_id_from_df = ""
                                    item_name_from_df = ""
                                    if "item_id" in df.columns and len(df) > 0:
                                        item_id_from_df = str(df["item_id"].iloc[0]) if pd.notna(df["item_id"].iloc[0]) else ""
                                    if "item_name" in df.columns and len(df) > 0:
                                        item_name_from_df = str(df["item_name"].iloc[0]) if pd.notna(df["item_name"].iloc[0]) else ""
                                    if not item_id_from_df:
                                        item_id_from_df = item_name_from_df
                                    
                                    from src.util.pipeline_utils import matches_item_filter
                                    if not matches_item_filter(fc_dir, file_rest_id, item_name_from_df, item_id_from_df, 
                                                              item_ids_filter, item_names_filter):
                                        # Item not in filter, skip it
                                        continue
                                
                                postprocessing_dfs.append(df)
                        except Exception as exc:
                            LOGGER.warning("Failed to load postprocessing CSV %s: %s", csv_path, exc)
        
        # If no postprocessing files, try loading from model results
        if not postprocessing_dfs:
            LOGGER.info("No postprocessing CSV files found in %s, falling back to model results...", POSTPROCESSING_DIR)
            model_results_df = load_model_results_to_postprocessing_format()
            if not model_results_df.empty:
                postprocessing_df = model_results_df
                LOGGER.info("Successfully loaded model results: %d rows (using model validation data as fallback)", len(postprocessing_df))
            else:
                error_msg = f"No postprocessing CSV files found in {POSTPROCESSING_DIR} and no model results found in {TRAINED_MODEL_RESULTS_DIR}. Cannot generate compiled results."
                pipeline_logger.log_general_error("compiled_result_generation", error_msg)
                LOGGER.error(error_msg)
                return
        else:
            # Combine all postprocessing DataFrames
            postprocessing_df = pd.concat(postprocessing_dfs, ignore_index=True)
            LOGGER.info("Loaded postprocessing results: %d rows from %d files", len(postprocessing_df), len(postprocessing_dfs))
            
            # Also load model results for items that don't have postprocessing
            model_results_df = load_model_results_to_postprocessing_format()
            if not model_results_df.empty:
                # Find items in model results that aren't in postprocessing
                postprocessing_items = set(
                    postprocessing_df.groupby(["foodcourt_id", "restaurant_id", "item_id"]).groups.keys()
                )
                model_items = set(
                    model_results_df.groupby(["foodcourt_id", "restaurant_id", "item_id"]).groups.keys()
                )
                missing_items = model_items - postprocessing_items
                
                if missing_items:
                    # Add missing items from model results
                    missing_dfs = []
                    for (fc_id, rest_id, item_id) in missing_items:
                        item_df = model_results_df[
                            (model_results_df["foodcourt_id"] == fc_id) &
                            (model_results_df["restaurant_id"] == rest_id) &
                            (model_results_df["item_id"] == item_id)
                        ]
                        if not item_df.empty:
                            missing_dfs.append(item_df)
                    
                    if missing_dfs:
                        missing_df = pd.concat(missing_dfs, ignore_index=True)
                        postprocessing_df = pd.concat([postprocessing_df, missing_df], ignore_index=True)
                        LOGGER.info("Added %d items from model results (missing postprocessing)", len(missing_items))
        
        if postprocessing_df.empty:
            LOGGER.warning("Combined data is empty")
            return
        
        # Create compiled results summaries
        capped_df = create_capped_compiled_results(postprocessing_df)
        original_df = create_original_compiled_results(postprocessing_df)
        
        # Save compiled results to restaurant tracker JSON files
        if restaurant_tracker:
            
            # Process each item and add compiled results to tracker
            items_processed_count = 0
            for (fc_id, rest_id, item_id), group in postprocessing_df.groupby(["foodcourt_id", "restaurant_id", "item_id"]):
                # Filter by item_ids if retrain_config specifies it
                if item_ids_filter is not None:
                    item_name_from_group = str(group.get("item_name", pd.Series([""])).iloc[0] if "item_name" in group.columns else "")
                    if not item_name_from_group:
                        item_name_from_group = str(item_id)
                    from src.util.pipeline_utils import matches_item_filter
                    if not matches_item_filter(fc_id, rest_id, item_name_from_group, str(item_id), item_ids_filter):
                        # Item not in item_ids filter, skip it
                        continue
                
                try:
                    # Get capped summary for this item
                    capped_item = None
                    if not capped_df.empty:
                        capped_items = capped_df[
                            (capped_df["foodcourt_id"] == fc_id) &
                            (capped_df["restaurant_id"] == rest_id) &
                            (capped_df["item_id"] == item_id)
                        ]
                        if not capped_items.empty:
                            row = capped_items.iloc[0]
                            capped_item = {
                                "model": str(row.get("model", "")),
                                "postProcessing_used": str(row.get("postProcessing_used", "No")) == "Yes",
                                "abs_avg_deviation": float(row.get("abs_avg_deviation", 0.0)),
                                "abs_avg_accuracy": float(row.get("abs_avg_accuracy", 0.0))
                            }
                    
                    # Get original summary for this item
                    original_item = None
                    if not original_df.empty:
                        original_items = original_df[
                            (original_df["foodcourt_id"] == fc_id) &
                            (original_df["restaurant_id"] == rest_id) &
                            (original_df["item_id"] == item_id)
                        ]
                        if not original_items.empty:
                            # Count unique models
                            item_group = group[group["item_count"] != 0]  # Exclude 0 count days
                            model_names = set()
                            for col in item_group.columns:
                                if "_predicted_count" in col and not col.endswith("_postprocessing"):
                                    model_name = col.replace("_predicted_count", "")
                                    model_names.add(model_name)
                            
                            # Get selected model from capped summary
                            selected_model = capped_item.get("model", "") if capped_item else ""
                            
                            original_item = {
                                "total_days": int(item_group["date"].nunique()) if "date" in item_group.columns else 0,
                                "active_days": int((item_group["item_count"] > 0).sum()) if "item_count" in item_group.columns else 0,
                                "models_count": len(model_names),
                                "models_tried": sorted(list(model_names)),
                                "model_selected": selected_model
                            }
                    
                    # Use defaults if summaries not available
                    if not capped_item:
                        capped_item = {
                            "model": "",
                            "postProcessing_used": False,
                            "abs_avg_deviation": 0.0,
                            "abs_avg_accuracy": 0.0
                        }
                    
                    if not original_item:
                        original_item = {
                            "total_days": 0,
                            "active_days": 0,
                            "models_count": 0,
                            "models_tried": [],
                            "model_selected": ""
                        }
                    
                    # Add compiled results to tracker
                    restaurant_tracker.add_compiled_results(
                        fc_id, rest_id, item_id,
                        capped_item, original_item
                    )
                    items_processed_count += 1
                except Exception as exc:
                    LOGGER.warning(f"Failed to save compiled results for {fc_id}/{rest_id}/{item_id}: {exc}")
        
        # Log summary of what was processed
        items_processed = len(postprocessing_df.groupby(["foodcourt_id", "restaurant_id", "item_id"]))
        LOGGER.info("Processed %d items - compiled results saved to restaurant tracker JSON files", items_processed)
        
        # Log process results
        results = {
            "Items processed": items_processed,
            "CAPPED results": len(capped_df) if not capped_df.empty else 0,
            "ORIGINAL results": len(original_df) if not original_df.empty else 0
        }
        pipeline_logger.log_process_results("compiled_result_generation", results)
        
        LOGGER.info("=" * 80)
        LOGGER.info("Compilation complete")
        LOGGER.info("=" * 80)
        
    except Exception as exc:
        LOGGER.error("Error during compilation: %s", exc, exc_info=True)
        pipeline_logger.log_general_error("compiled_result_generation", f"Error: {exc}", str(exc))
        raise


if __name__ == "__main__":
    main()
