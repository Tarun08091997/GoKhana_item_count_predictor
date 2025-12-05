"""
Restaurant-level tracking utility that uses JSON files to track item processing status.
Each restaurant has a JSON file: {foodcourt_id}/{restaurant_id}.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from src.util.file_saver import FileSaver

LOGGER = logging.getLogger(__name__)


def calculate_metrics_from_df(df: pd.DataFrame, actual_col: str = "actual_count", 
                              predicted_col: str = "predicted_count",
                              error_pct_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate metrics from a DataFrame with actual and predicted counts.
    
    Args:
        df: DataFrame with actual and predicted counts
        actual_col: Column name for actual counts
        predicted_col: Column name for predicted counts
        error_pct_col: Optional column name for error percentage (deprecated, kept for compatibility)
    
    Returns:
        Dict with: abs_avg_deviation, avg_abs_accuracy_pct, abs_avg_deviation_capped,
                   avg_abs_accuracy_pct_capped, total_days, active_days
        
    Note:
        - Ignores days where actual_count = 0 (only calculates on active days)
        - Accuracy formula: acc_pct = 100 - abs(actual - round(predicted)) / actual * 100
        - avg_abs_accuracy_pct = sum of all acc_pct / total active days
    """
    if df.empty:
        return {
            "abs_avg_deviation": 0.0,
            "avg_abs_accuracy_pct": 0.0,
            "avg_abs_accuracy_pct_capped": 0.0,
            "total_days": 0,
            "active_days": 0
        }
    
    # Filter out rows where actual_count is NaN
    df_clean = df.dropna(subset=[actual_col])
    
    if df_clean.empty:
        return {
            "abs_avg_deviation": 0.0,
            "avg_abs_accuracy_pct": 0.0,
            "avg_abs_accuracy_pct_capped": 0.0,
            "total_days": len(df),
            "active_days": 0
        }
    
    total_days = len(df_clean)
    active_days = int((df_clean[actual_col] > 0).sum())
    
    # For postprocessing metrics: IGNORE days where actual_count = 0 (as per requirements)
    # Filter to only active days (where actual_count > 0)
    df_active = df_clean[df_clean[actual_col] > 0].copy()
    
    if df_active.empty:
        # No active days, return zero metrics
        return {
            "abs_avg_deviation": 0.0,
            "avg_abs_accuracy_pct": 0.0,
            "avg_abs_accuracy_pct_capped": 0.0,
            "total_days": total_days,
            "active_days": active_days
        }
    
    # Calculate absolute deviations and accuracy percentages (only on active days)
    # Always use rounded predicted values (rounded to integer) for both calculations
    if predicted_col in df_active.columns:
        actual_vals = df_active[actual_col].values
        predicted_vals = df_active[predicted_col].values
        
        # Round predicted values to integers (always use rounded values)
        predicted_rounded = np.round(predicted_vals).astype(int)
        
        # Calculate absolute deviations using rounded predicted values
        abs_deviations = np.abs(predicted_rounded - actual_vals)
        abs_avg_deviation = float(np.mean(abs_deviations)) if len(abs_deviations) > 0 else 0.0
        
        # Calculate accuracy percentage for each day
        # Formula: 
        #   1. error = abs(actual_value - roundOff(predicted_val))
        #   2. error_pct = error / actualValue * 100
        #   3. acc_pct = 100 - error_pct
        # avg_acc_pct = sum_of_all_acc_pct / total_days (ignoring days where actual_count = 0)
        # Cap each day's accuracy at 0% minimum to prevent negative accuracy
        with np.errstate(divide='ignore', invalid='ignore'):
            # Only calculate for days where actual_count > 0 (already filtered, but double-check)
            # Step 1: Calculate absolute difference (error)
            error = np.abs(actual_vals - predicted_rounded)
            
            # Step 2: Calculate error percentage
            error_pct = np.where(actual_vals > 0,
                                error / actual_vals * 100.0,
                                0.0)
            
            # Step 3: Calculate accuracy percentage (100 - error_pct)
            accuracy_pct = np.where(actual_vals > 0,
                                   100.0 - error_pct,
                                   0.0)
            
            # Cap each day's accuracy at 0% minimum (can't have negative accuracy)
            accuracy_pct = np.maximum(accuracy_pct, 0.0)
        
        # Filter out NaN and Inf values
        valid_accuracies = accuracy_pct[~np.isnan(accuracy_pct) & ~np.isinf(accuracy_pct) & (actual_vals > 0)]
        
        # Calculate average accuracy percentage (already capped at 0% per day)
        avg_abs_accuracy_pct = float(np.mean(valid_accuracies)) if len(valid_accuracies) > 0 else 0.0
        
        # Capped version: ensure accuracy doesn't go below 0% (already capped per day, but ensure average is >= 0)
        avg_abs_accuracy_pct_capped = max(avg_abs_accuracy_pct, 0.0)
    else:
        abs_avg_deviation = 0.0
        avg_abs_accuracy_pct = 0.0
        avg_abs_accuracy_pct_capped = 0.0
    
    return {
        "abs_avg_deviation": abs_avg_deviation,
        "avg_abs_accuracy_pct": avg_abs_accuracy_pct,
        "avg_abs_accuracy_pct_capped": avg_abs_accuracy_pct_capped,
        "total_days": total_days,
        "active_days": active_days
    }


class RestaurantTracker:
    """
    Utility class for tracking item processing status at restaurant level.
    Uses JSON files: {foodcourt_id}/{restaurant_id}.json
    
    Structure (Item-centric):
    {
        "item_id_1": {
            "enrich_data": {
                "error": false,
                "file_name": "fc_id_rest_id_item_id_item_name_enrich_data.csv"
            },
            "preprocessing": {
                "error": false,
                "file_name": "fc_id_rest_id_item_id_item_name_preprocessing.csv"
            },
            "model_generation": {
                "error": false,
                "models": {
                    "XGBoost": {
                        "used": false,
                        "reason": "Insufficient data",
                        "training": {
                            "file_path": "path/to/file.xlsx",
                            "abs_avg_deviation": ...,
                            ...
                        },
                        "validation": {
                            "file_path": "path/to/file.xlsx",
                            ...
                        }
                    }
                }
            },
            "postprocessing": {...},
            "compiled_results": {...}
        }
    }
    """
    
    def __init__(self, file_saver: FileSaver):
        """
        Initialize RestaurantTracker with FileSaver instance.
        
        Args:
            file_saver: FileSaver instance for getting folder paths
        """
        self.file_saver = file_saver
        self.tracking_dir = file_saver.get_folder_path("restaurant_tracking", create=True)
        self._cache: Dict[str, Dict[str, Any]] = {}  # Cache for loaded JSON files
    
    def _get_tracking_file_path(self, foodcourt_id: str, restaurant_id: str) -> Path:
        """Get the path to the tracking JSON file for a restaurant."""
        foodcourt_dir = self.tracking_dir / foodcourt_id
        foodcourt_dir.mkdir(parents=True, exist_ok=True)
        return foodcourt_dir / f"{restaurant_id}.json"
    
    def _load_tracking_file(self, foodcourt_id: str, restaurant_id: str) -> Dict[str, Any]:
        """Load tracking data for a restaurant from JSON file."""
        cache_key = f"{foodcourt_id}_{restaurant_id}"
        
        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        file_path = self._get_tracking_file_path(foodcourt_id, restaurant_id)
        
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Migrate old structure to new structure if needed
                data = self._migrate_to_item_centric(data)
                
                # Ensure metadata exists
                if "_metadata" not in data:
                    data["_metadata"] = {}
                if "foodcourt_id" not in data["_metadata"]:
                    data["_metadata"]["foodcourt_id"] = foodcourt_id
                if "restaurant_id" not in data["_metadata"]:
                    data["_metadata"]["restaurant_id"] = restaurant_id
                
                self._cache[cache_key] = data
                return data
            except Exception as e:
                LOGGER.warning(f"Failed to load tracking file {file_path}: {e}")
                return {"_metadata": {"foodcourt_id": foodcourt_id, "restaurant_id": restaurant_id}}
        else:
            return {"_metadata": {"foodcourt_id": foodcourt_id, "restaurant_id": restaurant_id}}
    
    def _migrate_to_item_centric(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate old structure (step_itemid keys) to new structure (itemid with nested steps).
        
        Old: {"enrich_data_item1": {...}, "preprocessing_item1": {...}}
        New: {"item1": {"enrich_data": {...}, "preprocessing": {...}}}
        """
        if not data:
            return {}
        
        # Check if already in new format (has item_id keys with nested step keys)
        sample_key = next(iter(data.keys()))
        if sample_key and not any(step in sample_key for step in ["enrich_data", "preprocessing", "model_generation", "postprocessing", "compiled_results"]):
            # Looks like new format already
            return data
        
        # Migrate from old format
        migrated = {}
        
        for key, value in data.items():
            # Parse key like "enrich_data_itemid" or "step_itemid"
            parts = key.split("_", 1)
            if len(parts) < 2:
                # Unknown format, keep as is
                migrated[key] = value
                continue
            
            step_name = parts[0]
            item_id = parts[1]
            
            # Initialize item entry if not exists
            if item_id not in migrated:
                migrated[item_id] = {}
            
            # Add step data
            migrated[item_id][step_name] = value
        
        return migrated
    
    def _save_tracking_file(self, foodcourt_id: str, restaurant_id: str, data: Dict[str, Any]):
        """Save tracking data for a restaurant to JSON file."""
        file_path = self._get_tracking_file_path(foodcourt_id, restaurant_id)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Update cache
            cache_key = f"{foodcourt_id}_{restaurant_id}"
            self._cache[cache_key] = data
            
            LOGGER.debug(f"Saved tracking file: {file_path}")
        except Exception as e:
            LOGGER.error(f"Failed to save tracking file {file_path}: {e}")
            raise
    
    def add_success(self, foodcourt_id: str, restaurant_id: str, item_id: str, 
                   file_name: str, step_name: str,
                   foodcourt_name: Optional[str] = None,
                   restaurant_name: Optional[str] = None,
                   item_name: Optional[str] = None):
        """
        Record successful processing of an item.
        
        Args:
            foodcourt_id: Foodcourt ID
            restaurant_id: Restaurant ID
            item_id: Item ID (used as top-level key in JSON)
            file_name: Name of the saved file (can be found using output_paths.json)
            step_name: Step name (e.g., "enrich_data", "preprocessing")
            foodcourt_name: Optional foodcourt name to store in metadata
            restaurant_name: Optional restaurant name to store in metadata
            item_name: Optional item name to store with item data
        """
        if not item_id:
            item_id = "unknown"
        
        data = self._load_tracking_file(foodcourt_id, restaurant_id)
        
        # Update metadata with names
        if "_metadata" not in data:
            data["_metadata"] = {}
        if foodcourt_name:
            data["_metadata"]["foodcourt_name"] = foodcourt_name
        if restaurant_name:
            data["_metadata"]["restaurant_name"] = restaurant_name
        data["_metadata"]["foodcourt_id"] = foodcourt_id
        data["_metadata"]["restaurant_id"] = restaurant_id
        
        # Initialize item entry if not exists
        if item_id not in data:
            data[item_id] = {}
        
        # Store item_name with item data
        if item_name:
            data[item_id]["item_name"] = item_name
        
        # Add step data under item
        data[item_id][step_name] = {
            "error": False,
            "file_name": file_name
        }
        
        self._save_tracking_file(foodcourt_id, restaurant_id, data)
    
    def add_error(self, foodcourt_id: str, restaurant_id: str, item_id: str,
                 error_message: str, step_name: str,
                 foodcourt_name: Optional[str] = None,
                 restaurant_name: Optional[str] = None,
                 item_name: Optional[str] = None):
        """
        Record error during processing of an item.
        
        Args:
            foodcourt_id: Foodcourt ID
            restaurant_id: Restaurant ID
            item_id: Item ID (used as top-level key in JSON)
            error_message: Error message describing what went wrong
            step_name: Step name (e.g., "enrich_data", "preprocessing")
            foodcourt_name: Optional foodcourt name to store in metadata
            restaurant_name: Optional restaurant name to store in metadata
            item_name: Optional item name to store with item data
        """
        if not item_id:
            item_id = "unknown"
        
        data = self._load_tracking_file(foodcourt_id, restaurant_id)
        
        # Update metadata with names
        if "_metadata" not in data:
            data["_metadata"] = {}
        if foodcourt_name:
            data["_metadata"]["foodcourt_name"] = foodcourt_name
        if restaurant_name:
            data["_metadata"]["restaurant_name"] = restaurant_name
        data["_metadata"]["foodcourt_id"] = foodcourt_id
        data["_metadata"]["restaurant_id"] = restaurant_id
        
        # Initialize item entry if not exists
        if item_id not in data:
            data[item_id] = {}
        
        # Store item_name with item data
        if item_name:
            data[item_id]["item_name"] = item_name
        
        # Add step error under item
        data[item_id][step_name] = {
            "error": True,
            "msg": error_message
        }
        
        self._save_tracking_file(foodcourt_id, restaurant_id, data)
    
    def get_names(self, foodcourt_id: str, restaurant_id: str, item_id: Optional[str] = None) -> Dict[str, str]:
        """
        Get foodcourt_name, restaurant_name, and optionally item_name from tracking file.
        
        Args:
            foodcourt_id: Foodcourt ID
            restaurant_id: Restaurant ID
            item_id: Optional item ID to get item_name
        
        Returns:
            Dict with foodcourt_name, restaurant_name, and optionally item_name
        """
        data = self._load_tracking_file(foodcourt_id, restaurant_id)
        metadata = data.get("_metadata", {})
        
        result = {
            "foodcourt_name": metadata.get("foodcourt_name", foodcourt_id),
            "restaurant_name": metadata.get("restaurant_name", restaurant_id)
        }
        
        if item_id:
            item_data = data.get(item_id, {})
            result["item_name"] = item_data.get("item_name", item_id)
        
        return result
    
    def get_item_status(self, foodcourt_id: str, restaurant_id: str, item_id: str, 
                       step_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of an item for a specific step.
        
        Args:
            foodcourt_id: Foodcourt ID
            restaurant_id: Restaurant ID
            item_id: Item ID
            step_name: Step name
        
        Returns:
            Dict with status info or None if not found
        """
        if not item_id:
            item_id = "unknown"
        
        data = self._load_tracking_file(foodcourt_id, restaurant_id)
        item_data = data.get(item_id, {})
        return item_data.get(step_name)
    
    def has_error(self, foodcourt_id: str, restaurant_id: str, item_id: str, 
                 step_name: str) -> bool:
        """
        Check if an item has an error for a specific step.
        
        Args:
            foodcourt_id: Foodcourt ID
            restaurant_id: Restaurant ID
            item_id: Item ID
            step_name: Step name
        
        Returns:
            True if error exists, False otherwise
        """
        status = self.get_item_status(foodcourt_id, restaurant_id, item_id, step_name)
        if status:
            return status.get("error", False) == True or status.get("error", False) == 1
        return False
    
    def get_all_items(self, foodcourt_id: str, restaurant_id: str) -> Dict[str, Any]:
        """
        Get all tracked items for a restaurant.
        
        Args:
            foodcourt_id: Foodcourt ID
            restaurant_id: Restaurant ID
        
        Returns:
            Dict of all items with their status
        """
        return self._load_tracking_file(foodcourt_id, restaurant_id)
    
    def add_model_results(self, foodcourt_id: str, restaurant_id: str, item_id: str,
                         model_name: str, training_metrics: Dict[str, Any],
                         validation_metrics: Dict[str, Any], 
                         training_file_path: Optional[str] = None,
                         validation_file_path: Optional[str] = None,
                         used: bool = False,
                         reason: str = "",
                         step_name: str = "model_generation",
                         foodcourt_name: Optional[str] = None,
                         restaurant_name: Optional[str] = None,
                         item_name: Optional[str] = None):
        """
        Add model results (training and validation metrics) for an item.
        
        Args:
            foodcourt_id: Foodcourt ID
            restaurant_id: Restaurant ID
            item_id: Item ID
            model_name: Model name (e.g., "XGBoost", "MovingAverage")
            training_metrics: Dict with keys: abs_avg_deviation, avg_abs_accuracy_pct,
                            avg_abs_accuracy_pct_capped, total_days, active_days
            validation_metrics: Same structure as training_metrics
            training_file_path: Path to training results file (Excel)
            validation_file_path: Path to validation results file (Excel)
            used: Whether this model was selected/used for predictions
            reason: Reason why model was used or not used
            step_name: Step name (default: "model_generation")
        """
        if not item_id:
            item_id = "unknown"
        
        data = self._load_tracking_file(foodcourt_id, restaurant_id)
        
        # Update metadata with names
        if "_metadata" not in data:
            data["_metadata"] = {}
        if foodcourt_name:
            data["_metadata"]["foodcourt_name"] = foodcourt_name
        if restaurant_name:
            data["_metadata"]["restaurant_name"] = restaurant_name
        data["_metadata"]["foodcourt_id"] = foodcourt_id
        data["_metadata"]["restaurant_id"] = restaurant_id
        
        # Initialize item entry if not exists
        if item_id not in data:
            data[item_id] = {}
        
        # Store item_name with item data
        if item_name:
            data[item_id]["item_name"] = item_name
        
        # Initialize step entry if not exists
        if step_name not in data[item_id]:
            data[item_id][step_name] = {"error": False}
        
        # Initialize models dict if it doesn't exist
        if "models" not in data[item_id][step_name]:
            data[item_id][step_name]["models"] = {}
        
        # Add file paths to metrics
        training_with_path = training_metrics.copy()
        if training_file_path:
            training_with_path["file_path"] = training_file_path
        
        validation_with_path = validation_metrics.copy()
        if validation_file_path:
            validation_with_path["file_path"] = validation_file_path
        
        # Add model results
        data[item_id][step_name]["models"][model_name] = {
            "used": used,
            "reason": reason,
            "training": training_with_path,
            "validation": validation_with_path
        }
        
        self._save_tracking_file(foodcourt_id, restaurant_id, data)
    
    def add_postprocessing_results(self, foodcourt_id: str, restaurant_id: str, item_id: str,
                                  file_name: str, model_metrics: Dict[str, Dict[str, Any]],
                                  foodcourt_name: Optional[str] = None,
                                  restaurant_name: Optional[str] = None,
                                  item_name: Optional[str] = None):
        """
        Add postprocessing results with model metrics.
        
        Args:
            foodcourt_id: Foodcourt ID
            restaurant_id: Restaurant ID
            item_id: Item ID
            file_name: Name of the postprocessing CSV file
            model_metrics: Dict with model names as keys, each containing:
                         {"training": {...}, "validation": {...}}
                         Each with: abs_avg_deviation, avg_abs_accuracy_pct,
                         avg_abs_accuracy_pct_capped, total_days, active_days
        """
        if not item_id:
            item_id = "unknown"
        
        data = self._load_tracking_file(foodcourt_id, restaurant_id)
        
        # Update metadata with names
        if "_metadata" not in data:
            data["_metadata"] = {}
        if foodcourt_name:
            data["_metadata"]["foodcourt_name"] = foodcourt_name
        if restaurant_name:
            data["_metadata"]["restaurant_name"] = restaurant_name
        data["_metadata"]["foodcourt_id"] = foodcourt_id
        data["_metadata"]["restaurant_id"] = restaurant_id
        
        # Initialize item entry if not exists
        if item_id not in data:
            data[item_id] = {}
        
        # Store item_name with item data
        if item_name:
            data[item_id]["item_name"] = item_name
        
        # Add postprocessing data
        data[item_id]["postprocessing"] = {
            "error": False,
            "file_name": file_name,
            "models": model_metrics
        }
        
        self._save_tracking_file(foodcourt_id, restaurant_id, data)
    
    def add_compiled_results(self, foodcourt_id: str, restaurant_id: str, item_id: str,
                            capped_summary: Dict[str, Any],
                            original_summary: Dict[str, Any],
                            foodcourt_name: Optional[str] = None,
                            restaurant_name: Optional[str] = None,
                            item_name: Optional[str] = None):
        """
        Add compiled results summary for an item.
        
        Args:
            foodcourt_id: Foodcourt ID
            restaurant_id: Restaurant ID
            item_id: Item ID
            capped_summary: Dict with model, postProcessing_used, abs_avg_deviation, abs_avg_accuracy
            original_summary: Dict with total_days, active_days, models_count, models_tried, model_selected
        """
        if not item_id:
            item_id = "unknown"
        
        data = self._load_tracking_file(foodcourt_id, restaurant_id)
        
        # Update metadata with names
        if "_metadata" not in data:
            data["_metadata"] = {}
        if foodcourt_name:
            data["_metadata"]["foodcourt_name"] = foodcourt_name
        if restaurant_name:
            data["_metadata"]["restaurant_name"] = restaurant_name
        data["_metadata"]["foodcourt_id"] = foodcourt_id
        data["_metadata"]["restaurant_id"] = restaurant_id
        
        # Initialize item entry if not exists
        if item_id not in data:
            data[item_id] = {}
        
        # Store item_name with item data
        if item_name:
            data[item_id]["item_name"] = item_name
        
        # Add compiled results (no file_name since we don't create CSV files)
        data[item_id]["compiled_results"] = {
            "error": False,
            "capped_summary": capped_summary,
            "original_summary": original_summary
        }
        
        self._save_tracking_file(foodcourt_id, restaurant_id, data)

