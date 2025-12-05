"""
Script to compile validation errors from all models across all folders in Model Training directory.

For each item, it:
1. Looks for XGBoost validation file (if exists, uses it and ignores others)
2. Otherwise, compares Moving Average (decay) and Weekly Moving Average (weekday) 
   and picks the one with best (lowest) avg_abs_deviation
3. Filters out rows where actual_count = 0
4. Calculates pct_accuracy (avg of abs error_pct) and avg_abs_deviation (avg of abs error)
5. Outputs CSV with: foodcourt_id, name, restaurant_id, item_name, pct_accuracy, avg_abs_deviation
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
from pymongo import MongoClient
from bson import ObjectId
from src.util.config_parser import ConfigManger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Cache for MongoDB names to avoid repeated queries
_name_cache = {}

# Cache for item name mapping
_item_name_mapping = None

# Cache for training stats per (foodcourt, restaurant)
_training_stats_cache: Dict[Tuple[str, str], Dict[str, Tuple[int, int]]] = {}


def sanitize_name(name: str) -> str:
    """
    Sanitize item name to match folder naming convention.
    Same function as in models.py - converts to lowercase and replaces
    non-alphanumeric characters with underscores.
    """
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    cleaned = cleaned.strip("_")
    return cleaned or "item"


def load_item_name_mapping() -> Dict[Tuple[str, str, str], str]:
    """
    Load item name mapping from train_model_for_items.csv.
    Returns a dictionary mapping (foodcourt_id, restaurant_id, sanitized_item_name) -> original_item_name
    """
    global _item_name_mapping
    
    if _item_name_mapping is not None:
        return _item_name_mapping
    
    _item_name_mapping = {}
    
    try:
        csv_path = os.path.join("input_data", "train_model_for_items.csv")
        if not os.path.exists(csv_path):
            logging.warning(f"train_model_for_items.csv not found at {csv_path}")
            return _item_name_mapping
        
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_cols = ['foodcourtid', 'restaurant', 'menuitemname']
        if not all(col in df.columns for col in required_cols):
            logging.warning(f"Missing required columns in {csv_path}")
            return _item_name_mapping
        
        # Create mapping
        for _, row in df.iterrows():
            foodcourt_id = str(row['foodcourtid']).strip()
            restaurant_id = str(row['restaurant']).strip()
            original_item_name = str(row['menuitemname']).strip()
            
            # Sanitize the original name to match folder naming
            sanitized_name = sanitize_name(original_item_name)
            
            # Create key: (foodcourt_id, restaurant_id, sanitized_name)
            key = (foodcourt_id, restaurant_id, sanitized_name)
            _item_name_mapping[key] = original_item_name
        
        logging.info(f"Loaded {len(_item_name_mapping)} item name mappings from train_model_for_items.csv")
        
    except Exception as e:
        logging.error(f"Error loading item name mapping: {e}")
    
    return _item_name_mapping


def get_original_item_name(foodcourt_id: str, restaurant_id: str, sanitized_item_name: str) -> str:
    """
    Get the original item name from the mapping.
    Returns the original name if found, otherwise returns the sanitized name.
    """
    mapping = load_item_name_mapping()
    key = (foodcourt_id, restaurant_id, sanitized_item_name)
    return mapping.get(key, sanitized_item_name)


def load_training_stats(foodcourt_id: str, restaurant_id: str) -> Dict[str, Tuple[int, int]]:
    """
    Load training statistics (total days and non-zero days) for all items
    within a restaurant. Returns mapping: item_slug -> (total_days, non_zero_days)
    """
    cache_key = (foodcourt_id, restaurant_id)
    if cache_key in _training_stats_cache:
        return _training_stats_cache[cache_key]

    stats: Dict[str, Tuple[int, int]] = {}

    csv_path = Path("input_data") / "Model Training" / "preprocessed_data" / foodcourt_id / f"{restaurant_id}.csv"
    if not csv_path.exists():
        logging.warning(f"Preprocessed training data not found for {foodcourt_id}/{restaurant_id}: {csv_path}")
        _training_stats_cache[cache_key] = stats
        return stats

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            _training_stats_cache[cache_key] = stats
            return stats

        # Determine item name column
        name_col = None
        for candidate in ["itemname", "menuitemname", "item_name"]:
            if candidate in df.columns:
                name_col = candidate
                break
        if name_col is None:
            logging.warning(f"No item name column found in training data {csv_path}")
            _training_stats_cache[cache_key] = stats
            return stats

        # Determine count column
        count_col = None
        for candidate in ["count", "total_count"]:
            if candidate in df.columns:
                count_col = candidate
                break
        if count_col is None:
            logging.warning(f"No count column found in training data {csv_path}")
            _training_stats_cache[cache_key] = stats
            return stats

        df = df[[name_col, count_col]].copy()
        df.rename(columns={name_col: "itemname", count_col: "count"}, inplace=True)

        df["item_slug"] = df["itemname"].astype(str).apply(sanitize_name)
        grouped = df.groupby("item_slug")["count"].agg(
            total_days="count",
            non_zero_days=lambda s: int((s > 0).sum())
        ).reset_index()

        stats = {
            row["item_slug"]: (int(row["total_days"]), int(row["non_zero_days"]))
            for _, row in grouped.iterrows()
        }
        _training_stats_cache[cache_key] = stats
        return stats
    except Exception as exc:
        logging.error(f"Failed to load training stats from {csv_path}: {exc}")
        _training_stats_cache[cache_key] = stats
        return stats


def get_training_stats(foodcourt_id: str, restaurant_id: str, sanitized_item_name: str) -> Tuple[int, int]:
    """
    Return (total_training_days, total_non_zero_training_days) for the item.
    """
    stats_mapping = load_training_stats(foodcourt_id, restaurant_id)
    return stats_mapping.get(sanitized_item_name, (0, 0))

def get_mongo_names(foodcourt_id: str, restaurant_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch food court name and restaurant name from MongoDB.
    Uses caching to avoid repeated queries.
    Returns (foodcourt_name, restaurant_name) or (None, None) if not found.
    """
    # Check cache first
    cache_key_fc = f"fc_{foodcourt_id}"
    cache_key_rest = f"rest_{restaurant_id}" if restaurant_id else None
    
    foodcourt_name = _name_cache.get(cache_key_fc)
    restaurant_name = _name_cache.get(cache_key_rest) if cache_key_rest else None
    
    # If both are cached, return them
    if foodcourt_name is not None and (not restaurant_id or restaurant_name is not None):
        return foodcourt_name, restaurant_name
    
    try:
        config = ConfigManger()
        mongo_cfg = config.get_config("local_mongodb")
        
        client = MongoClient(mongo_cfg["LOCAL_MONGO_URI"])
        db = client[mongo_cfg["LOCAL_MONGO_DB"]]
        
        # Get food court name if not cached
        if foodcourt_name is None:
            try:
                foodcourt_objid = ObjectId(foodcourt_id) if len(foodcourt_id) == 24 else foodcourt_id
                fc_record = db[mongo_cfg["FOOD_COURT_COLL"].strip()].find_one({"_id": foodcourt_objid})
                if fc_record and "data" in fc_record and "name" in fc_record["data"]:
                    foodcourt_name = fc_record["data"]["name"]
                    _name_cache[cache_key_fc] = foodcourt_name
                else:
                    _name_cache[cache_key_fc] = None
            except Exception as e:
                logging.debug(f"Could not fetch food court name for {foodcourt_id}: {e}")
                _name_cache[cache_key_fc] = None
        
        # Get restaurant name if not cached and restaurant_id provided
        if restaurant_id and restaurant_name is None:
            try:
                restaurant_objid = ObjectId(restaurant_id) if len(restaurant_id) == 24 else restaurant_id
                rest_record = db["restaurant_data"].find_one({"_id": restaurant_objid})
                if rest_record and "data" in rest_record and "name" in rest_record["data"]:
                    restaurant_name = rest_record["data"]["name"]
                    _name_cache[cache_key_rest] = restaurant_name
                else:
                    _name_cache[cache_key_rest] = None
            except Exception as e:
                logging.debug(f"Could not fetch restaurant name for {restaurant_id}: {e}")
                _name_cache[cache_key_rest] = None
        
        client.close()
        return foodcourt_name, restaurant_name
    except Exception as e:
        logging.warning(f"Failed to connect to MongoDB for names lookup: {e}")
        return None, None


def process_validation_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Process a validation CSV file:
    - Round down actual_count values (floor)
    - Filter out rows where actual_count = 0
    - Return DataFrame with filtered data
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_cols = ['actual_count', 'error', 'error_pct']
        if not all(col in df.columns for col in required_cols):
            logging.warning(f"Missing required columns in {file_path}")
            return None
        
        # Round down actual_count and predicted_count values (floor)
        df['actual_count'] = df['actual_count'].apply(lambda x: int(np.floor(x)) if pd.notna(x) else x)
        if 'predicted_count' in df.columns:
            df['predicted_count'] = df['predicted_count'].apply(lambda x: int(np.floor(x)) if pd.notna(x) else x)
        
        # Filter out rows where actual_count = 0
        df_filtered = df[df['actual_count'] != 0].copy()
        
        if df_filtered.empty:
            logging.debug(f"No non-zero actual_count rows in {file_path}")
            return None
        
        return df_filtered
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None


def calculate_metrics(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate pct_accuracy and avg_abs_deviation from filtered DataFrame.
    - Cap absolute percentage errors to 100% (if abs(error_pct) > 100, cap to 100)
    Returns (pct_accuracy, avg_abs_deviation)
    """
    if df.empty:
        return 0.0, 0.0
    
    # Cap absolute percentage errors to 100%
    # If error_pct is -120, abs becomes 120, then cap to 100
    # If error_pct is 150, abs becomes 150, then cap to 100
    df_capped = df.copy()
    df_capped['error_pct_capped'] = df_capped['error_pct'].abs().clip(upper=100.0)
    
    # pct_accuracy = average of capped absolute values of error_pct
    pct_accuracy = 100 - df_capped['error_pct_capped'].mean()
    
    # avg_abs_deviation = average of absolute values of error
    avg_abs_deviation = df['error'].abs().mean()
    
    return pct_accuracy, avg_abs_deviation


def find_validation_files(item_folder: str, item_name: str) -> Dict[str, str]:
    """
    Find all validation files for an item.
    Returns dict with keys: 'xgboost', 'decay', 'weekday' and file paths as values.
    """
    files = {}
    
    # XGBoost validation file
    xgb_file = os.path.join(item_folder, f"{item_name}_validation_results.csv")
    if os.path.exists(xgb_file):
        files['xgboost'] = xgb_file
    
    # Moving average (decay) validation file
    decay_file = os.path.join(item_folder, f"{item_name}_validation_decay_results.csv")
    if os.path.exists(decay_file):
        files['decay'] = decay_file
    
    # Weekly moving average (weekday) validation file
    weekday_file = os.path.join(item_folder, f"{item_name}_validation_weekday_results.csv")
    if os.path.exists(weekday_file):
        files['weekday'] = weekday_file
    
    return files


def select_best_file(files: Dict[str, str], item_folder: str) -> Optional[Tuple[str, str]]:
    """
    Select the best validation file according to priority:
    1. If XGBoost exists, use it and ignore others
    2. Otherwise, compare decay and weekday, pick the one with best (lowest) avg_abs_deviation
    """
    # Priority 1: XGBoost
    if 'xgboost' in files:
        return files['xgboost'], 'xgboost'
    
    # Priority 2: Compare decay vs weekday
    if 'decay' in files and 'weekday' in files:
        # Process both and compare
        df_decay = process_validation_file(files['decay'])
        df_weekday = process_validation_file(files['weekday'])
        
        if df_decay is None and df_weekday is None:
            return None
        if df_decay is None:
            return files['weekday'], 'weekly_moving_average'
        if df_weekday is None:
            return files['decay'], 'moving_average_decay'
        
        # Compare avg_abs_deviation (lower is better)
        _, decay_dev = calculate_metrics(df_decay)
        _, weekday_dev = calculate_metrics(df_weekday)
        
        if decay_dev <= weekday_dev:
            return files['decay'], 'moving_average_decay'
        return files['weekday'], 'weekly_moving_average'
    
    # If only one exists, return it
    if 'decay' in files:
        return files['decay'], 'moving_average_decay'
    if 'weekday' in files:
        return files['weekday'], 'weekly_moving_average'
    
    return None


def process_model_training_folder(base_path: str) -> pd.DataFrame:
    """
    Process all folders in Model Training directory and compile validation errors.
    Returns DataFrame with results.
    """
    results = []
    
    base_path = Path(base_path)
    
    # Process each subfolder (XGBosst, preprocessed_data, enriched_data, food_court_data)
    for folder_name in ['XGBosst']:
        folder_path = base_path / folder_name
        
        if not folder_path.exists():
            logging.info(f"Folder {folder_name} does not exist, skipping...")
            continue
        
        logging.info(f"Processing folder: {folder_name}")
        
        # Iterate through food court folders
        for fc_folder in folder_path.iterdir():
            if not fc_folder.is_dir():
                continue
            
            foodcourt_id = fc_folder.name
            logging.info(f"  Processing food court: {foodcourt_id}")
            
            # Iterate through restaurant folders
            for rest_folder in fc_folder.iterdir():
                if not rest_folder.is_dir():
                    continue
                
                restaurant_id = rest_folder.name
                logging.debug(f"    Processing restaurant: {restaurant_id}")
                
                # Get names from MongoDB (cached)
                foodcourt_name, restaurant_name = get_mongo_names(foodcourt_id, restaurant_id)
                
                # Iterate through item folders
                for item_folder in rest_folder.iterdir():
                    if not item_folder.is_dir():
                        continue
                    
                    sanitized_item_name = item_folder.name
                    
                    # Get original item name from mapping
                    original_item_name = get_original_item_name(foodcourt_id, restaurant_id, sanitized_item_name)
                    
                    # Find validation files (use sanitized name for file lookup)
                    validation_files = find_validation_files(str(item_folder), sanitized_item_name)
                    
                    if not validation_files:
                        logging.debug(f"      No validation files found for {sanitized_item_name}")
                        continue
                    
                    # Select best file
                    selected = select_best_file(validation_files, str(item_folder))
                    
                    if selected is None:
                        logging.debug(f"      Could not select file for {sanitized_item_name}")
                        continue
                    
                    selected_file, model_type = selected
                    
                    # Process the selected file
                    df = process_validation_file(selected_file)
                    
                    if df is None or df.empty:
                        logging.debug(f"      No valid data for {sanitized_item_name}")
                        continue
                    
                    # Calculate metrics
                    pct_accuracy, avg_abs_deviation = calculate_metrics(df)
                    total_days, non_zero_days = get_training_stats(foodcourt_id, restaurant_id, sanitized_item_name)
                    
                    # Get absolute folder path
                    folder_path = os.path.abspath(str(item_folder))
                    
                    # Create Excel hyperlink
                    path_normalized = folder_path.replace('\\', '/')
                    if path_normalized.startswith('//'):
                        path_uri = f"file:{path_normalized}"
                    else:
                        if not path_normalized.startswith('/'):
                            path_uri = f"file:///{path_normalized}"
                        else:
                            path_uri = f"file://{path_normalized}"
                    # URL encode spaces and special characters
                    path_uri = path_uri.replace(' ', '%20')
                    folder_link = f'=HYPERLINK("{path_uri}", "Open Folder")'
                    
                    # Add to results (use original item name)
                    results.append({
                        'foodcourt_id': foodcourt_id,
                        'name': foodcourt_name if foodcourt_name else '',
                        'restaurant_id': restaurant_id,
                        'item_name': original_item_name,
                        'pct_accuracy': pct_accuracy,
                        'avg_abs_deviation': avg_abs_deviation,
                        'total_training_days': total_days,
                        'total_non_zero_training_days': non_zero_days,
                        'model_type': model_type,
                        'item_results_folder': folder_path,
                        'item_results_folder_link': folder_link
                    })
                    
                    logging.info(f"      Processed {original_item_name}: pct_accuracy={pct_accuracy:.2f}, avg_abs_deviation={avg_abs_deviation:.2f}")
    
    return pd.DataFrame(results)


def main():
    """Main function to compile validation errors."""
    # Get base path
    base_path = os.path.join("input_data", "Model Training")
    
    if not os.path.exists(base_path):
        logging.error(f"Model Training directory not found at {base_path}")
        return
    
    logging.info("Starting validation error compilation...")
    
    # Process all folders
    results_df = process_model_training_folder(base_path)
    
    if results_df.empty:
        logging.warning("No results found!")
        return
    
    # Save to CSV
    output_file = os.path.join(base_path, "val_error_compilation.csv")
    results_df.to_csv(output_file, index=False)
    
    logging.info(f"Compilation complete! Results saved to {output_file}")
    logging.info(f"Total items processed: {len(results_df)}")
    
    # Print summary
    print("\nSummary:")
    print(f"Total items: {len(results_df)}")
    print(f"Average pct_accuracy: {results_df['pct_accuracy'].mean():.2f}")
    print(f"Average avg_abs_deviation: {results_df['avg_abs_deviation'].mean():.2f}")


if __name__ == "__main__":
    main()

