"""
Shared utility module for loading and using the item filter CSV across all pipeline stages.
"""

import logging
from pathlib import Path
from typing import Optional, Set, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_item_filter(input_dir: Path) -> Optional[Set[Tuple[str, str, str]]]:
    """
    Load the filter CSV file that specifies which items to process.
    Returns a set of tuples (foodcourt_id, restaurant_id, item_name) to process.
    If file doesn't exist, returns None (process all items).
    
    Args:
        input_dir: Path to input_data directory
        
    Returns:
        Set of tuples (foodcourt_id, restaurant_id, item_name) or None
    """
    filter_csv_path = input_dir / "train_model_for_items.csv"
    
    if not filter_csv_path.exists():
        LOGGER.info("No filter CSV found at %s - processing all items", filter_csv_path)
        return None
    
    try:
        filter_df = pd.read_csv(filter_csv_path)
        
        # Handle different possible column name variations
        foodcourt_col = None
        restaurant_col = None
        item_col = None
        
        # Try to find foodcourt column
        for col in filter_df.columns:
            col_lower = col.lower().strip()
            if col_lower in ["food_court_id", "foodcourt_id", "foodcourtid", "foodcourt", "fc_id"]:
                foodcourt_col = col
                break
        
        # Try to find restaurant column
        for col in filter_df.columns:
            col_lower = col.lower().strip()
            if col_lower in ["restaurant_id", "restaurantid", "restaurant", "rest_id"]:
                restaurant_col = col
                break
        
        # Try to find item name column
        for col in filter_df.columns:
            col_lower = col.lower().strip()
            if col_lower in ["item_name", "itemname", "item", "menuitemname", "menu_item_name"]:
                item_col = col
                break
        
        if not foodcourt_col or not restaurant_col or not item_col:
            missing = []
            if not foodcourt_col:
                missing.append("food_court_id/foodcourt_id")
            if not restaurant_col:
                missing.append("restaurant_id")
            if not item_col:
                missing.append("item_name")
            LOGGER.error(
                "Filter CSV missing required columns: %s. Found columns: %s",
                ", ".join(missing),
                list(filter_df.columns)
            )
            return None
        
        # Create set of tuples (foodcourt_id, restaurant_id, item_name)
        # Normalize to strings and strip whitespace for matching
        filter_set = set()
        for _, row in filter_df.iterrows():
            fc_id = str(row[foodcourt_col]).strip() if pd.notna(row[foodcourt_col]) else ""
            rest_id = str(row[restaurant_col]).strip() if pd.notna(row[restaurant_col]) else ""
            item_name = str(row[item_col]).strip() if pd.notna(row[item_col]) else ""
            
            if fc_id and rest_id and item_name:
                filter_set.add((fc_id, rest_id, item_name))
        
        LOGGER.info("Loaded filter CSV with %d items to process", len(filter_set))
        return filter_set
    
    except Exception as exc:
        LOGGER.error("Failed to load filter CSV %s: %s", filter_csv_path, exc)
        return None


def should_process_item(
    foodcourt_id: str,
    restaurant_id: str,
    item_name: str,
    filter_set: Optional[Set[Tuple[str, str, str]]]
) -> bool:
    """
    Check if an item should be processed based on the filter set.
    If filter_set is None, process all items.
    
    Args:
        foodcourt_id: Foodcourt ID
        restaurant_id: Restaurant ID
        item_name: Item name
        filter_set: Set of tuples (foodcourt_id, restaurant_id, item_name) or None
        
    Returns:
        True if item should be processed, False otherwise
    """
    if filter_set is None:
        return True
    
    # Normalize for matching (strip and convert to string, case-insensitive for item name)
    fc_id = str(foodcourt_id).strip()
    rest_id = str(restaurant_id).strip()
    item = str(item_name).strip()
    
    # Check exact match first
    if (fc_id, rest_id, item) in filter_set:
        return True
    
    # Also check case-insensitive match for item name
    for filter_fc, filter_rest, filter_item in filter_set:
        if (fc_id == filter_fc and 
            rest_id == filter_rest and 
            item.lower() == filter_item.lower()):
            return True
    
    return False


def normalize_item_name_for_matching(item_name: str) -> str:
    """
    Normalize item name for matching (lowercase, strip whitespace).
    
    Args:
        item_name: Item name to normalize
        
    Returns:
        Normalized item name
    """
    return str(item_name).strip().lower()


def get_restaurant_foodcourt_pairs(filter_set: Optional[Set[Tuple[str, str, str]]]) -> Optional[Set[Tuple[str, str]]]:
    """
    Extract unique (foodcourt_id, restaurant_id) pairs from the filter set.
    
    Args:
        filter_set: Set of tuples (foodcourt_id, restaurant_id, item_name) or None
        
    Returns:
        Set of tuples (foodcourt_id, restaurant_id) or None
    """
    if filter_set is None:
        return None
    
    pairs = set()
    for fc_id, rest_id, _ in filter_set:
        pairs.add((fc_id, rest_id))
    
    return pairs


def should_process_restaurant(
    foodcourt_id: str,
    restaurant_id: str,
    filter_set: Optional[Set[Tuple[str, str, str]]]
) -> bool:
    """
    Check if a restaurant/foodcourt combination should be processed.
    If filter_set is None, process all restaurants.
    
    Args:
        foodcourt_id: Foodcourt ID
        restaurant_id: Restaurant ID
        filter_set: Set of tuples (foodcourt_id, restaurant_id, item_name) or None
        
    Returns:
        True if restaurant should be processed, False otherwise
    """
    if filter_set is None:
        return True
    
    # Normalize for matching
    fc_id = str(foodcourt_id).strip()
    rest_id = str(restaurant_id).strip()
    
    # Check if any item in this restaurant/foodcourt is in the filter
    for filter_fc, filter_rest, _ in filter_set:
        if fc_id == filter_fc and rest_id == filter_rest:
            return True
    
    return False

