"""
FRI Name Mapping Utility
Creates and maintains a mapping file for FRI (Foodcourt-Restaurant-Item) IDs to names.
This mapping is created during the enrichment step and used by the dashboard.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

LOGGER = logging.getLogger(__name__)

# Mapping file path - stored in dashboard folder for easy access
MAPPING_FILE = Path(__file__).parent.parent.parent / "dashboard" / "name_mapping.json"


def load_mapping() -> Dict:
    """Load the FRI name mapping from file. Appends/updates data, doesn't delete old data."""
    default_mapping = {
        "foodcourt_id_to_name": {},
        "foodcourt_name_to_id": {},
        "restaurant_name_to_id": {},
        "restaurant_id_to_name": {},
        "foodcourt_id_to_rest_id": {},  # {foodcourt_id: [restaurant_id1, restaurant_id2, ...]}
        "restaurant_id_to_foodcourt_id": {},
        "restaurant_id_to_item": {}  # {restaurant_id: {"item_id_to_name": {}, "item_name_to_id": {}}}
    }
    
    if MAPPING_FILE.exists():
        try:
            with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
                existing_mapping = json.load(f)
                # Merge with defaults to ensure all keys exist
                for key, default_value in default_mapping.items():
                    if key not in existing_mapping:
                        existing_mapping[key] = default_value
                    elif key == "restaurant_id_to_item":
                        # Ensure nested structure exists for each restaurant
                        if not isinstance(existing_mapping[key], dict):
                            existing_mapping[key] = {}
                        # Ensure each restaurant has the nested structure
                        for rest_id, rest_data in existing_mapping[key].items():
                            if not isinstance(rest_data, dict):
                                existing_mapping[key][rest_id] = {"item_id_to_name": {}, "item_name_to_id": {}}
                            if "item_id_to_name" not in existing_mapping[key][rest_id]:
                                existing_mapping[key][rest_id]["item_id_to_name"] = {}
                            if "item_name_to_id" not in existing_mapping[key][rest_id]:
                                existing_mapping[key][rest_id]["item_name_to_id"] = {}
                return existing_mapping
        except Exception as e:
            LOGGER.warning(f"Error loading mapping file: {e}, using default structure")
            return default_mapping
    
    return default_mapping


def save_mapping(mapping: Dict):
    """Save the FRI name mapping to file."""
    try:
        MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
    except Exception as e:
        LOGGER.error(f"Error saving mapping file: {e}")


def update_fri_mapping(
    foodcourt_id: str,
    restaurant_id: str,
    item_id: str,
    foodcourt_name: Optional[str] = None,
    restaurant_name: Optional[str] = None,
    item_name: Optional[str] = None
):
    """
    Update the FRI name mapping with new entries. Appends/updates data, doesn't delete old data.
    
    Creates/updates the following mappings:
    1. foodcourt_id_to_name
    2. foodcourt_name_to_id
    3. restaurant_name_to_id
    4. restaurant_id_to_name
    5. foodcourt_id_to_rest_id (list of restaurant IDs for each foodcourt)
    6. restaurant_id_to_foodcourt_id
    7. restaurant_id_to_item (contains item_id_to_name and item_name_to_id for each restaurant)
    
    Args:
        foodcourt_id: Foodcourt ID
        restaurant_id: Restaurant ID
        item_id: Item ID (FRI ID)
        foodcourt_name: Foodcourt name (optional)
        restaurant_name: Restaurant name (optional)
        item_name: Item name (optional)
    """
    mapping = load_mapping()
    
    # Initialize all required structures if they don't exist
    if "foodcourt_id_to_name" not in mapping:
        mapping["foodcourt_id_to_name"] = {}
    if "foodcourt_name_to_id" not in mapping:
        mapping["foodcourt_name_to_id"] = {}
    if "restaurant_name_to_id" not in mapping:
        mapping["restaurant_name_to_id"] = {}
    if "restaurant_id_to_name" not in mapping:
        mapping["restaurant_id_to_name"] = {}
    if "foodcourt_id_to_rest_id" not in mapping:
        mapping["foodcourt_id_to_rest_id"] = {}
    if "restaurant_id_to_foodcourt_id" not in mapping:
        mapping["restaurant_id_to_foodcourt_id"] = {}
    if "restaurant_id_to_item" not in mapping:
        mapping["restaurant_id_to_item"] = {}
    
    # 1. Update foodcourt_id_to_name and foodcourt_name_to_id
    if foodcourt_id:
        if foodcourt_name:
            # Update foodcourt_id_to_name (append/update, don't delete)
            mapping["foodcourt_id_to_name"][foodcourt_id] = foodcourt_name
            # Update foodcourt_name_to_id (append/update, don't delete)
            mapping["foodcourt_name_to_id"][foodcourt_name] = foodcourt_id
        elif foodcourt_id not in mapping["foodcourt_id_to_name"]:
            # Store ID as fallback if name not available
            mapping["foodcourt_id_to_name"][foodcourt_id] = foodcourt_id
    
    # 2. Update restaurant_id_to_name and restaurant_name_to_id
    if restaurant_id:
        if restaurant_name:
            # Update restaurant_id_to_name (append/update, don't delete)
            mapping["restaurant_id_to_name"][restaurant_id] = restaurant_name
            # Update restaurant_name_to_id (append/update, don't delete)
            # Note: restaurant names can be duplicated across foodcourts, so we store as foodcourt_id:restaurant_name -> restaurant_id
            # But for simplicity, we'll use restaurant_name -> restaurant_id (last one wins, or we could make it a list)
            mapping["restaurant_name_to_id"][restaurant_name] = restaurant_id
        elif restaurant_id not in mapping["restaurant_id_to_name"]:
            # Store ID as fallback if name not available
            mapping["restaurant_id_to_name"][restaurant_id] = restaurant_id
        
        # 6. Update restaurant_id_to_foodcourt_id
        if foodcourt_id:
            mapping["restaurant_id_to_foodcourt_id"][restaurant_id] = foodcourt_id
            
            # 5. Update foodcourt_id_to_rest_id (append restaurant_id to list, don't duplicate)
            if foodcourt_id not in mapping["foodcourt_id_to_rest_id"]:
                mapping["foodcourt_id_to_rest_id"][foodcourt_id] = []
            if restaurant_id not in mapping["foodcourt_id_to_rest_id"][foodcourt_id]:
                mapping["foodcourt_id_to_rest_id"][foodcourt_id].append(restaurant_id)
    
    # 7. Update restaurant_id_to_item (contains item_id_to_name and item_name_to_id for each restaurant)
    if restaurant_id and item_id:
        # Initialize restaurant_id_to_item structure if it doesn't exist
        if restaurant_id not in mapping["restaurant_id_to_item"]:
            mapping["restaurant_id_to_item"][restaurant_id] = {
                "item_id_to_name": {},
                "item_name_to_id": {}
            }
        
        # Ensure nested structure exists
        if "item_id_to_name" not in mapping["restaurant_id_to_item"][restaurant_id]:
            mapping["restaurant_id_to_item"][restaurant_id]["item_id_to_name"] = {}
        if "item_name_to_id" not in mapping["restaurant_id_to_item"][restaurant_id]:
            mapping["restaurant_id_to_item"][restaurant_id]["item_name_to_id"] = {}
        
        # Update item_id_to_name for this restaurant
        if item_name:
            mapping["restaurant_id_to_item"][restaurant_id]["item_id_to_name"][item_id] = item_name
            # Update item_name_to_id for this restaurant (append/update, don't delete)
            mapping["restaurant_id_to_item"][restaurant_id]["item_name_to_id"][item_name] = item_id
        elif item_id not in mapping["restaurant_id_to_item"][restaurant_id]["item_id_to_name"]:
            # Store ID as fallback if name not available
            mapping["restaurant_id_to_item"][restaurant_id]["item_id_to_name"][item_id] = item_id
    
    save_mapping(mapping)

