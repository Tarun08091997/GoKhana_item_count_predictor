"""
Mapping Manager for loading FRI name mappings.
The mapping file is now created by the enrichment step (step2_data_enrichment.py),
so this module only loads the existing mapping file.
"""

import json
from pathlib import Path
from typing import Dict

# Mapping file path (created by enrichment step)
MAPPING_FILE = Path(__file__).parent.parent / "name_mapping.json"


def load_mapping() -> Dict:
    """Load the FRI name mapping from file (created by enrichment step)."""
    if MAPPING_FILE.exists():
        try:
            with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading mapping file: {e}")
    
    return {
        "foodcourts": {},
        "restaurants": {},
        "items": {},
        "foodcourt_id_to_name": {},
        "restaurant_id_to_name": {},
        "item_id_to_name": {},
        # Bidirectional mappings for easy navigation
        "foodcourt_id_to_restaurant_ids": {},
        "restaurant_id_to_foodcourt_id": {},
        "restaurant_id_to_item_ids": {},
        "item_id_to_restaurant_id": {},
        "foodcourt_id_to_item_ids": {},
        "item_id_to_foodcourt_id": {},
    }
