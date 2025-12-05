"""
Utility to resolve foodcourt and restaurant names from cached mapping file.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from pymongo import MongoClient
from bson import ObjectId

from dashboard.utils.mapping_manager import load_mapping

# Global mapping cache (loaded once)
_mapping_cache: Optional[Dict] = None


def _get_mapping() -> Dict:
    """Get the mapping cache, loading it if necessary."""
    global _mapping_cache
    if _mapping_cache is None:
        _mapping_cache = load_mapping()
    return _mapping_cache


def get_foodcourt_name(foodcourt_id: str) -> str:
    """Get foodcourt name from cached mapping."""
    mapping = _get_mapping()
    return mapping.get("foodcourt_id_to_name", {}).get(foodcourt_id, foodcourt_id)


def get_restaurant_name(restaurant_id: str) -> str:
    """Get restaurant name from cached mapping."""
    mapping = _get_mapping()
    return mapping.get("restaurant_id_to_name", {}).get(restaurant_id, restaurant_id)


def get_item_name(item_id: str) -> str:
    """Get item name from cached mapping."""
    mapping = _get_mapping()
    return mapping.get("item_id_to_name", {}).get(item_id, item_id)


def reload_mapping():
    """Reload the mapping cache (useful after refresh)."""
    global _mapping_cache
    _mapping_cache = None
    _get_mapping()

