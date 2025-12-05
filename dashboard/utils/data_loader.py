"""
Utility functions for loading restaurant tracking data.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


def get_tracking_base_path() -> Path:
    """Get the base path for restaurant tracking data."""
    return Path(__file__).parent.parent.parent / "output_data" / "restaurant_tracking"


def get_pipeline_types() -> List[str]:
    """Get available pipeline types."""
    base_path = get_tracking_base_path()
    if not base_path.exists():
        return []
    return [d.name for d in base_path.iterdir() if d.is_dir()]


def get_foodcourt_ids(pipeline_type: str = "FRI_LEVEL") -> List[str]:
    """Get all foodcourt IDs for a given pipeline type."""
    base_path = get_tracking_base_path() / pipeline_type
    if not base_path.exists():
        return []
    return [d.name for d in base_path.iterdir() if d.is_dir()]


def get_restaurant_ids(pipeline_type: str, foodcourt_id: str) -> List[str]:
    """Get all restaurant IDs for a given foodcourt."""
    base_path = get_tracking_base_path() / pipeline_type / foodcourt_id
    if not base_path.exists():
        return []
    return [f.stem for f in base_path.glob("*.json")]


def load_restaurant_tracking(
    pipeline_type: str,
    foodcourt_id: str,
    restaurant_id: str
) -> Optional[Dict]:
    """Load restaurant tracking JSON file."""
    file_path = (
        get_tracking_base_path() / pipeline_type / foodcourt_id / f"{restaurant_id}.json"
    )
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def get_all_items_summary(pipeline_type: str = "FRI_LEVEL") -> pd.DataFrame:
    """
    Get summary of all items across all foodcourts and restaurants.
    Returns a DataFrame with columns: foodcourt_id, restaurant_id, item_id, 
    and various metrics.
    """
    rows = []
    foodcourt_ids = get_foodcourt_ids(pipeline_type)
    
    for foodcourt_id in foodcourt_ids:
        restaurant_ids = get_restaurant_ids(pipeline_type, foodcourt_id)
        
        for restaurant_id in restaurant_ids:
            data = load_restaurant_tracking(pipeline_type, foodcourt_id, restaurant_id)
            if not data:
                continue
            
            metadata = data.get("_metadata", {})
            
            # Iterate through items
            for item_id, item_data in data.items():
                if item_id == "_metadata":
                    continue
                
                row = {
                    "foodcourt_id": foodcourt_id,
                    "restaurant_id": restaurant_id,
                    "item_id": item_id,
                }
                
                # Add compiled results if available
                compiled = item_data.get("compiled_results", {})
                if compiled and not compiled.get("error", False):
                    capped_summary = compiled.get("capped_summary", {})
                    original_summary = compiled.get("original_summary", {})
                    
                    row.update({
                        "model": capped_summary.get("model", "N/A"),
                        "postprocessing_used": capped_summary.get("postProcessing_used", False),
                        "abs_avg_deviation": capped_summary.get("abs_avg_deviation", None),
                        "abs_avg_accuracy": capped_summary.get("abs_avg_accuracy", None),
                        "total_days": original_summary.get("total_days", None),
                        "active_days": original_summary.get("active_days", None),
                        "model_selected": original_summary.get("model_selected", "N/A"),
                    })
                else:
                    row.update({
                        "model": "N/A",
                        "postprocessing_used": False,
                        "abs_avg_deviation": None,
                        "abs_avg_accuracy": None,
                        "total_days": None,
                        "active_days": None,
                        "model_selected": "N/A",
                    })
                
                # Add error status
                row["has_error"] = any([
                    item_data.get("enrich_data", {}).get("error", False),
                    item_data.get("preprocessing", {}).get("error", False),
                    item_data.get("model_generation", {}).get("error", False),
                    item_data.get("postprocessing", {}).get("error", False),
                    item_data.get("compiled_results", {}).get("error", False),
                ])
                
                rows.append(row)
    
    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows)


def get_item_details(
    pipeline_type: str,
    foodcourt_id: str,
    restaurant_id: str,
    item_id: str
) -> Optional[Dict]:
    """Get detailed information for a specific item."""
    data = load_restaurant_tracking(pipeline_type, foodcourt_id, restaurant_id)
    if not data:
        return None
    
    return data.get(item_id)


def get_model_metrics(item_data: Dict) -> Dict:
    """Extract model metrics from item data."""
    model_gen = item_data.get("model_generation", {})
    if model_gen.get("error", False):
        return {}
    
    models = model_gen.get("models", {})
    metrics = {}
    
    for model_name, model_info in models.items():
        metrics[model_name] = {
            "used": model_info.get("used", False),
            "reason": model_info.get("reason", ""),
            "training": model_info.get("training", {}),
            "validation": model_info.get("validation", {}),
        }
    
    return metrics

