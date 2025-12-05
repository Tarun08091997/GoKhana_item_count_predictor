"""
Path and file name utilities for the pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Global pipeline type (set by run_pipeline.py, loaded from pipeline_hyperparameters.json)
_pipeline_type = "FRID_LEVEL"


class PathUtils:
    """
    Utility class for path and file name operations across the pipeline.
    
    This class consolidates all directory and file naming functions to avoid duplication.
    """
    
    @staticmethod
    def get_output_base_dir() -> Path:
        """Get the base output directory."""
        return Path(__file__).parent.parent.parent / "output_data"
    
    @staticmethod
    def get_input_base_dir() -> Path:
        """Get the base input directory."""
        return Path(__file__).parent.parent.parent / "input_data"
    
    @staticmethod
    def get_fr_data_path() -> Path:
        """Get the path to FR_data.json (renamed from fetch_progress.json)."""
        return PathUtils.get_input_base_dir() / "FR_data.json"
    
    @staticmethod
    def get_retrain_path() -> Path:
        """Get the path to retrain.json."""
        return PathUtils.get_input_base_dir() / "retrain.json"
    
    @staticmethod
    def get_pipeline_type() -> str:
        """Get the current pipeline type."""
        global _pipeline_type
        return _pipeline_type
    
    @staticmethod
    def set_pipeline_type(pipeline_type: str):
        """Set the current pipeline type."""
        global _pipeline_type
        _pipeline_type = pipeline_type
    
    @staticmethod
    def get_pipeline_log_dir(step_name: Optional[str] = None) -> Path:
        """
        Get the log directory for the current pipeline.
        
        Args:
            step_name: Optional step name (e.g., 'preprocessing') to create step-specific log dir
        
        Returns:
            Path to the log directory (e.g., output_data/logs/FRID_LEVEL or output_data/logs/FRID_LEVEL/preprocessing)
        """
        base_dir = PathUtils.get_output_base_dir()
        pipeline_type = PathUtils.get_pipeline_type()
        log_dir = base_dir / "logs" / pipeline_type
        
        if step_name:
            log_dir = log_dir / step_name
        
        return log_dir
    
    @staticmethod
    def sanitize_name(name: str) -> str:
        """
        Sanitize a name for use in filenames by replacing special characters with underscores.
        
        This function ensures consistent naming across all pipeline steps:
        - Preserves alphanumeric characters, underscores, and hyphens
        - Replaces all other characters (including spaces) with underscores
        - Strips leading/trailing underscores
        
        Examples:
            "North Indian Veg Thali" -> "North_Indian_Veg_Thali"
            "Item-Name (Special)" -> "Item_Name__Special_"
        """
        sanitized = "".join(ch if ch.isalnum() or ch in ('_', '-') else "_" for ch in str(name))
        return sanitized.strip("_")
    
    @staticmethod
    def get_file_name(foodcourt_id: str, restaurant_id: str, item_name: str, step_type: str, item_id: str = "") -> str:
        """
        Generate file name: {F_id}_{R_id}_{item_id}_{item_name}_{step_type}.csv
        
        This function ensures consistent naming across all pipeline steps.
        Item names are sanitized to handle special characters consistently.
        
        Naming Convention (used by all steps):
        - Format: {foodcourt_id}_{restaurant_id}_{item_id}_{sanitized_item_name}_{step_type}.csv
        - item_id is included if provided (non-empty)
        - item_name is sanitized using sanitize_name() to handle spaces and special characters
        - step_type can be: "enrich_data", "preprocessing", "postprocessing"
        
        IMPORTANT: Always read item_name from CSV files (source of truth), not from filenames.
        Filenames contain sanitized names (spaces->underscores), but CSV files contain original names.
        
        Args:
            foodcourt_id: Foodcourt ID
            restaurant_id: Restaurant ID
            item_name: Item name (will be sanitized for filename)
            step_type: Step type (e.g., "enrich_data", "preprocessing", "postprocessing")
            item_id: Item ID (optional, included in filename if provided)
        
        Returns:
            Filename string with .csv extension
        """
        # Sanitize item_name for filename - this ensures consistent naming
        # Spaces and special chars become underscores, preserving alphanumeric and existing underscores
        sanitized_item = PathUtils.sanitize_name(item_name)
        item_id_str = str(item_id).strip() if item_id else ""
        
        # Include item_id in filename if provided
        if item_id_str:
            return f"{foodcourt_id}_{restaurant_id}_{item_id_str}_{sanitized_item}_{step_type}.csv"
        else:
            # Fallback to old format if item_id not provided (for backward compatibility)
            return f"{foodcourt_id}_{restaurant_id}_{sanitized_item}_{step_type}.csv"
    
    @staticmethod
    def extract_item_name_from_filename(filename: str, foodcourt_id: str, restaurant_id: str, step_type: str) -> Optional[str]:
        """
        Extract item_name from a filename.
        
        This is a helper to reverse the filename generation process.
        Supports both old format (without item_id) and new format (with item_id).
        However, the item_name in the filename is sanitized (spaces->underscores),
        so the actual item_name should be read from the CSV file when possible.
        
        Args:
            filename: Filename like "{fc_id}_{rest_id}_{item_id}_{item_name}_{step_type}.csv" 
                      or old format "{fc_id}_{rest_id}_{item_name}_{step_type}.csv"
                      or model results: "{fc_id}_{rest_id}_{item_id}_{item_name}_{step_name}_{model_name}_{data_type}.csv"
            foodcourt_id: Expected foodcourt_id
            restaurant_id: Expected restaurant_id
            step_type: Expected step_type (e.g., "enrich_data", "preprocessing")
        
        Returns:
            Extracted item_name (sanitized version from filename) or None if format doesn't match
        """
        # Support both .csv and .xlsx (for backward compatibility)
        if not (filename.endswith('.csv') or filename.endswith('.xlsx') or filename.endswith('.xls')):
            return None
        
        base_name = filename.replace('.csv', '').replace('.xlsx', '').replace('.xls', '')
        expected_suffix = f"_{step_type}"
        
        # For model results, check for step_name pattern (e.g., "_model_generation_")
        if "_model_generation_" in base_name:
            # Model result file: {fc_id}_{rest_id}_{item_id}_{item_name}_model_generation_{model_name}_{data_type}
            prefix = base_name.split("_model_generation_")[0]
        elif base_name.endswith(expected_suffix):
            # Regular step file: {fc_id}_{rest_id}_{item_id}_{item_name}_{step_type}
            prefix = base_name[:-len(expected_suffix)]
        else:
            return None
        
        # Split by underscore
        parts = prefix.split('_')
        
        if len(parts) < 3:
            return None
        
        # Check if foodcourt_id and restaurant_id match
        if parts[0] != foodcourt_id or parts[1] != restaurant_id:
            return None
        
        # New format: {fc_id}_{rest_id}_{item_id}_{item_name}_{step_type}
        # Old format: {fc_id}_{rest_id}_{item_name}_{step_type}
        # If we have 4+ parts after removing step_type, assume new format (item_id is 3rd part)
        # If we have 3 parts, assume old format (item_name starts at 3rd part)
        if len(parts) >= 4:
            # New format: item_name starts from 4th part (index 3)
            item_name_sanitized = '_'.join(parts[3:])
        else:
            # Old format: item_name starts from 3rd part (index 2)
            item_name_sanitized = '_'.join(parts[2:])
        
        return item_name_sanitized
    
    @staticmethod
    def get_item_name_from_file(file_path: Path, preferred_columns: Optional[List[str]] = None) -> Optional[str]:
        """
        Extract item_name from a CSV or Excel file by reading the first row.
        This is the source of truth for item names.
        Supports both CSV and Excel files (for backward compatibility).
        
        Args:
            file_path: Path to CSV or Excel file
            preferred_columns: List of column names to check (default: ["itemname", "item_name", "item", "menu_item_name"])
        
        Returns:
            Item name as string, or None if not found
        """
        if preferred_columns is None:
            preferred_columns = ["itemname", "item_name", "item", "menu_item_name"]
        
        try:
            # Try CSV first (new format)
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, nrows=1)
            else:
                # Fallback to Excel (backward compatibility)
                df = pd.read_excel(file_path, sheet_name=0, nrows=1)
            
            for col in preferred_columns:
                if col in df.columns and not df[col].empty and not pd.isna(df[col].iloc[0]):
                    item_name = str(df[col].iloc[0]).strip()
                    if item_name:
                        return item_name
        except Exception:
            pass
        
        return None
    
    @staticmethod
    def get_model_file_name(foodcourt_id: str, restaurant_id: str, item_name: str, model_name: str, item_id: str = "") -> str:
        """Generate model file name: {F_id}_{R_id}_{item_id}_{item_name}_{model_name}.pkl"""
        sanitized_item = PathUtils.sanitize_name(item_name)
        item_id_str = str(item_id).strip() if item_id else ""
        
        # Include item_id in filename if provided
        if item_id_str:
            return f"{foodcourt_id}_{restaurant_id}_{item_id_str}_{sanitized_item}_{model_name}.pkl"
        else:
            # Fallback to old format if item_id not provided (for backward compatibility)
            return f"{foodcourt_id}_{restaurant_id}_{sanitized_item}_{model_name}.pkl"
    
    @staticmethod
    def get_result_file_name(foodcourt_id: str, restaurant_id: str, item_name: str, 
                             step_name: str, model_name: str, item_id: str = "", 
                             data_type: str = "") -> str:
        """
        Generate result file name: {F_id}_{R_id}_{item_id}_{item_name}_{step_name}_{model_name}_{data_type}.csv
        
        Args:
            foodcourt_id: Foodcourt ID
            restaurant_id: Restaurant ID
            item_name: Item name
            step_name: Step name (e.g., "model_generation")
            model_name: Model name (e.g., "XGBoost", "MovingAverage")
            item_id: Item ID (optional)
            data_type: "training" or "validation" (optional, if empty returns base name)
        
        Returns:
            Filename string with .csv extension
        """
        sanitized_item = PathUtils.sanitize_name(item_name)
        item_id_str = str(item_id).strip() if item_id else ""
        data_type_str = f"_{data_type}" if data_type else ""
        
        # Include item_id in filename if provided
        if item_id_str:
            return f"{foodcourt_id}_{restaurant_id}_{item_id_str}_{sanitized_item}_{step_name}_{model_name}{data_type_str}.csv"
        else:
            # Fallback to old format if item_id not provided (for backward compatibility)
            return f"{foodcourt_id}_{restaurant_id}_{sanitized_item}_{step_name}_{model_name}{data_type_str}.csv"


# Create a singleton instance for convenience
_path_utils = PathUtils()

# Convenience functions that delegate to the singleton instance
def get_output_base_dir() -> Path:
    """Get the base output directory."""
    return _path_utils.get_output_base_dir()

def get_input_base_dir() -> Path:
    """Get the base input directory."""
    return _path_utils.get_input_base_dir()

def get_fr_data_path() -> Path:
    """Get the path to FR_data.json."""
    return _path_utils.get_fr_data_path()

def get_retrain_path() -> Path:
    """Get the path to retrain.json."""
    return _path_utils.get_retrain_path()

def get_pipeline_type() -> str:
    """Get the current pipeline type."""
    return _path_utils.get_pipeline_type()

def set_pipeline_type(pipeline_type: str):
    """Set the current pipeline type."""
    _path_utils.set_pipeline_type(pipeline_type)

def get_pipeline_log_dir(step_name: Optional[str] = None) -> Path:
    """Get the log directory for the current pipeline."""
    return _path_utils.get_pipeline_log_dir(step_name)

def sanitize_name(name: str) -> str:
    """Sanitize a name for use in filenames."""
    return _path_utils.sanitize_name(name)

def get_file_name(foodcourt_id: str, restaurant_id: str, item_name: str, step_type: str, item_id: str = "") -> str:
    """Generate file name."""
    return _path_utils.get_file_name(foodcourt_id, restaurant_id, item_name, step_type, item_id)

def extract_item_name_from_filename(filename: str, foodcourt_id: str, restaurant_id: str, step_type: str) -> Optional[str]:
    """Extract item_name from a filename."""
    return _path_utils.extract_item_name_from_filename(filename, foodcourt_id, restaurant_id, step_type)

def get_item_name_from_file(file_path: Path, preferred_columns: Optional[List[str]] = None) -> Optional[str]:
    """Extract item_name from a CSV or Excel file."""
    return _path_utils.get_item_name_from_file(file_path, preferred_columns)

# Backward compatibility alias
get_item_name_from_excel = get_item_name_from_file

def get_model_file_name(foodcourt_id: str, restaurant_id: str, item_name: str, model_name: str, item_id: str = "") -> str:
    """Generate model file name."""
    return _path_utils.get_model_file_name(foodcourt_id, restaurant_id, item_name, model_name, item_id)

def get_result_file_name(foodcourt_id: str, restaurant_id: str, item_name: str, 
                         step_name: str, model_name: str, item_id: str = "", 
                         data_type: str = "") -> str:
    """Generate result file name."""
    return _path_utils.get_result_file_name(foodcourt_id, restaurant_id, item_name, step_name, model_name, item_id, data_type)

