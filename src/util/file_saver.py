"""
File saving utility class that uses output_paths.json configuration.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

LOGGER = logging.getLogger(__name__)


class FileSaver:
    """
    Utility class for saving files to configured output directories.
    Uses output_paths.json to determine where to save different types of files.
    """
    
    def __init__(self, config_path: Optional[Path] = None, pipeline_type: str = "FRID_LEVEL"):
        """
        Initialize FileSaver with configuration.
        
        Args:
            config_path: Path to output_paths.json (default: input_data/output_paths.json)
            pipeline_type: Pipeline type (e.g., "FRID_LEVEL", "FR_LEVEL", "I_LEVEL")
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "input_data" / "output_paths.json"
        
        self.config_path = config_path
        self.pipeline_type = pipeline_type
        self.config = self._load_config()
        self.base_output_dir = Path(__file__).parent.parent.parent / self.config.get("base_output_dir", "output_data")
        self.base_input_dir = Path(__file__).parent.parent.parent / self.config.get("base_input_dir", "input_data")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            LOGGER.debug(f"Loaded output paths configuration from {self.config_path}")
            return config
        except Exception as e:
            LOGGER.error(f"Failed to load output_paths.json: {e}")
            # Return default config
            return {
                "base_output_dir": "output_data",
                "base_input_dir": "input_data",
                "folders": {}
            }
    
    def get_folder_path(self, folder_key: str, create: bool = True) -> Path:
        """
        Get the full path for a folder key.
        
        Args:
            folder_key: Key from output_paths.json (e.g., "enrich_data", "preprocessing")
            create: Whether to create the directory if it doesn't exist
        
        Returns:
            Path object for the folder
        """
        folders = self.config.get("folders", {})
        folder_config = folders.get(folder_key)
        
        if not folder_config:
            LOGGER.warning(f"Folder key '{folder_key}' not found in config, using default")
            # Default to base_output_dir / folder_key
            path = self.base_output_dir / folder_key
        else:
            # Replace {pipeline_type} placeholder
            folder_path_str = folder_config.get("path", folder_key)
            folder_path_str = folder_path_str.replace("{pipeline_type}", self.pipeline_type)
            
            # Determine if it's relative to base_output_dir or base_input_dir
            if folder_path_str.startswith("input_data/"):
                path = Path(folder_path_str)
            else:
                path = self.base_output_dir / folder_path_str
        
        # Create directory if requested
        if create:
            path.mkdir(parents=True, exist_ok=True)
        
        return path
    
    def save_csv(self, df: pd.DataFrame, folder_key: str, filename: str, 
                 index: bool = False, encoding: str = 'utf-8', subdir: Optional[str] = None) -> Path:
        """
        Save a DataFrame as CSV file.
        
        Args:
            df: DataFrame to save
            folder_key: Key from output_paths.json (e.g., "enrich_data")
            filename: Filename (will be saved as .csv)
            index: Whether to include index in CSV
            encoding: File encoding
            subdir: Optional subdirectory within the folder (e.g., foodcourt_id)
        
        Returns:
            Path to saved file
        """
        folder_path = self.get_folder_path(folder_key, create=True)
        
        # Add subdirectory if provided
        if subdir:
            folder_path = folder_path / subdir
            folder_path.mkdir(parents=True, exist_ok=True)
        
        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename = f"{filename}.csv"
        
        file_path = folder_path / filename
        
        try:
            df.to_csv(file_path, index=index, encoding=encoding)
            LOGGER.debug(f"Saved CSV file: {file_path}")
            return file_path
        except Exception as e:
            LOGGER.error(f"Failed to save CSV file {file_path}: {e}")
            raise
    
    def save_excel(self, df: pd.DataFrame, folder_key: str, filename: str,
                   sheet_name: str = "Sheet1", index: bool = False) -> Path:
        """
        Save a DataFrame as Excel file.
        
        Args:
            df: DataFrame to save
            folder_key: Key from output_paths.json
            filename: Filename (will be saved as .xlsx)
            sheet_name: Sheet name for Excel file
            index: Whether to include index
        
        Returns:
            Path to saved file
        """
        folder_path = self.get_folder_path(folder_key, create=True)
        
        # Ensure filename has .xlsx extension
        if not filename.endswith('.xlsx'):
            filename = f"{filename}.xlsx"
        
        file_path = folder_path / filename
        
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=index)
            LOGGER.debug(f"Saved Excel file: {file_path}")
            return file_path
        except Exception as e:
            LOGGER.error(f"Failed to save Excel file {file_path}: {e}")
            raise
    
    def save_pickle(self, obj: Any, folder_key: str, filename: str) -> Path:
        """
        Save an object as pickle file.
        
        Args:
            obj: Object to pickle
            folder_key: Key from output_paths.json (e.g., "trained_models")
            filename: Filename (will be saved as .pkl)
        
        Returns:
            Path to saved file
        """
        import pickle
        
        folder_path = self.get_folder_path(folder_key, create=True)
        
        # Ensure filename has .pkl extension
        if not filename.endswith('.pkl'):
            filename = f"{filename}.pkl"
        
        file_path = folder_path / filename
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(obj, f)
            LOGGER.debug(f"Saved pickle file: {file_path}")
            return file_path
        except Exception as e:
            LOGGER.error(f"Failed to save pickle file {file_path}: {e}")
            raise
    
    def get_file_path(self, folder_key: str, filename: str) -> Path:
        """
        Get the full path for a file without saving it.
        
        Args:
            folder_key: Key from output_paths.json
            filename: Filename
        
        Returns:
            Path object for the file
        """
        folder_path = self.get_folder_path(folder_key, create=False)
        return folder_path / filename
    
    def file_exists(self, folder_key: str, filename: str) -> bool:
        """
        Check if a file exists.
        
        Args:
            folder_key: Key from output_paths.json
            filename: Filename
        
        Returns:
            True if file exists, False otherwise
        """
        file_path = self.get_file_path(folder_key, filename)
        return file_path.exists()
    
    def set_pipeline_type(self, pipeline_type: str):
        """Update the pipeline type."""
        self.pipeline_type = pipeline_type
    
    def save_step_snapshot(self, foodcourt_id: str, restaurant_id: str, label: str, 
                          df: Optional[pd.DataFrame], testing: bool = False) -> Optional[Path]:
        """
        Persist intermediate DataFrame when TESTING flag is enabled.
        
        Args:
            foodcourt_id: Foodcourt ID
            restaurant_id: Restaurant ID
            label: Label for the snapshot (e.g., "enriched", "aggregated")
            df: DataFrame to save (None to skip)
            testing: Whether testing mode is enabled
        
        Returns:
            Path to saved file if successful, None otherwise
        """
        if not testing or df is None or df.empty:
            return None
        
        # Create debug directory structure: output_data/_debug/{foodcourt_id}/
        debug_dir = self.base_output_dir / "_debug" / foodcourt_id
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize label for filename
        safe_label = label.lower().replace(" ", "_").replace("/", "_")
        filename = f"{restaurant_id}_{safe_label}.csv"
        file_path = debug_dir / filename
        
        try:
            df.to_csv(file_path, index=False, encoding="utf-8")
            LOGGER.debug(f"Saved debug snapshot: {file_path}")
            return file_path
        except Exception as exc:
            LOGGER.warning(f"Unable to save debug snapshot {file_path}: {exc}")
            return None
    
    def convert_parquet_to_csv(self, parquet_path: Path, foodcourt_id: str, 
                               restaurant_id: str, output_folder_key: str = "logs") -> Optional[Path]:
        """
        Convert parquet file to CSV and save in specified folder.
        
        Note: If output_folder_key is "logs", this function returns None to avoid creating csv_data directory.
        Only pipeline_logs.xlsx is needed in the logs directory.
        
        Args:
            parquet_path: Path to parquet file
            foodcourt_id: Foodcourt ID
            restaurant_id: Restaurant ID
            output_folder_key: Folder key from output_paths.json (default: "logs")
        
        Returns:
            Path to CSV file if successful, None otherwise (always None for logs folder)
        """
        # Don't create CSV files in logs directory - only pipeline_logs.xlsx is needed
        if output_folder_key == "logs":
            return None
        
        try:
            # Create csv_data directory structure within the specified folder
            folder_path = self.get_folder_path(output_folder_key, create=True)
            csv_data_dir = folder_path / "csv_data" / foodcourt_id
            csv_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Read parquet file
            df = pd.read_parquet(parquet_path)
            
            # Generate CSV filename: {restaurant_id}.csv
            csv_filename = f"{restaurant_id}.csv"
            csv_path = csv_data_dir / csv_filename
            
            # Save as CSV
            df.to_csv(csv_path, index=False, encoding="utf-8")
            LOGGER.debug(f"Converted parquet to CSV: {csv_path}")
            return csv_path
        except Exception as exc:
            LOGGER.warning(f"Failed to convert parquet to CSV for {parquet_path}: {exc}")
            return None

