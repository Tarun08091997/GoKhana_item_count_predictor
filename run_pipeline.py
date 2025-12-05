"""
Run the full GoKhana data-prep + modeling pipeline in sequence.

Pipeline Steps:
1. data_fetch - Fetch raw restaurant orders from MongoDB
2. enrich_data - Enrich restaurant orders with metadata, weather, holidays
3. preprocessing - Prepare data for modeling
4. model_generation - Train models (XGBoost, Moving Average, etc.)
5. postprocessing - Day-of-week redistribution for predictions
6. compiled_result_generation - Compile validation & training errors
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd

from src.util.pipeline_utils import (
    load_retrain_config, should_force_retrain, PipelineLogger,
    set_pipeline_start_time
)
from src.util.checkpoint_manager import get_checkpoint_manager
from src.util.path_utils import (
    get_output_base_dir, get_input_base_dir, get_fr_data_path, get_retrain_path,
    set_pipeline_type, get_pipeline_type
)
from src.util.progress_bar import ProgressBar
from src.util.file_saver import FileSaver
from src.util.restaurant_tracker import RestaurantTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("pipeline")

ROOT = Path(__file__).resolve().parent

# Input paths
INPUT_DIR = get_input_base_dir()
FR_DATA_PATH = get_fr_data_path()
RETRAIN_PATH = get_retrain_path()
FETCHED_DATA_DIR = INPUT_DIR / "fetched_data"

# Pipeline configuration - load from pipeline_hyperparameters.json
from src.util.pipeline_utils import get_active_pipeline_type, load_pipeline_config
pipeline_config = load_pipeline_config()
PIPELINE_TYPE = get_active_pipeline_type()  # Get active pipeline type from config

# Output paths - all outputs go to pipeline-specific subfolder
OUTPUT_DIR = get_output_base_dir()
# Note: data_fetch step is commented out, so DATA_FETCH_DIR is not needed
ENRICH_DATA_DIR = OUTPUT_DIR / "enrich_data" / PIPELINE_TYPE
PREPROCESSING_DIR = OUTPUT_DIR / "preprocessing" / PIPELINE_TYPE
POSTPROCESSING_DIR = OUTPUT_DIR / "postprocessing" / PIPELINE_TYPE
# Removed COMPILED_RESULTS_DIR - compiled results are saved to restaurant_tracker JSON files only
TRAINED_MODEL_DIR = OUTPUT_DIR / "trainedModel" / PIPELINE_TYPE
TRAINED_MODEL_MODELS_DIR = TRAINED_MODEL_DIR / "models"
TRAINED_MODEL_RESULTS_DIR = TRAINED_MODEL_DIR / "results"

# Set pipeline type before initializing logger
from src.util.pipeline_utils import get_pipeline_logger
from src.util.path_utils import set_pipeline_type
set_pipeline_type(PIPELINE_TYPE)

# Initialize logger (global)
# This will be accessible through pipeline_utils functions
pipeline_logger = get_pipeline_logger()

# Initialize utility instances to be passed to steps
file_saver = FileSaver(pipeline_type=PIPELINE_TYPE)
restaurant_tracker = RestaurantTracker(file_saver)
# ProgressBar instances will be created per step as needed


def _load_fr_data() -> dict:
    """Load FR_data.json (renamed from fetch_progress.json)."""
    if not FR_DATA_PATH.exists():
        LOGGER.warning("FR_data.json not found at %s; defaulting to empty map.", FR_DATA_PATH)
        return {}
    try:
        return json.loads(FR_DATA_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.error("Unable to read %s: %s", FR_DATA_PATH, exc)
        raise


def _check_file_exists(step_dir: Path, foodcourt_id: str, restaurant_id: str, 
                       item_name: Optional[str] = None, step_type: str = "") -> bool:
    """Check if output file exists for a step."""
    if item_name:
        from src.util.pipeline_utils import get_file_name
        filename = get_file_name(foodcourt_id, restaurant_id, item_name, step_type)
        # Files are stored in foodcourt_id subdirectory
        return (step_dir / foodcourt_id / filename).exists()
    else:
        # For restaurant-level files (like enrichment, preprocessing) - check for CSV
        return (step_dir / foodcourt_id / f"{restaurant_id}.csv").exists()


def _should_process_step(step_name: str, foodcourt_id: str, restaurant_id: str,
                        item_name: Optional[str] = None) -> Tuple[bool, bool]:
    """
    Determine if a step should be processed.
    Returns: (should_process, force_retrain)
    - should_process: True if step should run
    - force_retrain: True if we should force retrain even if data exists
    """
    from src.util.pipeline_utils import is_retrain_config_empty
    
    # Check if retrain.json is empty
    retrain_is_empty = is_retrain_config_empty()
    
    # Check if data exists
    # Note: model_generation uses trainedModel directory, not a separate model_generation folder
    step_dirs = {
        # "data_fetch": DATA_FETCH_DIR,  # Commented out since data_fetch step is not used
        "enrich_data": ENRICH_DATA_DIR,
        "preprocessing": PREPROCESSING_DIR,
        "postprocessing": POSTPROCESSING_DIR,
    }
    
    # For model_generation, check trainedModel directory
    if step_name == "model_generation":
        step_dir = TRAINED_MODEL_DIR
    else:
        step_dir = step_dirs.get(step_name)
    data_exists = False
    if step_dir and step_dir.exists():
        file_exists = _check_file_exists(step_dir, foodcourt_id, restaurant_id, item_name, step_name)
        if file_exists:
            data_exists = True
    
    # If retrain.json is empty and data exists, skip the step
    if retrain_is_empty and data_exists:
        return False, False
    
    # Check retrain.json for force retrain
    force_retrain = should_force_retrain(step_name, foodcourt_id, restaurant_id, item_name)
    
    if force_retrain:
        return True, True
    
    # If data doesn't exist, need to process
    if not data_exists:
        return True, False
    
    # Data exists but retrain.json has entries (might not match this specific item)
    # In this case, we still skip unless force_retrain is True (already checked above)
    return False, False


def run_data_fetch(prod_mode: bool = False):
    """
    Step 1: Fetch raw restaurant orders from MongoDB.
    If prod_mode=True, update data and rerun enrich_data for new data only.
    """
    LOGGER.info("=" * 80)
    LOGGER.info("STEP 1: DATA FETCH")
    LOGGER.info("=" * 80)
    
    try:
        from src.util.pipeline_utils import import_step_function
        fetch_main = import_step_function("data_fetch")
        fetch_main(prod_mode=prod_mode)
        LOGGER.info("Data fetch completed.")
    except Exception as exc:
        LOGGER.error("Error in data_fetch step: %s", exc)
        pipeline_logger.log_general_error("data_fetch", f"Error: {exc}", str(exc))
        raise


def run_enrich_data(prod_mode: bool = False, retrain_config: Optional[Dict] = None, 
                   file_saver=None, restaurant_tracker=None, checkpoint_manager=None):
    """
    Step 2: Enrich restaurant orders with metadata, weather, holidays.
    If prod_mode=True, only process new data added since last run.
    
    Args:
        prod_mode: If True, process only new data
        retrain_config: Retrain configuration from retrain.json
        file_saver: FileSaver instance for saving files
        restaurant_tracker: RestaurantTracker instance for tracking item status
        checkpoint_manager: CheckpointManager instance for checkpoint/resume functionality
    """
    LOGGER.info("=" * 80)
    LOGGER.info("STEP 2: ENRICH DATA")
    LOGGER.info("=" * 80)
    
    try:
        from src.util.pipeline_utils import import_step_function
        enrich_main = import_step_function("enrich_data")
        enrich_main(retrain_config=retrain_config, file_saver=file_saver, 
                   restaurant_tracker=restaurant_tracker, checkpoint_manager=checkpoint_manager)
        pipeline_logger.save()
        LOGGER.info("Enrichment completed. Logs saved to %s", pipeline_logger.log_path)
    except KeyboardInterrupt:
        LOGGER.warning("Enrichment step interrupted by user")
        pipeline_logger.log_general_error("enrich_data", "Step interrupted by user", "KeyboardInterrupt")
        pipeline_logger.save()  # Save logs even on interruption
        raise
    except Exception as exc:
        LOGGER.error("Error in enrich_data step: %s", exc)
        pipeline_logger.log_general_error("enrich_data", f"Error: {exc}", str(exc))
        pipeline_logger.save()  # Save logs even on error
        raise


def run_preprocessing(retrain_config: Optional[Dict] = None, file_saver=None, 
                     restaurant_tracker=None, checkpoint_manager=None):
    """
    Step 3: Preprocess data for modeling.
    
    Args:
        retrain_config: Retrain configuration from retrain.json
        file_saver: FileSaver instance for saving files
        restaurant_tracker: RestaurantTracker instance for tracking item status
        checkpoint_manager: CheckpointManager instance for checkpoint/resume functionality
    """
    LOGGER.info("=" * 80)
    LOGGER.info("STEP 3: PREPROCESSING")
    LOGGER.info("=" * 80)
    
    try:
        from src.util.pipeline_utils import import_step_function
        preprocess_main = import_step_function("preprocessing")
        preprocess_main(retrain_config=retrain_config, file_saver=file_saver, 
                       restaurant_tracker=restaurant_tracker, checkpoint_manager=checkpoint_manager)
        pipeline_logger.save()
        LOGGER.info("Preprocessing completed. Logs saved to %s", pipeline_logger.log_path)
    except Exception as exc:
        LOGGER.error("Error in preprocessing step: %s", exc)
        pipeline_logger.log_general_error("preprocessing", f"Error: {exc}", str(exc))
        raise


def run_model_generation(retrain_config: Optional[Dict] = None, file_saver=None, 
                        restaurant_tracker=None, checkpoint_manager=None):
    """
    Step 4: Train models for each item.
    
    Args:
        retrain_config: Retrain configuration from retrain.json
        file_saver: FileSaver instance for saving files
        restaurant_tracker: RestaurantTracker instance for tracking item status
    """
    LOGGER.info("=" * 80)
    LOGGER.info("STEP 4: MODEL GENERATION")
    LOGGER.info("=" * 80)
    
    try:
        from src.util.pipeline_utils import import_step_function
        model_main = import_step_function("model_generation")
        model_main(retrain_config=retrain_config, file_saver=file_saver, 
                  restaurant_tracker=restaurant_tracker, checkpoint_manager=checkpoint_manager)
        pipeline_logger.save()
        LOGGER.info("Model generation completed. Logs saved to %s", pipeline_logger.log_path)
    except Exception as exc:
        LOGGER.error("Error in model_generation step: %s", exc)
        pipeline_logger.log_general_error("model_generation", f"Error: {exc}", str(exc))
        raise


def run_postprocessing(retrain_config: Optional[Dict] = None, file_saver=None, 
                      restaurant_tracker=None, checkpoint_manager=None):
    """
    Step 5: Postprocess predictions with day-of-week redistribution.
    
    Args:
        retrain_config: Retrain configuration from retrain.json
        file_saver: FileSaver instance for saving files
        restaurant_tracker: RestaurantTracker instance for tracking item status
    """
    LOGGER.info("=" * 80)
    LOGGER.info("STEP 5: POSTPROCESSING")
    LOGGER.info("=" * 80)
    
    try:
        from src.util.pipeline_utils import import_step_function
        postprocess_main = import_step_function("postprocessing")
        postprocess_main(retrain_config=retrain_config, file_saver=file_saver, 
                        restaurant_tracker=restaurant_tracker)
        pipeline_logger.save()
        LOGGER.info("Postprocessing completed. Logs saved to %s", pipeline_logger.log_path)
    except Exception as exc:
        LOGGER.error("Error in postprocessing step: %s", exc)
        pipeline_logger.log_general_error("postprocessing", f"Error: {exc}", str(exc))
        raise


def run_compiled_result_generation(retrain_config: Optional[Dict] = None, file_saver=None, 
                                  restaurant_tracker=None, checkpoint_manager=None):
    """
    Step 6: Compile validation and training errors.
    Now saves to restaurant tracker JSON files instead of CSV.
    
    Args:
        retrain_config: Retrain configuration from retrain.json
        file_saver: FileSaver instance for saving files
        restaurant_tracker: RestaurantTracker instance for tracking item status
    """
    LOGGER.info("=" * 80)
    LOGGER.info("STEP 6: COMPILED RESULT GENERATION")
    LOGGER.info("=" * 80)
    
    try:
        from src.util.pipeline_utils import import_step_function
        compile_main = import_step_function("compiled_result_generation")
        compile_main(retrain_config=retrain_config, file_saver=file_saver, restaurant_tracker=restaurant_tracker)
        LOGGER.info("Compiled result generation completed.")
    except Exception as exc:
        LOGGER.error("Error in compiled_result_generation step: %s", exc)
        pipeline_logger.log_general_error("compiled_result_generation", f"Error: {exc}", str(exc))
        raise


def main(prod_mode: bool = False, use_checkpoint: bool = True, reset_checkpoint: bool = False):
    """
    Main pipeline orchestrator.
    
    Args:
        prod_mode: If True, update data and rerun enrich_data for new data only.
        use_checkpoint: If True, use checkpoint system to resume from last position.
        reset_checkpoint: If True, reset all checkpoints before starting.
    """
    # Log active pipeline type and configuration
    LOGGER.info("=" * 80)
    LOGGER.info("PIPELINE CONFIGURATION")
    LOGGER.info("=" * 80)
    LOGGER.info("Active Pipeline Type: %s", PIPELINE_TYPE)
    pipeline_info = pipeline_config.get("pipelines", {}).get(PIPELINE_TYPE, {})
    if pipeline_info:
        LOGGER.info("Description: %s", pipeline_info.get("description", "N/A"))
        steps = pipeline_info.get("steps", {})
        enabled_steps = [name for name, config in steps.items() if config.get("enabled", True)]
        LOGGER.info("Enabled Steps: %s", ", ".join(enabled_steps))
    LOGGER.info("=" * 80)
    
    # Ensure we run with project root on sys.path
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # Create output directories
    # Note: DATA_FETCH_DIR removed since data_fetch step is commented out
    # Note: COMPILED_RESULTS_DIR removed - compiled results are saved to restaurant_tracker JSON files
    for dir_path in [OUTPUT_DIR, ENRICH_DATA_DIR, PREPROCESSING_DIR,
                     POSTPROCESSING_DIR,
                     TRAINED_MODEL_DIR, TRAINED_MODEL_MODELS_DIR, TRAINED_MODEL_RESULTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("=" * 80)
    LOGGER.info("GO-KHANA PIPELINE - Starting Execution")
    LOGGER.info("Production Mode: %s", prod_mode)
    LOGGER.info("=" * 80)

    # Set global pipeline start time
    import time
    set_pipeline_start_time(time.time())

    # Initialize all database connections before any processing starts
    from src.util.connection_manager import get_connection_manager
    conn_mgr = get_connection_manager()
    LOGGER.info("Initializing database connections before processing...")
    try:
        conn_mgr.initialize_all_connections()
        LOGGER.info("✅ Database connections initialized successfully")
    except Exception as e:
        LOGGER.error(f"Failed to initialize database connections: {e}")
        raise

    # Initialize checkpoint manager
    checkpoint_manager = get_checkpoint_manager(PIPELINE_TYPE)
    
    if reset_checkpoint:
        checkpoint_manager.reset_all()
        LOGGER.info("Checkpoints reset. Starting fresh.")
    elif use_checkpoint:
        # Clear any in-progress items from previous run (in case of crash)
        checkpoint_manager.clear_in_progress()
        progress_summary = checkpoint_manager.get_progress_summary()
        LOGGER.info("Checkpoint system enabled. Progress summary:")
        for step_name, stats in progress_summary.items():
            if stats["completed"] > 0 or stats["failed"] > 0:
                LOGGER.info(f"  {step_name}: {stats['completed']} completed, "
                          f"{stats['failed']} failed, {stats['in_progress']} in progress")
    else:
        LOGGER.info("Checkpoint system disabled. Processing all items.")

    # Load retrain configuration
    retrain_config = load_retrain_config()
    from src.util.pipeline_utils import is_retrain_config_empty
    retrain_is_completely_empty = is_retrain_config_empty()
    
    LOGGER.info("Retrain configuration loaded: %s", {k: len(v) for k, v in retrain_config.items()})
    LOGGER.info("Retrain.json is completely empty: %s", retrain_is_completely_empty)
    
    # Import needed for helper function
    from src.util.pipeline_utils import get_retrain_config_for_step
    
    # Helper function to check if any later step has entries (for prerequisite logic)
    def has_later_step_entries(step_name: str) -> bool:
        """Check if any step after the given step has entries in retrain.json."""
        step_order = ["data_fetch", "enrich_data", "preprocessing", "model_generation", 
                     "postprocessing", "compiled_result_generation"]
        try:
            current_idx = step_order.index(step_name)
            for later_step in step_order[current_idx + 1:]:
                step_config = get_retrain_config_for_step(later_step)
                if (step_config.get("foodcourt_ids") or 
                    step_config.get("restaurant_ids") or 
                    step_config.get("item_ids") or
                    step_config.get("item_names")):
                    return True
        except ValueError:
            pass
        return False
    
    try:
        # Step 1: Data Fetch - COMMENTED OUT FOR TESTING
        # data_fetch_retrain = retrain_config.get("data_fetch", [])
        # if data_fetch_retrain or prod_mode:
        #     # If retrain.json has entries for data_fetch OR prod_mode is True, run it
        #     run_data_fetch(prod_mode=prod_mode)
        # elif retrain_is_completely_empty:
        #     # If retrain.json is completely empty, check if data_fetch needs to run
        #     # (This would check if fetched_data exists, but data_fetch logic is separate)
        #     run_data_fetch(prod_mode=prod_mode)
        # else:
        #     LOGGER.info("Skipping data_fetch (not in retrain.json and retrain.json is not empty)")

        # Step 2: Enrich Data
        enrich_step_config = get_retrain_config_for_step("enrich_data")
        enrich_foodcourt_ids = enrich_step_config.get("foodcourt_ids", [])
        enrich_restaurant_ids = enrich_step_config.get("restaurant_ids", [])
        enrich_item_ids = enrich_step_config.get("item_ids", [])
        enrich_item_names = enrich_step_config.get("item_names", [])
        enrich_has_config = bool(enrich_foodcourt_ids or enrich_restaurant_ids or enrich_item_ids or enrich_item_names)
        
        # Debug logging
        LOGGER.info(f"enrich_data config check: foodcourt_ids={len(enrich_foodcourt_ids)}, restaurant_ids={len(enrich_restaurant_ids)}, item_ids={len(enrich_item_ids)}, item_names={len(enrich_item_names)}, has_config={enrich_has_config}")
        
        # Check if later steps need enrichment (prerequisite check)
        later_steps_need_enrichment = has_later_step_entries("enrich_data")
        
        if enrich_has_config:
            # retrain.json has entries for enrich_data: force retrain ONLY those entries
            LOGGER.info("Processing enrich_data: retrain.json has entries - will process ONLY those")
            from src.util.pipeline_utils import import_step_function
            enrich_main = import_step_function("enrich_data")
            enrich_main(retrain_config=retrain_config, file_saver=file_saver, 
                       restaurant_tracker=restaurant_tracker,
                       checkpoint_manager=checkpoint_manager if use_checkpoint else None)
        elif later_steps_need_enrichment:
            # Later steps have entries, so we need to run enrichment for those items
            LOGGER.info("Processing enrich_data: Later steps have entries - will process items needed for those steps")
            # Create a temporary retrain config that includes items from later steps
            # This ensures enrichment runs for items needed by preprocessing/model_generation
            temp_retrain_config = retrain_config.copy()
            # Merge item_ids from later steps into enrich_data config
            all_item_ids = []
            seen_keys = set()
            for step_name in ["preprocessing", "model_generation", "postprocessing", "compiled_result_generation"]:
                step_config = get_retrain_config_for_step(step_name)
                step_item_ids = step_config.get("item_ids", [])
                # Add unique items (avoid duplicates by using a key)
                for item in step_item_ids:
                    # Create a unique key for the item
                    item_key = (item.get("foodcourt_id", ""), 
                               item.get("restaurant_id", ""), 
                               item.get("item_id", "") or item.get("item_name", ""))
                    if item_key not in seen_keys:
                        seen_keys.add(item_key)
                        all_item_ids.append(item)
            
            if all_item_ids:
                temp_retrain_config["enrich_data"] = {"item_ids": all_item_ids}
                LOGGER.info(f"Enriching {len(all_item_ids)} items needed by later steps")
            
            from src.util.pipeline_utils import import_step_function
            enrich_main = import_step_function("enrich_data")
            enrich_main(retrain_config=temp_retrain_config, file_saver=file_saver, 
                       restaurant_tracker=restaurant_tracker,
                       checkpoint_manager=checkpoint_manager if use_checkpoint else None)
        elif retrain_is_completely_empty:
            # retrain.json is completely empty: process all items that need processing (check previous step)
            LOGGER.info("Processing enrich_data: retrain.json is empty - will process items missing from previous step")
            from src.util.pipeline_utils import import_step_function
            enrich_main = import_step_function("enrich_data")
            enrich_main(retrain_config=retrain_config, file_saver=file_saver, 
                       restaurant_tracker=restaurant_tracker,
                       checkpoint_manager=checkpoint_manager if use_checkpoint else None)
        else:
            # retrain.json is not empty but enrich_data config is empty and no later steps need it
            LOGGER.info("Skipping enrich_data (enrich_data config is empty and no later steps require it)")

        # Step 3: Preprocessing
        preprocess_step_config = get_retrain_config_for_step("preprocessing")
        preprocess_foodcourt_ids = preprocess_step_config.get("foodcourt_ids", [])
        preprocess_restaurant_ids = preprocess_step_config.get("restaurant_ids", [])
        preprocess_item_ids = preprocess_step_config.get("item_ids", [])
        preprocess_item_names = preprocess_step_config.get("item_names", [])
        preprocess_has_config = bool(preprocess_foodcourt_ids or preprocess_restaurant_ids or preprocess_item_ids or preprocess_item_names)
        
        # Check if later steps need preprocessing (prerequisite check)
        later_steps_need_preprocessing = has_later_step_entries("preprocessing")
        
        if preprocess_has_config:
            LOGGER.info("Processing preprocessing: retrain.json has entries - will process ONLY those")
            from src.util.pipeline_utils import import_step_function
            preprocess_main = import_step_function("preprocessing")
            preprocess_main(retrain_config=retrain_config, 
                           file_saver=file_saver, restaurant_tracker=restaurant_tracker,
                           checkpoint_manager=checkpoint_manager if use_checkpoint else None)
        elif later_steps_need_preprocessing:
            # Later steps have entries, so we need to run preprocessing for those items
            LOGGER.info("Processing preprocessing: Later steps have entries - will process items needed for those steps")
            # Create a temporary retrain config that includes items from later steps
            temp_retrain_config = retrain_config.copy()
            # Merge item_ids from later steps into preprocessing config
            all_item_ids = []
            seen_keys = set()
            for step_name in ["model_generation", "postprocessing", "compiled_result_generation"]:
                step_config = get_retrain_config_for_step(step_name)
                step_item_ids = step_config.get("item_ids", [])
                # Add unique items (avoid duplicates by using a key)
                for item in step_item_ids:
                    # Create a unique key for the item
                    item_key = (item.get("foodcourt_id", ""), 
                               item.get("restaurant_id", ""), 
                               item.get("item_id", "") or item.get("item_name", ""))
                    if item_key not in seen_keys:
                        seen_keys.add(item_key)
                        all_item_ids.append(item)
            
            if all_item_ids:
                temp_retrain_config["preprocessing"] = {"item_ids": all_item_ids}
                LOGGER.info(f"Preprocessing {len(all_item_ids)} items needed by later steps")
            
            from src.util.pipeline_utils import import_step_function
            preprocess_main = import_step_function("preprocessing")
            preprocess_main(retrain_config=temp_retrain_config, 
                           file_saver=file_saver, restaurant_tracker=restaurant_tracker,
                           checkpoint_manager=checkpoint_manager if use_checkpoint else None)
        elif retrain_is_completely_empty:
            LOGGER.info("Processing preprocessing: retrain.json is empty - will process items missing from previous step")
            from src.util.pipeline_utils import import_step_function
            preprocess_main = import_step_function("preprocessing")
            preprocess_main(retrain_config=retrain_config, 
                           file_saver=file_saver, restaurant_tracker=restaurant_tracker,
                           checkpoint_manager=checkpoint_manager if use_checkpoint else None)
        else:
            LOGGER.info("Skipping preprocessing (preprocessing config is empty and no later steps require it)")

        # Step 4: Model Generation
        model_step_config = get_retrain_config_for_step("model_generation")
        model_foodcourt_ids = model_step_config.get("foodcourt_ids", [])
        model_restaurant_ids = model_step_config.get("restaurant_ids", [])
        model_item_ids = model_step_config.get("item_ids", [])
        model_item_names = model_step_config.get("item_names", [])
        model_has_config = bool(model_foodcourt_ids or model_restaurant_ids or model_item_ids or model_item_names)
        
        # Check if later steps need model generation (prerequisite check)
        later_steps_need_models = has_later_step_entries("model_generation")
        
        if model_has_config:
            LOGGER.info("Processing model_generation: retrain.json has entries - will process ONLY those")
            from src.util.pipeline_utils import import_step_function
            model_main = import_step_function("model_generation")
            model_main(retrain_config=retrain_config, 
                      file_saver=file_saver, restaurant_tracker=restaurant_tracker,
                      checkpoint_manager=checkpoint_manager if use_checkpoint else None)
        elif later_steps_need_models:
            # Later steps have entries, so we need to run model generation for those items
            LOGGER.info("Processing model_generation: Later steps have entries - will process items needed for those steps")
            # Create a temporary retrain config that includes items from later steps
            temp_retrain_config = retrain_config.copy()
            # Merge item_ids from later steps into model_generation config
            all_item_ids = []
            seen_keys = set()
            for step_name in ["postprocessing", "compiled_result_generation"]:
                step_config = get_retrain_config_for_step(step_name)
                step_item_ids = step_config.get("item_ids", [])
                # Add unique items (avoid duplicates by using a key)
                for item in step_item_ids:
                    # Create a unique key for the item
                    item_key = (item.get("foodcourt_id", ""), 
                               item.get("restaurant_id", ""), 
                               item.get("item_id", "") or item.get("item_name", ""))
                    if item_key not in seen_keys:
                        seen_keys.add(item_key)
                        all_item_ids.append(item)
            
            if all_item_ids:
                temp_retrain_config["model_generation"] = {"item_ids": all_item_ids}
                LOGGER.info(f"Generating models for {len(all_item_ids)} items needed by later steps")
            
            from src.util.pipeline_utils import import_step_function
            model_main = import_step_function("model_generation")
            model_main(retrain_config=temp_retrain_config, 
                      file_saver=file_saver, restaurant_tracker=restaurant_tracker,
                      checkpoint_manager=checkpoint_manager if use_checkpoint else None)
        elif retrain_is_completely_empty:
            LOGGER.info("Processing model_generation: retrain.json is empty - will process items missing from previous step")
            from src.util.pipeline_utils import import_step_function
            model_main = import_step_function("model_generation")
            model_main(retrain_config=retrain_config, 
                      file_saver=file_saver, restaurant_tracker=restaurant_tracker,
                      checkpoint_manager=checkpoint_manager if use_checkpoint else None)
        else:
            LOGGER.info("Skipping model_generation (model_generation config is empty and no later steps require it)")

        # Step 5: Postprocessing
        postprocess_step_config = get_retrain_config_for_step("postprocessing")
        postprocess_foodcourt_ids = postprocess_step_config.get("foodcourt_ids", [])
        postprocess_restaurant_ids = postprocess_step_config.get("restaurant_ids", [])
        postprocess_item_ids = postprocess_step_config.get("item_ids", [])
        postprocess_item_names = postprocess_step_config.get("item_names", [])
        postprocess_has_config = bool(postprocess_foodcourt_ids or postprocess_restaurant_ids or postprocess_item_ids or postprocess_item_names)
        
        # Check if later steps need postprocessing (prerequisite check)
        later_steps_need_postprocessing = has_later_step_entries("postprocessing")
        
        if postprocess_has_config:
            LOGGER.info("Processing postprocessing: retrain.json has entries - will process ONLY those")
            run_postprocessing(retrain_config=retrain_config, 
                             file_saver=file_saver, restaurant_tracker=restaurant_tracker,
                             checkpoint_manager=checkpoint_manager if use_checkpoint else None)
        elif later_steps_need_postprocessing:
            # Later steps have entries, so we need to run postprocessing for those items
            LOGGER.info("Processing postprocessing: Later steps have entries - will process items needed for those steps")
            # Create a temporary retrain config that includes items from later steps
            temp_retrain_config = retrain_config.copy()
            # Merge item_ids from later steps into postprocessing config
            all_item_ids = []
            seen_keys = set()
            for step_name in ["compiled_result_generation"]:
                step_config = get_retrain_config_for_step(step_name)
                step_item_ids = step_config.get("item_ids", [])
                # Add unique items (avoid duplicates by using a key)
                for item in step_item_ids:
                    # Create a unique key for the item
                    item_key = (item.get("foodcourt_id", ""), 
                               item.get("restaurant_id", ""), 
                               item.get("item_id", "") or item.get("item_name", ""))
                    if item_key not in seen_keys:
                        seen_keys.add(item_key)
                        all_item_ids.append(item)
            
            if all_item_ids:
                temp_retrain_config["postprocessing"] = {"item_ids": all_item_ids}
                LOGGER.info(f"Postprocessing {len(all_item_ids)} items needed by later steps")
            
            run_postprocessing(retrain_config=temp_retrain_config, 
                             file_saver=file_saver, restaurant_tracker=restaurant_tracker,
                             checkpoint_manager=checkpoint_manager if use_checkpoint else None)
        elif retrain_is_completely_empty:
            LOGGER.info("Processing postprocessing: retrain.json is empty - will process items missing from previous step")
            run_postprocessing(retrain_config=retrain_config, 
                             file_saver=file_saver, restaurant_tracker=restaurant_tracker,
                             checkpoint_manager=checkpoint_manager if use_checkpoint else None)
        else:
            LOGGER.info("Skipping postprocessing (postprocessing config is empty and no later steps require it)")

        # Step 6: Compiled Result Generation
        compile_step_config = get_retrain_config_for_step("compiled_result_generation")
        compile_foodcourt_ids = compile_step_config.get("foodcourt_ids", [])
        compile_restaurant_ids = compile_step_config.get("restaurant_ids", [])
        compile_item_ids = compile_step_config.get("item_ids", [])
        compile_item_names = compile_step_config.get("item_names", [])
        compile_has_config = bool(compile_foodcourt_ids or compile_restaurant_ids or compile_item_ids or compile_item_names)
        
        if compile_has_config:
            LOGGER.info("Processing compiled_result_generation: retrain.json has entries - will process ONLY those")
            run_compiled_result_generation(retrain_config=retrain_config, 
                                          file_saver=file_saver, restaurant_tracker=restaurant_tracker,
                                          checkpoint_manager=checkpoint_manager if use_checkpoint else None)
        elif retrain_is_completely_empty:
            LOGGER.info("Processing compiled_result_generation: retrain.json is empty - will process items missing from previous step")
            run_compiled_result_generation(retrain_config=retrain_config, 
                                          file_saver=file_saver, restaurant_tracker=restaurant_tracker,
                                          checkpoint_manager=checkpoint_manager if use_checkpoint else None)
        else:
            LOGGER.info("Skipping compiled_result_generation (compiled_result_generation config is empty in retrain.json, but other steps have entries)")

        # Restaurant tracking files are saved automatically per restaurant
        # No need to combine - each restaurant has its own JSON file
        
        # Save pipeline logs
        pipeline_logger.save()

        LOGGER.info("=" * 80)
        LOGGER.info("PIPELINE COMPLETED SUCCESSFULLY")
        LOGGER.info("=" * 80)

    except KeyboardInterrupt:
        LOGGER.warning("Pipeline interrupted by user (KeyboardInterrupt)")
        pipeline_logger.log_general_error("pipeline", "Pipeline interrupted by user", "KeyboardInterrupt")
        # Combine temp files and save final file_locator before exiting
        try:
            # Restaurant tracking files are saved automatically per restaurant
            pipeline_logger.save()
            LOGGER.info("File locator and logs saved before exit")
        except Exception as save_exc:
            LOGGER.error("Failed to save file locator/logs on interruption: %s", save_exc)
        raise
    except Exception as exc:
        LOGGER.error("Pipeline failed with error: %s", exc, exc_info=True)
        pipeline_logger.log_general_error("pipeline", f"Pipeline failed: {exc}", str(exc))
        pipeline_logger.save()
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run GoKhana pipeline")
    parser.add_argument("--prod-mode", action="store_true", 
                       help="Production mode: update data and rerun for new data only")
    parser.add_argument("--no-checkpoint", action="store_true",
                       help="Disable checkpoint/resume functionality (process all items)")
    parser.add_argument("--reset-checkpoint", action="store_true",
                       help="Reset all checkpoints before starting (start fresh)")
    args = parser.parse_args()
    main(prod_mode=args.prod_mode, 
         use_checkpoint=not args.no_checkpoint,
         reset_checkpoint=args.reset_checkpoint)
