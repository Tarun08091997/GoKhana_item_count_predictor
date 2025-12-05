"""
Checkpoint Manager for Pipeline Resume Functionality.

This module provides checkpoint/resume capabilities for the pipeline,
allowing it to resume from where it left off if interrupted.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from datetime import datetime

LOGGER = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoint files to track pipeline progress and enable resume functionality.
    
    Checkpoint file structure:
    {
        "metadata": {
            "created_at": "2025-01-01T12:00:00",
            "last_updated": "2025-01-01T13:00:00",
            "pipeline_type": "FRID_LEVEL"
        },
        "steps": {
            "enrich_data": {
                "completed_items": [
                    {"foodcourt_id": "...", "restaurant_id": "...", "item_id": "..."},
                    ...
                ],
                "failed_items": [...],
                "in_progress_items": [...]
            },
            "preprocessing": {...},
            "model_generation": {...},
            "postprocessing": {...},
            "compiled_result_generation": {...}
        }
    }
    """
    
    def __init__(self, checkpoint_path: Optional[Path] = None, pipeline_type: str = "FRID_LEVEL"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, uses default location.
            pipeline_type: Type of pipeline (FRID_LEVEL, etc.)
        """
        from src.util.path_utils import get_input_base_dir
        
        if checkpoint_path is None:
            checkpoint_path = get_input_base_dir() / "pipeline_checkpoint.json"
        
        self.checkpoint_path = checkpoint_path
        self.pipeline_type = pipeline_type
        self._checkpoint_data: Optional[Dict[str, Any]] = None
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load checkpoint data from file."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    self._checkpoint_data = json.load(f)
                LOGGER.info(f"Loaded checkpoint from {self.checkpoint_path}")
            except Exception as e:
                LOGGER.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
                self._checkpoint_data = self._create_empty_checkpoint()
        else:
            self._checkpoint_data = self._create_empty_checkpoint()
            LOGGER.info("No existing checkpoint found. Starting fresh.")
    
    def _create_empty_checkpoint(self) -> Dict[str, Any]:
        """Create an empty checkpoint structure."""
        return {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "pipeline_type": self.pipeline_type
            },
            "steps": {
                "enrich_data": {
                    "completed_items": [],
                    "failed_items": [],
                    "in_progress_items": []
                },
                "preprocessing": {
                    "completed_items": [],
                    "failed_items": [],
                    "in_progress_items": []
                },
                "model_generation": {
                    "completed_items": [],
                    "failed_items": [],
                    "in_progress_items": []
                },
                "postprocessing": {
                    "completed_items": [],
                    "failed_items": [],
                    "in_progress_items": []
                },
                "compiled_result_generation": {
                    "completed_items": [],
                    "failed_items": [],
                    "in_progress_items": []
                }
            }
        }
    
    def _save_checkpoint(self):
        """Save checkpoint data to file."""
        if self._checkpoint_data is None:
            return
        
        self._checkpoint_data["metadata"]["last_updated"] = datetime.now().isoformat()
        self._checkpoint_data["metadata"]["pipeline_type"] = self.pipeline_type
        
        try:
            # Create backup before saving
            if self.checkpoint_path.exists():
                backup_path = self.checkpoint_path.with_suffix('.json.bak')
                import shutil
                shutil.copy2(self.checkpoint_path, backup_path)
            
            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(self._checkpoint_data, f, indent=2, ensure_ascii=False)
            
            LOGGER.debug(f"Checkpoint saved to {self.checkpoint_path}")
        except Exception as e:
            LOGGER.error(f"Failed to save checkpoint: {e}")
    
    def _get_item_key(self, item: Dict[str, str]) -> str:
        """Generate a unique key for an item."""
        foodcourt_id = item.get("foodcourt_id", "")
        restaurant_id = item.get("restaurant_id", "")
        item_id = item.get("item_id", "")
        item_name = item.get("item_name", "")
        
        # Use item_id if available, otherwise use item_name
        identifier = item_id if item_id else item_name
        
        return f"{foodcourt_id}|{restaurant_id}|{identifier}"
    
    def _item_matches(self, item1: Dict[str, str], item2: Dict[str, str]) -> bool:
        """Check if two items match (same foodcourt, restaurant, and item)."""
        return self._get_item_key(item1) == self._get_item_key(item2)
    
    def is_item_completed(self, step_name: str, item: Dict[str, str]) -> bool:
        """
        Check if an item has been completed for a given step.
        
        Args:
            step_name: Name of the step (e.g., "enrich_data", "model_generation")
            item: Item dict with foodcourt_id, restaurant_id, and item_id/item_name
        
        Returns:
            True if item is completed, False otherwise
        """
        if self._checkpoint_data is None:
            return False
        
        step_data = self._checkpoint_data.get("steps", {}).get(step_name, {})
        completed_items = step_data.get("completed_items", [])
        
        for completed_item in completed_items:
            if self._item_matches(item, completed_item):
                return True
        
        return False
    
    def is_item_in_progress(self, step_name: str, item: Dict[str, str]) -> bool:
        """Check if an item is currently in progress."""
        if self._checkpoint_data is None:
            return False
        
        step_data = self._checkpoint_data.get("steps", {}).get(step_name, {})
        in_progress_items = step_data.get("in_progress_items", [])
        
        for in_progress_item in in_progress_items:
            if self._item_matches(item, in_progress_item):
                return True
        
        return False
    
    def mark_item_in_progress(self, step_name: str, item: Dict[str, str]):
        """Mark an item as in progress."""
        if self._checkpoint_data is None:
            return
        
        step_data = self._checkpoint_data.setdefault("steps", {}).setdefault(step_name, {
            "completed_items": [],
            "failed_items": [],
            "in_progress_items": []
        })
        
        in_progress_items = step_data["in_progress_items"]
        
        # Remove from in_progress if already there
        in_progress_items[:] = [i for i in in_progress_items if not self._item_matches(i, item)]
        
        # Add to in_progress
        in_progress_items.append(item.copy())
        
        self._save_checkpoint()
    
    def mark_item_completed(self, step_name: str, item: Dict[str, str]):
        """
        Mark an item as completed for a given step.
        
        Args:
            step_name: Name of the step
            item: Item dict with foodcourt_id, restaurant_id, and item_id/item_name
        """
        if self._checkpoint_data is None:
            return
        
        step_data = self._checkpoint_data.setdefault("steps", {}).setdefault(step_name, {
            "completed_items": [],
            "failed_items": [],
            "in_progress_items": []
        })
        
        completed_items = step_data["completed_items"]
        in_progress_items = step_data["in_progress_items"]
        failed_items = step_data["failed_items"]
        
        # Remove from in_progress and failed if present
        in_progress_items[:] = [i for i in in_progress_items if not self._item_matches(i, item)]
        failed_items[:] = [i for i in failed_items if not self._item_matches(i, item)]
        
        # Add to completed if not already there
        if not any(self._item_matches(i, item) for i in completed_items):
            completed_items.append(item.copy())
        
        self._save_checkpoint()
        LOGGER.debug(f"Marked item as completed for {step_name}: {self._get_item_key(item)}")
    
    def mark_item_failed(self, step_name: str, item: Dict[str, str], error: Optional[str] = None):
        """Mark an item as failed for a given step."""
        if self._checkpoint_data is None:
            return
        
        step_data = self._checkpoint_data.setdefault("steps", {}).setdefault(step_name, {
            "completed_items": [],
            "failed_items": [],
            "in_progress_items": []
        })
        
        failed_items = step_data["failed_items"]
        in_progress_items = step_data["in_progress_items"]
        
        # Remove from in_progress
        in_progress_items[:] = [i for i in in_progress_items if not self._item_matches(i, item)]
        
        # Add to failed if not already there
        failed_item = item.copy()
        if error:
            failed_item["error"] = error
        
        if not any(self._item_matches(i, failed_item) for i in failed_items):
            failed_items.append(failed_item)
        
        self._save_checkpoint()
        LOGGER.warning(f"Marked item as failed for {step_name}: {self._get_item_key(item)}")
    
    def filter_completed_items(self, step_name: str, items: List[Dict[str, str]], 
                               skip_completed: bool = True) -> List[Dict[str, str]]:
        """
        Filter out items that have already been completed.
        
        Args:
            step_name: Name of the step
            items: List of items to filter
            skip_completed: If True, skip completed items. If False, return all items.
        
        Returns:
            List of items that haven't been completed yet
        """
        if not skip_completed:
            return items
        
        remaining_items = []
        completed_count = 0
        
        for item in items:
            if self.is_item_completed(step_name, item):
                completed_count += 1
            else:
                remaining_items.append(item)
        
        if completed_count > 0:
            LOGGER.info(f"Skipping {completed_count} already completed items for {step_name}. "
                       f"Remaining: {len(remaining_items)} items to process.")
        
        return remaining_items
    
    def get_progress_summary(self) -> Dict[str, Dict[str, int]]:
        """Get a summary of progress for all steps."""
        if self._checkpoint_data is None:
            return {}
        
        summary = {}
        steps = self._checkpoint_data.get("steps", {})
        
        for step_name, step_data in steps.items():
            summary[step_name] = {
                "completed": len(step_data.get("completed_items", [])),
                "failed": len(step_data.get("failed_items", [])),
                "in_progress": len(step_data.get("in_progress_items", []))
            }
        
        return summary
    
    def reset_step(self, step_name: str):
        """Reset a step's checkpoint (clear all progress for that step)."""
        if self._checkpoint_data is None:
            return
        
        step_data = self._checkpoint_data.setdefault("steps", {}).setdefault(step_name, {
            "completed_items": [],
            "failed_items": [],
            "in_progress_items": []
        })
        
        step_data["completed_items"] = []
        step_data["failed_items"] = []
        step_data["in_progress_items"] = []
        
        self._save_checkpoint()
        LOGGER.info(f"Reset checkpoint for step: {step_name}")
    
    def reset_all(self):
        """Reset all checkpoints (start fresh)."""
        self._checkpoint_data = self._create_empty_checkpoint()
        self._save_checkpoint()
        LOGGER.info("Reset all checkpoints. Starting fresh.")
    
    def clear_in_progress(self, step_name: Optional[str] = None):
        """
        Clear in-progress items (useful when resuming after a crash).
        
        Args:
            step_name: If provided, clear only for this step. Otherwise, clear for all steps.
        """
        if self._checkpoint_data is None:
            return
        
        if step_name:
            step_data = self._checkpoint_data.get("steps", {}).get(step_name, {})
            step_data["in_progress_items"] = []
        else:
            for step_data in self._checkpoint_data.get("steps", {}).values():
                step_data["in_progress_items"] = []
        
        self._save_checkpoint()
        LOGGER.info(f"Cleared in-progress items for {step_name or 'all steps'}")


# Global checkpoint manager instance
_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager(pipeline_type: str = "FRID_LEVEL") -> CheckpointManager:
    """Get or create the global checkpoint manager instance."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager(pipeline_type=pipeline_type)
    return _checkpoint_manager


def reset_checkpoint_manager():
    """Reset the global checkpoint manager (useful for testing)."""
    global _checkpoint_manager
    _checkpoint_manager = None

