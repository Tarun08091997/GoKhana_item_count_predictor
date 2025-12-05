"""
View and manage pipeline checkpoint status.

Usage:
    python extraScripts/view_checkpoint.py                    # View summary
    python extraScripts/view_checkpoint.py --detailed          # Detailed view
    python extraScripts/view_checkpoint.py --reset-step STEP   # Reset a step
    python extraScripts/view_checkpoint.py --reset-all         # Reset all
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.util.checkpoint_manager import get_checkpoint_manager


def print_summary(checkpoint_manager):
    """Print a summary of checkpoint status."""
    summary = checkpoint_manager.get_progress_summary()
    
    print("\n" + "=" * 80)
    print("PIPELINE CHECKPOINT SUMMARY")
    print("=" * 80)
    
    total_completed = 0
    total_failed = 0
    
    for step_name, stats in summary.items():
        completed = stats["completed"]
        failed = stats["failed"]
        in_progress = stats["in_progress"]
        total_completed += completed
        total_failed += failed
        
        status = "✓" if completed > 0 else "○"
        print(f"{status} {step_name:30s} | Completed: {completed:4d} | "
              f"Failed: {failed:3d} | In Progress: {in_progress:2d}")
    
    print("=" * 80)
    print(f"Total Completed Items: {total_completed}")
    print(f"Total Failed Items: {total_failed}")
    print("=" * 80)
    
    # Show checkpoint file location
    checkpoint_path = checkpoint_manager.checkpoint_path
    if checkpoint_path.exists():
        file_size = checkpoint_path.stat().st_size
        print(f"\nCheckpoint file: {checkpoint_path}")
        print(f"File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")


def print_detailed(checkpoint_manager):
    """Print detailed checkpoint information."""
    if checkpoint_manager._checkpoint_data is None:
        print("No checkpoint data available.")
        return
    
    data = checkpoint_manager._checkpoint_data
    
    print("\n" + "=" * 80)
    print("DETAILED CHECKPOINT INFORMATION")
    print("=" * 80)
    
    # Metadata
    metadata = data.get("metadata", {})
    print("\nMetadata:")
    print(f"  Created: {metadata.get('created_at', 'N/A')}")
    print(f"  Last Updated: {metadata.get('last_updated', 'N/A')}")
    print(f"  Pipeline Type: {metadata.get('pipeline_type', 'N/A')}")
    
    # Steps
    steps = data.get("steps", {})
    for step_name, step_data in steps.items():
        completed = step_data.get("completed_items", [])
        failed = step_data.get("failed_items", [])
        in_progress = step_data.get("in_progress_items", [])
        
        print(f"\n{step_name}:")
        print(f"  Completed: {len(completed)} items")
        if completed:
            print(f"    First: {completed[0]}")
            if len(completed) > 1:
                print(f"    Last: {completed[-1]}")
        
        print(f"  Failed: {len(failed)} items")
        if failed:
            for item in failed[:5]:  # Show first 5 failed items
                error = item.get("error", "Unknown error")
                item_key = f"{item.get('foodcourt_id', '')}/{item.get('restaurant_id', '')}/{item.get('item_id', item.get('item_name', ''))}"
                print(f"    - {item_key}: {error}")
            if len(failed) > 5:
                print(f"    ... and {len(failed) - 5} more")
        
        print(f"  In Progress: {len(in_progress)} items")
        if in_progress:
            for item in in_progress:
                item_key = f"{item.get('foodcourt_id', '')}/{item.get('restaurant_id', '')}/{item.get('item_id', item.get('item_name', ''))}"
                print(f"    - {item_key}")


def main():
    parser = argparse.ArgumentParser(description="View and manage pipeline checkpoints")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed checkpoint information")
    parser.add_argument("--reset-step", type=str,
                       help="Reset checkpoint for a specific step (e.g., 'model_generation')")
    parser.add_argument("--reset-all", action="store_true",
                       help="Reset all checkpoints")
    parser.add_argument("--clear-in-progress", action="store_true",
                       help="Clear in-progress items (useful after a crash)")
    
    args = parser.parse_args()
    
    checkpoint_manager = get_checkpoint_manager()
    
    if args.reset_all:
        confirm = input("Are you sure you want to reset ALL checkpoints? (yes/no): ")
        if confirm.lower() == "yes":
            checkpoint_manager.reset_all()
            print("✓ All checkpoints reset.")
        else:
            print("Cancelled.")
    elif args.reset_step:
        confirm = input(f"Are you sure you want to reset checkpoint for '{args.reset_step}'? (yes/no): ")
        if confirm.lower() == "yes":
            checkpoint_manager.reset_step(args.reset_step)
            print(f"✓ Checkpoint reset for '{args.reset_step}'.")
        else:
            print("Cancelled.")
    elif args.clear_in_progress:
        checkpoint_manager.clear_in_progress()
        print("✓ In-progress items cleared.")
    elif args.detailed:
        print_detailed(checkpoint_manager)
    else:
        print_summary(checkpoint_manager)


if __name__ == "__main__":
    main()

