"""
Script to create retrain.json from train_model_for_items.csv using item_names instead of item_ids.
This script reads the CSV file and generates a retrain.json file with the same structure,
but uses item_names (menuitemname) instead of item_ids.
"""

import csv
import json
import os
from pathlib import Path


def create_retrain_json_from_item_names(csv_file_path, output_json_path):
    """
    Create retrain.json from CSV file using item_names instead of item_ids.
    
    Args:
        csv_file_path: Path to train_model_for_items.csv
        output_json_path: Path to output retrain.json file
    """
    # Initialize the output structure
    retrain_data = {
        "data_fetch": {},
        "enrich_data": {
            "item_names": []
        }
    }
    
    # Set to track unique combinations of foodcourt_id, restaurant_id, and item_name
    seen_combinations = set()
    
    # Read the CSV file
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            foodcourt_id = row['foodcourtid'].strip()
            restaurant_id = row['restaurant'].strip()
            item_name = row['menuitemname'].strip()
            
            # Skip rows with empty values
            if not foodcourt_id or not restaurant_id or not item_name:
                continue
            
            # Create a unique key for this combination
            combination_key = (foodcourt_id, restaurant_id, item_name)
            
            # Only add if we haven't seen this combination before
            if combination_key not in seen_combinations:
                seen_combinations.add(combination_key)
                
                retrain_data["enrich_data"]["item_names"].append({
                    "foodcourt_id": foodcourt_id,
                    "restaurant_id": restaurant_id,
                    "item_name": item_name
                })
    
    # Write to JSON file with pretty formatting
    with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(retrain_data, jsonfile, indent=2, ensure_ascii=False)
    
    print(f"Successfully created {output_json_path}")
    print(f"Total unique item_name entries: {len(retrain_data['enrich_data']['item_names'])}")
    
    return retrain_data


if __name__ == "__main__":
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_data_dir = project_root / "input_data"
    
    csv_file = input_data_dir / "train_model_for_items.csv"
    output_json = input_data_dir / "retrain.json"
    
    # Check if CSV file exists
    if not csv_file.exists():
        print(f"Error: CSV file not found at {csv_file}")
        exit(1)
    
    # Create the retrain.json file
    create_retrain_json_from_item_names(csv_file, output_json)

