"""
Script to fetch combined_foodcourt_restaurant_data from MongoDB
and save it in JSON format for progress tracking.
"""

import os
import json
import logging
from pymongo import MongoClient
from bson import ObjectId
from src.util.config_parser import ConfigManger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load config
config = ConfigManger()
cloud_config = config.read_config(type="config")

# Get MongoDB connection details for local MongoDB
local_mongo_uri = cloud_config["local_mongodb"]["LOCAL_MONGO_URI"]
local_mongo_db = cloud_config["local_mongodb"]["LOCAL_MONGO_DB"]
combined_foodcourt_coll = cloud_config["local_mongodb"]["COMBINED_FOODCOURT_COLL"]

# Get paths
current_path = os.path.dirname(os.path.abspath(__file__))
input_data_path = os.path.join(current_path, "input_data")
progress_json_path = os.path.join(input_data_path, "fetch_progress.json")


def main():
    """Fetch combined foodcourt-restaurant data and save to JSON."""
    logging.info("=" * 60)
    logging.info("Fetching Combined Foodcourt-Restaurant Data")
    logging.info("=" * 60)
    
    # Connect to MongoDB
    try:
        client = MongoClient(local_mongo_uri)
        db = client.get_database(local_mongo_db)
        collection = db.get_collection(combined_foodcourt_coll)
        logging.info(f"Connected to MongoDB: {local_mongo_db}.{combined_foodcourt_coll}")
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        return
    
    try:
        # Fetch all documents
        logging.info("Fetching all foodcourt-restaurant data...")
        documents = list(collection.find({}))
        logging.info(f"Fetched {len(documents)} documents")
        
        if not documents:
            logging.warning("No documents found in collection")
            return
        
        # Build progress structure
        progress = {}
        
        for doc in documents:
            foodcourt_id = str(doc.get("_id", ""))
            city_id = str(doc.get("cityId", ""))
            restaurants = doc.get("restaurants", [])
            
            if not foodcourt_id:
                logging.warning("Skipping document without foodcourt ID")
                continue
            
            # Initialize foodcourt entry
            progress[foodcourt_id] = {
                "cityId": city_id,
                "restaurants": {}
            }
            
            # Add restaurant entries
            for restaurant_id in restaurants:
                restaurant_id_str = str(restaurant_id)
                progress[foodcourt_id]["restaurants"][restaurant_id_str] = {
                    "last_fetched_date": None,
                    "is_completed": False
                }
        
        # Ensure input_data directory exists
        os.makedirs(input_data_path, exist_ok=True)
        
        # Save to JSON file
        with open(progress_json_path, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
        
        logging.info(f"✓ Progress JSON saved to: {progress_json_path}")
        logging.info(f"✓ Total foodcourts: {len(progress)}")
        
        total_restaurants = sum(len(fc["restaurants"]) for fc in progress.values())
        logging.info(f"✓ Total restaurants: {total_restaurants}")
        
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        import traceback
        logging.debug(traceback.format_exc())
    finally:
        client.close()
        logging.info("MongoDB connection closed")


if __name__ == "__main__":
    main()
