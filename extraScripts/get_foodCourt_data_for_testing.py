"""
Testing script to fetch foodcourt data for specific foodcourts.
Works with a hardcoded list of foodcourt IDs and uses fetch_progress_test.json.

Workflow:
1. For each foodcourt ID in the hardcoded list:
   - Check if it exists in fetch_progress_test.json
   - If not, copy the foodcourt object from fetch_progress.json
   - Check if all restaurants are completed
   - If all completed: copy from fetched_data to Model Training/food_court_data
   - If not all completed: fetch from MongoDB and save to Model Training/food_court_data
2. Uses input_data/Model Training/food_court_data/ as output directory
"""

import os
import sys
import json
import logging
import shutil
import pandas as pd
import pytz
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId
from src.util.config_parser import ConfigManger

# Hardcoded list of foodcourt IDs for testing
TEST_FOODCOURT_IDS = [

    "61e39f4b2acc405ba5c12202",
    "66751bbe191e5a001af454d4",
    "62720dcd1410b643ccc5eaf0",
    "60f172e2d1b6e328e744cf65",
    "620c8839d1702023497f3e64",
    "61862d3cc680d0688d9d3139",
    "61529bb6364f2332c0fe8b1c",
    "6272117f6ea9037357d4559e",
    "5f3652cde091b809e2467d96",
    "6051d1f64ec6952ca5b45dba",
    "5f338dce8f277f4c2f4ac99f",
    "63d74931822179001b3c8320",
    "655b3bdca4a02d001bcb78c3",
    "5f36401ce091b809e2467cf4",
    "61860f8c2d415a4a5c52e822",
    "618607cac665f41f98e72227",
    "66ffbf790724f300133f7c22",
    "627655b12847414390c38539",
    "62b976e8a94ae70e02f47540",
    "6288ac6574969f0ee1463b09",
    "61cd6892f1c6956f2d490104",
    "68ff220d34926e00133858d8",
    "621243047b7da3595490b804",
    "6368b12ed2915c10f4b68571",
    "62122f42e9692355fc21cd37",
    "62122c7a86769559bcb3861c",
    "628f0b429f179d6efab94aa1",
    "6471cfa0c3006d001707aee4",
    "6344e94b2db92732f3849cd4",
    "63245d4de5aff648ca6109f0",
    "633ac3e6f502b233052bd11f",
    "63469408bcf1a078aa63687c",
    "63245e835129027dc5432fb2",
    "679f3ef268109c00138554a7",
    "63245df1f909bf484ba32e5f",
    "6874f9e161afd1001ab2df38",
    "64e33cca77a4b10017691075",
    "65b48ebd8d7d8f001be02f79",
    "64deefcd895a9b001b1517a9",
    "66fb796c232d1500137baf13",
    "651ba1552d0b1d001b6c2e53",
    "667da1a6b159bc001a9160b6",
    "655f2bf0f2c700001bc727ac",
    "65b0f146a7f88a001b0aa203",
    "6895bf831d7a46001abf7996",
    "688074e938ecce001a806923",
    "688075110a2413001abceac1",
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load config
config = ConfigManger()
cloud_config = config.read_config(type="config")

# MongoDB connection for foodorder collection
connection_string = cloud_config["mongodb"]["connection_string"]
database_name = cloud_config["mongodb"]["db_name"]
collection_name = cloud_config["mongodb"]["collection_name"]

# Get paths
current_path = os.path.dirname(os.path.abspath(__file__))
input_data_path = os.path.join(current_path, "input_data")
progress_json_path = os.path.join(input_data_path, "fetch_progress.json")
progress_test_json_path = os.path.join(input_data_path, "fetch_progress_test.json")
fetched_data_path = os.path.join(input_data_path, "fetched_data")
model_training_path = os.path.join(input_data_path, "Model Training")
food_court_data_path = os.path.join(model_training_path, "food_court_data")
os.makedirs(food_court_data_path, exist_ok=True)


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=30, fill='█'):
    """Print a progress bar to the console."""
    if total == 0:
        return
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration >= total:
        sys.stdout.write('\n')


def get_mongo_collection(connection_string, database_nm, collection_nm):
    """Get MongoDB collection."""
    try:
        client = MongoClient(connection_string)
        db = client.get_database(database_nm)
        collection = db.get_collection(collection_nm)
        logging.info(f"Connected to MongoDB: {database_nm}.{collection_nm}")
        return collection, client
    except Exception as e:
        logging.error(f"Error connecting to MongoDB: {e}")
        raise


def to_objectid_if_possible(val):
    """Convert string to ObjectId if possible."""
    if isinstance(val, ObjectId):
        return val
    if isinstance(val, str) and len(val) == 24:
        try:
            return ObjectId(val)
        except Exception:
            return val
    return val


def fetch_restaurant_orders(collection, restaurant_id, include_preorders=True, progress_ctx=None, start_date=None):
    """
    Fetch foodorder data for a specific restaurant.
    Fetches ALL completed orders - uses placedtime for non-preorders and pickupdatetime for preorders as date.
    Uses index: data.parentId_1_createdAt_-1_data.orderstatus_1 for efficient querying.
    
    Args:
        collection: MongoDB collection
        restaurant_id: Restaurant ID (string or ObjectId)
        include_preorders: Whether to include preorder data
        progress_ctx: Progress context for progress bar
        start_date: Optional datetime or date string. If provided, only fetches data after this date.
    
    Returns:
        pd.DataFrame: Order data with is_preorder column indicating if order is preorder
    """
    restaurant_objid = to_objectid_if_possible(restaurant_id)
    
    if start_date:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
    
    all_data = []
    
    try:
        # 1. Fetch non-preorder data
        # For non-preorders: use placedtime as date field
        filter_criteria_non_preorder = {
            "data.parentId": restaurant_objid,
            "data.orderstatus": "completed",
            "$or": [
                {"data.preorder": {"$exists": False}},
                {"data.preorder": {"$exists": True, "$in": [False, None]}}
            ]
        }
        
        # Add date filter if start_date is provided
        if start_date:
            # Convert to datetime and then to ISO format for MongoDB
            if isinstance(start_date, pd.Timestamp):
                start_datetime = start_date.to_pydatetime()
            else:
                start_datetime = pd.to_datetime(start_date).to_pydatetime()
            # MongoDB expects datetime in UTC, but we need to account for timezone
            # Since we're filtering by date (not time), we want to include the entire start_date
            # So we set time to 00:00:00 in the timezone (+05:30)
            ist = pytz.timezone('Asia/Kolkata')
            start_datetime_ist = ist.localize(datetime.combine(start_datetime.date(), datetime.min.time()))
            start_datetime_utc = start_datetime_ist.astimezone(pytz.UTC)
            filter_criteria_non_preorder["data.placedtime"] = {"$gte": start_datetime_utc}
        
        pipeline_non_preorder = [
            {"$match": filter_criteria_non_preorder},
            {"$unwind": "$data.items"},
            {
                "$project": {
                    "_id": 1,
                    "foodcourtname": "$data.foodcourtname",
                    "foodcourt": "$data.foodcourt",
                    "restaurant": "$data.items.restaurant",
                    "restaurantname": "$data.items.restaurantname",
                    "restaurantmenuitem": "$data.items.restaurantmenuitem",
                    "orderid": "$data.orderid",
                    "itemname": "$data.items.menuitemname",
                    "count": "$data.items.count",
                    "itemprice": {
                        "$ifNull": [
                            "$data.items.totalprice",
                            "$data.items.itemprice"
                        ]
                    },
                    "date": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": {"$toDate": "$data.placedtime"},
                            "timezone": "+05:30"
                        }
                    },
                    "placedtime_raw": "$data.placedtime",
                    "orderstatus": "$data.orderstatus",
                    "is_preorder": {"$literal": False}
                }
            },
            {
                "$group": {
                    "_id": {
                        "foodcourt": "$foodcourt",
                        "foodcourtname": "$foodcourtname",
                        "restaurant": "$restaurant",
                        "restaurantname": "$restaurantname",
                        "restaurantmenuitem": "$restaurantmenuitem",
                        "itemname": "$itemname",
                        "orderid": "$orderid",
                        "date": "$date",
                        "orderstatus": "$orderstatus"
                    },
                    "total_count": {"$sum": "$count"},
                    "total_price": {"$sum": "$itemprice"},
                    "is_preorder": {"$first": "$is_preorder"},
                    "placedtime_raw": {"$first": "$placedtime_raw"}
                }
            },
        ]
        
        results_non_preorder = list(collection.aggregate(pipeline_non_preorder, allowDiskUse=True))
        all_data.extend(results_non_preorder)
        
        # 2. Fetch preorder data (if include_preorders is True)
        # For preorders: use pickupdatetime as date field
        if include_preorders:
            filter_criteria_preorder = {
                "data.parentId": restaurant_objid,
                "data.orderstatus": "completed",
                "data.preorder": True
            }
            
            # Add date filter if start_date is provided
            if start_date:
                ist = pytz.timezone('Asia/Kolkata')
                if isinstance(start_date, pd.Timestamp):
                    start_datetime = start_date.to_pydatetime()
                else:
                    start_datetime = pd.to_datetime(start_date).to_pydatetime()
                start_datetime_ist = ist.localize(datetime.combine(start_datetime.date(), datetime.min.time()))
                start_datetime_utc = start_datetime_ist.astimezone(pytz.UTC)
                filter_criteria_preorder["data.pickupdatetime"] = {"$gte": start_datetime_utc}
            
            pipeline_preorder = [
                {"$match": filter_criteria_preorder},
                {"$unwind": "$data.items"},
                {
                    "$project": {
                        "_id": 1,
                        "foodcourtname": "$data.foodcourtname",
                        "foodcourt": "$data.foodcourt",
                        "restaurant": "$data.items.restaurant",
                        "restaurantname": "$data.items.restaurantname",
                        "restaurantmenuitem": "$data.items.restaurantmenuitem",
                        "orderid": "$data.orderid",
                        "itemname": "$data.items.menuitemname",
                        "count": "$data.items.count",
                        "itemprice": {
                            "$ifNull": [
                                "$data.items.totalprice",
                                "$data.items.itemprice"
                            ]
                        },
                        "date": {
                            "$dateToString": {
                                "format": "%Y-%m-%d",
                                "date": {"$toDate": "$data.pickupdatetime"},
                                "timezone": "+05:30"
                            }
                        },
                        "pickupdatetime_raw": "$data.pickupdatetime",
                        "orderstatus": "$data.orderstatus",
                        "is_preorder": {"$literal": True}
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "foodcourt": "$foodcourt",
                            "foodcourtname": "$foodcourtname",
                            "restaurant": "$restaurant",
                            "restaurantname": "$restaurantname",
                            "restaurantmenuitem": "$restaurantmenuitem",
                            "itemname": "$itemname",
                            "orderid": "$orderid",
                            "date": "$date",
                            "orderstatus": "$orderstatus"
                        },
                        "total_count": {"$sum": "$count"},
                        "total_price": {"$sum": "$itemprice"},
                        "is_preorder": {"$first": "$is_preorder"},
                        "pickupdatetime_raw": {"$first": "$pickupdatetime_raw"}
                    }
                },
            ]
            
            results_preorder = list(collection.aggregate(pipeline_preorder, allowDiskUse=True))
            all_data.extend(results_preorder)
        
        if not all_data:
            return pd.DataFrame()
        
        # Process results into DataFrame
        processed_results = []
        total_records = len(all_data)
        progress_prefix = ""
        if progress_ctx:
            progress_prefix = (
                f"FC {progress_ctx['fc_idx']}/{progress_ctx['total_fc']} | "
                f"Rest {progress_ctx['rest_idx']}/{progress_ctx['total_rest']} | Copying"
            )
        else:
            # Fallback if no progress context
            progress_prefix = "Copying"

        for idx, record in enumerate(all_data, 1):
            try:
                # Get raw datetime (either placedtime or pickupdatetime)
                raw_datetime = None
                if record.get("placedtime_raw") is not None:
                    raw_datetime = record["placedtime_raw"]
                elif record.get("pickupdatetime_raw") is not None:
                    raw_datetime = record["pickupdatetime_raw"]
                
                processed_record = {
                    "foodcourtid": str(record["_id"].get("foodcourt", "")),
                    "foodcourtname": record["_id"].get("foodcourtname", ""),
                    "restaurant": str(record["_id"].get("restaurant", "")),
                    "restaurantname": record["_id"].get("restaurantname", ""),
                    "menuitemid": str(record["_id"].get("restaurantmenuitem", "")),
                    "itemname": record["_id"].get("itemname", ""),
                    "orderid": record["_id"].get("orderid", ""),
                    "date": record["_id"].get("date", ""),
                    "orderstatus": record["_id"].get("orderstatus", ""),
                    "is_preorder": record.get("is_preorder", False),
                    "total_count": record.get("total_count", 0),
                    "total_price": record.get("total_price", 0.0),
                    "raw_datetime": raw_datetime,  # Store raw datetime for IST conversion
                }
                processed_results.append(processed_record)

                if total_records > 0:
                    print_progress_bar(
                        idx,
                        total_records,
                        prefix=progress_prefix,
                        suffix=f"{idx}/{total_records} records",
                        length=40
                    )
            except Exception as e:
                logging.warning(f"Skipping record due to error: {e}")
                continue
        
        if processed_results:
            df = pd.DataFrame(processed_results)
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Create date_IST column: convert raw datetime to IST, then extract date
            ist = pytz.timezone('Asia/Kolkata')
            date_ist_list = []
            
            for idx, raw_dt in enumerate(df['raw_datetime']):
                if pd.isna(raw_dt) or raw_dt is None:
                    # Fallback to existing date if raw datetime is missing
                    date_ist_list.append(df.iloc[idx]['date'].date())
                else:
                    try:
                        # Handle MongoDB datetime objects (BSON datetime)
                        if hasattr(raw_dt, 'to_pydatetime'):
                            # It's a pandas Timestamp
                            dt = raw_dt
                        elif isinstance(raw_dt, datetime):
                            # It's a Python datetime
                            dt = pd.Timestamp(raw_dt)
                        else:
                            # Convert string or other format to datetime
                            dt = pd.to_datetime(raw_dt)
                        
                        # If datetime is timezone-naive, assume it's UTC
                        if dt.tz is None:
                            dt = dt.tz_localize(pytz.UTC)
                        
                        # Convert to IST
                        dt_ist = dt.astimezone(ist)
                        
                        # Extract date (date only, no time)
                        date_ist_list.append(dt_ist.date())
                    except Exception as e:
                        logging.warning(f"Error converting datetime to IST for record {idx}: {e}, using fallback date")
                        # Fallback to existing date
                        date_ist_list.append(df.iloc[idx]['date'].date())
            
            # Convert date_IST list to datetime and add to dataframe
            df['date_IST'] = pd.to_datetime(date_ist_list)
            
            # Drop the temporary raw_datetime column
            df = df.drop(columns=['raw_datetime'])
            
            # Sort by date_IST (primary) or date (fallback)
            if 'date_IST' in df.columns:
                df = df.sort_values('date_IST').reset_index(drop=True)
            else:
                df = df.sort_values('date').reset_index(drop=True)
            
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error fetching orders: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return pd.DataFrame()


def copy_foodcourt_folder(source_fc_id, dest_fc_id=None):
    """
    Copy entire foodcourt folder from fetched_data to food_court_data.
    
    Args:
        source_fc_id: Foodcourt ID in fetched_data
        dest_fc_id: Optional destination foodcourt ID (defaults to source_fc_id)
    """
    if dest_fc_id is None:
        dest_fc_id = source_fc_id
    
    source_dir = os.path.join(fetched_data_path, source_fc_id)
    dest_dir = os.path.join(food_court_data_path, dest_fc_id)
    
    if not os.path.exists(source_dir):
        logging.warning(f"Source directory not found: {source_dir}")
        return False
    
    try:
        # Remove destination if it exists
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        
        # Copy entire folder
        shutil.copytree(source_dir, dest_dir)
        logging.info(f"Copied folder from {source_dir} to {dest_dir}")
        return True
    except Exception as e:
        logging.error(f"Error copying folder {source_fc_id}: {e}")
        return False


def are_all_restaurants_completed(foodcourt_data):
    """Check if all restaurants in a foodcourt are completed."""
    restaurants = foodcourt_data.get("restaurants", {})
    if not restaurants:
        return False
    return all(rest.get("is_completed", False) for rest in restaurants.values())


def main():
    """Main function to fetch foodcourt data for testing."""
    logging.info("=" * 60)
    logging.info("Fetching Foodcourt Data for Testing")
    logging.info("=" * 60)
    
    # Load main progress JSON
    if not os.path.exists(progress_json_path):
        logging.error(f"Main progress JSON not found: {progress_json_path}")
        return
    
    with open(progress_json_path, 'r', encoding='utf-8') as f:
        main_progress = json.load(f)
    
    # Load or create test progress JSON
    if os.path.exists(progress_test_json_path):
        with open(progress_test_json_path, 'r', encoding='utf-8') as f:
            test_progress = json.load(f)
    else:
        test_progress = {}
    
    # Connect to MongoDB (will be used if needed)
    collection = None
    client = None
    
    try:
        # Process each foodcourt ID in the test list
        for fc_idx, foodcourt_id in enumerate(TEST_FOODCOURT_IDS, 1):
            logging.info(f"\n[{fc_idx}/{len(TEST_FOODCOURT_IDS)}] Processing foodcourt: {foodcourt_id}")
            
            # Check if foodcourt exists in test progress
            if foodcourt_id in test_progress:
                logging.info(f"Foodcourt {foodcourt_id} already exists in test progress")
                foodcourt_data = test_progress[foodcourt_id]
            else:
                # Copy from main progress if it exists
                if foodcourt_id in main_progress:
                    logging.info(f"Copying foodcourt {foodcourt_id} from main progress to test progress")
                    test_progress[foodcourt_id] = main_progress[foodcourt_id].copy()
                    foodcourt_data = test_progress[foodcourt_id]
                    # Save test progress
                    with open(progress_test_json_path, 'w', encoding='utf-8') as f:
                        json.dump(test_progress, f, indent=2, ensure_ascii=False)
                else:
                    logging.warning(f"Foodcourt {foodcourt_id} not found in main progress, skipping")
                    continue
            
            # Check if all restaurants are completed
            if are_all_restaurants_completed(foodcourt_data):
                logging.info(f"All restaurants completed for {foodcourt_id}, copying from fetched_data")
                # Copy entire folder
                if copy_foodcourt_folder(foodcourt_id):
                    logging.info(f"Successfully copied foodcourt {foodcourt_id}")
                else:
                    logging.warning(f"Failed to copy foodcourt {foodcourt_id}, will fetch from MongoDB")
                    # Fall through to fetch from MongoDB
                    if collection is None:
                        collection, client = get_mongo_collection(
                            connection_string,
                            database_name,
                            collection_name
                        )
                    # Fetch all restaurants for this foodcourt
                    restaurants = foodcourt_data.get("restaurants", {})
                    for rest_idx, (restaurant_id, restaurant_data) in enumerate(restaurants.items(), 1):
                        logging.info(f"Fetching restaurant {restaurant_id} ({rest_idx}/{len(restaurants)})")
                        progress_ctx = {
                            "fc_idx": fc_idx,
                            "total_fc": len(TEST_FOODCOURT_IDS),
                            "rest_idx": rest_idx,
                            "total_rest": len(restaurants),
                        }
                        df = fetch_restaurant_orders(collection, restaurant_id, progress_ctx=progress_ctx)
                        
                        if not df.empty:
                            foodcourt_dir = os.path.join(food_court_data_path, foodcourt_id)
                            os.makedirs(foodcourt_dir, exist_ok=True)
                            parquet_path = os.path.join(foodcourt_dir, f"{restaurant_id}.parquet")
                            df.to_parquet(parquet_path, index=False, engine='pyarrow')
                            logging.info(f"Saved {restaurant_id} to {parquet_path}")
            else:
                logging.info(f"Not all restaurants completed for {foodcourt_id}, fetching from MongoDB")
                # Fetch from MongoDB
                if collection is None:
                    collection, client = get_mongo_collection(
                        connection_string,
                        database_name,
                        collection_name
                    )
                
                restaurants = foodcourt_data.get("restaurants", {})
                for rest_idx, (restaurant_id, restaurant_data) in enumerate(restaurants.items(), 1):
                    is_completed = restaurant_data.get("is_completed", False)
                    if is_completed:
                        # Try to copy from fetched_data first
                        source_parquet = os.path.join(fetched_data_path, foodcourt_id, f"{restaurant_id}.parquet")
                        dest_parquet = os.path.join(food_court_data_path, foodcourt_id, f"{restaurant_id}.parquet")
                        if os.path.exists(source_parquet):
                            os.makedirs(os.path.dirname(dest_parquet), exist_ok=True)
                            shutil.copy2(source_parquet, dest_parquet)
                            logging.info(f"Copied {restaurant_id} from fetched_data")
                            continue
                    
                    logging.info(f"Fetching restaurant {restaurant_id} ({rest_idx}/{len(restaurants)})")
                    progress_ctx = {
                        "fc_idx": fc_idx,
                        "total_fc": len(TEST_FOODCOURT_IDS),
                        "rest_idx": rest_idx,
                        "total_rest": len(restaurants),
                    }
                    df = fetch_restaurant_orders(collection, restaurant_id, progress_ctx=progress_ctx)
                    
                    if not df.empty:
                        foodcourt_dir = os.path.join(food_court_data_path, foodcourt_id)
                        os.makedirs(foodcourt_dir, exist_ok=True)
                        parquet_path = os.path.join(foodcourt_dir, f"{restaurant_id}.parquet")
                        df.to_parquet(parquet_path, index=False, engine='pyarrow')
                        
                        # Update test progress
                        date_col = 'date_IST' if 'date_IST' in df.columns else 'date'
                        starting_date = df[date_col].min().strftime('%Y-%m-%d')
                        ending_date = df[date_col].max().strftime('%Y-%m-%d')
                        test_progress[foodcourt_id]["restaurants"][restaurant_id]["starting_date"] = starting_date
                        test_progress[foodcourt_id]["restaurants"][restaurant_id]["ending_date"] = ending_date
                        test_progress[foodcourt_id]["restaurants"][restaurant_id]["is_completed"] = True
                        
                        logging.info(f"Saved {restaurant_id} to {parquet_path}")
                
                # Save updated test progress
                with open(progress_test_json_path, 'w', encoding='utf-8') as f:
                    json.dump(test_progress, f, indent=2, ensure_ascii=False)
        
        logging.info("\n" + "=" * 60)
        logging.info("Processing complete!")
        logging.info("=" * 60)
        
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        import traceback
        logging.debug(traceback.format_exc())
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    main()

