"""
End-to-end pipeline to pull historical restaurant orders from MongoDB,
enrich them with derived fields, and persist both structured data and
summary statistics used downstream.

High-level flow
---------------
1. Read mongo + progress config files and connect to the `foodorder`
   collection.
2. Discover every foodcourt / restaurant pair from
   `input_data/FR_data.json`.
3. For each restaurant:
   - Run two aggregation pipelines (regular orders + preorders) to pull
     all completed orders after the optional `start_date`.
   - Explode every `data.items` entry to one row per menu item sold.
   - Aggregate by (foodcourt, restaurant, menu item, order, date) to get
     total quantity and revenue per item/day.
   - Reconstruct an IST-localized `date_IST` column, build per-item
     statistics, and append human-readable summaries to
     `input_data/restaurant_report.csv`.
   - Write the raw enriched dataset to Parquet under
     `input_data/fetched_data/{foodcourt}/{restaurant}.parquet` and mark
     the restaurant as complete in `FR_data.json`.
4. Once every restaurant has an initial snapshot, re-run in “Phase 2”
   mode to append incremental data beyond the previous ending date.

Columns produced in the Parquet output
--------------------------------------
- `foodcourtid` / `foodcourtname`: Mongo identifiers and display names.
- `restaurant` / `restaurantname`: Restaurant identifiers and names.
- `menuitemid` / `itemname`: Menu item identifier and name.
- `orderid`: Unique order reference in GoKhana.
- `date`: Sales date (string, yyyy-mm-dd, derived in IST).
- `date_IST`: pandas datetime localized to IST for precise ordering.
- `orderstatus`: Should always be `completed` (filter enforcement).
- `is_preorder`: Boolean flag; True if pulled from preorder pipeline.
- `total_count`: Quantity sold for that item/order/date grouping.
- `total_price`: Revenue for that grouping (totalprice fallback to price).
- `raw_datetime`: Internal helper used during processing (removed before save).
- Additional stats (outside the Parquet) include per-item totals,
  average per day metrics, order counts, and restaurant date ranges.
"""

import os
import sys
import json
import logging
import pandas as pd
import pytz
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId
from src.util.config_parser import ConfigManger
from src.util.progress_bar import ProgressBar

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
progress_json_path = os.path.join(input_data_path, "FR_data.json")
fetched_data_path = os.path.join(input_data_path, "fetched_data")
report_csv_path = os.path.join(input_data_path, "restaurant_report.csv")


# Progress bar functionality moved to pipeline_utils.ProgressBar


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

        # Initialize progress bar
        progress = None
        if total_records > 0:
            progress = ProgressBar(
                total=total_records,
                prefix=progress_prefix,
                suffix="records",
                length=40,
                show_elapsed=False
            )
        
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

                if progress:
                    progress.set_current(idx)
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


def calculate_restaurant_statistics(df, foodcourt_id, restaurant_id):
    """
    Calculate comprehensive statistics for a restaurant.
    
    Returns:
        list: List of dictionaries with statistics (one per item + summary rows)
    """
    if df.empty:
        return []
    
    # Overall statistics
    start_date = df['date'].min()
    end_date = df['date'].max()
    total_days = (end_date - start_date).days + 1
    days_with_data = df['date'].nunique()
    total_orders = df['orderid'].nunique()
    total_items_sold = df['total_count'].sum()
    total_revenue = df['total_price'].sum()
    
    # Calculate statistics per item
    item_stats = []
    
    for item_id in df['menuitemid'].unique():
        item_df = df[df['menuitemid'] == item_id]
        item_name = item_df['itemname'].iloc[0]
        
        # Item-specific statistics
        item_total_sold = item_df['total_count'].sum()
        item_total_revenue = item_df['total_price'].sum()
        item_first_sale = item_df['date'].min()
        item_last_sale = item_df['date'].max()
        item_days_with_sales = item_df['date'].nunique()
        item_total_days = (item_last_sale - item_first_sale).days + 1
        
        # Average per day (only for days with sales)
        item_avg_per_sale_day = item_total_sold / item_days_with_sales if item_days_with_sales > 0 else 0
        
        # Average per day (across all days from first to last sale)
        item_avg_per_all_days = item_total_sold / item_total_days if item_total_days > 0 else 0
        
        # Number of orders containing this item
        item_order_count = item_df['orderid'].nunique()
        
        item_stats.append({
            'foodcourt_id': foodcourt_id,
            'restaurant_id': restaurant_id,
            'restaurant_name': df['restaurantname'].iloc[0] if len(df) > 0 else '',
            'foodcourt_name': df['foodcourtname'].iloc[0] if len(df) > 0 else '',
            'item_id': item_id,
            'item_name': item_name,
            'item_first_sale_date': item_first_sale.strftime('%Y-%m-%d'),
            'item_last_sale_date': item_last_sale.strftime('%Y-%m-%d'),
            'item_total_instances_sold': int(item_total_sold),
            'item_total_revenue': float(item_total_revenue),
            'item_days_with_sales': int(item_days_with_sales),
            'item_total_days_range': int(item_total_days),
            'item_avg_per_sale_day': round(item_avg_per_sale_day, 2),
            'item_avg_per_all_days': round(item_avg_per_all_days, 2),
            'item_order_count': int(item_order_count),
            'restaurant_start_date': start_date.strftime('%Y-%m-%d'),
            'restaurant_end_date': end_date.strftime('%Y-%m-%d'),
            'restaurant_total_days': int(total_days),
            'restaurant_days_with_data': int(days_with_data),
            'restaurant_total_orders': int(total_orders),
            'restaurant_total_items_sold': int(total_items_sold),
            'restaurant_total_revenue': float(total_revenue),
            'statistic_type': 'item_detail'
        })
    
    # Add summary row at the beginning for this restaurant
    summary_row = {
        'foodcourt_id': foodcourt_id,
        'restaurant_id': restaurant_id,
        'restaurant_name': df['restaurantname'].iloc[0] if len(df) > 0 else '',
        'foodcourt_name': df['foodcourtname'].iloc[0] if len(df) > 0 else '',
        'item_id': 'SUMMARY',
        'item_name': 'RESTAURANT SUMMARY',
        'item_first_sale_date': start_date.strftime('%Y-%m-%d'),
        'item_last_sale_date': end_date.strftime('%Y-%m-%d'),
        'item_total_instances_sold': int(total_items_sold),
        'item_total_revenue': float(total_revenue),
        'item_days_with_sales': int(days_with_data),
        'item_total_days_range': int(total_days),
        'item_avg_per_sale_day': round(total_items_sold / days_with_data, 2) if days_with_data > 0 else 0,
        'item_avg_per_all_days': round(total_items_sold / total_days, 2) if total_days > 0 else 0,
        'item_order_count': int(total_orders),
        'restaurant_start_date': start_date.strftime('%Y-%m-%d'),
        'restaurant_end_date': end_date.strftime('%Y-%m-%d'),
        'restaurant_total_days': int(total_days),
        'restaurant_days_with_data': int(days_with_data),
        'restaurant_total_orders': int(total_orders),
        'restaurant_total_items_sold': int(total_items_sold),
        'restaurant_total_revenue': float(total_revenue),
        'statistic_type': 'restaurant_summary'
    }
    
    # Return summary first, then item details
    return [summary_row] + item_stats


def append_to_report(stats, foodcourt_id, restaurant_id, is_first_restaurant=False):
    """
    Append restaurant statistics to the report CSV file.
    
    Args:
        stats: List of statistics dictionaries
        foodcourt_id: Foodcourt ID
        restaurant_id: Restaurant ID
        is_first_restaurant: Whether this is the first restaurant (to write headers)
    """
    if not stats:
        return
    
    # Create DataFrame from stats
    stats_df = pd.DataFrame(stats)
    
    # Define column order
    column_order = [
        'foodcourt_id', 'foodcourt_name',
        'restaurant_id', 'restaurant_name',
        'statistic_type',
        'item_id', 'item_name',
        'restaurant_start_date', 'restaurant_end_date',
        'restaurant_total_days', 'restaurant_days_with_data',
        'restaurant_total_orders', 'restaurant_total_items_sold', 'restaurant_total_revenue',
        'item_first_sale_date', 'item_last_sale_date',
        'item_total_instances_sold', 'item_total_revenue',
        'item_days_with_sales', 'item_total_days_range',
        'item_avg_per_sale_day', 'item_avg_per_all_days',
        'item_order_count'
    ]
    
    # Reorder columns (only include columns that exist)
    existing_columns = [col for col in column_order if col in stats_df.columns]
    stats_df = stats_df[existing_columns]
    
    # If not first restaurant, add blank row before this restaurant's data
    if not is_first_restaurant:
        blank_row = pd.DataFrame([{col: '' for col in stats_df.columns}])
        stats_df = pd.concat([blank_row, stats_df], ignore_index=True)
    
    # Append to CSV (create new file if doesn't exist, append if exists)
    if is_first_restaurant or not os.path.exists(report_csv_path):
        stats_df.to_csv(report_csv_path, index=False, encoding='utf-8', mode='w')
    else:
        stats_df.to_csv(report_csv_path, index=False, encoding='utf-8', mode='a', header=False)


def main(prod_mode: bool = False):
    """Main function to fetch restaurant orders.

    When prod_mode is False, we assume fetched_data already contains the latest
    parquet exports and skip hitting MongoDB. This avoids unnecessary network
    calls when we only need to reuse existing data.
    """
    logging.info("=" * 60)
    logging.info("Fetching Restaurant Orders")
    logging.info("=" * 60)
    
    if not prod_mode:
        if not os.path.exists(fetched_data_path):
            logging.error("Cached fetched_data directory not found at %s", fetched_data_path)
            logging.error("Please run with --prod-mode once to generate the parquet files.")
            return
        
        foodcourt_dirs = [
            d for d in os.listdir(fetched_data_path)
            if os.path.isdir(os.path.join(fetched_data_path, d))
        ]
        logging.info(
            "Production mode disabled; using cached data from %s (foodcourts: %d)",
            fetched_data_path,
            len(foodcourt_dirs),
        )
        logging.info("Skipping MongoDB fetch since no new data should be pulled.")
        return
    
    # Load progress JSON
    if not os.path.exists(progress_json_path):
        logging.error(f"Progress JSON not found: {progress_json_path}")
        logging.error("Please run fetchData.py first to create the progress JSON")
        return
    
    with open(progress_json_path, 'r', encoding='utf-8') as f:
        progress = json.load(f)
    
    logging.info(f"Loaded progress for {len(progress)} foodcourts")
    
    # Connect to MongoDB
    try:
        collection, client = get_mongo_collection(
            connection_string,
            database_name,
            collection_name
        )
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        return
    
    try:
        # Track if this is the first restaurant with data (for report headers)
        is_first_restaurant = not os.path.exists(report_csv_path)
        total_foodcourts = len(progress)
        
        # Calculate total restaurants for progress tracking
        total_restaurants_all = sum(len(fc.get("restaurants", {})) for fc in progress.values())
        
        # Initialize counters
        processed_count = 0
        skipped_count = 0
        
        # PHASE 1: Check for incomplete restaurants
        incomplete_restaurants = []
        for foodcourt_id, foodcourt_data in progress.items():
            restaurants = foodcourt_data.get("restaurants", {})
            for restaurant_id, restaurant_data in restaurants.items():
                is_completed = restaurant_data.get("is_completed", False)
                if not is_completed:
                    incomplete_restaurants.append((foodcourt_id, restaurant_id, restaurant_data))
        
        # PHASE 1: Process incomplete restaurants
        if incomplete_restaurants:
            logging.info(f"\n{'='*60}")
            logging.info(f"PHASE 1: Processing {len(incomplete_restaurants)} incomplete restaurants")
            logging.info(f"{'='*60}")
            
            total_incomplete = len(incomplete_restaurants)
            phase1_processed = 0
            phase1_skipped = 0
            
            # Get foodcourt mapping for progress display
            foodcourt_mapping = {}
            for fc_idx, (fc_id, fc_data) in enumerate(progress.items(), 1):
                foodcourt_mapping[fc_id] = fc_idx
            
            for idx, (foodcourt_id, restaurant_id, restaurant_data) in enumerate(incomplete_restaurants, 1):
                # Get foodcourt index
                fc_idx = foodcourt_mapping.get(foodcourt_id, 1)
                
                # Fetch orders for this restaurant (fetch all data)
                progress_ctx = {
                    "fc_idx": fc_idx,
                    "total_fc": len(progress),
                    "rest_idx": idx,
                    "total_rest": total_incomplete,
                }
                df = fetch_restaurant_orders(
                    collection,
                    restaurant_id,
                    progress_ctx=progress_ctx
                )
                
                if df.empty:
                    phase1_skipped += 1
                    skipped_count += 1
                    # Mark as completed even if no data
                    if (foodcourt_id in progress and 
                        "restaurants" in progress[foodcourt_id] and 
                        restaurant_id in progress[foodcourt_id]["restaurants"]):
                        progress[foodcourt_id]["restaurants"][restaurant_id]["is_completed"] = True
                        with open(progress_json_path, 'w', encoding='utf-8') as f:
                            json.dump(progress, f, indent=2, ensure_ascii=False)
                    continue
                
                # Only create folder structure when we have data
                foodcourt_dir = os.path.join(fetched_data_path, foodcourt_id)
                os.makedirs(foodcourt_dir, exist_ok=True)
                
                # Save to Parquet
                parquet_path = os.path.join(foodcourt_dir, f"{restaurant_id}.parquet")
                df.to_parquet(parquet_path, index=False, engine='pyarrow')
                
                # Calculate date range using date_IST if available, otherwise use date
                date_col = 'date_IST' if 'date_IST' in df.columns else 'date'
                starting_date = df[date_col].min().strftime('%Y-%m-%d')
                ending_date = df[date_col].max().strftime('%Y-%m-%d')
                
                # Update progress JSON with starting_date, ending_date, and mark as completed
                if (foodcourt_id in progress and 
                    "restaurants" in progress[foodcourt_id] and 
                    restaurant_id in progress[foodcourt_id]["restaurants"]):
                    progress[foodcourt_id]["restaurants"][restaurant_id]["starting_date"] = starting_date
                    progress[foodcourt_id]["restaurants"][restaurant_id]["ending_date"] = ending_date
                    progress[foodcourt_id]["restaurants"][restaurant_id]["is_completed"] = True
                    # Save progress JSON
                    with open(progress_json_path, 'w', encoding='utf-8') as f:
                        json.dump(progress, f, indent=2, ensure_ascii=False)
                
                # Calculate statistics and append to report
                stats = calculate_restaurant_statistics(df, foodcourt_id, restaurant_id)
                append_to_report(stats, foodcourt_id, restaurant_id, is_first_restaurant)
                is_first_restaurant = False  # After first restaurant, always append
                
                phase1_processed += 1
                processed_count += 1
        
        # PHASE 2: Check if all restaurants are complete, then update with newest data
        all_complete = all(
            restaurant_data.get("is_completed", False)
            for foodcourt_data in progress.values()
            for restaurant_data in foodcourt_data.get("restaurants", {}).values()
        )
        
        if all_complete:
            logging.info(f"\n{'='*60}")
            logging.info(f"PHASE 2: All restaurants complete. Updating with newest data...")
            logging.info(f"{'='*60}")
            
            updated_count = 0
            phase2_skipped = 0
            
            # Iterate through all foodcourts and restaurants
            for fc_idx, (foodcourt_id, foodcourt_data) in enumerate(progress.items(), 1):
                restaurants = foodcourt_data.get("restaurants", {})
                total_restaurants = len(restaurants)
                
                # Process each restaurant
                for rest_idx, (restaurant_id, restaurant_data) in enumerate(restaurants.items(), 1):
                    # Get existing ending_date to fetch only new data
                    existing_ending_date = restaurant_data.get("ending_date")
                    start_date = None
                    if existing_ending_date:
                        # Fetch data after the existing ending_date
                        start_date = pd.to_datetime(existing_ending_date) + pd.Timedelta(days=1)
                    
                    # Fetch new orders for this restaurant
                    progress_ctx = {
                        "fc_idx": fc_idx,
                        "total_fc": total_foodcourts,
                        "rest_idx": rest_idx,
                        "total_rest": total_restaurants if total_restaurants else 1,
                    }
                    df_new = fetch_restaurant_orders(
                        collection,
                        restaurant_id,
                        progress_ctx=progress_ctx,
                        start_date=start_date
                    )
                    
                    if df_new.empty:
                        phase2_skipped += 1
                        skipped_count += 1
                        continue
                    
                    # Load existing data if parquet file exists
                    foodcourt_dir = os.path.join(fetched_data_path, foodcourt_id)
                    parquet_path = os.path.join(foodcourt_dir, f"{restaurant_id}.parquet")
                    
                    if os.path.exists(parquet_path):
                        df_existing = pd.read_parquet(parquet_path)
                        # Merge: combine and remove duplicates
                        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                        # Remove duplicates based on key columns
                        df_combined = df_combined.drop_duplicates(
                            subset=['orderid', 'menuitemid', 'date', 'is_preorder'],
                            keep='last'
                        )
                        df_combined = df_combined.sort_values('date').reset_index(drop=True)
                    else:
                        # No existing file, use only new data
                        df_combined = df_new
                    
                    # Save combined data to Parquet
                    os.makedirs(foodcourt_dir, exist_ok=True)
                    df_combined.to_parquet(parquet_path, index=False, engine='pyarrow')
                    
                    # Update ending_date (starting_date remains the same) using date_IST if available
                    date_col = 'date_IST' if 'date_IST' in df_combined.columns else 'date'
                    new_ending_date = df_combined[date_col].max().strftime('%Y-%m-%d')
                    existing_starting_date = restaurant_data.get("starting_date")
                    if not existing_starting_date:
                        existing_starting_date = df_combined[date_col].min().strftime('%Y-%m-%d')
                    
                    # Update progress JSON with new ending_date
                    if (foodcourt_id in progress and 
                        "restaurants" in progress[foodcourt_id] and 
                        restaurant_id in progress[foodcourt_id]["restaurants"]):
                        progress[foodcourt_id]["restaurants"][restaurant_id]["ending_date"] = new_ending_date
                        if not progress[foodcourt_id]["restaurants"][restaurant_id].get("starting_date"):
                            progress[foodcourt_id]["restaurants"][restaurant_id]["starting_date"] = existing_starting_date
                        # Save progress JSON
                        with open(progress_json_path, 'w', encoding='utf-8') as f:
                            json.dump(progress, f, indent=2, ensure_ascii=False)
                    
                    updated_count += 1
                    processed_count += 1
        else:
            pass  # Phase 2 skipped, no output needed
        
        # Final summary (minimal)
        print(f"\nComplete: {processed_count} restaurants processed, {skipped_count} skipped")
        
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        import traceback
        logging.debug(traceback.format_exc())
    finally:
        client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch restaurant order data")
    parser.add_argument(
        "--prod-mode",
        action="store_true",
        help="Enable production mode to fetch fresh data from MongoDB",
    )
    args = parser.parse_args()
    main(prod_mode=args.prod_mode)

