"""
Script to fetch foodorder data per restaurant from MongoDB.
Fetches orders filtered by restaurant_id (data.parentId) and orderstatus="completed".
Saves data to CSV files in input_data/food_court_data/{foodcourt_id}/{restaurant_id}.csv
Only creates folder structure when there's actual data to save.
"""

import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId
from config_parser import ConfigManger

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
food_court_data_path = os.path.join(input_data_path, "food_court_data")
report_csv_path = os.path.join(input_data_path, "restaurant_report.csv")


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


def fetch_restaurant_orders(collection, restaurant_id, include_preorders=True, progress_ctx=None):
    """
    Fetch foodorder data for a specific restaurant.
    Fetches ALL completed orders - uses placedtime for non-preorders and pickupdatetime for preorders as date.
    Uses index: data.parentId_1_createdAt_-1_data.orderstatus_1 for efficient querying.
    
    Args:
        collection: MongoDB collection
        restaurant_id: Restaurant ID (string or ObjectId)
        include_preorders: Whether to include preorder data
    
    Returns:
        pd.DataFrame: Order data with is_preorder column indicating if order is preorder
    """
    restaurant_objid = to_objectid_if_possible(restaurant_id)
    
    logging.info(f"Fetching orders for restaurant: {restaurant_id}")
    if include_preorders:
        logging.info("Including preorder data")
    
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
                    "is_preorder": {"$first": "$is_preorder"}
                }
            },
        ]
        
        results_non_preorder = list(collection.aggregate(pipeline_non_preorder, allowDiskUse=True))
        logging.info(f"Found {len(results_non_preorder)} non-preorder records")
        all_data.extend(results_non_preorder)
        
        # 2. Fetch preorder data (if include_preorders is True)
        # For preorders: use pickupdatetime as date field
        if include_preorders:
            filter_criteria_preorder = {
                "data.parentId": restaurant_objid,
                "data.orderstatus": "completed",
                "data.preorder": True
            }
            
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
                        "is_preorder": {"$first": "$is_preorder"}
                    }
                },
            ]
            
            results_preorder = list(collection.aggregate(pipeline_preorder, allowDiskUse=True))
            logging.info(f"Found {len(results_preorder)} preorder records")
            all_data.extend(results_preorder)
        
        logging.info(f"Total found {len(all_data)} order records")
        
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

        for idx, record in enumerate(all_data, 1):
            try:
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
                }
                processed_results.append(processed_record)

                if progress_ctx and total_records:
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
            # Sort by date
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
        logging.info(f"✓ Created report file: {report_csv_path}")
    else:
        stats_df.to_csv(report_csv_path, index=False, encoding='utf-8', mode='a', header=False)
        logging.info(f"✓ Appended to report: {len(stats)} rows")
    
    logging.info(f"  Report now contains statistics for restaurant: {restaurant_id}")


def main():
    """Main function to fetch restaurant orders."""
    logging.info("=" * 60)
    logging.info("Fetching Restaurant Orders")
    logging.info("=" * 60)
    
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
        
        # Iterate through foodcourts and restaurants
        processed_count = 0
        skipped_count = 0
        for fc_idx, (foodcourt_id, foodcourt_data) in enumerate(progress.items(), 1):
            restaurants = foodcourt_data.get("restaurants", {})
            city_id = foodcourt_data.get("cityId", "")
            total_restaurants = len(restaurants)
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing Foodcourt: {foodcourt_id}")
            logging.info(f"City ID: {city_id}")
            logging.info(f"Restaurants: {len(restaurants)}")
            logging.info(f"{'='*60}")
            
            # Process each restaurant
            for rest_idx, (restaurant_id, restaurant_data) in enumerate(restaurants.items(), 1):
                logging.info(f"\nProcessing Restaurant: {restaurant_id}")
                logging.info(
                    f"[Foodcourt {fc_idx}/{total_foodcourts}] "
                    f"[Restaurant {rest_idx}/{total_restaurants if total_restaurants else 1}]"
                )
                
                # Fetch orders for this restaurant
                progress_ctx = {
                    "fc_idx": fc_idx,
                    "total_fc": total_foodcourts,
                    "rest_idx": rest_idx,
                    "total_rest": total_restaurants if total_restaurants else 1,
                }
                df = fetch_restaurant_orders(
                    collection,
                    restaurant_id,
                    progress_ctx=progress_ctx
                )
                
                if df.empty:
                    logging.warning(f"No orders found for restaurant {restaurant_id}")
                    skipped_count += 1
                    # Skip to next restaurant if no data
                    continue
                
                # Only create folder structure when we have data
                foodcourt_dir = os.path.join(food_court_data_path, foodcourt_id)
                os.makedirs(foodcourt_dir, exist_ok=True)
                
                # Save to CSV
                csv_path = os.path.join(foodcourt_dir, f"{restaurant_id}.csv")
                df.to_csv(csv_path, index=False, encoding='utf-8')
                logging.info(f"✓ Saved {len(df):,} records to {csv_path}")
                logging.info(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
                
                # Calculate statistics and append to report
                logging.info("  Calculating statistics and updating report...")
                stats = calculate_restaurant_statistics(df, foodcourt_id, restaurant_id)
                append_to_report(stats, foodcourt_id, restaurant_id, is_first_restaurant)
                is_first_restaurant = False  # After first restaurant, always append
                
                processed_count += 1
        
        # Final summary
        logging.info("\n" + "="*60)
        logging.info("PROCESSING COMPLETE")
        logging.info("="*60)
        logging.info(f"Total foodcourts processed: {total_foodcourts}")
        logging.info(f"Total restaurants processed: {processed_count}")
        logging.info(f"Total restaurants skipped (no data): {skipped_count}")
        logging.info(f"Total restaurants in progress file: {total_restaurants_all}")
        logging.info(f"✓ Report saved to: {report_csv_path}")
        
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        import traceback
        logging.debug(traceback.format_exc())
    finally:
        client.close()
        logging.info("MongoDB connection closed")


if __name__ == "__main__":
    main()

