"""
Goal
----
Take raw restaurant order exports and enrich them into a modeling-ready dataset
where each (foodcourtid, restaurant, menuitemid, date) row contains canonical
metadata, total units sold, weather/holiday context, and gap-filled date ranges.

End-to-End Workflow
-------------------
Data Sources:
- FR_data.json: Contains foodcourt and restaurant metadata, including city IDs for weather joins
- fetched_data/: Contains raw restaurant order data as Parquet files organized by foodcourt_id
  Format: `input_data/fetched_data/{foodcourt_id}/{restaurant_id}.parquet`
- retrain.json: Controls which foodcourts/restaurants/items to process
  - If enrich_data list is non-empty: ONLY process entries specified in retrain.json (force retrain)
  - If enrich_data list is empty but other steps have entries: Skip enrichment step entirely
  - If retrain.json is completely empty: Process all items missing enriched data (check previous step)

Processing Flow:
1. Discover Work: Read `FR_data.json` to list every foodcourt and its restaurants along with
   city IDs used later for weather joins. Count foodcourts/restaurants in fetched_data/ and
   existing enrich_data/ outputs to determine what needs processing.

2. Filter by retrain.json: 
   - If enrich_data list in retrain.json is non-empty: Extract foodcourts/restaurants/items
     from retrain.json and ONLY process those (force retrain, replace existing files).
   - If enrich_data list is empty but retrain.json has entries for other steps: Skip this step.
   - If retrain.json is completely empty: Process all foodcourts/restaurants that don't have
     enriched data yet (check output_data/enrich_data/{pipeline_type}/).

3. Load Raw Orders: For each restaurant that has a Parquet file in `fetched_data/`, load the
   rows and prepare for enrichment. Skip restaurants without data in fetched_data/.
   Format: `input_data/fetched_data/{foodcourt_id}/{restaurant_id}.parquet`

4. Fetch Metadata: Query the `entityrecord` collection (with a local fallback) to build a
   normalized item-name lookup that maps raw names to canonical `menuitemid`, `isVeg`, `isSpicy`,
   and the original entityrecord name.

5. Enrich Orders: Normalize item names in the raw data, attach metadata, compute a unit `price`
   (total_price / total_count), and keep track of enrichment progress.

6. Aggregate by IDs: Group strictly on IDs (`foodcourtid`, `restaurant`, `menuitemid`, `date`)
   so each item/day pair has one row with summed counts, regardless of mismatched restaurant/
   foodcourt names in the raw export.

7. Canonicalize Names & Build Date Grid: Collapse metadata to a single canonical row per unique
   ID combo, then cross-join it with the continuous date range spanning the first to last observed
   sale; this ensures downstream models see a dense daily series even on zero-sale days.

8. Merge Metrics: Join the aggregated counts back onto the dense grid, fill missing count values
   with zeros, and forward/backfill metadata columns where necessary so each date carries the
   same canonical names and flags.

9. Add External Signals: Using the restaurant's `cityId` from FR_data.json, fetch weather rows
   from MySQL tables (`city_<cityId>`), merge them with the grid, join holiday metadata from
   `input_data/holidays_table.csv`, and compute weekday indicator columns.

10. Finalize Schema: Drop intermediate holiday columns, order the final fields, and split by item.
    Save each item as a separate Excel file to:
    `output_data/enrich_data/{pipeline_type}/{foodcourt_id}/{fc_id}_{rest_id}_{item_id}_{item_name}_enrich_data.csv`

11. Update File Locator: Add/update entries in file_locator.csv for all processed items. If a
    file already exists, it will be replaced (overwritten) and the locator entry will be updated.

12. Debug Support: When `TESTING` is True, write `_debug` CSV snapshots for every key step
    (enriched, aggregated, grid, weather, final) to simplify auditing.
"""

import importlib
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import pandas as pd
from bson import ObjectId
from pymongo import MongoClient
import re

mysql_connector = None
local_mongo_client = None

from src.util.config_parser import ConfigManger
from src.util.progress_bar import ProgressBar

# Keywords for filtering out beverage/MRP items
EXCLUDE_KEYWORDS = [
    "water",
    "soft drink",
    "cold drink",
    "coke",
    "pepsi",
    "soda",
    "coffee",
    "tea",
    "drink",
    "juice",
    "beverage",
    "shake",
    "mocktail",
    "latte",
    "espresso",
    "mocha",
    "cappuccino",
    "frappe",
    "mrp",
]
EXCLUDE_PATTERN = re.compile("|".join(EXCLUDE_KEYWORDS), re.IGNORECASE)


def is_excluded_item(name) -> bool:
    """Return True if the item name matches beverage or MRP keywords."""
    if not isinstance(name, str):
        return False
    return bool(EXCLUDE_PATTERN.search(name))


def filter_excluded_items(df: pd.DataFrame):
    """Remove rows where itemname matches beverage/MRP keywords."""
    if "itemname" not in df.columns:
        return df, pd.DataFrame()
    mask = df["itemname"].astype(str).apply(lambda x: not is_excluded_item(x))
    removed = df[~mask].copy()
    # Get unique items that were removed
    if "menuitemid" in removed.columns and "itemname" in removed.columns:
        removed_unique = removed[["menuitemid", "itemname"]].drop_duplicates()
    elif "itemname" in removed.columns:
        removed_unique = removed[["itemname"]].drop_duplicates()
        if "menuitemid" in df.columns:
            # Try to get menuitemid from original df
            removed_unique = removed[["itemname"]].merge(
                df[["itemname", "menuitemid"]].drop_duplicates(),
                on="itemname",
                how="left"
            )
    else:
        removed_unique = pd.DataFrame()
    return df[mask], removed_unique

# Feature flag to dump intermediate steps for inspection
TESTING = False  # Disabled to improve performance - only final enriched files are saved

# Configure logging once for the script.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)



def align_dates_to_ist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use the IST-normalized date column as the canonical date.

    - If `date_IST` exists, convert it to datetime, drop the original `date`,
      and write a new `date` column sourced from `date_IST`.
    - Otherwise, ensure `date` is parsed to datetime.
    """
    if df.empty:
        return df

    working_df = df.copy()

    if "date_IST" in working_df.columns:
        ist_series = pd.to_datetime(working_df["date_IST"], errors="coerce")
        ist_series = ist_series.dt.tz_localize(None)  # ensure timezone-naive
        working_df["date"] = ist_series
        working_df = working_df.drop(columns=["date_IST"])
    elif "date" in working_df.columns:
        working_df["date"] = pd.to_datetime(working_df["date"], errors="coerce")

    return working_df


def to_objectid_if_possible(value):
    """Convert a 24-char string to ObjectId if possible."""
    if isinstance(value, ObjectId):
        return value
    if isinstance(value, str) and len(value) == 24:
        try:
            return ObjectId(value)
        except Exception:
            return value
    return value


def normalize_item_name(name: Any) -> str:
    """Normalize item names for matching."""
    if name is None:
        return ""
    if isinstance(name, str):
        return name.strip().lower()
    # Handle pandas NA / numbers etc.
    if pd.isna(name):
        return ""
    return str(name).strip().lower()


def get_mongo_collection(connection_uri: str, db_name: str, collection_name: str):
    """Establish Mongo connection and return collection + client."""
    client = MongoClient(connection_uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    db = client.get_database(db_name)
    collection = db.get_collection(collection_name)
    from src.util.pipeline_utils import get_pipeline_logger
    logger = get_pipeline_logger()
    logger.log_connection_status(f"MongoDB {db_name}.{collection_name}", "connected")
    logging.info("✅ MongoDB %s.%s", db_name, collection_name)
    return collection, client


def get_local_mongo_collection(local_cfg, collection_name: str):
    """Fallback to local Mongo if cloud entityrecord is unavailable."""
    global local_mongo_client
    try:
        if local_mongo_client is None:
            local_mongo_client = MongoClient(local_cfg["LOCAL_MONGO_URI"], serverSelectionTimeoutMS=10000)
            local_mongo_client.admin.command("ping")
        db = local_mongo_client.get_database(local_cfg["LOCAL_MONGO_DB"])
        collection = db.get_collection(collection_name)
        return collection
    except Exception as exc:
        raise RuntimeError(f"Failed to connect to local MongoDB: {exc}") from exc


def get_mysql_connection(mysql_cfg: Dict[str, Any]):
    """Create and return a MySQL connection."""
    global mysql_connector
    if mysql_connector is None:
        try:
            mysql_connector = importlib.import_module("mysql.connector")
        except ImportError:
            logging.warning("mysql-connector-python not installed; weather enrichment disabled.")
            return None
    try:
        conn = mysql_connector.connect(
            host=mysql_cfg["host"],
            user=mysql_cfg["user"],
            password=mysql_cfg["password"],
            database=mysql_cfg["db_name"],
        )
        from src.util.pipeline_utils import get_pipeline_logger
        logger = get_pipeline_logger()
        logger.log_connection_status(f"MySQL {mysql_cfg['db_name']}", "connected")
        logging.info("✅ MySQL database: %s", mysql_cfg["db_name"])
        return conn
    except Exception as exc:
        from src.util.pipeline_utils import get_pipeline_logger
        logger = get_pipeline_logger()
        logger.log_connection_status(f"MySQL {mysql_cfg.get('db_name', 'Unknown')}", "failed", str(exc))
        logging.info("❌ MySQL database: %s (Error: %s)", mysql_cfg.get("db_name", "Unknown"), exc)
        raise RuntimeError(f"Failed to connect to MySQL: {exc}") from exc


def fetch_weather_data(mysql_conn, city_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch weather data from table city_<city_id> for the given date range.
    start_date and end_date must be YYYY-MM-DD strings.
    """
    if mysql_conn is None:
        logging.warning("MySQL connection unavailable; skipping weather fetch.")
        return pd.DataFrame()

    if not city_id:
        logging.warning("city_id missing; skipping weather fetch.")
        return pd.DataFrame()

    table_name = f"city_{city_id}"
    cursor = mysql_conn.cursor(dictionary=True)
    try:
        query = f"""
            SELECT `date`,
                   temperature_max,
                   temperature_min,
                   precipitation_sum,
                   rain_sum,
                   showers_sum,
                   weather_code,
                   weather_description
            FROM `{table_name}`
            WHERE `date` BETWEEN %s AND %s
            ORDER BY `date`;
        """
        cursor.execute(query, (start_date, end_date))
        rows = cursor.fetchall()
        if not rows:
            return pd.DataFrame()
        weather_df = pd.DataFrame(rows)
        weather_df["date"] = pd.to_datetime(weather_df["date"])
        return weather_df
    except Exception as exc:
        logging.error("Failed to fetch weather data for city %s: %s", city_id, exc)
        return pd.DataFrame()
    finally:
        cursor.close()


def load_holiday_table(path: str) -> pd.DataFrame:
    """Load holidays_table.csv with parsed dates."""
    if not os.path.exists(path):
        logging.warning("Holiday table not found at %s", path)
        return pd.DataFrame()
    try:
        holiday_df = pd.read_csv(path)
    except Exception as exc:
        logging.error("Failed to read holiday table %s: %s", path, exc)
        return pd.DataFrame()

    holiday_df = holiday_df.rename(
        columns={
            "date": "date",
            "holiday_name": "holiday_name",
            "day": "holiday_day",
            "type": "holiday_type",
        }
    )
    if "date" in holiday_df.columns:
        holiday_df["date"] = pd.to_datetime(holiday_df["date"], dayfirst=True, errors="coerce")
    return holiday_df


def fetch_item_metadata(collection, restaurant_id: str, item_entity_id: str) -> Dict[str, Dict[str, Any]]:
    """
    Fetch item metadata for a restaurant from entityrecord.

    Returns a dictionary keyed by normalized item name.
    """
    restaurant_objid = to_objectid_if_possible(restaurant_id)
    entity_objid = to_objectid_if_possible(item_entity_id)

    query = {
        "entityId": entity_objid,
        "data.parentId": restaurant_objid,
    }
    projection = {
        "_id": 1,
        "data.name": 1,
        "data.isVeg": 1,
        "data.isSpicy": 1,
    }

    lookup: Dict[str, Dict[str, Any]] = {}
    cursor = collection.find(query, projection=projection)
    for doc in cursor:
        name = doc.get("data", {}).get("name")
        normalized = normalize_item_name(name)
        if not normalized:
            continue
        if normalized in lookup:
            logging.debug(
                "Duplicate item name '%s' for restaurant %s. Keeping first occurrence.",
                name,
                restaurant_id,
            )
            continue
        lookup[normalized] = {
            "menuitemid": str(doc["_id"]),
            "isVeg": doc.get("data", {}).get("isVeg"),
            "isSpicy": doc.get("data", {}).get("isSpicy"),
            "raw_name": name,
        }

    return lookup


def get_item_metadata_with_fallback(
    primary_collection,
    restaurant_id: str,
    item_entity_id: str,
    local_cfg: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Try primary entityrecord; fallback to local FOOD_ITEM_COLL when needed."""
    primary_exc = None
    try:
        lookup = fetch_item_metadata(primary_collection, restaurant_id, item_entity_id)
        if lookup:
            return lookup
        # Primary lookup empty, try local (no warning - this is expected)
    except Exception as exc:
        # Primary lookup failed, try local (no warning - this is expected)
        primary_exc = exc

    local_collection = get_local_mongo_collection(local_cfg, local_cfg.get("FOOD_ITEM_COLL", "food_item_record"))
    try:
        lookup = fetch_item_metadata(local_collection, restaurant_id, item_entity_id)
        if not lookup:
            error_msg = f"Unable to fetch item metadata for restaurant {restaurant_id} from both primary and local sources."
            if primary_exc:
                error_msg += f" Primary error: {primary_exc}"
            raise RuntimeError(error_msg)
        return lookup
    except Exception as local_exc:
        # Both primary and local failed - this is an error
        error_msg = f"Unable to fetch item metadata for restaurant {restaurant_id} from both primary and local sources."
        if primary_exc:
            error_msg += f" Primary error: {primary_exc}"
        error_msg += f" Local error: {local_exc}"
        raise RuntimeError(error_msg)


def enrich_orders(
    df: pd.DataFrame,
    metadata_lookup: Dict[str, Dict[str, Any]],
    progress_ctx: Dict[str, Any] | None = None,
    step_start_time: Optional[float] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Enrich a restaurant dataframe with metadata.

    Returns (enriched_df, match_count).
    """
    if df.empty:
        return df, 0

    working_df = df.copy()
    if "menuitemid" not in working_df.columns:
        working_df["menuitemid"] = ""
    if "itemname" not in working_df.columns:
        working_df["itemname"] = ""
    working_df["isVeg"] = working_df.get("isVeg")
    working_df["isSpicy"] = working_df.get("isSpicy")
    if "price" not in working_df.columns:
        if {"total_price", "total_count"}.issubset(set(working_df.columns)):
            counts = pd.to_numeric(working_df["total_count"], errors="coerce").replace(0, pd.NA)
            working_df["price"] = pd.to_numeric(working_df["total_price"], errors="coerce") / counts
        else:
            working_df["price"] = pd.NA

    working_df["__normalized_itemname"] = working_df["itemname"].apply(normalize_item_name)

    total_records = len(working_df)
    if total_records == 0:
        return working_df.drop(columns=["__normalized_itemname"]), 0

    progress_prefix = ""
    if progress_ctx:
        progress_prefix = (
            f"FC {progress_ctx['fc_idx']}/{progress_ctx['total_fc']} | "
            f"Rest {progress_ctx['rest_idx']}/{progress_ctx['total_rest']} | Enriching"
        )

    # Initialize progress bar
    progress = None
    if progress_prefix and total_records > 0:
        progress = ProgressBar(
            total=total_records,
            prefix=progress_prefix,
            suffix="rows",
            length=40,
            show_elapsed=True
        )
    
    match_count = 0
    # Track FRI mappings for this enrichment batch
    fri_mappings = {}  # {(foodcourt_id, restaurant_id, item_id): item_name}
    
    for idx, row_index in enumerate(working_df.index, 1):
        normalized_name = working_df.at[row_index, "__normalized_itemname"]
        metadata = metadata_lookup.get(normalized_name)
        if metadata:
            match_count += 1
            item_id = metadata.get("menuitemid")
            item_name = metadata.get("raw_name", normalized_name)
            working_df.at[row_index, "menuitemid"] = item_id
            working_df.at[row_index, "isVeg"] = metadata.get("isVeg")
            working_df.at[row_index, "isSpicy"] = metadata.get("isSpicy")
            
            # Store FRI mapping (will be saved after enrichment completes)
            # Note: foodcourt_id and restaurant_id are not in working_df, they're passed via context
            # We'll handle mapping update after enrichment when we have all IDs

        if progress:
            progress.set_current(idx)

    working_df = working_df.drop(columns=["__normalized_itemname"])
    return working_df, match_count


def consolidate_items_by_normalized_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate items with the same normalized name but different menuitemids.
    This ensures that items with identical normalized names use the same canonical menuitemid.
    
    Strategy:
    1. Group by restaurant and normalized item name
    2. For each group, choose the latest menuitemid (based on date or order of appearance)
    3. Update all rows in the group to use the latest menuitemid
    """
    if df.empty or "menuitemid" not in df.columns or "itemname" not in df.columns:
        return df
    
    working_df = df.copy()
    
    # Ensure restaurant column exists for grouping
    if "restaurant" not in working_df.columns:
        logging.warning("'restaurant' column not found, consolidating without restaurant grouping")
        restaurant_col = None
    else:
        restaurant_col = "restaurant"
    
    # Create normalized name column for grouping
    working_df["__normalized_itemname"] = working_df["itemname"].apply(normalize_item_name)
    
    # Determine grouping columns
    if restaurant_col:
        group_cols = [restaurant_col, "__normalized_itemname"]
    else:
        group_cols = ["__normalized_itemname"]
    
    # Group by restaurant and normalized name, get the latest menuitemid for each group
    name_to_canonical_id = {}
    
    for group_key, group in working_df.groupby(group_cols):
        # Get all non-empty menuitemids for this normalized name
        menuitemids = group["menuitemid"].dropna()
        menuitemids = menuitemids[menuitemids.astype(str).str.strip() != ""]
        
        if len(menuitemids) == 0:
            continue
        
        # Determine latest ID based on date if available, otherwise use last occurrence
        if "date" in group.columns:
            # Sort by date (most recent first) and get the menuitemid from the latest date
            group_sorted = group.sort_values("date", ascending=False, na_position='last')
            latest_row = group_sorted.iloc[0]
            canonical_id = str(latest_row["menuitemid"])
        else:
            # No date column, use the last occurrence (assuming later IDs appear later)
            # Try to determine "latest" by comparing IDs if they're sortable
            try:
                # If IDs are ObjectIds or similar, sort them
                menuitemids_list = menuitemids.astype(str).tolist()
                canonical_id = sorted(menuitemids_list, reverse=True)[0]  # Latest by string sort
            except:
                # Fallback: use the last occurrence in the group
                canonical_id = str(menuitemids.iloc[-1])
        
        # Create key for lookup: (restaurant_id, normalized_name) or just normalized_name
        if restaurant_col:
            restaurant_id = str(group_key[0]) if isinstance(group_key, tuple) else str(group_key)
            normalized_name = str(group_key[1]) if isinstance(group_key, tuple) else str(group_key)
            lookup_key = (restaurant_id, normalized_name)
        else:
            normalized_name = str(group_key)
            lookup_key = normalized_name
        
        name_to_canonical_id[lookup_key] = canonical_id
    
    # Update all rows to use canonical menuitemid
    if name_to_canonical_id:
        def get_canonical_id(row):
            normalized = normalize_item_name(str(row.get("itemname", "")))
            current_id = str(row.get("menuitemid", ""))
            
            # Create lookup key
            if restaurant_col:
                restaurant_id_val = str(row.get(restaurant_col, ""))
                lookup_key = (restaurant_id_val, normalized)
            else:
                lookup_key = normalized
            
            canonical_id = name_to_canonical_id.get(lookup_key, current_id)
            return canonical_id if canonical_id else current_id
        
        working_df["menuitemid"] = working_df.apply(get_canonical_id, axis=1)
    
    # Drop temporary column
    working_df = working_df.drop(columns=["__normalized_itemname"])
    
    return working_df


def aggregate_enriched_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate enriched data using IDs only (foodcourtid, restaurant, menuitemid, date).
    Sums total_count for rows with the same ID combination and keeps supporting metadata.
    """
    if df.empty:
        return df

    working_df = df.copy()

    # Ensure date is normalized to YYYY-MM-DD strings to avoid time-based duplicates.
    if "date" in working_df.columns:
        working_df["date"] = pd.to_datetime(working_df["date"]).dt.strftime("%Y-%m-%d")

    # Group by IDs only (not names) to aggregate same items
    grouping_cols = [
        "foodcourtid",
        "restaurant",
        "menuitemid",
        "date",
    ]
    grouping_cols = [col for col in grouping_cols if col in working_df.columns]

    if not grouping_cols:
        return df

    # Build aggregation map: sum counts, keep first value for other metadata
    agg_map: Dict[str, Any] = {"total_count": "sum"}
    
    # Keep metadata columns with first value (names, flags, etc.)
    metadata_cols = [
        "foodcourtname",
        "restaurantname",
        "itemname",
        "is_preorder",
        "isVeg",
        "isSpicy",
        "price",
    ]
    for col in metadata_cols:
        if col in working_df.columns:
            agg_map[col] = "first"

    aggregated_df = (
        working_df.groupby(grouping_cols, dropna=False).agg(agg_map).reset_index()
    )

    aggregated_df = aggregated_df.rename(columns={"total_count": "count"})
    aggregated_df = aggregated_df.sort_values("date")

    return aggregated_df


def build_item_date_grid(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """Create complete date × item grid to ensure continuous coverage."""
    if aggregated_df.empty:
        return aggregated_df

    working_df = aggregated_df.copy()
    working_df["date"] = pd.to_datetime(working_df["date"])

    min_date = working_df["date"].min()
    max_date = working_df["date"].max()
    if pd.isna(min_date) or pd.isna(max_date):
        return aggregated_df

    date_range = pd.date_range(min_date, max_date, freq="D")
    date_df = pd.DataFrame({"date": date_range})
    date_df["__key"] = 1

    item_id_cols = [
        col
        for col in ["foodcourtid", "restaurant", "menuitemid"]
        if col in working_df.columns
    ]
    metadata_cols = [
        "itemname",
        "price",
        "isVeg",
        "isSpicy",
        "restaurantname",
        "foodcourtname",
    ]
    metadata_cols = [col for col in metadata_cols if col in working_df.columns]
    item_cols = item_id_cols + [col for col in metadata_cols if col not in item_id_cols]
    if not item_cols:
        return working_df

    items_df = working_df[item_cols].copy()
    if item_id_cols:
        # Ensure a single canonical metadata row per unique ID combination.
        items_df = (
            items_df.sort_values(item_id_cols)
            .drop_duplicates(subset=item_id_cols, keep="first")
        )
    else:
        items_df = items_df.drop_duplicates()
    items_df["__key"] = 1

    grid_df = items_df.merge(date_df, on="__key").drop(columns="__key")
    return grid_df


def merge_with_item_date_grid(grid_df: pd.DataFrame, aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """Merge aggregated restaurant data onto the full item-date grid."""
    if grid_df.empty:
        return aggregated_df

    working_grid = grid_df.copy()
    working_grid["date"] = pd.to_datetime(working_grid["date"])

    working_agg = aggregated_df.copy()
    if "date" in working_agg.columns:
        working_agg["date"] = pd.to_datetime(working_agg["date"])

    merge_cols = ["menuitemid", "date"]
    merged_df = working_grid.merge(working_agg, on=merge_cols, how="left", suffixes=("", "_agg"))

    cols_to_drop = [col for col in merged_df.columns if col.endswith("_agg")]
    cols_to_drop += [col for col in ["is_preorder", "unique_orders"] if col in merged_df.columns]
    if cols_to_drop:
        merged_df = merged_df.drop(columns=cols_to_drop)

    # Fill metric columns with zeros when missing
    metric_cols = ["count"]
    for col in metric_cols:
        if col in merged_df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                filled = merged_df[col].fillna(0)
            merged_df[col] = filled.infer_objects(copy=False)

    # Forward fill remaining metadata columns from grid values
    for col in ["itemname", "is_preorder", "isVeg", "isSpicy", "restaurantname", "foodcourtname"]:
        if col in merged_df.columns and merged_df[col].isna().any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                filled = merged_df[col].ffill().bfill()
            merged_df[col] = filled.infer_objects(copy=False)

    return merged_df


def merge_weather_into_grid(grid_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Attach weather columns to the grid dataframe."""
    if grid_df.empty or weather_df.empty:
        return grid_df

    working_grid = grid_df.copy()
    working_grid["date"] = pd.to_datetime(working_grid["date"])
    working_weather = weather_df.copy()
    working_weather["date"] = pd.to_datetime(working_weather["date"])

    merged = working_grid.merge(working_weather, on="date", how="left")
    return merged


def merge_holiday_data(df: pd.DataFrame, holiday_df: pd.DataFrame) -> pd.DataFrame:
    """Merge holiday metadata and add flag columns."""
    if df.empty or "date" not in df.columns:
        return df

    working_df = df.copy()
    working_df["date"] = pd.to_datetime(working_df["date"]).dt.normalize()

    flag_cols = ["is_minor", "is_major", "is_sandwich"]

    def ensure_flag_columns():
        for col in flag_cols:
            if col not in working_df.columns:
                working_df[col] = 0

    if holiday_df is None or holiday_df.empty or "date" not in holiday_df.columns:
        ensure_flag_columns()
        return working_df

    working_holidays = holiday_df.copy()
    working_holidays["date"] = pd.to_datetime(working_holidays["date"]).dt.normalize()

    working_df = working_df.merge(
        working_holidays,
        on="date",
        how="left",
        suffixes=("", "_holiday"),
    )
    matched_holidays = working_df["holiday_type"].notna().sum()

    ensure_flag_columns()
    type_series = working_df.get("holiday_type")
    if type_series is None:
        type_series = pd.Series([None] * len(working_df), index=working_df.index)
    else:
        type_series = (
            type_series.astype(str)
            .str.strip()
            .str.lower()
            .replace("sandwitch", "sandwich")
        )

    working_df["is_minor"] = (type_series == "minor").astype(int)
    working_df["is_major"] = (type_series == "major").astype(int)
    working_df["is_sandwich"] = (type_series == "sandwich").astype(int)

    return working_df


def add_weekday_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_sun ... is_sat flags based on the date."""
    if df.empty or "date" not in df.columns:
        return df
    working_df = df.copy()
    working_df["date"] = pd.to_datetime(working_df["date"])
    weekday_flags = [
        (6, "is_sun"),
        (0, "is_mon"),
        (1, "is_tue"),
        (2, "is_wed"),
        (3, "is_thu"),
        (4, "is_fri"),
        (5, "is_sat"),
    ]
    weekday_series = working_df["date"].dt.weekday
    for weekday_value, col_name in weekday_flags:
        working_df[col_name] = (weekday_series == weekday_value).astype(int)
    return working_df


def reorder_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop holiday metadata columns and reorder to the requested schema."""
    if df.empty:
        return df

    working_df = df.copy()
    drop_cols = ["holiday_name", "holiday_day", "holiday_type"]
    for col in drop_cols:
        if col in working_df.columns:
            working_df = working_df.drop(columns=col)

    desired_order = [
        "foodcourtid",
        "foodcourtname",
        "restaurant",
        "restaurantname",
        "menuitemid",
        "itemname",
        "price",
        "isVeg",
        "isSpicy",
        "date",
        "count",
        "is_mon",
        "is_tue",
        "is_wed",
        "is_thu",
        "is_fri",
        "is_sat",
        "is_sun",
    ]

    existing_order = [col for col in desired_order if col in working_df.columns]
    remaining_cols = [col for col in working_df.columns if col not in existing_order]
    return working_df[existing_order + remaining_cols]


def load_progress(progress_path: str) -> Dict[str, Any]:
    """Load FR_data.json (renamed from fetch_progress.json)."""
    with open(progress_path, "r", encoding="utf-8") as progress_file:
        return json.load(progress_file)


def _initialize_connections():
    """
    Initialize all database connections upfront before processing starts.
    This ensures connections are established before any items are processed.
    """
    from src.util.connection_manager import get_connection_manager
    
    logging.info("Initializing database connections...")
    
    # Initialize ConnectionManager connections (for get_mongo_names, etc.)
    # This already handles local MongoDB and MySQL initialization
    conn_mgr = get_connection_manager()
    try:
        conn_mgr.initialize_all_connections()
        logging.info("✅ All database connections initialized via ConnectionManager")
    except Exception as e:
        logging.error(f"Error initializing ConnectionManager connections: {e}")
        raise RuntimeError(f"Failed to initialize database connections: {e}") from e
    
    # Also initialize step2-specific global connections for backward compatibility
    # These are used by legacy functions in step2
    from src.util.config_parser import ConfigManger
    config_manager = ConfigManger()
    local_cfg = config_manager.read_config("local_mongodb") or {}
    weather_cfg = config_manager.read_config("weather_record") or {}
    
    # Initialize local MongoDB connection (for legacy code)
    global local_mongo_client
    try:
        if local_mongo_client is None:
            from pymongo import MongoClient
            local_uri = local_cfg.get("LOCAL_MONGO_URI")
            if local_uri:
                local_mongo_client = MongoClient(local_uri, serverSelectionTimeoutMS=10000)
                local_mongo_client.admin.command("ping")
                logging.debug(f"✅ Legacy local MongoDB client initialized: {local_uri}")
            else:
                logging.warning("LOCAL_MONGO_URI not found in config, legacy MongoDB client not initialized")
    except KeyError as e:
        logging.warning(f"Missing config key for local MongoDB: {e} (ConnectionManager connection should still work)")
        # Don't raise - ConnectionManager already has the connection
    except Exception as e:
        logging.warning(f"Failed to initialize legacy local MongoDB client: {e} (ConnectionManager connection should still work)")
        # Don't raise - ConnectionManager already has the connection
    
    # MySQL connector module will be loaded lazily when needed
    # No need to test connection here since ConnectionManager handles it


def main(retrain_config: Optional[dict] = None, file_saver=None, restaurant_tracker=None, checkpoint_manager=None):
    """
    Main function for data enrichment.
    
    Args:
        retrain_config: Optional retrain configuration dict from retrain.json
        file_saver: Optional FileSaver instance for saving files (if None, uses default paths)
        restaurant_tracker: Optional RestaurantTracker instance for tracking item status
        checkpoint_manager: Optional CheckpointManager instance for checkpoint/resume functionality
    """
    # Initialize all database connections before processing starts
    _initialize_connections()
    
    config = ConfigManger().read_config(type="config")
    if not config:
        logging.error("Unable to load config/config.json")
        return

    try:
        mongo_cfg = config["mongodb"]
        local_cfg = config.get("local_mongodb", {})
        item_entity_id = config["entity_ids"]["item_entity_id"]
    except KeyError as exc:
        logging.error("Missing key in config: %s", exc)
        return

    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to root from src/
    input_dir = os.path.join(current_dir, "input_data")
    # Use FR_data.json instead of fetch_progress_test.json
    progress_path = os.path.join(input_dir, "FR_data.json")
    holiday_csv_path = os.path.join(input_dir, "holidays_table.csv")
    # Read from fetched_data instead of Model Training/food_court_data
    raw_base_path = os.path.join(input_dir, "fetched_data")
    # Write to output_data/enrich_data/{pipeline_type} instead of Model Training/enriched_data
    # Use file_saver if provided, otherwise use default paths
    if file_saver:
        processed_base_path = str(file_saver.get_folder_path("enrich_data", create=True))
    else:
        from src.util.pipeline_utils import get_output_base_dir, get_pipeline_type
        output_base = get_output_base_dir()
        pipeline_type = get_pipeline_type()
        processed_base_path = output_base / "enrich_data" / pipeline_type
        processed_base_path.mkdir(parents=True, exist_ok=True)
        processed_base_path = str(processed_base_path)

    if not os.path.exists(progress_path):
        logging.error("FR_data.json not found at %s. This file is required.", progress_path)
        return

    try:
        progress = load_progress(progress_path)
    except json.JSONDecodeError as exc:
        logging.error("Unable to parse progress JSON: %s", exc)
        return

    try:
        collection, client = get_mongo_collection(
            mongo_cfg["connection_string"],
            mongo_cfg["db_name"],
            mongo_cfg["entity_record"],
        )
    except Exception as exc:
        # Primary connection failed, try local fallback (no warning - this is expected)
        local_cfg = config.get("local_mongodb", {})
        fallback_conn_str = local_cfg.get("LOCAL_MONGO_URI")
        fallback_db = local_cfg.get("LOCAL_MONGO_DB")
        fallback_coll = local_cfg.get("FOODORDER_MONGO_DB")
        if not all([fallback_conn_str, fallback_db, fallback_coll]):
            logging.error("Local Mongo fallback not configured; aborting.")
            return
        try:
            collection, client = get_mongo_collection(
                fallback_conn_str,
                fallback_db,
                fallback_coll,
            )
        except Exception as fallback_exc:
            # Both primary and fallback failed - this is an error
            logging.error("Both primary and local Mongo connections failed. Primary: %s, Fallback: %s", exc, fallback_exc)
            logging.error("Cannot proceed without database connection. Please review configuration.")
            return
    weather_cfg = config.get("weather_record") or config.get("mysql") or {}
    mysql_conn = get_mysql_connection(weather_cfg)
    holiday_df = load_holiday_table(holiday_csv_path)
    from src.util.pipeline_utils import get_pipeline_logger
    pipeline_logger = get_pipeline_logger()
    pipeline_logger.log_connection_status(f"Holiday Data", "success", f"{len(holiday_df)} rows loaded")
    logging.info("✅ Holiday data: %d rows loaded", len(holiday_df))

    # ============================================================================
    # SUMMARY: Count foodcourts and restaurants in different locations
    # ============================================================================
    logging.info("=" * 80)
    logging.info("DATA SUMMARY - Before Processing")
    logging.info("=" * 80)
    
    # Count in FR_data.json
    fr_foodcourts = len(progress)
    fr_restaurants = sum(len(fc_info.get("restaurants", {})) for fc_info in progress.values())
    logging.info("FR_data.json: %d foodcourts, %d restaurants", fr_foodcourts, fr_restaurants)
    
    # Count in fetched_data - SCAN DIRECTORY, don't use FR_data.json
    fetched_foodcourts = 0
    fetched_restaurants = 0
    foodcourts_with_data = []
    fetched_data_structure = {}  # {foodcourt_id: [restaurant_ids]}
    
    # Scan fetched_data/ directory directly
    if os.path.exists(raw_base_path):
        for fc_dir in os.listdir(raw_base_path):
            fc_path = os.path.join(raw_base_path, fc_dir)
            if os.path.isdir(fc_path):
                parquet_files = [f for f in os.listdir(fc_path) if f.endswith('.parquet')]
                if parquet_files:
                    fetched_foodcourts += 1
                    # Extract restaurant_ids from parquet filenames (restaurant_id.parquet)
                    restaurant_ids = [f.replace('.parquet', '') for f in parquet_files]
                    fetched_restaurants += len(restaurant_ids)
                    foodcourts_with_data.append(fc_dir)
                    fetched_data_structure[fc_dir] = restaurant_ids
    
    logging.info("fetched_data/: %d foodcourts, %d restaurants (parquet files)", 
                fetched_foodcourts, fetched_restaurants)
    
    # Count in enrich_data output (current step)
    enrich_foodcourts = 0
    enrich_restaurants = 0
    if os.path.exists(processed_base_path):
        for fc_dir in os.listdir(processed_base_path):
            fc_path = os.path.join(processed_base_path, fc_dir)
            if os.path.isdir(fc_path):
                enrich_foodcourts += 1
                # Count CSV files (each represents an item, but we want unique restaurants)
                csv_files = [f for f in os.listdir(fc_path) if f.endswith('.csv')]
                # Extract unique restaurant IDs from filenames
                restaurant_ids = set()
                for fname in csv_files:
                    # Filename format: {fc_id}_{rest_id}_{item_id}_{item_name}_enrich_data.csv
                    parts = fname.split('_')
                    if len(parts) >= 2:
                        restaurant_ids.add(parts[1])  # restaurant_id is second part
                enrich_restaurants += len(restaurant_ids)
    logging.info("enrich_data/ (output): %d foodcourts, %d restaurants (from existing files)", 
                enrich_foodcourts, enrich_restaurants)
    
    # Note: FR_data.json is only for metadata, not for determining what to process
    # We process what's in fetched_data/ (or retrain.json if specified)
    
    # Calculate what needs processing: fetched_data/ - enrich_data/
    needs_processing = fetched_restaurants - enrich_restaurants
    logging.info("=" * 80)
    logging.info("PROCESSING DECISION:")
    logging.info("  - Restaurants in fetched_data/: %d", fetched_restaurants)
    logging.info("  - Restaurants in enrich_data/: %d", enrich_restaurants)
    logging.info("  - Restaurants needing enrichment: %d (difference + retrain config)", needs_processing)
    logging.info("=" * 80)
    
    # Log process start with summary
    summary = {
        "fetched_data/ (input)": f"{fetched_foodcourts} foodcourts, {fetched_restaurants} restaurants",
        "enrich_data/ (existing)": f"{enrich_foodcourts} foodcourts, {enrich_restaurants} restaurants",
        "Needs processing": f"{needs_processing} restaurants (difference + retrain config)"
    }
    pipeline_logger.log_process_start("enrich_data", summary)
    
    logging.info("Process started")
    logging.info("=" * 80)
    
    # ============================================================================
    # PROCESSING: Use only foodcourts with data in fetched_data
    # If retrain.json is non-empty, ONLY process entries from retrain.json
    # If retrain.json is empty, process all foodcourts that need processing (check previous level)
    # ============================================================================
    from src.util.pipeline_utils import should_force_retrain, is_retrain_config_empty, load_retrain_config
    
    # Check if retrain.json is empty for enrich_data step
    if retrain_config is None:
        retrain_config = load_retrain_config()
    retrain_is_completely_empty = is_retrain_config_empty()
    
    # Get enrich_data config (new format: dict with foodcourt_ids and restaurant_ids)
    from src.util.pipeline_utils import get_retrain_config_for_step
    enrich_step_config = get_retrain_config_for_step("enrich_data")
    enrich_foodcourt_ids = enrich_step_config.get("foodcourt_ids", [])
    enrich_restaurant_ids = enrich_step_config.get("restaurant_ids", [])
    enrich_item_ids = enrich_step_config.get("item_ids", [])
    enrich_item_names = enrich_step_config.get("item_names", [])
    enrich_retrain_is_empty = (len(enrich_foodcourt_ids) == 0 and len(enrich_restaurant_ids) == 0 and 
                               len(enrich_item_ids) == 0 and len(enrich_item_names) == 0)
    
    # Extract foodcourts and restaurants from retrain.json
    # Priority: foodcourt_ids > restaurant_ids > item_ids
    retrain_foodcourts = set(enrich_foodcourt_ids) if enrich_foodcourt_ids else set()
    retrain_restaurant_ids_set = set(enrich_restaurant_ids) if enrich_restaurant_ids else set()
    retrain_restaurants = {}  # {foodcourt_id: set of restaurant_ids}
    retrain_item_ids = enrich_item_ids if enrich_item_ids else []
    
    # If foodcourt_ids is empty, extract from restaurant_ids, item_ids, or item_names
    if not retrain_foodcourts:
        # Extract foodcourts from restaurant_ids (if they have foodcourt_id in dict format)
        if enrich_restaurant_ids:
            for rest_entry in enrich_restaurant_ids:
                if isinstance(rest_entry, dict) and "foodcourt_id" in rest_entry:
                    fc_id = rest_entry.get("foodcourt_id", "").strip()
                    if fc_id:
                        retrain_foodcourts.add(fc_id)
        
        # Extract foodcourts from item_ids
        if enrich_item_ids:
            for item in enrich_item_ids:
                if isinstance(item, dict):
                    fc_id = item.get("foodcourt_id", "").strip()
                    if fc_id:
                        retrain_foodcourts.add(fc_id)
        
        # Extract foodcourts from item_names
        if enrich_item_names:
            for item in enrich_item_names:
                if isinstance(item, dict):
                    fc_id = item.get("foodcourt_id", "").strip()
                    if fc_id:
                        retrain_foodcourts.add(fc_id)
    
    # Build retrain_restaurants mapping
    # If foodcourt_ids are specified, initialize all of them
    for fc_id in retrain_foodcourts:
        if fc_id not in retrain_restaurants:
            retrain_restaurants[fc_id] = set()
    
    # If restaurant_ids are specified, add them to their foodcourts
    if enrich_restaurant_ids:
        for rest_entry in enrich_restaurant_ids:
            if isinstance(rest_entry, dict):
                # New format: {foodcourt_id, restaurant_id}
                fc_id = rest_entry.get("foodcourt_id", "").strip()
                rest_id = rest_entry.get("restaurant_id", "").strip()
                if fc_id and rest_id:
                    if fc_id not in retrain_restaurants:
                        retrain_restaurants[fc_id] = set()
                    retrain_restaurants[fc_id].add(rest_id)
            elif isinstance(rest_entry, str):
                # Old format: simple restaurant_id string
                # Apply to all foodcourts in retrain_foodcourts
                for fc_id in retrain_foodcourts:
                    retrain_restaurants[fc_id].add(rest_entry)
    
    # Extract restaurants from item_ids
    if enrich_item_ids:
        for item in enrich_item_ids:
            if isinstance(item, dict):
                fc_id = item.get("foodcourt_id", "").strip()
                rest_id = item.get("restaurant_id", "").strip()
                if fc_id and rest_id:
                    if fc_id not in retrain_restaurants:
                        retrain_restaurants[fc_id] = set()
                    retrain_restaurants[fc_id].add(rest_id)
    
    # Extract restaurants from item_names
    if enrich_item_names:
        for item in enrich_item_names:
            if isinstance(item, dict):
                fc_id = item.get("foodcourt_id", "").strip()
                rest_id = item.get("restaurant_id", "").strip()
                if fc_id and rest_id:
                    if fc_id not in retrain_restaurants:
                        retrain_restaurants[fc_id] = set()
                    retrain_restaurants[fc_id].add(rest_id)
    
    # Filter foodcourts to process
    # IMPORTANT: Use retrain.json OR scan fetched_data/, NOT FR_data.json
    foodcourts_to_process = []
    # Check if ANY of the filters are specified (foodcourt_ids, restaurant_ids, item_ids, or item_names)
    has_any_filter = (len(enrich_foodcourt_ids) > 0 or 
                     len(enrich_restaurant_ids) > 0 or 
                     len(enrich_item_ids) > 0 or
                     len(enrich_item_names) > 0)
    
    if has_any_filter:
        # enrich_data has filters in retrain.json: process based on what's specified
        if retrain_foodcourts:
            # We have foodcourts to process (from foodcourt_ids, restaurant_ids, or item_ids)
            logging.info("enrich_data retrain.json is non-empty: Processing %d foodcourts from retrain.json", len(retrain_foodcourts))
            if len(enrich_foodcourt_ids) > 0:
                logging.info("  - Filtering by foodcourt_ids: %d foodcourts", len(enrich_foodcourt_ids))
            if len(enrich_restaurant_ids) > 0:
                logging.info("  - Filtering by restaurant_ids: %d restaurants", len(enrich_restaurant_ids))
            if len(enrich_item_ids) > 0:
                logging.info("  - Filtering by item_ids: %d items", len(enrich_item_ids))
            if len(enrich_item_names) > 0:
                logging.info("  - Filtering by item_names: %d items", len(enrich_item_names))
        
        if retrain_foodcourts:
            for foodcourt_id in retrain_foodcourts:
                # Check if this foodcourt has data in fetched_data/
                if foodcourt_id in foodcourts_with_data:
                    foodcourts_to_process.append(foodcourt_id)
                else:
                    logging.warning("Foodcourt %s in retrain.json but no data in fetched_data/", foodcourt_id)
        else:
            # This shouldn't happen if we extracted foodcourts correctly, but log a warning
            logging.warning("enrich_data has filters but no foodcourts could be extracted. Check retrain.json format.")
    elif retrain_is_completely_empty:
        # retrain.json is completely empty: process ALL foodcourts from fetched_data/ that need processing
        logging.info("retrain.json is completely empty: Processing all foodcourts from fetched_data/ that need enrichment")
        for foodcourt_id in foodcourts_with_data:
            output_dir = Path(processed_base_path) / foodcourt_id
            if not output_dir.exists():
                # No enriched data exists for this foodcourt, need to process
                foodcourts_to_process.append(foodcourt_id)
            else:
                # Check if there are restaurants without enriched data
                # Get restaurant_ids from fetched_data structure
                fetched_rest_ids = fetched_data_structure.get(foodcourt_id, [])
                for restaurant_id in fetched_rest_ids:
                    existing_files = list(output_dir.glob(f"{restaurant_id}_*.csv"))
                    if len(existing_files) == 0:
                        # At least one restaurant needs processing
                        foodcourts_to_process.append(foodcourt_id)
                        break
    else:
        # enrich_data list is empty but retrain.json has entries for other steps: skip this step
        # But only if enrich_retrain_is_empty is True (no item_names, item_ids, foodcourt_ids, or restaurant_ids)
        if enrich_retrain_is_empty:
            logging.info("enrich_data list is empty in retrain.json (but other steps have entries): Skipping enrichment step")
            return
        # If we have item_names or item_ids, we should process them
        # This means enrich_retrain_is_empty is False, so we should continue
    
    total_foodcourts = len(foodcourts_to_process)
    total_restaurants = 0
    restaurants_with_data = 0
    processed_files = 0
    total_rows = 0
    total_matched_rows = 0
    
    # Create a mapping of foodcourt_id to index for progress display (only for foodcourts to process)
    fc_to_index = {fc_id: idx for idx, fc_id in enumerate(foodcourts_to_process, 1)}
    
    # Start time tracking for this step
    step_start_time = time.time()
    
    # Track errors silently
    error_tracking = {
        "skipped_no_parquet": [],
        "skipped_empty": [],
        "skipped_already_exists": [],
        "failed_read": [],
        "failed_processing": [],
        "no_items": [],
        "other_errors": []
    }

    try:
        # Process only foodcourts in foodcourts_to_process (from retrain.json or fetched_data/)
        for foodcourt_id in foodcourts_to_process:
            # Skip if no data in fetched_data
            if foodcourt_id not in foodcourts_with_data:
                continue
            
            fc_idx = fc_to_index[foodcourt_id]
            
            # Get metadata from FR_data.json (ONLY for cityId and names)
            foodcourt_info = progress.get(foodcourt_id, {})
            city_id = foodcourt_info.get("cityId", "")
            
            # Get restaurant list from fetched_data structure (not FR_data.json)
            fetched_rest_ids = fetched_data_structure.get(foodcourt_id, [])
            
            # Get restaurant metadata from FR_data.json for names (if available)
            fr_restaurants = foodcourt_info.get("restaurants", {})
            
            # Filter restaurants based on retrain.json OR use all from fetched_data
            restaurants_to_process = []
            if not enrich_retrain_is_empty and retrain_foodcourts:
                # enrich_data has foodcourt_ids/restaurant_ids/item_ids: only process restaurants from retrain.json
                if foodcourt_id in retrain_restaurants:
                    retrain_rest_set = retrain_restaurants[foodcourt_id]
                    if len(retrain_rest_set) == 0:
                        # Empty set: check if item_ids are specified
                        if enrich_item_ids and len(enrich_item_ids) > 0:
                            # item_ids specified: only process restaurants that have items in item_ids
                            # Extract unique restaurants from item_ids for this foodcourt
                            restaurants_from_items = set()
                            for item in enrich_item_ids:
                                if isinstance(item, dict):
                                    item_fc_id = item.get("foodcourt_id", "").strip()
                                    item_rest_id = item.get("restaurant_id", "").strip()
                                    if item_fc_id == foodcourt_id and item_rest_id:
                                        restaurants_from_items.add(item_rest_id)
                            # Only process restaurants that have items in item_ids
                            restaurants_to_process = [r for r in fetched_rest_ids if r in restaurants_from_items]
                        else:
                            # No item_ids, empty restaurant set means process all restaurants in this foodcourt
                            restaurants_to_process = fetched_rest_ids.copy()
                    else:
                        # Process only specified restaurants (that are in both fetched_data and retrain lists)
                        restaurants_to_process = [r for r in fetched_rest_ids if r in retrain_rest_set]
                else:
                    # Foodcourt not in retrain.json (shouldn't happen, but handle gracefully)
                    continue
            elif retrain_is_completely_empty:
                # retrain.json is completely empty: process all restaurants from fetched_data that need processing
                restaurants_to_process = fetched_rest_ids.copy()
                # Filter out restaurants that already have enriched data
                output_dir = Path(processed_base_path) / foodcourt_id
                if output_dir.exists():
                    restaurants_to_process = [
                        r_id for r_id in restaurants_to_process
                        if len(list(output_dir.glob(f"{r_id}_*.csv"))) == 0
                    ]
            else:
                # enrich_data list is empty but other steps have entries: skip (shouldn't reach here)
                continue
            
            total_restaurants_in_fc = len(restaurants_to_process)
            for rest_idx, restaurant_id in enumerate(restaurants_to_process, 1):
                total_restaurants += 1
                
                raw_parquet_path = os.path.join(raw_base_path, foodcourt_id, f"{restaurant_id}.parquet")
                if not os.path.exists(raw_parquet_path):
                    # Track silently
                    error_tracking["skipped_no_parquet"].append({
                        "foodcourt_id": foodcourt_id,
                        "restaurant_id": restaurant_id,
                        "reason": "No raw Parquet file found"
                    })
                    continue
                
                # Check retrain logic from FR_data.json (via retrain.json)
                force_retrain = should_force_retrain("enrich_data", foodcourt_id, restaurant_id)
                
                # Compare fetched_data/ with enrich_data/ to determine if processing is needed
                # Process if: (parquet exists in fetched_data AND enriched data missing in enrich_data) OR (force retrain)
                output_dir = Path(processed_base_path) / foodcourt_id
                enriched_files_exist = False
                if output_dir.exists():
                    # Check if we have any enriched files for this restaurant
                    existing_files = list(output_dir.glob(f"{restaurant_id}_*.csv"))
                    enriched_files_exist = len(existing_files) > 0
                
                # Skip if enriched data exists and not forcing retrain (only when retrain.json is completely empty)
                if retrain_is_completely_empty and enriched_files_exist and not force_retrain:
                    error_tracking["skipped_already_exists"].append({
                        "foodcourt_id": foodcourt_id,
                        "restaurant_id": restaurant_id,
                        "reason": "Enriched data already exists"
                    })
                    continue
                
                # If enrich_data list in retrain.json is non-empty, always process (force retrain)
                if not enrich_retrain_is_empty:
                    force_retrain = True

                restaurants_with_data += 1
                try:
                    df = pd.read_parquet(raw_parquet_path)
                except Exception as exc:
                    error_tracking["failed_read"].append({
                        "foodcourt_id": foodcourt_id,
                        "restaurant_id": restaurant_id,
                        "reason": f"Failed to read parquet: {exc}"
                    })
                    continue

                if df.empty:
                    error_tracking["skipped_empty"].append({
                        "foodcourt_id": foodcourt_id,
                        "restaurant_id": restaurant_id,
                        "reason": "Raw Parquet is empty"
                    })
                    continue

                try:
                    # Get foodcourt and restaurant names for logging
                    from src.util.pipeline_utils import get_mongo_names, get_pipeline_logger, get_file_locator, matches_item_filter
                    foodcourt_name, restaurant_name = get_mongo_names(foodcourt_id, restaurant_id)
                    if not foodcourt_name:
                        foodcourt_name = foodcourt_id
                    if not restaurant_name:
                        restaurant_name = restaurant_id
                    
                    # NORMALIZE ITEMS FIRST (before filtering)
                    # Fetch metadata to normalize item names/IDs
                    # This ensures filtering uses normalized IDs/names
                    # Note: collection, item_entity_id, and local_cfg are defined in main() function scope
                    metadata_lookup = None
                    if "itemname" in df.columns and not df.empty:
                        # Fetch metadata lookup for normalization
                        # collection, item_entity_id, and local_cfg are from outer scope (main function)
                        try:
                            metadata_lookup = get_item_metadata_with_fallback(
                                collection,
                                restaurant_id,
                                item_entity_id,
                                local_cfg,
                            )
                            
                            # Normalize item names/IDs in dataframe using metadata
                            if metadata_lookup:
                                # Add normalized name column
                                df["__normalized_itemname"] = df["itemname"].apply(normalize_item_name)
                                
                                # Update menuitemid and itemname with normalized values
                                for idx, row in df.iterrows():
                                    normalized_name = df.at[idx, "__normalized_itemname"]
                                    metadata = metadata_lookup.get(normalized_name)
                                    if metadata:
                                        # Update with normalized/canonical values
                                        df.at[idx, "menuitemid"] = metadata.get("menuitemid", df.at[idx, "menuitemid"] if "menuitemid" in df.columns else "")
                                        df.at[idx, "itemname"] = metadata.get("raw_name", normalized_name)
                                
                                # Remove temporary column
                                df = df.drop(columns=["__normalized_itemname"])
                                
                                logging.debug(f"Normalized {len(df)} rows for {foodcourt_id}/{restaurant_id} using metadata")
                        except Exception as norm_exc:
                            # If normalization fails, continue with raw data (will be normalized later in enrich_orders)
                            logging.warning(f"Failed to normalize items before filtering for {foodcourt_id}/{restaurant_id}: {norm_exc}. Will normalize during enrichment.")
                    
                    # Filter by item_ids or item_names AFTER normalization (so we use normalized IDs/names)
                    if (enrich_item_ids and len(enrich_item_ids) > 0) or (enrich_item_names and len(enrich_item_names) > 0):
                        # Filter dataframe to only include items in item_ids
                        # Build a set of matching item identifiers for fast lookup
                        matching_items = set()
                        
                        # First, identify which items in enrich_item_ids match this restaurant
                        for item_filter in enrich_item_ids:
                            if isinstance(item_filter, dict):
                                filter_fc_id = str(item_filter.get("foodcourt_id", "")).strip()
                                filter_rest_id = str(item_filter.get("restaurant_id", "")).strip()
                                
                                # Check if this filter matches current foodcourt/restaurant
                                if filter_fc_id == foodcourt_id and filter_rest_id == restaurant_id:
                                    # Get item identifier from filter
                                    filter_item_id = str(item_filter.get("item_id", "")).strip()
                                    filter_item_name = str(item_filter.get("item_name", "")).strip().lower() if item_filter.get("item_name") else ""
                                    
                                    if filter_item_id:
                                        matching_items.add(("id", filter_item_id))
                                    if filter_item_name:
                                        matching_items.add(("name", filter_item_name))
                        
                        # Also check item_names filter (can be array of strings or array of objects)
                        if enrich_item_names and len(enrich_item_names) > 0:
                            # Check if it's object format (like item_ids) or simple string array
                            first_item = enrich_item_names[0] if enrich_item_names else None
                            is_object_format = isinstance(first_item, dict)
                            
                            if is_object_format:
                                # Format: [{foodcourt_id, restaurant_id, item_name}, ...]
                                for item_name_entry in enrich_item_names:
                                    if isinstance(item_name_entry, dict):
                                        entry_fc_id = str(item_name_entry.get("foodcourt_id", "")).strip()
                                        entry_rest_id = str(item_name_entry.get("restaurant_id", "")).strip()
                                        entry_item_name = str(item_name_entry.get("item_name", "")).strip().lower()
                                        
                                        # Check if this filter matches current foodcourt/restaurant
                                        if entry_fc_id == foodcourt_id and entry_rest_id == restaurant_id:
                                            if entry_item_name:
                                                matching_items.add(("name", entry_item_name))
                            else:
                                # Format: ["item name 1", "item name 2", ...] - simple array
                                item_names_lower = [str(name).strip().lower() for name in enrich_item_names]
                                for name in item_names_lower:
                                    matching_items.add(("name", name))
                        
                        if not matching_items:
                            # No items in item_ids or item_names match this restaurant, skip it
                            error_tracking["no_items"].append({
                                "foodcourt_id": foodcourt_id,
                                "restaurant_id": restaurant_id,
                                "reason": f"No items in item_ids/item_names filter match this restaurant ({len(enrich_item_ids)} item_ids, {len(enrich_item_names)} item_names specified)"
                            })
                            continue
                        
                        # Filter dataframe rows
                        item_mask = pd.Series([False] * len(df), index=df.index)
                        
                        # Check each row against matching items
                        for idx, row in df.iterrows():
                            row_item_id = str(row.get("menuitemid", "")).strip() if "menuitemid" in row and pd.notna(row.get("menuitemid")) else ""
                            row_item_name = str(row.get("itemname", "")).strip().lower() if "itemname" in row and pd.notna(row.get("itemname")) else ""
                            
                            # Check if row matches any item in matching_items
                            if row_item_id and ("id", row_item_id) in matching_items:
                                item_mask[idx] = True
                            elif row_item_name and ("name", row_item_name) in matching_items:
                                item_mask[idx] = True
                        
                        # Filter dataframe
                        df = df[item_mask].copy()
                        
                        if df.empty:
                            error_tracking["no_items"].append({
                                "foodcourt_id": foodcourt_id,
                                "restaurant_id": restaurant_id,
                                "reason": f"No rows match item_ids filter after filtering ({len(matching_items)} items expected)"
                            })
                            continue
                        
                        # Removed INFO log: "Filtered to X rows matching item_ids filter..."
                    
                    # Filter out beverage/MRP items BEFORE enrichment to avoid processing them
                    removed_beverage_items = pd.DataFrame()
                    if "itemname" in df.columns:
                        df, removed_beverage_items = filter_excluded_items(df)
                        
                        # Log and track removed beverage/MRP items
                        if not removed_beverage_items.empty:
                            pipeline_logger = get_pipeline_logger()
                            file_locator = get_file_locator()
                            
                            for _, row in removed_beverage_items.iterrows():
                                removed_item_id = str(row.get("menuitemid", "")) if "menuitemid" in row else ""
                                removed_item_name = str(row.get("itemname", ""))
                                if not removed_item_id:
                                    removed_item_id = removed_item_name
                                
                                error_reason = "MRP / BEVERAGE"
                                
                                # Log to enrichment_logs
                                pipeline_logger.log_enrichment_error(
                                    foodcourt_id, foodcourt_name,
                                    restaurant_id, restaurant_name,
                                    removed_item_id, removed_item_name,
                                    error_reason
                                )
                                
                                # Add discard reason to file_locator
                                file_locator.add_discard_reason(
                                    foodcourt_id, foodcourt_name,
                                    restaurant_id, restaurant_name,
                                    removed_item_id, removed_item_name,
                                    "enrich_data",
                                    "MRP / BEVERAGE"
                                )
                                
                                # Track error in restaurant tracker
                                if restaurant_tracker:
                                    from src.util.pipeline_utils import get_all_names
                                    fc_name, rest_name, item_name_val = get_all_names(
                                        foodcourt_id, restaurant_id, removed_item_id, None
                                    )
                                    restaurant_tracker.add_error(
                                        foodcourt_id, restaurant_id, removed_item_id or removed_item_name,
                                        error_reason, "enrich_data",
                                        foodcourt_name=fc_name, restaurant_name=rest_name, item_name=removed_item_name or item_name_val
                                    )
                    
                    # If all items were filtered out, skip this restaurant
                    if df.empty:
                        error_tracking["no_items"].append({
                            "foodcourt_id": foodcourt_id,
                            "restaurant_id": restaurant_id,
                            "reason": f"All items filtered out as beverage/MRP (removed {len(removed_beverage_items)} items)"
                        })
                        continue
                    
                    # Track diagnostics for detailed error reporting
                    diagnostics = {
                        "initial_rows": len(df) + len(removed_beverage_items) if not removed_beverage_items.empty else len(df),
                        "beverage_items_removed": len(removed_beverage_items),
                        "has_date_column": "date" in df.columns or "date_IST" in df.columns,
                        "metadata_items_found": 0,
                        "enriched_rows": 0,
                        "matched_rows": 0,
                        "aggregated_rows": 0,
                        "grid_rows": 0,
                        "final_rows": 0,
                        "has_valid_dates": False,
                        "weather_data_found": False,
                        "items_in_final": 0
                    }
                    
                    # Check date columns before alignment
                    if "date" not in df.columns and "date_IST" not in df.columns:
                        error_tracking["no_items"].append({
                            "foodcourt_id": foodcourt_id,
                            "restaurant_id": restaurant_id,
                            "reason": "No date column found in raw data (missing 'date' or 'date_IST')"
                        })
                        continue
                    
                    df = align_dates_to_ist(df)
                    
                    # Check if dates are valid after alignment
                    if "date" in df.columns:
                        valid_dates = pd.to_datetime(df["date"], errors="coerce").notna().sum()
                        diagnostics["has_valid_dates"] = valid_dates > 0
                        if valid_dates == 0:
                            error_tracking["no_items"].append({
                                "foodcourt_id": foodcourt_id,
                                "restaurant_id": restaurant_id,
                                "reason": "No valid dates found in data after alignment"
                            })
                            continue
                    else:
                        error_tracking["no_items"].append({
                            "foodcourt_id": foodcourt_id,
                            "restaurant_id": restaurant_id,
                            "reason": "Date column missing after alignment"
                        })
                        continue

                    # Get metadata lookup (if not already fetched for normalization)
                    if metadata_lookup is None:
                        metadata_lookup = get_item_metadata_with_fallback(
                            collection,
                            restaurant_id,
                            item_entity_id,
                            local_cfg,
                        )
                    diagnostics["metadata_items_found"] = len(metadata_lookup) if metadata_lookup else 0
                    
                    if diagnostics["metadata_items_found"] == 0:
                        error_tracking["no_items"].append({
                            "foodcourt_id": foodcourt_id,
                            "restaurant_id": restaurant_id,
                            "reason": "No item metadata found in MongoDB for this restaurant"
                        })
                        continue

                    progress_ctx = {
                        "fc_idx": fc_idx,
                        "total_fc": total_foodcourts if total_foodcourts else 1,
                        "rest_idx": rest_idx,
                        "total_rest": total_restaurants_in_fc if total_restaurants_in_fc else 1,
                    }

                    # Pass step start time for progress tracking
                    enriched_df, matched_rows = enrich_orders(
                        df, metadata_lookup, progress_ctx=progress_ctx, step_start_time=step_start_time
                    )
                    
                    # Consolidate items with same normalized name but different menuitemids
                    # This ensures items like "Veg Combo Meal" with different IDs are merged
                    enriched_df = consolidate_items_by_normalized_name(enriched_df)
                    
                    diagnostics["enriched_rows"] = len(enriched_df)
                    diagnostics["matched_rows"] = matched_rows
                    total_rows += len(enriched_df)
                    total_matched_rows += matched_rows
                    
                    # Update FRI name mapping after enrichment (now we have normalized IDs and names)
                    # Extract unique FRI combinations from enriched data
                    # Note: foodcourt_name and restaurant_name are already fetched earlier (line ~1174)
                    if not enriched_df.empty and "menuitemid" in enriched_df.columns:
                        from src.util.name_mapping import update_fri_mapping
                        
                        # Use already fetched names (foodcourt_name, restaurant_name from line ~1174)
                        fc_name_to_use = foodcourt_name if 'foodcourt_name' in locals() and foodcourt_name else foodcourt_id
                        rest_name_to_use = restaurant_name if 'restaurant_name' in locals() and restaurant_name else restaurant_id
                        
                        # Get unique item IDs and their names from enriched data
                        if "itemname" in enriched_df.columns:
                            item_groups = enriched_df.groupby("menuitemid")
                            for item_id, item_group in item_groups:
                                item_id_str = str(item_id)
                                # Get item name from the group (use first non-null value)
                                item_name = item_group["itemname"].dropna().iloc[0] if not item_group["itemname"].dropna().empty else item_id_str
                                
                                # Update FRI mapping
                                update_fri_mapping(
                                    foodcourt_id, restaurant_id, item_id_str,
                                    foodcourt_name=fc_name_to_use,
                                    restaurant_name=rest_name_to_use,
                                    item_name=str(item_name) if item_name else None
                                )
                    
                    # Check if any items matched
                    if matched_rows == 0:
                        # Check if itemname column exists and has data
                        if "itemname" in df.columns:
                            unique_items = df["itemname"].nunique()
                            error_tracking["no_items"].append({
                                "foodcourt_id": foodcourt_id,
                                "restaurant_id": restaurant_id,
                                "reason": f"No items matched metadata (found {unique_items} unique items in data, {diagnostics['metadata_items_found']} items in metadata lookup)"
                            })
                        else:
                            error_tracking["no_items"].append({
                                "foodcourt_id": foodcourt_id,
                                "restaurant_id": restaurant_id,
                                "reason": f"No items matched metadata (no 'itemname' column in data, {diagnostics['metadata_items_found']} items in metadata lookup)"
                            })
                        continue

                    aggregated_df = aggregate_enriched_orders(enriched_df)
                    diagnostics["aggregated_rows"] = len(aggregated_df)
                    
                    if aggregated_df.empty:
                        error_tracking["no_items"].append({
                            "foodcourt_id": foodcourt_id,
                            "restaurant_id": restaurant_id,
                            "reason": f"Aggregation resulted in empty dataframe (enriched {diagnostics['enriched_rows']} rows, matched {diagnostics['matched_rows']} rows)"
                        })
                        continue

                    grid_df = build_item_date_grid(aggregated_df)
                    diagnostics["grid_rows"] = len(grid_df)
                    
                    if grid_df.empty:
                        # Check if dates are valid in aggregated_df
                        if "date" in aggregated_df.columns:
                            valid_dates_agg = pd.to_datetime(aggregated_df["date"], errors="coerce").notna().sum()
                            if valid_dates_agg == 0:
                                error_tracking["no_items"].append({
                                    "foodcourt_id": foodcourt_id,
                                    "restaurant_id": restaurant_id,
                                    "reason": "Date grid build failed: no valid dates in aggregated data"
                                })
                            else:
                                error_tracking["no_items"].append({
                                    "foodcourt_id": foodcourt_id,
                                    "restaurant_id": restaurant_id,
                                    "reason": f"Date grid build resulted in empty dataframe (aggregated {diagnostics['aggregated_rows']} rows)"
                                })
                        else:
                            error_tracking["no_items"].append({
                                "foodcourt_id": foodcourt_id,
                                "restaurant_id": restaurant_id,
                                "reason": "Date grid build failed: no date column in aggregated data"
                            })
                        continue

                    merged_grid_df = merge_with_item_date_grid(grid_df, aggregated_df)

                    weather_df = pd.DataFrame()
                    weather_status = "not_attempted"
                    if not aggregated_df.empty and mysql_conn:
                        min_date = pd.to_datetime(aggregated_df["date"]).min()
                        max_date = pd.to_datetime(aggregated_df["date"]).max()
                        if pd.notna(min_date) and pd.notna(max_date):
                            weather_df = fetch_weather_data(
                                mysql_conn,
                                city_id,
                                min_date.strftime("%Y-%m-%d"),
                                max_date.strftime("%Y-%m-%d"),
                            )
                            if not weather_df.empty:
                                diagnostics["weather_data_found"] = True
                                weather_status = "found"
                            else:
                                weather_status = f"not_found (city_id: {city_id}, date_range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})"
                        else:
                            weather_status = "skipped_invalid_dates"
                    elif not mysql_conn:
                        weather_status = "mysql_not_connected"
                    else:
                        weather_status = "skipped_empty_aggregated"

                    final_df = merge_weather_into_grid(merged_grid_df, weather_df)
                    final_df = merge_holiday_data(final_df, holiday_df)
                    final_df = add_weekday_flags(final_df)
                    final_df = reorder_final_columns(final_df)
                    diagnostics["final_rows"] = len(final_df)

                    # Note: Beverage/MRP items are already filtered before enrichment
                    # No need to filter again here

                    # Split by item and save as Excel files (one per item)
                    output_dir = Path(processed_base_path) / foodcourt_id
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    if final_df.empty:
                        # Provide detailed reason why final_df is empty
                        reason_parts = []
                        reason_parts.append(f"Initial rows: {diagnostics['initial_rows']}")
                        reason_parts.append(f"Enriched rows: {diagnostics['enriched_rows']}")
                        reason_parts.append(f"Matched rows: {diagnostics['matched_rows']}")
                        reason_parts.append(f"Aggregated rows: {diagnostics['aggregated_rows']}")
                        reason_parts.append(f"Grid rows: {diagnostics['grid_rows']}")
                        reason_parts.append(f"Weather: {weather_status}")
                        reason = "Final enriched dataframe is empty. " + " | ".join(reason_parts)
                        error_tracking["failed_processing"].append({
                            "foodcourt_id": foodcourt_id,
                            "restaurant_id": restaurant_id,
                            "reason": reason
                        })
                        continue
                    
                    # Get item identifier column
                    item_id_col = None
                    item_name_col = None
                    for col in ["menuitemid", "item_identifier", "itemname"]:
                        if col in final_df.columns:
                            if col == "menuitemid" or col == "item_identifier":
                                item_id_col = col
                            if col == "itemname":
                                item_name_col = col
                    
                    if not item_id_col and not item_name_col:
                        error_tracking["failed_processing"].append({
                            "foodcourt_id": foodcourt_id,
                            "restaurant_id": restaurant_id,
                            "reason": "No item identifier column found in final dataframe"
                        })
                        continue
                    
                    # Get foodcourt and restaurant names (try to get from MongoDB or use IDs)
                    from src.util.pipeline_utils import get_mongo_names
                    foodcourt_name, restaurant_name = get_mongo_names(foodcourt_id, restaurant_id)
                    if not foodcourt_name:
                        foodcourt_name = foodcourt_id
                    if not restaurant_name:
                        restaurant_name = restaurant_id
                    
                    # Split by item and save
                    from src.util.pipeline_utils import get_file_name, save_dataframe_to_excel
                    
                    # Group by item
                    if item_id_col:
                        group_col = item_id_col
                    else:
                        group_col = item_name_col
                    
                    # Count unique items in final_df
                    if group_col in final_df.columns:
                        unique_items = final_df[group_col].nunique()
                        diagnostics["items_in_final"] = unique_items
                    else:
                        unique_items = 0
                    
                    items_processed = 0
                    items_skipped_existing = 0
                    saved_files_for_validation = []  # Track files for validation
                    for item_value, item_df in final_df.groupby(group_col):
                        # Get item name and ID
                        item_id = str(item_value)
                        item_name = ""
                        if item_name_col and item_name_col in item_df.columns:
                            item_name = str(item_df[item_name_col].iloc[0])
                        if not item_name:
                            item_name = item_id
                        
                        # Note: item_ids/item_names filtering is already done BEFORE enrichment (at line ~1255)
                        # So all items in final_df should already match the filter
                        # But we still check here for safety and to determine force_retrain
                        if (enrich_item_ids and len(enrich_item_ids) > 0) or (enrich_item_names and len(enrich_item_names) > 0):
                            # Double-check item matches filter (should always be True since we filtered earlier)
                            from src.util.pipeline_utils import matches_item_filter
                            if not matches_item_filter(foodcourt_id, restaurant_id, item_name, item_id, 
                                                      enrich_item_ids, enrich_item_names):
                                # Item not in filter, skip it (shouldn't happen, but safety check)
                                continue
                            # Item matches filter, force retrain
                            force_retrain_item = True
                        elif not enrich_retrain_is_empty:
                            # enrich_data has foodcourt_ids/restaurant_ids: process all items for matching restaurants
                            force_retrain_item = True  # All items for restaurants in retrain.json
                        else:
                            # retrain.json is completely empty: check if file exists
                            force_retrain_item = should_force_retrain("enrich_data", foodcourt_id, restaurant_id, item_name, item_id)
                        
                        # Create item dict for checkpoint
                        item_dict = {
                            "foodcourt_id": foodcourt_id,
                            "restaurant_id": restaurant_id,
                            "item_id": item_id if item_id else None,
                            "item_name": item_name if item_name else None
                        }
                        
                        # Check checkpoint: skip if already completed (unless force retrain)
                        if checkpoint_manager:
                            if checkpoint_manager.is_item_completed("enrich_data", item_dict) and not force_retrain_item:
                                items_skipped_existing += 1
                                continue
                        
                        # Generate filename
                        filename = get_file_name(foodcourt_id, restaurant_id, item_name, "enrich_data", item_id)
                        output_path = output_dir / filename
                        
                        # Check if file exists and not forcing retrain (only when retrain.json is completely empty)
                        # Removed file existence check to improve performance
                        # If forcing retrain or retrain.json is non-empty, always process (replace existing file)
                        # Skip check only when retrain.json is completely empty and not forcing retrain
                        # (File will be overwritten anyway if it exists)
                        if retrain_is_completely_empty and not force_retrain_item:
                            # In this case, we could check, but removed for performance
                            # Files will be overwritten if they exist during save
                            pass
                        
                        # Mark as in progress before processing
                        if checkpoint_manager:
                            checkpoint_manager.mark_item_in_progress("enrich_data", item_dict)
                        
                        try:
                            export_df = item_df.copy()
                            if not export_df.empty and "date" in export_df.columns:
                                export_df["date"] = pd.to_datetime(export_df["date"]).dt.strftime("%Y-%m-%d")
                            
                            # Save as CSV - use file_saver if provided
                            if file_saver:
                                # Use file_saver to save to enrich_data folder with foodcourt_id subdirectory
                                output_path = file_saver.save_csv(export_df, "enrich_data", filename, 
                                                                  index=False, subdir=foodcourt_id)
                            else:
                                save_dataframe_to_excel(export_df, output_path, sheet_name="Enriched Data")  # Function now saves CSV
                            
                            # Track saved file for validation
                            saved_files_for_validation.append(output_path)
                            
                            # Removed validation check to improve performance
                            # Validation was checking file existence which slows down the pipeline
                            
                            # Mark as completed after successful save
                            if checkpoint_manager:
                                checkpoint_manager.mark_item_completed("enrich_data", item_dict)
                            
                            # Track success in restaurant tracker
                            if restaurant_tracker:
                                from src.util.pipeline_utils import get_all_names
                                fc_name, rest_name, item_name_val = get_all_names(
                                    foodcourt_id, restaurant_id, item_id, filename
                                )
                                restaurant_tracker.add_success(
                                    foodcourt_id, restaurant_id, item_id or item_name,
                                    filename, "enrich_data",
                                    foodcourt_name=fc_name, restaurant_name=rest_name, item_name=item_name or item_name_val
                                )
                            
                            
                            items_processed += 1
                            processed_files += 1
                        except Exception as exc:
                            error_msg = f"Failed to write Excel file for item {item_name}: {exc}"
                            # Mark as failed in checkpoint
                            if checkpoint_manager:
                                checkpoint_manager.mark_item_failed("enrich_data", item_dict, error_msg)
                            
                            error_tracking["other_errors"].append({
                                "foodcourt_id": foodcourt_id,
                                "restaurant_id": restaurant_id,
                                "reason": error_msg
                            })
                            # Track error in restaurant tracker
                            if restaurant_tracker:
                                from src.util.pipeline_utils import get_all_names
                                fc_name, rest_name, item_name_val = get_all_names(
                                    foodcourt_id, restaurant_id, item_id, filename if 'filename' in locals() else None
                                )
                                restaurant_tracker.add_error(
                                    foodcourt_id, restaurant_id, item_id or item_name,
                                    error_msg, "enrich_data",
                                    foodcourt_name=fc_name, restaurant_name=rest_name, item_name=item_name or item_name_val
                                )
                    
                    if items_processed == 0:
                        # Provide detailed reason why no items were processed
                        reason_parts = []
                        reason_parts.append(f"Initial rows: {diagnostics['initial_rows']}")
                        reason_parts.append(f"Metadata items found: {diagnostics['metadata_items_found']}")
                        reason_parts.append(f"Enriched rows: {diagnostics['enriched_rows']}")
                        reason_parts.append(f"Matched rows: {diagnostics['matched_rows']}")
                        reason_parts.append(f"Aggregated rows: {diagnostics['aggregated_rows']}")
                        reason_parts.append(f"Grid rows: {diagnostics['grid_rows']}")
                        reason_parts.append(f"Final rows: {diagnostics['final_rows']}")
                        reason_parts.append(f"Unique items in final: {unique_items}")
                        if items_skipped_existing > 0:
                            reason_parts.append(f"Items skipped (already exist): {items_skipped_existing}")
                        reason_parts.append(f"Weather: {weather_status}")
                        
                        if unique_items == 0:
                            reason = "No items in final dataframe. " + " | ".join(reason_parts)
                        elif items_skipped_existing == unique_items:
                            reason = f"All {unique_items} items already exist (skipped). " + " | ".join(reason_parts)
                        else:
                            reason = "No items processed. " + " | ".join(reason_parts)
                        
                        error_tracking["no_items"].append({
                            "foodcourt_id": foodcourt_id,
                            "restaurant_id": restaurant_id,
                            "reason": reason
                        })
                        # Track error in restaurant tracker (for restaurant-level error)
                        if restaurant_tracker:
                            from src.util.pipeline_utils import get_all_names
                            fc_name, rest_name, _ = get_all_names(foodcourt_id, restaurant_id, None, None)
                            restaurant_tracker.add_error(
                                foodcourt_id, restaurant_id, "all_items",
                                reason, "enrich_data",
                                foodcourt_name=fc_name, restaurant_name=rest_name, item_name=None
                            )
                except Exception as exc:
                    error_msg = f"Processing error: {exc}"
                    error_tracking["failed_processing"].append({
                        "foodcourt_id": foodcourt_id,
                        "restaurant_id": restaurant_id,
                        "reason": error_msg
                    })
                    # Track error in restaurant tracker
                    if restaurant_tracker:
                        from src.util.pipeline_utils import get_all_names
                        fc_name, rest_name, _ = get_all_names(foodcourt_id, restaurant_id, None, None)
                        restaurant_tracker.add_error(
                            foodcourt_id, restaurant_id, "all_items",
                            error_msg, "enrich_data",
                            foodcourt_name=fc_name, restaurant_name=rest_name, item_name=None
                        )
                    continue

    finally:
        client.close()
        if mysql_conn:
            mysql_conn.close()

    # Log errors to enrichment_logs sheet (no CSV conversion, just use parquet path)
    # Get names for logging
    from src.util.pipeline_utils import get_mongo_names
    
    # Log all error tracking entries to enrichment_logs
    for error_type, error_list in error_tracking.items():
        for error in error_list:
            fc_id = error.get("foodcourt_id", "")
            rest_id = error.get("restaurant_id", "")
            reason = error.get("reason", "Unknown error")
            
            # Get names
            fc_name, rest_name = get_mongo_names(fc_id, rest_id)
            
            # Use parquet path directly (no CSV conversion needed)
            input_file_link = "N/A"
            parquet_path = Path(raw_base_path) / fc_id / f"{rest_id}.parquet"
            if parquet_path.exists():
                input_file_link = str(parquet_path)
            else:
                input_file_link = "Parquet file not found"
            
            # Log to enrichment_logs (no item_id/item_name for restaurant-level errors)
            pipeline_logger.log_enrichment_log(
                fc_id, fc_name or fc_id,
                rest_id, rest_name or rest_id,
                None, None,  # item_id, item_name
                input_file_link,
                reason
            )
    
    # Log process results
    results = {
        "Total restaurants listed": total_restaurants,
        "Restaurants with raw data": restaurants_with_data,
        "Processed Excel files written": processed_files,
        "Total rows processed": total_rows,
        "Rows with metadata matches": total_matched_rows,
        "Match rate": f"{(total_matched_rows/total_rows*100) if total_rows > 0 else 0:.2f}%"
    }
    pipeline_logger.log_process_results("enrich_data", results)


if __name__ == "__main__":
    main()

