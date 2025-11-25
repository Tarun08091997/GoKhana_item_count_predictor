"""
Goal
----
Take raw restaurant order exports and enrich them into a modeling-ready dataset
where each (foodcourtid, restaurant, menuitemid, date) row contains canonical
metadata, total units sold, weather/holiday context, and gap-filled date ranges.

End-to-End Workflow
-------------------
1. Discover Work: Read `fetch_progress_test.json` to list every foodcourt and its
   restaurants along with city IDs used later for weather joins.
2. Load Raw Orders: For each restaurant that already has a CSV under
   `input_data/food_court_data/{foodcourtid}/{restaurant}.csv`, load the rows and
   prepare for enrichment; skip restaurants without data.
3. Fetch Metadata: Query the `entityrecord` collection (with a local fallback) to
   build a normalized item-name lookup that maps raw names to canonical `menuitemid`,
   `isVeg`, `isSpicy`, and the original entityrecord name.
4. Enrich Orders: Normalize item names in the raw CSV, attach metadata, compute a
   unit `price` (total_price / total_count), and keep track of enrichment progress.
5. Aggregate by IDs: Group strictly on IDs (`foodcourtid`, `restaurant`,
   `menuitemid`, `date`) so each item/day pair has one row with summed counts,
   regardless of mismatched restaurant/foodcourt names in the raw export.
6. Canonicalize Names & Build Date Grid: Collapse metadata to a single canonical
   row per unique ID combo, then cross-join it with the continuous date range
   spanning the first to last observed sale; this ensures downstream models see a
   dense daily series even on zero-sale days.
7. Merge Metrics: Join the aggregated counts back onto the dense grid, fill
   missing count values with zeros, and forward/backfill metadata columns where
   necessary so each date carries the same canonical names and flags.
8. Add External Signals: Using the restaurant's `cityId`, fetch weather rows from
   MySQL tables (`city_<cityId>`), merge them with the grid, join holiday metadata
   from `input_data/holidays_table.csv`, and compute weekday indicator columns.
9. Finalize Schema: Drop intermediate holiday columns, order the final fields, and
   persist the enriched dataframe to `input_data/enriched_data/{foodcourt}/{restaurant}.csv`.
10. Debug Support: When `TESTING` is True, write `_debug` CSV snapshots for every
    key step (enriched, aggregated, grid, weather, final) to simplify auditing.
"""

import importlib
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Tuple, Any, Optional

import pandas as pd
from bson import ObjectId
from pymongo import MongoClient

mysql_connector = None
local_mongo_client = None

from config_parser import ConfigManger

# Feature flag to dump intermediate steps for inspection
TESTING = True

# Configure logging once for the script.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def print_progress_bar(iteration, total, prefix="", suffix="", decimals=1, length=30, fill="█"):
    """Render a simple terminal progress bar (mirrors fetch_restaurant_orders.py)."""
    if total == 0:
        return
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    sys.stdout.write(f"\r{prefix} |{bar}| {percent}% {suffix}")
    sys.stdout.flush()
    if iteration >= total:
        sys.stdout.write("\n")


def save_step_snapshot(base_dir: str, foodcourt_id: str, restaurant_id: str, label: str, df: Optional[pd.DataFrame]):
    """Persist intermediate DataFrame when TESTING flag is enabled."""
    if not TESTING or df is None:
        return
    debug_dir = os.path.join(base_dir, "_debug", foodcourt_id)
    os.makedirs(debug_dir, exist_ok=True)
    safe_label = label.lower().replace(" ", "_")
    path = os.path.join(debug_dir, f"{restaurant_id}_{safe_label}.csv")
    try:
        df.to_csv(path, index=False, encoding="utf-8")
    except Exception as exc:
        logging.warning("Unable to save debug snapshot %s: %s", path, exc)


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
    logging.info("Connected to MongoDB %s.%s", db_name, collection_name)
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
        logging.info("Connected to local MongoDB %s.%s", local_cfg["LOCAL_MONGO_DB"], collection_name)
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
        logging.info("Connected to MySQL database: %s", mysql_cfg["db_name"])
        return conn
    except Exception as exc:
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
    try:
        lookup = fetch_item_metadata(primary_collection, restaurant_id, item_entity_id)
        if lookup:
            return lookup
        logging.warning("Primary metadata lookup empty for restaurant %s; trying local", restaurant_id)
    except Exception as exc:
        logging.error("Primary metadata lookup failed for %s: %s", restaurant_id, exc)

    local_collection = get_local_mongo_collection(local_cfg, local_cfg.get("FOOD_ITEM_COLL", "food_item_record"))
    lookup = fetch_item_metadata(local_collection, restaurant_id, item_entity_id)
    if not lookup:
        raise RuntimeError(f"Unable to fetch item metadata for restaurant {restaurant_id} from both primary and local sources.")
    logging.info("Fetched metadata for restaurant %s from local Mongo", restaurant_id)
    return lookup


def enrich_orders(
    df: pd.DataFrame,
    metadata_lookup: Dict[str, Dict[str, Any]],
    progress_ctx: Dict[str, Any] | None = None,
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

    match_count = 0
    for idx, row_index in enumerate(working_df.index, 1):
        normalized_name = working_df.at[row_index, "__normalized_itemname"]
        metadata = metadata_lookup.get(normalized_name)
        if metadata:
            match_count += 1
            working_df.at[row_index, "menuitemid"] = metadata.get("menuitemid")
            working_df.at[row_index, "isVeg"] = metadata.get("isVeg")
            working_df.at[row_index, "isSpicy"] = metadata.get("isSpicy")

        if progress_prefix:
            print_progress_bar(
                idx,
                total_records,
                prefix=progress_prefix,
                suffix=f"{idx}/{total_records} rows",
                length=40,
            )

    working_df = working_df.drop(columns=["__normalized_itemname"])
    return working_df, match_count


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
            merged_df[col] = merged_df[col].fillna(0)

    # Forward fill remaining metadata columns from grid values
    for col in ["itemname", "is_preorder", "isVeg", "isSpicy", "restaurantname", "foodcourtname"]:
        if col in merged_df.columns and merged_df[col].isna().any():
            merged_df[col] = merged_df[col].ffill().bfill()

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
    if matched_holidays:
        logging.info("Matched %d holiday rows", matched_holidays)

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
    """Load fetch_progress_test.json."""
    with open(progress_path, "r", encoding="utf-8") as progress_file:
        return json.load(progress_file)


def main():
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

    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "input_data")
    progress_path = os.path.join(input_dir, "fetch_progress_test.json")
    holiday_csv_path = os.path.join(input_dir, "holidays_table.csv")
    raw_base_path = os.path.join(input_dir, "food_court_data")
    processed_base_path = os.path.join(input_dir, "enriched_data")
    os.makedirs(processed_base_path, exist_ok=True)

    if not os.path.exists(progress_path):
        logging.error("Progress file not found: %s", progress_path)
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
        logging.warning("Primary Mongo connection failed (%s); trying local", exc)
        local_cfg = config.get("local_mongodb", {})
        fallback_conn_str = local_cfg.get("LOCAL_MONGO_URI")
        fallback_db = local_cfg.get("LOCAL_MONGO_DB")
        fallback_coll = local_cfg.get("FOODORDER_MONGO_DB")
        if not all([fallback_conn_str, fallback_db, fallback_coll]):
            logging.error("Local Mongo fallback not configured; aborting.")
            return
        collection, client = get_mongo_collection(
            fallback_conn_str,
            fallback_db,
            fallback_coll,
        )
    weather_cfg = config.get("weather_record") or config.get("mysql") or {}
    mysql_conn = get_mysql_connection(weather_cfg)
    holiday_df = load_holiday_table(holiday_csv_path)
    logging.info("Loaded %d holiday rows", len(holiday_df))

    total_foodcourts = len(progress)
    total_restaurants = 0
    restaurants_with_data = 0
    processed_files = 0
    total_rows = 0
    total_matched_rows = 0

    try:
        for fc_idx, (foodcourt_id, foodcourt_info) in enumerate(progress.items(), 1):
            restaurants = foodcourt_info.get("restaurants", {})
            total_restaurants_in_fc = len(restaurants)
            city_id = foodcourt_info.get("cityId", "")
            for rest_idx, restaurant_id in enumerate(restaurants.keys(), 1):
                total_restaurants += 1
                raw_csv_path = os.path.join(raw_base_path, foodcourt_id, f"{restaurant_id}.csv")
                if not os.path.exists(raw_csv_path):
                    logging.info("Skipping %s/%s (no raw CSV found)", foodcourt_id, restaurant_id)
                    continue

                restaurants_with_data += 1
                try:
                    df = pd.read_csv(raw_csv_path)
                except Exception as exc:
                    logging.error("Failed to read %s: %s", raw_csv_path, exc)
                    continue

                if df.empty:
                    logging.info("Skipping %s/%s (raw CSV is empty)", foodcourt_id, restaurant_id)
                    continue

                metadata_lookup = get_item_metadata_with_fallback(
                    collection,
                    restaurant_id,
                    item_entity_id,
                    local_cfg,
                )

                progress_ctx = {
                    "fc_idx": fc_idx,
                    "total_fc": total_foodcourts if total_foodcourts else 1,
                    "rest_idx": rest_idx,
                    "total_rest": total_restaurants_in_fc if total_restaurants_in_fc else 1,
                }

                enriched_df, matched_rows = enrich_orders(df, metadata_lookup, progress_ctx=progress_ctx)
                total_rows += len(enriched_df)
                total_matched_rows += matched_rows
                save_step_snapshot(processed_base_path, foodcourt_id, restaurant_id, "step_enriched", enriched_df)

                aggregated_df = aggregate_enriched_orders(enriched_df)
                save_step_snapshot(processed_base_path, foodcourt_id, restaurant_id, "step_aggregated", aggregated_df)

                grid_df = build_item_date_grid(aggregated_df)
                save_step_snapshot(processed_base_path, foodcourt_id, restaurant_id, "step_grid", grid_df)

                merged_grid_df = merge_with_item_date_grid(grid_df, aggregated_df)
                save_step_snapshot(processed_base_path, foodcourt_id, restaurant_id, "step_grid_merged", merged_grid_df)

                weather_df = pd.DataFrame()
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
                save_step_snapshot(processed_base_path, foodcourt_id, restaurant_id, "step_weather", weather_df)

                final_df = merge_weather_into_grid(merged_grid_df, weather_df)
                final_df = merge_holiday_data(final_df, holiday_df)
                final_df = add_weekday_flags(final_df)
                final_df = reorder_final_columns(final_df)
                save_step_snapshot(processed_base_path, foodcourt_id, restaurant_id, "step_final", final_df)

                output_dir = os.path.join(processed_base_path, foodcourt_id)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{restaurant_id}.csv")
                try:
                    export_df = final_df.copy()
                    if not export_df.empty and "date" in export_df.columns:
                        export_df["date"] = pd.to_datetime(export_df["date"]).dt.strftime("%Y-%m-%d")
                    export_df.to_csv(output_path, index=False, encoding="utf-8")
                    processed_files += 1
                    logging.info(
                        "Processed %s/%s -> %s (matched %d of %d rows | final rows %d)",
                        foodcourt_id,
                        restaurant_id,
                        output_path,
                        matched_rows,
                        len(enriched_df),
                        len(export_df),
                    )
                except Exception as exc:
                    logging.error("Failed to write processed CSV for %s/%s: %s", foodcourt_id, restaurant_id, exc)

    finally:
        client.close()
        logging.info("MongoDB connection closed")
        if mysql_conn:
            mysql_conn.close()
            logging.info("MySQL connection closed")

    logging.info("=" * 60)
    logging.info("ENRICHMENT SUMMARY")
    logging.info("=" * 60)
    logging.info("Total restaurants listed: %d", total_restaurants)
    logging.info("Restaurants with raw CSV: %d", restaurants_with_data)
    logging.info("Processed CSV files written: %d", processed_files)
    logging.info("Total rows processed: %d", total_rows)
    logging.info("Rows with metadata matches: %d", total_matched_rows)


if __name__ == "__main__":
    main()

