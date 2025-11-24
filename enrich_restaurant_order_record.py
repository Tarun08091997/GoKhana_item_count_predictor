"""
Enrich restaurant-level order CSVs with metadata from entityrecord items.

Workflow
--------
1. Read the fetch_progress.json file to know which restaurants exist.
2. For every restaurant that already has a raw CSV in input_data/food_court_data,
   query entityrecord for its items (using entityId + data.parentId filter).
3. Build a normalized-name lookup (lowercase + trimmed) so we can match the
   CSV item rows against entityrecord documents.
4. Replace menuitemid with the entityrecord _id and append isVeg/isSpicy flags.
5. Save the enriched CSV in input_data/added_data/{foodcourt}/{restaurant}.csv.
"""

import json
import logging
import os
import sys
from typing import Dict, Tuple, Any

import pandas as pd
from bson import ObjectId
from pymongo import MongoClient

from config_parser import ConfigManger

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
    client = MongoClient(connection_uri)
    db = client.get_database(db_name)
    collection = db.get_collection(collection_name)
    logging.info("Connected to MongoDB %s.%s", db_name, collection_name)
    return collection, client


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


def load_progress(progress_path: str) -> Dict[str, Any]:
    """Load fetch_progress.json."""
    with open(progress_path, "r", encoding="utf-8") as progress_file:
        return json.load(progress_file)


def main():
    config = ConfigManger().read_config(type="config")
    if not config:
        logging.error("Unable to load config/config.json")
        return

    try:
        mongo_cfg = config["mongodb"]
        item_entity_id = config["entity_ids"]["item_entity_id"]
    except KeyError as exc:
        logging.error("Missing key in config: %s", exc)
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "input_data")
    progress_path = os.path.join(input_dir, "fetch_progress.json")
    raw_base_path = os.path.join(input_dir, "food_court_data")
    enriched_base_path = os.path.join(input_dir, "added_data")
    os.makedirs(enriched_base_path, exist_ok=True)

    if not os.path.exists(progress_path):
        logging.error("Progress file not found: %s", progress_path)
        return

    try:
        progress = load_progress(progress_path)
    except json.JSONDecodeError as exc:
        logging.error("Unable to parse progress JSON: %s", exc)
        return

    collection, client = get_mongo_collection(
        mongo_cfg["connection_string"],
        mongo_cfg["db_name"],
        mongo_cfg["entity_record"],
    )

    total_foodcourts = len(progress)
    total_restaurants = 0
    restaurants_with_data = 0
    enriched_files = 0
    total_rows = 0
    total_matched_rows = 0

    try:
        for fc_idx, (foodcourt_id, foodcourt_info) in enumerate(progress.items(), 1):
            restaurants = foodcourt_info.get("restaurants", {})
            total_restaurants_in_fc = len(restaurants)
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

                metadata_lookup = fetch_item_metadata(collection, restaurant_id, item_entity_id)

                progress_ctx = {
                    "fc_idx": fc_idx,
                    "total_fc": total_foodcourts if total_foodcourts else 1,
                    "rest_idx": rest_idx,
                    "total_rest": total_restaurants_in_fc if total_restaurants_in_fc else 1,
                }

                enriched_df, matched_rows = enrich_orders(df, metadata_lookup, progress_ctx=progress_ctx)
                total_rows += len(enriched_df)
                total_matched_rows += matched_rows

                # Ensure output directory exists
                output_dir = os.path.join(enriched_base_path, foodcourt_id)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{restaurant_id}.csv")

                try:
                    enriched_df.to_csv(output_path, index=False, encoding="utf-8")
                    enriched_files += 1
                    logging.info(
                        "Enriched %s/%s -> %s (matched %d of %d rows)",
                        foodcourt_id,
                        restaurant_id,
                        output_path,
                        matched_rows,
                        len(enriched_df),
                    )
                except Exception as exc:
                    logging.error("Failed to write enriched CSV for %s/%s: %s", foodcourt_id, restaurant_id, exc)

    finally:
        client.close()
        logging.info("MongoDB connection closed")

    logging.info("=" * 60)
    logging.info("ENRICHMENT SUMMARY")
    logging.info("=" * 60)
    logging.info("Total restaurants listed: %d", total_restaurants)
    logging.info("Restaurants with raw CSV: %d", restaurants_with_data)
    logging.info("Enriched CSV files written: %d", enriched_files)
    logging.info("Total rows processed: %d", total_rows)
    logging.info("Rows with metadata matches: %d", total_matched_rows)


if __name__ == "__main__":
    main()

