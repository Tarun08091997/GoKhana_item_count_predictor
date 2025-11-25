"""
Goal
----
Train and evaluate per-item XGBoost regressors using the engineered features
from `input_data/preprocessed_data`, then store predictions, models, and a
restaurant-level analysis summary under `input_data/XGBosst/`.

Workflow
--------
1. Discover Inputs: Iterate through `input_data/preprocessed_data/{foodcourt}/`
   directories, loading each `{restaurant}.csv` into memory and parsing the date
   column plus the canonical item identifier (`menuitemid` or fallback `itemname`).
2. Feature Selection: Keep only the feature columns present in the file
   (`FEATURE_COLUMNS` acts as the allow-list). Skip datasets that lack the minimum
   number of usable features so we do not train ill-posed models.
3. Item Filtering: Group rows by `menuitemid` (or fallback identifier) and drop
   items flagged with `predict_model == 0`, beverages, or series shorter than the
   required training window.
4. Dataset Splits: Split each item’s timeseries into a pre period (default:
   dates before `SPLIT_DATE`, falling back to a 70/30 chronological split) and a
   post period. Require at least `MIN_TRAIN_ROWS` before training.
5. Model Training: For every eligible item, convert features to numeric matrices,
   train an `xgboost.XGBRegressor` on the pre period, and capture RMSE metrics on
   both the training (pre) set and the post set.
6. Persist Outputs: Under `input_data/XGBosst/{foodcourt}/{restaurant}/{item}/`,
   write `{item_slug}_pre_results.csv`, `{item_slug}_post_results.csv`, and
   `{item_slug}.json` (serialized model). Result CSVs include metadata columns,
   features, actual counts, predictions, and residuals.
7. Analysis Summary: Append per-item metrics (train/test rows, RMSEs, identifiers)
   to an in-memory list that is later written/appended to
   `input_data/XGBosst/analysis.csv` with blank rows separating restaurants.
8. Logging & Errors: Emit informative logs for skipped items (e.g., insufficient
   rows, missing features) and continue processing the remaining restaurants.
"""

import importlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import xgboost as xgb  # type: ignore
else:
    xgb = None  # type: ignore


def ensure_xgboost():
    global xgb
    if xgb is None:
        try:
            xgb = importlib.import_module("xgboost")
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "xgboost must be installed to run models.py. "
                "Install with `pip install xgboost`."
            ) from exc


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input_data"
PREPROCESSED_DIR = INPUT_DIR / "preprocessed_data"
OUTPUT_BASE_DIR = INPUT_DIR / "XGBosst"
ANALYSIS_PATH = OUTPUT_BASE_DIR / "analysis.csv"

FEATURE_COLUMNS = [
    "price",
    "isVeg",
    "isSpicy",
    "is_mon",
    "is_tue",
    "is_wed",
    "is_thu",
    "is_fri",
    "is_sat",
    "is_sun",
    "predict_model",
    "avg_3_day",
    "avg_7_day",
    "lag_7_day",
    "lag_14_day",
    "lag_21_day",
    "lag_28_day",
    "avg_1_month",
    "avg_2_month",
    "avg_3_month",
]

ITEM_IDENTIFIER_COL = "item_identifier"
SPLIT_DATE = pd.Timestamp("2025-10-01")
MIN_TRAIN_ROWS = 20
MIN_FEATURES_REQUIRED = 8

BEVERAGE_PATTERN = re.compile(r"(coffee|coffe|chai|tea|drink|juice|shake|mocktail)", re.IGNORECASE)


@dataclass
class ItemResult:
    foodcourt_id: str
    restaurant_id: str
    item_id: str
    item_name: str
    train_rows: int
    test_rows: int
    train_rmse: float
    post_rmse: float


# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #


def sanitize_name(name: str) -> str:
    """Make filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "item"


def load_preprocessed_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Missing 'date' column in preprocessed file")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "menuitemid" in df.columns:
        df[ITEM_IDENTIFIER_COL] = df["menuitemid"].astype(str)
    else:
        logging.warning("Column 'menuitemid' missing in %s; using itemname as identifier", path)
        if "itemname" not in df.columns:
            raise ValueError("Missing both 'menuitemid' and 'itemname' columns")
        df[ITEM_IDENTIFIER_COL] = df["itemname"].astype(str)

    df = df.dropna(subset=["date", ITEM_IDENTIFIER_COL])
    return df.sort_values("date").reset_index(drop=True)


def subset_features(df: pd.DataFrame) -> List[str]:
    cols = [col for col in FEATURE_COLUMNS if col in df.columns]
    return cols


def split_datasets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pre_df = df[df["date"] < SPLIT_DATE].copy()
    post_df = df[df["date"] >= SPLIT_DATE].copy()
    if pre_df.empty or post_df.empty:
        n = len(df)
        split_idx = max(1, int(n * 0.7))
        if split_idx >= n:
            split_idx = n - 1
        pre_df = df.iloc[:split_idx].copy()
        post_df = df.iloc[split_idx:].copy()
    return pre_df, post_df


def prepare_matrices(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    meta_cols = ["date"]
    if "menuitemid" in df.columns:
        meta_cols.append("menuitemid")
    if "itemname" in df.columns:
        meta_cols.append("itemname")
    meta = df[meta_cols].copy()
    X = df[feature_cols].copy()
    if "date" in X.columns:
        X = X.drop(columns=["date"])
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    y = pd.to_numeric(df["count"], errors="coerce").fillna(0.0).values
    return X, y, meta


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> "xgb.XGBRegressor":
    ensure_xgboost()
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: "xgb.XGBRegressor", X: pd.DataFrame, y: np.ndarray) -> Tuple[np.ndarray, float]:
    preds = model.predict(X)
    rmse = float(np.sqrt(np.mean((preds - y) ** 2))) if len(y) else float("nan")
    return preds, rmse


def save_results(
    dataset_name: str,
    meta_df: pd.DataFrame,
    df_features: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
):
    result_df = pd.concat(
        [meta_df.reset_index(drop=True), df_features.reset_index(drop=True)],
        axis=1,
    )
    if "predict_model" in result_df.columns:
        result_df = result_df.drop(columns=["predict_model"])
    result_df["actual_count"] = y_true
    result_df["predicted_count"] = y_pred
    result_df["error"] = y_pred - y_true
    result_df.to_csv(output_dir / f"{dataset_name}_results.csv", index=False, encoding="utf-8")


def process_item(
    df_item: pd.DataFrame,
    feature_cols: List[str],
    item_slug: str,
    output_dir: Path,
) -> Optional[ItemResult]:
    if len(df_item) < MIN_TRAIN_ROWS + 5:
        logging.info("Skipping item %s (insufficient rows)", item_slug)
        return None

    pre_df, post_df = split_datasets(df_item)
    if len(pre_df) < MIN_TRAIN_ROWS or len(post_df) == 0:
        logging.info("Skipping item %s (split too small)", item_slug)
        return None

    X_train_df, y_train, train_meta = prepare_matrices(pre_df, feature_cols)
    X_test_df, y_test, test_meta = prepare_matrices(post_df, feature_cols)

    model = train_model(X_train_df.values, y_train)

    y_train_pred, train_rmse = evaluate(model, X_train_df, y_train)
    y_test_pred, post_rmse = evaluate(model, X_test_df, y_test)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(f"{item_slug}_pre", train_meta, X_train_df, y_train, y_train_pred, output_dir)
    save_results(f"{item_slug}_post", test_meta, X_test_df, y_test, y_test_pred, output_dir)
    model.save_model(output_dir / f"{item_slug}.json")

    return ItemResult(
        foodcourt_id=df_item["foodcourtid"].iloc[0],
        restaurant_id=df_item["restaurant"].iloc[0],
        item_id=str(df_item[ITEM_IDENTIFIER_COL].iloc[0]),
        item_name=str(df_item["itemname"].iloc[0]) if "itemname" in df_item.columns else str(df_item[ITEM_IDENTIFIER_COL].iloc[0]),
        train_rows=len(pre_df),
        test_rows=len(post_df),
        train_rmse=train_rmse,
        post_rmse=post_rmse,
    )


def append_analysis(summary_rows: List[Dict[str, object]]):
    if not summary_rows:
        return
    summary_df = pd.DataFrame(summary_rows)
    try:
        if ANALYSIS_PATH.exists():
            existing = pd.read_csv(ANALYSIS_PATH)
            summary_df = pd.concat([existing, summary_df], ignore_index=True)
        summary_df.to_csv(ANALYSIS_PATH, index=False, encoding="utf-8")
    except PermissionError:
        logging.error("Permission denied writing analysis to %s", ANALYSIS_PATH)


# --------------------------------------------------------------------------- #
# Main execution
# --------------------------------------------------------------------------- #


def main():
    ensure_xgboost()
    if not PREPROCESSED_DIR.exists():
        logging.error("Preprocessed directory not found: %s", PREPROCESSED_DIR)
        return

    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, object]] = []

    for fc_dir in sorted(PREPROCESSED_DIR.iterdir()):
        if not fc_dir.is_dir():
            continue

        for csv_path in sorted(fc_dir.glob("*.csv")):
            restaurant_id = csv_path.stem
            try:
                df = load_preprocessed_file(csv_path)
            except Exception as exc:
                logging.error("Skipping %s: %s", csv_path, exc)
                continue

            feature_cols = subset_features(df)
            if len(feature_cols) < MIN_FEATURES_REQUIRED:
                logging.warning("Skipping %s (not enough feature columns)", csv_path)
                continue

            df_grouped = df.groupby(ITEM_IDENTIFIER_COL)
            restaurant_rows: List[Dict[str, object]] = []

            for item_id, item_df in df_grouped:
                if "predict_model" in item_df.columns:
                    if item_df["predict_model"].iloc[0] == 0:
                        logging.info("Skipping %s (predict_model == 0)", item_id)
                        continue
                item_slug = sanitize_name(str(item_df["itemname"].iloc[0]))
                item_out_dir = OUTPUT_BASE_DIR / fc_dir.name / restaurant_id / item_slug
                result = process_item(item_df, feature_cols, item_slug, item_out_dir)
                if result:
                    restaurant_rows.append(
                        {
                            "foodcourtid": result.foodcourt_id,
                            "restaurant": result.restaurant_id,
                            "item_id": result.item_id,
                            "item_name": result.item_name,
                            "train_rows": result.train_rows,
                            "test_rows": result.test_rows,
                            "train_rmse": round(result.train_rmse, 3),
                            "post_rmse": round(result.post_rmse, 3),
                        }
                    )

            if restaurant_rows:
                summary_rows.extend(restaurant_rows)
                summary_rows.append({})  # blank row separator
                logging.info(
                    "Processed restaurant %s/%s with %d items",
                    fc_dir.name,
                    restaurant_id,
                    len(restaurant_rows),
                )

    append_analysis(summary_rows)
    logging.info("XGBoost modeling complete. Analysis stored in %s", ANALYSIS_PATH)


if __name__ == "__main__":
    main()

