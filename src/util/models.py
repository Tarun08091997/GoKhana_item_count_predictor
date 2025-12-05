"""Model strategy definitions for restaurant demand forecasting."""

from __future__ import annotations

import importlib
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #


@dataclass
class ItemResult:
    foodcourt_id: str
    restaurant_id: str
    item_id: str
    item_name: str
    train_rows: int
    train_rmse: float
    train_rmspe: float


@dataclass
class ModelTrainingResult:
    item_result: ItemResult
    validation_summary: Dict[str, object]
    model_name: str
    output_path: str


@dataclass
class TrainingBundle:
    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    full_item_df: pd.DataFrame
    feature_cols: List[str]
    output_dir: Path
    metadata: Dict[str, str]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def sanitize_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    cleaned = cleaned.strip("_")
    return cleaned or "item"


def _extract_item_id(df: pd.DataFrame) -> str:
    for col in ("item_identifier", "menuitemid", "itemname"):
        if col in df.columns and len(df[col]):
            return str(df[col].iloc[0])
    return ""


def _extract_item_name(df: pd.DataFrame) -> str:
    if "itemname" in df.columns and len(df["itemname"]):
        return str(df["itemname"].iloc[0])
    return _extract_item_id(df)


def _ensure_xgboost():
    try:
        return importlib.import_module("xgboost")
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("xgboost must be installed. Run `pip install xgboost`.") from exc


def _prepare_matrices(
    df: pd.DataFrame, feature_cols: List[str]
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    meta_cols = ["date"]
    if "menuitemid" in df.columns:
        meta_cols.append("menuitemid")
    if "itemname" in df.columns:
        meta_cols.append("itemname")
    meta = df[meta_cols].copy()

    features = df[feature_cols].copy()
    if "date" in features.columns:
        features = features.drop(columns=["date"])
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)

    y = pd.to_numeric(df["count"], errors="coerce").fillna(0.0).values
    return features, y, meta


def _compute_percentage_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(y_true != 0, (y_pred - y_true) / y_true, np.nan)
    return pct


def _compute_rmspe(pct_errors: np.ndarray) -> float:
    if pct_errors.size == 0:
        return float("nan")
    valid = ~np.isnan(pct_errors)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(pct_errors[valid]))))


def _save_results(
    dataset_name: str,
    meta_df: pd.DataFrame,
    features_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pct_errors: np.ndarray,
    output_dir: Path,
):
    # Extract metadata (foodcourt, restaurant, item info) - only first row values
    foodcourt_id = ""
    foodcourt_name = ""
    restaurant_id = ""
    restaurant_name = ""
    item_id = ""
    item_name = ""
    
    if len(meta_df) > 0:
        for col in ["foodcourt_id", "foodcourtid"]:
            if col in meta_df.columns:
                foodcourt_id = str(meta_df[col].iloc[0]) if pd.notna(meta_df[col].iloc[0]) else ""
                break
        for col in ["foodcourt_name", "foodcourtname"]:
            if col in meta_df.columns:
                foodcourt_name = str(meta_df[col].iloc[0]) if pd.notna(meta_df[col].iloc[0]) else ""
                break
        for col in ["restaurant_id", "restaurant"]:
            if col in meta_df.columns:
                restaurant_id = str(meta_df[col].iloc[0]) if pd.notna(meta_df[col].iloc[0]) else ""
                break
        for col in ["restaurant_name", "restaurantname"]:
            if col in meta_df.columns:
                restaurant_name = str(meta_df[col].iloc[0]) if pd.notna(meta_df[col].iloc[0]) else ""
                break
        for col in ["menuitemid", "item_identifier"]:
            if col in meta_df.columns:
                item_id = str(meta_df[col].iloc[0]) if pd.notna(meta_df[col].iloc[0]) else ""
                break
        for col in ["itemname"]:
            if col in meta_df.columns:
                item_name = str(meta_df[col].iloc[0]) if pd.notna(meta_df[col].iloc[0]) else ""
                break
    
    # Create simplified result with only essential columns (data table)
    result_df = pd.DataFrame()
    
    # Add date, actual count, predicted count, error, error percentage
    # Use date type (not datetime) - convert datetime to date
    if "date" in meta_df.columns:
        # Convert to datetime first, then to date type (not datetime)
        date_series = pd.to_datetime(meta_df["date"])
        result_df["date"] = date_series.dt.date
    result_df["actual_count"] = y_true
    # Round predicted values before calculating errors
    y_pred_rounded = np.round(y_pred).astype(int)
    result_df["predicted_count"] = y_pred_rounded
    result_df["error"] = y_pred_rounded - y_true
    # Recalculate error percentage using rounded predicted values
    # Suppress divide by zero warning - we handle it with np.where
    with np.errstate(divide='ignore', invalid='ignore'):
        result_df["error_pct"] = np.where(
            y_true != 0,
            ((y_pred_rounded - y_true) / y_true) * 100.0,
            0.0
        )
    result_df["error_pct"] = np.nan_to_num(result_df["error_pct"], nan=0.0)
    
    # Save as Excel with metadata header at top
    output_path_excel = output_dir / f"{dataset_name}_results.xlsx"
    try:
        with pd.ExcelWriter(output_path_excel, engine='openpyxl') as writer:
            # Create metadata DataFrame for header
            metadata_df = pd.DataFrame({
                "Field": ["Foodcourt ID", "Foodcourt Name", "Restaurant ID", "Restaurant Name", "Item ID", "Item Name"],
                "Value": [foodcourt_id, foodcourt_name, restaurant_id, restaurant_name, item_id, item_name]
            })
            
            # Write metadata to first rows
            metadata_df.to_excel(writer, index=False, sheet_name='Results', startrow=0)
            
            # Write empty row for gap
            # Then write data table
            result_df.to_excel(writer, index=False, sheet_name='Results', startrow=len(metadata_df) + 2)
        
        LOGGER.debug("Saved Excel results with metadata header to %s", output_path_excel.name)
    except Exception as exc:
        LOGGER.warning("Failed to save Excel file for %s: %s", dataset_name, exc)
    
    # Also save as CSV (simple format, no metadata header)
    output_path_csv = output_dir / f"{dataset_name}_results.csv"
    result_df.to_csv(output_path_csv, index=False, encoding="utf-8")


def _build_validation_summary(
    bundle: TrainingBundle,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pct_errors: np.ndarray,
) -> Dict[str, object]:
    model_date = pd.to_datetime(bundle.metadata.get("model_date"))
    validation_df = bundle.validation_df

    active_days = int(np.sum(validation_df["count"] > 0)) if not validation_df.empty else 0
    total_count = float(np.nansum(y_true)) if len(y_true) else 0.0
    total_predicted = float(np.nansum(y_pred)) if len(y_pred) else 0.0
    abs_errors = np.abs(y_pred - y_true) if len(y_true) else np.array([])
    avg_abs_dev = float(np.mean(abs_errors)) if abs_errors.size else float("nan")
    
    # Calculate raw_accuracy with proper NaN handling to avoid RuntimeWarning
    if len(pct_errors) == 0:
        raw_accuracy = float("nan")
    else:
        abs_pct_errors = np.abs(pct_errors)
        valid_errors = abs_pct_errors[~np.isnan(abs_pct_errors)]
        if len(valid_errors) > 0:
            raw_accuracy = float(max(0.0, 1.0 - np.mean(valid_errors)))
        else:
            raw_accuracy = float("nan")

    recent_window_start = model_date - pd.Timedelta(days=14)
    recent_activity = bundle.full_item_df[
        (bundle.full_item_df["date"] >= recent_window_start)
        & (bundle.full_item_df["date"] < model_date)
        & (bundle.full_item_df["count"] > 0)
    ]
    active_recently = bool(len(recent_activity))

    summary = {
        "foodcourtid": bundle.metadata.get("foodcourtid", ""),
        "foodcourtname": bundle.metadata.get("foodcourtname", ""),
        "restaurant": bundle.metadata.get("restaurant", ""),
        "restaurantname": bundle.metadata.get("restaurantname", ""),
        "menuitemname": bundle.metadata.get("item_name", bundle.metadata.get("item_slug")),
        "active_days": active_days,
        "total_count": total_count,
        "predicted_count": total_predicted,
        "avg_accuracy": raw_accuracy,
        "percentage_average_accuracy": raw_accuracy * 100 if not np.isnan(raw_accuracy) else float("nan"),
        "average_absolute_deviation": avg_abs_dev,
        "active_in_last_two_weeks": active_recently,
    }
    return summary


# --------------------------------------------------------------------------- #
# Model implementations
# --------------------------------------------------------------------------- #


class BaseModel:
    name = "base"

    def can_train(self, bundle: TrainingBundle) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def train(self, bundle: TrainingBundle) -> ModelTrainingResult:  # pragma: no cover - interface
        raise NotImplementedError


class XGBoostModel(BaseModel):
    name = "xgboost"
    min_train_rows = 20

    def __init__(self):
        self._xgb = None

    def can_train(self, bundle: TrainingBundle) -> bool:
        if bundle.metadata.get("predict_model", "1") != "1":
            return False
        return len(bundle.train_df) >= self.min_train_rows

    def train(self, bundle: TrainingBundle) -> ModelTrainingResult:
        if self._xgb is None:
            self._xgb = _ensure_xgboost()

        # Filter out rows with count = 0 for XGBoost training
        train_df_filtered = bundle.train_df[bundle.train_df["count"] > 0].copy()
        validation_df_filtered = bundle.validation_df[bundle.validation_df["count"] > 0].copy()
        
        # Check if we have enough data after filtering
        if len(train_df_filtered) < self.min_train_rows:
            LOGGER.warning(f"Insufficient non-zero data for XGBoost: {len(train_df_filtered)} rows (need {self.min_train_rows})")
            return None
        
        X_train_df, y_train, train_meta = _prepare_matrices(train_df_filtered, bundle.feature_cols)
        X_val_df, y_val, val_meta = _prepare_matrices(validation_df_filtered, bundle.feature_cols)

        model = self._xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        )
        model.fit(X_train_df.values, y_train)

        y_train_pred = model.predict(X_train_df)
        train_pct = _compute_percentage_errors(y_train, y_train_pred)
        train_rmse = float(np.sqrt(np.mean((y_train_pred - y_train) ** 2)))
        train_rmspe = _compute_rmspe(train_pct)

        y_val_pred = model.predict(X_val_df)
        val_pct = _compute_percentage_errors(y_val, y_val_pred)

        bundle.output_dir.mkdir(parents=True, exist_ok=True)
        slug = bundle.metadata["item_slug"]
        _save_results(f"{slug}_train", train_meta, X_train_df, y_train, y_train_pred, train_pct, bundle.output_dir)
        _save_results(f"{slug}_validation", val_meta, X_val_df, y_val, y_val_pred, val_pct, bundle.output_dir)
        
        # Save XGBoost model to trainedModel/{pipeline_type}/models/XGBoost/
        from src.util.pipeline_utils import get_output_base_dir, get_model_file_name, get_pipeline_type
        OUTPUT_BASE = get_output_base_dir()
        PIPELINE_TYPE = get_pipeline_type()
        XGBOOST_MODELS_DIR = OUTPUT_BASE / "trainedModel" / PIPELINE_TYPE / "models" / "XGBoost"
        XGBOOST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        foodcourt_id = bundle.metadata.get("foodcourtid", "")
        restaurant_id = bundle.metadata.get("restaurant", "")
        item_name = bundle.metadata.get("item_name", slug)
        
        # Extract item_id for file naming
        from src.util.pipeline_utils import update_model_location, get_mongo_names
        item_id = _extract_item_id(bundle.full_item_df)
        
        model_filename = get_model_file_name(foodcourt_id, restaurant_id, item_name, "XGBoost", item_id)
        model_path = XGBOOST_MODELS_DIR / model_filename
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        LOGGER.debug("Saved XGBoost model to %s", model_path)
        
        # Update model_location.xlsx
        fc_name, rest_name = get_mongo_names(foodcourt_id, restaurant_id)
        fc_name, rest_name = get_mongo_names(foodcourt_id, restaurant_id)
        if not fc_name:
            fc_name = foodcourt_id
        if not rest_name:
            rest_name = restaurant_id
        update_model_location(
            foodcourt_id, fc_name or foodcourt_id,
            restaurant_id, rest_name or restaurant_id,
            item_id, _extract_item_name(bundle.full_item_df),
            "XGBoost", model_path
        )
        
        # Also save results to trainedModel/{pipeline_type}/results/ with proper structure
        # Save separate CSV files for training and validation
        from src.util.pipeline_utils import get_result_file_name, get_pipeline_type, get_file_locator
        PIPELINE_TYPE = get_pipeline_type()
        TRAINED_MODEL_RESULTS_DIR = OUTPUT_BASE / "trainedModel" / PIPELINE_TYPE / "results"
        TRAINED_MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        LOGGER.debug(f"Saving XGBoost results to directory: {TRAINED_MODEL_RESULTS_DIR}")
        
        # Create separate training and validation CSV files
        training_filename = get_result_file_name(foodcourt_id, restaurant_id, item_name, "model_generation", "XGBoost", item_id, "training")
        validation_filename = get_result_file_name(foodcourt_id, restaurant_id, item_name, "model_generation", "XGBoost", item_id, "validation")
        training_path = TRAINED_MODEL_RESULTS_DIR / training_filename
        validation_path = TRAINED_MODEL_RESULTS_DIR / validation_filename
        
        LOGGER.debug(f"Training CSV path: {training_path}")
        LOGGER.debug(f"Validation CSV path: {validation_path}")
        
        # Round predicted values first, then calculate errors
        y_train_pred_rounded = np.round(y_train_pred).astype(int)
        y_val_pred_rounded = np.round(y_val_pred).astype(int)
        
        # Create result DataFrame with proper structure
        # Suppress divide by zero warnings - we handle it with np.where
        with np.errstate(divide='ignore', invalid='ignore'):
            # Use date type (not datetime) - convert datetime to date
            # Date is in train_meta, not X_train_df
            train_dates = None
            if "date" in train_meta.columns:
                train_dates = pd.to_datetime(train_meta["date"]).dt.date
            else:
                train_dates = [None] * len(y_train)
            
            train_result_df = pd.DataFrame({
                "date": train_dates,
                "actual_count": y_train,
                "predicted_count": y_train_pred_rounded,
                "err": (y_train_pred_rounded - y_train),
                "pct_error": np.where(
                    y_train != 0,
                    ((y_train_pred_rounded - y_train) / y_train) * 100.0,
                    0.0
                )
            })
            train_result_df["pct_error"] = np.nan_to_num(train_result_df["pct_error"], nan=0.0)
            
            # Use date type (not datetime) - convert datetime to date
            # Date is in val_meta, not X_val_df
            val_dates = None
            if "date" in val_meta.columns:
                val_dates = pd.to_datetime(val_meta["date"]).dt.date
            else:
                val_dates = [None] * len(y_val)
            
            val_result_df = pd.DataFrame({
                "date": val_dates,
                "actual_count": y_val,
                "predicted_count": y_val_pred_rounded,  # Already rounded above
                "err": (y_val_pred_rounded - y_val),
                "pct_error": np.where(
                    y_val != 0,
                    ((y_val_pred_rounded - y_val) / y_val) * 100.0,
                    0.0
                )
            })
            val_result_df["pct_error"] = np.nan_to_num(val_result_df["pct_error"], nan=0.0)
            
            # Note: Metadata columns (foodcourt_id, restaurant_id, item_id, item_name, model_name) 
            # are not included as they can be extracted from the filename
        
        # Save separate CSV files for training and validation
        try:
            if not train_result_df.empty:
                # Ensure directory exists
                training_path.parent.mkdir(parents=True, exist_ok=True)
                train_result_df.to_csv(training_path, index=False, encoding='utf-8')
                LOGGER.debug(f"Saved XGBoost training results to {training_path}")
            else:
                LOGGER.warning(f"Training result DataFrame is empty, skipping save to {training_path}")
        except Exception as exc:
            LOGGER.error(f"Failed to save XGBoost training results to {training_path}: {exc}", exc_info=True)
        
        try:
            if not val_result_df.empty:
                # Ensure directory exists
                validation_path.parent.mkdir(parents=True, exist_ok=True)
                val_result_df.to_csv(validation_path, index=False, encoding='utf-8')
                LOGGER.debug(f"Saved XGBoost validation results to {validation_path}")
            else:
                LOGGER.warning(f"Validation result DataFrame is empty, skipping save to {validation_path}")
        except Exception as exc:
            LOGGER.error(f"Failed to save XGBoost validation results to {validation_path}: {exc}", exc_info=True)
        
        # Add validation file to file_locator (use validation file as primary reference)
        get_file_locator().add_file(
            foodcourt_id, fc_name or foodcourt_id,
            restaurant_id, rest_name or restaurant_id,
            item_id, _extract_item_name(bundle.full_item_df),
            "model_generation", validation_path, model_name="XGBoost"
        )

        validation_summary = _build_validation_summary(bundle, y_val, y_val_pred, val_pct)
        validation_summary["model_name"] = self.name

        bundle.metadata["model_name"] = self.name
        item_result = ItemResult(
            foodcourt_id=bundle.metadata.get("foodcourtid", ""),
            restaurant_id=bundle.metadata.get("restaurant", ""),
            item_id=_extract_item_id(bundle.full_item_df),
            item_name=_extract_item_name(bundle.full_item_df),
            train_rows=len(bundle.train_df),
            train_rmse=train_rmse,
            train_rmspe=train_rmspe,
        )

        return ModelTrainingResult(
            item_result=item_result,
            validation_summary=validation_summary,
            model_name=self.name,
            output_path=str(bundle.output_dir.resolve()),
        )


class WeeklyMovingAverageModel(BaseModel):
    name = "weekly_moving_average"
    window_days = 90  # 3 months (approximately 90 days)
    decay_factor = 0.02  # Mild exponential decay factor (α)

    def can_train(self, bundle: TrainingBundle) -> bool:
        return len(bundle.train_df) >= 1

    def _decay_based_weighted_average(self, df: pd.DataFrame, prediction_date: pd.Timestamp) -> float:
        """
        Method 1: DECAY-BASED WEIGHTS (EXPONENTIAL DECAY BUT MILD)
        
        Uses exponential decay: weight = exp(α * days_ago)
        where α is a small positive constant (mild decay).
        More recent days get exponentially higher weights.
        """
        if df.empty:
            return 0.0
        
        # Get 3 months window (90 days) before prediction date
        window_start = prediction_date - pd.Timedelta(days=self.window_days - 1)
        recent = df[df["date"] >= window_start].copy()
        
        if recent.empty:
            # Fallback to overall average if no data in 3-month window
            all_nonzero = df[df["count"] > 0]["count"]
            return float(all_nonzero.mean() if len(all_nonzero) > 0 else 0.0)
        
        # Filter to only days with sales (count > 0)
        recent_nonzero = recent[recent["count"] > 0].copy()
        
        if recent_nonzero.empty:
            # Fallback to overall average of non-zero days
            all_nonzero = df[df["count"] > 0]["count"]
            return float(all_nonzero.mean() if len(all_nonzero) > 0 else 0.0)
        
        # Calculate days ago from prediction date
        recent_nonzero["days_ago"] = (prediction_date - recent_nonzero["date"]).dt.days
        
        # Calculate exponential decay weights: weight = exp(-α * days_ago)
        # More recent days (smaller days_ago) get higher weights
        # α is small for mild decay
        # Use negative decay_factor so that smaller days_ago (more recent) gets higher weight
        recent_nonzero["weight"] = np.exp(-self.decay_factor * recent_nonzero["days_ago"].values)
        
        # Calculate weighted average
        counts = recent_nonzero["count"].values
        weights = recent_nonzero["weight"].values
        
        weighted_sum = np.sum(weights * counts)
        total_weight = np.sum(weights)
        
        weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0.0
        return float(weighted_avg)

    def _weekday_aware_weighted_average(self, df: pd.DataFrame, prediction_date: pd.Timestamp) -> float:
        """
        Method 2: WEEKDAY-AWARE WEIGHTING (SMART REGULATION)
        
        Gives higher weights to historical data points that fall on the same
        day of the week as the prediction date.
        For example, if predicting for Monday, historical Mondays get 3x weight.
        Same weekday = 3.0, adjacent weekdays = 1.5, others = 1.0
        """
        if df.empty:
            return 0.0
        
        # Get 3 months window (90 days) before prediction date
        window_start = prediction_date - pd.Timedelta(days=self.window_days - 1)
        recent = df[df["date"] >= window_start].copy()
        
        if recent.empty:
            # Fallback to overall average if no data in 3-month window
            all_nonzero = df[df["count"] > 0]["count"]
            return float(all_nonzero.mean() if len(all_nonzero) > 0 else 0.0)
        
        # Filter to only days with sales (count > 0)
        recent_nonzero = recent[recent["count"] > 0].copy()
        
        if recent_nonzero.empty:
            # Fallback to overall average of non-zero days
            all_nonzero = df[df["count"] > 0]["count"]
            return float(all_nonzero.mean() if len(all_nonzero) > 0 else 0.0)
        
        # Get day of week for prediction date (0=Monday, 6=Sunday)
        prediction_weekday = prediction_date.weekday()
        recent_nonzero["weekday"] = pd.to_datetime(recent_nonzero["date"]).dt.weekday
        
        # Calculate weekday-aware weights
        # Same weekday: 3.0, adjacent (±1 day): 1.5, others: 1.0
        def get_weekday_weight(weekday_diff):
            abs_diff = abs(weekday_diff)
            if abs_diff == 0:
                return 3.0  # Same weekday
            elif abs_diff == 1 or abs_diff == 6:  # Adjacent (considering wrap-around Sunday-Monday)
                return 1.5
            else:
                return 1.0
        
        recent_nonzero["weekday_diff"] = (recent_nonzero["weekday"] - prediction_weekday + 7) % 7
        recent_nonzero["weekday_diff"] = recent_nonzero["weekday_diff"].apply(
            lambda x: min(x, 7 - x)  # Handle wrap-around
        )
        recent_nonzero["weight"] = recent_nonzero["weekday_diff"].apply(get_weekday_weight)
        
        # Also apply mild recency boost (exponential decay)
        recent_nonzero["days_ago"] = (prediction_date - recent_nonzero["date"]).dt.days
        # Use negative decay_factor so that smaller days_ago (more recent) gets higher weight
        recency_boost = np.exp(-self.decay_factor * recent_nonzero["days_ago"].values)
        
        # Combine weekday weight with recency boost
        recent_nonzero["weight"] = recent_nonzero["weight"].values * recency_boost
        
        # Calculate weighted average
        counts = recent_nonzero["count"].values
        weights = recent_nonzero["weight"].values
        
        weighted_sum = np.sum(weights * counts)
        total_weight = np.sum(weights)
        
        weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0.0
        return float(weighted_avg)

    def _predict(self, df: pd.DataFrame, avg: float) -> np.ndarray:
        return np.full(len(df), avg, dtype=float) if len(df) else np.array([])

    def train(self, bundle: TrainingBundle) -> ModelTrainingResult:
        model_date = pd.to_datetime(bundle.metadata.get("model_date"))
        
        # Get actual values
        y_train = bundle.train_df["count"].values
        y_val = bundle.validation_df["count"].values

        # Prepare matrices for saving results
        bundle.output_dir.mkdir(parents=True, exist_ok=True)
        slug = bundle.metadata["item_slug"]
        X_train_df, _, train_meta = _prepare_matrices(bundle.train_df, bundle.feature_cols)
        X_val_df, _, val_meta = _prepare_matrices(bundle.validation_df, bundle.feature_cols)

        # Method 1: Decay-based weighted average
        LOGGER.debug("Calculating decay-based weighted average for %s", slug)
        decay_avg = self._decay_based_weighted_average(bundle.train_df, model_date)
        
        y_train_pred_decay = self._predict(bundle.train_df, decay_avg)
        y_val_pred_decay = self._predict(bundle.validation_df, decay_avg)

        train_pct_decay = _compute_percentage_errors(y_train, y_train_pred_decay)
        val_pct_decay = _compute_percentage_errors(y_val, y_val_pred_decay)

        # Save decay-based results
        _save_results(f"{slug}_train_decay", train_meta, X_train_df, y_train, y_train_pred_decay, train_pct_decay, bundle.output_dir)
        _save_results(f"{slug}_validation_decay", val_meta, X_val_df, y_val, y_val_pred_decay, val_pct_decay, bundle.output_dir)

        # Method 2: Weekday-aware weighted average
        LOGGER.debug("Calculating weekday-aware weighted average for %s", slug)
        
        # For validation set, calculate prediction per date (weekday-aware)
        y_val_pred_weekday = np.array([
            self._weekday_aware_weighted_average(
                bundle.train_df, 
                pd.to_datetime(date_val)
            )
            for date_val in bundle.validation_df["date"]
        ])
        
        # For train set, use average across all training dates
        weekday_avg = self._weekday_aware_weighted_average(bundle.train_df, bundle.train_df["date"].max())
        y_train_pred_weekday = self._predict(bundle.train_df, weekday_avg)

        train_pct_weekday = _compute_percentage_errors(y_train, y_train_pred_weekday)
        val_pct_weekday = _compute_percentage_errors(y_val, y_val_pred_weekday)

        # Save weekday-aware results
        _save_results(f"{slug}_train_weekday", train_meta, X_train_df, y_train, y_train_pred_weekday, train_pct_weekday, bundle.output_dir)
        _save_results(f"{slug}_validation_weekday", val_meta, X_val_df, y_val, y_val_pred_weekday, val_pct_weekday, bundle.output_dir)

        # Build validation summaries for both methods
        validation_summary_decay = _build_validation_summary(bundle, y_val, y_val_pred_decay, val_pct_decay)
        validation_summary_decay["model_name"] = f"{self.name}_decay"
        validation_summary_decay["method"] = "decay_based"
        validation_summary_decay["prediction_method"] = "exponential_decay_mild"
        validation_summary_decay["avg_prediction_value"] = decay_avg

        validation_summary_weekday = _build_validation_summary(bundle, y_val, y_val_pred_weekday, val_pct_weekday)
        validation_summary_weekday["model_name"] = f"{self.name}_weekday"
        validation_summary_weekday["method"] = "weekday_aware"
        validation_summary_weekday["prediction_method"] = "weekday_aware_weighting"
        validation_summary_weekday["avg_prediction_value"] = weekday_avg

        # Choose the better performing method based on validation RMSPE
        decay_val_rmspe = _compute_rmspe(val_pct_decay)
        weekday_val_rmspe = _compute_rmspe(val_pct_weekday)
        
        if not np.isnan(decay_val_rmspe) and not np.isnan(weekday_val_rmspe):
            use_method = "decay" if decay_val_rmspe <= weekday_val_rmspe else "weekday"
        else:
            use_method = "decay"  # Default
        
        bundle.metadata["model_name"] = self.name
        bundle.metadata["method_used"] = use_method
        bundle.metadata["decay_val_rmspe"] = decay_val_rmspe
        bundle.metadata["weekday_val_rmspe"] = weekday_val_rmspe
        
        # Calculate metrics for the selected method
        if use_method == "decay":
            y_train_pred_final = y_train_pred_decay
            y_val_pred_final = y_val_pred_decay
            train_pct_final = train_pct_decay
            val_pct_final = val_pct_decay
        else:
            y_train_pred_final = y_train_pred_weekday
            y_val_pred_final = y_val_pred_weekday
            train_pct_final = train_pct_weekday
            val_pct_final = val_pct_weekday
        
        item_result = ItemResult(
            foodcourt_id=bundle.metadata.get("foodcourtid", ""),
            restaurant_id=bundle.metadata.get("restaurant", ""),
            item_id=_extract_item_id(bundle.full_item_df),
            item_name=_extract_item_name(bundle.full_item_df),
            train_rows=len(bundle.train_df),
            train_rmse=float(np.sqrt(np.mean((y_train_pred_final - y_train) ** 2))) if len(y_train) else float("nan"),
            train_rmspe=_compute_rmspe(train_pct_final),
        )

        # Save comparison CSV with both methods side-by-side
        # Use date type (not datetime) - convert datetime to date
        validation_dates = pd.to_datetime(bundle.validation_df["date"]).dt.date if "date" in bundle.validation_df.columns else [None] * len(y_val)
        comparison_data = {
            "date": validation_dates,
            "actual_count": y_val,
            "predicted_decay": y_val_pred_decay,
            "predicted_weekday": y_val_pred_weekday,
            "error_decay": y_val_pred_decay - y_val,
            "error_weekday": y_val_pred_weekday - y_val,
            "error_pct_decay": val_pct_decay * 100,
            "error_pct_weekday": val_pct_weekday * 100,
        }
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(bundle.output_dir / f"{slug}_comparison.csv", index=False, encoding="utf-8")
        LOGGER.debug("Saved comparison results to %s_comparison.csv (Decay RMSPE: %.3f, Weekday RMSPE: %.3f)", 
                   slug, decay_val_rmspe, weekday_val_rmspe)

        # Save model parameters/config for moving average model
        model_config = {
            "model_type": "weekly_moving_average",
            "window_days": self.window_days,
            "decay_factor": self.decay_factor,
            "selected_method": use_method,
            "decay_average": float(decay_avg),
            "weekday_average": float(weekday_avg),
            "decay_val_rmspe": float(decay_val_rmspe) if not np.isnan(decay_val_rmspe) else None,
            "weekday_val_rmspe": float(weekday_val_rmspe) if not np.isnan(weekday_val_rmspe) else None,
            "model_date": bundle.metadata.get("model_date", ""),
            "training_data_start": str(bundle.train_df["date"].min()) if not bundle.train_df.empty else "",
            "training_data_end": str(bundle.train_df["date"].max()) if not bundle.train_df.empty else "",
            "validation_data_start": str(bundle.validation_df["date"].min()) if not bundle.validation_df.empty else "",
            "validation_data_end": str(bundle.validation_df["date"].max()) if not bundle.validation_df.empty else "",
        }
        
        # Save to trainedModel/{pipeline_type}/models/MovingAverage/
        from src.util.pipeline_utils import get_output_base_dir, get_model_file_name, get_pipeline_type
        OUTPUT_BASE = get_output_base_dir()
        PIPELINE_TYPE = get_pipeline_type()
        MOVING_AVG_MODELS_DIR = OUTPUT_BASE / "trainedModel" / PIPELINE_TYPE / "models" / "MovingAverage"
        MOVING_AVG_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        foodcourt_id = bundle.metadata.get("foodcourtid", "")
        restaurant_id = bundle.metadata.get("restaurant", "")
        item_name = bundle.metadata.get("item_name", slug)
        
        # Extract item_id for file naming
        from src.util.pipeline_utils import update_model_location, get_mongo_names
        item_id = _extract_item_id(bundle.full_item_df)
        
        model_filename = get_model_file_name(foodcourt_id, restaurant_id, item_name, "MovingAverage", item_id)
        model_config_path = MOVING_AVG_MODELS_DIR / model_filename
        with open(model_config_path, "wb") as f:
            pickle.dump(model_config, f)
        LOGGER.debug("Saved MovingAverage model to %s", model_config_path)
        
        # Update model_location.xlsx
        fc_name, rest_name = get_mongo_names(foodcourt_id, restaurant_id)
        fc_name, rest_name = get_mongo_names(foodcourt_id, restaurant_id)
        if not fc_name:
            fc_name = foodcourt_id
        if not rest_name:
            rest_name = restaurant_id
        update_model_location(
            foodcourt_id, fc_name or foodcourt_id,
            restaurant_id, rest_name or restaurant_id,
            item_id, _extract_item_name(bundle.full_item_df),
            "MovingAverage", model_config_path
        )
        
        # Also save results to trainedModel/{pipeline_type}/results/ with proper structure
        # Save separate CSV files for training and validation
        from src.util.pipeline_utils import get_result_file_name, get_pipeline_type, get_file_locator
        PIPELINE_TYPE = get_pipeline_type()
        TRAINED_MODEL_RESULTS_DIR = OUTPUT_BASE / "trainedModel" / PIPELINE_TYPE / "results"
        TRAINED_MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        LOGGER.debug(f"Saving MovingAverage results to directory: {TRAINED_MODEL_RESULTS_DIR}")
        
        # Generate separate training and validation filenames
        training_filename = get_result_file_name(foodcourt_id, restaurant_id, item_name, "model_generation", "MovingAverage", item_id, "training")
        validation_filename = get_result_file_name(foodcourt_id, restaurant_id, item_name, "model_generation", "MovingAverage", item_id, "validation")
        training_path = TRAINED_MODEL_RESULTS_DIR / training_filename
        validation_path = TRAINED_MODEL_RESULTS_DIR / validation_filename
        
        LOGGER.debug(f"Training CSV path: {training_path}")
        LOGGER.debug(f"Validation CSV path: {validation_path}")
        
        # Create result DataFrame with proper structure (use selected method)
        # Round predicted values first, then calculate errors
        y_train_pred_rounded = np.round(y_train_pred_final).astype(int)
        y_val_pred_rounded = np.round(y_val_pred_final).astype(int)
        
        # Suppress divide by zero warnings - we handle it with np.where
        with np.errstate(divide='ignore', invalid='ignore'):
            # Use date type (not datetime) - convert datetime to date
            # Date is in train_meta, not X_train_df
            train_dates = None
            if "date" in train_meta.columns:
                train_dates = pd.to_datetime(train_meta["date"]).dt.date
            else:
                train_dates = [None] * len(y_train)
            
            train_result_df = pd.DataFrame({
                "date": train_dates,
                "actual_count": y_train,
                "predicted_count": y_train_pred_rounded,
                "err": (y_train_pred_rounded - y_train),
                "pct_error": np.where(
                    y_train != 0,
                    ((y_train_pred_rounded - y_train) / y_train) * 100.0,
                    0.0
                )
            })
            train_result_df["pct_error"] = np.nan_to_num(train_result_df["pct_error"], nan=0.0)
            
            # Note: Metadata columns (foodcourt_id, restaurant_id, item_id, item_name, model_name) 
            # are not included as they can be extracted from the filename
            
            # Use date type (not datetime) - convert datetime to date
            # Date is in val_meta, not X_val_df
            val_dates = None
            if "date" in val_meta.columns:
                val_dates = pd.to_datetime(val_meta["date"]).dt.date
            else:
                val_dates = [None] * len(y_val)
            
            val_result_df = pd.DataFrame({
                "date": val_dates,
                "actual_count": y_val,
                "predicted_count": y_val_pred_rounded,
                "err": (y_val_pred_rounded - y_val),
                "pct_error": np.where(
                    y_val != 0,
                    ((y_val_pred_rounded - y_val) / y_val) * 100.0,
                    0.0
                )
            })
            val_result_df["pct_error"] = np.nan_to_num(val_result_df["pct_error"], nan=0.0)
            
            # Note: Metadata columns (foodcourt_id, restaurant_id, item_id, item_name, model_name) 
            # are not included as they can be extracted from the filename
        
        # Save separate CSV files for training and validation
        try:
            if not train_result_df.empty:
                # Ensure directory exists
                training_path.parent.mkdir(parents=True, exist_ok=True)
                train_result_df.to_csv(training_path, index=False, encoding='utf-8')
                LOGGER.debug(f"Saved MovingAverage training results to {training_path}")
            else:
                LOGGER.warning(f"Training result DataFrame is empty, skipping save to {training_path}")
        except Exception as exc:
            LOGGER.error(f"Failed to save MovingAverage training results to {training_path}: {exc}", exc_info=True)
        
        try:
            if not val_result_df.empty:
                # Ensure directory exists
                validation_path.parent.mkdir(parents=True, exist_ok=True)
                val_result_df.to_csv(validation_path, index=False, encoding='utf-8')
                LOGGER.debug(f"Saved MovingAverage validation results to {validation_path}")
            else:
                LOGGER.warning(f"Validation result DataFrame is empty, skipping save to {validation_path}")
        except Exception as exc:
            LOGGER.error(f"Failed to save MovingAverage validation results to {validation_path}: {exc}", exc_info=True)
        
        # Add validation file to file_locator (use validation file as primary reference)
        get_file_locator().add_file(
            foodcourt_id, fc_name or foodcourt_id,
            restaurant_id, rest_name or restaurant_id,
            item_id, _extract_item_name(bundle.full_item_df),
            "model_generation", validation_path, model_name="MovingAverage"
        )
        
        LOGGER.debug("Saved model configuration to %s", model_config_path.name)

        # Return primary result with combined summary that includes both methods
        # The validation_summary will be added twice (once for each method) in process_item
        combined_summary = {
            **validation_summary_decay,
            "decay_rmse": float(np.sqrt(np.mean((y_val_pred_decay - y_val) ** 2))) if len(y_val) else float("nan"),
            "weekday_rmse": float(np.sqrt(np.mean((y_val_pred_weekday - y_val) ** 2))) if len(y_val) else float("nan"),
            "decay_val_rmspe": decay_val_rmspe,
            "weekday_val_rmspe": weekday_val_rmspe,
            "best_method": use_method,
        }
        
        # Store both summaries separately for easy access
        combined_summary["_decay_summary"] = validation_summary_decay
        combined_summary["_weekday_summary"] = validation_summary_weekday

        return ModelTrainingResult(
            item_result=item_result,
            validation_summary=combined_summary,
            model_name=self.name,
            output_path=str(bundle.output_dir.resolve()),
        )


class ProphetModel(BaseModel):
    name = "prophet"
    min_train_rows = 20

    def __init__(self):
        self._prophet = None

    def can_train(self, bundle: TrainingBundle) -> bool:
        if bundle.metadata.get("predict_model", "1") != "1":
            return False
        return len(bundle.train_df) >= self.min_train_rows

    def train(self, bundle: TrainingBundle) -> ModelTrainingResult:
        try:
            from prophet import Prophet
        except ImportError as exc:
            LOGGER.error("prophet must be installed. Run `pip install prophet`.")
            return None

        # Prophet needs continuous dates - keep all rows including count=0
        # Ensure dates are continuous (fill missing dates with count=0)
        train_df = bundle.train_df.copy()
        validation_df = bundle.validation_df.copy()
        
        # Create continuous date range
        train_start = train_df["date"].min()
        train_end = train_df["date"].max()
        val_start = validation_df["date"].min()
        val_end = validation_df["date"].max()
        
        all_dates = pd.date_range(start=train_start, end=val_end, freq='D')
        train_dates = pd.date_range(start=train_start, end=train_end, freq='D')
        val_dates = pd.date_range(start=val_start, end=val_end, freq='D')
        
        # Create continuous DataFrames
        train_continuous = pd.DataFrame({"ds": train_dates})
        train_continuous = train_continuous.merge(
            train_df[["date", "count"]].rename(columns={"date": "ds"}),
            on="ds", how="left"
        )
        train_continuous["y"] = train_continuous["count"].fillna(0.0)
        train_continuous = train_continuous[["ds", "y"]]
        
        val_continuous = pd.DataFrame({"ds": val_dates})
        val_continuous = val_continuous.merge(
            validation_df[["date", "count"]].rename(columns={"date": "ds"}),
            on="ds", how="left"
        )
        val_continuous["y"] = val_continuous["count"].fillna(0.0)
        val_continuous = val_continuous[["ds", "y"]]

        # Train Prophet model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(train_continuous)

        # Predict on training and validation
        train_forecast = model.predict(train_continuous[["ds"]])
        val_forecast = model.predict(val_continuous[["ds"]])
        
        y_train_pred = np.maximum(0, np.round(train_forecast["yhat"].values)).astype(int)
        y_val_pred = np.maximum(0, np.round(val_forecast["yhat"].values)).astype(int)
        y_train = train_continuous["y"].values
        y_val = val_continuous["y"].values

        # Calculate metrics
        train_pct = _compute_percentage_errors(y_train, y_train_pred)
        train_rmse = float(np.sqrt(np.mean((y_train_pred - y_train) ** 2)))
        train_rmspe = _compute_rmspe(train_pct)
        val_pct = _compute_percentage_errors(y_val, y_val_pred)

        # Save model
        bundle.output_dir.mkdir(parents=True, exist_ok=True)
        slug = bundle.metadata["item_slug"]
        model_path = bundle.output_dir / f"{slug}_prophet.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save results
        from src.util.pipeline_utils import get_output_base_dir, get_model_file_name, get_result_file_name, get_pipeline_type, update_model_location, get_mongo_names, get_file_locator
        OUTPUT_BASE = get_output_base_dir()
        PIPELINE_TYPE = get_pipeline_type()
        PROPHET_MODELS_DIR = OUTPUT_BASE / "trainedModel" / PIPELINE_TYPE / "models" / "Prophet"
        PROPHET_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        TRAINED_MODEL_RESULTS_DIR = OUTPUT_BASE / "trainedModel" / PIPELINE_TYPE / "results"
        TRAINED_MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        foodcourt_id = bundle.metadata.get("foodcourtid", "")
        restaurant_id = bundle.metadata.get("restaurant", "")
        item_name = bundle.metadata.get("item_name", slug)
        item_id = _extract_item_id(bundle.full_item_df)

        # Save model to proper location
        model_filename = get_model_file_name(foodcourt_id, restaurant_id, item_name, "Prophet", item_id)
        final_model_path = PROPHET_MODELS_DIR / model_filename
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(model_path, final_model_path)

        # Update model location
        fc_name, rest_name = get_mongo_names(foodcourt_id, restaurant_id)
        update_model_location(
            foodcourt_id, fc_name or foodcourt_id,
            restaurant_id, rest_name or restaurant_id,
            item_id, _extract_item_name(bundle.full_item_df),
            "Prophet", final_model_path
        )

        # Save CSV results
        training_filename = get_result_file_name(foodcourt_id, restaurant_id, item_name, "model_generation", "Prophet", item_id, "training")
        validation_filename = get_result_file_name(foodcourt_id, restaurant_id, item_name, "model_generation", "Prophet", item_id, "validation")
        training_path = TRAINED_MODEL_RESULTS_DIR / training_filename
        validation_path = TRAINED_MODEL_RESULTS_DIR / validation_filename

        train_dates = train_continuous["ds"].dt.date
        val_dates = val_continuous["ds"].dt.date

        with np.errstate(divide='ignore', invalid='ignore'):
            train_result_df = pd.DataFrame({
                "date": train_dates,
                "actual_count": y_train,
                "predicted_count": y_train_pred,
                "err": (y_train_pred - y_train),
                "pct_error": np.where(
                    y_train != 0,
                    ((y_train_pred - y_train) / y_train) * 100.0,
                    0.0
                )
            })
            train_result_df["pct_error"] = np.nan_to_num(train_result_df["pct_error"], nan=0.0)

            val_result_df = pd.DataFrame({
                "date": val_dates,
                "actual_count": y_val,
                "predicted_count": y_val_pred,
                "err": (y_val_pred - y_val),
                "pct_error": np.where(
                    y_val != 0,
                    ((y_val_pred - y_val) / y_val) * 100.0,
                    0.0
                )
            })
            val_result_df["pct_error"] = np.nan_to_num(val_result_df["pct_error"], nan=0.0)

        train_result_df.to_csv(training_path, index=False, encoding='utf-8')
        val_result_df.to_csv(validation_path, index=False, encoding='utf-8')

        # Add to file locator
        get_file_locator().add_file(
            foodcourt_id, fc_name or foodcourt_id,
            restaurant_id, rest_name or restaurant_id,
            item_id, _extract_item_name(bundle.full_item_df),
            "model_generation", validation_path, model_name="Prophet"
        )

        validation_summary = _build_validation_summary(bundle, y_val, y_val_pred, val_pct)
        validation_summary["model_name"] = self.name

        item_result = ItemResult(
            foodcourt_id=bundle.metadata.get("foodcourtid", ""),
            restaurant_id=bundle.metadata.get("restaurant", ""),
            item_id=_extract_item_id(bundle.full_item_df),
            item_name=_extract_item_name(bundle.full_item_df),
            train_rows=len(train_df),
            train_rmse=train_rmse,
            train_rmspe=train_rmspe,
        )

        return ModelTrainingResult(
            item_result=item_result,
            validation_summary=validation_summary,
            model_name=self.name,
            output_path=str(bundle.output_dir.resolve()),
        )


AVAILABLE_MODELS: List[BaseModel] = [
    XGBoostModel(),
    ProphetModel(),
    WeeklyMovingAverageModel(),
]


