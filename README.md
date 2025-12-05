# GoKhana Demand Forecasting Pipeline

## Project AIM

The GoKhana Demand Forecasting Pipeline is an end-to-end machine learning system designed to predict restaurant menu item demand across multiple food courts. The pipeline automates data extraction, enrichment, preprocessing, model training, and result compilation to generate accurate demand forecasts that help optimize inventory management, reduce waste, and improve operational efficiency.

### Key Objectives:
- **Automated Data Pipeline**: Extract historical order data from MongoDB and enrich it with external signals (weather, holidays)
- **Intelligent Model Selection**: Automatically assign appropriate forecasting models (XGBoost or Moving Average) based on data characteristics
- **Scalable Processing**: Process thousands of foodcourt-restaurant-item combinations efficiently
- **Comprehensive Logging**: Track all processing steps, errors, and file locations for easy debugging and auditing
- **Flexible Retraining**: Support selective retraining at foodcourt, restaurant, or item level via configuration

---

## Pipeline Architecture

The pipeline consists of **6 sequential steps**, each transforming data and passing it to the next stage:

```
Data Fetch → Data Enrichment → Preprocessing → Model Generation → Postprocessing → Compiled Results
```

---

## Pipeline Steps

### Step 1: Data Fetch (`step1_fetch_data.py`)

**Purpose**: Extract raw restaurant order data from MongoDB

**Input Sources**:
- MongoDB `foodorder` collection (restaurant orders)
- `input_data/FR_data.json` (foodcourt/restaurant configuration)

**Process**:
1. Connects to MongoDB using credentials from `config/config.json`
2. Reads foodcourt and restaurant list from `FR_data.json`
3. For each restaurant, runs aggregation pipelines to fetch:
   - Regular completed orders
   - Preorders
4. Explodes order items into individual rows
5. Aggregates by (foodcourt, restaurant, menu item, date)
6. Calculates total quantity sold and revenue per item/day
7. Normalizes dates to IST timezone

**Output**:
- Parquet files: `input_data/fetched_data/{foodcourt_id}/{restaurant_id}.parquet`
- Contains: `foodcourtid`, `restaurant`, `menuitemid`, `itemname`, `date`, `total_count`, `total_price`, `is_preorder`

**Key Features**:
- Supports incremental updates (`prod_mode`)
- Progress tracking in `FR_data.json`
- Handles timezone conversion (IST)

---

### Step 2: Data Enrichment (`step2_data_enrichment.py`)

**Purpose**: Enrich raw order data with metadata, weather, and holiday information

**Input Sources**:
- Parquet files from Step 1
- MongoDB `entityrecord` collection (item metadata)
- MySQL weather tables (`city_{cityId}`)
- `input_data/holidays_table.csv`

**Process**:
1. Loads raw parquet files for each restaurant
2. Fetches item metadata from MongoDB:
   - Item canonical names
   - `isVeg`, `isSpicy` flags
   - Menu item IDs
3. Normalizes and matches item names
4. Aggregates by (foodcourtid, restaurant, menuitemid, date)
5. Creates dense date grid (fills missing dates with zeros)
6. Merges weather data from MySQL:
   - Temperature (max/min)
   - Precipitation
   - Weather codes
7. Merges holiday data:
   - Holiday flags (`is_minor`, `is_major`, `is_sandwich`)
8. Adds weekday indicators (`is_mon` through `is_sun`)
9. Calculates unit price (`total_price / total_count`)

**Output**:
- Excel files: `output_data/enrich_data/{foodcourt_id}/{fc_id}_{rest_id}_{item_name}_enrich_data.xlsx`
- Contains: All metadata, weather, holidays, weekday flags, price, count

**Key Features**:
- Handles missing metadata gracefully (fallback to local MongoDB)
- Gap-fills missing dates for continuous time series
- One file per item for efficient downstream processing

---

### Step 3: Data Preprocessing (`step3_data_preprocessing.py`)

**Purpose**: Prepare enriched data for machine learning models

**Input Sources**:
- Excel files from Step 2
- `input_data/train_model_for_items.csv` (filter list)

**Process**:
1. Loads enriched Excel files (one per item)
2. Filters out beverage/MRP items (coffee, tea, drinks, etc.)
3. Removes leading zero-count rows (finds first sale date per item)
4. Assigns `predict_model` value:
   - **Model 1 (XGBoost)**: ≥6 months data span + recent_total > 50 + active
   - **Model 2 (Moving Average)**: 3-6 months data span
   - **Model 3 (Moving Average)**: <3 months or low activity
5. Generates temporal features:
   - Rolling averages: `avg_3_day`, `avg_7_day`
   - Lag features: `lag_7_day`, `lag_14_day`, `lag_21_day`, `lag_28_day`
   - Monthly averages: `avg_1_month`, `avg_2_month`, `avg_3_month`
6. Reorders columns for model compatibility

**Output**:
- Excel files: `output_data/preprocessing/{foodcourt_id}/{fc_id}_{rest_id}_{item_name}_preprocessing.xlsx`
- Contains: All features, `predict_model` assignment, temporal features

**Key Features**:
- Intelligent model assignment based on data quality
- Feature engineering for time series patterns
- Handles items with minimal data (assigns to Model 3)

---

### Step 4: Model Generation (`step4_model_generation.py`)

**Purpose**: Train forecasting models for each item

**Input Sources**:
- Excel files from Step 3
- `input_data/train_model_for_items.csv` (required filter)

**Process**:
1. Loads preprocessed data for each item
2. Splits data into training and validation sets:
   - Training: All data before validation window
   - Validation: Last 7 days before `MODEL_DATE` (2025-11-02)
3. Selects model based on `predict_model`:
   - **XGBoost Model** (`predict_model=1`):
     - Gradient boosting regressor
     - 300 estimators, learning rate 0.05
     - Uses all temporal features
   - **Moving Average Model** (`predict_model=2 or 3`):
     - Two methods: Decay-based and Weekday-aware
     - Selects best method based on validation RMSPE
4. Trains model and generates predictions
5. Calculates metrics:
   - RMSE, RMSPE
   - Average absolute deviation
   - Percentage errors

**Output**:
- Model files: `output_data/trainedModel/models/{model_type}/{fc_id}_{rest_id}_{item_name}_{model_name}.pkl`
- Result files: `output_data/trainedModel/results/{fc_id}_{rest_id}_{item_name}_model_generation_{model_name}.xlsx`
  - Sheets: "Training Data", "Validation Data"
  - Columns: `date`, `actual_count`, `predicted_count`, `err`, `pct_error`

**Key Features**:
- Automatic model selection
- Moving Average uses best of two methods
- Comprehensive validation metrics

---

### Step 5: Postprocessing (`step5_postprocessing.py`)

**Purpose**: Redistribute predictions based on day-of-week patterns

**Input Sources**:
- Validation predictions from Step 4
- Training data from Step 3

**Process**:
1. Loads validation predictions (7-day window)
2. Loads training data for the same item
3. Calculates day-of-week percentages from last 3 months of training data:
   - Groups by week
   - Calculates % contribution of each weekday (Mon-Sun)
   - Averages across weeks
4. Redistributes 7-day prediction totals:
   - Preserves total predicted count
   - Distributes based on historical weekday patterns
5. Calculates postprocessing errors

**Output**:
- Excel files: `output_data/postprocessing/{foodcourt_id}/{fc_id}_{rest_id}_{item_name}_postprocessing.xlsx`
- Contains: Original predictions, redistributed predictions, errors

**Key Features**:
- Improves prediction accuracy by accounting for weekly patterns
- Maintains total predicted volume
- Only affects validation predictions

---

### Step 6: Compiled Result Generation (`step6_compiled_result_generation.py`)

**Purpose**: Compile all model errors into comprehensive summary tables

**Input Sources**:
- Model results from Step 4
- Postprocessing results from Step 5
- `input_data/train_model_for_items.csv`

**Process**:
1. Loads validation results for all models (XGBoost, MovingAverage, Postprocessing)
2. For each item, calculates:
   - Average absolute error
   - Average percentage error
   - Capped average error (errors capped at 100%)
3. Creates two summary files:
   - **compiled_error.xlsx**: Raw errors (no cap)
   - **capped_compiled_data.xlsx**: Capped errors (for averaging)

**Output**:
- `output_data/compiled_results/compiled_error.xlsx`
- `output_data/compiled_results/capped_compiled_data.xlsx`
- Columns: Foodcourt/Restaurant/Item info, error metrics per model

**Key Features**:
- Side-by-side comparison of all models
- Both capped and uncapped error metrics
- One row per item with all model results

---

## File Structure

### Core Pipeline Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `run_pipeline.py` | Main orchestrator | Coordinates all 6 steps, handles retrain logic, saves logs |
| `step1_fetch_data.py` | Data extraction | Fetches orders from MongoDB, saves to Parquet |
| `step2_data_enrichment.py` | Data enrichment | Adds metadata, weather, holidays, creates item-level files |
| `step3_data_preprocessing.py` | Feature engineering | Generates temporal features, assigns models |
| `step4_model_generation.py` | Model training | Trains XGBoost/MovingAverage, saves models and results |
| `step5_postprocessing.py` | Prediction refinement | Redistributes predictions by weekday patterns |
| `step6_compiled_result_generation.py` | Result compilation | Aggregates errors across all models |

### Utility Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `pipeline_utils.py` | Core utilities | File locator, logging, Excel operations, MongoDB name fetching |
| `filter_utils.py` | Filter management | Loads `train_model_for_items.csv`, item filtering logic |
| `models.py` | Model definitions | XGBoostModel, WeeklyMovingAverageModel classes |
| `config_parser.py` | Configuration | Reads `config/config.json` for MongoDB/MySQL credentials |

### Configuration Files

| File | Purpose |
|------|---------|
| `config/config.json` | MongoDB and MySQL connection settings |
| `input_data/FR_data.json` | Foodcourt/restaurant configuration and progress tracking |
| `input_data/retrain.json` | Retrain configuration (which items to force retrain) |
| `input_data/train_model_for_items.csv` | Filter list (which items to process) |
| `input_data/holidays_table.csv` | Holiday calendar data |

### Output Structure

```
output_data/
├── data_fetch/          # (Future: fetched data copies)
├── enrich_data/         # Enriched Excel files (one per item)
│   └── {fc_id}/
│       └── {fc_id}_{rest_id}_{item_name}_enrich_data.xlsx
├── preprocessing/       # Preprocessed Excel files (one per item)
│   └── {fc_id}/
│       └── {fc_id}_{rest_id}_{item_name}_preprocessing.xlsx
├── trainedModel/
│   ├── models/         # Trained model files (.pkl)
│   │   ├── XGBoost/
│   │   └── MovingAverage/
│   └── results/        # Model results (Excel)
│       └── {fc_id}_{rest_id}_{item_name}_model_generation_{model}.xlsx
├── postprocessing/     # Postprocessed predictions
│   └── {fc_id}/
│       └── {fc_id}_{rest_id}_{item_name}_postprocessing.xlsx
├── compiled_results/   # Final error summaries
│   ├── compiled_error.xlsx
│   └── capped_compiled_data.xlsx
├── logs/               # Pipeline logs
│   └── pipeline_logs.xlsx
└── file_locator.xlsx   # Hyperlinks to all output files
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL DATA SOURCES                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│   MongoDB     │         │     MySQL     │         │  CSV Files    │
│               │         │               │         │               │
│ • foodorder   │         │ • city_{id}   │         │ • holidays_   │
│   collection  │         │   tables      │         │   table.csv   │
│               │         │               │         │               │
│ • entityrecord│         │ Weather data: │         │ • train_model │
│   collection  │         │ • temperature │         │   _for_items  │
│               │         │ • precipitation│        │               │
│ • food_court_ │         │ • weather_code│         │ • FR_data.json│
│   record      │         │               │         │               │
│               │         │               │         │ • retrain.json │
│ • restaurant_ │         │               │         │               │
│   data        │         │               │         │               │
└───────────────┘         └───────────────┘         └───────────────┘
        │                           │                           │
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   STEP 1: FETCH DATA          │
                    │   step1_fetch_data.py         │
                    │                               │
                    │ • Query MongoDB foodorder    │
                    │ • Aggregate orders by item   │
                    │ • Save to Parquet             │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   STEP 2: DATA ENRICHMENT     │
                    │   step2_data_enrichment.py    │
                    │                               │
                    │ • Fetch item metadata (Mongo)│
                    │ • Fetch weather (MySQL)       │
                    │ • Merge holidays (CSV)        │
                    │ • Create date grid            │
                    │ • Save Excel (one per item)   │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   STEP 3: PREPROCESSING       │
                    │   step3_data_preprocessing.py │
                    │                               │
                    │ • Filter beverages            │
                    │ • Generate temporal features  │
                    │ • Assign predict_model        │
                    │ • Save Excel (one per item)   │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   STEP 4: MODEL GENERATION   │
                    │   step4_model_generation.py   │
                    │                               │
                    │ • Split train/validation      │
                    │ • Train XGBoost or MA         │
                    │ • Save model (.pkl)           │
                    │ • Save results (Excel)        │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   STEP 5: POSTPROCESSING      │
                    │   step5_postprocessing.py     │
                    │                               │
                    │ • Calculate weekday patterns │
                    │ • Redistribute predictions    │
                    │ • Save Excel                  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   STEP 6: COMPILED RESULTS     │
                    │   step6_compiled_result_       │
                    │   generation.py               │
                    │                               │
                    │ • Aggregate all model errors  │
                    │ • Create summary tables       │
                    │ • Save compiled_error.xlsx    │
                    │ • Save capped_compiled_data   │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │        OUTPUT DATA            │
                    │                               │
                    │ • Models (.pkl files)        │
                    │ • Results (Excel files)       │
                    │ • Logs (pipeline_logs.xlsx)   │
                    │ • File locator (hyperlinks)  │
                    └───────────────────────────────┘
```

---

## Data Transformations

### Step 1 → Step 2
- **Input**: Raw MongoDB orders (JSON documents)
- **Transformation**: 
  - Explode order items → individual rows
  - Aggregate by (foodcourt, restaurant, item, date)
  - Calculate totals (count, price)
- **Output**: Parquet files with structured order data

### Step 2 → Step 3
- **Input**: Enriched data with metadata, weather, holidays
- **Transformation**:
  - Filter beverages/MRP items
  - Generate temporal features (lags, averages)
  - Assign model type (1, 2, or 3)
- **Output**: Preprocessed data with features ready for ML

### Step 3 → Step 4
- **Input**: Preprocessed data with features
- **Transformation**:
  - Split into train/validation sets
  - Train model (XGBoost or Moving Average)
  - Generate predictions
  - Calculate errors
- **Output**: Trained models and prediction results

### Step 4 → Step 5
- **Input**: Validation predictions (7 days)
- **Transformation**:
  - Calculate historical weekday patterns
  - Redistribute predictions maintaining total
- **Output**: Postprocessed predictions with weekday adjustment

### Step 5 → Step 6
- **Input**: All model results (XGBoost, MA, Postprocessing)
- **Transformation**:
  - Aggregate errors per item
  - Calculate average and capped errors
  - Create summary tables
- **Output**: Comprehensive error compilation

---

## Database Connections

### MongoDB Collections

| Collection | Purpose | Data Fetched |
|------------|---------|--------------|
| `foodorder` | Restaurant orders | Order details, items, dates, prices |
| `entityrecord` | Item metadata | Item names, isVeg, isSpicy flags |
| `food_court_record` | Foodcourt info | Foodcourt names, city IDs |
| `restaurant_data` | Restaurant info | Restaurant names |

### MySQL Tables

| Table Pattern | Purpose | Data Fetched |
|---------------|---------|--------------|
| `city_{cityId}` | Weather data | Temperature, precipitation, weather codes per date |

---

## Usage

### Basic Usage

```bash
# Run entire pipeline
python run_pipeline.py

# Run in production mode (incremental updates)
python run_pipeline.py --prod-mode
```

### Retraining Specific Items

Edit `input_data/retrain.json`:

```json
{
  "enrich_data": [
    {"foodcourt_id": "5f338dce8f277f4c2f4ac99f", "restaurant_id": "60f172e2d1b6e328e744cf65", "item_name": "Biryani"}
  ],
  "model_generation": [
    {"foodcourt_id": "5f338dce8f277f4c2f4ac99f", "restaurant_id": "60f172e2d1b6e328e744cf65"}
  ]
}
```

### Filtering Items

Create `input_data/train_model_for_items.csv`:

```csv
foodcourt_id,restaurant_id,item_name
5f338dce8f277f4c2f4ac99f,60f172e2d1b6e328e744cf65,Biryani
5f338dce8f277f4c2f4ac99f,60f172e2d1b6e328e744cf65,Butter Chicken
```

---

## Requirements

See `requirements.txt` for full list. Key dependencies:
- pandas
- numpy
- xgboost
- pymongo
- mysql-connector-python
- openpyxl

---

## Configuration

Configure database connections in `config/config.json`:

```json
{
  "mongodb": {
    "connection_string": "mongodb://...",
    "db_name": "...",
    "entity_record": "..."
  },
  "mysql": {
    "host": "...",
    "user": "...",
    "password": "...",
    "db_name": "..."
  }
}
```

---

## Output Files

### File Locator (`file_locator.xlsx`)
- Single sheet with hyperlinks to all output files
- Columns: Foodcourt ID/Name, Restaurant ID/Name, Item ID/Name, File paths with clickable links

### Pipeline Logs (`logs/pipeline_logs.xlsx`)
- Multiple sheets for different error types:
  - `enrichment_errors`
  - `preprocessing_errors`
  - `model_training_errors`
  - `postprocessing_errors`
  - `data_fetch_errors`
  - `general_errors`

---

## Notes

- All intermediate and final files are saved as Excel format (`.xlsx`)
- Files are organized by foodcourt ID in subdirectories
- Each item gets its own file at enrichment and preprocessing stages
- Models are saved as `.pkl` files organized by model type
- Results include both training and validation data in separate sheets

---

## Support

For issues or questions, refer to:
- Pipeline logs: `output_data/logs/pipeline_logs.xlsx`
- File locator: `output_data/file_locator.xlsx`
- Individual step logs in console output

