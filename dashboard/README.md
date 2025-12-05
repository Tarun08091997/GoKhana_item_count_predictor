# GoKhana Pipeline Dashboard

Streamlit dashboard for visualizing and analyzing GoKhana pipeline results.

## Features

- **FRI Result**: Foodcourt-Restaurant-Item level results
  - FRI ID and names display
  - Training and validation days (total and active)
  - Enrich data and preprocessing file downloads
  - Model results with detailed metrics
  - Table view and detailed view modes

## Installation

Make sure you have Streamlit installed:

```bash
pip install streamlit
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Running the Dashboard

From the project root directory:

```bash
streamlit run dashboard/app.py
```

Or navigate to the dashboard folder:

```bash
cd dashboard
streamlit run app.py
```

## Structure

```
dashboard/
├── app.py                 # Main Streamlit application
├── pages/                 # Dashboard pages/tabs
│   ├── fri_result.py      # FRI Result page
│   └── result_compilation.py  # (Legacy - can be removed)
├── utils/                 # Utility functions
│   ├── data_loader.py     # Data loading utilities
│   └── name_resolver.py   # MongoDB name resolution
└── README.md
```

## Usage

1. Select a foodcourt from the sidebar
2. Select a restaurant from the sidebar
3. View results:
   - **Table View**: Quick overview with all items in a table
   - **Detailed View**: Expandable cards with full details
4. For each item:
   - View FRI ID and item name
   - See training/validation total and active days
   - Download enrich_data and preprocessing CSV files
   - View model results with metrics:
     - abs_avg_deviation
     - abs_avg_error_pct
     - abs_avg_deviation_capped
     - abs_avg_error_pct_capped
   - See both training and validation metrics for each model

