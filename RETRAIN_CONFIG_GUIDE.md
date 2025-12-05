# Retrain Configuration Guide

The `retrain.json` file controls which foodcourts, restaurants, and items should be reprocessed in the pipeline.

## Location
`input_data/retrain.json`

## Supported Formats

### 1. Using IDs (Traditional Method)
```json
{
  "model_generation": {
    "foodcourt_ids": ["5f338dce8f277f4c2f4ac99f", "5f36401ce091b809e2467cf4"],
    "restaurant_ids": ["62b96ba6a3b4411d5b0b508a", "63085a2195355f366f7fa0f6"]
  }
}
```

### 2. Using Foodcourt Names (NEW - Easier Method) ✨
You can now use foodcourt **names** instead of IDs! The system will automatically resolve them using `name_mapping.json`.

**Note**: Restaurant names are NOT supported because restaurants can have the same name across different foodcourts. You must specify restaurants using both `foodcourt_id` and `restaurant_id`.

```json
{
  "model_generation": {
    "foodcourt_names": ["CG-EPIP Kakadu-BLR", "CG-Campus-BLR"]
  }
}
```

### 3. Specifying Restaurants (Foodcourt + Restaurant IDs Required)
Since restaurant names can be duplicated across foodcourts, you must specify both `foodcourt_id` and `restaurant_id` when targeting specific restaurants:

```json
{
  "model_generation": {
    "restaurant_ids": [
      {"foodcourt_id": "62720dcd1410b643ccc5eaf0", "restaurant_id": "62b96ba6a3b4411d5b0b508a"},
      {"foodcourt_id": "62720dcd1410b643ccc5eaf0", "restaurant_id": "63085a2195355f366f7fa0f6"}
    ]
  }
}
```

### 4. Mixed Format (Foodcourt Names + Restaurant Pairs)
You can mix foodcourt names with restaurant pairs:

```json
{
  "model_generation": {
    "foodcourt_names": ["CG-Campus-BLR"],
    "restaurant_ids": [
      {"foodcourt_id": "62720dcd1410b643ccc5eaf0", "restaurant_id": "62b96ba6a3b4411d5b0b508a"}
    ]
  }
}
```

### 5. Backward Compatibility (Simple Restaurant IDs)
For backward compatibility, you can still use simple restaurant ID strings, but only when the foodcourt is already specified in `foodcourt_ids`:

```json
{
  "model_generation": {
    "foodcourt_ids": ["62720dcd1410b643ccc5eaf0"],
    "restaurant_ids": ["62b96ba6a3b4411d5b0b508a", "63085a2195355f366f7fa0f6"]
  }
}
```

## Pipeline Steps

The configuration supports all pipeline steps:
- `data_fetch`
- `enrich_data`
- `preprocessing`
- `model_generation`
- `postprocessing`
- `compiled_result_generation`

## How Name Resolution Works

1. The system loads `name_mapping.json` (from `dashboard/name_mapping.json` or root `name_mapping.json`)
2. **Foodcourt names** are matched **case-insensitively** against the mapping
3. If an exact match is found, the corresponding ID is used
4. If no exact match, a partial match (contains) is attempted
5. Unresolved names will log a warning but won't stop the pipeline
6. **Restaurant names are NOT resolved** - use foodcourt_id + restaurant_id pairs instead

## Finding Names and IDs

### Using the Dashboard
1. Open the dashboard
2. Use the name mapping feature to browse foodcourts and restaurants
3. For foodcourts: Copy the exact name you see in the dashboard
4. For restaurants: Copy both the foodcourt ID and restaurant ID

### From name_mapping.json
You can also check `dashboard/name_mapping.json` directly:
- `foodcourt_id_to_name`: Maps foodcourt IDs to names
- `restaurant_id_to_name`: Maps restaurant IDs to names
- `restaurant_id_to_foodcourt_id`: Maps restaurant IDs to their foodcourt IDs

## Examples

### Example 1: Retrain models for specific foodcourts (using names)
```json
{
  "model_generation": {
    "foodcourt_names": [
      "CG-EPIP Kakadu-BLR",
      "CG-Campus-BLR",
      "CG-DV B4-BLR"
    ]
  }
}
```

### Example 2: Retrain models for specific restaurants (using foodcourt + restaurant pairs)
```json
{
  "model_generation": {
    "restaurant_ids": [
      {"foodcourt_id": "62720dcd1410b643ccc5eaf0", "restaurant_id": "62b96ba6a3b4411d5b0b508a"},
      {"foodcourt_id": "62720dcd1410b643ccc5eaf0", "restaurant_id": "63085a2195355f366f7fa0f6"},
      {"foodcourt_id": "5f3652cde091b809e2467d96", "restaurant_id": "5f3653f6696cd8152e8e933f"}
    ]
  }
}
```

### Example 3: Retrain multiple steps
```json
{
  "preprocessing": {
    "foodcourt_names": ["CG-EPIP Kakadu-BLR"]
  },
  "model_generation": {
    "foodcourt_names": ["CG-EPIP Kakadu-BLR"],
    "restaurant_ids": [
      {"foodcourt_id": "62720dcd1410b643ccc5eaf0", "restaurant_id": "62b96ba6a3b4411d5b0b508a"}
    ]
  },
  "postprocessing": {
    "foodcourt_names": ["CG-EPIP Kakadu-BLR"]
  }
}
```

### Example 4: Empty config (process all missing items)
```json
{
  "data_fetch": {},
  "enrich_data": {},
  "preprocessing": {},
  "model_generation": {},
  "postprocessing": {},
  "compiled_result_generation": {}
}
```
When all sections are empty, the pipeline processes all items that need processing (checks if output files exist).

## Behavior

- **If a section has entries**: Only those foodcourts/restaurants are processed (forces retraining even if files exist)
- **If a section is empty but others have entries**: That step is skipped
- **If all sections are empty**: All items that need processing are handled (normal pipeline run)
- **Restaurant matching**: 
  - If `restaurant_ids` contains objects with `{foodcourt_id, restaurant_id}`, both must match
  - If `restaurant_ids` contains simple strings, the restaurant ID must match AND the foodcourt must be in `foodcourt_ids`
  - If `restaurant_ids` is empty but `foodcourt_ids` has entries, all restaurants in those foodcourts are processed

## Important Notes

- **Restaurant names are NOT supported** because the same restaurant name can exist in multiple foodcourts
- Always specify both `foodcourt_id` and `restaurant_id` when targeting specific restaurants
- Foodcourt names are resolved at pipeline startup with clear logging
- The resolved IDs are logged for transparency
- Both exact and partial name matching are supported for foodcourts (partial matches log a notice)
