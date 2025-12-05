# Pipeline Checkpoint/Resume System

## Overview

The checkpoint system allows the pipeline to resume from where it left off if interrupted. It tracks which items have been completed at each step, so you don't have to reprocess items that were already successfully processed.

## How It Works

1. **Checkpoint File**: A `pipeline_checkpoint.json` file is created in `input_data/` directory
2. **Progress Tracking**: Each step tracks:
   - `completed_items`: Items that were successfully processed
   - `failed_items`: Items that failed with errors
   - `in_progress_items`: Items currently being processed (cleared on startup)

3. **Resume Logic**: When you restart the pipeline:
   - Items already completed are automatically skipped
   - Only remaining items are processed
   - Failed items are retried (unless you manually remove them from checkpoint)

## Usage

### Basic Usage (Checkpoint Enabled by Default)

```bash
python run_pipeline.py
```

The checkpoint system is **enabled by default**. It will:
- Skip items already completed
- Resume from where it left off
- Show progress summary on startup

### Disable Checkpoint System

```bash
python run_pipeline.py --no-checkpoint
```

This will process all items from scratch, ignoring checkpoint file.

### Reset Checkpoints (Start Fresh)

```bash
python run_pipeline.py --reset-checkpoint
```

This will clear all checkpoints and start processing from the beginning.

### Production Mode with Checkpoints

```bash
python run_pipeline.py --prod-mode
```

Checkpoints work with production mode as well.

## Checkpoint File Structure

The checkpoint file (`input_data/pipeline_checkpoint.json`) has this structure:

```json
{
  "metadata": {
    "created_at": "2025-01-01T12:00:00",
    "last_updated": "2025-01-01T13:00:00",
    "pipeline_type": "FRI_LEVEL"
  },
  "steps": {
    "enrich_data": {
      "completed_items": [...],
      "failed_items": [...],
      "in_progress_items": []
    },
    "preprocessing": {...},
    "model_generation": {...},
    "postprocessing": {...},
    "compiled_result_generation": {...}
  }
}
```

## Example Scenarios

### Scenario 1: Pipeline Crashes After 300 Items

**Before**: You processed 300 items, then the system crashed.

**On Resume**: 
- Pipeline detects 300 items are already completed
- Skips those 300 items
- Continues from item 301
- Saves time by not reprocessing

### Scenario 2: Power Outage During Model Generation

**Before**: You were training models for 500 items, power went out at item 250.

**On Resume**:
- Pipeline clears "in_progress" items (from crash)
- Finds 250 items already completed
- Resumes from item 251
- No duplicate work

### Scenario 3: Need to Reprocess Everything

**Before**: You want to start fresh, ignoring previous checkpoints.

**Solution**:
```bash
python run_pipeline.py --reset-checkpoint
```

## Viewing Checkpoint Status

You can view the checkpoint file directly:

```bash
cat input_data/pipeline_checkpoint.json
```

Or use Python to get a summary:

```python
from src.util.checkpoint_manager import get_checkpoint_manager

checkpoint = get_checkpoint_manager()
summary = checkpoint.get_progress_summary()
print(summary)
```

## Manual Checkpoint Management

### Reset a Specific Step

```python
from src.util.checkpoint_manager import get_checkpoint_manager

checkpoint = get_checkpoint_manager()
checkpoint.reset_step("model_generation")  # Reset only model generation
```

### Clear All Checkpoints

```python
from src.util.checkpoint_manager import get_checkpoint_manager

checkpoint = get_checkpoint_manager()
checkpoint.reset_all()  # Start completely fresh
```

### Remove Failed Items from Checkpoint

Edit `input_data/pipeline_checkpoint.json` manually and remove items from the `failed_items` array for the step you want to retry.

## Best Practices

1. **Keep Checkpoint File**: Don't delete `pipeline_checkpoint.json` unless you want to start fresh
2. **Backup Before Reset**: The checkpoint file is automatically backed up (`.bak` file) before updates
3. **Monitor Progress**: Check the checkpoint file periodically to see progress
4. **Handle Failures**: Review `failed_items` to understand what went wrong

## Troubleshooting

### Checkpoint File Corrupted

If the checkpoint file is corrupted:
1. Delete `input_data/pipeline_checkpoint.json`
2. Restart pipeline - it will create a new checkpoint file
3. Note: You'll lose progress tracking, but pipeline will still work

### Want to Retry Failed Items

1. Open `input_data/pipeline_checkpoint.json`
2. Find the step with failed items
3. Move items from `failed_items` to `completed_items` (or delete them)
4. Restart pipeline - it will retry those items

### Checkpoint Not Working

1. Check if `--no-checkpoint` flag was used
2. Verify checkpoint file exists: `input_data/pipeline_checkpoint.json`
3. Check file permissions
4. Review logs for checkpoint-related messages

## Integration with Steps

The checkpoint system is currently integrated with:
- ✅ **Model Generation** (step4) - Most time-consuming step
- ⏳ Other steps can be integrated similarly

To integrate other steps, pass `checkpoint_manager` parameter and use:
- `checkpoint_manager.is_item_completed(step_name, item_dict)` - Check if done
- `checkpoint_manager.mark_item_in_progress(step_name, item_dict)` - Mark as starting
- `checkpoint_manager.mark_item_completed(step_name, item_dict)` - Mark as done
- `checkpoint_manager.mark_item_failed(step_name, item_dict, error)` - Mark as failed

