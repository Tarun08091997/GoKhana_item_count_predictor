"""
Convert Parquet files to CSV format.

This script can:
- Convert a single parquet file to CSV
- Convert all parquet files in a directory (recursively)
- Preserve directory structure in output
- Handle errors gracefully

Usage:
    python convert_parquet_to_csv.py <input_path> [output_path]
    
Examples:
    # Convert single file
    python convert_parquet_to_csv.py data.parquet
    
    # Convert single file with custom output
    python convert_parquet_to_csv.py data.parquet output.csv
    
    # Convert all parquet files in directory (recursive)
    python convert_parquet_to_csv.py input_data/fetched_data
    
    # Convert directory with custom output directory
    python convert_parquet_to_csv.py input_data/fetched_data output_data
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def convert_single_parquet(input_path: Path, output_path: Optional[Path] = None) -> bool:
    """
    Convert a single parquet file to CSV.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output CSV file. If None, uses input path with .csv extension
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not input_path.exists():
            LOGGER.error("Input file not found: %s", input_path)
            return False
        
        if not input_path.suffix.lower() == ".parquet":
            LOGGER.warning("File does not have .parquet extension: %s", input_path)
        
        LOGGER.info("Reading parquet file: %s", input_path)
        df = pd.read_parquet(input_path)
        
        if df.empty:
            LOGGER.warning("Parquet file is empty: %s", input_path)
        
        # Determine output path
        if output_path is None:
            output_path = input_path.with_suffix(".csv")
        else:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        LOGGER.info("Writing CSV file: %s (%d rows, %d columns)", output_path, len(df), len(df.columns))
        df.to_csv(output_path, index=False, encoding="utf-8")
        
        LOGGER.info("Successfully converted: %s -> %s", input_path.name, output_path.name)
        return True
        
    except Exception as exc:
        LOGGER.error("Failed to convert %s: %s", input_path, exc)
        return False


def convert_directory(input_dir: Path, output_dir: Optional[Path] = None, preserve_structure: bool = True) -> tuple[int, int]:
    """
    Convert all parquet files in a directory (recursively).
    
    Args:
        input_dir: Directory containing parquet files
        output_dir: Output directory. If None, converts in place (replaces .parquet with .csv)
        preserve_structure: If True, maintains directory structure in output
        
    Returns:
        Tuple of (successful_conversions, failed_conversions)
    """
    if not input_dir.exists():
        LOGGER.error("Input directory not found: %s", input_dir)
        return (0, 0)
    
    if not input_dir.is_dir():
        LOGGER.error("Input path is not a directory: %s", input_dir)
        return (0, 0)
    
    # Find all parquet files
    parquet_files = list(input_dir.rglob("*.parquet"))
    
    if not parquet_files:
        LOGGER.warning("No parquet files found in: %s", input_dir)
        return (0, 0)
    
    LOGGER.info("Found %d parquet file(s) in %s", len(parquet_files), input_dir)
    
    successful = 0
    failed = 0
    
    for parquet_file in parquet_files:
        if preserve_structure and output_dir is not None:
            # Maintain relative path structure
            relative_path = parquet_file.relative_to(input_dir)
            csv_output = output_dir / relative_path.with_suffix(".csv")
        elif output_dir is not None:
            # Flatten structure - all CSVs in output_dir
            csv_output = output_dir / parquet_file.with_suffix(".csv").name
        else:
            # Convert in place
            csv_output = None
        
        if convert_single_parquet(parquet_file, csv_output):
            successful += 1
        else:
            failed += 1
    
    return (successful, failed)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert Parquet files to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Input parquet file or directory containing parquet files",
    )
    parser.add_argument(
        "output_path",
        type=str,
        nargs="?",
        default=None,
        help="Output CSV file or directory (optional). If not provided, converts in place",
    )
    parser.add_argument(
        "--no-preserve-structure",
        action="store_true",
        help="Flatten directory structure when converting directories (default: preserve structure)",
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path).resolve() if args.output_path else None
    
    LOGGER.info("=" * 60)
    LOGGER.info("Parquet to CSV Converter")
    LOGGER.info("=" * 60)
    
    # Check if input is a file or directory
    if input_path.is_file():
        # Single file conversion
        success = convert_single_parquet(input_path, output_path)
        if success:
            LOGGER.info("Conversion completed successfully!")
            sys.exit(0)
        else:
            LOGGER.error("Conversion failed!")
            sys.exit(1)
    
    elif input_path.is_dir():
        # Directory conversion
        preserve_structure = not args.no_preserve_structure
        
        if output_path and not output_path.exists():
            LOGGER.info("Creating output directory: %s", output_path)
            output_path.mkdir(parents=True, exist_ok=True)
        
        successful, failed = convert_directory(input_path, output_path, preserve_structure)
        
        LOGGER.info("=" * 60)
        LOGGER.info("Conversion Summary")
        LOGGER.info("=" * 60)
        LOGGER.info("Successful: %d", successful)
        LOGGER.info("Failed: %d", failed)
        LOGGER.info("Total: %d", successful + failed)
        
        if failed == 0:
            LOGGER.info("All conversions completed successfully!")
            sys.exit(0)
        else:
            LOGGER.warning("Some conversions failed. Check logs above for details.")
            sys.exit(1)
    
    else:
        LOGGER.error("Input path does not exist: %s", input_path)
        sys.exit(1)


if __name__ == "__main__":
    main()

