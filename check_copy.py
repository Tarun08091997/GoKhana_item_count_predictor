"""
Check for duplicate records in enriched data files.

This script scans all CSV files in input_data/enriched_data/ and checks for
duplicate records based on the combination of 'date' and 'menuitemid'.
It reports any duplicates found and optionally fixes them by keeping the last occurrence.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def check_duplicates_in_file(file_path: Path) -> Tuple[bool, int, pd.DataFrame]:
    """
    Check for duplicates in a single CSV file based on date + menuitemid.
    
    Returns:
        (has_duplicates, duplicate_count, duplicate_rows_df)
    """
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except Exception as exc:
        logging.error("Failed to read %s: %s", file_path, exc)
        return False, 0, pd.DataFrame()
    
    if df.empty:
        return False, 0, pd.DataFrame()
    
    # Check if required columns exist
    if "date" not in df.columns or "menuitemid" not in df.columns:
        logging.warning(
            "File %s missing required columns (date or menuitemid). Skipping.",
            file_path
        )
        return False, 0, pd.DataFrame()
    
    # Convert date to datetime for consistent comparison
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Find duplicates based on date + menuitemid
    duplicate_mask = df.duplicated(subset=["date", "menuitemid"], keep=False)
    duplicate_count = duplicate_mask.sum()
    has_duplicates = duplicate_count > 0
    
    if has_duplicates:
        duplicate_rows = df[duplicate_mask].copy()
        return True, duplicate_count, duplicate_rows
    
    return False, 0, pd.DataFrame()


def fix_duplicates_in_file(file_path: Path) -> Tuple[bool, int]:
    """
    Fix duplicates in a file by keeping the last occurrence of each date+menuitemid combination.
    
    Returns:
        (success, duplicates_removed_count)
    """
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except Exception as exc:
        logging.error("Failed to read %s: %s", file_path, exc)
        return False, 0
    
    if df.empty:
        return True, 0
    
    if "date" not in df.columns or "menuitemid" not in df.columns:
        logging.warning(
            "File %s missing required columns (date or menuitemid). Skipping.",
            file_path
        )
        return False, 0
    
    original_count = len(df)
    
    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Remove duplicates, keeping the last occurrence
    df_cleaned = df.sort_values(["menuitemid", "date"]).drop_duplicates(
        subset=["date", "menuitemid"], keep="last"
    )
    
    duplicates_removed = original_count - len(df_cleaned)
    
    if duplicates_removed > 0:
        # Save the cleaned dataframe
        df_cleaned.to_csv(file_path, index=False, encoding="utf-8")
        logging.info(
            "Fixed %s: Removed %d duplicate(s), %d rows remaining",
            file_path,
            duplicates_removed,
            len(df_cleaned),
        )
        return True, duplicates_removed
    
    return True, 0


def scan_enriched_data(base_path: Path, fix_duplicates: bool = False) -> Dict[str, any]:
    """
    Scan all enriched data files for duplicates.
    
    Args:
        base_path: Path to input_data/enriched_data directory
        fix_duplicates: If True, automatically fix duplicates by keeping last occurrence
    
    Returns:
        Dictionary with scan results
    """
    if not base_path.exists():
        logging.error("Enriched data directory not found: %s", base_path)
        return {}
    
    results = {
        "total_files": 0,
        "files_with_duplicates": 0,
        "total_duplicate_rows": 0,
        "files_fixed": 0,
        "duplicates_removed": 0,
        "file_details": [],
    }
    
    # Find all CSV files (excluding _debug folder)
    csv_files = []
    for root, dirs, files in os.walk(base_path):
        # Skip _debug directories
        if "_debug" in root:
            continue
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(Path(root) / file)
    
    results["total_files"] = len(csv_files)
    logging.info("Found %d CSV files to check", len(csv_files))
    
    for file_path in csv_files:
        has_duplicates, duplicate_count, duplicate_rows = check_duplicates_in_file(file_path)
        
        file_detail = {
            "file": str(file_path.relative_to(base_path)),
            "has_duplicates": has_duplicates,
            "duplicate_count": duplicate_count,
        }
        
        if has_duplicates:
            results["files_with_duplicates"] += 1
            results["total_duplicate_rows"] += duplicate_count
            logging.warning(
                "Found %d duplicate row(s) in %s",
                duplicate_count,
                file_path.relative_to(base_path),
            )
            
            # Log sample duplicate rows
            if not duplicate_rows.empty:
                sample = duplicate_rows.head(5)
                logging.debug("Sample duplicate rows:\n%s", sample[["date", "menuitemid", "count"]].to_string())
            
            # Fix if requested
            if fix_duplicates:
                success, removed = fix_duplicates_in_file(file_path)
                if success:
                    results["files_fixed"] += 1
                    results["duplicates_removed"] += removed
                    file_detail["fixed"] = True
                    file_detail["duplicates_removed"] = removed
                else:
                    file_detail["fixed"] = False
        
        results["file_details"].append(file_detail)
    
    return results


def print_summary(results: Dict[str, any]):
    """Print a summary of the duplicate check results."""
    print("\n" + "=" * 80)
    print("DUPLICATE CHECK SUMMARY")
    print("=" * 80)
    print(f"Total files scanned: {results['total_files']}")
    print(f"Files with duplicates: {results['files_with_duplicates']}")
    print(f"Total duplicate rows found: {results['total_duplicate_rows']}")
    
    if results.get("files_fixed", 0) > 0:
        print(f"\nFiles fixed: {results['files_fixed']}")
        print(f"Total duplicates removed: {results['duplicates_removed']}")
    
    if results["files_with_duplicates"] > 0:
        print("\nFiles with duplicates:")
        for detail in results["file_details"]:
            if detail["has_duplicates"]:
                status = "FIXED" if detail.get("fixed", False) else "NOT FIXED"
                print(f"  - {detail['file']}: {detail['duplicate_count']} duplicate(s) [{status}]")
    else:
        print("\n✓ No duplicates found in any files!")
    
    print("=" * 80 + "\n")


def main():
    """Main function to run the duplicate check."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check for duplicate records in enriched data files"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix duplicates by keeping the last occurrence",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to enriched_data directory (default: input_data/enriched_data)",
    )
    
    args = parser.parse_args()
    
    # Determine base path
    if args.path:
        base_path = Path(args.path)
    else:
        current_dir = Path(__file__).parent
        base_path = current_dir / "input_data" / "enriched_data"
    
    logging.info("Scanning enriched data in: %s", base_path)
    
    # Scan for duplicates
    results = scan_enriched_data(base_path, fix_duplicates=args.fix)
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()


