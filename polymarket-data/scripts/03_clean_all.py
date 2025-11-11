#!/usr/bin/env python3
"""
Script to clean and combine all market data into a single Parquet file

Usage:
    python scripts/03_clean_all.py
    python scripts/03_clean_all.py --markets data/raw/markets.json
    python scripts/03_clean_all.py --resample-freq 30min
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from polymarket_data.clean import clean_and_combine_data
from polymarket_data.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Main entry point for data cleaning script
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean and combine Polymarket data into Parquet.\n\n"
            "Prerequisites:\n"
            "  1. Run scripts/01_fetch_markets.py first\n"
            "  2. Run scripts/02_fetch_history.py second\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--markets",
        type=Path,
        default=Path(settings.raw_data_dir) / "markets.json",
        help="Input markets JSON file from step 1",
    )
    parser.add_argument(
        "--price-history-dir",
        type=Path,
        default=Path(settings.raw_data_dir) / "price_history",
        help="Input directory with price history JSON files from step 2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.processed_data_dir) / "polymarket_data.parquet",
        help="Output Parquet file path",
    )
    parser.add_argument(
        "--resample-freq",
        type=str,
        default=None,
        help=f"Resampling frequency (default: {settings.resample_frequency})",
    )

    args = parser.parse_args()

    if not args.markets.exists():
        logger.error(f"Markets file not found: {args.markets}")
        logger.error(
            "Please run scripts/01_fetch_markets.py first to generate markets.json"
        )
        sys.exit(1)

    if not args.price_history_dir.exists():
        logger.error(f"Price history directory not found: {args.price_history_dir}")
        logger.error(
            "Please run scripts/02_fetch_history.py first to generate price histories"
        )
        sys.exit(1)

    resample_freq = args.resample_freq or settings.resample_frequency

    logger.info("=" * 60)
    logger.info("Polymarket Data Cleaner & Combiner")
    logger.info("=" * 60)
    logger.info(f"Markets file: {args.markets}")
    logger.info(f"Price history dir: {args.price_history_dir}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Resample frequency: {resample_freq}")
    logger.info("=" * 60)

    try:
        df = clean_and_combine_data(
            markets_file=args.markets,
            price_history_dir=args.price_history_dir,
            output_file=args.output,
            resample_freq=resample_freq,
        )

        logger.info("=" * 60)
        logger.info("Data Processing Complete!")
        logger.info("=" * 60)

        if df.empty:
            logger.warning("No data was processed (empty DataFrame)")
            sys.exit(0)

        logger.info("\nDataset Summary:")
        logger.info(f"  Total rows: {len(df):,}")
        logger.info(f"  Columns: {len(df.columns)}")

        if "market_id" in df.columns:
            logger.info(f"  Unique markets: {df['market_id'].nunique()}")

        if "token_id" in df.columns:
            logger.info(f"  Unique tokens: {df['token_id'].nunique()}")

        if "won" in df.columns:
            win_counts = df["won"].value_counts().sort_index()
            total = len(df)
            logger.info(f"\nTarget Variable ('won') Distribution:")
            for value, count in win_counts.items():
                pct = (count / total) * 100
                label = "Won" if value == 1 else "Lost"
                logger.info(f"  {label} ({value}): {count:,} ({pct:.1f}%)")

        if "timestamp" in df.columns:
            logger.info(f"\nTime Range:")
            logger.info(f"  Start: {df['timestamp'].min()}")
            logger.info(f"  End: {df['timestamp'].max()}")

        if "category" in df.columns and "market_id" in df.columns:
            logger.info(f"\nTop Categories:")
            category_counts = (
                df.groupby("category")["market_id"]
                .nunique()
                .sort_values(ascending=False)
            )

            for category, count in category_counts.head(10).items():
                logger.info(f"  {category}: {count} markets")

            if len(category_counts) > 10:
                logger.info(f"  ... and {len(category_counts) - 10} more categories")

        logger.info(f"\nColumns ({len(df.columns)}):")
        for col in df.columns:
            logger.info(f"  - {col}")

        logger.info(f"\nOutput saved to: {args.output}")

    except ValueError as e:
        logger.error(f"No data to process: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error processing data: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
