#!/usr/bin/env python3
"""
Script to clean and combine all market data into a single Parquet file

Usage:
    # Process from S3 and save to S3
    POLYMARKET_S3_BUCKET=cs230-polymarket-data-1 python scripts/03_clean_all.py --s3
    
    # Process from local files (legacy)
    python scripts/03_clean_all.py --markets data/raw/markets.json
    
    # Other options
    python scripts/03_clean_all.py --s3 --resample-freq 30min
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
        default=None,
        help="Input markets JSON file from step 1 (local path)",
    )
    parser.add_argument(
        "--price-history-dir",
        type=Path,
        default=None,
        help="Input directory with price history JSON files from step 2 (local path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet file path (local path)",
    )
    parser.add_argument(
        "--resample-freq",
        type=str,
        default=None,
        help=f"Resampling frequency (default: {settings.resample_frequency})",
    )
    parser.add_argument(
        "--s3",
        action="store_true",
        help="Read from S3 and write output to S3. "
        "Requires POLYMARKET_S3_BUCKET env var.",
    )

    args = parser.parse_args()
    use_s3 = args.s3

    # Validate configuration
    if use_s3:
        if not settings.use_s3:
            logger.error(
                "S3 mode requested but POLYMARKET_S3_BUCKET env var not set.\n"
                "Set it with: export POLYMARKET_S3_BUCKET=your-bucket-name"
            )
            sys.exit(1)
    else:
        # Local mode: check files exist
        markets_file = args.markets or Path(settings.raw_data_dir) / "markets.json"
        price_history_dir = args.price_history_dir or Path(settings.raw_data_dir) / "price_history"
        
        if not markets_file.exists():
            logger.error(f"Markets file not found: {markets_file}")
            logger.error(
                "Please run scripts/01_fetch_markets.py first to generate markets.json"
            )
            sys.exit(1)

        if not price_history_dir.exists():
            logger.error(f"Price history directory not found: {price_history_dir}")
            logger.error(
                "Please run scripts/02_fetch_history.py first to generate price histories"
            )
            sys.exit(1)

    resample_freq = args.resample_freq or settings.resample_frequency

    logger.info("=" * 60)
    logger.info("Polymarket Data Cleaner & Combiner")
    logger.info("=" * 60)
    
    if use_s3:
        logger.info(f"Mode: S3 (bucket: {settings.s3_bucket})")
        logger.info(f"Markets source: s3://{settings.s3_bucket}/{settings.get_s3_markets_key()}")
        logger.info(f"Price history source: s3://{settings.s3_bucket}/{settings.s3_prefix}/raw/price_history/")
        logger.info(f"Output: s3://{settings.s3_bucket}/{settings.get_s3_processed_key()}")
    else:
        markets_file = args.markets or Path(settings.raw_data_dir) / "markets.json"
        price_history_dir = args.price_history_dir or Path(settings.raw_data_dir) / "price_history"
        output_file = args.output or Path(settings.processed_data_dir) / "polymarket_data.parquet"
        logger.info(f"Markets file: {markets_file}")
        logger.info(f"Price history dir: {price_history_dir}")
        logger.info(f"Output file: {output_file}")
    
    logger.info(f"Resample frequency: {resample_freq}")
    logger.info("=" * 60)

    try:
        if use_s3:
            df = clean_and_combine_data(
                resample_freq=resample_freq,
                use_s3=True,
                s3_output_key=settings.get_s3_processed_key(),
            )
            output_location = f"s3://{settings.s3_bucket}/{settings.get_s3_processed_key()}"
        else:
            markets_file = args.markets or Path(settings.raw_data_dir) / "markets.json"
            price_history_dir = args.price_history_dir or Path(settings.raw_data_dir) / "price_history"
            output_file = args.output or Path(settings.processed_data_dir) / "polymarket_data.parquet"
            
            df = clean_and_combine_data(
                markets_file=markets_file,
                price_history_dir=price_history_dir,
                output_file=output_file,
                resample_freq=resample_freq,
            )
            output_location = str(output_file)

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

        logger.info(f"\nOutput saved to: {output_location}")

    except ValueError as e:
        logger.error(f"No data to process: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error processing data: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
