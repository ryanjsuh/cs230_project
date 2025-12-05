#!/usr/bin/env python3
"""
Quickly grab partial cleaned data from whatever price histories exist.

This script is optimized for speed: it first loads the manifest of processed
tokens from S3, then only processes markets that have at least one token
with data. Much faster than iterating through ALL markets.

Usage:
    # Grab partial data from S3 (recommended)
    POLYMARKET_S3_BUCKET=cs230-polymarket-data-1 python scripts/grab_partial.py --s3

    # Save to local file instead of S3
    POLYMARKET_S3_BUCKET=cs230-polymarket-data-1 python scripts/grab_partial.py --s3 --output partial.parquet

    # Limit to first N markets with data (for testing)
    python scripts/grab_partial.py --s3 --max-markets 1000
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from polymarket_data.clean import MarketDataProcessor
from polymarket_data.config import settings
from polymarket_data.fetch_markets import Market

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quickly grab partial cleaned data from existing price histories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--s3",
        action="store_true",
        help="Read from S3 (requires POLYMARKET_S3_BUCKET env var)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet file (default: S3 if --s3, else data/processed/partial.parquet)",
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=None,
        help="Limit to first N markets with data",
    )
    parser.add_argument(
        "--resample-freq",
        type=str,
        default=None,
        help=f"Resampling frequency (default: {settings.resample_frequency})",
    )

    args = parser.parse_args()
    use_s3 = args.s3

    if use_s3 and not settings.use_s3:
        logger.error(
            "S3 mode requested but POLYMARKET_S3_BUCKET env var not set.\n"
            "Set it with: export POLYMARKET_S3_BUCKET=your-bucket-name"
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Partial Data Grabber (Fast Mode)")
    logger.info("=" * 60)

    processor = MarketDataProcessor(
        resample_freq=args.resample_freq,
        use_s3=use_s3,
    )

    # Step 1: Get the set of tokens we actually have data for
    logger.info("Loading list of available price histories...")
    if use_s3:
        from polymarket_data.s3_utils import load_processed_tokens_from_s3, get_existing_token_ids_from_s3
        
        # Try manifest first (faster), fall back to listing S3 objects
        available_tokens = load_processed_tokens_from_s3()
        if not available_tokens:
            logger.info("No manifest found, listing S3 objects...")
            available_tokens = get_existing_token_ids_from_s3()
    else:
        price_history_dir = Path(settings.raw_data_dir) / "price_history"
        if price_history_dir.exists():
            available_tokens = {
                f.stem for f in price_history_dir.glob("*.json")
                if not f.name.endswith(".tmp")
            }
        else:
            available_tokens = set()

    logger.info(f"Found {len(available_tokens)} tokens with price history data")

    if not available_tokens:
        logger.error("No price history data found. Run 02_fetch_history.py first.")
        sys.exit(1)

    # Step 2: Load markets and filter to those with available data
    logger.info("Loading markets...")
    markets = processor.load_markets()
    
    markets_with_data: list[Market] = []
    for market in markets:
        token_ids = market.get_token_ids()
        if any(tid in available_tokens for tid in token_ids):
            markets_with_data.append(market)

    logger.info(
        f"Found {len(markets_with_data)}/{len(markets)} markets "
        f"with at least one token that has price history"
    )

    if args.max_markets:
        markets_with_data = markets_with_data[:args.max_markets]
        logger.info(f"Limited to {len(markets_with_data)} markets")

    if not markets_with_data:
        logger.error("No markets have price history data available.")
        sys.exit(1)

    # Step 3: Process only markets with data
    logger.info(f"Processing {len(markets_with_data)} markets...")
    
    import pandas as pd
    all_dfs = []
    processed_count = 0

    for i, market in enumerate(markets_with_data, 1):
        if i % 100 == 0:
            logger.info(f"Processing market {i}/{len(markets_with_data)}...")

        df = processor.process_market(market)
        if df is not None:
            all_dfs.append(df)
            processed_count += 1

    logger.info(f"Successfully processed {processed_count} markets")

    if not all_dfs:
        logger.error("No data was successfully processed.")
        sys.exit(1)

    # Step 4: Finalize and save
    df = processor._finalize_dataframe(all_dfs)

    if args.output:
        output_path = args.output
        processor.save_parquet(df, output_path=output_path)
        output_location = str(output_path)
    elif use_s3:
        s3_key = settings.get_s3_processed_key("partial_polymarket_data.parquet")
        processor.save_parquet(df, s3_key=s3_key)
        output_location = f"s3://{settings.s3_bucket}/{s3_key}"
    else:
        output_path = Path(settings.processed_data_dir) / "partial_polymarket_data.parquet"
        processor.save_parquet(df, output_path=output_path)
        output_location = str(output_path)

    logger.info("=" * 60)
    logger.info("Partial Data Extraction Complete!")
    logger.info("=" * 60)
    logger.info(f"  Total rows: {len(df):,}")
    logger.info(f"  Unique markets: {df['market_id'].nunique()}")
    logger.info(f"  Unique tokens: {df['token_id'].nunique()}")
    if "won" in df.columns:
        win_counts = df["won"].value_counts().sort_index()
        for value, count in win_counts.items():
            label = "Won" if value == 1 else "Lost"
            logger.info(f"  {label}: {count:,}")
    logger.info(f"  Output: {output_location}")


if __name__ == "__main__":
    main()

