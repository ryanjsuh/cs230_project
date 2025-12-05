#!/usr/bin/env python3
"""
Script to quickly grab partial cleaned data from whatever price histories exist.

This script is optimized: it first loads the manifest of processed
tokens from S3, then only processes markets that have at least one token
with data. Saves checkpoints every N markets so you can stop early and still
have usable data. Much faster than iterating through ALL markets.

Usage:
    # Grab partial data from S3 with checkpoints every 2000 markets (default)
    POLYMARKET_S3_BUCKET=cs230-polymarket-data-1 python scripts/grab_partial.py --s3

    # Custom checkpoint interval
    python scripts/grab_partial.py --s3 --checkpoint-interval 1000

    # Save to local file instead of S3
    python scripts/grab_partial.py --s3 --output-dir data/processed/checkpoints

    # Limit to first N markets with data (for testing)
    python scripts/grab_partial.py --s3 --max-markets 1000
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from polymarket_data.clean import MarketDataProcessor
from polymarket_data.config import settings
from polymarket_data.fetch_markets import Market

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Save a checkpoint and return the output location
def save_checkpoint(
    processor: MarketDataProcessor,
    all_dfs: list[pd.DataFrame],
    checkpoint_num: int,
    use_s3: bool,
    output_dir: Path | None,
) -> str | None:
    if not all_dfs:
        return None
    
    try:
        checkpoint_df = processor._finalize_dataframe(all_dfs)
        filename = f"checkpoint_{checkpoint_num:04d}.parquet"
        
        if use_s3:
            s3_key = settings.get_s3_processed_key(f"checkpoints/{filename}")
            processor.save_parquet(checkpoint_df, s3_key=s3_key)
            return f"s3://{settings.s3_bucket}/{s3_key}"
        else:
            output_path = output_dir / filename
            processor.save_parquet(checkpoint_df, output_path=output_path)
            return str(output_path)
    except Exception as e:
        logger.error(f"Failed to save checkpoint {checkpoint_num}: {e}")
        return None


# Combine all checkpoint files into a single final parquet
def combine_checkpoints(
    processor: MarketDataProcessor,
    checkpoint_files: list[str],
    use_s3: bool,
    output_dir: Path | None,
) -> str:
    logger.info(f"Combining {len(checkpoint_files)} checkpoint files...")
    
    all_dfs = []
    for checkpoint_path in checkpoint_files:
        try:
            if use_s3:
                # Extract S3 key from path
                from polymarket_data.s3_utils import download_bytes_from_s3
                import io
                
                # Parse s3://bucket/key format
                s3_key = checkpoint_path.replace(f"s3://{settings.s3_bucket}/", "")
                parquet_bytes = download_bytes_from_s3(s3_key)
                df = pd.read_parquet(io.BytesIO(parquet_bytes))
            else:
                df = pd.read_parquet(checkpoint_path)
            all_dfs.append(df)
            logger.info(f"  Loaded {checkpoint_path}: {len(df):,} rows")
        except Exception as e:
            logger.warning(f"  Failed to load {checkpoint_path}: {e}")
    
    if not all_dfs:
        raise ValueError("No checkpoint data could be loaded")
    
    # Combine and deduplicate (in case of overlaps)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp", "token_id"], keep="first")
    combined = combined.sort_values("timestamp")
    
    # Save final combined file
    if use_s3:
        s3_key = settings.get_s3_processed_key("polymarket_data_partial.parquet")
        processor.save_parquet(combined, s3_key=s3_key)
        return f"s3://{settings.s3_bucket}/{s3_key}"
    else:
        output_path = output_dir / "polymarket_data_partial.parquet"
        processor.save_parquet(combined, output_path=output_path)
        return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quickly grab partial cleaned data with checkpointing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--s3",
        action="store_true",
        help="Read from S3 (requires POLYMARKET_S3_BUCKET env var)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for checkpoint files (default: S3 or data/processed/checkpoints)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=2000,
        help="Save checkpoint every N markets (default: 2000)",
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
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help="Skip combining checkpoints at the end (just save individual checkpoint files)",
    )

    args = parser.parse_args()
    use_s3 = args.s3
    checkpoint_interval = args.checkpoint_interval

    if use_s3 and not settings.use_s3:
        logger.error(
            "S3 mode requested but POLYMARKET_S3_BUCKET env var not set.\n"
            "Set it with: export POLYMARKET_S3_BUCKET=your-bucket-name"
        )
        sys.exit(1)

    # Set up output directory for local mode
    output_dir = args.output_dir
    if not use_s3 and output_dir is None:
        output_dir = Path(settings.processed_data_dir) / "checkpoints"
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Partial Data Grabber (with Checkpointing)")
    logger.info("=" * 60)
    logger.info(f"Checkpoint interval: every {checkpoint_interval} markets")
    if use_s3:
        logger.info(f"Output: S3 bucket {settings.s3_bucket}")
    else:
        logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

    processor = MarketDataProcessor(
        resample_freq=args.resample_freq,
        use_s3=use_s3,
    )

    # Step 1: Get the set of tokens we actually have data for
    logger.info("Loading list of available price histories...")
    if use_s3:
        from polymarket_data.s3_utils import load_processed_tokens_from_s3, get_existing_token_ids_from_s3
        
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

    # Step 3: Process markets with checkpointing
    logger.info(f"Processing {len(markets_with_data)} markets with checkpoints every {checkpoint_interval}...")
    
    all_dfs: list[pd.DataFrame] = []
    processed_count = 0
    checkpoint_num = 0
    checkpoint_files: list[str] = []
    total_rows = 0

    for i, market in enumerate(markets_with_data, 1):
        if i % 100 == 0:
            logger.info(f"Processing market {i}/{len(markets_with_data)}...")

        df = processor.process_market(market)
        if df is not None:
            all_dfs.append(df)
            processed_count += 1

        # Save checkpoint every N markets
        if i % checkpoint_interval == 0 and all_dfs:
            checkpoint_num += 1
            logger.info(f"=" * 40)
            logger.info(f"CHECKPOINT {checkpoint_num}: Saving {len(all_dfs)} markets...")
            
            checkpoint_path = save_checkpoint(
                processor, all_dfs, checkpoint_num, use_s3, output_dir
            )
            
            if checkpoint_path:
                checkpoint_files.append(checkpoint_path)
                rows_in_checkpoint = sum(len(df) for df in all_dfs)
                total_rows += rows_in_checkpoint
                logger.info(f"CHECKPOINT {checkpoint_num}: Saved to {checkpoint_path}")
                logger.info(f"CHECKPOINT {checkpoint_num}: {rows_in_checkpoint:,} rows in this batch")
                logger.info(f"CHECKPOINT {checkpoint_num}: {total_rows:,} total rows so far")
            
            # Clear memory
            all_dfs.clear()
            gc.collect()
            logger.info(f"CHECKPOINT {checkpoint_num}: Memory cleared")
            logger.info(f"=" * 40)

    # Save any remaining data as final checkpoint
    if all_dfs:
        checkpoint_num += 1
        logger.info(f"Saving final checkpoint {checkpoint_num} with {len(all_dfs)} remaining markets...")
        
        checkpoint_path = save_checkpoint(
            processor, all_dfs, checkpoint_num, use_s3, output_dir
        )
        
        if checkpoint_path:
            checkpoint_files.append(checkpoint_path)
            rows_in_checkpoint = sum(len(df) for df in all_dfs)
            total_rows += rows_in_checkpoint
            logger.info(f"Final checkpoint saved: {checkpoint_path}")
        
        all_dfs.clear()
        gc.collect()

    logger.info("=" * 60)
    logger.info(f"Processing Complete!")
    logger.info(f"  Total markets processed: {processed_count}")
    logger.info(f"  Total checkpoints: {len(checkpoint_files)}")
    logger.info(f"  Total rows (approx): {total_rows:,}")
    logger.info("=" * 60)

    # Step 4: Combine checkpoints into final file
    if checkpoint_files and not args.skip_combine:
        logger.info("Combining checkpoints into final file...")
        try:
            final_path = combine_checkpoints(processor, checkpoint_files, use_s3, output_dir)
            logger.info("=" * 60)
            logger.info("FINAL OUTPUT READY!")
            logger.info(f"  Location: {final_path}")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Failed to combine checkpoints: {e}")
            logger.info("Individual checkpoint files are still available:")
            for cp in checkpoint_files:
                logger.info(f"  - {cp}")
    else:
        logger.info("Checkpoint files saved:")
        for cp in checkpoint_files:
            logger.info(f"  - {cp}")
        logger.info("\nYou can use any checkpoint file directly as training data!")


if __name__ == "__main__":
    main()
