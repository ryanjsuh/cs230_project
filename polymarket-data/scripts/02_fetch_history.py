#!/usr/bin/env python3
"""
Script to fetch price history for markets from Polymarket CLOB API

Usage:
    # Fetch to S3
    POLYMARKET_S3_BUCKET=cs230-polymarket-data-1 python scripts/02_fetch_history.py --s3
    
    # Fetch to local directory (legacy)
    python scripts/02_fetch_history.py --output data/raw/price_history
    
    # Other options
    python scripts/02_fetch_history.py --s3 --max-tokens 50
    python scripts/02_fetch_history.py --s3 --start-ts 1698700000 --end-ts 1698800000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from polymarket_data.config import settings
from polymarket_data.fetch_history import fetch_market_price_histories
from polymarket_data.fetch_markets import Market

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Load markets from JSON file
def load_markets(markets_file: Path | None = None, use_s3: bool = False) -> list[Market]:
    if use_s3:
        from polymarket_data.s3_utils import download_json_from_s3
        
        s3_key = settings.get_s3_markets_key()
        logger.info(f"Loading markets from S3: s3://{settings.s3_bucket}/{s3_key}")
        data = download_json_from_s3(s3_key)
    else:
        if markets_file is None:
            raise ValueError("markets_file required when not using S3")
        logger.info(f"Loading markets from {markets_file}")
        with open(markets_file) as f:
            data = json.load(f)

    markets: list[Market] = []
    skipped = 0

    for i, market_dict in enumerate(data):
        try:
            market = Market.model_validate(market_dict)
            markets.append(market)
        except Exception as e:
            logger.warning(
                f"Skipping market at index {i} (failed to parse): {e}"
            )
            skipped += 1

    logger.info(
        f"Loaded {len(markets)} markets"
        + (f" (skipped {skipped})" if skipped > 0 else "")
    )

    return markets


# Extract all unique token IDs from markets
def extract_token_ids(markets: list[Market]) -> list[str]:
    all_token_ids = []
    for market in markets:
        token_ids = market.get_token_ids()
        all_token_ids.extend(token_ids)

    unique_ids = list(dict.fromkeys(all_token_ids))
    unique_ids = [tid for tid in unique_ids if tid]

    logger.info(
        f"Extracted {len(unique_ids)} unique token IDs from {len(markets)} markets"
    )

    return unique_ids


# Build mapping of token_id -> (start_ts, end_ts) from market metadata
# If token appears in multiple markets, uses earliest start and latest end
def build_token_time_ranges(markets: list[Market]) -> dict[str, tuple[int, int]]:
    token_ranges: dict[str, tuple[int, int]] = {}
    markets_with_ranges = 0
    markets_without_ranges = 0

    for market in markets:
        time_range = market.get_time_range()
        if time_range is None:
            markets_without_ranges += 1
            logger.debug(
                f"Market {market.id or market.slug or 'unknown'}: "
                "could not determine time range"
            )
            continue

        markets_with_ranges += 1
        start_ts, end_ts = time_range
        token_ids = market.get_token_ids()

        for token_id in token_ids:
            if not token_id:
                continue

            if token_id in token_ranges:
                prev_start, prev_end = token_ranges[token_id]
                token_ranges[token_id] = (
                    min(prev_start, start_ts),
                    max(prev_end, end_ts),
                )
            else:
                token_ranges[token_id] = (start_ts, end_ts)

    logger.info(
        f"Built time ranges for {len(token_ranges)} tokens from "
        f"{markets_with_ranges} markets (skipped {markets_without_ranges} markets)"
    )

    return token_ranges


# Main entry point for price history fetching script
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch price history for Polymarket tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--markets",
        type=Path,
        default=None,
        help="Input markets JSON file from step 1 (local path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for price history JSON files (local path)",
    )
    parser.add_argument(
        "--s3",
        action="store_true",
        help="Use S3 for both input (markets.json) and output (price histories). "
        "Requires POLYMARKET_S3_BUCKET env var.",
    )

    time_group = parser.add_argument_group("time range")
    time_group.add_argument(
        "--start-ts",
        type=int,
        default=None,
        help="Override start timestamp (unix seconds) for all tokens. "
        "By default, uses timestamps from market metadata.",
    )
    time_group.add_argument(
        "--end-ts",
        type=int,
        default=None,
        help="Override end timestamp (unix seconds) for all tokens. "
        "By default, uses timestamps from market metadata.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to fetch (for testing)",
    )

    args = parser.parse_args()

    has_start = args.start_ts is not None
    has_end = args.end_ts is not None

    if has_start ^ has_end:
        parser.error("Both --start-ts and --end-ts must be provided together")

    use_override_range = has_start and has_end
    use_s3 = args.s3

    # Validate S3 configuration
    if use_s3:
        if not settings.use_s3:
            logger.error(
                "S3 mode requested but POLYMARKET_S3_BUCKET env var not set.\n"
                "Set it with: export POLYMARKET_S3_BUCKET=your-bucket-name"
            )
            sys.exit(1)
    else:
        # Local mode: check markets file exists
        markets_file = args.markets or Path(settings.raw_data_dir) / "markets.json"
        if not markets_file.exists():
            logger.error(f"Markets file not found: {markets_file}")
            logger.error(
                "Please run scripts/01_fetch_markets.py first to generate markets.json"
            )
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Polymarket Price History Fetcher")
    logger.info("=" * 60)
    
    if use_s3:
        logger.info(f"Mode: S3 (bucket: {settings.s3_bucket})")
        logger.info(f"Markets source: s3://{settings.s3_bucket}/{settings.get_s3_markets_key()}")
        logger.info(f"Output: s3://{settings.s3_bucket}/{settings.s3_prefix}/raw/price_history/")
    else:
        markets_file = args.markets or Path(settings.raw_data_dir) / "markets.json"
        output_dir = args.output or Path(settings.raw_data_dir) / "price_history"
        logger.info(f"Markets file: {markets_file}")
        logger.info(f"Output directory: {output_dir}")
    
    if use_override_range:
        logger.info(
            f"Using override time range: {args.start_ts} to {args.end_ts} (unix seconds)"
        )
    else:
        logger.info("Using time ranges from market metadata")
    if args.max_tokens:
        logger.info(f"Max tokens (testing): {args.max_tokens}")
    logger.info("=" * 60)

    try:
        # Load markets
        if use_s3:
            markets = load_markets(use_s3=True)
        else:
            markets_file = args.markets or Path(settings.raw_data_dir) / "markets.json"
            markets = load_markets(markets_file=markets_file)

        if not markets:
            logger.error("No valid markets loaded")
            sys.exit(1)

        token_ids = extract_token_ids(markets)

        if not token_ids:
            logger.warning("No token IDs found in markets")
            sys.exit(0)

        token_time_ranges = None
        if not use_override_range:
            token_time_ranges = build_token_time_ranges(markets)

            token_time_ranges = {
                tid: token_time_ranges[tid]
                for tid in token_ids
                if tid in token_time_ranges
            }

            if token_time_ranges:
                logger.info(
                    f"Using per-token time ranges for {len(token_time_ranges)}/{len(token_ids)} tokens"
                )
            else:
                logger.warning(
                    "No time ranges available from market metadata. "
                    "You may need to provide --start-ts and --end-ts."
                )

        if args.max_tokens:
            token_ids = token_ids[: args.max_tokens]
            logger.info(f"Limited to {len(token_ids)} tokens for testing")
            if token_time_ranges:
                token_time_ranges = {
                    tid: token_time_ranges[tid]
                    for tid in token_ids
                    if tid in token_time_ranges
                }

        logger.info(f"\nFetching price history for {len(token_ids)} tokens...")
        
        if use_s3:
            histories = fetch_market_price_histories(
                token_ids=token_ids,
                start_ts=args.start_ts,
                end_ts=args.end_ts,
                token_time_ranges=token_time_ranges,
                use_s3=True,
            )
            output_location = f"s3://{settings.s3_bucket}/{settings.s3_prefix}/raw/price_history/"
        else:
            output_dir = args.output or Path(settings.raw_data_dir) / "price_history"
            histories = fetch_market_price_histories(
                token_ids=token_ids,
                output_dir=output_dir,
                start_ts=args.start_ts,
                end_ts=args.end_ts,
                token_time_ranges=token_time_ranges,
            )
            output_location = str(output_dir)

        logger.info("=" * 60)
        logger.info(f"Fetched price histories for {len(token_ids)} tokens")
        logger.info(f"Data saved to: {output_location}")
        logger.info("=" * 60)

        if histories:
            non_empty = [h for h in histories if h.history]
            empty_count = len(histories) - len(non_empty)

            total_points = sum(len(h.history) for h in histories)
            avg_points = total_points / len(non_empty) if non_empty else 0

            logger.info("\nSummary:")
            logger.info(f"  Total tokens requested: {len(token_ids)}")
            logger.info(f"  Tokens with price data: {len(non_empty)}")
            if empty_count > 0:
                logger.info(f"  Tokens with no data: {empty_count}")
            logger.info(f"  Total price points: {total_points}")
            if non_empty:
                logger.info(f"  Average points per token: {avg_points:.1f}")

    except Exception as e:
        logger.error(f"Error fetching price histories: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
