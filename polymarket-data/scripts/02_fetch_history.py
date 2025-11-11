#!/usr/bin/env python3
"""
Script to fetch price history for markets from Polymarket CLOB API

Usage:
    python scripts/02_fetch_history.py
    python scripts/02_fetch_history.py --start-ts 1698700000 --end-ts 1698800000
    python scripts/02_fetch_history.py --max-tokens 50
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
def load_markets(markets_file: Path) -> list[Market]:
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
        default=Path(settings.raw_data_dir) / "markets.json",
        help="Input markets JSON file from step 1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.raw_data_dir) / "price_history",
        help="Output directory for price history JSON files",
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

    if not args.markets.exists():
        logger.error(f"Markets file not found: {args.markets}")
        logger.error(
            "Please run scripts/01_fetch_markets.py first to generate markets.json"
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Polymarket Price History Fetcher")
    logger.info("=" * 60)
    logger.info(f"Markets file: {args.markets}")
    logger.info(f"Output directory: {args.output}")
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
        markets = load_markets(args.markets)

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
        histories = fetch_market_price_histories(
            token_ids=token_ids,
            output_dir=args.output,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
            token_time_ranges=token_time_ranges,
        )

        logger.info("=" * 60)
        logger.info(f"Fetched {len(histories)} token histories")
        logger.info(f"Data saved to: {args.output}")
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
