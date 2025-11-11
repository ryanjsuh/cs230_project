#!/usr/bin/env python3
"""Script to fetch price history for markets from Polymarket CLOB API.

This script reads the markets.json file from step 1, extracts token IDs,
and fetches time-series price data for each token from the CLOB API.

Usage:
    # Using interval (default)
    python scripts/02_fetch_history.py
    python scripts/02_fetch_history.py --interval 1d

    # Using timestamp range
    python scripts/02_fetch_history.py --start-ts 1698700000 --end-ts 1698800000

    # Testing with limited tokens
    python scripts/02_fetch_history.py --max-tokens 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for direct script execution
# Note: For production, prefer running as module (python -m scripts.02_fetch_history)
# or installing the package (pip install -e .)
sys.path.insert(0, str(Path(__file__).parent.parent))

from polymarket_data.config import settings
from polymarket_data.fetch_history import fetch_market_price_histories
from polymarket_data.fetch_markets import Market

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_markets(markets_file: Path) -> list[Market]:
    """Load markets from JSON file with error handling.

    Args:
        markets_file: Path to markets JSON file from step 1

    Returns:
        List of Market objects (skips any that fail to parse)
    """
    logger.info(f"Loading markets from {markets_file}")

    with open(markets_file) as f:
        data = json.load(f)

    # Note: loads all into memory - fine for CS230 scale
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


def extract_token_ids(markets: list[Market]) -> list[str]:
    """Extract all unique token IDs from markets.

    Uses Market.get_token_ids() which handles multiple key formats
    (token_id, tokenId, id).

    Args:
        markets: List of Market objects

    Returns:
        List of unique token IDs
    """
    all_token_ids = []
    for market in markets:
        token_ids = market.get_token_ids()
        all_token_ids.extend(token_ids)

    # Remove duplicates while preserving order
    unique_ids = list(dict.fromkeys(all_token_ids))

    # Filter out empty strings
    unique_ids = [tid for tid in unique_ids if tid]

    logger.info(
        f"Extracted {len(unique_ids)} unique token IDs from {len(markets)} markets"
    )

    return unique_ids


def main() -> None:
    """Main entry point for price history fetching script."""
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

    # Time range options (mutually exclusive: interval OR start/end timestamps)
    time_group = parser.add_argument_group("time range (choose one method)")
    time_group.add_argument(
        "--interval",
        type=str,
        default=None,
        choices=["1m", "1h", "6h", "1d", "1w", "max"],
        help="Price data interval (e.g., 1h). Mutually exclusive with --start-ts/--end-ts",
    )
    time_group.add_argument(
        "--start-ts",
        type=int,
        default=None,
        help="Start timestamp (unix seconds). Requires --end-ts. Mutually exclusive with --interval",
    )
    time_group.add_argument(
        "--end-ts",
        type=int,
        default=None,
        help="End timestamp (unix seconds). Requires --start-ts. Mutually exclusive with --interval",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to fetch (for testing)",
    )

    args = parser.parse_args()

    # Validate time range arguments
    has_interval = args.interval is not None
    has_timestamps = args.start_ts is not None or args.end_ts is not None

    if has_interval and has_timestamps:
        parser.error("Cannot use both --interval and --start-ts/--end-ts")

    if has_timestamps and (args.start_ts is None or args.end_ts is None):
        parser.error("Both --start-ts and --end-ts must be provided together")

    # Default to 1h interval if nothing specified
    if not has_interval and not has_timestamps:
        args.interval = "1h"
        logger.info("No time range specified, defaulting to --interval 1h")

    # Validate input file exists
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
    if args.interval:
        logger.info(f"Interval: {args.interval}")
    else:
        logger.info(f"Time range: {args.start_ts} to {args.end_ts} (unix seconds)")
    if args.max_tokens:
        logger.info(f"Max tokens (testing): {args.max_tokens}")
    logger.info("=" * 60)

    try:
        # Load markets (with error handling for bad markets)
        markets = load_markets(args.markets)

        if not markets:
            logger.error("No valid markets loaded")
            sys.exit(1)

        # Extract token IDs
        token_ids = extract_token_ids(markets)

        if not token_ids:
            logger.warning("No token IDs found in markets")
            sys.exit(0)

        # Limit tokens if requested (for testing)
        if args.max_tokens:
            token_ids = token_ids[: args.max_tokens]
            logger.info(f"Limited to {len(token_ids)} tokens for testing")

        # Fetch price histories
        logger.info(f"\nFetching price history for {len(token_ids)} tokens...")
        histories = fetch_market_price_histories(
            token_ids=token_ids,
            output_dir=args.output,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
            interval=args.interval,
        )

        logger.info("=" * 60)
        logger.info(f"Fetched {len(histories)} token histories")
        logger.info(f"Data saved to: {args.output}")
        logger.info("=" * 60)

        # Print summary stats
        if histories:
            # Count non-empty histories
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
