#!/usr/bin/env python3
"""Script to fetch resolved markets from Polymarket Gamma API.

This script fetches all markets that were closed/resolved within the last
365 days (configurable) and saves them to a JSON file.

Usage:
    python scripts/01_fetch_markets.py
    python scripts/01_fetch_markets.py --lookback-days 180
    python scripts/01_fetch_markets.py --output data/raw/my_markets.json
    python scripts/01_fetch_markets.py --max-markets 100  # For testing
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for direct script execution
# Note: For production, prefer running as module (python -m scripts.01_fetch_markets)
# or installing the package (pip install -e .)
sys.path.insert(0, str(Path(__file__).parent.parent))

from polymarket_data.config import settings
from polymarket_data.fetch_markets import fetch_resolved_markets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for market fetching script."""
    parser = argparse.ArgumentParser(
        description="Fetch resolved Polymarket markets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=settings.lookback_days,
        help=f"Days to look back for resolved markets (default: {settings.lookback_days})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.raw_data_dir) / "markets.json",
        help="Output path for markets JSON file",
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=None,
        help="Maximum number of markets to fetch (for testing)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Polymarket Market Fetcher")
    logger.info("=" * 60)
    logger.info(f"Lookback days: {args.lookback_days}")
    logger.info(f"Output path: {args.output}")
    if args.max_markets:
        logger.info(f"Max markets (testing): {args.max_markets}")
    logger.info("=" * 60)

    try:
        # Fetch markets
        markets = fetch_resolved_markets(
            lookback_days=args.lookback_days,
            output_path=args.output,
            max_markets=args.max_markets,
        )

        logger.info("=" * 60)
        logger.info(f"Successfully fetched {len(markets)} markets")
        logger.info(f"Saved to: {args.output}")
        logger.info("=" * 60)

        # Print some summary stats
        if markets:
            logger.info("\nSummary:")
            logger.info(f"  Total markets: {len(markets)}")

            # Count by category
            categories = {}
            for market in markets:
                cat = market.category or "Unknown"
                categories[cat] = categories.get(cat, 0) + 1

            logger.info(f"  Unique categories: {len(categories)}")
            if categories:
                logger.info("  Top categories:")
                for cat, count in sorted(
                    categories.items(), key=lambda x: x[1], reverse=True
                )[:5]:
                    logger.info(f"    {cat}: {count}")

    except Exception as e:
        logger.error(f"Error fetching markets: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
