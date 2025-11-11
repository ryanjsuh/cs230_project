"""Data cleaning and processing for Polymarket time-series data."""

import json
import logging
from pathlib import Path

import pandas as pd

from polymarket_data.config import settings
from polymarket_data.fetch_history import TokenPriceHistory
from polymarket_data.fetch_markets import Market

logger = logging.getLogger(__name__)


class MarketDataProcessor:
    """Process and clean Polymarket market and price data."""

    def __init__(
        self,
        markets_file: Path | str,
        price_history_dir: Path | str,
        resample_freq: str | None = None,
    ) -> None:
        """Initialize data processor.

        Args:
            markets_file: Path to markets JSON file
            price_history_dir: Directory containing price history JSON files
            resample_freq: Pandas frequency string for resampling
                          (default: from settings)
        """
        self.markets_file = Path(markets_file)
        self.price_history_dir = Path(price_history_dir)
        self.resample_freq = resample_freq or settings.resample_frequency

    def load_markets(self) -> list[Market]:
        """Load markets from JSON file.

        Returns:
            List of Market objects
        """
        logger.info(f"Loading markets from {self.markets_file}")

        with open(self.markets_file) as f:
            data = json.load(f)

        markets = []
        for market_dict in data:
            try:
                market = Market.model_validate(market_dict)
                markets.append(market)
            except Exception as e:
                logger.warning(f"Skipping invalid market: {e}")
                continue

        logger.info(f"Loaded {len(markets)} markets")
        return markets

    def load_token_history(self, token_id: str) -> TokenPriceHistory | None:
        """Load price history for a single token.

        Args:
            token_id: Token ID to load

        Returns:
            TokenPriceHistory or None if file doesn't exist or is invalid
        """
        file_path = self.price_history_dir / f"{token_id}.json"

        if not file_path.exists():
            logger.debug(f"No price history file for token {token_id}")
            return None

        try:
            with open(file_path) as f:
                data = json.load(f)

            # Validate required keys
            if "token_id" not in data or "history" not in data:
                logger.warning(
                    f"Invalid format for {token_id}: missing token_id or history"
                )
                return None

            # Convert from saved format back to TokenPriceHistory
            history_points = []
            for point in data["history"]:
                try:
                    # Require both timestamp_unix and price
                    history_points.append({
                        "t": point["timestamp_unix"],
                        "p": point["price"],
                    })
                except KeyError as e:
                    logger.warning(
                        f"Skipping invalid price point in {token_id}: "
                        f"missing {e}"
                    )
                    continue

            history = TokenPriceHistory(
                token_id=data["token_id"],
                history=history_points,
            )
            return history

        except Exception as e:
            logger.warning(f"Failed to load price history for {token_id}: {e}")
            return None

    def process_token_data(
        self,
        market: Market,
        token_id: str,
        token_index: int,
    ) -> pd.DataFrame | None:
        """Process price data for a single token.

        Note: Assumes token ordering matches outcome ordering in the market.
        This is typically true for Polymarket but is an assumption.

        Args:
            market: Market object containing metadata
            token_id: Token ID to process
            token_index: Index of this token in the market's outcomes

        Returns:
            DataFrame with processed data or None if no data
        """
        # Load price history
        history = self.load_token_history(token_id)
        if not history or not history.history:
            logger.debug(f"No price data for token {token_id}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(
            [
                {"timestamp": point.timestamp, "price": point.p}
                for point in history.history
            ]
        )

        # Set timestamp as index
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Resample to fixed frequency and forward-fill
        # Note: This creates a regular time grid but may vary in length
        # across markets depending on their trading history
        df = df.resample(self.resample_freq).mean()
        df["price"] = df["price"].ffill()

        # Drop rows with NaN (at the beginning if no data to ffill from)
        df = df.dropna()

        if df.empty:
            return None

        # Add market metadata with None handling
        df["market_id"] = market.id if market.id else "unknown"
        df["condition_id"] = market.condition_id if market.condition_id else "unknown"
        df["question"] = market.question
        df["category"] = market.category or "Unknown"
        df["token_id"] = token_id
        df["token_index"] = token_index

        # Add outcome label if available
        # Assumes token order matches outcome order (typical for Polymarket)
        if market.outcomes and token_index < len(market.outcomes):
            df["outcome"] = market.outcomes[token_index]
        else:
            df["outcome"] = f"Outcome_{token_index}"

        # Add resolution information
        if market.closed_time:
            try:
                closed_dt = pd.to_datetime(market.closed_time)
                df["resolution_time"] = closed_dt

                # Calculate time until resolution (in hours)
                # Both timestamps are UTC so this subtraction is safe
                df["hours_to_resolution"] = (
                    (closed_dt - df.index).total_seconds() / 3600
                )
            except Exception as e:
                logger.warning(
                    f"Failed to parse closed_time for market {market.id}: {e}"
                )
                df["resolution_time"] = None
                df["hours_to_resolution"] = None
        else:
            df["resolution_time"] = None
            df["hours_to_resolution"] = None

        # Reset index to make timestamp a column
        df = df.reset_index()

        return df

    def process_market(self, market: Market) -> pd.DataFrame | None:
        """Process all tokens for a single market.

        Args:
            market: Market to process

        Returns:
            Combined DataFrame for all tokens in the market, or None
        """
        token_ids = market.get_token_ids()

        if not token_ids:
            logger.debug(f"No token IDs for market {market.id}")
            return None

        market_dfs = []

        for idx, token_id in enumerate(token_ids):
            df = self.process_token_data(market, token_id, idx)
            if df is not None:
                market_dfs.append(df)

        if not market_dfs:
            return None

        # Combine all tokens for this market
        combined = pd.concat(market_dfs, ignore_index=True)

        logger.debug(
            f"Processed market {market.id}: "
            f"{len(token_ids)} tokens, {len(combined)} rows"
        )

        return combined

    def process_all_markets(self) -> pd.DataFrame:
        """Process all markets and combine into single DataFrame.

        Note: Loads all data into memory - suitable for CS230 scale datasets.
        For larger datasets, consider batch processing.

        Returns:
            Combined DataFrame with all market data
        """
        markets = self.load_markets()

        logger.info(f"Processing {len(markets)} markets...")

        all_dfs = []
        processed_count = 0
        skipped_count = 0

        for i, market in enumerate(markets, 1):
            if i % 10 == 0:
                logger.info(f"Processing market {i}/{len(markets)}...")

            df = self.process_market(market)

            if df is not None:
                all_dfs.append(df)
                processed_count += 1
            else:
                skipped_count += 1

        logger.info(
            f"Processed {processed_count} markets "
            f"(skipped {skipped_count} with no data)"
        )

        if not all_dfs:
            raise ValueError("No market data to process")

        # Combine all markets
        combined = pd.concat(all_dfs, ignore_index=True)

        # Sort by timestamp
        combined = combined.sort_values("timestamp")

        logger.info(
            f"Combined dataset: {len(combined)} rows, {len(combined.columns)} columns"
        )

        return combined

    def save_parquet(self, df: pd.DataFrame, output_path: Path | str) -> None:
        """Save DataFrame to Parquet file.

        Args:
            df: DataFrame to save
            output_path: Path to output Parquet file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_path, index=False, engine="pyarrow")

        logger.info(f"Saved {len(df)} rows to {output_path}")

        # Log file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"File size: {size_mb:.2f} MB")


def clean_and_combine_data(
    markets_file: Path | str,
    price_history_dir: Path | str,
    output_file: Path | str,
    resample_freq: str | None = None,
) -> pd.DataFrame:
    """Main entry point for data cleaning and processing.

    Args:
        markets_file: Path to markets JSON file
        price_history_dir: Directory with price history JSON files
        output_file: Path to output Parquet file
        resample_freq: Pandas frequency string for resampling
                      (default: from settings)

    Returns:
        Processed DataFrame
    """
    processor = MarketDataProcessor(
        markets_file=markets_file,
        price_history_dir=price_history_dir,
        resample_freq=resample_freq,
    )

    # Process all data
    df = processor.process_all_markets()

    # Save to Parquet
    processor.save_parquet(df, output_file)

    return df
