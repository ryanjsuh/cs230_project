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

        # Add final price (price at resolution time) for feature engineering
        # Use the last available price before/at resolution
        if not df.empty:
            df["final_price"] = df["price"].iloc[-1]
        else:
            df["final_price"] = None

        # Reset index to make timestamp a column
        df = df.reset_index()

        return df

    def _determine_winning_outcome(self, market: Market) -> int | None:
        """Determine which outcome index won the market based on outcome_prices.

        For resolved markets, the winning outcome should have a price close to 1.0.
        If outcome_prices are not available or market is not resolved, returns None.

        Args:
            market: Market object

        Returns:
            Index of winning outcome (0-based), or None if cannot determine
        """
        # Only process resolved/closed markets
        if not market.closed:
            return None

        # Need outcome_prices to determine winner
        if not market.outcome_prices or len(market.outcome_prices) == 0:
            return None

        # Parse outcome prices (they come as strings from API)
        try:
            prices = [float(p) for p in market.outcome_prices]
        except (ValueError, TypeError) as e:
            logger.debug(
                f"Could not parse outcome_prices for market {market.id}: {e}"
            )
            return None

        # Find outcome with price closest to 1.0 (winner)
        # In Polymarket, winning outcome resolves to ~1.0, losers to ~0.0
        winning_idx = None
        max_price = -1.0

        for idx, price in enumerate(prices):
            if price > max_price:
                max_price = price
                winning_idx = idx

        # Sanity check: winner should have price > 0.5
        if winning_idx is not None and max_price > 0.5:
            return winning_idx

        return None

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

        # Determine winning outcome for resolved markets
        winning_outcome_idx = self._determine_winning_outcome(market)

        market_dfs = []

        for idx, token_id in enumerate(token_ids):
            df = self.process_token_data(market, token_id, idx)
            if df is not None:
                # Add target variable: 1 if this token won, 0 otherwise
                # Only add target for resolved markets with known winner
                if winning_outcome_idx is not None:
                    df["won"] = (idx == winning_outcome_idx).astype(int)
                else:
                    # For unresolved markets or markets without outcome_prices,
                    # set won to None (will be filtered out later)
                    df["won"] = None

                market_dfs.append(df)

        if not market_dfs:
            return None

        # Combine all tokens for this market
        combined = pd.concat(market_dfs, ignore_index=True)

        logger.debug(
            f"Processed market {market.id}: "
            f"{len(token_ids)} tokens, {len(combined)} rows, "
            f"winner_idx={winning_outcome_idx}"
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

        # Filter to only include resolved markets with known winners
        # (i.e., rows where 'won' is not None)
        initial_rows = len(combined)
        combined = combined[combined["won"].notna()].copy()
        filtered_rows = len(combined)

        if filtered_rows < initial_rows:
            logger.info(
                f"Filtered out {initial_rows - filtered_rows} rows "
                f"from unresolved markets or markets without outcome_prices"
            )

        if combined.empty:
            raise ValueError(
                "No resolved market data with known winners to process. "
                "Ensure markets are closed and have outcome_prices."
            )

        # Sort by timestamp
        combined = combined.sort_values("timestamp")

        # Ensure won column is integer type (0 or 1)
        combined["won"] = combined["won"].astype(int)

        # Data quality checks
        logger.info(
            f"Combined dataset: {len(combined)} rows, {len(combined.columns)} columns"
        )

        # Log target variable distribution
        if "won" in combined.columns:
            win_counts = combined["won"].value_counts()
            logger.info(
                f"Target variable distribution: "
                f"Won={win_counts.get(1, 0)}, Lost={win_counts.get(0, 0)}"
            )

        # Check for missing values in critical columns
        critical_cols = ["price", "timestamp", "market_id", "won"]
        missing_info = {}
        for col in critical_cols:
            if col in combined.columns:
                missing_count = combined[col].isna().sum()
                if missing_count > 0:
                    missing_info[col] = missing_count

        if missing_info:
            logger.warning(f"Missing values in critical columns: {missing_info}")

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
