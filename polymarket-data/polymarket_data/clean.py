"""
Data cleaning and processing for Polymarket time-series data
"""

import json
import logging
from pathlib import Path

import pandas as pd

from polymarket_data.config import settings
from polymarket_data.fetch_history import TokenPriceHistory
from polymarket_data.fetch_markets import Market

logger = logging.getLogger(__name__)

# Process and clean Polymarket market and price data
class MarketDataProcessor:

    # Initialize data processor
    def __init__(
        self,
        markets_file: Path | str | None = None,
        price_history_dir: Path | str | None = None,
        resample_freq: str | None = None,
        use_s3: bool = False,
    ) -> None:
        self.use_s3 = use_s3
        self.resample_freq = resample_freq or settings.resample_frequency
        
        if use_s3:
            self.markets_file = None
            self.price_history_dir = None
        else:
            if markets_file is None or price_history_dir is None:
                raise ValueError(
                    "markets_file and price_history_dir required when not using S3"
                )
            self.markets_file = Path(markets_file)
            self.price_history_dir = Path(price_history_dir)

    # Load markets from JSON file
    def load_markets(self) -> list[Market]:
        if self.use_s3:
            from polymarket_data.s3_utils import download_json_from_s3
            
            s3_key = settings.get_s3_markets_key()
            logger.info(f"Loading markets from S3: {s3_key}")
            data = download_json_from_s3(s3_key)
        else:
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

    # Load price history for a single token
    def load_token_history(self, token_id: str) -> TokenPriceHistory | None:
        try:
            if self.use_s3:
                from polymarket_data.s3_utils import download_json_from_s3, s3_object_exists
                
                s3_key = settings.get_s3_price_history_key(token_id)
                if not s3_object_exists(s3_key):
                    logger.debug(f"No price history in S3 for token {token_id}")
                    return None
                
                data = download_json_from_s3(s3_key)
            else:
                file_path = self.price_history_dir / f"{token_id}.json"

                if not file_path.exists():
                    logger.debug(f"No price history file for token {token_id}")
                    return None

                with open(file_path) as f:
                    data = json.load(f)

            if "token_id" not in data or "history" not in data:
                logger.warning(
                    f"Invalid format for {token_id}: missing token_id or history"
                )
                return None

            history_points = []
            for point in data["history"]:
                try:
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

    # Process price data for a single token
    def process_token_data(
        self,
        market: Market,
        token_id: str,
        token_index: int,
    ) -> pd.DataFrame | None:        
        # Assumes token ordering matches outcome ordering
        history = self.load_token_history(token_id)
        if not history or not history.history:
            logger.debug(f"No price data for token {token_id}")
            return None

        df = pd.DataFrame(
            [
                {"timestamp": point.timestamp, "price": point.p}
                for point in history.history
            ]
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        df = df.resample(self.resample_freq).mean()
        df["price"] = df["price"].ffill()
        df = df.dropna()

        if df.empty:
            return None

        df["market_id"] = market.id if market.id else "unknown"
        df["condition_id"] = market.condition_id if market.condition_id else "unknown"
        df["question"] = market.question
        df["category"] = market.category or "Unknown"
        df["token_id"] = token_id
        df["token_index"] = token_index

        if market.outcomes and token_index < len(market.outcomes):
            df["outcome"] = market.outcomes[token_index]
        else:
            df["outcome"] = f"Outcome_{token_index}"

        if market.closed_time:
            try:
                closed_dt = pd.to_datetime(market.closed_time)
                df["resolution_time"] = closed_dt
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

        if not df.empty:
            df["final_price"] = df["price"].iloc[-1]
        else:
            df["final_price"] = None

        df = df.reset_index()

        return df

    # Determine winning outcome index from outcome_prices   
    def _determine_winning_outcome(self, market: Market) -> int | None:
        # Returns None if market not resolved or outcome_prices unavailable
        if not market.closed:
            return None

        if not market.outcome_prices or len(market.outcome_prices) == 0:
            return None

        try:
            prices = [float(p) for p in market.outcome_prices]
        except (ValueError, TypeError) as e:
            logger.debug(
                f"Could not parse outcome_prices for market {market.id}: {e}"
            )
            return None

        winning_idx = None
        max_price = -1.0

        for idx, price in enumerate(prices):
            if price > max_price:
                max_price = price
                winning_idx = idx

        if winning_idx is not None and max_price > 0.5:
            return winning_idx

        return None

    # Process all tokens for a single market
    def process_market(self, market: Market) -> pd.DataFrame | None:
        token_ids = market.get_token_ids()

        if not token_ids:
            logger.debug(f"No token IDs for market {market.id}")
            return None

        winning_outcome_idx = self._determine_winning_outcome(market)
        market_dfs = []

        for idx, token_id in enumerate(token_ids):
            df = self.process_token_data(market, token_id, idx)
            if df is not None:
                # Only add target for resolved markets with known winner
                if winning_outcome_idx is not None:
                    df["won"] = int(idx == winning_outcome_idx)
                else:
                    df["won"] = None

                market_dfs.append(df)

        if not market_dfs:
            return None

        combined = pd.concat(market_dfs, ignore_index=True)

        logger.debug(
            f"Processed market {market.id}: "
            f"{len(token_ids)} tokens, {len(combined)} rows, "
            f"winner_idx={winning_outcome_idx}"
        )

        return combined

    # Finalize combined DataFrame with filtering and sorting
    def _finalize_dataframe(self, all_dfs: list[pd.DataFrame]) -> pd.DataFrame:
        if not all_dfs:
            raise ValueError("No market data to process")

        combined = pd.concat(all_dfs, ignore_index=True)

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

        combined = combined.sort_values("timestamp")
        combined["won"] = combined["won"].astype(int)

        logger.info(
            f"Combined dataset: {len(combined)} rows, {len(combined.columns)} columns"
        )

        if "won" in combined.columns:
            win_counts = combined["won"].value_counts()
            logger.info(
                f"Target variable distribution: "
                f"Won={win_counts.get(1, 0)}, Lost={win_counts.get(0, 0)}"
            )

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

    # Process all markets and combine into single DataFrame
    def process_all_markets(
        self,
        checkpoint_interval: int | None = None,
        checkpoint_path: Path | str | None = None,
        checkpoint_s3_key: str | None = None,
        max_markets: int | None = None,
    ) -> pd.DataFrame:
        """
        Process all markets with optional checkpointing.
        
        Args:
            checkpoint_interval: Save checkpoint every N markets (e.g., 1000)
            checkpoint_path: Local path for checkpoint parquet files
            checkpoint_s3_key: S3 key for checkpoint parquet files
            max_markets: Stop after processing this many markets (for partial runs)
        """
        markets = self.load_markets()
        
        if max_markets:
            markets = markets[:max_markets]
            logger.info(f"Limited to first {max_markets} markets")

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

            # Checkpoint save
            if checkpoint_interval and i % checkpoint_interval == 0 and all_dfs:
                logger.info(f"Saving checkpoint at market {i}...")
                try:
                    checkpoint_df = self._finalize_dataframe(all_dfs.copy())
                    if checkpoint_s3_key:
                        self.save_parquet(checkpoint_df, s3_key=checkpoint_s3_key)
                    elif checkpoint_path:
                        self.save_parquet(checkpoint_df, output_path=checkpoint_path)
                    logger.info(f"Checkpoint saved: {processed_count} markets, {len(checkpoint_df)} rows")
                except Exception as e:
                    logger.warning(f"Checkpoint save failed: {e}")

        logger.info(
            f"Processed {processed_count} markets "
            f"(skipped {skipped_count} with no data)"
        )

        return self._finalize_dataframe(all_dfs)

    # Save DataFrame to Parquet file
    def save_parquet(
        self,
        df: pd.DataFrame,
        output_path: Path | str | None = None,
        s3_key: str | None = None,
    ) -> None:
        """
        Save DataFrame to Parquet file.
        
        Args:
            df: DataFrame to save
            output_path: Local file path (ignored if s3_key provided)
            s3_key: S3 object key to upload directly to S3
        """
        if s3_key is not None:
            import io
            from polymarket_data.s3_utils import upload_bytes_to_s3
            
            # Write parquet to in-memory buffer
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False, engine="pyarrow")
            parquet_bytes = buffer.getvalue()
            
            upload_bytes_to_s3(
                parquet_bytes,
                s3_key,
                content_type="application/octet-stream",
            )
            
            size_mb = len(parquet_bytes) / (1024 * 1024)
            logger.info(f"Saved {len(df)} rows to S3: {s3_key}")
            logger.info(f"File size: {size_mb:.2f} MB")
            return
        
        if output_path is None:
            raise ValueError("Either output_path or s3_key must be provided")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_path, index=False, engine="pyarrow")

        logger.info(f"Saved {len(df)} rows to {output_path}")

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"File size: {size_mb:.2f} MB")

# Clean and combine market data into a single DataFrame
def clean_and_combine_data(
    markets_file: Path | str | None = None,
    price_history_dir: Path | str | None = None,
    output_file: Path | str | None = None,
    resample_freq: str | None = None,
    use_s3: bool = False,
    s3_output_key: str | None = None,
    checkpoint_interval: int | None = None,
    max_markets: int | None = None,
) -> pd.DataFrame:
    processor = MarketDataProcessor(
        markets_file=markets_file,
        price_history_dir=price_history_dir,
        resample_freq=resample_freq,
        use_s3=use_s3,
    )

    df = processor.process_all_markets(
        checkpoint_interval=checkpoint_interval,
        checkpoint_path=output_file,
        checkpoint_s3_key=s3_output_key,
        max_markets=max_markets,
    )
    
    if s3_output_key:
        processor.save_parquet(df, s3_key=s3_output_key)
    elif output_file:
        processor.save_parquet(df, output_path=output_file)
    else:
        logger.warning("No output path specified, DataFrame not saved")

    return df
