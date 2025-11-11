"""Fetch price history from Polymarket CLOB API."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from polymarket_data.config import settings

logger = logging.getLogger(__name__)


class PricePoint(BaseModel):
    """Single price observation for a token."""

    t: int = Field(..., description="Unix timestamp in seconds")
    p: float = Field(..., description="Price (0.0 to 1.0)")

    @property
    def timestamp(self) -> datetime:
        """Convert unix timestamp to datetime."""
        return datetime.fromtimestamp(self.t, tz=timezone.utc)


class TokenPriceHistory(BaseModel):
    """Price history for a single token."""

    token_id: str = Field(..., description="Token ID")
    history: list[PricePoint] = Field(
        default_factory=list, description="List of price points"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with readable timestamps."""
        return {
            "token_id": self.token_id,
            "history": [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "timestamp_unix": point.t,
                    "price": point.p,
                }
                for point in self.history
            ],
        }


class PriceHistoryFetcher:
    """Fetches price history from Polymarket CLOB API.

    CLOB API reference:
    https://docs.polymarket.com/developers/CLOB/timeseries
    https://docs.polymarket.com/api-reference/pricing/get-price-history-for-a-traded-token

    Important: interval is mutually exclusive with startTs/endTs.
    Valid intervals: 1m, 1h, 6h, 1d, 1w, max
    """

    def __init__(self, client: httpx.Client | None = None) -> None:
        """Initialize price history fetcher.

        Args:
            client: Optional httpx client. If None, creates a new one.
        """
        self.client = client or httpx.Client(timeout=settings.request_timeout)
        self._owned_client = client is None

    def __enter__(self) -> "PriceHistoryFetcher":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        if self._owned_client:
            self.client.close()

    def fetch_token_history(
        self,
        token_id: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
        interval: str | None = None,
    ) -> TokenPriceHistory:
        """Fetch price history for a single token.

        Note: interval is mutually exclusive with start_ts/end_ts.
        If interval is provided, start_ts and end_ts are ignored.

        Args:
            token_id: Token ID to fetch history for
            start_ts: Start timestamp (unix seconds). Requires end_ts.
            end_ts: End timestamp (unix seconds). Requires start_ts.
            interval: Price interval from: 1m, 1h, 6h, 1d, 1w, max.
                     Mutually exclusive with start_ts/end_ts.

        Returns:
            TokenPriceHistory object

        Raises:
            httpx.HTTPStatusError: If request fails
            ValueError: If invalid parameter combination
        """
        # Build query params - use "market" as documented
        params: dict[str, Any] = {"market": token_id}

        # Validate and add time parameters
        if interval is not None:
            # Using interval - ignore start/end
            valid_intervals = ["1m", "1h", "6h", "1d", "1w", "max"]
            if interval not in valid_intervals:
                raise ValueError(
                    f"Invalid interval '{interval}'. "
                    f"Must be one of: {valid_intervals}"
                )
            params["interval"] = interval
            if start_ts is not None or end_ts is not None:
                logger.warning(
                    "Both interval and start_ts/end_ts provided. "
                    "Using interval only (per API docs)."
                )
        elif start_ts is not None and end_ts is not None:
            # Using start/end timestamps
            params["startTs"] = start_ts
            params["endTs"] = end_ts
        elif start_ts is not None or end_ts is not None:
            raise ValueError("Both start_ts and end_ts must be provided together")
        else:
            # No time params - API will return some default range
            logger.debug(f"Fetching all available history for token {token_id}")

        logger.debug(f"Fetching price history for token {token_id}: params={params}")

        try:
            # Use query params, not path segment
            response = self.client.get(
                settings.clob_price_history_url,
                params=params,
            )
            response.raise_for_status()

            # Rate limiting
            time.sleep(settings.rate_limit_delay)

            # Parse response
            data = response.json()

            # Handle response format
            if isinstance(data, dict) and "history" in data:
                # Expected format per docs: {"history": [...]}
                history = [
                    PricePoint.model_validate(point) for point in data["history"]
                ]
            elif isinstance(data, list):
                # Alternative format (if API returns bare list)
                logger.info(
                    f"Got bare list response for token {token_id} "
                    "(expected dict with 'history' key)"
                )
                history = [PricePoint.model_validate(point) for point in data]
            else:
                logger.warning(
                    f"Unexpected response format for token {token_id}: {type(data)}"
                )
                history = []

            return TokenPriceHistory(token_id=token_id, history=history)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"No price history found for token {token_id}")
                return TokenPriceHistory(token_id=token_id, history=[])
            else:
                logger.error(
                    f"HTTP error fetching token {token_id}: "
                    f"{e.response.status_code} - {e.response.text}"
                )
                raise

        except Exception as e:
            logger.error(f"Error fetching price history for token {token_id}: {e}")
            raise

    def fetch_multiple_tokens(
        self,
        token_ids: list[str],
        start_ts: int | None = None,
        end_ts: int | None = None,
        interval: str | None = None,
    ) -> list[TokenPriceHistory]:
        """Fetch price history for multiple tokens.

        Args:
            token_ids: List of token IDs to fetch
            start_ts: Start timestamp (unix seconds)
            end_ts: End timestamp (unix seconds)
            interval: Price interval (mutually exclusive with start_ts/end_ts)

        Returns:
            List of TokenPriceHistory objects
        """
        results: list[TokenPriceHistory] = []

        logger.info(f"Fetching price history for {len(token_ids)} tokens")

        for i, token_id in enumerate(token_ids, 1):
            logger.info(f"Fetching token {i}/{len(token_ids)}: {token_id}")

            try:
                history = self.fetch_token_history(
                    token_id=token_id,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    interval=interval,
                )
                results.append(history)

            except Exception as e:
                logger.error(f"Failed to fetch token {token_id}: {e}")
                # Continue with other tokens
                continue

        logger.info(
            f"Successfully fetched {len(results)}/{len(token_ids)} token histories"
        )
        return results

    def save_histories(
        self,
        histories: list[TokenPriceHistory],
        output_dir: Path | str,
    ) -> None:
        """Save token price histories to individual JSON files.

        Args:
            histories: List of TokenPriceHistory objects
            output_dir: Directory to save JSON files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        for history in histories:
            if not history.history:
                logger.warning(f"Skipping token {history.token_id} - no price data")
                continue

            output_path = output_dir / f"{history.token_id}.json"

            with open(output_path, "w") as f:
                json.dump(history.to_dict(), f, indent=2)

            logger.debug(
                f"Saved {len(history.history)} price points for "
                f"{history.token_id} to {output_path}"
            )
            saved_count += 1

        logger.info(f"Saved {saved_count}/{len(histories)} token histories to {output_dir}")


def fetch_market_price_histories(
    token_ids: list[str],
    output_dir: Path | str | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
    interval: str | None = None,
) -> list[TokenPriceHistory]:
    """Fetch price histories for market tokens.

    Main entry point for price history fetching.

    Args:
        token_ids: List of token IDs (called "market" in CLOB API)
        output_dir: Optional directory to save price histories
        start_ts: Start timestamp (unix seconds, requires end_ts)
        end_ts: End timestamp (unix seconds, requires start_ts)
        interval: Price interval from: 1m, 1h, 6h, 1d, 1w, max.
                 Mutually exclusive with start_ts/end_ts.
                 If None and no timestamps, API returns default range.

    Returns:
        List of TokenPriceHistory objects
    """
    with PriceHistoryFetcher() as fetcher:
        histories = fetcher.fetch_multiple_tokens(
            token_ids=token_ids,
            start_ts=start_ts,
            end_ts=end_ts,
            interval=interval,
        )

        # Save if output directory provided
        if output_dir:
            fetcher.save_histories(histories, output_dir)

        return histories
