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

# Maximum chunk size for API requests (15 days in seconds)
# The API has limits on how long a time range can be
MAX_CHUNK_SECONDS = 60 * 60 * 24 * 15  # 15 days
MIN_CHUNK_SECONDS = 60 * 60  # 1 hour minimum


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
        # If using interval, fetch directly
        if interval is not None:
            valid_intervals = ["1m", "1h", "6h", "1d", "1w", "max"]
            if interval not in valid_intervals:
                raise ValueError(
                    f"Invalid interval '{interval}'. "
                    f"Must be one of: {valid_intervals}"
                )
            if start_ts is not None or end_ts is not None:
                logger.warning(
                    "Both interval and start_ts/end_ts provided. "
                    "Using interval only (per API docs)."
                )
            return self._fetch_with_interval(token_id, interval)

        # If using timestamps, use chunking logic
        if start_ts is not None and end_ts is not None:
            if end_ts <= start_ts:
                raise ValueError("end_ts must be greater than start_ts")
            return self._fetch_with_timestamps_chunked(token_id, start_ts, end_ts)

        if start_ts is not None or end_ts is not None:
            raise ValueError("Both start_ts and end_ts must be provided together")

        # No time params - API will return some default range
        logger.debug(f"Fetching all available history for token {token_id}")
        return self._fetch_with_interval(token_id, "max")

    def _fetch_with_interval(
        self, token_id: str, interval: str
    ) -> TokenPriceHistory:
        """Fetch history using interval parameter."""
        params: dict[str, Any] = {"market": token_id, "interval": interval}
        logger.debug(f"Fetching price history for token {token_id}: params={params}")

        try:
            response = self.client.get(
                settings.clob_price_history_url,
                params=params,
            )
            response.raise_for_status()
            time.sleep(settings.rate_limit_delay)
            data = response.json()
            history = self._parse_history_response(token_id, data)
            return TokenPriceHistory(token_id=token_id, history=history)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"No price history found for token {token_id}")
                return TokenPriceHistory(token_id=token_id, history=[])
            raise

    def _fetch_with_timestamps_chunked(
        self, token_id: str, start_ts: int, end_ts: int
    ) -> TokenPriceHistory:
        """Fetch history using timestamps with automatic chunking for long ranges."""
        duration = end_ts - start_ts

        # If range is small enough, try direct fetch
        if duration <= MAX_CHUNK_SECONDS:
            try:
                return self._fetch_chunk(token_id, start_ts, end_ts)
            except httpx.HTTPStatusError as e:
                if self._is_interval_too_long_error(e):
                    # Even though range is small, API rejected it - split further
                    logger.debug(
                        f"API rejected range {start_ts}-{end_ts} for token {token_id}, "
                        "splitting into smaller chunks"
                    )
                    return self._fetch_chunked_recursive(token_id, start_ts, end_ts)
                raise

        # Range is too long - split into chunks
        logger.debug(
            f"Range {start_ts}-{end_ts} ({duration} seconds) exceeds max chunk size "
            f"({MAX_CHUNK_SECONDS} seconds), splitting into chunks"
        )
        return self._fetch_chunked_recursive(token_id, start_ts, end_ts)

    def _fetch_chunked_recursive(
        self, token_id: str, start_ts: int, end_ts: int
    ) -> TokenPriceHistory:
        """Recursively fetch history by splitting into smaller chunks."""
        all_points: list[PricePoint] = []
        current_start = start_ts

        while current_start < end_ts:
            current_end = min(current_start + MAX_CHUNK_SECONDS, end_ts)

            try:
                chunk_points = self._fetch_chunk(token_id, current_start, current_end)
                all_points.extend(chunk_points)
                current_start = current_end
            except httpx.HTTPStatusError as e:
                if self._is_interval_too_long_error(e):
                    # Chunk is still too long - split it in half
                    duration = current_end - current_start
                    if duration <= MIN_CHUNK_SECONDS:
                        logger.error(
                            f"Cannot split range {current_start}-{current_end} "
                            f"further (duration {duration}s <= min {MIN_CHUNK_SECONDS}s)"
                        )
                        raise

                    midpoint = current_start + (duration // 2)
                    logger.debug(
                        f"Splitting chunk {current_start}-{current_end} at {midpoint}"
                    )
                    # Recursively fetch both halves
                    first_half = self._fetch_chunked_recursive(
                        token_id, current_start, midpoint
                    )
                    second_half = self._fetch_chunked_recursive(
                        token_id, midpoint, current_end
                    )
                    all_points.extend(first_half.history)
                    all_points.extend(second_half.history)
                    current_start = current_end
                else:
                    raise

        # Deduplicate and sort by timestamp
        unique_points = self._deduplicate_points(all_points)
        return TokenPriceHistory(token_id=token_id, history=unique_points)

    def _fetch_chunk(
        self, token_id: str, start_ts: int, end_ts: int
    ) -> list[PricePoint]:
        """Fetch a single chunk of price history."""
        params: dict[str, Any] = {
            "market": token_id,
            "startTs": start_ts,
            "endTs": end_ts,
        }
        logger.debug(
            f"Fetching chunk for token {token_id}: {start_ts} to {end_ts} "
            f"({end_ts - start_ts} seconds)"
        )

        response = self.client.get(
            settings.clob_price_history_url,
            params=params,
        )

        if response.status_code == 404:
            logger.debug(
                f"Token {token_id} returned 404 for range {start_ts}-{end_ts}"
            )
            time.sleep(settings.rate_limit_delay)
            return []

        response.raise_for_status()
        time.sleep(settings.rate_limit_delay)
        data = response.json()
        return self._parse_history_response(token_id, data)

    def _parse_history_response(
        self, token_id: str, data: Any
    ) -> list[PricePoint]:
        """Parse API response into PricePoint list."""
        if isinstance(data, dict) and "history" in data:
            payload = data.get("history") or []
        elif isinstance(data, list):
            logger.info(
                f"Got bare list response for token {token_id} "
                "(expected dict with 'history' key)"
            )
            payload = data
        else:
            logger.warning(
                f"Unexpected response format for token {token_id}: {type(data)}"
            )
            return []

        return [PricePoint.model_validate(point) for point in payload]

    def _is_interval_too_long_error(self, error: httpx.HTTPStatusError) -> bool:
        """Check if error indicates the time interval is too long."""
        if error.response is None or error.response.status_code != 400:
            return False

        try:
            error_text = error.response.text.lower()
            return "interval is too long" in error_text or "startts and endts interval is too long" in error_text
        except Exception:
            return False

    def _deduplicate_points(self, points: list[PricePoint]) -> list[PricePoint]:
        """Remove duplicate timestamps and sort by timestamp."""
        # Use dict to keep last point for each timestamp
        unique: dict[int, PricePoint] = {}
        for point in points:
            unique[point.t] = point

        # Sort by timestamp
        return [unique[ts] for ts in sorted(unique.keys())]

    def fetch_multiple_tokens(
        self,
        token_ids: list[str],
        start_ts: int | None = None,
        end_ts: int | None = None,
        interval: str | None = None,
        token_time_ranges: dict[str, tuple[int, int]] | None = None,
    ) -> list[TokenPriceHistory]:
        """Fetch price history for multiple tokens.

        Args:
            token_ids: List of token IDs to fetch
            start_ts: Default start timestamp (unix seconds) for tokens without
                an explicit range in token_time_ranges
            end_ts: Default end timestamp (unix seconds) for tokens without
                an explicit range in token_time_ranges
            interval: Price interval (mutually exclusive with start_ts/end_ts).
                Only used if no timestamps provided.
            token_time_ranges: Optional per-token (start_ts, end_ts) overrides.
                Takes precedence over start_ts/end_ts.

        Returns:
            List of TokenPriceHistory objects
        """
        results: list[TokenPriceHistory] = []

        if token_time_ranges:
            logger.info(
                f"Fetching price history for {len(token_ids)} tokens "
                f"with per-token time ranges"
            )
        else:
            logger.info(f"Fetching price history for {len(token_ids)} tokens")

        for i, token_id in enumerate(token_ids, 1):
            logger.info(f"Fetching token {i}/{len(token_ids)}: {token_id}")

            # Determine time range for this token
            token_start_ts = start_ts
            token_end_ts = end_ts

            if token_time_ranges and token_id in token_time_ranges:
                token_start_ts, token_end_ts = token_time_ranges[token_id]
                logger.debug(
                    f"Using per-token range for {token_id}: "
                    f"{token_start_ts} to {token_end_ts}"
                )

            try:
                history = self.fetch_token_history(
                    token_id=token_id,
                    start_ts=token_start_ts,
                    end_ts=token_end_ts,
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
    token_time_ranges: dict[str, tuple[int, int]] | None = None,
) -> list[TokenPriceHistory]:
    """Fetch price histories for market tokens.

    Main entry point for price history fetching.

    Args:
        token_ids: List of token IDs (called "market" in CLOB API)
        output_dir: Optional directory to save price histories
        start_ts: Default start timestamp (unix seconds) for tokens without
            an explicit range in token_time_ranges
        end_ts: Default end timestamp (unix seconds) for tokens without
            an explicit range in token_time_ranges
        interval: Price interval from: 1m, 1h, 6h, 1d, 1w, max.
                 Only used if no timestamps provided.
        token_time_ranges: Optional per-token (start_ts, end_ts) overrides.
            Takes precedence over start_ts/end_ts.

    Returns:
        List of TokenPriceHistory objects
    """
    with PriceHistoryFetcher() as fetcher:
        histories = fetcher.fetch_multiple_tokens(
            token_ids=token_ids,
            start_ts=start_ts,
            end_ts=end_ts,
            interval=interval,
            token_time_ranges=token_time_ranges,
        )

        # Save if output directory provided
        if output_dir:
            fetcher.save_histories(histories, output_dir)

        return histories
