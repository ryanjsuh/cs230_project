"""
Fetch price history from Polymarket CLOB API
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from polymarket_data.config import settings

logger = logging.getLogger(__name__)

# Maximum chunk size for API requests (15 days in secs)
MAX_CHUNK_SECONDS = 60 * 60 * 24 * 15
MIN_CHUNK_SECONDS = 60 * 60


# Single price observation for a token
class PricePoint(BaseModel):

    t: int = Field(..., description="Unix timestamp in seconds")
    p: float = Field(..., description="Price (0.0 to 1.0)")

    # Convert unix timestamp to datetime
    @property
    def timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.t, tz=timezone.utc)


# Price history for a single token
class TokenPriceHistory(BaseModel):

    token_id: str = Field(..., description="Token ID")
    history: list[PricePoint] = Field(
        default_factory=list, description="List of price points"
    )

    # Convert to dictionary with readable timestamps
    def to_dict(self) -> dict[str, Any]:
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


# Fetches price history from Polymarket CLOB API
# Interval is mutually exclusive with startTs/endTs
# Valid intervals: 1m, 1h, 6h, 1d, 1w, max
class PriceHistoryFetcher:

    # Initialize price history fetcher
    def __init__(self, client: httpx.Client | None = None) -> None:
        self.client = client or httpx.Client(timeout=settings.request_timeout)
        self._owned_client = client is None

    # Context manager entry
    def __enter__(self) -> "PriceHistoryFetcher":
        return self

    # Context manager exit
    def __exit__(self, *args: Any) -> None:
        if self._owned_client:
            self.client.close()

    # Fetch price history for a single token
    def fetch_token_history(
        self,
        token_id: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
        interval: str | None = None,
    ) -> TokenPriceHistory:
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

        if start_ts is not None and end_ts is not None:
            if end_ts <= start_ts:
                raise ValueError("end_ts must be greater than start_ts")
            return self._fetch_with_timestamps_chunked(token_id, start_ts, end_ts)

        if start_ts is not None or end_ts is not None:
            raise ValueError("Both start_ts and end_ts must be provided together")

        logger.debug(f"Fetching all available history for token {token_id}")
        return self._fetch_with_interval(token_id, "max")

    # Fetch history using interval parameter
    def _fetch_with_interval(
        self, token_id: str, interval: str
    ) -> TokenPriceHistory:
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

    # Fetch history using timestamps with auto chunking for long ranges
    def _fetch_with_timestamps_chunked(
        self, token_id: str, start_ts: int, end_ts: int
    ) -> TokenPriceHistory:
        duration = end_ts - start_ts

        if duration <= MAX_CHUNK_SECONDS:
            try:
                points = self._fetch_chunk(token_id, start_ts, end_ts)
                return TokenPriceHistory(token_id=token_id, history=points)
            except httpx.HTTPStatusError as e:
                if self._is_interval_too_long_error(e):
                    logger.debug(
                        f"API rejected range {start_ts}-{end_ts} for token {token_id}, "
                        "splitting into smaller chunks"
                    )
                    return self._fetch_chunked_recursive(token_id, start_ts, end_ts)
                raise

        logger.debug(
            f"Range {start_ts}-{end_ts} ({duration} seconds) exceeds max chunk size "
            f"({MAX_CHUNK_SECONDS} seconds), splitting into chunks"
        )
        return self._fetch_chunked_recursive(token_id, start_ts, end_ts)

    # Recursively fetch history by splitting into smaller chunks
    def _fetch_chunked_recursive(
        self, token_id: str, start_ts: int, end_ts: int
    ) -> TokenPriceHistory:
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

        unique_points = self._deduplicate_points(all_points)
        return TokenPriceHistory(token_id=token_id, history=unique_points)

    # Fetch a single chunk of price history
    def _fetch_chunk(
        self, token_id: str, start_ts: int, end_ts: int
    ) -> list[PricePoint]:
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

    # Parse API response into PricePoint list
    def _parse_history_response(
        self, token_id: str, data: Any
    ) -> list[PricePoint]:
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

    # Check if error indicates the time interval is too long
    def _is_interval_too_long_error(self, error: httpx.HTTPStatusError) -> bool:
        if error.response is None or error.response.status_code != 400:
            return False

        try:
            error_text = error.response.text.lower()
            return "interval is too long" in error_text or "startts and endts interval is too long" in error_text
        except Exception:
            return False

    # Remove duplicate timestamps and sort by timestamp
    def _deduplicate_points(self, points: list[PricePoint]) -> list[PricePoint]:
        unique: dict[int, PricePoint] = {}
        for point in points:
            unique[point.t] = point

        return [unique[ts] for ts in sorted(unique.keys())]

    # Fetch price history for multiple tokens
    def fetch_multiple_tokens(
        self,
        token_ids: list[str],
        start_ts: int | None = None,
        end_ts: int | None = None,
        interval: str | None = None,
        token_time_ranges: dict[str, tuple[int, int]] | None = None,
        output_dir: Path | str | None = None,
        resume: bool = True,
        use_s3: bool = False,
    ) -> list[TokenPriceHistory]:
        results: list[TokenPriceHistory] = []
        
        # Track which tokens already have files
        existing_files: set[str] = set()
        if resume:
            if use_s3:
                # Check S3 for existing files
                from polymarket_data.s3_utils import get_existing_token_ids_from_s3
                existing_files = get_existing_token_ids_from_s3()
                if existing_files:
                    logger.info(
                        f"Found {len(existing_files)} existing price history files in S3. "
                        "Skipping already-fetched tokens (resume mode)."
                    )
            elif output_dir:
                output_path = Path(output_dir)
                if output_path.exists():
                    # Check for both completed files and temp files
                    existing_files = {
                        f.stem for f in output_path.glob("*.json")
                        if not f.name.endswith(".tmp")
                    }
                    # Clean up
                    temp_files = list(output_path.glob("*.json.tmp"))
                    if temp_files:
                        logger.info(
                            f"Found {len(temp_files)} leftover temp files from previous run. "
                            "Cleaning up..."
                        )
                        for temp_file in temp_files:
                            try:
                                temp_file.unlink()
                            except Exception as e:
                                logger.warning(f"Failed to remove temp file {temp_file}: {e}")
                    
                    if existing_files:
                        logger.info(
                            f"Found {len(existing_files)} existing price history files. "
                            "Skipping already-fetched tokens (resume mode)."
                        )

        if token_time_ranges:
            logger.info(
                f"Fetching price history for {len(token_ids)} tokens "
                f"with per-token time ranges"
            )
        else:
            logger.info(f"Fetching price history for {len(token_ids)} tokens")

        skipped_count = 0
        saved_count = 0
        for i, token_id in enumerate(token_ids, 1):
            # Skip if file already exists
            if resume and token_id in existing_files:
                skipped_count += 1
                if skipped_count % 1000 == 0:
                    logger.info(f"Skipped {skipped_count} already-fetched tokens...")
                continue
            
            if i % 100 == 0:
                logger.info(
                    f"Progress: {i}/{len(token_ids)} tokens processed, "
                    f"{saved_count} saved, {skipped_count} skipped"
                )
            
            logger.info(f"Fetching token {i}/{len(token_ids)}: {token_id}")

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
                
                if use_s3:
                    # Upload directly to S3
                    try:
                        self._save_single_history_to_s3(history)
                        saved_count += 1
                    except Exception as save_error:
                        logger.error(
                            f"Failed to upload token {token_id} to S3: {save_error}"
                        )
                elif output_dir:
                    try:
                        self._save_single_history(history, output_dir)
                        saved_count += 1
                    except Exception as save_error:
                        logger.error(
                            f"Failed to save token {token_id} after fetching: {save_error}"
                        )
                else:
                    results.append(history)

            except Exception as e:
                logger.error(f"Failed to fetch token {token_id}: {e}")
                continue

        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} already-fetched tokens")
        logger.info(
            f"Successfully fetched {len(results)}/{len(token_ids)} token histories"
        )
        return results

    # Save a single token history to file for incremental saves
    def _save_single_history(
        self,
        history: TokenPriceHistory,
        output_dir: Path | str,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not history.history:
            logger.info(f"Skipping token {history.token_id} - no price data")
            return

        output_path = output_dir / f"{history.token_id}.json"

        try:
            temp_path = output_path.with_suffix(".json.tmp")
            with open(temp_path, "w") as f:
                json.dump(history.to_dict(), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename
            temp_path.rename(output_path)
            
            logger.info(
                f"Saved {len(history.history)} price points for "
                f"{history.token_id} to {output_path}"
            )
        except Exception as e:
            logger.error(
                f"Failed to save token {history.token_id} to {output_path}: {e}"
            )
            temp_path = output_path.with_suffix(".json.tmp")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise

    # Save a single token history directly to S3
    def _save_single_history_to_s3(self, history: TokenPriceHistory) -> None:
        if not history.history:
            logger.info(f"Skipping token {history.token_id} - no price data")
            return

        from polymarket_data.config import settings
        from polymarket_data.s3_utils import upload_json_to_s3

        s3_key = settings.get_s3_price_history_key(history.token_id)
        upload_json_to_s3(history.to_dict(), s3_key)
        
        logger.info(
            f"Uploaded {len(history.history)} price points for "
            f"{history.token_id} to S3"
        )

    # Save token price histories to individual JSON files
    def save_histories(
        self,
        histories: list[TokenPriceHistory],
        output_dir: Path | str,
    ) -> None:
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


# Fetch price histories for multiple tokens
def fetch_market_price_histories(
    token_ids: list[str],
    output_dir: Path | str | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
    interval: str | None = None,
    token_time_ranges: dict[str, tuple[int, int]] | None = None,
    resume: bool = True,
    use_s3: bool = False,
) -> list[TokenPriceHistory]:
    with PriceHistoryFetcher() as fetcher:
        histories = fetcher.fetch_multiple_tokens(
            token_ids=token_ids,
            start_ts=start_ts,
            end_ts=end_ts,
            interval=interval,
            token_time_ranges=token_time_ranges,
            output_dir=output_dir,
            resume=resume,
            use_s3=use_s3,
        )
        return histories
