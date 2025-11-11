"""Fetch market metadata from Polymarket Gamma API."""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from polymarket_data.config import settings

logger = logging.getLogger(__name__)


class Market(BaseModel):
    """Polymarket market metadata model.

    Field names match Gamma API response format (camelCase).
    See: https://docs.polymarket.com/developers/gamma-markets-api/get-markets
    """

    # Core identifiers - make optional for robustness
    id: str | None = Field(None, description="Market ID")
    condition_id: str | None = Field(
        None, alias="conditionId", description="Condition ID"
    )
    slug: str | None = Field(None, description="URL slug")
    question: str = Field(..., description="Market question")

    # Dates and status
    end_date: str | None = Field(
        None, alias="endDate", description="Market end date ISO"
    )
    game_start_time: str | None = Field(
        None, alias="gameStartTime", description="Event start time"
    )
    closed: bool = Field(False, description="Whether market is closed")
    closed_time: str | None = Field(
        None, alias="closedTime", description="When market closed (ISO)"
    )
    created_at: str | None = Field(
        None, alias="createdAt", description="Creation time"
    )

    # Market metadata
    category: str | None = Field(None, description="Market category")
    tags: list[str] = Field(default_factory=list, description="Market tags")
    description: str | None = Field(None, description="Market description")

    # Trading info
    tokens: list[dict[str, Any]] = Field(
        default_factory=list, description="Token information"
    )
    clob_token_ids: list[str] | None = Field(
        None,
        alias="clobTokenIds",
        description="CLOB token IDs (JSON string array from API)",
    )
    volume: str | None = Field(None, description="Trading volume")
    liquidity: str | None = Field(None, description="Market liquidity")

    # Outcomes
    outcomes: list[str] = Field(
        default_factory=list, description="Possible outcomes"
    )
    outcome_prices: list[str] = Field(
        default_factory=list,
        alias="outcomePrices",
        description="Current outcome prices",
    )

    # Additional fields that may be present
    rewards: dict[str, Any] | None = Field(None, description="Reward information")
    archive: bool | None = Field(None, description="Whether archived")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("outcomes", "outcome_prices", mode="before")
    @classmethod
    def parse_json_list(cls, v: Any) -> list[str]:
        """Parse JSON string lists from Gamma API.

        The Gamma API sometimes returns outcomes and outcomePrices as
        JSON-encoded strings instead of actual arrays. This validator
        handles both formats.

        Args:
            v: Either a list or a JSON string representing a list

        Returns:
            Parsed list of strings
        """
        if isinstance(v, str):
            # Parse JSON string to list
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
                else:
                    logger.warning(f"Expected list in JSON string, got {type(parsed)}")
                    return []
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON string: {e}")
                return []
        elif isinstance(v, list):
            # Already a list, return as-is
            return v
        else:
            # Unexpected type
            logger.warning(f"Unexpected type for list field: {type(v)}")
            return []

    @field_validator("clob_token_ids", mode="before")
    @classmethod
    def parse_clob_token_ids(cls, v: Any) -> list[str] | None:
        """Parse clobTokenIds from Gamma API.

        The Gamma API returns clobTokenIds as a JSON-encoded string array.
        This validator handles both string and list formats.

        Args:
            v: Either a list, a JSON string representing a list, or None

        Returns:
            Parsed list of token ID strings, or None if not present/invalid
        """
        if v is None:
            return None

        if isinstance(v, str):
            # Parse JSON string to list
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    # Convert all items to strings
                    return [str(item) for item in parsed]
                else:
                    logger.warning(
                        f"Expected list in clobTokenIds JSON string, got {type(parsed)}"
                    )
                    return None
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse clobTokenIds JSON string: {e}")
                return None
        elif isinstance(v, list):
            # Already a list, convert to strings
            return [str(item) for item in v]
        else:
            # Unexpected type
            logger.warning(f"Unexpected type for clobTokenIds: {type(v)}")
            return None

    def is_resolved_within_lookback(self, lookback_days: int) -> bool:
        """Check if market was closed/resolved within the lookback period.

        Uses the lookback_days argument consistently (not settings).
        Gamma uses 'closed' and 'closedTime' for resolution tracking.

        Args:
            lookback_days: Number of days to look back from now

        Returns:
            True if market closed within lookback period
        """
        if not self.closed or not self.closed_time:
            return False

        try:
            closed_dt = datetime.fromisoformat(
                self.closed_time.replace("Z", "+00:00")
            )
            cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            return closed_dt >= cutoff
        except (ValueError, AttributeError) as e:
            logger.warning(
                f"Invalid closedTime for market {self.id}: "
                f"{self.closed_time} - {e}"
            )
            return False

    def get_token_ids(self) -> list[str]:
        """Extract token IDs for price history fetching.

        Primary source: clobTokenIds field (from Gamma API)
        Fallback: tokens field with multiple key formats

        Returns:
            List of token IDs
        """
        ids: list[str] = []

        # Primary: Use clobTokenIds if available (most reliable source)
        if self.clob_token_ids:
            ids.extend(self.clob_token_ids)

        # Fallback: Extract from tokens array (for backwards compatibility)
        if not ids and self.tokens:
            for token in self.tokens:
                # Try multiple possible key names
                tid = token.get("token_id") or token.get("tokenId") or token.get("id")
                if tid:
                    ids.append(str(tid))

        return ids


class MarketFetcher:
    """Fetches markets from Polymarket Gamma API with pagination.

    Uses limit/offset pagination as documented:
    https://docs.polymarket.com/developers/gamma-markets-api/fetch-markets-guide

    Uses synchronous requests since pagination is inherently sequential.
    """

    def __init__(self, client: httpx.Client | None = None) -> None:
        """Initialize market fetcher.

        Args:
            client: Optional httpx client. If None, creates a new one.
        """
        self.client = client or httpx.Client(timeout=settings.request_timeout)
        self._owned_client = client is None

    def __enter__(self) -> "MarketFetcher":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        if self._owned_client:
            self.client.close()

    def fetch_page(
        self,
        offset: int = 0,
        limit: int | None = None,
        closed: bool | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Fetch a single page of markets from Gamma API.

        Per docs, Gamma returns a JSON array of markets (not {"data": [...]}).
        https://docs.polymarket.com/developers/gamma-markets-api/get-markets

        Args:
            offset: Pagination offset
            limit: Number of markets per page
            closed: Filter by closed status (True for resolved markets)
            **kwargs: Additional query parameters (e.g., tag_id, category)

        Returns:
            List of market dictionaries

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        params: dict[str, Any] = {
            "offset": offset,
            "limit": limit or settings.default_page_size,
        }

        # Add closed filter if specified
        if closed is not None:
            params["closed"] = str(closed).lower()

        # Add any additional filters
        params.update(kwargs)

        logger.debug(
            f"Fetching markets: offset={offset}, limit={params['limit']}, "
            f"closed={closed}"
        )

        response = self.client.get(
            settings.gamma_markets_url,
            params=params,
        )
        response.raise_for_status()

        # Rate limiting - respect documented limits (Gamma: 125 requests / 10s)
        time.sleep(settings.rate_limit_delay)

        # Gamma returns a bare array, not {"data": [...]}
        data = response.json()
        if not isinstance(data, list):
            logger.warning(
                f"Expected list response from Gamma, got {type(data)}. "
                "API format may have changed."
            )
            return []

        return data

    def fetch_all_markets(
        self,
        closed: bool | None = None,
        max_markets: int | None = None,
        **kwargs: Any,
    ) -> list[Market]:
        """Fetch all markets with pagination.

        Args:
            closed: Filter by closed status (True for resolved/closed markets)
            max_markets: Maximum number of markets to fetch (for testing)
            **kwargs: Additional filters to pass to API

        Returns:
            List of Market objects
        """
        markets: list[Market] = []
        offset = 0
        page_size = settings.default_page_size

        logger.info(f"Starting market fetch: closed={closed}, filters={kwargs}")

        while True:
            page_data = self.fetch_page(
                offset=offset, limit=page_size, closed=closed, **kwargs
            )

            if not page_data:
                logger.info(f"No more markets found at offset {offset}")
                break

            # Parse markets
            for market_dict in page_data:
                try:
                    market = Market.model_validate(market_dict)
                    markets.append(market)
                except Exception as e:
                    logger.warning(
                        f"Failed to parse market "
                        f"{market_dict.get('id', 'unknown')}: {e}"
                    )
                    continue

            logger.info(f"Fetched {len(page_data)} markets (total: {len(markets)})")

            # Check if we've reached max_markets
            if max_markets and len(markets) >= max_markets:
                markets = markets[:max_markets]
                logger.info(f"Reached max_markets limit: {max_markets}")
                break

            # Check if we've reached the end (fewer results than page size)
            if len(page_data) < page_size:
                logger.info("Reached end of results (partial page)")
                break

            offset += page_size

        logger.info(f"Finished fetching {len(markets)} total markets")
        return markets

    def filter_by_closed_date(
        self, markets: list[Market], lookback_days: int
    ) -> list[Market]:
        """Filter markets closed within the lookback period.

        Args:
            markets: List of markets to filter
            lookback_days: Number of days to look back from now

        Returns:
            Filtered list of markets
        """
        filtered = [
            m for m in markets if m.is_resolved_within_lookback(lookback_days)
        ]
        logger.info(
            f"Filtered {len(filtered)}/{len(markets)} markets closed within "
            f"{lookback_days} days"
        )
        return filtered

    def save_markets(self, markets: list[Market], output_path: Path | str) -> None:
        """Save markets to JSON file.

        Saves as a JSON array to match Gamma API response format.

        Args:
            markets: List of markets to save
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dicts using model_dump with by_alias=True to preserve format
        data = [m.model_dump(by_alias=True) for m in markets]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(markets)} markets to {output_path}")


def fetch_resolved_markets(
    lookback_days: int | None = None,
    output_path: Path | str | None = None,
    max_markets: int | None = None,
) -> list[Market]:
    """Fetch and filter closed/resolved markets from Gamma API.

    Main entry point for market fetching.

    Args:
        lookback_days: Number of days to look back (defaults to settings value)
        output_path: Optional path to save markets JSON
        max_markets: Maximum number of markets to fetch (for testing/debugging)

    Returns:
        List of filtered Market objects
    """
    # Use settings as default, but allow override
    if lookback_days is None:
        lookback_days = settings.lookback_days

    # Calculate end_date_min to filter at API level (more efficient)
    # This avoids fetching 60k+ old markets dating back to 2020
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    end_date_min = cutoff_date.strftime("%Y-%m-%d")

    logger.info(f"Fetching markets with end_date >= {end_date_min}")

    with MarketFetcher() as fetcher:
        # Fetch closed markets with date filter
        all_markets = fetcher.fetch_all_markets(
            closed=True,
            max_markets=max_markets,
            end_date_min=end_date_min,  # API-level date filter
        )

        # Filter by closed date within lookback period (double-check)
        filtered_markets = fetcher.filter_by_closed_date(all_markets, lookback_days)

        # Save if output path provided
        if output_path:
            fetcher.save_markets(filtered_markets, output_path)

        return filtered_markets
