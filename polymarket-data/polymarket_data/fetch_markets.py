"""
Fetch market metadata from Polymarket Gamma API
"""

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


# Polymarket market metadata model
# https://docs.polymarket.com/developers/gamma-markets-api/get-markets
class Market(BaseModel):

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

    # Parse JSON string lists from Gamma API (Handles both JSON-encoded strings and actual arrays)
    @field_validator("outcomes", "outcome_prices", mode="before")
    @classmethod
    def parse_json_list(cls, v: Any) -> list[str]:
        if isinstance(v, str):
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
            return v
        else:
            logger.warning(f"Unexpected type for list field: {type(v)}")
            return []

    # Parse clobTokenIds from Gamma API (Handles both JSON-encoded strings and actual arrays)
    @field_validator("clob_token_ids", mode="before")
    @classmethod
    def parse_clob_token_ids(cls, v: Any) -> list[str] | None:
        if v is None:
            return None

        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
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
            return [str(item) for item in v]
        else:
            logger.warning(f"Unexpected type for clobTokenIds: {type(v)}")
            return None

    # Check if market was closed/resolved within the lookback period
    def is_resolved_within_lookback(self, lookback_days: int) -> bool:
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

    # Extract token IDs for price history fetching (primary: clobTokenIds field, fallback: tokens field)
    def get_token_ids(self) -> list[str]:
        ids: list[str] = []

        if self.clob_token_ids:
            ids.extend(self.clob_token_ids)

        if not ids and self.tokens:
            for token in self.tokens:
                tid = token.get("token_id") or token.get("tokenId") or token.get("id")
                if tid:
                    ids.append(str(tid))

        return ids

    # Parse an ISO timestamp string into a timezone-aware UTC datetime
    def _parse_iso_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None

        try:
            normalized = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
        except (ValueError, AttributeError) as e:
            logger.debug(f"Failed to parse timestamp '{value}': {e}")
            return None

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(timezone.utc)

    # Get start timestamp (unix seconds) for price history fetching
    # Uses gameStartTime if available, otherwise falls back to createdAt
    def get_start_ts(self) -> int | None:
        start_dt = self._parse_iso_datetime(self.game_start_time)
        if start_dt is None:
            start_dt = self._parse_iso_datetime(self.created_at)

        if start_dt is None:
            return None

        return int(start_dt.timestamp())

    # Get end timestamp (unix seconds) for price history fetching
    # Uses closedTime if available, otherwise falls back to endDate
    def get_end_ts(self) -> int | None:
        end_dt = self._parse_iso_datetime(self.closed_time)
        if end_dt is None:
            end_dt = self._parse_iso_datetime(self.end_date)

        if end_dt is None:
            return None

        return int(end_dt.timestamp())

    # Get start and end timestamps as a tuple for price history fetching
    def get_time_range(self) -> tuple[int, int] | None:
        start_ts = self.get_start_ts()
        end_ts = self.get_end_ts()

        if start_ts is None or end_ts is None:
            return None

        if end_ts <= start_ts:
            logger.warning(
                f"Market {self.id}: end_ts ({end_ts}) <= start_ts ({start_ts}), "
                "adjusting end_ts to be 1 second after start_ts"
            )
            end_ts = start_ts + 1

        return (start_ts, end_ts)


# Fetches markets from Polymarket Gamma API with pagination
# Uses limit/offset pagination and synchronous requests
class MarketFetcher:

    # Initialize market fetcher
    def __init__(self, client: httpx.Client | None = None) -> None:
        self.client = client or httpx.Client(timeout=settings.request_timeout)
        self._owned_client = client is None

    # Context manager entry
    def __enter__(self) -> "MarketFetcher":
        return self

    # Context manager exit
    def __exit__(self, *args: Any) -> None:
        if self._owned_client:
            self.client.close()

    # Fetch a single page of markets from Gamma API
    # Gamma returns a JSON array of markets (not {"data": [...]})
    def fetch_page(
        self,
        offset: int = 0,
        limit: int | None = None,
        closed: bool | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "offset": offset,
            "limit": limit or settings.default_page_size,
        }

        if closed is not None:
            params["closed"] = str(closed).lower()

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

        time.sleep(settings.rate_limit_delay)

        data = response.json()
        if not isinstance(data, list):
            logger.warning(
                f"Expected list response from Gamma, got {type(data)}. "
                "API format may have changed."
            )
            return []

        return data

    # Fetch all markets with pagination
    def fetch_all_markets(
        self,
        closed: bool | None = None,
        max_markets: int | None = None,
        **kwargs: Any,
    ) -> list[Market]:
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

            if max_markets and len(markets) >= max_markets:
                markets = markets[:max_markets]
                logger.info(f"Reached max_markets limit: {max_markets}")
                break

            if len(page_data) < page_size:
                logger.info("Reached end of results (partial page)")
                break

            offset += page_size

        logger.info(f"Finished fetching {len(markets)} total markets")
        return markets

    # Filter markets closed within the lookback period
    def filter_by_closed_date(
        self, markets: list[Market], lookback_days: int
    ) -> list[Market]:
        filtered = [
            m for m in markets if m.is_resolved_within_lookback(lookback_days)
        ]
        logger.info(
            f"Filtered {len(filtered)}/{len(markets)} markets closed within "
            f"{lookback_days} days"
        )
        return filtered

    # Save markets to JSON file as a JSON array
    def save_markets(self, markets: list[Market], output_path: Path | str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [m.model_dump(by_alias=True) for m in markets]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(markets)} markets to {output_path}")


# Main entry point for market fetching
def fetch_resolved_markets(
    lookback_days: int | None = None,
    output_path: Path | str | None = None,
    max_markets: int | None = None,
) -> list[Market]:
    if lookback_days is None:
        lookback_days = settings.lookback_days

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    end_date_min = cutoff_date.strftime("%Y-%m-%d")

    logger.info(f"Fetching markets with end_date >= {end_date_min}")

    with MarketFetcher() as fetcher:
        all_markets = fetcher.fetch_all_markets(
            closed=True,
            max_markets=max_markets,
            end_date_min=end_date_min,
        )

        filtered_markets = fetcher.filter_by_closed_date(all_markets, lookback_days)

        if output_path:
            fetcher.save_markets(filtered_markets, output_path)

        return filtered_markets
