"""Tests for market fetching functionality."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import httpx
import pytest

from polymarket_data.fetch_markets import Market, MarketFetcher


class TestMarket:
    """Test Market model and methods."""

    def test_market_parse_with_all_fields(self) -> None:
        """Test parsing a market with all fields present."""
        data = {
            "id": "123",
            "conditionId": "abc",
            "slug": "test-market",
            "question": "Will this test pass?",
            "endDate": "2024-12-31T00:00:00Z",
            "closed": True,
            "closedTime": "2024-11-01T00:00:00Z",
            "category": "Technology",
            "tags": ["test", "python"],
            "tokens": [{"tokenId": "token1"}, {"tokenId": "token2"}],
            "outcomes": ["Yes", "No"],
            "outcomePrices": ["0.6", "0.4"],
        }

        market = Market.model_validate(data)

        assert market.id == "123"
        assert market.condition_id == "abc"
        assert market.slug == "test-market"
        assert market.question == "Will this test pass?"
        assert market.closed is True
        assert market.category == "Technology"
        assert len(market.tokens) == 2

    def test_market_parse_with_minimal_fields(self) -> None:
        """Test parsing a market with only required fields."""
        data = {"question": "Minimal market?"}

        market = Market.model_validate(data)

        assert market.question == "Minimal market?"
        assert market.id is None
        assert market.condition_id is None
        assert market.closed is False

    def test_is_resolved_within_lookback_true(self) -> None:
        """Test that recent closed markets are within lookback."""
        # Market closed 10 days ago
        closed_time = datetime.now(timezone.utc) - timedelta(days=10)
        data = {
            "question": "Recent market?",
            "closed": True,
            "closedTime": closed_time.isoformat(),
        }

        market = Market.model_validate(data)
        assert market.is_resolved_within_lookback(lookback_days=30) is True

    def test_is_resolved_within_lookback_false(self) -> None:
        """Test that old closed markets are outside lookback."""
        # Market closed 400 days ago
        closed_time = datetime.now(timezone.utc) - timedelta(days=400)
        data = {
            "question": "Old market?",
            "closed": True,
            "closedTime": closed_time.isoformat(),
        }

        market = Market.model_validate(data)
        assert market.is_resolved_within_lookback(lookback_days=365) is False

    def test_is_resolved_within_lookback_not_closed(self) -> None:
        """Test that non-closed markets return False."""
        data = {"question": "Active market?", "closed": False}

        market = Market.model_validate(data)
        assert market.is_resolved_within_lookback(lookback_days=365) is False

    def test_get_token_ids_with_tokenId(self) -> None:
        """Test extracting token IDs with camelCase tokenId."""
        data = {
            "question": "Market with tokens?",
            "tokens": [
                {"tokenId": "token1", "outcome": "Yes"},
                {"tokenId": "token2", "outcome": "No"},
            ],
        }

        market = Market.model_validate(data)
        token_ids = market.get_token_ids()

        assert len(token_ids) == 2
        assert "token1" in token_ids
        assert "token2" in token_ids

    def test_get_token_ids_with_token_id(self) -> None:
        """Test extracting token IDs with snake_case token_id."""
        data = {
            "question": "Market with tokens?",
            "tokens": [
                {"token_id": "token1"},
                {"token_id": "token2"},
            ],
        }

        market = Market.model_validate(data)
        token_ids = market.get_token_ids()

        assert len(token_ids) == 2
        assert "token1" in token_ids
        assert "token2" in token_ids

    def test_get_token_ids_with_id_fallback(self) -> None:
        """Test extracting token IDs falls back to generic 'id'."""
        data = {
            "question": "Market with tokens?",
            "tokens": [
                {"id": "token1"},
                {"id": "token2"},
            ],
        }

        market = Market.model_validate(data)
        token_ids = market.get_token_ids()

        assert len(token_ids) == 2
        assert "token1" in token_ids
        assert "token2" in token_ids

    def test_get_token_ids_empty(self) -> None:
        """Test extracting token IDs from empty tokens list."""
        data = {"question": "Market without tokens?", "tokens": []}

        market = Market.model_validate(data)
        token_ids = market.get_token_ids()

        assert len(token_ids) == 0


class TestMarketFetcher:
    """Test MarketFetcher functionality."""

    @pytest.fixture
    def mock_client(self) -> Mock:
        """Create a mock httpx client."""
        return Mock(spec=httpx.Client)

    def test_fetch_page_success(self, mock_client: Mock) -> None:
        """Test successful page fetch."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = [
            {"question": "Market 1"},
            {"question": "Market 2"},
        ]
        mock_client.get.return_value = mock_response

        with patch("time.sleep"):  # Skip rate limit delay in tests
            fetcher = MarketFetcher(client=mock_client)
            result = fetcher.fetch_page(offset=0, limit=2)

        assert len(result) == 2
        assert result[0]["question"] == "Market 1"
        mock_response.raise_for_status.assert_called_once()

    def test_fetch_page_with_closed_filter(self, mock_client: Mock) -> None:
        """Test page fetch with closed filter."""
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_client.get.return_value = mock_response

        with patch("time.sleep"):
            fetcher = MarketFetcher(client=mock_client)
            fetcher.fetch_page(offset=0, closed=True)

        # Check that closed param was passed correctly
        call_kwargs = mock_client.get.call_args[1]
        assert call_kwargs["params"]["closed"] == "true"

    def test_fetch_all_markets_single_page(self, mock_client: Mock) -> None:
        """Test fetching all markets when there's only one page."""
        closed_time = datetime.now(timezone.utc).isoformat()
        mock_response = Mock()
        mock_response.json.return_value = [
            {"question": "Market 1", "closed": True, "closedTime": closed_time},
            {"question": "Market 2", "closed": True, "closedTime": closed_time},
        ]
        mock_client.get.return_value = mock_response

        with patch("time.sleep"):
            fetcher = MarketFetcher(client=mock_client)
            markets = fetcher.fetch_all_markets(closed=True)

        assert len(markets) == 2
        assert markets[0].question == "Market 1"

    def test_fetch_all_markets_pagination(self, mock_client: Mock) -> None:
        """Test fetching all markets across multiple pages."""
        closed_time = datetime.now(timezone.utc).isoformat()

        # First page returns 100 markets, second page returns 50
        def mock_response_json() -> list[dict[str, str]]:
            if mock_client.get.call_count == 1:
                return [
                    {"question": f"Market {i}", "closed": True, "closedTime": closed_time}
                    for i in range(100)
                ]
            else:
                return [
                    {"question": f"Market {i}", "closed": True, "closedTime": closed_time}
                    for i in range(100, 150)
                ]

        mock_response = Mock()
        mock_response.json.side_effect = mock_response_json
        mock_client.get.return_value = mock_response

        with patch("time.sleep"):
            with patch("polymarket_data.fetch_markets.settings") as mock_settings:
                mock_settings.default_page_size = 100
                mock_settings.gamma_markets_url = "http://test.com"
                mock_settings.rate_limit_delay = 0

                fetcher = MarketFetcher(client=mock_client)
                markets = fetcher.fetch_all_markets(closed=True)

        # Should stop after second page (50 < 100 page size)
        assert len(markets) == 150

    def test_filter_by_closed_date(self, mock_client: Mock) -> None:
        """Test filtering markets by closed date."""
        recent_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        old_time = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()

        markets = [
            Market(
                question="Recent market",
                closed=True,
                closedTime=recent_time,
            ),
            Market(
                question="Old market",
                closed=True,
                closedTime=old_time,
            ),
            Market(question="Active market", closed=False),
        ]

        fetcher = MarketFetcher(client=mock_client)
        filtered = fetcher.filter_by_closed_date(markets, lookback_days=365)

        # Only the recent market should pass
        assert len(filtered) == 1
        assert filtered[0].question == "Recent market"
