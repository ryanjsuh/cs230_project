"""Tests for price history fetching functionality."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import Mock, patch

import httpx
import pytest

from polymarket_data.fetch_history import (
    PriceHistoryFetcher,
    PricePoint,
    TokenPriceHistory,
)


class TestPricePoint:
    """Test PricePoint model."""

    def test_price_point_parse(self) -> None:
        """Test parsing a price point."""
        data = {"t": 1698764400, "p": 0.65}

        point = PricePoint.model_validate(data)

        assert point.t == 1698764400
        assert point.p == 0.65

    def test_price_point_timestamp_property(self) -> None:
        """Test timestamp conversion to datetime."""
        point = PricePoint(t=1698764400, p=0.65)

        dt = point.timestamp

        assert isinstance(dt, datetime)
        assert dt.tzinfo == timezone.utc
        assert dt.timestamp() == 1698764400


class TestTokenPriceHistory:
    """Test TokenPriceHistory model."""

    def test_token_price_history_empty(self) -> None:
        """Test creating empty price history."""
        history = TokenPriceHistory(token_id="test123", history=[])

        assert history.token_id == "test123"
        assert len(history.history) == 0

    def test_token_price_history_with_points(self) -> None:
        """Test creating price history with data points."""
        points = [
            PricePoint(t=1698764400, p=0.60),
            PricePoint(t=1698768000, p=0.65),
        ]
        history = TokenPriceHistory(token_id="test123", history=points)

        assert history.token_id == "test123"
        assert len(history.history) == 2
        assert history.history[0].p == 0.60

    def test_to_dict(self) -> None:
        """Test converting price history to dict."""
        points = [PricePoint(t=1698764400, p=0.60)]
        history = TokenPriceHistory(token_id="test123", history=points)

        result = history.to_dict()

        assert result["token_id"] == "test123"
        assert len(result["history"]) == 1
        assert result["history"][0]["price"] == 0.60
        assert result["history"][0]["timestamp_unix"] == 1698764400
        assert "timestamp" in result["history"][0]


class TestPriceHistoryFetcher:
    """Test PriceHistoryFetcher functionality."""

    @pytest.fixture
    def mock_client(self) -> Mock:
        """Create a mock httpx client."""
        return Mock(spec=httpx.Client)

    def test_fetch_token_history_basic(self, mock_client: Mock) -> None:
        """Test fetching token history with explicit time range."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "history": [
                {"t": 1698764400, "p": 0.60},
                {"t": 1698768000, "p": 0.65},
            ]
        }
        mock_client.get.return_value = mock_response

        with patch("time.sleep"):
            fetcher = PriceHistoryFetcher(client=mock_client)
            result = fetcher.fetch_token_history(
                token_id="token123",
                start_ts=1698700000,
                end_ts=1698800000,
            )

        assert result.token_id == "token123"
        assert len(result.history) == 2
        assert result.history[0].p == 0.60

        # Verify request params
        call_kwargs = mock_client.get.call_args[1]
        assert call_kwargs["params"]["market"] == "token123"
        assert call_kwargs["params"]["startTs"] == 1698700000
        assert call_kwargs["params"]["endTs"] == 1698800000

    def test_fetch_token_history_with_timestamps(
        self, mock_client: Mock
    ) -> None:
        """Test fetching token history with start/end timestamps."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"history": [{"t": 1698764400, "p": 0.60}]}
        mock_client.get.return_value = mock_response

        with patch("time.sleep"):
            fetcher = PriceHistoryFetcher(client=mock_client)
            result = fetcher.fetch_token_history(
                token_id="token123",
                start_ts=1698700000,
                end_ts=1698800000,
            )

        assert result.token_id == "token123"
        assert len(result.history) == 1

        # Verify request params
        call_kwargs = mock_client.get.call_args[1]
        assert call_kwargs["params"]["startTs"] == 1698700000
        assert call_kwargs["params"]["endTs"] == 1698800000
        assert "interval" not in call_kwargs["params"]

    def test_fetch_token_history_missing_end_ts(self, mock_client: Mock) -> None:
        """Test that providing only start_ts raises ValueError."""
        fetcher = PriceHistoryFetcher(client=mock_client)

        with pytest.raises(
            ValueError, match="start_ts and end_ts must be provided together"
        ):
            fetcher.fetch_token_history(token_id="token123", start_ts=1698700000)

    def test_fetch_token_history_404(self, mock_client: Mock) -> None:
        """Test handling of 404 response (no data for token)."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_client.get.return_value = mock_response

        with patch("time.sleep"):
            fetcher = PriceHistoryFetcher(client=mock_client)
            result = fetcher.fetch_token_history(
                token_id="nonexistent",
                start_ts=1698700000,
                end_ts=1698800000,
            )

        # Should return empty history instead of raising
        assert result.token_id == "nonexistent"
        assert len(result.history) == 0

    def test_fetch_token_history_bare_list_response(
        self, mock_client: Mock
    ) -> None:
        """Test handling of bare list response (alternative format)."""
        mock_response = Mock()
        mock_response.status_code = 200
        # Return bare list instead of dict with "history"
        mock_response.json.return_value = [
            {"t": 1698764400, "p": 0.60},
            {"t": 1698768000, "p": 0.65},
        ]
        mock_client.get.return_value = mock_response

        with patch("time.sleep"):
            fetcher = PriceHistoryFetcher(client=mock_client)
            result = fetcher.fetch_token_history(
                token_id="token123",
                start_ts=1698700000,
                end_ts=1698800000,
            )

        # Should still parse correctly
        assert result.token_id == "token123"
        assert len(result.history) == 2

    def test_fetch_multiple_tokens(self, mock_client: Mock) -> None:
        """Test fetching history for multiple tokens."""

        def mock_response_json() -> dict[str, list[dict[str, float]]]:
            # Return different data based on which token is being fetched
            token_id = mock_client.get.call_args[1]["params"]["market"]
            return {
                "history": [
                    {"t": 1698764400, "p": 0.60 if token_id == "token1" else 0.70}
                ]
            }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = mock_response_json
        mock_client.get.return_value = mock_response

        with patch("time.sleep"):
            fetcher = PriceHistoryFetcher(client=mock_client)
            results = fetcher.fetch_multiple_tokens(
                token_ids=["token1", "token2"],
                start_ts=1698700000,
                end_ts=1698800000,
            )

        assert len(results) == 2
        assert results[0].token_id == "token1"
        assert results[1].token_id == "token2"
        assert results[0].history[0].p == 0.60
        assert results[1].history[0].p == 0.70

    def test_fetch_multiple_tokens_with_per_token_ranges(
        self, mock_client: Mock
    ) -> None:
        """Ensure per-token ranges override the shared range for each request."""
        captured_params: list[dict[str, Any]] = []

        def mock_get(url: str, params: dict[str, Any]) -> Mock:
            captured_params.append(params)
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"history": []}
            return mock_response

        mock_client.get.side_effect = mock_get

        token_ranges = {"token1": (10, 20), "token2": (30, 40)}

        with patch("time.sleep"):
            fetcher = PriceHistoryFetcher(client=mock_client)
            fetcher.fetch_multiple_tokens(
                token_ids=["token1", "token2"],
                start_ts=1,
                end_ts=2,
                token_time_ranges=token_ranges,
            )

        assert len(captured_params) == 2
        assert captured_params[0]["startTs"] == 10
        assert captured_params[0]["endTs"] == 20
        assert captured_params[1]["startTs"] == 30
        assert captured_params[1]["endTs"] == 40

    def test_fetch_token_history_splits_long_range(self, mock_client: Mock) -> None:
        """Verify that long ranges are automatically split into smaller chunks."""
        spans: list[tuple[int, int]] = []

        def mock_get(url: str, params: dict[str, Any]) -> Mock:
            spans.append((params["startTs"], params["endTs"]))
            duration = params["endTs"] - params["startTs"]

            if duration > 5:
                request = httpx.Request("GET", url)
                response = httpx.Response(
                    400,
                    request=request,
                    content=b"startTs and endTs interval is too long",
                )
                raise httpx.HTTPStatusError("Bad Request", request=request, response=response)

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "history": [{"t": params["startTs"], "p": 0.5}]
            }
            return mock_response

        mock_client.get.side_effect = mock_get

        with patch("time.sleep"):
            fetcher = PriceHistoryFetcher(
                client=mock_client,
                max_chunk_seconds=10,
                min_chunk_seconds=2,
            )
            result = fetcher.fetch_token_history(
                token_id="token123",
                start_ts=0,
                end_ts=20,
            )

        # Ensure multiple spans were attempted and the result contains merged data
        assert len(spans) > 2
        assert [point.t for point in result.history] == [0, 5, 10, 15]

    def test_fetch_multiple_tokens_with_errors(
        self, mock_client: Mock
    ) -> None:
        """Test that errors on individual tokens don't stop the batch."""

        def mock_get(*args: Any, **kwargs: Any) -> Mock:
            token_id = kwargs["params"]["market"]
            if token_id == "bad_token":
                raise Exception("Network error")
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"history": [{"t": 1698764400, "p": 0.60}]}
            return mock_response

        mock_client.get.side_effect = mock_get

        with patch("time.sleep"):
            fetcher = PriceHistoryFetcher(client=mock_client)
            results = fetcher.fetch_multiple_tokens(
                token_ids=["good_token", "bad_token", "another_good"],
                start_ts=1698700000,
                end_ts=1698800000,
            )

        # Should have 2 successful results, bad_token skipped
        assert len(results) == 2
        assert results[0].token_id == "good_token"
        assert results[1].token_id == "another_good"
