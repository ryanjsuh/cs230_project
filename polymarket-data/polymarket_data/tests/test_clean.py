"""Tests for data cleaning and processing."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from polymarket_data.clean import MarketDataProcessor
from polymarket_data.fetch_history import PricePoint, TokenPriceHistory
from polymarket_data.fetch_markets import Market


class TestMarketDataProcessor:
    """Test MarketDataProcessor functionality."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_markets_file(self, temp_dir: Path) -> Path:
        """Create sample markets JSON file."""
        markets = [
            {
                "id": "market1",
                "conditionId": "cond1",
                "question": "Will it rain?",
                "category": "Weather",
                "closed": True,
                "closedTime": "2024-11-01T00:00:00Z",
                "outcomes": ["Yes", "No"],
                "tokens": [{"tokenId": "token1"}, {"tokenId": "token2"}],
            }
        ]

        markets_file = temp_dir / "markets.json"
        with open(markets_file, "w") as f:
            json.dump(markets, f)

        return markets_file

    @pytest.fixture
    def sample_price_history_dir(self, temp_dir: Path) -> Path:
        """Create sample price history directory with files."""
        price_dir = temp_dir / "price_history"
        price_dir.mkdir()

        # Create history for token1
        history1 = {
            "token_id": "token1",
            "history": [
                {
                    "timestamp": "2024-10-01T00:00:00Z",
                    "timestamp_unix": 1727740800,
                    "price": 0.60,
                },
                {
                    "timestamp": "2024-10-01T01:00:00Z",
                    "timestamp_unix": 1727744400,
                    "price": 0.65,
                },
            ],
        }

        with open(price_dir / "token1.json", "w") as f:
            json.dump(history1, f)

        # Create history for token2
        history2 = {
            "token_id": "token2",
            "history": [
                {
                    "timestamp": "2024-10-01T00:00:00Z",
                    "timestamp_unix": 1727740800,
                    "price": 0.40,
                },
                {
                    "timestamp": "2024-10-01T01:00:00Z",
                    "timestamp_unix": 1727744400,
                    "price": 0.35,
                },
            ],
        }

        with open(price_dir / "token2.json", "w") as f:
            json.dump(history2, f)

        return price_dir

    def test_load_markets(
        self, sample_markets_file: Path, temp_dir: Path
    ) -> None:
        """Test loading markets from JSON file."""
        processor = MarketDataProcessor(
            markets_file=sample_markets_file,
            price_history_dir=temp_dir / "dummy",
        )

        markets = processor.load_markets()

        assert len(markets) == 1
        assert markets[0].question == "Will it rain?"

    def test_load_markets_skips_invalid(self, temp_dir: Path) -> None:
        """Test that invalid markets are skipped."""
        markets_file = temp_dir / "markets.json"

        # Mix of valid and invalid markets
        data = [
            {"question": "Valid market"},
            {"invalid": "no question field"},
            {"question": "Another valid market"},
        ]

        with open(markets_file, "w") as f:
            json.dump(data, f)

        processor = MarketDataProcessor(
            markets_file=markets_file,
            price_history_dir=temp_dir / "dummy",
        )

        markets = processor.load_markets()

        # Should load 2 valid markets, skip 1 invalid
        assert len(markets) == 2

    def test_load_token_history_success(
        self, sample_price_history_dir: Path, temp_dir: Path
    ) -> None:
        """Test loading token history successfully."""
        processor = MarketDataProcessor(
            markets_file=temp_dir / "dummy.json",
            price_history_dir=sample_price_history_dir,
        )

        history = processor.load_token_history("token1")

        assert history is not None
        assert history.token_id == "token1"
        assert len(history.history) == 2
        assert history.history[0].p == 0.60

    def test_load_token_history_missing_file(self, temp_dir: Path) -> None:
        """Test loading token history for non-existent file."""
        processor = MarketDataProcessor(
            markets_file=temp_dir / "dummy.json",
            price_history_dir=temp_dir / "empty",
        )

        history = processor.load_token_history("nonexistent")

        assert history is None

    def test_load_token_history_invalid_format(self, temp_dir: Path) -> None:
        """Test loading token history with invalid format."""
        price_dir = temp_dir / "price_history"
        price_dir.mkdir()

        # Invalid format - missing required keys
        with open(price_dir / "bad_token.json", "w") as f:
            json.dump({"invalid": "format"}, f)

        processor = MarketDataProcessor(
            markets_file=temp_dir / "dummy.json",
            price_history_dir=price_dir,
        )

        history = processor.load_token_history("bad_token")

        assert history is None

    def test_load_token_history_missing_price_keys(
        self, temp_dir: Path
    ) -> None:
        """Test handling of price points with missing keys."""
        price_dir = temp_dir / "price_history"
        price_dir.mkdir()

        # History with some invalid points
        data = {
            "token_id": "token1",
            "history": [
                {"timestamp_unix": 1727740800, "price": 0.60},  # Valid
                {"timestamp_unix": 1727744400},  # Missing price
                {"price": 0.65},  # Missing timestamp_unix
            ],
        }

        with open(price_dir / "token1.json", "w") as f:
            json.dump(data, f)

        processor = MarketDataProcessor(
            markets_file=temp_dir / "dummy.json",
            price_history_dir=price_dir,
        )

        history = processor.load_token_history("token1")

        # Should load only the valid point
        assert history is not None
        assert len(history.history) == 1
        assert history.history[0].p == 0.60

    def test_process_token_data(
        self,
        sample_markets_file: Path,
        sample_price_history_dir: Path,
    ) -> None:
        """Test processing token data with resampling."""
        processor = MarketDataProcessor(
            markets_file=sample_markets_file,
            price_history_dir=sample_price_history_dir,
            resample_freq="1h",
        )

        markets = processor.load_markets()
        market = markets[0]

        df = processor.process_token_data(market, "token1", 0)

        assert df is not None
        assert "timestamp" in df.columns
        assert "price" in df.columns
        assert "market_id" in df.columns
        assert "outcome" in df.columns
        assert "hours_to_resolution" in df.columns

        # Check metadata
        assert df["market_id"].iloc[0] == "market1"
        assert df["outcome"].iloc[0] == "Yes"  # First token = first outcome

    def test_process_token_data_with_none_ids(
        self, temp_dir: Path, sample_price_history_dir: Path
    ) -> None:
        """Test processing token data with None market IDs."""
        # Create market with None IDs
        markets_file = temp_dir / "markets.json"
        data = [
            {
                "question": "Test market",
                "tokens": [{"tokenId": "token1"}],
                "outcomes": ["Yes"],
                "closed": True,
                "closedTime": "2024-11-01T00:00:00Z",
            }
        ]

        with open(markets_file, "w") as f:
            json.dump(data, f)

        processor = MarketDataProcessor(
            markets_file=markets_file,
            price_history_dir=sample_price_history_dir,
        )

        markets = processor.load_markets()
        market = markets[0]

        df = processor.process_token_data(market, "token1", 0)

        assert df is not None
        # Should default to "unknown" for None IDs
        assert df["market_id"].iloc[0] == "unknown"
        assert df["condition_id"].iloc[0] == "unknown"

    def test_process_market(
        self,
        sample_markets_file: Path,
        sample_price_history_dir: Path,
    ) -> None:
        """Test processing entire market with multiple tokens."""
        processor = MarketDataProcessor(
            markets_file=sample_markets_file,
            price_history_dir=sample_price_history_dir,
            resample_freq="1h",
        )

        markets = processor.load_markets()
        market = markets[0]

        df = processor.process_market(market)

        assert df is not None
        # Should have data for both tokens
        assert len(df["token_id"].unique()) == 2
        assert "token1" in df["token_id"].values
        assert "token2" in df["token_id"].values

    def test_save_parquet(self, temp_dir: Path) -> None:
        """Test saving DataFrame to Parquet file."""
        processor = MarketDataProcessor(
            markets_file=temp_dir / "dummy.json",
            price_history_dir=temp_dir / "dummy",
        )

        # Create sample DataFrame
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1h"),
            "price": [0.5 + i * 0.01 for i in range(10)],
            "market_id": ["market1"] * 10,
        })

        output_file = temp_dir / "output.parquet"
        processor.save_parquet(df, output_file)

        # Verify file exists and can be read
        assert output_file.exists()

        loaded_df = pd.read_parquet(output_file)
        assert len(loaded_df) == 10
        assert "timestamp" in loaded_df.columns
