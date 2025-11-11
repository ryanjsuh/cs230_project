"""Configuration for Polymarket API endpoints and settings."""

from datetime import timedelta
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for Polymarket data collection."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Gamma API endpoints (market metadata)
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    gamma_markets_endpoint: str = "/markets"

    # CLOB API endpoints (price history)
    clob_base_url: str = "https://clob.polymarket.com"
    clob_price_history_endpoint: str = "/prices-history"

    # Rate limiting (from Polymarket docs)
    # Standard tier: 100 requests per minute per IP
    requests_per_minute: int = 100
    rate_limit_buffer: float = 0.8  # Use 80% to stay safe

    # Data collection settings
    lookback_days: int = 365
    market_status_filter: Literal["resolved", "active", "closed"] = "resolved"

    # Time series resampling
    resample_frequency: str = "15min"  # Pandas frequency string

    # File paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    # Pagination
    default_page_size: int = 100
    max_page_size: int = 100

    # Timeouts
    request_timeout: int = 30  # seconds

    @property
    def gamma_markets_url(self) -> str:
        """Get full Gamma markets URL."""
        return f"{self.gamma_base_url}{self.gamma_markets_endpoint}"

    @property
    def clob_price_history_url(self) -> str:
        """Get full CLOB price history URL."""
        return f"{self.clob_base_url}{self.clob_price_history_endpoint}"

    @property
    def rate_limit_delay(self) -> float:
        """Calculate delay between requests in seconds."""
        effective_rpm = self.requests_per_minute * self.rate_limit_buffer
        return 60.0 / effective_rpm

    @property
    def lookback_timedelta(self) -> timedelta:
        """Get lookback period as timedelta."""
        return timedelta(days=self.lookback_days)


# Global settings instance
settings = Settings()
