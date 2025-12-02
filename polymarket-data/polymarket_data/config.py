"""
Configuration for Polymarket API endpoints and settings
"""

from datetime import timedelta
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Application settings for Polymarket data collection
class Settings(BaseSettings):

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

    # Rate limiting
    requests_per_minute: int = 100
    rate_limit_buffer: float = 0.8

    # Data collection settings
    lookback_days: int = 365
    market_status_filter: Literal["resolved", "active", "closed"] = "resolved"

    # Time series resampling
    resample_frequency: str = "15min"

    # File paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    # AWS S3 settings
    s3_bucket: str | None = Field(
        default=None,
        alias="POLYMARKET_S3_BUCKET",
        description="S3 bucket name for storing data (e.g., cs230-polymarket-data-1)",
    )
    s3_prefix: str = Field(
        default="polymarket",
        alias="POLYMARKET_S3_PREFIX",
        description="Prefix/folder within the S3 bucket",
    )
    s3_region: str = Field(
        default="us-east-1",
        alias="AWS_DEFAULT_REGION",
        description="AWS region for S3 bucket",
    )

    # Pagination
    default_page_size: int = 100
    max_page_size: int = 100

    # Timeouts
    request_timeout: int = 30

    # Get full Gamma markets URL
    @property
    def gamma_markets_url(self) -> str:
        return f"{self.gamma_base_url}{self.gamma_markets_endpoint}"

    # Get full CLOB price history URL
    @property
    def clob_price_history_url(self) -> str:
        return f"{self.clob_base_url}{self.clob_price_history_endpoint}"

    # Calculate delay between requests in secs
    @property
    def rate_limit_delay(self) -> float:
        effective_rpm = self.requests_per_minute * self.rate_limit_buffer
        return 60.0 / effective_rpm

    # Get lookback period as timedelta
    @property
    def lookback_timedelta(self) -> timedelta:
        return timedelta(days=self.lookback_days)

    # Check if S3 storage is configured
    @property
    def use_s3(self) -> bool:
        return self.s3_bucket is not None and len(self.s3_bucket) > 0

    # Get S3 URI for raw data directory
    @property
    def s3_raw_data_uri(self) -> str:
        if not self.s3_bucket:
            raise ValueError("S3 bucket not configured")
        return f"s3://{self.s3_bucket}/{self.s3_prefix}/raw"

    # Get S3 URI for processed data directory
    @property
    def s3_processed_data_uri(self) -> str:
        if not self.s3_bucket:
            raise ValueError("S3 bucket not configured")
        return f"s3://{self.s3_bucket}/{self.s3_prefix}/processed"

    # Get S3 key for markets JSON file
    def get_s3_markets_key(self) -> str:
        return f"{self.s3_prefix}/raw/markets.json"

    # Get S3 key for a token's price history JSON file
    def get_s3_price_history_key(self, token_id: str) -> str:
        return f"{self.s3_prefix}/raw/price_history/{token_id}.json"

    # Get S3 key for processed parquet file
    def get_s3_processed_key(self, filename: str = "polymarket_data.parquet") -> str:
        return f"{self.s3_prefix}/processed/{filename}"


settings = Settings()
