# Polymarket Data Collection Pipeline

Fetch, process, and store Polymarket prediction market data.

## Quick Start

This pipeline streams data directly to AWS S3.

### Prerequisites

1. **AWS CLI configured** with S3 access (see [docs/aws_setup.md](docs/aws_setup.md))
2. **S3 bucket created** (e.g., `cs230-polymarket-data-1`)
3. **Environment variable set**:
   ```bash
   export POLYMARKET_S3_BUCKET=cs230-polymarket-data-1
   ```

### Installation

```bash
cd polymarket-data
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Step 1: Fetch all resolved markets → S3
python scripts/01_fetch_markets.py --s3 --all-markets

# Step 2: Fetch price histories → S3 (streams directly, resumable)
python scripts/02_fetch_history.py --s3

# Step 3: Clean and combine into Parquet → S3
python scripts/03_clean_all.py --s3
```

## Local Mode (Legacy)

If you prefer local storage (requires ~300GB+ disk space):

```bash
# Fetch markets
python scripts/01_fetch_markets.py --output data/raw/markets.json

# Fetch price histories
python scripts/02_fetch_history.py --output data/raw/price_history

# Clean and combine
python scripts/03_clean_all.py
```

## Pipeline Overview

| Script                | Input          | Output                    | Description                         |
| --------------------- | -------------- | ------------------------- | ----------------------------------- |
| `01_fetch_markets.py` | Polymarket API | `markets.json`            | Fetches resolved market metadata    |
| `02_fetch_history.py` | `markets.json` | `price_history/*.json`    | Fetches per-token price time series |
| `03_clean_all.py`     | Both above     | `polymarket_data.parquet` | Cleans, resamples, and combines     |

## Command Options

### 01_fetch_markets.py

```bash
--s3              # Upload directly to S3
--all-markets     # Fetch ALL resolved markets (no date filter)
--lookback-days N # Only markets resolved in last N days (default: 365)
--max-markets N   # Limit for testing
```

### 02_fetch_history.py

```bash
--s3              # Read markets from S3, upload histories to S3
--max-tokens N    # Limit for testing
--start-ts T      # Override start timestamp (unix seconds)
--end-ts T        # Override end timestamp (unix seconds)
```

### 03_clean_all.py

```bash
--s3              # Read from S3, write to S3
--resample-freq F # Resampling frequency (default: 15min)
```

## Output Schema

The final Parquet file contains:

| Column                | Type     | Description                                    |
| --------------------- | -------- | ---------------------------------------------- |
| `timestamp`           | datetime | Price observation time                         |
| `price`               | float    | Token price (0-1)                              |
| `market_id`           | string   | Unique market identifier                       |
| `token_id`            | string   | CLOB token identifier                          |
| `question`            | string   | Market question text                           |
| `category`            | string   | Market category                                |
| `outcome`             | string   | Outcome name (e.g., "Yes", "No")               |
| `won`                 | int      | Target variable: 1 if outcome won, 0 otherwise |
| `hours_to_resolution` | float    | Hours until market resolved                    |
| `final_price`         | float    | Price at resolution                            |

## Configuration

Environment variables (or `.env` file):

| Variable               | Default      | Description                           |
| ---------------------- | ------------ | ------------------------------------- |
| `POLYMARKET_S3_BUCKET` | -            | S3 bucket name (required for S3 mode) |
| `POLYMARKET_S3_PREFIX` | `polymarket` | Prefix/folder within bucket           |
| `AWS_DEFAULT_REGION`   | `us-east-1`  | AWS region                            |

## Development

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
mypy .
```

## License

For academic use only (CS 230 project).
