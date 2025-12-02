# AWS S3 Setup Guide

This guide walks through setting up AWS S3 storage for the Polymarket data collection pipeline.

## Prerequisites

- AWS account with credits
- macOS with Homebrew installed

## Step 1: Create IAM User

1. Log in to [AWS Console](https://console.aws.amazon.com/)
2. Navigate to **IAM** → **Users** → **Create user**
3. Enter a username (e.g., `polymarket-access`)
4. Select **Programmatic access** (CLI)
5. Attach the `AmazonS3FullAccess` policy (or a more restrictive custom policy)
6. Complete user creation
7. **Save the Access Key ID and Secret Access Key** - you cannot retrieve the secret later!

## Step 2: Install and Configure AWS CLI

```bash
# Install AWS CLI via Homebrew
brew install awscli

# Configure with your credentials
aws configure
```

When prompted, enter:

- **AWS Access Key ID**: Your access key
- **AWS Secret Access Key**: Your secret key
- **Default region**: `us-east-1` (or your preferred region)
- **Default output format**: `json`

## Step 3: Verify CLI Works

```bash
# Check authentication
aws sts get-caller-identity

# List buckets (will be empty if none exist)
aws s3 ls
```

## Step 4: Create S3 Bucket

```bash
# Create bucket (name must be globally unique)
aws s3 mb s3://cs230-polymarket-data-1

# Optional: Enable versioning for rollback safety
aws s3api put-bucket-versioning \
  --bucket cs230-polymarket-data-1 \
  --versioning-configuration Status=Enabled

# Verify bucket was created
aws s3 ls
```

## Step 5: Set Environment Variable

Add to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
export POLYMARKET_S3_BUCKET=cs230-polymarket-data-1
```

Then reload:

```bash
source ~/.zshrc  # or ~/.bashrc
```

## Bucket Structure

The scripts will create the following structure in your bucket:

```
s3://cs230-polymarket-data-1/
└── polymarket/
    ├── raw/
    │   ├── markets.json           # Market metadata
    │   └── price_history/         # Per-token price history
    │       ├── {token_id_1}.json
    │       ├── {token_id_2}.json
    │       └── ...
    └── processed/
        └── polymarket_data.parquet  # Final cleaned dataset
```

## Usage

Once configured, run the scripts with the `--s3` flag:

```bash
# Step 1: Fetch markets → S3
python scripts/01_fetch_markets.py --s3 --all-markets

# Step 2: Fetch price histories → S3 (streams directly, no local storage)
python scripts/02_fetch_history.py --s3

# Step 3: Clean and combine → S3
python scripts/03_clean_all.py --s3
```

## Monitoring Storage Usage

```bash
# Check bucket size
aws s3 ls s3://cs230-polymarket-data-1 --recursive --summarize

# List price history files
aws s3 ls s3://cs230-polymarket-data-1/polymarket/raw/price_history/ | wc -l
```

## Cost Considerations

- **Storage**: ~$0.023/GB/month for S3 Standard
- **Requests**: ~$0.0004 per 1,000 PUT requests
- **Data transfer**: Free within same region, ~$0.09/GB outbound

For a ~300GB dataset:

- Storage: ~$7/month
- One-time upload: ~$0.50 (assuming ~1M files)

## Troubleshooting

### "S3 bucket not configured" error

Ensure the environment variable is set:

```bash
echo $POLYMARKET_S3_BUCKET
# Should print: cs230-polymarket-data-1
```

### "Access Denied" errors

Check IAM permissions:

```bash
aws iam get-user
aws iam list-attached-user-policies --user-name YOUR_USERNAME
```

### Slow uploads

The scripts upload each token's price history individually. This is intentional for:

1. Resumability (can restart without re-fetching)
2. Memory efficiency (no local disk usage)

For very large datasets, consider running on an EC2 instance in the same region as your bucket.
