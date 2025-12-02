"""
S3 utility functions for uploading/downloading data directly to/from AWS S3
"""

import io
import json
import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError

from polymarket_data.config import settings

logger = logging.getLogger(__name__)

# Get a boto3 S3 client using default credentials
def get_s3_client():
    return boto3.client("s3", region_name=settings.s3_region)


# Upload JSON data directly to S3 without writing to local disk
def upload_json_to_s3(data: Any, key: str, bucket: str | None = None) -> str:
    bucket = bucket or settings.s3_bucket
    if not bucket:
        raise ValueError("S3 bucket not configured. Set POLYMARKET_S3_BUCKET env var.")
    
    s3_client = get_s3_client()
    
    # Serialize to JSON in memory
    json_bytes = json.dumps(data, indent=2).encode("utf-8")
    file_obj = io.BytesIO(json_bytes)
    
    logger.debug(f"Uploading {len(json_bytes)} bytes to s3://{bucket}/{key}")
    
    s3_client.upload_fileobj(
        file_obj,
        bucket,
        key,
        ExtraArgs={"ContentType": "application/json"},
    )
    
    s3_uri = f"s3://{bucket}/{key}"
    logger.info(f"Uploaded to {s3_uri}")
    return s3_uri


# Download and parse JSON data from S3 without writing to local disk
def download_json_from_s3(key: str, bucket: str | None = None) -> Any:
    bucket = bucket or settings.s3_bucket
    if not bucket:
        raise ValueError("S3 bucket not configured. Set POLYMARKET_S3_BUCKET env var.")
    
    s3_client = get_s3_client()
    
    logger.debug(f"Downloading from s3://{bucket}/{key}")
    
    file_obj = io.BytesIO()
    s3_client.download_fileobj(bucket, key, file_obj)
    file_obj.seek(0)
    
    data = json.load(file_obj)
    logger.debug(f"Downloaded and parsed JSON from s3://{bucket}/{key}")
    return data


# Upload raw bytes directly to S3
def upload_bytes_to_s3(
    data: bytes,
    key: str,
    bucket: str | None = None,
    content_type: str = "application/octet-stream",
) -> str:
    bucket = bucket or settings.s3_bucket
    if not bucket:
        raise ValueError("S3 bucket not configured. Set POLYMARKET_S3_BUCKET env var.")
    
    s3_client = get_s3_client()
    file_obj = io.BytesIO(data)
    
    logger.debug(f"Uploading {len(data)} bytes to s3://{bucket}/{key}")
    
    s3_client.upload_fileobj(
        file_obj,
        bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    
    s3_uri = f"s3://{bucket}/{key}"
    logger.info(f"Uploaded to {s3_uri}")
    return s3_uri


# Download raw bytes from S3
def download_bytes_from_s3(key: str, bucket: str | None = None) -> bytes:
    bucket = bucket or settings.s3_bucket
    if not bucket:
        raise ValueError("S3 bucket not configured. Set POLYMARKET_S3_BUCKET env var.")
    
    s3_client = get_s3_client()
    
    file_obj = io.BytesIO()
    s3_client.download_fileobj(bucket, key, file_obj)
    file_obj.seek(0)
    
    return file_obj.read()


# Check if an S3 object exists
def s3_object_exists(key: str, bucket: str | None = None) -> bool:
    bucket = bucket or settings.s3_bucket
    if not bucket:
        raise ValueError("S3 bucket not configured. Set POLYMARKET_S3_BUCKET env var.")
    
    s3_client = get_s3_client()
    
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


# List all object keys under a given S3 prefix
def list_s3_objects(prefix: str, bucket: str | None = None) -> list[str]:
    bucket = bucket or settings.s3_bucket
    if not bucket:
        raise ValueError("S3 bucket not configured. Set POLYMARKET_S3_BUCKET env var.")
    
    s3_client = get_s3_client()
    
    keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                keys.append(obj["Key"])
    
    return keys


# Get set of token IDs that already have price history files in S3
def get_existing_token_ids_from_s3(bucket: str | None = None) -> set[str]:
    bucket = bucket or settings.s3_bucket
    if not bucket:
        return set()
    
    prefix = f"{settings.s3_prefix}/raw/price_history/"
    keys = list_s3_objects(prefix, bucket)
    
    # Extract token IDs from keys like "polymarket/raw/price_history/{token_id}.json"
    token_ids = set()
    for key in keys:
        if key.endswith(".json"):
            # Get filename without extension
            filename = key.rsplit("/", 1)[-1]
            token_id = filename[:-5]  # Remove ".json"
            token_ids.add(token_id)
    
    return token_ids

