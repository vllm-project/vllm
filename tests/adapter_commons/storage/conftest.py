"""Test fixtures for storage provider tests."""

import os
import tempfile
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

# Test data
TEST_ADAPTER_CONTENT = b"mock adapter content"
TEST_ADAPTER_SIZE = len(TEST_ADAPTER_CONTENT)


@pytest.fixture(scope="session", autouse=True)
def aws_credentials():
    """Configure mock AWS credentials."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture(scope="function")
def mock_s3():
    """Create a mocked S3 environment."""
    with mock_aws():
        s3 = boto3.client("s3")
        # Create test bucket
        s3.create_bucket(Bucket="test-bucket")
        # Upload test adapter
        s3.put_object(
            Bucket="test-bucket",
            Key="adapters/test.bin",
            Body=TEST_ADAPTER_CONTENT
        )
        yield s3


@pytest.fixture(scope="function")
def mock_s3_error():
    """Create a mocked S3 environment that simulates errors."""
    with mock_aws():
        s3 = boto3.client("s3")
        # Create bucket but don't upload any objects
        s3.create_bucket(Bucket="test-bucket")
        yield s3


@pytest.fixture(scope="function")
def large_adapter_content():
    """Create large test adapter content."""
    return os.urandom(1024 * 1024)  # 1MB


@pytest.fixture(scope="function")
def mock_s3_with_large_file(large_adapter_content):
    """Create a mocked S3 environment with a large test file."""
    with mock_aws():
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket="test-bucket")
        s3.put_object(
            Bucket="test-bucket",
            Key="adapters/large.bin",
            Body=large_adapter_content
        )
        yield s3 