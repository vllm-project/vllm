# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cartridge manager for downloading and caching KV cache cartridges."""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import filelock
import torch

logger = logging.getLogger(__name__)


class CartridgeManager:
    """
    Manager for downloading and caching KV cache cartridges from S3 or local paths.

    This class handles:
    - Downloading cartridges from S3 URIs (s3://bucket/path/to/cartridge.pt)
    - Loading cartridges from local file paths
    - Caching downloaded files to avoid re-downloading
    - Thread-safe file locking during downloads
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the CartridgeManager.

        Args:
            cache_dir: Directory to cache downloaded cartridges.
                      Defaults to ~/.cache/vllm/cartridges
        """
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "vllm", "cartridges"
            )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CartridgeManager initialized with cache_dir: {self.cache_dir}")

    def _get_cache_path(self, uri: str) -> Path:
        """
        Get the local cache path for a given URI.

        Args:
            uri: The S3 URI or identifier of the cartridge

        Returns:
            Path to the cached file
        """
        # Use hash of URI as filename to avoid path traversal issues
        uri_hash = hashlib.sha256(uri.encode()).hexdigest()
        return self.cache_dir / f"{uri_hash}.pt"

    def _download_from_s3(self, s3_uri: str, local_path: Path) -> None:
        """
        Download a file from S3 to a local path.

        Args:
            s3_uri: S3 URI (e.g., s3://bucket/path/to/file.pt)
            local_path: Local path to save the downloaded file

        Raises:
            ImportError: If boto3 is not installed
            Exception: If download fails
        """
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError as e:
            raise ImportError(
                "boto3 is required to download cartridges from S3. "
                "Please install it with: pip install boto3"
            ) from e

        # Parse S3 URI
        parsed = urlparse(s3_uri)
        if parsed.scheme != "s3":
            raise ValueError(f"Invalid S3 URI: {s3_uri}. Expected s3:// scheme.")

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        if not bucket or not key:
            raise ValueError(
                f"Invalid S3 URI: {s3_uri}. Expected format: s3://bucket/path/to/file"
            )

        logger.info(f"Downloading cartridge from S3: {s3_uri}")

        try:
            s3_client = boto3.client("s3")

            # Download to a temporary file first, then move to final location
            temp_path = local_path.with_suffix(".tmp")
            s3_client.download_file(bucket, key, str(temp_path))
            temp_path.rename(local_path)

            logger.info(f"Successfully downloaded cartridge to {local_path}")
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to download from S3: {e}")
            # Clean up partial download
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to download cartridge from {s3_uri}: {e}") from e

    def get_cartridge(
        self,
        cartridge_id: str,
        source: str = "s3",
        force_redownload: bool = False,
    ) -> torch.Tensor:
        """
        Get a cartridge, downloading from S3 or loading from local path.

        Args:
            cartridge_id: The identifier/path of the cartridge.
                         For S3: s3://bucket/path/to/cartridge.pt
                         For local: /path/to/local/cartridge.pt
            source: Source type ('s3' or 'local')
            force_redownload: If True, re-download even if cached

        Returns:
            Loaded cartridge tensor data

        Raises:
            FileNotFoundError: If local file doesn't exist
            RuntimeError: If download or loading fails
        """
        if source == "local":
            # Load directly from local path
            local_path = Path(cartridge_id)
            if not local_path.exists():
                raise FileNotFoundError(f"Local cartridge not found: {cartridge_id}")

            logger.info(f"Loading cartridge from local path: {cartridge_id}")
            return torch.load(local_path, map_location="cpu")

        elif source == "s3":
            # Get cache path
            cache_path = self._get_cache_path(cartridge_id)
            lock_path = cache_path.with_suffix(".lock")

            # Use file lock to prevent concurrent downloads
            with filelock.FileLock(lock_path, timeout=300):
                # Check if we need to download
                should_download = force_redownload or not cache_path.exists()

                if should_download:
                    logger.info(
                        f"Downloading cartridge (force_redownload={force_redownload})"
                    )
                    self._download_from_s3(cartridge_id, cache_path)
                else:
                    logger.info(f"Using cached cartridge: {cache_path}")

                # Load the cartridge
                try:
                    return torch.load(cache_path, map_location="cpu")
                except Exception as e:
                    logger.error(f"Failed to load cartridge from {cache_path}: {e}")
                    # If loading failed, try re-downloading once
                    if not should_download:
                        logger.info("Retrying with fresh download...")
                        self._download_from_s3(cartridge_id, cache_path)
                        return torch.load(cache_path, map_location="cpu")
                    raise RuntimeError(f"Failed to load cartridge: {e}") from e

        else:
            raise ValueError(f"Unknown source type: {source}. Expected 's3' or 'local'.")

    def clear_cache(self) -> None:
        """Clear all cached cartridges."""
        logger.info("Clearing cartridge cache...")
        for file in self.cache_dir.glob("*.pt"):
            file.unlink()
        for file in self.cache_dir.glob("*.lock"):
            file.unlink()
        logger.info("Cache cleared")


# Global cartridge manager instance
_global_cartridge_manager: Optional[CartridgeManager] = None


def get_cartridge_manager() -> CartridgeManager:
    """Get or create the global cartridge manager instance."""
    global _global_cartridge_manager
    if _global_cartridge_manager is None:
        _global_cartridge_manager = CartridgeManager()
    return _global_cartridge_manager
