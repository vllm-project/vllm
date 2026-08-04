# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict
import os

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class SageMakerHyperPodConnectorAdapter(ConnectorAdapter):
    """Adapter for SageMaker HyperPod connectors."""

    def __init__(self) -> None:
        super().__init__("sagemaker-hyperpod://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """
        Create a SageMaker HyperPod connector from the given context.

        Args:
            context: Connector context containing configuration

        Returns:
            Initialized SageMaker HyperPod connector

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If shared memory initialization fails
        """
        # Local import to avoid circular dependencies
        # Local
        from .sagemaker_hyperpod_connector import SageMakerHyperPodConnector

        config = context.config
        assert config is not None, "Config must not be None"
        assert context.loop is not None, "context.loop must not be None"
        assert context.local_cpu_backend is not None, (
            "context.local_cpu_backend must not be None"
        )

        # Default configuration values with type hints
        defaults: Dict[str, Any] = {
            "bucket": "lmcache",
            "shared_memory_name": "shared_memory",
            "max_concurrent_requests": 100,
            "max_connections": 256,
            "max_connections_per_host": 128,
            "timeout_ms": 5000,
            "lease_ttl_s": 30.0,
            "put_stream_chunk_bytes": 65536,
            "use_https": False,
            "max_lease_size_mb": None,
        }

        # Extract and validate configuration
        extra_config = config.extra_config or {}

        bucket_name = str(
            extra_config.get("sagemaker_hyperpod_bucket", defaults["bucket"])
        )
        shared_memory_name = extra_config.get(
            "sagemaker_hyperpod_shared_memory_name", defaults["shared_memory_name"]
        )
        max_concurrent_requests = self._get_positive_int(
            extra_config,
            "sagemaker_hyperpod_max_concurrent_requests",
            int(defaults["max_concurrent_requests"]),  # Cast to int
        )
        max_connections = self._get_positive_int(
            extra_config,
            "sagemaker_hyperpod_max_connections",
            int(defaults["max_connections"]),  # Cast to int
        )
        max_connections_per_host = self._get_positive_int(
            extra_config,
            "sagemaker_hyperpod_max_connections_per_host",
            int(defaults["max_connections_per_host"]),  # Cast to int
        )
        timeout_ms = self._get_positive_int(
            extra_config,
            "sagemaker_hyperpod_timeout_ms",
            int(defaults["timeout_ms"]),  # Cast to int
        )
        lease_ttl_s = self._get_positive_float(
            extra_config,
            "sagemaker_hyperpod_lease_ttl_s",
            float(defaults["lease_ttl_s"]),  # Cast to float
        )
        put_stream_chunk_bytes = self._get_positive_int(
            extra_config,
            "sagemaker_hyperpod_put_stream_chunk_bytes",
            int(defaults["put_stream_chunk_bytes"]),  # Cast to int
        )
        use_https = bool(
            extra_config.get("sagemaker_hyperpod_use_https", defaults["use_https"])
        )
        max_lease_size_mb = extra_config.get(
            "sagemaker_hyperpod_max_lease_size_mb", defaults["max_lease_size_mb"]
        )

        if max_lease_size_mb is not None:
            try:
                max_lease_size_mb = float(max_lease_size_mb)
                if max_lease_size_mb <= 0:
                    raise ValueError(
                        f"sagemaker_hyperpod_max_lease_size_mb must be positive,"
                        f" got {max_lease_size_mb}"
                    )
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid value for sagemaker_hyperpod_max_lease_size_mb:"
                    f" {max_lease_size_mb}"
                ) from e

        # Parse and construct URL
        url = self._parse_url(context.url, use_https)

        logger.info(
            f"Creating SageMaker HyperPod connector: url={url}, "
            f"bucket={bucket_name}, shared_memory={shared_memory_name}, "
            f"max_connections={max_connections}, "
            f"max_concurrent_requests={max_concurrent_requests}, "
            f"timeout_ms={timeout_ms}, lease_ttl_s={lease_ttl_s}s"
            f"max_lease_size_mb="
            f"{max_lease_size_mb if max_lease_size_mb else 'unlimited'}"
        )

        # Create connector instance
        connector = SageMakerHyperPodConnector(
            sagemaker_hyperpod_url=url,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            bucket_name=bucket_name,
            shared_memory_name=shared_memory_name,
            max_concurrent_requests=max_concurrent_requests,
            max_connections=max_connections,
            max_connections_per_host=max_connections_per_host,
            timeout_ms=timeout_ms,
            lease_ttl_s=lease_ttl_s,
            put_stream_chunk_bytes=put_stream_chunk_bytes,
            max_lease_size_mb=max_lease_size_mb,
        )

        logger.info("SageMaker HyperPod connector created successfully")
        return connector

    @staticmethod
    def _parse_url(url: str, use_https: bool) -> str:
        """
        Parse and normalize the SageMaker HyperPod URL.

        Args:
            url: Raw URL from context (e.g., "sagemaker-hyperpod://127.0.0.1:9200")
            use_https: Whether to use HTTPS protocol

        Returns:
            Normalized HTTP/HTTPS URL
        """
        assert url, "SageMaker HyperPod URL must not be empty"

        expanded_url = os.path.expandvars(url)

        # Strip the sagemaker-hyperpod:// prefix
        raw_url = expanded_url.removeprefix("sagemaker-hyperpod://")

        assert raw_url, (
            "SageMaker HyperPod URL must contain host information after 'sagemaker-hyperpod://'"
        )

        # If URL already has protocol, use it as-is
        if raw_url.startswith("http://") or raw_url.startswith("https://"):
            return raw_url

        # Otherwise, add appropriate protocol
        protocol = "https" if use_https else "http"
        return f"{protocol}://{raw_url}"

    @staticmethod
    def _get_positive_int(config_dict: dict, key: str, default: int) -> int:
        """
        Extract a positive integer from config with validation.

        Args:
            config_dict: Configuration dictionary
            key: Configuration key
            default: Default value if key not found

        Returns:
            Validated positive integer

        Raises:
            ValueError: If value is not a positive integer
        """
        value = config_dict.get(key, default)
        try:
            int_value = int(value)
            if int_value <= 0:
                raise ValueError(f"{key} must be positive, got {int_value}")
            return int_value
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid value for {key}: {value}") from e

    @staticmethod
    def _get_positive_float(config_dict: dict, key: str, default: float) -> float:
        """
        Extract a positive float from config with validation.

        Args:
            config_dict: Configuration dictionary
            key: Configuration key
            default: Default value if key not found

        Returns:
            Validated positive float

        Raises:
            ValueError: If value is not a positive float
        """
        value = config_dict.get(key, default)
        try:
            float_value = float(value)
            if float_value <= 0:
                raise ValueError(f"{key} must be positive, got {float_value}")
            return float_value
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid value for {key}: {value}") from e
