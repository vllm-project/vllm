# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class S3ConnectorAdapter(ConnectorAdapter):
    """Adapter for S3 Server connectors."""

    def __init__(self) -> None:
        super().__init__("s3://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .s3_connector import S3Connector

        config = context.config
        assert config is not None

        # Get config from extra_config with defaults
        extra_config = config.extra_config if config.extra_config is not None else {}

        self.save_chunk_meta = bool(extra_config.get("save_chunk_meta", False))
        assert not self.save_chunk_meta, "save_chunk_meta must be False for S3"

        self.s3_num_io_threads = int(extra_config.get("s3_num_io_threads", 64))
        self.s3_prefer_http2 = bool(extra_config.get("s3_prefer_http2", True))
        self.s3_region = extra_config.get("s3_region", None)
        assert self.s3_region is not None, "s3_region is required"
        self.s3_region = str(self.s3_region)
        self.s3_enable_s3express = bool(extra_config.get("s3_enable_s3express", False))
        self.disable_tls = bool(extra_config.get("disable_tls", False))
        self.aws_access_key_id = extra_config.get("aws_access_key_id", None)
        self.aws_secret_access_key = extra_config.get("aws_secret_access_key", None)
        if context.metadata is None:
            raise ValueError("metadata is required for S3Connector")

        logger.info(f"Creating S3 connector for URL: {context.url}")

        s3_endpoint = context.url

        return S3Connector(
            s3_endpoint=s3_endpoint,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            s3_num_io_threads=self.s3_num_io_threads,
            s3_prefer_http2=self.s3_prefer_http2,
            s3_region=self.s3_region,
            s3_enable_s3express=self.s3_enable_s3express,
            disable_tls=self.disable_tls,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )
