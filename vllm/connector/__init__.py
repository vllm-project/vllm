# SPDX-License-Identifier: Apache-2.0

import enum

from vllm.connector.base_connector import (BaseConnector, BaseFileConnector,
                                           BaseKVConnector)
from vllm.connector.redis import RedisConnector
from vllm.connector.s3 import S3Connector
from vllm.logger import init_logger
from vllm.transformers_utils.utils import parse_connector_type

logger = init_logger(__name__)


class ConnectorType(str, enum.Enum):
    FS = "filesystem"
    KV = "KV"


def create_remote_connector(url, **kwargs) -> BaseConnector:
    connector_type = parse_connector_type(url)
    if connector_type == "redis":
        return RedisConnector(url)
    elif connector_type == "s3":
        return S3Connector(url)
    else:
        raise ValueError(f"Invalid connector type: {url}")


def get_connector_type(client: BaseConnector) -> ConnectorType:
    if isinstance(client, BaseKVConnector):
        return ConnectorType.KV
    if isinstance(client, BaseFileConnector):
        return ConnectorType.FS

    raise ValueError(f"Invalid connector type: {client}")


__all__ = [
    "BaseConnector",
    "BaseFileConnector",
    "BaseKVConnector",
    "RedisConnector",
    "HPKVConnector",
    "S3Connector",
    "ConnectorType",
    "create_remote_connector",
    "get_connector_type",
]
