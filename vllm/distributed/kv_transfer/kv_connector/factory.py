from typing import TYPE_CHECKING

from .base import KVConnectorBase

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class KVConnectorFactory:

    @staticmethod
    def create_connector(rank: int, local_rank: int,
                         config: "VllmConfig") -> KVConnectorBase:
        if config.kv_transfer_config.kv_connector == 'PyNcclConnector':
            from .pynccl_connector.connector import PyNcclConnector
            return PyNcclConnector(rank, local_rank, config)
        else:
            raise ValueError(f"Unsupported connector type: "
                             f"{config.kv_connector}")
