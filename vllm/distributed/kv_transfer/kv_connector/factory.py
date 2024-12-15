from typing import TYPE_CHECKING

from .base import KVConnectorBase

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class KVConnectorFactory:

    @staticmethod
    def create_connector(rank: int, local_rank: int,
                         config: "VllmConfig") -> KVConnectorBase:
        supported_kv_connector = ["PyNcclConnector", "MooncakeConnector"]
        if config.kv_transfer_config.kv_connector in supported_kv_connector:
            from .simple_connector import SimpleConnector
            return SimpleConnector(rank, local_rank, config)
        else:
            raise ValueError(f"Unsupported connector type: "
                             f"{config.kv_connector}")
