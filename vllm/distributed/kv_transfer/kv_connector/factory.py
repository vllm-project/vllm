
from .base import KVConnectorBase

class KVConnectorFactory:
    
    @staticmethod
    def create_connector(
        rank: int,
        local_rank: int,
        config
    ) -> KVConnectorBase:
        if config.kv_connector == 'PyNcclConnector':
            from .pynccl_connector.pynccl_connector import PyNcclConnector 
            return PyNcclConnector(rank, local_rank, config)
        else:
            raise ValueError(f"Unsupported connector type: "
                             f"{config.kv_connector}")