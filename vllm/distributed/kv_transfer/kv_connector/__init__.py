
from .base import KVConnectorBase

class KVConnectorFactory:
    
    @staticmethod
    def create_connector(
        local_rank: int,
        config
    ) -> KVConnectorBase:
        if config.kv_connector == 'PyNcclConnector':
            from . import PyNcclConnector 
            return PyNcclConnector(local_rank, config)
        else:
            raise ValueError(f"Unsupported connector type: "
                             f"{config.kv_connector}")