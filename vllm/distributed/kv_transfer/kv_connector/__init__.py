
from .base import KVConnectorBase
from vllm.config import ParallelConfig

class KVConnectorFactory:
    
    @staticmethod
    def create_connector(
        config: ParallelConfig
    ) -> KVConnectorBase:
        if config.kv_connector == 'LMCacheConnector':
            from .lmcache_connector import LMCacheConnector
            return LMCacheConnector(config)
        elif config.kv_connector == 'TorchDistributedConnector':
            from .torch_distributed_connector import TorchDistributedConnector
            return TorchDistributedConnector(config)
        else:
            raise ValueError(f"Unsupported connector type: {connector_type}")