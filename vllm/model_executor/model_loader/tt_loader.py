from typing import Optional

from torch import nn

from vllm.model_executor.model_loader.loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import get_model_architecture
from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig, 
                         ParallelConfig, SchedulerConfig)


class TTModelLoader(BaseModelLoader):
    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig,
                   cache_config: CacheConfig) -> nn.Module:
        """Load a model with the given configurations."""
        
        # For TT models, prepend "TT" to the architecture name, e.g. "TTLlamaForCausalLM"
        arch_names = model_config.hf_config.architectures
        assert len(model_config.hf_config.architectures) == 1
        arch_names[0] = "TT" + arch_names[0]
        
        model_class, _ = get_model_architecture(model_config)
        model = model_class.initialize_vllm_model(model_config.hf_config, device_config.device, scheduler_config.max_num_seqs)
        return model
    
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError