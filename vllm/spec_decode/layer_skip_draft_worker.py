from typing import Optional, List, Tuple
import torch.nn as nn
from vllm.worker.worker_base import WorkerBase
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest

class LayerSkipDraftWorker(WorkerBase):
    """Minimal worker for layer-skip draft generation.
    
    This worker uses a provided model runner (EarlyExitModelRunner) that wraps
    the scorer's model runner to perform early exit at a specified layer.
    """
    
    def __init__(self, model_runner, vllm_config, *args, **kwargs):
        # Initialize WorkerBase to set up all required attributes
        super().__init__(vllm_config)
        
        # Store the provided model runner (our EarlyExitModelRunner wrapper)
        self.model_runner = model_runner
        
        # Override device/config attributes to match the model runner
        self.device = model_runner.device
        self.device_config = model_runner.device_config
        self.model_config = model_runner.model_config
    
    def init_device(self) -> None:
        """Device is already initialized by the scorer worker."""
        pass
    
    def load_model(self) -> None:
        """Model weights are shared from scorer via our model runner wrapper."""
        pass
    
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Memory is managed by the scorer worker."""
        return 0, 0
    
    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """KV cache is managed by the scorer worker."""
        pass
    
    def get_cache_block_size_bytes(self) -> int:
        """Delegate to model runner."""
        return self.model_runner.get_cache_block_size_bytes()
    
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> Optional[List[SamplerOutput]]:
        """Execute using the wrapped model runner."""
        return self.model_runner.execute_model(execute_model_req)
    
    def get_model(self) -> nn.Module:
        """Return the model from the wrapped runner."""
        return self.model_runner.model