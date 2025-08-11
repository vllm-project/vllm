from typing import TYPE_CHECKING, Optional  
import torch  
from vllm.logger import init_logger  
from .interface import Platform, PlatformEnum, _Backend  
  
if TYPE_CHECKING:  
    from vllm.config import VllmConfig  
  
logger = init_logger(__name__)  
  
class MpsPlatform(Platform):  
    _enum = PlatformEnum.MPS  # You'll need to add this to PlatformEnum  
    device_name: str = "mps"  
    device_type: str = "mps"  
    dispatch_key: str = "MPS"  
      
    @classmethod  
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,  
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],  
                             block_size: int, use_v1: bool,  
                             use_mla: bool) -> str:  
        # MPS attention backend not yet implemented
        raise NotImplementedError("MPS attention backend not yet implemented. "
                                  "This is a placeholder for future MPS attention support.")  
      
    @classmethod  
    def set_device(cls, device: torch.device) -> None:  
        # MPS doesn't need explicit device setting like CUDA
        pass  
      
    @classmethod  
    def get_device_name(cls, device_id: int = 0) -> str:  
        return "Apple MPS"  
      
    @classmethod  
    @classmethod  
    def get_device_total_memory(cls, device_id: int = 0) -> int:  
        # On Apple Silicon, MPS uses unified memory with the CPU.
        # We can use psutil to get the total system memory.
        import psutil
        return psutil.virtual_memory().total
    
    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        # Similar to CUDA, MPS can support async output processing
        # Only disable it when enforce_eager is True and not using V1
        from vllm import envs
        if enforce_eager and not envs.VLLM_USE_V1:
            logger.warning(
                "To see benefits of async output processing, enable MPS "
                "graph. Since, enforce-eager is enabled, async output "
                "processor cannot be used")
            return False
        return True
    
    @classmethod
    def inference_mode(cls):
        """A device-specific wrapper of `torch.inference_mode`."""
        return torch.no_grad()
    
    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Check and update the configuration for MPS platform."""
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            # Use CPU worker for MPS since they're similar
            parallel_config.worker_cls = "vllm.worker.worker.Worker"
        
        # MPS supports single GPU only
        if parallel_config.tensor_parallel_size > 1:
            raise RuntimeError("MPS backend does not support tensor parallelism")
        if parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError("MPS backend does not support pipeline parallelism")
        
        # Set default block size if not set
        cache_config = vllm_config.cache_config
        if cache_config.block_size is None:
            cache_config.block_size = 16  # Default block size
        
        # Disable features that may not work well with MPS
        compilation_config = vllm_config.compilation_config
        compilation_config.use_cudagraph = False  # MPS doesn't support CUDA graphs
    
    @classmethod
    @classmethod
    def get_current_memory_usage(cls, device: torch.device) -> int:
        """Get current memory usage for MPS device."""
        # torch.mps.current_allocated_memory() is available from PyTorch 1.13.
        # It returns the current memory allocated on the MPS device.
        return torch.mps.current_allocated_memory()