import logging
from abc import ABC
import torch
from vllm.config import ModelConfig
from multiprocessing import shared_memory  
import numpy as np 
import fcntl
from unittest.mock import patch  
logger = logging.getLogger(__name__)

LOCK_FILE = "/tmp/vllm_routed_experts.lock"  # Shared lock file path

def lock_file(fp):
    fcntl.flock(fp, fcntl.LOCK_EX)

def unlock_file(fp):
    fcntl.flock(fp, fcntl.LOCK_UN)

_global_experts_capturer = None

class RoutedExpertsCapturer(ABC):
    @staticmethod
    def create(enable: bool):
        """Create a global singleton instance"""
        global _global_experts_capturer
        if _global_experts_capturer is not None:
            raise RuntimeError("Experts capturer already created.")

        if enable:
            _global_experts_capturer = _RoutedExpertsCapturerReal()
        else:
            _global_experts_capturer = _RoutedExpertsCapturerNoop()
        return _global_experts_capturer

    @staticmethod
    def get_instance():
        if _global_experts_capturer is None:
            logger.info("Experts capturer not initialized.")
        return _global_experts_capturer

    def init_buffer(self, max_num_batched_tokens: int, max_num_kv_tokens: int, model_config: ModelConfig):
        raise NotImplementedError

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        raise NotImplementedError

    def clear_buffer(self):
        raise NotImplementedError

    def save_captured_experts(self, indices: np.ndarray):
        raise NotImplementedError


class _RoutedExpertsCapturerReal(RoutedExpertsCapturer):
    """Capturer for routed experts with host buffer"""
    def __init__(self):
        self._experts_capturer_device_buffer = None

    def init_buffer(self, max_num_batched_tokens: int, max_num_kv_tokens: int, model_config: ModelConfig, enable_shared_memory: bool):
        if (
            model_config.enable_return_routed_experts
            and self._experts_capturer_device_buffer is None
        ):
            self._experts_capturer_device_buffer = torch.zeros(
                (
                    max_num_batched_tokens,
                    model_config.hf_text_config.num_hidden_layers,
                    model_config.hf_text_config.num_experts_per_tok,
                ),
                dtype=torch.int32,
                device="cuda",
            )

            if enable_shared_memory:
                # Compute required shared memory size  
                shape = (  
                    max_num_kv_tokens,  
                    model_config.hf_text_config.num_hidden_layers,  
                    model_config.hf_text_config.num_experts_per_tok,  
                )  
                nbytes = np.prod(shape) * np.dtype(np.int32).itemsize 

                # 创建共享内存  
                with open(LOCK_FILE, "wb") as lockfp:
                    self._shm = shared_memory.SharedMemory(  
                        create=True,   
                        size=nbytes,  
                        name="vllm_routed_experts_buffer"  # Fixed name for worker access  
                    )  

                    # 创建 numpy array 视图  
                    self._host_buffer_view = np.ndarray(  
                        shape, dtype=np.int32, buffer=self._shm.buf  
                    ) 
                    self._host_buffer_view.fill(0)

                logger.debug(  
                    f"Created shared memory buffer '{self._shm.name}' "  
                    f"with shape {shape}"  
                )
            else:
                self._shm = None
                self._host_buffer_view = None

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        if self._experts_capturer_device_buffer is None:
            raise RuntimeError("Buffer not initialized.")
        batch_size, num_routed_experts = topk_ids.shape
        self._experts_capturer_device_buffer[:batch_size, layer_id, : ] = topk_ids

    def clear_buffer(self):
        self._experts_capturer_device_buffer.zero_()


    def save_captured_experts(self, indices: np.ndarray):
        # Copy the entire batch from GPU to shared memory (via numpy view)  
        with open(LOCK_FILE, "wb+") as fp:
            if self._host_buffer_view is not None:
                num_tokens = len(indices)
                data = self._experts_capturer_device_buffer[:num_tokens, :, :].cpu().numpy()  
                self._host_buffer_view[indices, :, :] = data

    def __del__(self):  
        """Clean up shared memory"""  
        if self._shm is not None:  
            self._shm.close()  
            self._shm.unlink()  # Delete shared memory



class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def init_buffer(self, max_num_batched_tokens: int, max_num_kv_tokens: int, model_config: ModelConfig, enable_shared_memory: bool):
        pass

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        pass

    def clear_buffer(self): 
        pass
    
    def save_captured_experts(self, indices: np.ndarray): 
        pass


_global_experts_reader = None

class RoutedExpertsReader(ABC):
    @staticmethod
    def create(enable: bool):
        """Create a global singleton instance"""
        global _global_experts_reader
        if _global_experts_reader is not None:
            raise RuntimeError("Experts Reader already created.")

        if enable:
            _global_experts_reader = _RoutedExpertsReaderReal()
        else:
            _global_experts_reader = _RoutedExpertsReaderNoop()
        return _global_experts_reader

    @staticmethod
    def get_instance():
        if _global_experts_reader is None:
            logger.info("Experts reader not initialized.")
            # raise RuntimeError("Experts reader not initialized.")
        return _global_experts_reader

    def get_routed_experts(self,num_tokens: int):
        raise NotImplementedError


class _RoutedExpertsReaderReal:  
    """Reader class in worker process"""  
    def __init__(self):  
        self._shm = None  
          
    def attach_buffer(self, max_num_kv_tokens: int, model_config: ModelConfig):  
        if self._shm is None:  
            shape = (  
                max_num_kv_tokens,  
                model_config.hf_text_config.num_hidden_layers,  
                model_config.hf_text_config.num_experts_per_tok,  
            )  
              
            # Attach to existing shared memory  
            with open(LOCK_FILE, "rb+") as fp:
                with patch(  
                    "multiprocessing.resource_tracker.register",  
                    lambda *args, **kwargs: None,  
                ):  
                    self._shm = shared_memory.SharedMemory(  
                        name="vllm_routed_experts_buffer"  
                    )  
                
                self._host_buffer_view = np.ndarray(  
                    shape, dtype=np.int32, buffer=self._shm.buf  
                )  
              
    def get_routed_experts(self, indices: np.ndarray):  
        """Read data from shared memory, return routed experts for the corresponding request"""  
        with open(LOCK_FILE, "rb+") as fp:
            if self._host_buffer_view is None:  
                raise RuntimeError("Buffer not attached.")  
            return self._host_buffer_view[indices, :, :].copy()
      
    def __del__(self):  
        """Only close, do not delete shared memory"""  
        if self._shm is not None:  
            self._shm.close()  # Note: reader does not call unlink()

class _RoutedExpertsReaderNoop: 
    def attach_buffer(self, max_num_kv_tokens: int, model_config: ModelConfig):
        pass
    def get_routed_experts(self, indices: np.ndarray):  
        return None