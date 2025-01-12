from collections import deque
from typing import Deque, Dict, Optional
import torch
from vllm.config import ModelConfig, CacheConfig
from vllm.logger import init_logger
import psutil

logger = init_logger(__name__)

class GlobalCache:
    """
    For now just use a simple Dict to store golbal kv cache 
    and a Deque to evict the least used key.
    It can be easily extended and integrated with other kv cache pools 
    that shared with other vllm instances. 
    """
    def __init__(self, max_mem_util: float):
        self.cachedBlockNum: int = 0
        self.max_mem_util: float = max_mem_util
        self.blockHashDict_k: Dict[int, Dict[int, torch.Tensor]] = {}
        self.blockHashDict_v: Dict[int, Dict[int, torch.Tensor]] = {}
        self.cachedBlockHashQ: Deque[int] = deque()

    def setGlabalCacheBlockNum(
            self, model_config: ModelConfig, 
            cache_config: CacheConfig, 
            dtype: torch.dtype):
        if self.cachedBlockNum > 0:
            logger.warning("global kv cache already enabled")
            return        
        if cache_config.num_global_cache_blocks <= 0:
            logger.warning("num_global_cache_blocks is not valid")
            return
        available_mem = psutil.virtual_memory().available * self.max_mem_util
        num_kv_heads = model_config.hf_text_config.num_attention_heads
        head_size = (model_config.hf_text_config.hidden_size // 
            model_config.hf_text_config.num_attention_heads)
        num_attention_layers = model_config.hf_config.num_hidden_layers
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        key_size_bytes = (dtype_size * cache_config.block_size * 
            num_kv_heads * head_size * num_attention_layers)
        if (key_size_bytes * 2 * cache_config.num_global_cache_blocks > 
            available_mem):
            logger.warning("num_global_cache_blocks too large, can not enable "
                "global kv cache, at most %d blocks can be used", 
                available_mem // (key_size_bytes * 2))
            return
        self.cachedBlockNum = cache_config.num_global_cache_blocks     
        logger.info("global kv cache enabled: %d", self.cachedBlockNum)

    def writeCache(
            self, h: int, idx: int, 
            k_block_tensor: torch.Tensor, v_block_tensor: torch.Tensor):
        if self.cachedBlockNum == 0:
            return
        if len(self.cachedBlockHashQ) == self.cachedBlockNum:
            poped_block_hash = self.cachedBlockHashQ.popleft()
            del self.blockHashDict_k[poped_block_hash]
            del self.blockHashDict_v[poped_block_hash]
        if (h not in self.blockHashDict_k or 
            h not in self.blockHashDict_v):
            self.blockHashDict_k[h] = {}
            self.blockHashDict_v[h] = {}
        else:
            self.cachedBlockHashQ.remove(h)

        self.blockHashDict_k[h][idx] = \
            k_block_tensor.to(device="cpu", non_blocking=True)
        self.blockHashDict_v[h][idx] = \
            v_block_tensor.to(device="cpu", non_blocking=True)
        self.cachedBlockHashQ.append(h)

    def readCache(self, h: int, idx: int, device: torch.device):
        if self.cachedBlockNum == 0:
            return
        if not self.checkExist(h):
            return
        self.cachedBlockHashQ.remove(h)
        self.cachedBlockHashQ.append(h)
        return self.blockHashDict_k[h][idx].to(device, non_blocking=True), \
                self.blockHashDict_v[h][idx].to(device, non_blocking=True)
    
    def checkExist(self, h: Optional[int]):
        return h in self.blockHashDict_k and h in self.blockHashDict_v

global_cache_instance = GlobalCache(max_mem_util=0.8)