import enum
from functools import lru_cache
from typing import Optional, Type

import torch

from vllm.wde.encode_only.layers.attention.backends.abstract import EncodeOnlyAttentionBackend
from vllm.wde.core.llm_engine import LLMEngine
from vllm.logger import init_logger
logger = init_logger(__name__)


class _Backend(enum.Enum):
    FLASH_ATTN = enum.auto()
    XFORMERS = enum.auto()
    ROCM_FLASH = enum.auto()
    TORCH_SDPA = enum.auto()
    OPENVINO = enum.auto()
    FLASHINFER = enum.auto()
    PALLAS = enum.auto()
    IPEX = enum.auto()


@lru_cache(maxsize=None)
def get_attn_backend(
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    kv_cache_dtype: Optional[str],
    block_size: int,
    is_blocksparse: bool = False,
) -> Type[EncodeOnlyAttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""

    from vllm.wde.encode_only.layers.attention.backends.flash_attn import EncodeOnlyFlashAttentionBackend
    return EncodeOnlyFlashAttentionBackend


class AttnBackend:
    @classmethod
    def from_engine(cls, engine: LLMEngine):
        from vllm.wde.encode_only.layers.attention.backends.flash_attn import EncodeOnlyFlashAttentionBackend
        return EncodeOnlyFlashAttentionBackend