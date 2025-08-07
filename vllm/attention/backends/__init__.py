"""
Attention backends for vLLM.
"""

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.flash_attn import FlashAttentionBackend
from vllm.attention.backends.flash_attn_3 import FlashAttention3Backend

try:
    from vllm.attention.backends.hpu_attn import HPUAttentionBackend

    _HPU_AVAILABLE = True
except ImportError:
    _HPU_AVAILABLE = False

__all__ = [
    "AttentionBackend",
    "FlashAttentionBackend",
    "FlashAttention3Backend",
]

if _HPU_AVAILABLE:
    __all__.append("HPUAttentionBackend")
