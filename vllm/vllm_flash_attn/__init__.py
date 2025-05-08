__version__ = "2.7.2.post1"

# Use relative import to support build-from-source installation in vLLM
from .flash_attn_interface import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
    sparse_attn_func,
    sparse_attn_varlen_func,
    is_fa_version_supported,
    fa_version_unsupported_reason
)