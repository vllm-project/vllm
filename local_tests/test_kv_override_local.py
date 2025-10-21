import pytest
from vllm.config import ModelConfig, VllmConfig
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.kv_cache_interface import FullAttentionSpec, CrossAttentionSpec

def new_kv(block_size=16, num_kv_heads=2, head_size=64):
    # dtype is accessed for size only; torch dtype not strictly needed for logic paths
    import torch
    return FullAttentionSpec(block_size=block_size, num_kv_heads=num_kv_heads, head_size=head_size, dtype=torch.float32)

def new_cross(block_size=16, num_kv_heads=2, head_size=64):
    import torch
    return CrossAttentionSpec(block_size=block_size, num_kv_heads=num_kv_heads, head_size=head_size, dtype=torch.float32)

def test_override_small_single_type():
    m = ModelConfig(max_model_len=32)
    v = VllmConfig(model_config=m)
    specs = {"l1": new_kv(), "l2": new_kv()}
    v.cache_config.num_gpu_blocks_override = 1
    per_block_total = sum(s.page_size_bytes for s in specs.values())
    avail = per_block_total * 100
    with pytest.raises(ValueError):
        get_kv_cache_configs(v, [specs], [avail])

def test_override_must_cover_worst_layer():
    m = ModelConfig(max_model_len=1024)
    v = VllmConfig(model_config=m)
    specs = {"decoder": new_kv(), "cross": new_cross()}
    v.cache_config.num_gpu_blocks_override = 96  # < 128 needed for cross-attn
    per_block_total = sum(s.page_size_bytes for s in specs.values())
    avail = per_block_total * 100
    with pytest.raises(ValueError):
        get_kv_cache_configs(v, [specs], [avail])
