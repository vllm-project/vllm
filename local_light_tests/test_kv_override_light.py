import sys, types
import pytest

# Stub heavy import that kv_cache_utils pulls transitively
# before importing kv_cache_utils
mod = types.ModuleType('vllm.v1.request');
mod.Request = type('Request', (), {})
sys.modules['vllm.v1.request'] = mod

from vllm.v1.core import kv_cache_utils as kvc

# Minimal config object with required fields
class DummyConfig:
    def __init__(self, max_model_len: int, override: int | None, max_enc_tokens: int = 2048):
        self.model_config = types.SimpleNamespace(max_model_len=max_model_len)
        self.cache_config = types.SimpleNamespace(num_gpu_blocks_override=override)
        self.scheduler_config = types.SimpleNamespace(max_num_encoder_input_tokens=max_enc_tokens)

# Minimal KV spec implementations used only by check_enough_kv_cache_memory
class FullSpec:
    def __init__(self, block_size: int, per_token_bytes: int):
        self.block_size = block_size
        self.page_size_bytes = block_size * per_token_bytes
    def max_memory_usage_bytes(self, vllm_config):
        # ceil(max_model_len / block_size) * page_size
        from vllm.utils import cdiv
        return cdiv(vllm_config.model_config.max_model_len, self.block_size) * self.page_size_bytes

class CrossSpec(FullSpec):
    def max_memory_usage_bytes(self, vllm_config):
        # ceil(max_num_encoder_input_tokens / block_size) * page_size
        from vllm.utils import cdiv
        return cdiv(vllm_config.scheduler_config.max_num_encoder_input_tokens, self.block_size) * self.page_size_bytes


PER_TOKEN_BYTES = 2 * 2 * 64 * 4  # 2(KV) * num_heads * head_size * dtype(bytes)


def test_override_small_single_type():
    cfg = DummyConfig(max_model_len=32, override=1)
    specs = {
        'l1': FullSpec(16, PER_TOKEN_BYTES),
        'l2': FullSpec(16, PER_TOKEN_BYTES),
    }
    per_block_total = sum(s.page_size_bytes for s in specs.values())
    available = per_block_total * 100
    with pytest.raises(ValueError):
        kvc.check_enough_kv_cache_memory(cfg, specs, available)


def test_override_must_cover_worst_layer_blocks_in_heterogeneous_model():
    # full needs ceil(1024/16)=64 blocks; cross needs ceil(2048/16)=128 blocks
    cfg = DummyConfig(max_model_len=1024, override=96, max_enc_tokens=2048)
    specs = {
        'decoder_self': FullSpec(16, PER_TOKEN_BYTES),
        'cross': CrossSpec(16, PER_TOKEN_BYTES),
    }
    per_block_total = sum(s.page_size_bytes for s in specs.values())
    available = per_block_total * 100
    with pytest.raises(ValueError):
        kvc.check_enough_kv_cache_memory(cfg, specs, available)
