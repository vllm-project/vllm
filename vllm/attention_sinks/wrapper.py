import torch
from torch import nn
from typing import Tuple

from vllm.attention.selector import which_attn_to_use
from vllm.attention_sinks import StreamingAttentionSink
from vllm.config import CacheConfig, ModelConfig
from vllm.lora.utils import replace_submodule


def apply_attn_sinks_to_model(
    model: nn.Module,
    model_config: ModelConfig,
    cache_config: CacheConfig,
    chunked_prefill_enabled: bool,
) -> None:
    # need to grab a {Model}Attention object for get_attention_sink
    model_attn = None
    # save StreamingAttentionSinks for initializing _RopeIdentity modules
    attn_sink_modules = {}

    for module_name, module in model.named_modules(remove_duplicate=False):
        parts = module_name.split(".")
        if len(parts) == 4 and "Attention" in module.__class__.__name__:
            # e.g. 'LlamaAttention'
            model_attn = module
        elif len(parts) == 5 and parts[-1] == "attn":
            # e.g. 'model.layers.21.self_attn.attn'
            assert module.__class__.__name__ == "Attention"
            attn_sink_module = _get_attention_sink(model_attn, model_config, cache_config, chunked_prefill_enabled)
            replace_submodule(model, module_name, attn_sink_module)
            layer_idx = parts[2]
            attn_sink_modules[layer_idx] = attn_sink_module

    for module_name, module in model.named_modules(remove_duplicate=False):
        parts = module_name.split(".")
        if len(parts) == 5 and parts[-1] == "rotary_emb":
            # e.g. 'model.layers.21.self_attn.rotary_emb'
            assert module.__class__.__name__ == "RotaryEmbedding"
            layer_idx = parts[2]
            rope_patch = _RopeIdentity(attn_sink_modules[layer_idx])
            replace_submodule(model, module_name, rope_patch)


class _RopeIdentity(nn.Module):
    """Used to patch rotary_emb.
    StreamingAttentionSink will call rotary_emb, so the rotary_emb
    inside the model's attention module should not change q, k.
    Instead, `positions` will be passed into StreamingAttentionSink
    to be used in that module's forward function.
    """
    def __init__(self, attn_sink_module: StreamingAttentionSink):
        super().__init__()
        assert isinstance(attn_sink_module, StreamingAttentionSink)
        self.attn_sink_module = attn_sink_module
    
    def forward(self, positions, q, k) -> Tuple[torch.Tensor, torch.Tensor]:
        self.attn_sink_module.save_positions(positions)
        return q, k


def _get_attention_sink(
    model_attn: nn.Module,
    model_config: ModelConfig,
    cache_config: CacheConfig,
    chunked_prefill_enabled: bool,
) -> StreamingAttentionSink:
    num_kv_heads = getattr(model_attn, "num_kv_heads", model_attn.num_heads)
    attn_backend = which_attn_to_use(
        model_attn.num_heads,
        model_attn.head_dim,
        num_kv_heads,
        cache_config.sliding_window,
        model_config.dtype,
        cache_config.cache_dtype,
        cache_config.block_size
    )

    return StreamingAttentionSink(
        model_config.max_model_len,
        cache_config.block_size,
        cache_config.cache_dtype,
        attn_backend,
        num_kv_heads,
        model_attn.head_dim,
        getattr(model_attn.attn, "kv_scale", 1.0),
        getattr(model_attn, "rotary_emb", None),
        model_attn.attn,
        chunked_prefill_enabled
    )
