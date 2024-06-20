from functools import lru_cache
import torch

from vllm.attention_sinks import get_attention_sink
from vllm.config import CacheConfig
from vllm.lora.utils import replace_submodule


def apply_attn_sinks_to_model(
    model: torch.nn.Module,
    cache_config: CacheConfig,
    max_model_len: int,
    dtype: torch.dtype
):
    # need to grab a {Model}Attention object for get_attention_sink
    model_attn = None
    for module_name, module in model.named_modules(remove_duplicate=False):
        parts = module_name.split(".")
        
        if len(parts) == 4 and "Attention" in module.__class__.__name__: # sus
            # e.g. 'LlamaAttention'
            model_attn = module
        
        if len(parts) != 5:
            continue
        if parts[-1] == "attn":
            # e.g. 'model.layers.21.self_attn.attn'
            attn_sink_layer = get_attention_sink(model_attn, cache_config, max_model_len, dtype)
            replace_submodule(model, module_name, attn_sink_layer)
        elif parts[-1] == "rotary_emb":
            # e.g. 'model.layers.21.self_attn.rotary_emb'
            replace_submodule(model, module_name, _get_identity_module())


class _Identity(torch.nn.Module):
    """Used to patch rotary_emb.
    StreamingAttentionSink will call rotary_emb, so the rotary_emb
    inside the model's attention module should do nothing.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, positions, q, k):
        return q, k


@lru_cache(maxsize=1)
def _get_identity_module():
    # should this be cached?
    return _Identity()
