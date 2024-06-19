from functools import lru_cache
from torch import nn

from vllm.attention_sinks import get_attention_sink
from vllm.config import CacheConfig
from vllm.lora.utils import replace_submodule


def apply_attn_sinks_to_model(
    model: nn.Module,
    cache_config: CacheConfig,
    max_model_len: int
):
    # need to grab a {Model}Attention object for get_attention_sink
    model_attn = None
    for module_name, module in model.named_modules(remove_duplicate=False):
        parts = module_name.split(".")
        
        if len(parts) == 4 and "Attention" in module.__class__.__name__: # sus
            model_attn = module
        
        if len(parts) != 5:
            continue
        if parts[-1] == "attn":
            # e.g. 'model.layers.21.self_attn.attn' (llama)
            attn_sink_layer = get_attention_sink(model_attn, cache_config, max_model_len)
            replace_submodule(model, module_name, attn_sink_layer)
        elif parts[-1] == "rotary_emb":
            replace_submodule(model, module_name, _get_identity_module())


class Identity(nn.Module):
    """Used to patch rotary_emb."""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


@lru_cache(maxsize=1)
def _get_identity_module():
    return Identity()