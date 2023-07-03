# Adapted from https://huggingface.co/mosaicml/mpt-7b/blob/main/configuration_mpt.py
from typing import Dict, Optional, Union

from transformers import PretrainedConfig
attn_config_defaults: Dict = {'attn_type': 'multihead_attention', 'attn_pdrop': 0.0, 'attn_impl': 'triton', 'qk_ln': False, 'clip_qkv': None, 'softmax_scale': None, 'prefix_lm': False, 'attn_uses_sequence_id': False, 'alibi': False, 'alibi_bias_max': 8}
init_config_defaults: Dict = {'name': 'kaiming_normal_', 'fan_mode': 'fan_in', 'init_nonlinearity': 'relu', 'init_div_is_residual': True, 'emb_init_std': None, 'emb_init_uniform_lim': None, 'init_std': None, 'init_gain': 0.0}


class MPTConfig(PretrainedConfig):
    model_type = "mpt"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
    }

    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        n_layers: int = 24,
        expansion_ratio: int = 4,
        max_seq_len: int = 2048,
        vocab_size: int = 50368,
        resid_pdrop: float = 0.0,
        emb_pdrop: float = 0.0,
        learned_pos_emb: bool = True,
        attn_config: Dict = attn_config_defaults,
        init_device: str = "cpu",
        logit_scale: Optional[Union[float, str]] = None,
        no_bias: bool = False,
        verbose: int = 0,
        embedding_fraction: float = 1.0,
        norm_type: str = "low_precision_layernorm",
        use_cache: bool = False,
        init_config: Dict = init_config_defaults,
        **kwargs,
    ) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.learned_pos_emb = learned_pos_emb
        self.attn_config = attn_config
        self.init_device = init_device
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.verbose = verbose
        self.embedding_fraction = embedding_fraction
        self.norm_type = norm_type
        self.use_cache = use_cache
        self.init_config = init_config
        if 'name' in kwargs:
            del kwargs['name']
        if 'loss_fn' in kwargs:
            del kwargs['loss_fn']
        super().__init__(**kwargs)
