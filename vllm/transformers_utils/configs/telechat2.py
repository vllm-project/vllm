# SPDX-License-Identifier: Apache-2.0

# adapted from https://www.modelscope.cn/models/TeleAI/TeleChat2-3B/resolve/master/configuration_telechat2.py
""" Telechat configuration compatible with LlamaConfig. """

from transformers.configuration_utils import PretrainedConfig


class Telechat2Config(PretrainedConfig):

    model_type = "telechat"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
        "intermediate_size": "ffn_hidden_size",
        "rms_norm_eps": "layer_norm_epsilon"
    }

    def __init__(
        self,
        vocab_size=160256,
        hidden_size=4096,
        n_layer=30,
        n_head=32,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        ffn_hidden_size=12288,
        training_seqlen=8192,
        logn=True,
        embed_layernorm=False,
        hidden_act="silu",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.apply_residual_connection_post_layernorm = (
            apply_residual_connection_post_layernorm)
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.logn = logn
        self.training_seqlen = training_seqlen
        self.embed_layernorm = embed_layernorm
        self.num_key_value_heads = kwargs.pop("num_key_value_heads", None)
        self.ffn_hidden_size = ffn_hidden_size
        self.hidden_act = hidden_act
        super().__init__(bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id,
                         **kwargs)
