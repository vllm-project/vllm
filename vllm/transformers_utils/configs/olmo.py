# coding=utf-8
# adapted from https://github.com/allenai/OLMo/blob/v0.2.4/hf_olmo/configuration_olmo.py
"""OLMo configuration"""
from transformers import PretrainedConfig


class OLMoConfig(PretrainedConfig):
    model_type = 'olmo'
    attribute_map = {
        'num_attention_heads': 'n_heads',
        'hidden_size': 'd_model',
        'num_hidden_layers': 'n_layers',
    }

    # Note that the defaults for these attributes are equivalent to the base GPT2 model.
    def __init__(
        self,
        d_model=768,
        n_heads=12,
        n_layers=12,
        mlp_ratio=4,
        mlp_hidden_size=None,
        activation_type="swiglu",
        block_type="sequential",
        block_group_size=1,
        alibi=False,
        alibi_bias_max=8.0,
        rope=False,
        rope_full_precision=True,
        multi_query_attention=False,
        attention_layer_norm=False,
        layer_norm_type="default",
        layer_norm_with_affine=True,
        attention_layer_norm_with_affine=True,
        max_sequence_length=1024,
        include_bias=True,
        bias_for_layer_norm=None,
        scale_logits=False,
        vocab_size=50257,
        embedding_size=50304,
        weight_tying=True,
        eos_token_id=50256,
        pad_token_id=50256,
        **kwargs,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_size = mlp_hidden_size
        self.activation_type = activation_type
        self.block_type = block_type
        self.block_group_size = block_group_size
        self.alibi = alibi
        self.alibi_bias_max = alibi_bias_max
        self.rope = rope
        self.rope_full_precision = rope_full_precision
        self.multi_query_attention = multi_query_attention
        self.attention_layer_norm = attention_layer_norm
        self.layer_norm_type = layer_norm_type
        self.layer_norm_with_affine = layer_norm_with_affine
        self.attention_layer_norm_with_affine = attention_layer_norm_with_affine
        self.max_sequence_length = max_sequence_length
        self.include_bias = include_bias
        self.bias_for_layer_norm = bias_for_layer_norm
        self.scale_logits = scale_logits
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.weight_tying = weight_tying
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        super().__init__(**kwargs)
