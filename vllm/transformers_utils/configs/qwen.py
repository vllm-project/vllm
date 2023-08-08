# Copyright (c) Alibaba Cloud.
# LICENSE: https://huggingface.co/Qwen/Qwen-7B/blob/main/LICENSE

from transformers import PretrainedConfig


class QWenConfig(PretrainedConfig):
    model_type = "qwen"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "max_position_embeddings": "n_positions",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=151851,
        n_embd=4096,
        n_layer=32,
        n_head=32,
        n_inner=None,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        eos_token_id=151643,
        apply_residual_connection_post_layernorm=False,
        bf16=True,
        kv_channels=128,
        rotary_pct=1.0,
        rotary_emb_base=10000,
        use_dynamic_ntk=False,
        use_logn_attn=False,
        use_flash_attn=True,
        ffn_hidden_size=22016,
        no_bias=True,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.eos_token_id = eos_token_id
        super().__init__(eos_token_id=eos_token_id,
                         tie_word_embeddings=tie_word_embeddings,
                         **kwargs)

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.apply_residual_connection_post_layernorm = (
            apply_residual_connection_post_layernorm)
        self.bf16 = bf16
        self.kv_channels = kv_channels
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.use_dynamic_ntk = use_dynamic_ntk
        self.use_logn_attn = use_logn_attn
        self.use_flash_attn = use_flash_attn
        self.ffn_hidden_size = ffn_hidden_size
        self.no_bias = no_bias
        self.tie_word_embeddings = tie_word_embeddings
