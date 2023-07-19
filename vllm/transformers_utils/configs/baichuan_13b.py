# Copyright (c) 2023, Baichuan Intelligent Technology. All rights reserved.

from transformers.configuration_utils import PretrainedConfig


class BaichuanConfig(PretrainedConfig):
    model_type = "baichuan"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=64000,
        hidden_size=5120,
        intermediate_size=13696,
        num_hidden_layers=40,
        num_attention_heads=40,
        hidden_act="silu",
        model_max_length=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        gradient_checkpointing=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.model_max_length = model_max_length
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.gradient_checkpointing = (gradient_checkpointing,)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
