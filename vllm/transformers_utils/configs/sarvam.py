# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from packaging.version import Version
from transformers import PretrainedConfig
from transformers import __version__ as TRANSFORMERS_VERSION


class SarvamMLAConfig(PretrainedConfig):
    model_type = "sarvam_mla"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=262144,
        hidden_size=4096,
        intermediate_size=11008,
        moe_intermediate_size=4864,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_experts=128,
        num_experts_per_tok=4,
        num_shared_experts=2,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_parameters=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings

        # MLA-specific parameters
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        # Handle rope_scaling from config
        # (Sarvam uses rope_scaling instead of rope_parameters)
        # rope_scaling should be converted to rope_parameters for compatibility
        if rope_scaling is not None:
            rope_parameters = rope_scaling
        elif rope_parameters is None:
            rope_parameters = {}

        if (
            isinstance(rope_parameters, dict)
            and rope_theta is not None
            and "rope_theta" not in rope_parameters
        ):
            rope_parameters["rope_theta"] = rope_theta

        self.rope_parameters = rope_parameters or {}

        # For Transformers v5.3+, set ignore_keys_at_rope_validation to skip validation
        # of custom rope parameters like mscale_all_dim and beta_fast/beta_slow
        if Version(TRANSFORMERS_VERSION) >= Version("5.3.0.dev0"):
            # These keys are custom to SarvamMLA's deepseek_yarn rope type
            # and may not be recognized by the standard validation
            kwargs["ignore_keys_at_rope_validation"] = [
                "mscale_all_dim",
                "beta_fast",
                "beta_slow",
                "factor",
                "original_max_position_embeddings",
                "mscale",
            ]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        # Set these AFTER super().__init__() because transformers v4's
        # PretrainedConfig.__init__ has these as explicit params with different
        # defaults (e.g. tie_word_embeddings=True) that would overwrite our values.
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
