# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import PretrainedConfig


class MiniMaxM3TextConfig(PretrainedConfig):
    """Config for the MiniMax M3 text backbone (MiniMaxM3SparseForCausalLM).

    Defaults mirror the ``text_config`` of the MiniMax-M3-preview checkpoint.
    """

    model_type = "minimax_m3_text"
    architectures = ["MiniMaxM3SparseForCausalLM"]

    def __init__(
        self,
        vocab_size: int = 200064,
        hidden_size: int = 6144,
        intermediate_size: int = 3072,
        dense_intermediate_size: int = 12288,
        shared_intermediate_size: int = 3072,
        num_hidden_layers: int = 60,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        max_position_embeddings: int = 524288,
        rms_norm_eps: float = 1e-6,
        use_gemma_norm: bool = True,
        attention_output_gate: bool = False,
        rope_theta: float = 5000000,
        rotary_dim: int = 64,
        partial_rotary_factor: float = 0.5,
        hidden_act: str = "swigluoai",
        swiglu_alpha: float = 1.702,
        # SwiGLU-OAI uses the (up + 1) bias, i.e. beta=1.0 (matches the
        # reference: gate * sigmoid(gate * alpha) * (up + 1)). The checkpoint
        # config omits swiglu_beta, so this default must stay 1.0.
        swiglu_beta: float = 1.0,
        swiglu_limit: float = 7.0,
        use_qk_norm: bool = True,
        qk_norm_type: str = "per_head",
        num_local_experts: int = 128,
        num_experts_per_tok: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "sigmoid",
        use_routing_bias: bool = True,
        routed_scaling_factor: float = 2.0,
        num_mtp_modules: int = 1,
        moe_layer_freq: list[int] | None = None,
        sparse_attention_config: dict[str, Any] | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dense_intermediate_size = dense_intermediate_size
        self.shared_intermediate_size = shared_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_gemma_norm = use_gemma_norm
        self.attention_output_gate = attention_output_gate
        self.rope_theta = rope_theta
        self.rotary_dim = rotary_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.hidden_act = hidden_act
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.use_routing_bias = use_routing_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.num_mtp_modules = num_mtp_modules
        # First 3 layers are dense; the remaining 57 are sparse MoE.
        self.moe_layer_freq = (
            moe_layer_freq if moe_layer_freq is not None else [0] * 3 + [1] * 57
        )
        self.sparse_attention_config = (
            sparse_attention_config
            if sparse_attention_config is not None
            else {
                "use_sparse_attention": True,
                "sparse_index_dim": 128,
                "sparse_num_index_heads": 4,
                "sparse_topk_blocks": 16,
                "sparse_block_size": 128,
                "sparse_disable_index_value": [0] * 3 + [1] * 57,
                "sparse_score_type": "max",
                "sparse_init_block": 0,
                "sparse_local_block": 1,
                "sparse_attention_freq": [0] * 3 + [1] * 57,
            }
        )
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class MiniMaxM3MTPConfig(MiniMaxM3TextConfig):
    """Config for a standalone MiniMax M3 MTP (multi-token prediction) head.

    The MTP transformer layer is structurally a single MiniMax M3 decoder
    layer, so this reuses the text backbone schema. Standalone MTP checkpoints
    use ``model_type='minimax_m3_mtp'`` and a single hidden layer.
    """

    model_type = "minimax_m3_mtp"
    architectures = ["MiniMaxM3MTP"]

    def __init__(self, num_hidden_layers: int = 1, **kwargs):
        super().__init__(num_hidden_layers=num_hidden_layers, **kwargs)


class MiniMaxM3Config(PretrainedConfig):
    """Top-level MiniMax M3 (VL) config.

    Holds the text backbone as ``text_config`` so that
    ``config.get_text_config()`` extracts the MiniMaxM3SparseForCausalLM
    backbone. Vision components are kept as a raw dict passthrough and are
    not modeled here.
    """

    model_type = "minimax_m3_vl"

    def __init__(
        self,
        text_config: dict | MiniMaxM3TextConfig | None = None,
        vision_config: dict | None = None,
        **kwargs,
    ):
        if text_config is None:
            text_config = MiniMaxM3TextConfig()
        elif isinstance(text_config, dict):
            text_config = MiniMaxM3TextConfig(**text_config)
        self.text_config = text_config
        self.vision_config = vision_config

        self.hidden_size = text_config.hidden_size

        super().__init__(**kwargs)
