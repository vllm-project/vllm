# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import DeepseekV2Config, MiniMaxM2Config, OlmoeConfig


class DeepseekV2CompressedConfig(DeepseekV2Config):
    """Compatibility alias for legacy DeepSeek-V2 MoNE checkpoints."""

    model_type = "deepseek_v2_compressed"


class MiniMaxM2CompressedConfig(MiniMaxM2Config):
    """Compatibility alias for legacy MiniMax MoNE checkpoints."""

    model_type = "minimax_m2_compressed"

    def __init__(
        self,
        use_routing_bias: bool = True,
        scoring_func: str = "sigmoid",
        approximate_experts: dict | None = None,
        approximate_expert_init_tokens: dict | None = None,
        rotary_dim: int | None = None,
        partial_rotary_factor: float = 1.0,
        rope_theta: float = 5_000_000.0,
        **kwargs,
    ) -> None:
        head_dim = kwargs.get("head_dim", 128)
        if rotary_dim is None:
            rotary_dim = int(head_dim * partial_rotary_factor)
        else:
            partial_rotary_factor = rotary_dim / head_dim

        rope_parameters = dict(kwargs.pop("rope_parameters", None) or {})
        rope_parameters.setdefault("rope_type", "default")
        rope_parameters.setdefault("rope_theta", rope_theta)
        rope_parameters.setdefault("partial_rotary_factor", partial_rotary_factor)

        kwargs.setdefault("bos_token_id", 1)
        kwargs.setdefault("eos_token_id", 2)
        super().__init__(rope_parameters=rope_parameters, **kwargs)

        if approximate_experts is not None:
            approximate_experts = {
                int(layer): list(experts)
                for layer, experts in approximate_experts.items()
            }
        if approximate_expert_init_tokens is not None:
            approximate_expert_init_tokens = {
                int(layer): list(tokens)
                for layer, tokens in approximate_expert_init_tokens.items()
            }

        self.use_routing_bias = use_routing_bias
        self.scoring_func = scoring_func
        self.approximate_experts = approximate_experts
        self.approximate_expert_init_tokens = approximate_expert_init_tokens
        self.rotary_dim = rotary_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta
        self.num_experts = self.num_local_experts


class OlmoeCompressedConfig(OlmoeConfig):
    """Compatibility alias for legacy OLMoE MoNE checkpoints."""

    model_type = "olmoe_compressed"


__all__ = [
    "DeepseekV2CompressedConfig",
    "MiniMaxM2CompressedConfig",
    "OlmoeCompressedConfig",
]
