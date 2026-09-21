# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AliceAI model configuration."""

from typing import Any

from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig


class AliceAIConfig(Qwen3NextConfig):
    model_type = "alice_ai"
    attribute_map = {"num_nextn_predict_layers": "mtp_num_hidden_layers"}

    def __init__(
        self,
        vocab_size: int = 129024,
        max_position_embeddings: int = 262144,
        rope_theta: float = 1_000_000.0,
        linear_num_key_heads: int = 32,
        block_attn_res_block_size: int = 4,
        router_score_function: str = "sigmoid",
        router_bias_correction: bool = True,
        kda_allow_negative_eigenvalues: bool = False,
        mtp_num_hidden_layers: int = 1,
        number_of_conv_states: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            linear_num_key_heads=linear_num_key_heads,
            block_attn_res_block_size=block_attn_res_block_size,
            router_score_function=router_score_function,
            router_bias_correction=router_bias_correction,
            kda_allow_negative_eigenvalues=kda_allow_negative_eigenvalues,
            mtp_num_hidden_layers=mtp_num_hidden_layers,
            number_of_conv_states=number_of_conv_states,
            **kwargs,
        )


__all__ = ["AliceAIConfig"]
