# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.gdn.kda_linear_attn import KDAAttentionBase
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader


def _ya_sharded_weight_loader(
    shard_axis: int,
    model_dtype: torch.dtype,
    *,
    reshape_a_log: bool = False,
):
    weight_loader = sharded_weight_loader(shard_axis)

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        if reshape_a_log and loaded_weight.ndim == 1:
            loaded_weight = loaded_weight.view(1, 1, -1, 1)
        loaded_weight = loaded_weight.to(model_dtype).float()
        weight_loader(param, loaded_weight)

    return loader


@PluggableLayer.register("ya_gated_delta_net_attention")
class YAGatedDeltaNetAttention(KDAAttentionBase):
    def __init__(
        self,
        config: PretrainedConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        *,
        reduce_results: bool = True,
    ) -> None:
        num_heads = config.linear_num_key_heads  # type: ignore[attr-defined]
        num_value_heads = config.linear_num_value_heads  # type: ignore[attr-defined]
        head_dim = config.linear_key_head_dim  # type: ignore[attr-defined]
        value_head_dim = config.linear_value_head_dim  # type: ignore[attr-defined]
        conv_size = config.linear_conv_kernel_dim  # type: ignore[attr-defined]
        allow_neg_eigval = getattr(
            config,
            "kda_allow_neg_eigval",
            getattr(config, "allow_neg_eigval", False),
        )

        if num_heads != num_value_heads:
            raise ValueError(
                "KDA requires linear_num_key_heads == linear_num_value_heads, "
                f"got {num_heads} and {num_value_heads}"
            )
        if head_dim != value_head_dim:
            raise ValueError(
                "KDA requires linear_key_head_dim == linear_value_head_dim, "
                f"got {head_dim} and {value_head_dim}"
            )

        speculative_config = vllm_config.speculative_config
        if speculative_config and speculative_config.num_speculative_tokens > 0:
            raise NotImplementedError(
                "YAGatedDeltaNetAttention does not support speculative decoding"
            )

        model_dtype = vllm_config.model_config.dtype
        super().__init__(
            config,
            vllm_config,
            prefix,
            num_heads=num_heads,
            head_dim=head_dim,
            conv_size=conv_size,
            conv_params_dtype=model_dtype,
            norm_eps=config.rms_norm_eps,  # type: ignore[attr-defined]
            norm_dtype=model_dtype,
            allow_neg_eigval=allow_neg_eigval,
            reduce_results=reduce_results,
            dt_bias_weight_loader=_ya_sharded_weight_loader(
                0,
                model_dtype,
            ),
            a_log_weight_loader=_ya_sharded_weight_loader(
                2,
                model_dtype,
                reshape_a_log=True,
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._forward_kda(hidden_states)
