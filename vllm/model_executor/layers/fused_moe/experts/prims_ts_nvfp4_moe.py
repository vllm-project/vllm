# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsModular,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_prims_ts_fused_moe

# The architectures FlashInfer builds Prims-TS MoE configs for.
_PRIMS_TS_DEVICE_CAPABILITIES = (100, 103, 107)

# Any batch compiles the configurations of every token count; keep it small.
_PRECOMPILE_NUM_TOKENS = 16


class PrimsTsNvFp4ExpertsModular(TrtLlmNvFp4ExpertsModular):
    """NvFp4 MoE with FlashInfer's Prims-TS (CuTe DSL) expert GEMMs.

    Routing and finalize stay on TRTLLM-Gen, and the op takes the arguments and
    the shuffled weight layout of `trtllm_fp4_block_scale_routed_moe`, so
    only the op differs from TrtLlmNvFp4ExpertsModular.
    """

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        capability = p.get_device_capability()
        return (
            p.is_cuda()
            and capability is not None
            and capability.to_int() in _PRIMS_TS_DEVICE_CAPABILITIES
            and has_flashinfer_prims_ts_fused_moe()
        )

    @staticmethod
    def _routed_moe(**kwargs):
        import flashinfer

        return flashinfer.fused_moe.prims_ts_fp4_block_scale_routed_moe(**kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        # Read by prims_ts_moe_warmup.
        layer.prims_ts_moe_warmup_provider = self

    def untuned_precompile_kwargs(self, layer: torch.nn.Module) -> dict[str, Any]:
        """The arguments of `precompile_untuned` for `layer`'s weights."""
        device = layer.w13_weight.device
        num_tokens = _PRECOMPILE_NUM_TOKENS
        if self.expects_unquantized_inputs:
            hidden_states = torch.ones(
                (num_tokens, self.hidden_dim), dtype=torch.bfloat16, device=device
            )
            a1q_scale = None
        else:
            hidden_states = torch.zeros(
                (num_tokens, self.hidden_dim // 2), dtype=torch.uint8, device=device
            )
            a1q_scale = torch.zeros(
                (num_tokens, self.hidden_dim // 16), dtype=torch.uint8, device=device
            ).view(torch.float8_e4m3fn)
        topk_ids = torch.arange(num_tokens * self.topk, device=device).view(
            num_tokens, self.topk
        )
        return self._routed_moe_kwargs(
            hidden_states=hidden_states,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=torch.full(
                (num_tokens, self.topk), 1 / self.topk, device=device
            ),
            topk_ids=(topk_ids % layer.global_num_experts).to(torch.int32),
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            a1q_scale=a1q_scale,
        )

    @staticmethod
    def precompile_untuned(kwargs: dict[str, Any]) -> int:
        """Compile the expert GEMMs that untuned launches with `kwargs`'s
        weights and settings use at any token count. Returns how many FC1/FC2
        configurations that is."""
        import flashinfer

        return flashinfer.fused_moe.prims_ts_fp4_block_scale_routed_moe_precompile(
            **kwargs
        )
