# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash adapter for the DeepGEMM MegaMoE kernel."""

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    DeepGemmMegaMoEBackend,
    get_deep_gemm_mega_moe_backend,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MegaMoEExperts


class Glm5NextMegaMoEExperts(DeepseekV4MegaMoEExperts):
    """MegaMoE experts loaded from GLM-5.3-Flash's block-FP8 checkpoint
    (E4M3 weights, one float32 scale per ``weight_block_size`` block), run by
    the FP8xFP8 kernel."""

    def __init__(self, *args, weight_block_size: tuple[int, int], **kwargs):
        super().__init__(*args, **kwargs)
        block_m, block_k = weight_block_size
        e, h, i = self.num_local_experts, self.hidden_size, self.intermediate_size
        if h % block_m or h % block_k or i % block_m or i % block_k:
            raise ValueError(
                "GLM-5.3 MegaMoE requires hidden and intermediate sizes that are "
                f"multiples of the FP8 block size {weight_block_size}."
            )

        def param(shape: tuple[int, ...], dtype: torch.dtype) -> nn.Parameter:
            p = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
            set_weight_attrs(p, {"weight_loader": self.weight_loader})
            return p

        # The base class allocates MXFP4 loader parameters; this checkpoint's
        # are block FP8, under the names its expert mapping produces
        # (``experts.routed_experts.w13_weight`` / ``...w13_weight_scale_inv``).
        self.w13_weight = None
        self.w13_weight_scale = None
        self.w2_weight = None
        self.w2_weight_scale = None
        loader = nn.Module()
        loader.w13_weight = param((e, 2 * i, h), torch.float8_e4m3fn)
        loader.w13_weight_scale_inv = param(
            (e, 2 * i // block_m, h // block_k), torch.float32
        )
        loader.w2_weight = param((e, h, i), torch.float8_e4m3fn)
        loader.w2_weight_scale_inv = param(
            (e, h // block_m, i // block_k), torch.float32
        )
        self.routed_experts: nn.Module | None = loader

    def _ensure_backend(self) -> DeepGemmMegaMoEBackend:
        if self._backend is None:
            self._backend = get_deep_gemm_mega_moe_backend(
                torch.device("cuda", torch.accelerator.current_device_index()),
                self.hidden_size,
                self.intermediate_size,
                mma_type="fp8xfp8",
            )
        return self._backend

    def finalize_weights(self, shared_experts: nn.Module | None = None) -> None:
        loader = self.routed_experts
        if loader is not None:
            self.w13_weight = loader.w13_weight
            self.w13_weight_scale = loader.w13_weight_scale_inv
            self.w2_weight = loader.w2_weight
            self.w2_weight_scale = loader.w2_weight_scale_inv
            self.routed_experts = None
        super().finalize_weights(shared_experts)
