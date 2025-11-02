# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_fbgemm_gpu_gen_ai

if current_platform.is_cuda_alike() and has_fbgemm_gpu_gen_ai():
    from fbgemm_gpu.experimental.gen_ai.moe import grouped_gemm, silu_mul


class MetaShufflingMoERoutedExperts:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
            cls.instance._initialized = False
        return cls.instance

    def __init__(
        self,
        quant_config: FusedMoEQuantConfig | None,
    ) -> None:
        if self._initialized:
            return
        self.quant_config = quant_config
        self._initialized: bool = True

    @staticmethod
    def _apply_activation(
        activation: str,
        x0: torch.Tensor,
        x1: torch.Tensor,
        valid_count: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if activation == "silu":
            return silu_mul(x0, x1, valid_token_count=valid_count)
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    def run(
        self,
        x: torch.Tensor,
        token_counts: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        activation: str,
        scores: torch.Tensor,  # scores,
        apply_router_weight_on_input: bool,
        num_valid_tokens: torch.Tensor | None = None,
        shared_out: torch.Tensor | None = None,
        token_indices: torch.Tensor | None = None,
    ):
        if x.shape[0] == 0:
            return x
        D = x.shape[-1]
        assert w1.shape[-1] == x.shape[-1]
        HD_L = w2.shape[-1]
        if envs.VLLM_META_SHUFFLING_GEMM_BACKEND == "cutlass":
            token_counts = token_counts.to(torch.int64)
            y = torch.ops.fbgemm.bf16bf16bf16_grouped_stacked(x, w1, token_counts)
        else:
            y = grouped_gemm(
                x,
                w1.view(-1, D),
                token_counts,
                use_fast_accum=True,
                _use_warp_specialization=False,
            )

        # TODO: Add support for scores multiplication after gemm
        z = self._apply_activation(
            activation, y[:, :HD_L], y[:, HD_L:], valid_count=num_valid_tokens
        )

        if envs.VLLM_META_SHUFFLING_GEMM_BACKEND == "cutlass":
            out = torch.ops.fbgemm.bf16bf16bf16_grouped_stacked(z, w2, token_counts)
        else:
            out = grouped_gemm(
                z,
                w2.view(-1, HD_L),
                token_counts,
                use_fast_accum=True,
                _use_warp_specialization=False,
                _output_tensor=shared_out,
                _scatter_add_indices=token_indices,
            )

        return out
