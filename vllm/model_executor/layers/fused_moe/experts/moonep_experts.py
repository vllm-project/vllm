# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoonEP experts: grouped GEMM over MoonEP's expert-grouped activations.

MoonEP dispatch delivers tokens already contiguous per weight row
(``[NvS, H]`` in ``cu_seqlens[E+B]`` segment order), so the expert compute
is three grouped GEMMs over those segments with no permute/unpermute:

    gate = grouped_mm(x, w_gate[E+B])      # [NvS, I]
    up   = grouped_mm(x, w_up[E+B])        # [NvS, I]
    act  = silu(gate) * up * route_weight  # route weights applied here
    out  = grouped_mm(act, w_down[E+B])    # [NvS, H]

Rows ``[E, E+B)`` are the redundant-expert prefetch slots that
``MoonEPPrepareAndFinalize`` fills before ``apply`` runs. Empty segments
(including unused prefetch slots) are skipped by the grouped GEMM.

Weight layout: MoonEP's ``prefetch_weight`` requires each projection to be
its own contiguous ``[E+B, ., .]`` tensor, so gate and up are separate
tensors rather than vLLM's interleaved ``[E, 2I, H]`` ``w13``. The modular
kernel passes ``w1``/``w2`` from the layer; ``w1`` is the gate tensor and
the up tensor is supplied via ``set_up_weight``.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform


def moonep_grouped_gemm(
    a: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """``out[seg_g] = a[seg_g] @ w[g].T`` for each ``cu_seqlens`` segment.

    Args:
        a: ``[NvS, K]`` activations in segment order.
        w: ``[G, N, K]`` weights (vLLM ``[E, N, K]`` convention).
        cu_seqlens: ``[G]`` int32 cumulative segment end offsets.

    Returns:
        ``[NvS, N]``; rows past ``cu_seqlens[-1]`` are zero-filled.
    """
    return torch._grouped_mm(a, w.transpose(1, 2), offs=cu_seqlens)


class MoonEPExperts(mk.FusedMoEExpertsModular):
    """Grouped-GEMM experts over MoonEP's ``[NvS, H]`` layout (BF16)."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(moe_config, quant_config, max_num_tokens, num_dispatchers)
        assert quant_config.quant_dtype is None, (
            "MoonEPExperts supports unquantized BF16 only"
        )
        self._up_weight: torch.Tensor | None = None

    def set_up_weight(self, up_weight: torch.Tensor) -> None:
        """Attach the ``[E+B, I, H]`` up-projection weight tensor."""
        assert up_weight.ndim == 3 and up_weight.is_contiguous()
        self._up_weight = up_weight

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return moe_parallel_config.use_moonep_kernels

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key is None and activation_key is None

    def supports_chunking(self) -> bool:
        # NvS is static and segments must stay whole.
        return False

    def supports_expert_map(self) -> bool:
        # MoonEP addresses global expert rows directly.
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        # a1 is [NvS, H] in segment order, not token-major, so the base
        # implementation's topk_ids/a1 row-count check does not apply.
        assert a1.dim() == 2 and w1.dim() == 3 and w2.dim() == 3
        num_rows, intermediate, hidden = w1.shape  # [E+B, I, H]
        assert a1.size(1) == hidden
        return num_rows, a1.size(0), intermediate, hidden, topk_ids.size(1)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # M is NvS. torch._grouped_mm allocates its own outputs, so no
        # workspaces are needed.
        return ((0,), (0,), (M, K))

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        # expert_map is ignored: MoonEP dispatches on global expert ids and
        # the weights are addressed by global [E+B] row, on every rank.
        assert not apply_router_weight_on_input
        assert activation == MoEActivation.SILU
        assert self._up_weight is not None, "set_up_weight() not called"
        assert expert_tokens_meta is not None
        # MoonEPPrepareAndFinalize returns route weights in NvS order as the
        # dispatched topk_weights, and per-row segment lengths as
        # expert_tokens_meta.
        route_weights_nvs = topk_weights
        assert route_weights_nvs.dim() == 1
        cu_seqlens = torch.cumsum(
            expert_tokens_meta.expert_num_tokens, dim=0, dtype=torch.int32
        )

        assert cu_seqlens.numel() == w1.size(0)

        gate = moonep_grouped_gemm(hidden_states, w1, cu_seqlens)
        up = moonep_grouped_gemm(hidden_states, self._up_weight, cu_seqlens)
        act = torch.nn.functional.silu(gate)
        act.mul_(up)
        act.mul_(route_weights_nvs.to(act.dtype).unsqueeze(-1))
        # Padding rows past the last segment come out zero-filled.
        output.copy_(moonep_grouped_gemm(act, w2, cu_seqlens))
