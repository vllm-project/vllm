# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native BF16 MoE implementation selected by ``--moe-backend hpc``."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform
from vllm.utils.hpc import has_hpc, hpc_fuse_moe_bf16


class HPCBf16Experts(mk.FusedMoEPermuteExpertsUnpermute):
    """Hopper BF16 SwiGLU MoE via the hpc-ops compatibility API."""

    # hpc-ops partitions one byte workspace into aligned routing, task-map,
    # TMA, and GEMM regions. Keep a conservative fixed reserve for metadata
    # and task maps after accounting for the tensor-shaped regions below.
    _WORKSPACE_FIXED_RESERVE_BYTES = 1 << 20

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        assert quant_config.weight_quant_dtype is None
        self.device = moe_config.device
        self.num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        self.ep_size = moe_config.moe_parallel_config.ep_size
        self.tp_rank = moe_config.moe_parallel_config.tp_rank
        self.tp_size = moe_config.moe_parallel_config.tp_size
        self.out_dtype = moe_config.in_dtype

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cuda()
            and current_platform.is_device_capability(90)
            and has_hpc()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (None, None)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return moe_parallel_config.dp_size == 1

    def supports_expert_map(self) -> bool:
        return False

    def supports_chunking(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

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
        rows = M * topk
        # N is the fused gate/up width (2 * intermediate size). The large
        # regions are gate input/output and down input/output. workspace2 is
        # allocated with the activation dtype and reinterpreted as uint8 in
        # apply(), so round bytes to BF16 elements here.
        tensor_region_bytes = rows * (4 * K + 3 * N + 64)
        metadata_reserve_bytes = (
            local_num_experts * 4096 + self._WORKSPACE_FIXED_RESERVE_BYTES
        )
        workspace_bytes = tensor_region_bytes + metadata_reserve_bytes
        workspace_elements = (workspace_bytes + 1) // 2
        return (M, K), (workspace_elements,), (M, K)

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
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        assert activation == MoEActivation.SILU
        assert expert_map is None
        assert a1q_scale is None and a2_scale is None
        assert workspace2 is not None
        # The vLLM workspace manager owns this allocation. Distinct live CUDA
        # Graphs therefore do not fall back to hpc-ops' process-global scratch
        # state, and sequential chunks may safely reuse the same range.
        assert workspace2.is_contiguous()
        workspace = workspace2.view(torch.uint8)
        hpc_fuse_moe_bf16(
            hidden_states,
            w1,
            w2,
            topk_ids,
            topk_weights,
            rank_ep=self.ep_rank,
            num_expert_total=global_num_experts,
            output=output,
            workspace=workspace,
        )
