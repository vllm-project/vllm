# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_triton_kernels

logger = init_logger(__name__)

use_legacy_triton_kernels = False

if has_triton_kernels():
    try:
        import triton_kernels.swiglu
        from triton_kernels.matmul_ogs import (
            FnSpecs,
            FusedActivation,
            GatherIndx,
            RoutingData,
            ScatterIndx,
            matmul_ogs,
        )
        from triton_kernels.tensor import (
            BIT,
            Bitmatrix,
        )
        from triton_kernels.topk import topk

        if current_platform.is_rocm():
            try:
                from triton_kernels.tensor import (
                    SparseMatrix,
                    make_ragged_tensor_metadata,
                )
            except ImportError:
                logger.warning_once("Using legacy triton_kernels on ROCm")
                use_legacy_triton_kernels = True
    except (AttributeError, ImportError) as e:
        logger.error(
            "Failed to import Triton kernels. Please make sure your triton "
            "version is compatible. Error: %s",
            e,
        )


@triton.jit
def pack_bitmatrix(
    bitmatrix,
    topk_ids,
    n_rows,  # n_rows in bitmatrix / topk_ids
    bm_cols: tl.constexpr,  # n int32_t bitpacks in bitmatrix
    n_expts_act,  # num_topk
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Packs topk_ids into a bitmatrix.
    code reference:
    https://github.com/triton-lang/triton/blob/dd1bbc52b34d202dfe5ffea1e04fb16166c5c04e/python/triton_kernels/bench/distributed.py#L264
    """
    pid_m = tl.program_id(0)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    offsets = offsets_m[:, None] * n_expts_act + offsets_k[None, :]
    mask = (offsets_m < n_rows)[:, None] & (offsets_k < n_expts_act)[None, :]
    indices = tl.load(topk_ids + offsets, mask=mask, other=-1)
    div = indices // 32
    rem = indices % 32
    one = tl.cast(1, tl.uint32)

    # Iterate through all the relevant bitmatrix columns.
    for i in range(bm_cols):
        # When BLOCK_SIZE_K=32, offs is just the column index.
        offs = tl.arange(0, BLOCK_SIZE_K // 32) + i * (BLOCK_SIZE_K // 32)
        # All topks that need to go into this column has the correct bit set.
        # Other bits are 0. x is a 2D tensor.
        x = tl.where(
            div[:, :, None] == offs[None, None, :], (one << rem)[:, :, None], 0
        )
        # Reduce x to get a single int32_t bitpack.
        y = tl.reduce_or(x, axis=1)
        bitmatrix_ptrs = bitmatrix + offsets_m[:, None] * bm_cols + offs[None, :]
        tl.store(bitmatrix_ptrs, y, mask=offsets_m[:, None] < n_rows)


def legacy_routing_from_bitmatrix(
    bitmatrix: "Bitmatrix",
    expt_scal: torch.Tensor,
    expt_indx: torch.Tensor,
    n_expts_tot: int,
    n_expts_act: int,
) -> tuple["RoutingData", "GatherIndx", "ScatterIndx"]:
    """
    Replacement for the removed triton_kernels.routing.routing_from_bitmatrix.
    Creates routing data from a bitmatrix representation.
    """
    if use_legacy_triton_kernels:
        from triton_kernels.routing import routing_from_bitmatrix

        return routing_from_bitmatrix(
            bitmatrix, expt_scal, expt_indx, n_expts_tot, n_expts_act
        )
    sparse_logits = SparseMatrix(indx=expt_indx, vals=expt_scal, mask=bitmatrix)
    dispatch_indx = sparse_logits.mask_metadata.row_sorted_indx
    combine_indx = sparse_logits.mask_metadata.col_sorted_indx
    ragged_batch_metadata = make_ragged_tensor_metadata(
        sparse_logits.mask_metadata.col_sum,
        dispatch_indx.shape[0],
    )
    gate_scal = sparse_logits.vals.flatten()[combine_indx]
    routing_data = RoutingData(
        gate_scal,
        ragged_batch_metadata.block_sizes,
        n_expts_tot,
        n_expts_act,
        ragged_batch_metadata,
    )
    gather_idx = GatherIndx(combine_indx, dispatch_indx)
    scatter_idx = ScatterIndx(dispatch_indx, combine_indx)
    return routing_data, gather_idx, scatter_idx


def legacy_routing(
    logits: torch.Tensor,
    n_expts_act: int,
    sm_first: bool = False,
) -> tuple["RoutingData", "GatherIndx", "ScatterIndx"]:
    """
    Replacement for the removed triton_kernels.routing.routing function.
    Computes routing data from gating logits.
    """
    if use_legacy_triton_kernels:
        from triton_kernels.routing import routing

        return routing(logits, n_expts_act, sm_first=sm_first)
    if sm_first:
        logits = torch.softmax(logits, dim=-1)
    sparse_logits = topk(logits, n_expts_act, apply_softmax=not sm_first)
    return legacy_routing_from_bitmatrix(
        sparse_logits.mask,
        sparse_logits.vals,
        sparse_logits.indx,
        logits.shape[-1],
        n_expts_act,
    )


def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: str = "silu",
    quant_config: FusedMoEQuantConfig | None = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    routing_data, gather_idx, scatter_idx = legacy_routing(
        gating_output, topk, sm_first=not renormalize
    )

    output = torch.empty_like(hidden_states)

    return triton_kernel_fused_experts(
        output,
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_idx,
        scatter_idx,
        topk=topk,
        activation=activation,
        quant_config=quant_config,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )


# This is a triton implementation of the fused_experts function
def triton_kernel_fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    routing_data,  # RoutingData
    gather_indx,  # GatherIndx
    scatter_indx,  # ScatterIndx
    topk: int,
    activation: str = "silu",
    quant_config: FusedMoEQuantConfig | None = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    intermediate_cache: torch.Tensor | None = None,
    a1q_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if quant_config is None:
        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

    # type check, uint8 means mxfp4
    assert hidden_states.dtype == torch.bfloat16
    assert quant_config.w1_bias is None or quant_config.w1_bias.dtype == torch.float32
    assert quant_config.w2_bias is None or quant_config.w2_bias.dtype == torch.float32

    # Shape check, only check non-mxfp4
    assert hidden_states.ndim == 2
    assert hidden_states.shape[-1] == w1.shape[-2]
    assert w2.shape[-1] == w1.shape[1]

    batch_dim = 1
    M, K = hidden_states.shape[-2:]
    E, _, N = w1.shape

    if global_num_experts == -1:
        global_num_experts = E

    if intermediate_cache is None:
        intermediate_cache = torch.empty(
            (batch_dim, M * topk, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    # Add batch_dim to output buffer because matmul_ogs expects 3D output
    intermediate_cache = _resize_cache(
        intermediate_cache, (batch_dim, M * topk, N // 2)
    )
    output_tensor = _resize_cache(output_tensor, (batch_dim, M, K))

    act = (
        FusedActivation(
            FnSpecs(
                "swiglu",
                triton_kernels.swiglu.swiglu_fn,
                ("alpha", "limit"),
                reduction_n=2,
            ),
            (swiglu_alpha, swiglu_limit),
        )
        if not use_legacy_triton_kernels
        else FusedActivation(
            FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")),
            (swiglu_alpha, swiglu_limit),
            2,
        )
    )
    gammas = routing_data.gate_scal if routing_data else None

    matmul_ogs(
        hidden_states,
        w1,
        quant_config.w1_bias,
        routing_data,
        gather_indx=gather_indx,
        precision_config=quant_config.w1_precision,
        gammas=gammas if apply_router_weight_on_input else None,
        fused_activation=act,
        y=intermediate_cache,
    )

    matmul_ogs(
        intermediate_cache.view(M * topk, N // 2),
        w2,
        quant_config.w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        precision_config=quant_config.w2_precision,
        gammas=None if apply_router_weight_on_input else gammas,
        y=output_tensor,
    )
    output_tensor = output_tensor.view(M, K)
    return output_tensor


def make_routing_data(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_local_experts: int,
) -> tuple["RoutingData", torch.Tensor, torch.Tensor]:
    topk_ids = topk_ids.to(torch.int16)
    topk_weights = topk_weights.to(torch.bfloat16)

    n_rows, num_topk = topk_ids.size()

    BLOCK_SIZE_M = 512
    BLOCK_SIZE_K = 32

    bm_cols = triton.cdiv(num_local_experts, BLOCK_SIZE_K)  # n_bitpacks
    bitmatrix = torch.zeros(
        (n_rows, bm_cols), dtype=torch.uint32, device=topk_ids.device
    )

    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
    pack_bitmatrix[grid](
        bitmatrix,
        topk_ids,
        n_rows,
        bm_cols,
        num_topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    bitmatrix_shape = [n_rows, bm_cols * 32]
    bitmatrix_shape_max = [n_rows, None]
    bitmatrix = (
        Bitmatrix(
            bitmatrix, dtype=BIT, shape=bitmatrix_shape, shape_max=bitmatrix_shape_max
        )
        if not use_legacy_triton_kernels
        else Bitmatrix(
            bitmatrix,
            shape=bitmatrix_shape,
            shape_max=bitmatrix_shape_max,
            scratchpad=None,
        )
    )

    # matmul_ogs expects invalid topk_weights to be -1s
    topk_weights = torch.where(topk_ids == -1, -1.0, topk_weights)
    routing_data, gather_indx, scatter_indx = legacy_routing_from_bitmatrix(
        bitmatrix, topk_weights, topk_ids, num_local_experts, num_topk
    )

    return routing_data, gather_indx, scatter_indx


class BaseOAITritonExperts(mk.FusedMoEPermuteExpertsUnpermute):
    @staticmethod
    def _supports_current_device() -> bool:
        raise NotImplementedError(
            "OAITritonExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        raise NotImplementedError(
            "OAITritonExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        raise NotImplementedError(
            "OAITritonExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_activation(activation: str) -> bool:
        raise NotImplementedError(
            "OAITritonExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        raise NotImplementedError(
            "OAITritonExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    def supports_expert_map(self) -> bool:
        return True

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        """
        Extract the MoE problem size from the given tensor arguments:
        - a: The hidden states, input to the MoE layer.
        - w1: The first set of expert weights.
        - w2: The second set of expert weights.
        - topk_ids: The topk ids.
        Note: extracting the problem shape from the weight and activation
        tensors is not obvious.  It needs to be done this way specifically
        due to subtle issues with particular kernels, e.g. the int4 kernels
        divide the trailing dimension by two, so it's not "correct" to
        extract N or K from the trailing dimension of w1 or w2.  Similarly,
        some kernels transpose the weights, so this needs to be kept in mind.
        Note: This implementation covers most cases. However, if experts
        require a specialized implementation, like MarlinExperts, they are free
        to override this function.
        """
        assert w1.dim() == 3 and w2.dim() == 3
        E, _, N = w1.size()
        K = a1.size(-1)

        assert a1.dim() == 2
        assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
        M = a1.size(0)

        assert topk_ids.dim() == 2
        topk = topk_ids.size(1)

        return E, M, N, K, topk

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Weight application and reduction happens in the fused_experts kernel.
        return TopKWeightAndReduceNoOP()

    def _make_routing_data(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_local_experts: int,
    ) -> tuple["RoutingData", torch.Tensor, torch.Tensor]:
        return make_routing_data(topk_ids, topk_weights, num_local_experts)


class OAITritonExperts(BaseOAITritonExperts):
    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def supports_chunking(self) -> bool:
        return True

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # workspace are allocated inside the kernel
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (0, 0)
        workspace2 = (M * topk, activation_out_dim)
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if expert_map is not None:
            topk_ids = expert_map[topk_ids]

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        routing_data, gather_indx, scatter_indx = self._make_routing_data(
            topk_ids, topk_weights, local_num_experts
        )

        topk = topk_ids.size(1)
        triton_kernel_fused_experts(
            output,
            hidden_states,
            w1,
            w2,
            routing_data,
            gather_indx,
            scatter_indx,
            topk=topk,
            activation=activation,
            quant_config=self.quant_config,
            apply_router_weight_on_input=False,
            global_num_experts=local_num_experts,
            expert_map=None,  # applied already
            intermediate_cache=workspace2,
            a1q_scale=a1q_scale,
        )


class UnfusedOAITritonExperts(BaseOAITritonExperts):
    """
    A Triton based MoE expert class that operates on expert standard
    format and explicitly keeps the activation and reduction (moe_sum) steps
    unfused from the matmul_ogs kernel. This exposes injection points
    for activation and moe_sum.

    One use case for it is to inject LoRA modules on the activation and moe_sum.
    """

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def supports_chunking(self) -> bool:
        return True

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # workspace are allocated inside the kernel
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M * topk, activation_out_dim)
        workspace2 = (M * topk, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor):
        ops.moe_sum(input, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        # Use local variable to help mypy narrow the type after None check
        quant_config = self.quant_config
        if quant_config is None:
            quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

        if expert_map is not None:
            topk_ids = expert_map[topk_ids]

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        routing_data, gather_indx, scatter_indx = self._make_routing_data(
            topk_ids, topk_weights, local_num_experts
        )

        topk = topk_ids.size(1)

        # type check, uint8 means mxfp4
        assert hidden_states.dtype == torch.bfloat16
        assert (
            quant_config.w1_bias is None or quant_config.w1_bias.dtype == torch.float32
        )
        assert (
            quant_config.w2_bias is None or quant_config.w2_bias.dtype == torch.float32
        )

        # Shape check, only check non-mxfp4
        assert hidden_states.ndim == 2
        assert hidden_states.shape[-1] == w1.shape[-2]
        assert w2.shape[-1] == w1.shape[1]

        batch_dim = 1
        M, K = hidden_states.shape
        E, _, N = w1.shape

        if global_num_experts == -1:
            global_num_experts = E

        # Note that the output tensor might be in workspace13
        intermediate_cache1 = _resize_cache(workspace2, (batch_dim, M * topk, N))
        intermediate_cache3 = _resize_cache(workspace2, (batch_dim, M * topk, K))
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(workspace13, (M * topk, activation_out_dim))

        gammas = routing_data.gate_scal if routing_data else None

        matmul_ogs(
            hidden_states,
            w1,
            quant_config.w1_bias,
            routing_data,
            gather_indx=gather_indx,
            precision_config=quant_config.w1_precision,
            gammas=gammas if apply_router_weight_on_input else None,
            fused_activation=None,
            y=intermediate_cache1,
        )

        self.activation(
            activation,
            intermediate_cache2,
            intermediate_cache1.view(-1, N)[gather_indx.dst_indx],
        )

        # matmul_ogs grouped reduction fuse sum across multiple experts:
        # y[dst_indx // n_expts_act, :] += x
        # Need to set n_expts_act to 1 to unfuse moe_sum
        routing_data.n_expts_act = 1

        matmul_ogs(
            intermediate_cache2[gather_indx.src_indx],
            w2,
            quant_config.w2_bias,
            routing_data,
            scatter_indx=scatter_indx,
            precision_config=quant_config.w2_precision,
            gammas=None if apply_router_weight_on_input else gammas,
            y=intermediate_cache3,
        )

        self.moe_sum(intermediate_cache3.view(-1, topk, K), output)
