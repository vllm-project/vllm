# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.lora_experts_mixin import (
    LoRAExpertsMixin,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_triton_kernels

from ..utils import swiglu_limit_func

logger = init_logger(__name__)


def _triton_kernel_moe_supports_current_device() -> bool:
    # Shared device gate for the OAI Triton MoE expert classes.
    # Platform-aware to avoid ROCm capability aliasing — cap (9, 0)
    # matches both gfx90a (verified) and gfx906 (unverified), so we
    # dispatch on gfx-string helpers instead of the cap tuple on ROCm.
    p = current_platform
    if p.is_cuda():
        cap = p.get_device_capability()
        # Keep the original `(9, 0) <= cap < (11, 0)` window on
        # CUDA (covers Hopper SM90 and Blackwell SM100, excludes
        # SM120) — this PR is ROCm-scoped and the broader CUDA
        # range was not validated.
        return cap is not None and (9, 0) <= (cap.major, cap.minor) < (11, 0)
    if p.is_rocm():
        from vllm.platforms.rocm import on_gfx1x, on_gfx9

        # gfx9 family: gfx90a (MI200), gfx942/gfx950 (MI3xx);
        # on_gfx9() already excludes gfx906/gfx908.
        # gfx1x family: gfx11xx (RDNA3/3.5) and gfx12xx (RDNA4);
        # on_gfx1x() excludes gfx10xx (RDNA1/RDNA2).
        return on_gfx9() or on_gfx1x()
    return False


def _patch_make_bitmatrix_metadata() -> None:
    """Monkey-patch make_bitmatrix_metadata to support non-power-of-2 top_k.

    triton's tl.arange requires a power-of-2 range.  The original kernel
    computes BLOCK_SIZE = BLOCK_PER_TOK * TOKS_PER_ROW (= 32 * top_k).  For
    DeepSeek-V4 with top_k=6 this gives 192, which is not a power of 2 and
    causes a compile error at the first forward pass.

    Fix: define a drop-in replacement kernel that accepts an extra constexpr
    BLOCK_SIZE_PADDED (next power of 2 >= BLOCK_SIZE) and uses it for the
    tl.arange call while keeping the actual BLOCK_SIZE as the stride between
    thread-blocks so that all flat indices into NonzeroIndx stay correct.
    Elements beyond BLOCK_SIZE are masked out (col_indx = 0xffff) and ignored.

    This function is called once at module load time and patches the function
    inside the triton_kernels tensor module so that SparseMatrix.__post_init__
    picks up the fixed version transparently.
    """
    import torch
    import triton
    import triton.language as tl

    try:
        from vllm.third_party.triton_kernels.tensor_details import (
            bitmatrix as _bm,
        )
        from vllm.third_party.triton_kernels.tensor_details.bitmatrix import (
            BitmatrixMetadata,
            _keyed_add,
            cdiv,
        )
        from vllm.third_party.triton_kernels.tensor_details.bitmatrix_details.sum_bitmatrix_rows import (  # noqa: E501
            sum_bitmatrix_rows,
        )
    except ImportError:
        return

    @triton.jit
    def _stage2_pow2(
        ColSortedIndx,
        RowSortedIndx,
        NonzeroIndx,
        n_tokens,
        ColPartialSum,
        stride_pm,
        stride_pn,
        ColOffs,
        TOKS_PER_ROW: tl.constexpr,
        BLOCK_PER_TOK: tl.constexpr,
        BLOCK_SIZE_PADDED: tl.constexpr,
    ):
        # Actual number of elements per block (may not be a power of 2).
        BLOCK_SIZE: tl.constexpr = BLOCK_PER_TOK * TOKS_PER_ROW
        tl.static_assert(BLOCK_SIZE_PADDED <= 32768)
        if isinstance(n_tokens, tl.tensor) and n_tokens.dtype.is_ptr():
            n_tokens = tl.load(n_tokens)
        nonzero_indx_size = n_tokens * TOKS_PER_ROW
        pid_m = tl.program_id(0)
        # Use BLOCK_SIZE_PADDED (a power of 2) for tl.arange, but stride by
        # the actual BLOCK_SIZE so flat positions in NonzeroIndx are correct.
        # Elements with offs_local >= BLOCK_SIZE have offs_global beyond the
        # valid range, get col_indx = 0xffff, and are filtered by the mask
        # below without producing any output.
        offs_local = tl.arange(0, BLOCK_SIZE_PADDED)
        offs_global = pid_m * BLOCK_SIZE + offs_local
        mask = offs_global < nonzero_indx_size
        col_indx = tl.load(NonzeroIndx + offs_global, mask=mask, other=-1).to(tl.uint32)
        kv_pairs = ((col_indx << 16) | offs_local).to(tl.uint32)
        kv_pairs = tl.sort(kv_pairs, 0)
        col_indx = kv_pairs >> 16
        offs_global = pid_m * BLOCK_SIZE + (kv_pairs & 0xFFFF)
        mask = col_indx != 0xFFFF
        x = kv_pairs & 0xFFFF0000 | 0x00000001
        cols_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
        exclusive_run_lengths = (cols_and_inclusive_run_lengths - 1) & 0xFFFF
        row_sorted_indx = tl.load(
            ColPartialSum + pid_m * stride_pm + col_indx * stride_pn, mask=mask
        )
        row_sorted_indx += tl.load(ColOffs + col_indx, mask=mask)
        row_sorted_indx += exclusive_run_lengths
        tl.store(RowSortedIndx + offs_global, row_sorted_indx, mask=mask)
        tl.store(ColSortedIndx + row_sorted_indx, offs_global, mask=mask)

    def _make_bitmatrix_metadata_pow2_safe(nonzero_indx, bitmatrix):
        assert nonzero_indx.ndim == 2
        PARTIAL_BLOCK_M = 32
        col_sum, col_partial_sum = sum_bitmatrix_rows(
            bitmatrix, partials_block_size=PARTIAL_BLOCK_M
        )
        device = bitmatrix.device
        n_indx = nonzero_indx.numel()
        n_cols = bitmatrix.shape[1]
        col_offs = torch.empty(n_cols, dtype=torch.int32, device=device)
        combined_indx = torch.empty(n_indx * 2, dtype=torch.int32, device=device)
        col_sorted_indx = combined_indx[:n_indx]
        row_sorted_indx = combined_indx[n_indx:]
        MEMSET_BLOCK = 1024
        memset_grid = (cdiv(n_indx * 2, MEMSET_BLOCK) + n_cols + 1,)
        _bm._bitmatrix_metadata_compute_stage1[memset_grid](
            combined_indx,
            n_indx * 2,
            -1,
            MEMSET_BLOCK,
            col_sum,
            col_offs,
            col_sum.shape[0],
            col_partial_sum,
            col_partial_sum.shape[0],
            col_partial_sum.stride(0),
            col_partial_sum.stride(1),
            BLOCK_M=512,
            BLOCK_N=512,
        )
        toks_per_row = nonzero_indx.shape[-1]
        block_size = PARTIAL_BLOCK_M * toks_per_row
        # Next power of 2 >= block_size (required by tl.arange).
        block_size_padded = 1 << (max(block_size, 1) - 1).bit_length()
        compute_grid = (cdiv(bitmatrix.shape_max[0], PARTIAL_BLOCK_M),)
        _stage2_pow2[compute_grid](
            col_sorted_indx,
            row_sorted_indx,
            nonzero_indx,
            bitmatrix.shape[0],
            col_partial_sum,
            col_partial_sum.stride(0),
            col_partial_sum.stride(1),
            col_offs,
            TOKS_PER_ROW=toks_per_row,
            BLOCK_PER_TOK=PARTIAL_BLOCK_M,
            BLOCK_SIZE_PADDED=block_size_padded,
        )
        return BitmatrixMetadata(
            col_sum=col_sum,
            col_sorted_indx=col_sorted_indx,
            row_sorted_indx=row_sorted_indx,
        )

    # The most reliable patch point: SparseMatrix.__post_init__ looks up
    # make_bitmatrix_metadata via its own __globals__ dict (the tensor.py
    # module dict).  Patching through __globals__ works regardless of how
    # sys.modules maps "triton_kernels.tensor" vs
    # "vllm.third_party.triton_kernels.tensor".
    from triton_kernels.tensor import SparseMatrix as _SparseMatrix

    _SparseMatrix.__post_init__.__globals__["make_bitmatrix_metadata"] = (
        _make_bitmatrix_metadata_pow2_safe
    )
    # Also patch the bitmatrix module itself in case it is imported directly.
    _bm.make_bitmatrix_metadata = _make_bitmatrix_metadata_pow2_safe


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

        try:
            from triton_kernels.tensor import (
                SparseMatrix,
                make_ragged_tensor_metadata,
            )
        except ImportError:
            if current_platform.is_rocm():
                logger.warning_once("Using legacy triton_kernels on ROCm")
                use_legacy_triton_kernels = True
            else:
                raise
        if not use_legacy_triton_kernels:
            _patch_make_bitmatrix_metadata()
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
    valid = indices >= 0
    div = indices // 32
    rem = indices % 32
    one = tl.cast(1, tl.uint32)

    # Iterate through all the relevant bitmatrix columns.
    for i in range(bm_cols):
        # When BLOCK_SIZE_K=32, offs is just the column index.
        offs = tl.arange(0, BLOCK_SIZE_K // 32) + i * (BLOCK_SIZE_K // 32)
        # All topks that need to go into this column has the correct bit set.
        # Other bits are 0. x is a 2D tensor.
        # Guard with `valid` to prevent negative indices from producing
        # spurious bits (on HIP, -1 // 32 == 0 and 1 << (-1 % 32) sets
        # bit 31).
        x = tl.where(
            valid[:, :, None] & (div[:, :, None] == offs[None, None, :]),
            (one << rem)[:, :, None],
            0,
        )
        # Reduce x to get a single int32_t bitpack.
        y = tl.reduce_or(x, axis=1)
        bitmatrix_ptrs = bitmatrix + offsets_m[:, None] * bm_cols + offs[None, :]
        tl.store(bitmatrix_ptrs, y, mask=offsets_m[:, None] < n_rows)


def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: MoEActivation = MoEActivation.SWIGLUOAI,
    quant_config: FusedMoEQuantConfig | None = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    unpadded_N_w1=None,
    unpadded_K_w1=None,
    unpadded_N_w2=None,
    unpadded_K_w2=None,
) -> torch.Tensor:
    from triton_kernels.topk import topk as topk_fn

    sm_first = not renormalize
    logits = gating_output
    if sm_first:
        logits = torch.softmax(logits, dim=-1)
    topk_result = topk_fn(logits, topk, apply_softmax=not sm_first)
    # topk may return a tuple (vals, indx, bitmatrix) or a
    # SparseMatrix depending on the triton_kernels version.
    if isinstance(topk_result, tuple):
        topk_weights, topk_ids_raw, _ = topk_result
    else:
        topk_weights = topk_result.vals
        topk_ids_raw = topk_result.indx

    if expert_map is not None:
        # topk_ids_raw contains global expert IDs - remap to local.
        topk_ids = expert_map[topk_ids_raw.to(torch.long)]
        local_num_experts = w1.shape[0]
        routing_data, gather_idx, scatter_idx = make_routing_data(
            topk_ids, topk_weights, local_num_experts
        )
        # expert_map already applied; pass None downstream.
        effective_expert_map = None
        effective_global_num_experts = local_num_experts
    else:
        topk_ids = topk_ids_raw.to(torch.long)
        routing_data, gather_idx, scatter_idx = make_routing_data(
            topk_ids, topk_weights, gating_output.shape[-1]
        )
        effective_expert_map = expert_map
        effective_global_num_experts = global_num_experts

    output = torch.empty_like(hidden_states)
    effective_quant_config = (
        quant_config if quant_config is not None else FUSED_MOE_UNQUANTIZED_CONFIG
    )

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
        quant_config=effective_quant_config,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=effective_global_num_experts,
        expert_map=effective_expert_map,
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
    activation: MoEActivation = MoEActivation.SWIGLUOAI,
    quant_config: FusedMoEQuantConfig | None = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    intermediate_cache: torch.Tensor | None = None,
    a1q_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton implementation of fused expert computation using OAI kernels."""
    assert activation == MoEActivation.SWIGLUOAI, (
        "Only SWIGLUOAI activation is supported"
    )
    assert quant_config is not None

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

    if use_legacy_triton_kernels:
        from triton_kernels.routing import routing_from_bitmatrix

        return routing_from_bitmatrix(
            bitmatrix, topk_weights, topk_ids, num_local_experts, num_topk
        )

    sparse_logits = SparseMatrix(indx=topk_ids, vals=topk_weights, mask=bitmatrix)
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
        num_local_experts,
        num_topk,
        ragged_batch_metadata,
    )
    gather_indx = GatherIndx(combine_indx, dispatch_indx)
    scatter_indx = ScatterIndx(dispatch_indx, combine_indx)
    return routing_data, gather_indx, scatter_indx


class BaseOAITritonExperts(mk.FusedMoEExpertsModular):
    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        return _triton_kernel_moe_supports_current_device() and has_triton_kernels()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, None),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        raise NotImplementedError

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

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
        assert len(w1.shape) == 3 and len(w2.shape) == 3
        E, _, N = w1.shape
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
    """OAI Triton-based fused MoE expert implementation."""

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SWIGLUOAI

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

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
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if self.quant_config is None:
            self.quant_config: FusedMoEQuantConfig = FUSED_MOE_UNQUANTIZED_CONFIG

        if expert_map is not None:
            topk_ids = expert_map[topk_ids]

        local_num_experts = w1.shape[0]
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


class UnfusedOAITritonExperts(LoRAExpertsMixin, BaseOAITritonExperts):
    """
    A Triton based MoE expert class that operates on expert standard
    format and explicitly keeps the activation and reduction (moe_sum) steps
    unfused from the matmul_ogs kernel. This exposes injection points
    for activation and moe_sum.

    One use case for it is to inject LoRA modules on the activation and moe_sum.
    """

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUSTEP,
        ]

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

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
        # workspace are allocated inside the kernel
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M * topk, activation_out_dim)
        workspace2 = (M * topk, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor):
        ops.moe_sum(input, output)

    def activation(
        self,
        activation: MoEActivation,
        output: torch.Tensor,
        input: torch.Tensor,
    ) -> None:
        quant_config = self.quant_config or FUSED_MOE_UNQUANTIZED_CONFIG
        if activation == MoEActivation.SWIGLUOAI:
            alpha = (
                quant_config.gemm1_alpha
                if quant_config.gemm1_alpha is not None
                else 1.702
            )
            limit = (
                quant_config.gemm1_clamp_limit
                if quant_config.gemm1_clamp_limit is not None
                else 7.0
            )
            torch.ops._C.swigluoai_and_mul(output, input, alpha, limit)
        elif (
            activation == MoEActivation.SILU
            and quant_config.gemm1_clamp_limit is not None
        ):
            swiglu_limit_func(
                output,
                input,
                quant_config.gemm1_clamp_limit,
            )
        else:
            super().activation(activation, output, input)

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
    ):
        # Use local variable to help mypy narrow the type after None check
        quant_config = self.quant_config
        if quant_config is None:
            quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

        global_topk_ids = topk_ids
        if expert_map is not None:
            topk_ids = expert_map[topk_ids]

        local_num_experts = w1.shape[0]
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

        # w13 LoRA: gather the activation input from expert-sorted
        # intermediate_cache1, then add the LoRA delta in-place on that copy
        # before passing it to activation — exactly mirroring the old
        # decorator approach which modified the gathered tensor in-place.
        act_input = intermediate_cache1.view(-1, N)[gather_indx.dst_indx]

        sorted_token_ids_lora = None
        expert_ids_lora = None
        num_tokens_post_padded_lora = None
        token_lora_mapping = None
        lora_context = self._lora_context
        if lora_context is not None:
            (
                sorted_token_ids_lora,
                expert_ids_lora,
                num_tokens_post_padded_lora,
                token_lora_mapping,
            ) = self.apply_w13_lora(
                lora_context,
                y=act_input,
                x=hidden_states,
                topk_ids=global_topk_ids,
                topk_weights=topk_weights,
                expert_map=expert_map,
                w1=w1,
                w2=w2,
                num_tokens=M,
                top_k_num=topk,
            )

        self.activation(
            activation,
            intermediate_cache2,
            act_input,
        )

        # matmul_ogs grouped reduction fuses sum across multiple experts:
        # y[dst_indx // n_expts_act, :] += x
        # Set n_expts_act to 1 to unfuse the sum so we can do it manually via moe_sum.
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

        # w2 LoRA: after matmul_ogs with scatter_indx, intermediate_cache3 is
        # in token-topk order, matching the (M, topk, K) layout add_lora_w2 expects.
        if lora_context is not None:
            self.apply_w2_lora(
                lora_context,
                y=intermediate_cache3.view(-1, topk, K),
                x=intermediate_cache2,
                topk_weights=topk_weights,
                sorted_token_ids_lora=sorted_token_ids_lora,
                expert_ids_lora=expert_ids_lora,
                num_tokens_post_padded_lora=num_tokens_post_padded_lora,
                token_lora_mapping=token_lora_mapping,
                num_tokens=M,
                w1=w1,
                w2=w2,
                top_k_num=topk,
            )

        self.moe_sum(intermediate_cache3.view(-1, topk, K), output)


class OAITritonMxfp4ExpertsMonolithic(mk.FusedMoEExpertsMonolithic):
    """Monolithic Triton MXFP4 expert. Wraps triton_kernel_moe_forward()."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.topk = moe_config.experts_per_token
        self.renormalize = moe_config.routing_method in (
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return _triton_kernel_moe_supports_current_device() and has_triton_kernels()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, None),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SWIGLUOAI

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
            and moe_parallel_config.dp_size <= 1
        )

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        return triton_kernel_moe_forward(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            gating_output=router_logits,
            topk=self.topk,
            renormalize=self.renormalize,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            quant_config=self.quant_config,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
