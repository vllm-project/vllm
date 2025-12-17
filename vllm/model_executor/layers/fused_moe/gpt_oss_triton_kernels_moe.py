# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_triton_kernels

logger = init_logger(__name__)

if has_triton_kernels():
    try:
        import triton_kernels.swiglu
        from triton_kernels.matmul_ogs import FnSpecs, FusedActivation, matmul_ogs
        from triton_kernels.routing import (
            ExptData,
            GatherIndx,
            RoutingData,
            ScatterIndx,
            routing_from_bitmatrix,
        )
        from triton_kernels.tensor import Bitmatrix
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

    for i in range(bm_cols):
        offs = tl.arange(0, BLOCK_SIZE_K // 32) + i * (BLOCK_SIZE_K // 32)
        x = tl.where(
            div[:, :, None] == offs[None, None, :], (one << rem)[:, :, None], 0
        )
        y = tl.reduce_or(x, axis=1)
        bitmatrix_ptrs = bitmatrix + offsets_m[:, None] * bm_cols + offs[None, :]
        tl.store(bitmatrix_ptrs, y, mask=offsets_m[:, None] < n_rows)


@triton.jit
def _global_barrier_cas(
    barrier_ptr,
    num_programs: tl.constexpr,
    phase: tl.constexpr,
):
    counter_ptr = barrier_ptr + phase

    # each pid increments counter when it arrives
    arrived = tl.atomic_add(counter_ptr, 1, sem="acq_rel")

    if arrived == num_programs - 1:
        # last pid to arrive resets the counter
        tl.atomic_xchg(counter_ptr, 0, sem="release")
    else:
        while tl.atomic_cas(counter_ptr, 0, 0) != 0:
            pass


@triton.jit
def _cdiv(n, d):
    return (n + d - 1) // d


@triton.jit
def _fused_topk_softmax_routing_kernel(
    logits_ptr,
    stride_logits_m,
    stride_logits_n,
    weights_ptr,
    stride_weights_m,
    stride_weights_k,
    indices_ptr,
    stride_indices_m,
    stride_indices_k,
    hist_ptr,  # [N] global histogram
    expt_offs_ptr,  # [N] prefix sum of histogram (token_offs_raw)
    partial_hist_ptr,  # [num_programs, N] per-program histogram
    stride_partial_m,
    stride_partial_n,
    gate_scal_ptr,  # [M * topk] reordered weights
    topk_index_ptr,  # [M * topk] gather indices
    gate_index_ptr,  # [M * topk] scatter indices
    token_offs_pad_ptr,  # [NUM_BLOCK_SIZES, N+1] padded offsets for each block_m
    stride_token_offs_m,
    stride_token_offs_n,
    block_pid_map_ptr,  # [NUM_BLOCK_SIZES, max_n_tiles] pid mapping
    stride_pid_map_m,
    stride_pid_map_n,
    max_n_tiles,
    barrier_ptr,  # [2] barrier counters
    M,  # num_tokens (can be dynamic)
    N: tl.constexpr,  # num_experts
    topk: tl.constexpr,
    num_programs: tl.constexpr,
    RENORM: tl.constexpr,
    ROWS_PER_PID: tl.constexpr,
    NUM_BLOCK_SIZES: tl.constexpr,  # number of block_m sizes (4 or 5)
    BLOCK_M_LOG2_START: tl.constexpr,  # log2 of smallest block_m (4 for 16)
):
    """
    Fully fused kernel combining:
    - Phase 1: topk + softmax + hist accumulation
    - Phase 2: prefixsum + ExptData init (single program)
    - Phase 3: gather/scatter indices + block_pid_map
    """
    pid = tl.program_id(0)

    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, topk)

    local_hist = tl.zeros([N], dtype=tl.int32)

    row_start = pid * ROWS_PER_PID
    row_end = tl.minimum(row_start + ROWS_PER_PID, M)

    for row_idx in range(row_start, row_end):
        logits_row_ptr = logits_ptr + row_idx * stride_logits_m
        logits = tl.load(logits_row_ptr + offs_n * stride_logits_n)

        if not RENORM:
            logits = tl.softmax(logits, dim=0)

        topk_vals = tl.full([topk], float("-inf"), dtype=tl.float32)
        topk_idxs = tl.zeros([topk], dtype=tl.int32)

        for k in tl.static_range(topk):
            cur_max = tl.max(logits, axis=0)
            cur_idx = tl.argmax(logits, axis=0)

            topk_vals = tl.where(offs_k == k, cur_max, topk_vals)
            topk_idxs = tl.where(offs_k == k, cur_idx, topk_idxs)

            mask_selected = offs_n == cur_idx
            logits = tl.where(mask_selected, float("-inf"), logits)

        if RENORM:
            topk_vals = tl.softmax(topk_vals, dim=0)

        weights_row_ptr = weights_ptr + row_idx * stride_weights_m
        indices_row_ptr = indices_ptr + row_idx * stride_indices_m
        tl.store(weights_row_ptr + offs_k * stride_weights_k, topk_vals)
        tl.store(indices_row_ptr + offs_k * stride_indices_k, topk_idxs.to(tl.int16))

        for k in tl.static_range(topk):
            expert_id = tl.sum(tl.where(offs_k == k, topk_idxs, 0))
            tl.atomic_add(hist_ptr + expert_id, 1, sem="relaxed")
            local_hist = tl.where(offs_n == expert_id, local_hist + 1, local_hist)

    partial_hist_row_ptr = partial_hist_ptr + pid * stride_partial_m
    tl.store(partial_hist_row_ptr + offs_n * stride_partial_n, local_hist)

    _global_barrier_cas(barrier_ptr, num_programs, phase=0)

    if pid == 0:
        hist = tl.load(hist_ptr + offs_n)

        cumsum = 0
        for i in tl.static_range(N):
            h = tl.sum(tl.where(offs_n == i, hist, 0))
            tl.store(expt_offs_ptr + i, cumsum)
            cumsum = cumsum + h
        tl.store(expt_offs_ptr + N, cumsum)

        for size_idx in tl.static_range(NUM_BLOCK_SIZES):
            block_m_log2 = (
                BLOCK_M_LOG2_START + size_idx
            )  # 4, 5, 6, 7 (for 16, 32, 64, 128)
            block_m = 1 << block_m_log2

            cumsum_pad = 0
            for i in tl.static_range(N):
                h = tl.sum(tl.where(offs_n == i, hist, 0))
                n_tiles = (h + block_m - 1) // block_m  # cdiv
                offs_pad_ptr = token_offs_pad_ptr + size_idx * stride_token_offs_m
                tl.store(offs_pad_ptr + i * stride_token_offs_n, cumsum_pad)
                cumsum_pad = cumsum_pad + n_tiles
            offs_pad_ptr = token_offs_pad_ptr + size_idx * stride_token_offs_m
            tl.store(offs_pad_ptr + N * stride_token_offs_n, cumsum_pad)

        for size_idx in tl.static_range(NUM_BLOCK_SIZES):
            pid_map_row_ptr = block_pid_map_ptr + size_idx * stride_pid_map_m
            for tile_idx in range(max_n_tiles):
                tl.store(pid_map_row_ptr + tile_idx * stride_pid_map_n, -1)

        for size_idx in tl.static_range(NUM_BLOCK_SIZES):
            block_m_log2 = BLOCK_M_LOG2_START + size_idx
            block_m = 1 << block_m_log2

            offs_pad_ptr = token_offs_pad_ptr + size_idx * stride_token_offs_m
            pid_map_row_ptr = block_pid_map_ptr + size_idx * stride_pid_map_m

            for expert_id in range(N):
                h = tl.load(hist_ptr + expert_id)
                n_tiles = (h + block_m - 1) // block_m
                tile_start = tl.load(offs_pad_ptr + expert_id * stride_token_offs_n)

                for block_idx in range(n_tiles):
                    # Pack (block_idx << 16) | expert_id
                    packed_val = (block_idx << 16) | expert_id
                    tl.store(
                        pid_map_row_ptr + (tile_start + block_idx) * stride_pid_map_n,
                        packed_val,
                    )

    _global_barrier_cas(barrier_ptr, num_programs, phase=1)

    expt_offs = tl.load(expt_offs_ptr + offs_n)

    prior_contrib = tl.zeros([N], dtype=tl.int32)
    for p in range(pid):
        prior_partial_ptr = partial_hist_ptr + p * stride_partial_m
        prior_hist = tl.load(prior_partial_ptr + offs_n * stride_partial_n)
        prior_contrib = prior_contrib + prior_hist

    local_offset = tl.zeros([N], dtype=tl.int32)

    for row_idx in range(row_start, row_end):
        weights_row_ptr = weights_ptr + row_idx * stride_weights_m
        indices_row_ptr = indices_ptr + row_idx * stride_indices_m
        weights = tl.load(weights_row_ptr + offs_k * stride_weights_k)
        expert_ids = tl.load(indices_row_ptr + offs_k * stride_indices_k).to(tl.int32)

        for k in tl.static_range(topk):
            expert_id = tl.sum(tl.where(offs_k == k, expert_ids, 0))
            weight_val = tl.sum(tl.where(offs_k == k, weights, 0.0))
            flat_idx = row_idx * topk + k

            expert_base = tl.sum(tl.where(offs_n == expert_id, expt_offs, 0))
            expert_prior = tl.sum(tl.where(offs_n == expert_id, prior_contrib, 0))
            expert_local = tl.sum(tl.where(offs_n == expert_id, local_offset, 0))

            global_pos = expert_base + expert_prior + expert_local

            tl.store(gate_scal_ptr + global_pos, weight_val)
            tl.store(topk_index_ptr + global_pos, flat_idx)
            tl.store(gate_index_ptr + flat_idx, global_pos)

            local_offset = tl.where(offs_n == expert_id, local_offset + 1, local_offset)


def fused_routing(
    router_logits: torch.Tensor,
    topk: int,
    renormalize: bool = True,
) -> tuple["RoutingData", "GatherIndx", "ScatterIndx"]:
    """
    Fully fused topk, softmax, routing, and ExptData computation.
    """
    M, N = router_logits.shape
    device = router_logits.device
    dtype = router_logits.dtype

    # Validate constraints
    BLOCK_N = triton.next_power_of_2(N)
    topk_padded = triton.next_power_of_2(topk)
    assert (BLOCK_N == N) and (topk_padded == topk), (
        f"N and topk must be power of 2, got N={N}, topk={topk}"
    )

    weights = torch.empty((M, topk), device=device, dtype=dtype)
    indices = torch.empty((M, topk), device=device, dtype=torch.int16)
    hist = torch.zeros(N, device=device, dtype=torch.int32)
    expt_offs = torch.empty(N + 1, device=device, dtype=torch.int32)

    n_gates = M * topk
    gate_scal = torch.empty(n_gates, device=device, dtype=dtype)
    topk_index = torch.empty(n_gates, device=device, dtype=torch.int32)
    gate_index = torch.empty(n_gates, device=device, dtype=torch.int32)

    # block_m sizes: 16, 32, 64, 128 (NUM_BLOCK_SIZES=4)
    BLOCK_M_LOG2_START = 4
    NUM_BLOCK_SIZES = 4

    if n_gates <= N:
        max_n_tiles = n_gates
    else:
        min_block_m = 1 << BLOCK_M_LOG2_START  # 16
        max_n_tiles = N - 1 - ((N - n_gates - 1) // min_block_m)

    token_offs_pad = torch.empty(
        (NUM_BLOCK_SIZES, N + 1), device=device, dtype=torch.int32
    )
    block_pid_map = torch.full(
        (NUM_BLOCK_SIZES, max_n_tiles), -1, device=device, dtype=torch.int32
    )

    barrier = torch.zeros(2, device=device, dtype=torch.int32)

    device_props = torch.cuda.get_device_properties(device)
    num_sms = device_props.multi_processor_count

    max_num_programs = 64
    ROWS_PER_PID = 4
    desired_programs = triton.cdiv(M, ROWS_PER_PID)
    num_programs = min(desired_programs, num_sms, max_num_programs)
    ROWS_PER_PID = triton.cdiv(M, num_programs)

    partial_hist = torch.zeros((num_programs, N), device=device, dtype=torch.int32)

    _fused_topk_softmax_routing_kernel[(num_programs,)](
        logits_ptr=router_logits,
        stride_logits_m=router_logits.stride(0),
        stride_logits_n=router_logits.stride(1),
        weights_ptr=weights,
        stride_weights_m=weights.stride(0),
        stride_weights_k=weights.stride(1),
        indices_ptr=indices,
        stride_indices_m=indices.stride(0),
        stride_indices_k=indices.stride(1),
        hist_ptr=hist,
        expt_offs_ptr=expt_offs,
        partial_hist_ptr=partial_hist,
        stride_partial_m=partial_hist.stride(0),
        stride_partial_n=partial_hist.stride(1),
        gate_scal_ptr=gate_scal,
        topk_index_ptr=topk_index,
        gate_index_ptr=gate_index,
        token_offs_pad_ptr=token_offs_pad,
        stride_token_offs_m=token_offs_pad.stride(0),
        stride_token_offs_n=token_offs_pad.stride(1),
        block_pid_map_ptr=block_pid_map,
        stride_pid_map_m=block_pid_map.stride(0),
        stride_pid_map_n=block_pid_map.stride(1),
        max_n_tiles=max_n_tiles,
        barrier_ptr=barrier,
        M=M,
        N=N,
        topk=topk,
        num_programs=num_programs,
        RENORM=renormalize,
        ROWS_PER_PID=ROWS_PER_PID,
        NUM_BLOCK_SIZES=NUM_BLOCK_SIZES,
        BLOCK_M_LOG2_START=BLOCK_M_LOG2_START,
    )

    gate_scal = gate_scal.to(torch.bfloat16)
    weights = weights.to(torch.bfloat16)

    token_offs_pad_dict = {
        (1 << (BLOCK_M_LOG2_START + i)): token_offs_pad[i]
        for i in range(NUM_BLOCK_SIZES)
    }
    block_pid_map_dict = {
        (1 << (BLOCK_M_LOG2_START + i)): block_pid_map[i]
        for i in range(NUM_BLOCK_SIZES)
    }

    # token_offs_raw is expt_offs[:N]
    token_offs_raw = expt_offs[: N + 1]

    expt_data = ExptData(
        hist=hist,
        token_offs_raw=token_offs_raw,
        token_offs_pad=token_offs_pad_dict,
        block_pid_map=block_pid_map_dict,
    )

    gather_index = GatherIndx(topk_index, gate_index)
    scatter_index = ScatterIndx(gate_index, topk_index)
    routing_data = RoutingData(
        gate_scal=gate_scal,
        expt_hist=hist,
        n_expts_tot=N,
        n_expts_act=topk,
        expt_data=expt_data,
    )
    return routing_data, gather_index, scatter_index


def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1,
    w2,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: str = "silu",
    quant_config: FusedMoEQuantConfig | None = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    routing_data, gather_idx, scatter_idx = fused_routing(
        gating_output, topk, renormalize
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


def triton_kernel_fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,
    w2,
    routing_data,
    gather_indx,
    scatter_indx,
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

    assert hidden_states.dtype == torch.bfloat16
    assert quant_config.w1_bias is None or quant_config.w1_bias.dtype == torch.float32
    assert quant_config.w2_bias is None or quant_config.w2_bias.dtype == torch.float32

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

    intermediate_cache = _resize_cache(
        intermediate_cache, (batch_dim, M * topk, N // 2)
    )
    output_tensor = _resize_cache(output_tensor, (batch_dim, M, K))

    act = FusedActivation(
        FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")),
        (swiglu_alpha, swiglu_limit),
        2,
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

    bm_cols = triton.cdiv(num_local_experts, BLOCK_SIZE_K)
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
    bitmatrix = Bitmatrix(
        bitmatrix, shape=bitmatrix_shape, shape_max=bitmatrix_shape_max, scratchpad=None
    )

    topk_weights = torch.where(topk_ids == -1, -1.0, topk_weights)
    routing_data, gather_indx, scatter_indx = routing_from_bitmatrix(
        bitmatrix, topk_weights, topk_ids, num_local_experts, num_topk
    )

    return routing_data, gather_indx, scatter_indx


class BaseOAITritonExperts(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(self, quant_config: FusedMoEQuantConfig):
        super().__init__(quant_config)

    def supports_expert_map(self) -> bool:
        return True

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
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
        return TopKWeightAndReduceNoOP()

    def _make_routing_data(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_local_experts: int,
    ) -> tuple["RoutingData", torch.Tensor, torch.Tensor]:
        return make_routing_data(topk_ids, topk_weights, num_local_experts)


class OAITritonExperts(BaseOAITritonExperts):
    def __init__(self, quant_config: FusedMoEQuantConfig):
        assert quant_config.use_mxfp4_w4a16, "Supports only mxfp4_w4a16"
        super().__init__(quant_config)

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.Standard,
            mk.FusedMoEActivationFormat.Standard,
        )

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
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (0, 0)
        workspace2 = (M * topk, N // 2)
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
            expert_map=None,
            intermediate_cache=workspace2,
            a1q_scale=a1q_scale,
        )


class UnfusedOAITritonExperts(BaseOAITritonExperts):
    def __init__(self, quant_config: FusedMoEQuantConfig):
        assert quant_config.use_mxfp4_w4a16, "Supports only mxfp4_w4a16"
        super().__init__(quant_config)

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.Standard,
            mk.FusedMoEActivationFormat.Standard,
        )

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
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (M * topk, N // 2)
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
        if self.quant_config is None:
            self.quant_config = FUSED_MOE_UNQUANTIZED_CONFIG
        if expert_map is not None:
            topk_ids = expert_map[topk_ids]

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        routing_data, gather_indx, scatter_indx = self._make_routing_data(
            topk_ids, topk_weights, local_num_experts
        )

        topk = topk_ids.size(1)

        assert hidden_states.dtype == torch.bfloat16
        assert (
            self.quant_config.w1_bias is None
            or self.quant_config.w1_bias.dtype == torch.float32
        )
        assert (
            self.quant_config.w2_bias is None
            or self.quant_config.w2_bias.dtype == torch.float32
        )

        assert hidden_states.ndim == 2
        assert hidden_states.shape[-1] == w1.shape[-2]
        assert w2.shape[-1] == w1.shape[1]

        batch_dim = 1
        M, K = hidden_states.shape
        E, _, N = w1.shape

        if global_num_experts == -1:
            global_num_experts = E

        intermediate_cache1 = _resize_cache(workspace2, (batch_dim, M * topk, N))
        intermediate_cache3 = _resize_cache(workspace2, (batch_dim, M * topk, K))
        intermediate_cache2 = _resize_cache(workspace13, (M * topk, N // 2))

        gammas = routing_data.gate_scal if routing_data else None

        matmul_ogs(
            hidden_states,
            w1,
            self.quant_config.w1_bias,
            routing_data,
            gather_indx=gather_indx,
            precision_config=self.quant_config.w1_precision,
            gammas=gammas if apply_router_weight_on_input else None,
            fused_activation=None,
            y=intermediate_cache1,
        )

        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        routing_data.n_expts_act = 1

        matmul_ogs(
            intermediate_cache2,
            w2,
            self.quant_config.w2_bias,
            routing_data,
            scatter_indx=scatter_indx,
            precision_config=self.quant_config.w2_precision,
            gammas=None if apply_router_weight_on_input else gammas,
            y=intermediate_cache3,
        )

        self.moe_sum(intermediate_cache3.view(-1, topk, K), output)
