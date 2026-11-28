# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 routed experts on Hopper: Humming expert GEMMs (bf16
activations, NVFP4 / MXFP8 weights) in place of the TRT-LLM-gen ones.

Rows stay in the flat (token * topk + slot) order: the gate/up GEMM
gathers each row's token, the down GEMM reads and writes row by row, and
`ops.moe.combine` sums them with the identity `src2dst` (-1: not local).
"""

import json
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units

from .activation import _situ
from .linear import _k_group_tile, _prefetch_k_group_tile, k_group_partials
from .moe import _load_bias, _local_experts, _select_experts, _spin_until

if TYPE_CHECKING:
    from vllm.utils.humming import HummingInputSchema

# Row block size when no Humming tuning range covers the row count.
_DEFAULT_BLOCK_M = 64


@triton.jit
def _select_kernel(
    Partials,
    Bias,
    ExpertBounds,
    IsPadding,
    Ids,
    Weights,
    Src2Dst,
    T,
    route_scale,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
    SORT: tl.constexpr,
):
    """Local expert ids (-1: not local), routing weights and identity
    `src2dst` of the flat rows of one token."""
    token = tl.program_id(0) + tl.arange(0, 1)
    bias = _load_bias(Bias, E, E_POW2)
    ids, weights = _select_experts(
        Partials, bias, token, T, route_scale, E, E_POW2, GROUPS, TOPK, SORT
    )
    padding = None
    if IsPadding is not None:
        padding = tl.load(IsPadding + token)
    local = _local_experts(
        ids, tl.load(ExpertBounds), tl.load(ExpertBounds + 1), padding
    )
    rows = token[:, None] * TOPK + tl.arange(0, TOPK)[None, :]
    tl.store(Ids + rows, local)
    tl.store(Weights + rows, weights)
    tl.store(Src2Dst + rows, tl.where(local >= 0, rows, -1))


def select_experts(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    expert_bounds: torch.Tensor,
    is_padding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The router and top-k of `ops.moe.route`, without its layout.

    Returns the local expert id of each token's slots `[T, topk]` (-1: not
    one of the local experts `expert_bounds`, or a padding token), their
    routing weights `[T * topk]` and the identity `src2dst` `[T * topk]`
    of the local rows (-1 elsewhere).
    """
    tokens = x.shape[0]
    experts = gate_weight.shape[0]
    partials = k_group_partials(x, gate_weight)
    ids = torch.empty((tokens, topk), dtype=torch.int32, device=x.device)
    weights = torch.empty(tokens * topk, dtype=torch.float32, device=x.device)
    src2dst = torch.empty(tokens * topk, dtype=torch.int32, device=x.device)
    _select_kernel[(tokens,)](
        partials,
        bias,
        expert_bounds,
        is_padding,
        ids,
        weights,
        src2dst,
        tokens,
        route_scale,
        E=experts,
        E_POW2=triton.next_power_of_2(experts),
        GROUPS=partials.shape[0],
        TOPK=topk,
        SORT=tokens <= 128,
        num_warps=4 if tokens <= 256 else 1,
    )
    return ids, weights, src2dst


# Batches up to this many tokens route and align in one launch.
ALIGNED_ROUTING_MAX_TOKENS = 32
# Completion counters (`Inkling2Gate.counters`), one 128-byte line apart.
_ROUTERS_DONE = tl.constexpr(0)
_TOKENS_DONE = tl.constexpr(32)


@triton.jit
def _align_rows(
    Ids,
    SortedIds,
    ExpertIds,
    NumPadded,
    numel,
    max_padded,
    max_blocks,
    NUM_LOCAL: tl.constexpr,
    LOCAL_POW2: tl.constexpr,
    ROWS_POW2: tl.constexpr,
    BLOCKS_POW2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
):
    """`moe_align_block_size` (its one-block small-batch kernel, invalid ids
    ignored) of the local expert ids `Ids` [numel], bitwise: each expert's
    rows padded to `BLOCK_SIZE`, in the order that kernel's `STRIDE`
    counting threads give them (row i goes to thread i % STRIDE)."""
    rows = tl.arange(0, ROWS_POW2)
    experts = tl.arange(0, LOCAL_POW2)
    ids = tl.load(Ids + rows, rows < numel, other=-1, cache_modifier=".cg")
    valid = (ids >= 0) & (ids < NUM_LOCAL)
    counts = tl.sum(
        ((ids[None, :] == experts[:, None]) & valid[None, :]).to(tl.int32), 1
    )
    padded = (counts + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    ends = tl.cumsum(padded, 0)
    total = tl.sum(padded)
    # Padding before each expert's rows: its padded start minus its dense one.
    pad_before = (ends - padded) - (tl.cumsum(counts, 0) - counts)
    # Rows sorted by (expert, counting thread, iteration); invalid rows last.
    key = (rows % STRIDE) * ROWS_POW2 + rows // STRIDE
    expert_key = tl.where(valid, ids, LOCAL_POW2)
    order = tl.sort(expert_key * (STRIDE * ROWS_POW2) + key)
    sorted_expert = order // (STRIDE * ROWS_POW2)
    sorted_key = order % (STRIDE * ROWS_POW2)
    sorted_row = (sorted_key % ROWS_POW2) * STRIDE + sorted_key // ROWS_POW2
    dense = sorted_expert < NUM_LOCAL
    shift = tl.gather(pad_before, tl.where(dense, sorted_expert, 0), 0)
    for start in range(0, max_padded, ROWS_POW2):
        offs = start + rows
        tl.store(SortedIds + offs, numel, offs < max_padded)
    tl.debug_barrier()
    tl.store(SortedIds + rows + shift, sorted_row, dense)
    blocks = tl.arange(0, BLOCKS_POW2)
    block_expert = tl.sum(
        (ends[None, :] <= blocks[:, None] * BLOCK_SIZE).to(tl.int32), 1
    )
    block_expert = tl.where(blocks * BLOCK_SIZE < total, block_expert, -1)
    tl.store(ExpertIds + blocks, block_expert, blocks < max_blocks)
    tl.store(NumPadded, total)


@triton.jit
def _route_align_kernel(
    X,
    GateWeight,
    Bias,
    Partials,
    ExpertBounds,
    IsPadding,
    Counters,
    Ids,
    Weights,
    Src2Dst,
    SortedIds,
    ExpertIds,
    NumPadded,
    T,
    route_scale,
    max_padded,
    max_blocks,
    H: tl.constexpr,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    M_TILES: tl.constexpr,
    NUM_LOCAL: tl.constexpr,
    LOCAL_POW2: tl.constexpr,
    ROWS_POW2: tl.constexpr,
    BLOCKS_POW2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
):
    # Programs [0, ROUTERS) compute the router K-group partials (the tiles
    # of `k_group_partials`). Program ROUTERS + t waits for them and selects
    # token t's experts (`_select_kernel`); the last token program to finish
    # aligns all rows for the expert GEMMs (`moe_align_block_size`) and
    # resets the counters. All programs of a launch are resident at once.
    pid = tl.program_id(0)
    N_TILES: tl.constexpr = (E + BN - 1) // BN
    ROUTERS: tl.constexpr = N_TILES * M_TILES * GROUPS
    if pid < ROUTERS:
        pid_n = pid % N_TILES
        pid_m = (pid // N_TILES) % M_TILES
        pid_k = pid // (N_TILES * M_TILES)
        if GROUPS > 1 and pid_m == 0:
            _prefetch_k_group_tile(GateWeight, E, H, BN, pid_n, pid_k)
        _k_group_tile(
            X, GateWeight, Partials, T, E, H, BM, BN, GROUPS > 1, pid_n, pid_m, pid_k
        )
        tl.debug_barrier()
        tl.atomic_add(Counters + _ROUTERS_DONE, 1, sem="release")
    else:
        token = (pid - ROUTERS) + tl.arange(0, 1)
        bias = _load_bias(Bias, E, E_POW2)
        padding = None
        if IsPadding is not None:
            padding = tl.load(IsPadding + token)
        lo = tl.load(ExpertBounds)
        hi = tl.load(ExpertBounds + 1)
        _spin_until(Counters + _ROUTERS_DONE, ROUTERS)
        ids, weights = _select_experts(
            Partials, bias, token, T, route_scale, E, E_POW2, GROUPS, TOPK, True
        )
        local = _local_experts(ids, lo, hi, padding)
        rows = token[:, None] * TOPK + tl.arange(0, TOPK)[None, :]
        tl.store(Ids + rows, local)
        tl.store(Weights + rows, weights)
        tl.store(Src2Dst + rows, tl.where(local >= 0, rows, -1))
        tl.debug_barrier()
        if tl.atomic_add(Counters + _TOKENS_DONE, 1, sem="acq_rel") == T - 1:
            _align_rows(
                Ids,
                SortedIds,
                ExpertIds,
                NumPadded,
                T * TOPK,
                max_padded,
                max_blocks,
                NUM_LOCAL,
                LOCAL_POW2,
                ROWS_POW2,
                BLOCKS_POW2,
                BLOCK_SIZE,
                STRIDE,
            )
            tl.store(Counters + _ROUTERS_DONE, 0)
            tl.store(Counters + _TOKENS_DONE, 0)


def use_aligned_routing(
    tokens: int, experts: int, hidden: int, device: torch.device
) -> bool:
    """Whether `select_experts_aligned` serves `tokens`: few enough, and its
    programs (which wait for each other, launched cooperatively) all fit on
    `device` at once."""
    if not 0 < tokens <= ALIGNED_ROUTING_MAX_TOKENS:
        return False
    block_n = 32 if tokens <= 16 else 64
    routers = (
        triton.cdiv(experts, block_n)
        * triton.cdiv(tokens, 16)
        * triton.cdiv(hidden, 1024)
    )
    return routers + tokens <= num_compute_units(device.index or 0)


def align_sizes(rows: int, num_local: int, block_size: int) -> tuple[int, int]:
    """`moe_align_block_size`'s `sorted_ids` and `expert_ids` lengths."""
    max_padded = rows + num_local * (block_size - 1)
    if rows < num_local:
        max_padded = min(rows * block_size, max_padded)
    return max_padded, triton.cdiv(max_padded, block_size)


def select_experts_aligned(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    expert_bounds: torch.Tensor,
    num_local: int,
    counters: torch.Tensor,
    block_size: int,
    is_padding: torch.Tensor | None = None,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """`select_experts` and `moe_align_block_size(ids, block_size,
    num_local, ignore_invalid_experts=True)` of its local ids in one
    launch, where `use_aligned_routing`, for the `num_local` experts of
    `expert_bounds`. The selection is bitwise that of `select_experts`, and
    the layout bitwise that of `moe_align_block_size` for up to 64 local
    experts (its one-block kernel). With more, `moe_align_block_size`
    places rows by atomics: the expert of each block, the padded count and
    each expert's set of rows are the same; their order, and so which of
    the expert's blocks holds a row, may differ. `counters`
    (`Inkling2Gate`) must be zero; the launch leaves them zero."""
    tokens, hidden = x.shape
    experts = gate_weight.shape[0]
    assert use_aligned_routing(tokens, experts, hidden, x.device)
    assert x.is_contiguous()
    rows = tokens * topk
    device = x.device
    # The tiles of `k_group_partials` (split K) for this many tokens.
    block_m, block_n = 16, 32 if tokens <= 16 else 64
    groups = triton.cdiv(hidden, 1024)
    partials = torch.empty(
        (groups, tokens, experts), dtype=torch.float32, device=device
    )
    ids = torch.empty((tokens, topk), dtype=torch.int32, device=device)
    weights = torch.empty(rows, dtype=torch.float32, device=device)
    src2dst = torch.empty(rows, dtype=torch.int32, device=device)
    max_padded, max_blocks = align_sizes(rows, num_local, block_size)
    sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
    expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
    num_padded = torch.empty(1, dtype=torch.int32, device=device)
    m_tiles = triton.cdiv(tokens, block_m)
    routers = triton.cdiv(experts, block_n) * m_tiles * groups
    _route_align_kernel[(routers + tokens,)](
        x,
        gate_weight,
        bias,
        partials,
        expert_bounds,
        is_padding,
        counters,
        ids,
        weights,
        src2dst,
        sorted_ids,
        expert_ids,
        num_padded,
        tokens,
        route_scale,
        max_padded,
        max_blocks,
        H=hidden,
        E=experts,
        E_POW2=triton.next_power_of_2(experts),
        GROUPS=groups,
        TOPK=topk,
        BM=block_m,
        BN=block_n,
        M_TILES=m_tiles,
        NUM_LOCAL=num_local,
        LOCAL_POW2=triton.next_power_of_2(num_local),
        ROWS_POW2=max(triton.next_power_of_2(rows), 16),
        BLOCKS_POW2=max(triton.next_power_of_2(max_blocks), 16),
        BLOCK_SIZE=block_size,
        STRIDE=max(num_local, 32),
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
        launch_cooperative_grid=True,
    )
    return (ids, weights, src2dst), (sorted_ids, expert_ids, num_padded)


@triton.jit
def _swiglu_kernel(
    GateUp,
    Weights,
    Src2Dst,
    Out,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
):
    """bf16 softcapped SwiGLU of a local row times its routing weight."""
    row = tl.program_id(0)
    if tl.load(Src2Dst + row) < 0:
        return
    start = tl.program_id(1) * BLOCK
    gamma = tl.load(Weights + row)
    values = _situ(GateUp, row, start, gamma, N, BLOCK, GATE_CAP, UP_CAP)
    tl.store(Out + row * N + start + tl.arange(0, BLOCK), values.to(tl.bfloat16))


def swiglu(
    gate_up: torch.Tensor,
    weights: torch.Tensor,
    src2dst: torch.Tensor,
    gate_cap: float,
    up_cap: float,
) -> torch.Tensor:
    """`activation.swiglu_quantize`'s SiTU of the interleaved (gate, up)
    columns times the routing weights, in bf16, for the local rows (the
    others are left unset)."""
    rows, n = gate_up.shape[0], gate_up.shape[1] // 2
    block = 256 if n % 256 == 0 else 128
    assert n % block == 0
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=gate_up.device)
    _swiglu_kernel[(rows, n // block)](
        gate_up,
        weights,
        src2dst,
        out,
        N=n,
        BLOCK=block,
        GATE_CAP=gate_cap,
        UP_CAP=up_cap,
        num_warps=2,
    )
    return out


class HummingExperts:
    """Humming tensors and kernel configs of the two expert projections.

    Args:
        weights: Per projection (`w13`, `w2`), the row-major checkpoint
            weights `[E, N, K / pack]`, block scales `[E, N, K / group]`
            and, for NVFP4, the per-expert global scales `[E]` (else None).
        shapes: `(N, K)` of each projection.
        nvfp4: NVFP4 (else MXFP8) weights.
        input_schema: Quantized input contract, or None for BF16 activations.

    """

    def __init__(
        self,
        weights: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]],
        shapes: tuple[tuple[int, int], tuple[int, int]],
        nvfp4: bool,
        input_schema: "HummingInputSchema | None" = None,
    ) -> None:
        from vllm.utils.humming import (
            GemmType,
            HummingInputSchema,
            ModeloptMxfp8WeightSchema,
            ModeloptNvfp4WeightSchema,
            get_heuristics_config,
            prepare_and_transform_humming_tensors,
        )

        self.configs = {}
        self.tensors: dict[str, dict[str, torch.Tensor]] = {}
        self.tunings: dict[str, list] = {}
        self.tuning_strs: dict[str, str] = {}
        for name, (n, k) in zip(("w13", "w2"), shapes):
            weight, scale, global_scale = weights[name]
            num_experts = weight.shape[0]
            tensors = {"weight": weight, "weight_scale": scale}
            if nvfp4:
                source = ModeloptNvfp4WeightSchema()
                tensors["weight_scale_2"] = global_scale
            else:
                source = ModeloptMxfp8WeightSchema()
            schema, tensors = source.convert_humming(
                tensors=tensors,
                shape_n_stacks=[n],
                shape_k_stacks=[k],
                param_dtype=torch.bfloat16,
                num_experts=num_experts,
            )
            config, tensors = prepare_and_transform_humming_tensors(
                tensors=tensors,
                shape_n=n,
                shape_k=k,
                pad_n_to_multiple=256,
                pad_k_to_multiple=128,
                input_schema=input_schema or HummingInputSchema(),
                weight_schema=schema,
                num_experts=num_experts,
                torch_dtype=torch.bfloat16,
                device=weight.device,
            )
            assert config.shape_n == n and config.shape_k == k, (n, k)
            tuning = get_heuristics_config(
                layer_config=config,
                use_f16_accum=False,
                use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
                gemm_type=GemmType.INDEXED,
            )
            # Stream-K adds the K slices into the output with bf16 atomics,
            # whose order (and so the result) changes from run to run.
            for _, _, kernel_config in tuning:
                kernel_config["use_stream_k"] = False
            self.configs[name] = config
            self.tensors[name] = tensors
            self.tunings[name] = tuning
            self.tuning_strs[name] = json.dumps(tuning)
        self.num_experts = num_experts
        self.compute_config = json.dumps(
            {
                "use_batch_invariant": envs.VLLM_BATCH_INVARIANT,
                "use_f16_accum": False,
                "gemm_type": GemmType.INDEXED.value,
            }
        )
        self.locks = torch.zeros(2048, dtype=torch.int32, device=weight.device)

    def _valid_rows(self, rows: int, num_global_experts: int) -> int:
        """Expected local rows, which pick the kernel configs."""
        return -(-rows * self.num_experts // num_global_experts)

    def block_size(
        self, ids_shape: tuple[int, int], num_global_experts: int
    ) -> int | None:
        """The row block size of both projections for `ids` of this shape,
        or None when they differ."""
        valid_rows = self._valid_rows(ids_shape[0] * ids_shape[1], num_global_experts)
        sizes = {self._block_m(name, valid_rows) for name in ("w13", "w2")}
        return sizes.pop() if len(sizes) == 1 else None

    def _block_m(self, name: str, rows: int) -> int:
        for lower, upper, config in self.tunings[name]:
            if lower < rows <= upper:
                return config["block_shape"][0]
        return _DEFAULT_BLOCK_M

    def _gemm(
        self,
        name: str,
        inputs: torch.Tensor,
        rows: int,
        alignment: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        top_k: int,
        valid_rows: int,
    ) -> torch.Tensor:
        from vllm.utils.humming import humming_forward

        config = self.configs[name]
        tensors = self.tensors[name]
        out = torch.empty(
            (rows, config.shape_n), dtype=torch.bfloat16, device=inputs.device
        )
        sorted_ids, expert_ids, num_tokens_padded = alignment
        return humming_forward(
            config,
            inputs=inputs,
            weight=tensors["weight"],
            weight_scale=tensors.get("weight_scale"),
            weight_scale_2=tensors.get("weight_scale_2"),
            outputs=out,
            locks=self.locks,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            top_k=top_k,
            valid_shape_m=valid_rows,
            compute_config=self.compute_config,
            tuning_config=self.tuning_strs[name],
        )

    def __call__(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        src2dst: torch.Tensor,
        num_global_experts: int,
        gate_cap: float,
        up_cap: float,
        alignment: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """bf16 `[T * topk, H]` down-projection rows of the local slots of
        `ids` (`select_experts`); the other rows are left unset. `alignment`
        is their `moe_align_block_size` for `block_size` when the caller
        has it (`select_experts_aligned`)."""
        tokens, topk = ids.shape
        rows = tokens * topk
        valid_rows = self._valid_rows(rows, num_global_experts)
        block_m = {name: self._block_m(name, valid_rows) for name in ("w13", "w2")}
        if alignment is not None:
            assert set(block_m.values()) == {
                self.block_size(ids.shape, num_global_experts)
            }
            alignments = {size: alignment for size in block_m.values()}
        else:
            alignments = {
                size: moe_align_block_size(
                    ids, size, self.num_experts, ignore_invalid_experts=True
                )[:3]
                for size in set(block_m.values())
            }
        gate_up = self._gemm(
            "w13", x, rows, alignments[block_m["w13"]], topk, valid_rows
        )
        activated = swiglu(gate_up, weights, src2dst, gate_cap, up_cap)
        del gate_up
        return self._gemm(
            "w2", activated, rows, alignments[block_m["w2"]], 1, valid_rows
        )
