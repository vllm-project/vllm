# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# The affine summary and merge equations follow the FLA context-parallel
# implementation (fla/ops/cp/chunk_delta_h.py, MIT license), specialized to the
# KDA gate convention (per-dim log2 gate cumsum `gk`, pre-gated `kg`, no scalar
# gate, no DPLR) and to serving-side zig-zag chunk sharding.
"""KCP (two-pass parallel scan) for KDA layers under hybrid PCP.

Each rank prepares its prefill segments (see ``HybridPCPLayout`` in
``vllm.v1.worker.gpu.pcp_manager``) once with FlashKDA's preparation kernel,
then:

1. Computes every segment's affine summary S_end = M @ S_in + S from a zero
   initial state, with FlashKDA's recurrence over the value columns plus the
   identity's columns (the transition M).
2. All-gathers the BF16 [S^T; M^T] summaries and chain-merges them in chunk
   order in FP32. This yields every segment's initial state and every
   request's final state; all ranks publish the final state so the replicated
   state caches stay coherent.
3. Runs FlashKDA's recurrence on its segments from the merged initial states.

The short convolution needs each segment's preceding (kernel_size - 1) raw
inputs. Ranks all-gather their segment tails, assemble every initial and final
window from the tails and the cached prefix, and run the ordinary kernels.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pcp_group
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.pcp_manager import HybridPCPLayout


@dataclass
class KcpPlan:
    """Per-step KCP indices, derived once from the step's layout.

    Mamba groups share the plan; each group's state slots come from its own
    bound layout.

    Every rank exchanges two per-layer summaries in fixed physical layouts:

    * Chunk summaries: ``[P ranks, 2 parts, N]`` gathered in one all-gather,
      where part 0 is slot ``rank`` and part 1 is slot ``2P - 1 - rank``.
    * Convolution tails: the last ``halo`` raw inputs of both parts,
      ``[N, 2, halo]`` per rank and right-aligned. Short and empty parts leave
      rows absent.

    Convolution windows select from a pool whose first ``N * halo`` rows are each
    request's cached convolution prefix, followed by the gathered tails.
    """

    layout: "HybridPCPLayout"
    halo_size: int
    # Rows of the gathered [P, 2, N] summaries; per [N, 2P] (request, slot),
    # whether any rank has tokens there and the local segment owning it or -1.
    summary_idx: torch.Tensor
    slot_present: torch.Tensor
    segment_of_slot: torch.Tensor
    # Local prefill token of each [N, 2, halo] tail row, or -1 if absent.
    tail_src_idx: torch.Tensor
    # Convolution pool rows of each segment's initial and request's final window.
    halo_idx: torch.Tensor
    final_halo_idx: torch.Tensor
    # Short-convolution inputs over the local segments; window row 0 is
    # NULL_BLOCK_ID, so segments use rows 1..L.
    conv_metadata: SimpleNamespace
    segment_ids: torch.Tensor
    segment_has_initial_state: torch.Tensor
    # Exchange buffers and views the step's KDA layers share.
    buffers: dict = field(default_factory=dict, repr=False)


def plan_for(layout: "HybridPCPLayout", halo_size: int) -> KcpPlan:
    """This step's KCP plan, built on first use (the model builds it ahead of
    the forward)."""
    plan = layout.shared.get("kcp")
    if plan is None:
        plan = layout.shared["kcp"] = _build_plan(layout, halo_size)
    assert plan.halo_size == halo_size
    return plan


def _build_plan(layout: "HybridPCPLayout", halo: int) -> KcpPlan:
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
    from vllm.v1.worker.gpu.pcp_manager import upload_pinned

    world, rank = layout.world, layout.rank
    N, num_slots = layout.num_prefill_reqs, 2 * world
    chunk_lens = -(-layout.prefill_lens // num_slots)
    slot_lens = layout.slot_lens
    scan_cu = layout.scan_cu

    def part(slot: int) -> int:
        return int(slot >= world)

    def owner(slot: int) -> int:
        return slot if slot < world else num_slots - 1 - slot

    def window(n: int, end: int) -> list[int]:
        """Pool rows of request n's raw inputs at offsets [end - halo, end)."""
        rows = []
        for position in range(end - halo, end):
            if position < 0:
                rows.append(n * halo + position + halo)
                continue
            slot = position // int(chunk_lens[n])
            offset = position - slot * int(chunk_lens[n])
            column = halo - int(slot_lens[n, slot]) + offset
            assert column >= 0
            tail = ((owner(slot) * N + n) * 2 + part(slot)) * halo
            rows.append(N * halo + tail + column)
        return rows

    summary_idx, halo_idx = [], []
    tail_src_idx = np.full(N * 2 * halo, -1, dtype=np.int64)
    segments = zip(layout.segment_request, layout.segment_slot, layout.segment_start)
    for i, (n, slot, start) in enumerate(segments):
        n, slot, end = int(n), int(slot), int(scan_cu[i + 1])
        take = min(halo, end - int(scan_cu[i]))
        destination = (n * 2 + part(slot)) * halo + halo - take
        tail_src_idx[destination : destination + take] = np.arange(end - take, end)
        summary_idx.append((rank * 2 + part(slot)) * N + n)
        halo_idx.append(window(n, int(start)))
    final_halo_idx = [
        window(n, int(length)) for n, length in enumerate(layout.prefill_lens)
    ]
    num_segments = layout.num_segments
    segment_of_slot = np.full(N * num_slots, -1)
    segment_of_slot[layout.segment_request * num_slots + layout.segment_slot] = (
        np.arange(num_segments)
    )

    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        torch.from_numpy(scan_cu).to(torch.int32), device=torch.device("cpu")
    )
    device = layout.scan_cu_seqlens.device
    indices = upload_pinned(
        device,
        torch.int64,
        summary_idx=summary_idx,
        tail_src_idx=tail_src_idx,
        halo_idx=np.asarray(halo_idx, dtype=np.int64).reshape(-1, halo),
        final_halo_idx=final_halo_idx,
    )
    lengths = upload_pinned(
        device,
        torch.int32,
        segment_of_slot=segment_of_slot,
        segment_ids=np.arange(1, num_segments + 1),
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )
    flags = upload_pinned(
        device,
        torch.bool,
        segment_has_initial_state=np.ones(num_segments, dtype=np.bool_),
        slot_present=slot_lens.reshape(-1) > 0,
    )
    batch_ptr = lengths.pop("batch_ptr")
    token_chunk_offset_ptr = lengths.pop("token_chunk_offset_ptr")
    for metadata in nums_dict.values():
        metadata["batch_ptr"] = batch_ptr
        metadata["token_chunk_offset_ptr"] = token_chunk_offset_ptr
    return KcpPlan(
        layout=layout,
        halo_size=halo,
        conv_metadata=SimpleNamespace(
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        ),
        **indices,
        **lengths,
        **flags,
    )


# FlashKDA's workspace bytes per (head, 16-token tile) for head_dim 128.
WORKSPACE_BYTES_PER_TILE = 13824


@dataclass
class KcpSegments:
    """A rank's local prefill segments after FlashKDA's preparation."""

    v: torch.Tensor
    beta_t: torch.Tensor
    workspace: torch.Tensor


def prepare_segments(
    plan: KcpPlan,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
) -> KcpSegments:
    """Run FlashKDA's preparation on [1, T, H, D] segments with raw gates."""
    import vllm._flashkda_C  # noqa: F401

    _, tokens, heads, dim = q.shape
    assert dim == 128
    beta_t = beta[0].t().contiguous()
    # FlashKDA's get_workspace_size, computed here to skip an op call per layer.
    tiles = (tokens + 15) // 16 + plan.layout.num_segments
    workspace = torch.empty(
        heads * tiles * WORKSPACE_BYTES_PER_TILE, dtype=torch.uint8, device=q.device
    )
    torch.ops._flashkda_C.kcp_flash_prepare(
        q.contiguous(),
        k.contiguous(),
        raw_g.contiguous(),
        beta_t,
        a_log.view(-1),
        dt_bias.view(heads, dim),
        plan.layout.scan_cu_seqlens,
        workspace,
        dim**-0.5,
        lower_bound,
    )
    return KcpSegments(v.contiguous(), beta_t, workspace)


def scan_segments(
    plan: KcpPlan,
    segments: KcpSegments | None,
    initial_states: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Write every local segment's outputs from its merged initial state."""
    if segments is not None:
        torch.ops._flashkda_C.kcp_flash_scan(
            segments.v,
            segments.beta_t,
            segments.workspace,
            plan.layout.scan_cu_seqlens,
            initial_states,
            out,
        )


@triton.jit
def _merge_states_kernel(
    summaries,
    cache,
    slots,
    has_initial,
    slot_present,
    segment_of_slot,
    initial,
    N,
    cache_stride,
    S: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
):
    """Advance value-first states through every chronological slot.

    Summaries are BF16 [P, 2, N, H, V + K, K] blocks [S^T; M^T]; the state
    after a slot is S^T + state @ M^T in FP32. M^T is exact in BF16, so two
    BF16 MMAs of the state's high and low halves keep about 16 bits. Each
    request starts from its cached state (zero without one) and publishes the
    final state to the cache. Absent slots are skipped. The state before each
    slot a local segment owns is that segment's initial state.
    """
    nh = tl.program_id(0).to(tl.int64)
    n, h = nh // H, nh % H
    v = tl.program_id(1) * BV + tl.arange(0, BV)[:, None]
    k = tl.arange(0, K)[None, :]
    kk = tl.arange(0, K)[:, None]
    slot = tl.load(slots + n).to(tl.int64)
    cached = cache + slot * cache_stride + (h * V + v) * K + k
    state = tl.load(cached).to(tl.float32)
    state = tl.where(tl.load(has_initial + n) != 0, state, 0.0)
    for c in range(0, S):
        if tl.load(slot_present + n * S + c) != 0:
            segment = tl.load(segment_of_slot + n * S + c).to(tl.int64)
            if segment >= 0:
                tl.store(initial + ((segment * H + h) * V + v) * K + k, state)
            # Slot c < P is rank c's part 0; slot c >= P is rank 2P-1-c's part 1.
            physical = tl.where(c < S // 2, 2 * c, 2 * (S - 1 - c) + 1).to(tl.int64)
            block = summaries + ((physical * N + n) * H + h) * (V + K) * K
            after = tl.load(block + v * K + k).to(tl.float32)
            transition_t = tl.load(block + (V + kk) * K + k)
            high = state.to(tl.bfloat16)
            low = (state - high.to(tl.float32)).to(tl.bfloat16)
            product = tl.dot(high, transition_t, out_dtype=tl.float32)
            state = after + tl.dot(low, transition_t, product, out_dtype=tl.float32)
    tl.store(cached, state.to(cache.dtype.element_ty))


def kcp_merge(
    summaries: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    has_initial: torch.Tensor,
    slot_present: torch.Tensor,
    segment_of_slot: torch.Tensor,
    initial: torch.Tensor,
) -> None:
    """Merge the gathered summaries in chronological slot order.

    ``summaries`` is [2P physical slots, N, H, V + K, K] laid out [rank][part]:
    physical slot 2r holds chronological slot r and 2r + 1 holds 2P - 1 - r.
    Request n starts from ``cache[slots[n]]`` if ``has_initial[n]`` (else
    zero) and its final state is written back there. ``slot_present`` and
    ``segment_of_slot`` are [N, 2P]: whether any rank has tokens in the slot,
    and the local segment owning it or -1. Writes the FP32 value-first
    ``initial`` states of the local segments.
    """
    S, N, H, VK, K = summaries.shape
    V = VK - K
    assert summaries.is_contiguous() and cache[0].is_contiguous()
    block = 64
    _merge_states_kernel[(N * H, V // block)](
        summaries,
        cache,
        slots,
        has_initial,
        slot_present,
        segment_of_slot,
        initial,
        N,
        cache.stride(0),
        S=S,
        H=H,
        V=V,
        K=K,
        BV=block,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def _pack_conv_tail_rows(
    x,
    source_indices,
    out,
    stride_token: tl.int64,
    stride_channel: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    channel_tiles: tl.constexpr = tl.cdiv(C, BLOCK)
    token = (tl.program_id(0) // channel_tiles).to(tl.int64)
    channel = (tl.program_id(0) % channel_tiles).to(tl.int64) * BLOCK
    channel += tl.arange(0, BLOCK)
    source = tl.load(source_indices + token).to(tl.int64)
    value = tl.load(
        x + tl.maximum(source, 0) * stride_token + channel * stride_channel,
        (source >= 0) & (channel < C),
        other=0.0,
    )
    tl.store(out + token * C + channel, value, channel < C)


def gather_conv_tails(plan: KcpPlan, qkv: torch.Tensor) -> torch.Tensor:
    """All-gather every rank's raw prefill tails as [P * N * 2 * halo, C].

    Each rank packs its [N * 2 * halo, C] tail rows straight into the
    all-gather buffer.
    """
    rows, channels = plan.tail_src_idx.numel(), qkv.shape[-1]
    group = get_pcp_group()
    buffers = plan.buffers.get("conv_tails")
    if buffers is None:
        buffers = plan.buffers["conv_tails"] = group.all_gather_buffer(
            (rows, channels), qkv.dtype, key="kcp_conv_tails"
        )
    gathered, tails = buffers
    _pack_conv_tail_rows[(rows * triton.cdiv(channels, 256),)](
        qkv,
        plan.tail_src_idx,
        tails,
        qkv.stride(0),
        qkv.stride(1),
        channels,
        BLOCK=256,
    )
    return group.all_gather_in_place(gathered)


@triton.jit
def _conv_windows_kernel(
    conv_state,
    state_indices,
    has_initial_state,
    tails,
    pool_rows,
    out,
    stride_state_slot,
    stride_state_channel,
    stride_state_column,
    stride_out_row,
    stride_out_channel,
    stride_out_column,
    num_prefix_rows,
    C: tl.constexpr,
    HALO: tl.constexpr,
    OUT_BY_SLOT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Select HALO raw inputs per row from cached prefixes and gathered tails."""
    row = tl.program_id(0).to(tl.int64)
    channel = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = channel < C
    values = ()  # type: tuple
    for column in tl.static_range(HALO):
        source = tl.load(pool_rows + row * HALO + column).to(tl.int64)
        from_prefix = source < num_prefix_rows
        request = tl.where(from_prefix, source // HALO, 0)
        slot = tl.load(state_indices + request).to(tl.int64)
        continued = tl.load(has_initial_state + request).to(tl.int1)
        prefix = tl.load(
            conv_state
            + slot * stride_state_slot
            + channel * stride_state_channel
            + (source % HALO) * stride_state_column,
            mask=mask & from_prefix & continued,
            other=0.0,
        )
        tail = tl.load(
            tails + tl.maximum(source - num_prefix_rows, 0) * C + channel,
            mask=mask & (source >= num_prefix_rows),
            other=0.0,
        )
        values += (tl.where(from_prefix, prefix, tail.to(prefix.dtype)),)
    # Load every column before storing: the final windows overwrite the prefix.
    # Final windows go to the requests' cache slots; initial windows start at
    # row 1 because row 0 is NULL_BLOCK_ID to the conv kernel.
    destination = tl.load(state_indices + row).to(tl.int64) if OUT_BY_SLOT else row + 1
    for column in tl.static_range(HALO):
        tl.store(
            out
            + destination * stride_out_row
            + channel * stride_out_channel
            + column * stride_out_column,
            values[column],
            mask=mask,
        )


def conv_windows(
    plan: KcpPlan,
    tails: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    final: bool = False,
) -> torch.Tensor:
    """Gather halo windows from cached prefixes and every rank's ``tails``.

    ``conv_state`` is the (..., C, halo) cache view and ``state_indices`` the
    prefill requests' slots. By default returns the segments' initial windows
    as [1 + L, C, halo] in the cache's layout (row 0 is NULL_BLOCK_ID to the
    conv kernel; segments use rows 1..L). ``final`` instead writes every
    prefill request's final window into its conv cache slot.
    """
    channels = conv_state.shape[1]
    assert conv_state.shape[-1] == plan.halo_size
    assert tails.shape[-1] == channels and tails.is_contiguous()
    if final:
        rows, out = plan.final_halo_idx, conv_state
    else:
        rows = plan.halo_idx
        shape = (plan.layout.num_segments + 1, channels, plan.halo_size)
        if conv_state.stride(-1) == 1:
            out = conv_state.new_empty(shape)
        else:
            out = conv_state.new_empty(shape[0], shape[2], shape[1]).transpose(1, 2)
    if rows.shape[0]:
        block = 1024
        _conv_windows_kernel[(rows.shape[0], triton.cdiv(channels, block))](
            conv_state,
            state_indices,
            plan.layout.prefill_has_initial_state,
            tails,
            rows,
            out,
            *conv_state.stride(),
            *out.stride(),
            plan.layout.num_prefill_reqs * plan.halo_size,
            C=channels,
            HALO=plan.halo_size,
            OUT_BY_SLOT=final,
            BLOCK=block,
        )
    return out


def local_summaries(
    plan: KcpPlan, segments: KcpSegments | None, gathered: torch.Tensor
) -> None:
    """Write every local segment's summary into the BF16
    [P ranks * 2 parts * N, H, V + K, K] exchange buffer."""
    if segments is not None:
        torch.ops._flashkda_C.kcp_flash_summary(
            segments.v,
            segments.beta_t,
            segments.workspace,
            plan.layout.scan_cu_seqlens,
            plan.summary_idx,
            gathered,
        )


def exchange_and_scan(
    plan: KcpPlan,
    segments: KcpSegments | None,
    cache: torch.Tensor,
    slots: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Exchange summaries, merge and scan; ranks without prefill pass ``None``.

    Summaries are written straight into this rank's [2, N] block of the
    all-gather buffer and exchanged with one all-gather. Every rank publishes
    each request's final state to the replicated recurrent ``cache`` at the
    prefill requests' ``slots``; ``out`` receives the local prefill tokens'
    outputs.
    """
    N, (_, H, V, K) = plan.layout.num_prefill_reqs, cache.shape
    gathered = plan.buffers.get("summaries")
    if gathered is None:
        gathered, _ = get_pcp_group().all_gather_buffer(
            (2 * N, H, V + K, K), torch.bfloat16, key="kcp_summaries"
        )
        plan.buffers["summaries"] = gathered
    local_summaries(plan, segments, gathered)
    get_pcp_group().all_gather_in_place(gathered)
    initial = cache.new_empty(plan.layout.num_segments, H, V, K, dtype=torch.float32)
    kcp_merge(
        gathered.view(2 * plan.layout.world, N, H, V + K, K),
        cache,
        slots,
        plan.layout.prefill_has_initial_state,
        plan.slot_present,
        plan.segment_of_slot,
        initial,
    )
    scan_segments(plan, segments, initial, out)
