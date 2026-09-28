# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM KDA context parallelism (KCP) under hybrid PCP.

The end-to-end test runs the real layer on every emulated PCP rank of one GPU
and compares outputs and replicated caches with the ordinary unpartitioned
path. Rank threads run one at a time; they switch only inside collectives.
"""

import threading
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only kernels"),
    pytest.mark.skipif(
        not current_platform.is_device_capability_family(100),
        reason="KCP FlashKDA kernels are built for SM100-family GPUs",
    ),
]

LOWER_BOUND = -5.0


def _hybrid_layout(world, rank, lengths, prefilling, computed, device="cuda"):
    """One rank's hybrid-PCP layout for a global batch of the given requests."""
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

    qsl = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    manager = PCPManager(world, rank, torch.device(device))
    manager._global_batch = SimpleNamespace(
        num_reqs=len(lengths),
        num_scheduled_tokens=np.asarray(lengths, dtype=np.int32),
        is_prefilling_np=np.asarray(prefilling),
        num_computed_tokens_np=np.asarray(computed, dtype=np.int32),
        query_start_loc_np=qsl,
        num_draft_tokens_per_req=None,
    )
    segments = manager._get_rank_segments(rank, lengths, np.asarray(prefilling), qsl)
    manager._local_segments = tuple(segments)
    return manager.build_hybrid_layout(), segments, qsl


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("halo", [1, 3, 4])
@pytest.mark.parametrize(
    "lengths,prefilling",
    [
        ([1, 2, 1, 3], [False, True, False, True]),  # empty slots and parts
        ([1, 1, 37, 150], [False, False, True, True]),
        ([3], [True]),  # ranks without prefill tokens
        ([64, 1, 17], [True, True, True]),  # one-token prefill
    ],
)
def test_kcp_plan_maps_summaries_and_halos_to_global_positions(
    world, halo, lengths, prefilling
):
    """Summary rows follow the [rank][part] exchange layout, slots are marked
    present and owned, and every segment's initial window and every request's
    final window select the right raw inputs from [cached prefix, tails]."""
    from vllm.models.glm5next.nvidia.ops import kcp

    computed = [0, 9, 0, 7][: len(lengths)]
    prefill_rows = np.flatnonzero(prefilling)
    num_decodes = len(lengths) - len(prefill_rows)
    slots = 2 * world

    def position(request, offset):
        """A value naming one token of one request, including cached prefixes."""
        return 1000 * request + computed[request] + offset

    plans, tails = [], []
    for rank in range(world):
        layout, segments, qsl = _hybrid_layout(
            world, rank, lengths, prefilling, computed
        )
        plan = kcp.plan_for(layout, halo)
        prefill_segments = segments[num_decodes:]
        local = [
            position(s.global_batch_req_idx, i - qsl[s.global_batch_req_idx])
            for s in prefill_segments
            for i in range(s.global_batch_slice.start, s.global_batch_slice.stop)
        ]
        segment_slots = list(zip(layout.segment_request, layout.segment_slot))
        assert plan.summary_idx.tolist() == [
            (rank * 2 + int(slot >= world)) * len(prefill_rows) + n
            for n, slot in segment_slots
        ]
        segment_of_slot = [-1] * (len(prefill_rows) * slots)
        for i, (n, slot) in enumerate(segment_slots):
            segment_of_slot[n * slots + slot] = i
        assert plan.segment_of_slot.tolist() == segment_of_slot
        present = []
        for g in prefill_rows:
            chunk = -(-lengths[g] // slots)
            present += [lengths[g] - c * chunk > 0 for c in range(slots)]
        assert plan.slot_present.tolist() == present
        assert kcp.plan_for(layout, halo) is plan
        tail = torch.full(plan.tail_src_idx.shape, -1)
        source = plan.tail_src_idx.cpu()
        valid = source >= 0
        tail[valid] = torch.tensor(local, dtype=torch.int64)[source[valid]]
        tails.append(tail)
        plans.append((plan, prefill_segments, qsl))

    # The pool is [cached prefix of each request, gathered tails].
    gathered = torch.cat(tails)
    for plan, prefill_segments, qsl in plans:
        prefix = torch.tensor(
            [position(g, j - halo) for g in prefill_rows for j in range(halo)]
        )
        pool = torch.cat((prefix, gathered))
        for segment, rows in zip(prefill_segments, plan.halo_idx.cpu()):
            g = segment.global_batch_req_idx
            start = segment.global_batch_slice.start - qsl[g]
            expected = [position(g, start - halo + j) for j in range(halo)]
            assert pool[rows].tolist() == expected
        for g, rows in zip(prefill_rows, plan.final_halo_idx.cpu()):
            expected = [position(g, lengths[g] - halo + j) for j in range(halo)]
            assert pool[rows].tolist() == expected


@pytest.mark.parametrize("counts", [[8], [8, 8, 8], [8, 3, 0], [1, 8, 3, 0]])
@torch.inference_mode()
def test_kcp_merge_matches_sequential_fp32(counts):
    """The merge of BF16 [S^T; M^T] summaries matches a
    sequential FP32 chain from the cached states over the present slots,
    place segment initial states and publish final states to the cache.
    Absent slots hold garbage and must be skipped."""
    from vllm.models.glm5next.nvidia.ops.kcp import kcp_merge

    torch.manual_seed(7)
    slots, heads, dim = 8, 2, 128
    N = len(counts)
    summaries = torch.randn(slots, N, heads, 2 * dim, dim, device="cuda") * 0.003
    summaries[..., dim:, :] += 0.95 * torch.eye(dim, device="cuda")
    present = torch.arange(slots)[None, :] < torch.tensor(counts)[:, None]
    summaries = summaries.bfloat16()
    chrono = summaries.float()
    cache = torch.randn(2 * N + 1, heads, dim, dim, device="cuda") * 0.2
    original = cache.clone()
    cache_slots = torch.randperm(2 * N, device="cuda")[:N] + 1
    has_initial = torch.arange(N, device="cuda") != N - 1
    base = torch.where(has_initial[:, None, None, None], cache[cache_slots], 0.0)
    expected_inits = base.new_empty(N, slots, heads, dim, dim)
    expected_final = torch.empty_like(base)
    for request in range(N):
        state = base[request]
        for slot in range(slots):
            expected_inits[request, slot] = state
            if present[request, slot]:
                block = chrono[slot, request]
                state = block[:, :dim] + state @ block[:, dim:]
        expected_final[request] = state

    # Rank r's [part 0, part 1] blocks hold slots r and 2P - 1 - r.
    order = [c for r in range(slots // 2) for c in (r, slots - 1 - r)]
    physical = summaries[order].contiguous()
    # Local segments own every other present slot, in (request, slot) order.
    owned = [(n, c) for n in range(N) for c in range(0, slots, 2) if present[n, c]]
    segment_of_slot = torch.full((N * slots,), -1, dtype=torch.int32)
    for i, (n, c) in enumerate(owned):
        segment_of_slot[n * slots + c] = i
    initial = torch.empty(len(owned), heads, dim, dim, device="cuda")
    kcp_merge(
        physical,
        cache,
        cache_slots,
        has_initial,
        present.flatten().cuda(),
        segment_of_slot.cuda(),
        initial,
    )
    torch.testing.assert_close(
        initial,
        torch.stack([expected_inits[n, c] for n, c in owned]),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(cache[cache_slots], expected_final, atol=1e-5, rtol=1e-5)
    untouched = torch.ones(len(cache), dtype=torch.bool, device="cuda")
    untouched[cache_slots] = False
    assert torch.equal(cache[untouched], original[untouched])


@torch.inference_mode()
def test_kcp_flash_summaries_are_affine_segment_transitions():
    """Summary rows hold FlashKDA's zero-state final state (S^T) and the
    transition (M^T) for ragged and empty segments, at the given rows."""
    import vllm._flashkda_C  # noqa: F401

    from vllm.models.glm5next.nvidia.ops import kcp

    ops = torch.ops._flashkda_C
    torch.manual_seed(3)
    heads, dim = 4, 128
    lengths = [37, 0, 16, 1, 300]
    cu = torch.tensor(np.r_[0, np.cumsum(lengths)], dtype=torch.int32, device="cuda")
    tokens, count = int(cu[-1]), len(lengths)
    q, k, g = (
        torch.randn(1, tokens, heads, dim, device="cuda").bfloat16() for _ in range(3)
    )
    v = (torch.randn(1, tokens, heads, dim, device="cuda") * 0.5).bfloat16()
    beta = torch.randn(1, tokens, heads, device="cuda").bfloat16()
    a_log = 0.5 * torch.randn(heads, device="cuda")
    dt_bias = 0.1 * torch.randn(heads, dim, device="cuda")
    workspace = torch.empty(
        ops.get_workspace_size(tokens, heads, count), dtype=torch.uint8, device="cuda"
    )
    # prepare_segments sizes the workspace without the op call.
    tiles = (tokens + 15) // 16 + count
    assert workspace.numel() == heads * tiles * kcp.WORKSPACE_BYTES_PER_TILE
    beta_t = beta[0].t().contiguous()
    ops.kcp_flash_prepare(
        q, k, g, beta_t, a_log, dt_bias, cu, workspace, dim**-0.5, LOWER_BOUND
    )
    rows = torch.tensor([6, 0, 3, 5, 1], device="cuda")
    summaries = torch.full(
        (8, heads, 2 * dim, dim), float("nan"), device="cuda"
    ).bfloat16()
    ops.kcp_flash_summary(v, beta_t, workspace, cu, rows, summaries)
    untouched = torch.ones(8, dtype=torch.bool)
    untouched[rows.cpu()] = False
    assert summaries[untouched.cuda()].isnan().all()
    summary = summaries[rows].float()

    def final_state(initial):
        out = torch.empty_like(q)
        final = torch.empty(count, heads, dim, dim, device="cuda")
        ops.fwd(
            q,
            k,
            v,
            g,
            beta,
            dim**-0.5,
            out,
            torch.empty_like(workspace),
            a_log,
            dt_bias,
            LOWER_BOUND,
            initial,
            final,
            cu,
            None,
            None,
        )
        return final

    zero = final_state(torch.zeros(count, heads, dim, dim, device="cuda"))
    torch.testing.assert_close(summary[:, :, :dim], zero, atol=1e-3, rtol=1e-2)
    initial = torch.randn(count, heads, dim, dim, device="cuda") * 0.3
    expected = final_state(initial)
    torch.testing.assert_close(
        summary[:, :, :dim] + initial @ summary[:, :, dim:],
        expected,
        atol=2e-2,
        rtol=2e-2,
    )
    empty = lengths.index(0)
    assert (summary[empty, :, :dim] == 0).all()
    assert torch.equal(
        summary[empty, :, dim:], torch.eye(dim, device="cuda").expand(heads, dim, dim)
    )


class _EmulatedGroup:
    """Serialized rank threads that exchange tensors only in all_gather."""

    def __init__(self, world: int):
        self.world = world
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(world)
        self.parts: list[torch.Tensor | None] = [None] * world
        self.local = threading.local()

    def _wait(self):
        self.lock.release()
        self.barrier.wait()
        self.lock.acquire()

    def all_gather(self, tensor, dim=0):
        self.parts[self.local.rank] = tensor
        self._wait()
        gathered = torch.cat(self.parts, dim)
        self._wait()
        return gathered

    def all_gather_buffer(self, shape, dtype, parts=1, key="ag_buffer"):
        output = torch.full(
            (parts * self.world * shape[0], *shape[1:]),
            float("nan"),
            dtype=dtype,
            device="cuda",
        )
        view = output.view(parts, self.world, *shape)
        return output, view[:, self.local.rank]

    def all_gather_in_place(self, output, stream=None):
        local = output.chunk(self.world)[self.local.rank].clone()
        output.copy_(self.all_gather(local))
        return output

    def run(self, fn):
        errors = []

        def body(rank):
            self.local.rank = rank
            with self.lock:
                try:
                    fn(rank)
                except BaseException as error:  # noqa: BLE001
                    errors.append(error)
                    self.barrier.abort()

        threads = [threading.Thread(target=body, args=(r,)) for r in range(self.world)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]


def _layer(heads, dim, cache):
    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention

    torch.manual_seed(1)
    channels = heads * dim
    convs = [
        SimpleNamespace(weight=torch.randn(channels, 1, 4, device="cuda") * 0.3)
        for _ in range(3)
    ]
    layer = SimpleNamespace(
        prefix="kda",
        kv_cache=cache,
        q_conv1d=SimpleNamespace(weight=convs[0].weight, bias=None),
        k_conv1d=convs[1],
        v_conv1d=convs[2],
        _merged_conv_weight=None,
        _conv_state_dim_first=True,
        A_log=0.5 * torch.randn(heads, device="cuda"),
        dt_bias=0.1 * torch.randn(channels, device="cuda"),
        local_num_heads=heads,
        head_dim=dim,
        local_projection_size=channels,
        kda_safe_gate=True,
        kda_lower_bound=LOWER_BOUND,
        kda_prefill_backend="triton",
    )
    for name in ("_forward", "_forward_kcp", "_conv_state_and_weights"):
        method = getattr(Glm5NextLinearAttention, name)
        setattr(layer, name, MethodType(getattr(method, "__wrapped__", method), layer))
    return layer


def _reference_metadata(lengths, computed, slots, num_decodes):
    """Ordinary GDN metadata of the unpartitioned batch, decodes first."""
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    qsl = torch.tensor(np.r_[0, np.cumsum(lengths)], dtype=torch.int32)
    nums_dict, batch_ptr, offsets = compute_causal_conv1d_metadata(
        qsl, device=torch.device("cuda")
    )
    return GDNAttentionMetadata(
        num_prefills=len(lengths) - num_decodes,
        num_prefill_tokens=int(sum(lengths[num_decodes:])),
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=int(sum(lengths)),
        has_initial_state=torch.tensor(computed, device="cuda") > 0,
        non_spec_query_start_loc=qsl.cuda(),
        non_spec_state_indices_tensor=slots,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=offsets,
    )


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize(
    "lengths,computed,num_decodes",
    [
        # Two decodes, a short prefill with empty slots, a continued prefill
        # and a fresh prefill spanning several chunks.
        ([1, 1, 5, 150, 70], [8, 3, 0, 64, 0], 2),
        # One short continued prefill: some ranks hold no prefill tokens.
        ([3], [5], 0),
    ],
)
@torch.inference_mode()
def test_kcp_layer_matches_unpartitioned_forward(
    monkeypatch, world, lengths, computed, num_decodes
):
    from vllm.models.glm5next.common import kda
    from vllm.models.glm5next.nvidia.ops import kcp

    torch.manual_seed(5)
    heads, dim, halo, padding = 2, 128, 3, 2
    channels = heads * dim
    num_slots = 2 * len(lengths) + 1
    # Distinct, shuffled cache slots; slot 0 is NULL_BLOCK_ID and the spare
    # slots must stay untouched.
    slots = torch.randperm(num_slots - 1, device="cuda")[: len(lengths)] + 1
    conv = torch.randn(num_slots, 3 * channels, halo, device="cuda").bfloat16()
    state = torch.randn(num_slots, heads, dim, dim, device="cuda") * 0.1
    tokens = sum(lengths)
    qkv = torch.randn(tokens, 3 * channels, device="cuda").bfloat16()
    gate = torch.randn(1, tokens, heads, dim, device="cuda").bfloat16()
    beta = torch.randn(1, tokens, heads, device="cuda").bfloat16()
    prefilling = np.arange(len(lengths)) >= num_decodes

    def forward(layer, metadata, qkv, gate, beta, out):
        context = SimpleNamespace(attn_metadata={"kda": metadata})
        monkeypatch.setattr(kda, "get_forward_context", lambda: context)
        layer._forward(qkv, gate, beta, out)

    reference_cache = (conv.clone(), state.clone())
    reference = torch.empty(1, tokens, heads, dim, device="cuda").bfloat16()
    forward(
        _layer(heads, dim, reference_cache),
        _reference_metadata(lengths, computed, slots, num_decodes),
        qkv,
        gate,
        beta,
        reference,
    )

    group = _EmulatedGroup(world)
    monkeypatch.setattr(kcp, "get_pcp_group", lambda: group)
    contexts: dict[int, SimpleNamespace] = {}
    monkeypatch.setattr(kda, "get_forward_context", lambda: contexts[group.local.rank])
    results = {}

    def run_rank(rank):
        layout, segments, _ = _hybrid_layout(world, rank, lengths, prefilling, computed)
        index = torch.tensor(
            [i for s in segments for i in range(*s.global_batch_slice.indices(tokens))],
            device="cuda",
            dtype=torch.int64,
        )
        pad = torch.zeros(padding, dtype=torch.int64, device="cuda")
        local = torch.cat((index, pad))
        metadata = _reference_metadata(lengths, computed, slots, num_decodes)
        metadata.pcp_layout = layout.with_state_indices(slots)
        contexts[rank] = SimpleNamespace(attn_metadata={"kda": metadata})
        cache = (conv.clone(), state.clone())
        out = torch.full((1, len(local), heads, dim), 7.0, device="cuda").bfloat16()
        # The layer sees q|k|v as a column slice of a wider projection.
        local_qkv = qkv[local]
        wide = torch.cat((local_qkv, torch.zeros_like(local_qkv[:, :7])), dim=1)
        _layer(heads, dim, cache)._forward(
            wide[:, : local_qkv.shape[1]], gate[:, local], beta[:, local], out
        )
        results[rank] = (index, out, cache)

    group.run(run_rank)

    spare = torch.ones(num_slots, dtype=torch.bool, device="cuda")
    spare[slots] = False
    for index, out, (rank_conv, rank_state) in results.values():
        torch.testing.assert_close(
            out[0, : len(index)], reference[0, index], atol=4e-3, rtol=2e-2
        )
        assert (out[0, len(index) :] == 0).all()
        torch.testing.assert_close(rank_conv, reference_cache[0], atol=0, rtol=0)
        # Decodes use the recurrent kernel and short prefills are stitched from
        # BF16 segment summaries, so states differ within output tolerance.
        torch.testing.assert_close(
            rank_state[slots], reference_cache[1][slots], atol=4e-3, rtol=2e-2
        )
        assert torch.equal(rank_state[spare], state[spare])
