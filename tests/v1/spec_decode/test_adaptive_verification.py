# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import parallel_state
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.worker.gpu.async_utils import StepTimingSample
from vllm.v1.worker.gpu.spec_decode import adaptive_verification as adaptive_module
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    assign_draft_token_budget,
    maybe_create_adaptive_verification_manager,
    resolve_adaptive_cudagraph_mode,
)


def make_manager(
    confidences: np.ndarray, verify_cost_ms: np.ndarray
) -> AdaptiveVerificationManager:
    num_reqs, num_steps = confidences.shape
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.num_speculative_steps = num_steps
    manager._stale_confidences = [SimpleNamespace(np=confidences)]  # type: ignore[list-item]  # CPU-only confidence buffer.
    manager._stale_idx = 0
    manager.req_states = SimpleNamespace(
        req_id_to_index={"low": 0, "high": 1},
        num_computed_tokens_np=np.ones(num_reqs, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.ones(num_reqs, dtype=np.int32)),
    )
    manager.cost_tables = (np.zeros(num_reqs + 1), verify_cost_ms)
    manager._max_total_logits = 1 << 30
    manager.num_bonus_tokens = 1
    return manager


@pytest.fixture(autouse=True)
def single_rank_tp_group(monkeypatch):
    """In-process tests have no TP group: its broadcasts return their input."""
    group = SimpleNamespace(
        broadcast_object=lambda obj, src=0: obj, broadcast=lambda t, src=0: t
    )
    monkeypatch.setattr(adaptive_module, "get_tp_group", lambda: group)


def tp_consensus_worker(rank: int, world_size: int, store: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    # A real TP group, minus the device communicator so it runs on CPU.
    parallel_state._TP = parallel_state.GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=0,
        torch_distributed_backend="gloo",
        use_device_communicator=False,
        use_message_queue_broadcaster=True,
        group_name="tp",
    )

    # CPU stand-ins for the Triton budget kernel and the pinned H2D copy.
    def assign_budget(probs, idx_mapping, capacities, budget, survival):
        capacities.copy_(global_topk_budget(probs, idx_mapping, capacities, budget))

    def tensor_h2d(array, *, out):
        return out.copy_(torch.from_numpy(array))

    adaptive_module.assign_draft_token_budget = assign_budget
    adaptive_module.async_tensor_h2d = tensor_h2d

    def layout(stale, current, verify_cost_ms):
        """Each rank's batch layout for one step with one draft per request."""
        manager = make_manager(stale, np.array(verify_cost_ms))
        manager._confidence_probs = torch.from_numpy(current)
        manager._batch_draft_capacity = torch.empty(2, dtype=torch.int32)
        manager._num_non_draft_tokens = torch.empty(2, dtype=torch.int32)
        manager._cu_num_logits = torch.empty(3, dtype=torch.int32)
        manager.query_start_loc = torch.empty(3, dtype=torch.int32)
        manager._draft_survival = None
        manager._copy_admitted_drafts_to_host = False
        num_tokens = manager.get_num_tokens(
            {"low": 2, "high": 2}, {"low": [0], "high": [0]}
        )
        cu_num_logits, query_start_loc, *_ = manager.reallocate_drafts(
            ["low", "high"], torch.arange(2)
        )
        layouts: list = [None] * world_size
        dist.all_gather_object(
            layouts, (num_tokens, cu_num_logits.tolist(), query_start_loc.tolist())
        )
        return layouts

    # Confidences differ across ranks by one fp32 ULP around a tie.
    up = np.nextafter(np.float32(0.5), np.float32(1))
    down = np.nextafter(np.float32(0.5), np.float32(0))
    # Budget: accepted/cost of 0 and 1 drafts tie at exactly 0.5.
    stale = np.array([[up if rank == 0 else down], [0.1]], dtype=np.float32)
    budget_layouts = layout(stale, stale, [1, 1, 1, 1.25, 100])
    # Boundaries: same budget (1 of 2 drafts), but the requests' current
    # confidences swap order, so each rank would admit a different draft.
    stale = np.full((2, 1), 0.5, dtype=np.float32)
    current = np.array([[up], [down]] if rank == 0 else [[down], [up]])
    boundary_layouts = layout(stale, current, [1, 1, 1, 1, 100])

    parallel_state._TP.destroy()
    dist.destroy_process_group()
    for layouts in (budget_layouts, boundary_layouts):
        assert all(layout == layouts[0] for layout in layouts), layouts


@pytest.mark.parametrize("world_size", [2, 4])
def test_tp_ranks_agree_on_batch_layout(world_size, tmp_path):
    """TP ranks must agree on the draft budget and request boundaries even when
    their draft confidences differ slightly."""
    mp.spawn(
        tp_consensus_worker,
        args=(world_size, str(tmp_path / "store")),
        nprocs=world_size,
    )


@pytest.mark.parametrize(
    ("mode", "piecewise_capture_available", "expected"),
    [
        ("FULL_DECODE_ONLY", True, "FULL_DECODE_ONLY"),
        ("FULL_DECODE_ONLY", False, "FULL_DECODE_ONLY"),
        ("FULL", True, "FULL_AND_PIECEWISE"),
        ("FULL", False, "FULL_DECODE_ONLY"),
        ("FULL_AND_PIECEWISE", True, "FULL_AND_PIECEWISE"),
        ("FULL_AND_PIECEWISE", False, "FULL_DECODE_ONLY"),
    ],
)
def test_resolve_adaptive_cudagraph_mode(mode, piecewise_capture_available, expected):
    assert (
        resolve_adaptive_cudagraph_mode(
            CUDAGraphMode[mode], piecewise_capture_available=piecewise_capture_available
        )
        == CUDAGraphMode[expected]
    )


@pytest.mark.parametrize(
    "target_support,target_bound,device_offsets,additional,error",
    [
        pytest.param(AttentionCGSupport.ALWAYS, None, True, None, None, id="always"),
        pytest.param(
            AttentionCGSupport.UNIFORM_BATCH, 8, True, None, None, id="bounded"
        ),
        pytest.param(
            AttentionCGSupport.UNIFORM_BATCH,
            None,
            True,
            None,
            "TargetBackend allows none",
            id="unsupported",
        ),
        pytest.param(
            AttentionCGSupport.UNIFORM_BATCH,
            7,
            True,
            None,
            "up to 8, but TargetBackend allows at most 7",
            id="too-narrow",
        ),
        pytest.param(
            AttentionCGSupport.UNIFORM_BATCH,
            8,
            False,
            None,
            "trims verification requests",
            id="host-query-lens",
        ),
        pytest.param(
            AttentionCGSupport.ALWAYS,
            None,
            True,
            (AttentionCGSupport.UNIFORM_BATCH, "EncoderBackend"),
            "EncoderBackend allows none",
            id="additional-group",
        ),
    ],
)
def test_manager_checks_target_varlen_cudagraph_bound(
    monkeypatch, target_support, target_bound, device_offsets, additional, error
):
    """Target builders must replay varlen decode graphs of the verification
    width; draft-only groups are not checked."""

    def group(backend_name, layer_name, support, bound=None, mismatch=True):
        class Builder(AttentionMetadataBuilder):
            _cudagraph_support = support

            def __init__(self):
                pass

            def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
                raise NotImplementedError

            if bound is not None:

                @classmethod
                def get_varlen_cudagraph_max_query_len(cls, *_args):
                    return bound

        backend = type(
            backend_name,
            (),
            {"supports_device_cpu_query_lens_mismatch": staticmethod(lambda: mismatch)},
        )
        builder = Builder()
        return SimpleNamespace(
            layer_names=[layer_name],
            backend=backend,
            kv_cache_spec=None,
            get_metadata_builder=lambda _index: builder,
        )

    groups = [
        [
            group(
                "TargetBackend", "target", target_support, target_bound, device_offsets
            ),
            group("DraftBackend", "draft", AttentionCGSupport.UNIFORM_BATCH),
        ]
    ]
    created = object()
    monkeypatch.setattr(
        adaptive_module,
        "AdaptiveVerificationManager",
        lambda *_args, **_kwargs: created,
    )

    with pytest.raises(ValueError, match=error) if error else nullcontext():
        manager = maybe_create_adaptive_verification_manager(
            enable_adaptive_verification=True,
            attn_groups=groups,
            req_states=SimpleNamespace(num_speculative_steps=7),
            query_start_loc=object(),
            num_bonus_tokens=1,
            max_total_logits=1,
            vllm_config=None,
            target_layer_names={"target"},
            additional_attn_cg_support=additional,
        )
        assert manager is created


def test_budget_stops_where_marginal_drafts_stop_paying_for_themselves():
    # Verification is cheap up to two extra tokens, then jumps 100x; only the
    # highest-confidence draft is worth the cheap slot.
    manager = make_manager(
        np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32),
        np.array([1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0]),
    )

    manager.get_num_tokens(
        {"low": 3, "high": 3},
        {"low": [1, 2], "high": [3, 4]},
    )
    assert manager._batch_budget is not None
    valid_drafts, num_non_draft_tokens, draft_budget = manager._batch_budget

    assert draft_budget == 1
    assert valid_drafts == {"low": 2, "high": 2}
    assert num_non_draft_tokens == {"low": 1, "high": 1}


def test_profiled_batches_seed_cost_curves_via_consumer(monkeypatch):
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.req_states = SimpleNamespace(max_num_batched_tokens=4096, max_num_reqs=64)
    manager.num_speculative_steps = 7
    manager.num_bonus_tokens = 1
    curves: dict[str, list[tuple[int, float]]] = {}
    monkeypatch.setattr(
        manager,
        "set_cost_curves",
        lambda draft, verify: curves.update(draft=draft, verify=verify),
    )

    timings = [
        StepTimingSample(
            forward_ms=float(batch["num_tokens"]),
            drafter_ms=1.0,
            num_target_tokens=batch["num_tokens"],
            num_reqs=batch["num_tokens"] // 8,
            # Only the captured sizes replay a graph; the tail sizes run eager.
            full_cudagraph=batch["num_tokens"] <= 1024,
        )
        for batch in manager.batches_to_profile([8, 1024])
    ]
    manager.set_initial_cost_curves(timings)

    # Tail beyond the last capture size: 1.5x then doubling to the max.
    assert curves["verify"] == [
        (8, 8.0),
        (1024, 1024.0),
        (1536, 1536.0),
        (2048, 2048.0),
        (4096, 4096.0),
    ]
    # Eager batches must not contribute to the draft curve: keyed by request
    # count they would land inside the captured range and, once made monotonic,
    # smear that eager cost across every larger request count.
    assert curves["draft"] == [(1, 1.0), (128, 1.0)]


def test_compact_batch_preserves_totals_and_bounds():
    # The CPU placeholder layout must keep the batch total equal to the GPU
    # total and every verification row within decode_query_len, or downstream
    # CPU metadata desyncs from the reallocated GPU boundaries.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9], [1.0, 1.0]], dtype=np.float32),
        np.array([1.0] * 44 + [100.0] * 3),
    )
    manager.req_states.req_id_to_index["prefill"] = 2
    manager.req_states.num_computed_tokens_np = np.zeros(3, dtype=np.int32)
    manager.req_states.prefill_len.np = np.array([0, 0, 60], dtype=np.int32)
    num_tokens = manager.get_num_tokens(
        {"low": 3, "high": 3, "prefill": 40},
        {"low": [1, 2], "high": [3, 4]},
    )
    scheduled = np.array([3, 3, 40], dtype=np.int32)
    drafts = np.array([2, 2, 0], dtype=np.int32)
    cu_num_logits_np = np.array([0, 3, 6, 7], dtype=np.int32)
    compacted, _ = manager.compact_batch(drafts, scheduled, cu_num_logits_np)

    assert int(compacted.sum()) == num_tokens
    num_steps = manager.num_speculative_steps
    assert (compacted[:2] <= 1 + num_steps).all()
    assert compacted[2] == 40


def test_budget_caps_at_one_rejection_sampler_chunk():
    # The chunked verification path cannot address the compacted logits
    # layout, so the budget must keep total logits within a single chunk.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9]], dtype=np.float32),
        np.ones(7),
    )
    manager._max_total_logits = 3  # 2 bonus logits + at most 1 draft
    manager.get_num_tokens(
        {"low": 3, "high": 3},
        {"low": [1, 2], "high": [3, 4]},
    )
    assert manager._batch_budget is not None
    _, _, draft_budget = manager._batch_budget
    assert draft_budget <= 1


def test_zero_budget_rebuilds_cpu_cu_num_logits():
    # When one bonus row per request already overflows a verification chunk, the
    # budget clamps to zero but the batch still needs chunking. Every capacity is
    # zeroed on device, so the CPU can name that layout exactly -- and must, since
    # _iter_request_chunks slices the compacted logits with these offsets.
    #
    # The third request is a chunked prefill (no drafts, still mid-prompt). The
    # runner gives *every* request num_bonus_tokens logits rows regardless
    # (num_logits = num_draft_tokens_per_req + num_bonus_tokens), so the rebuilt
    # offsets stay uniform rather than skipping non-verification rows.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9], [1.0, 1.0]], dtype=np.float32),
        np.ones(64),
    )
    manager.req_states.req_id_to_index["prefill"] = 2
    manager.req_states.num_computed_tokens_np = np.zeros(3, dtype=np.int32)
    manager.req_states.prefill_len.np = np.array([0, 0, 60], dtype=np.int32)
    manager._max_total_logits = 2  # < 3 requests * 1 bonus token

    manager.get_num_tokens(
        {"low": 3, "high": 3, "prefill": 40},
        {"low": [1, 2], "high": [3, 4]},
    )
    assert manager._batch_budget is not None
    _, _, draft_budget = manager._batch_budget
    assert draft_budget == 0

    scheduled = np.array([3, 3, 40], dtype=np.int32)
    drafts = np.array([2, 2, 0], dtype=np.int32)
    scheduled_cu_num_logits = np.array([0, 3, 6, 7], dtype=np.int32)
    compacted, cu_num_logits_np = manager.compact_batch(
        drafts, scheduled, scheduled_cu_num_logits
    )

    # One bonus row per request, matching cumsum(capacities + num_bonus_tokens)
    # with every capacity zeroed -- the prefill row included.
    expected = np.arange(4, dtype=np.int32) * manager.num_bonus_tokens
    assert np.array_equal(cu_num_logits_np, expected)
    assert cu_num_logits_np.dtype == scheduled_cu_num_logits.dtype
    # The prefill keeps its scheduled tokens; only drafts are dropped.
    assert np.array_equal(compacted, np.array([1, 1, 40], dtype=np.int32))


def global_topk_budget(
    confidence_probs: torch.Tensor,
    idx_mapping: torch.Tensor,
    capacities: torch.Tensor,
    draft_budget: int,
) -> torch.Tensor:
    survival = confidence_probs[idx_mapping].cumprod(dim=1)
    steps = torch.arange(survival.shape[1], device=survival.device)
    survival = survival.masked_fill(steps >= capacities[:, None], -float("inf"))
    admitted = torch.zeros(survival.numel(), dtype=torch.bool, device=survival.device)
    admitted[survival.flatten().topk(draft_budget).indices] = True
    return admitted.view_as(survival).sum(dim=1, dtype=torch.int32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_steps", [3, 8])
@pytest.mark.parametrize("num_reqs", [1, 2, 300])
def test_draft_budget_kernel_matches_global_topk(num_steps, num_reqs):
    """With distinct survivals the per-request admits equal a global top-k."""
    max_num_reqs = 512
    gen = torch.Generator(device="cuda").manual_seed(num_reqs * num_steps)
    confidence_probs = torch.rand(max_num_reqs, num_steps, device="cuda", generator=gen)
    idx_mapping = torch.randperm(max_num_reqs, device="cuda", generator=gen)
    idx_mapping = idx_mapping[:num_reqs]
    scheduled = torch.randint(
        1, num_steps + 1, (num_reqs,), dtype=torch.int32, device="cuda", generator=gen
    )
    scheduled[0] = num_steps
    survival = torch.empty(max_num_reqs * num_steps, dtype=torch.int32, device="cuda")
    num_scheduled = int(scheduled.sum())
    # The caller only invokes the kernel when the budget binds.
    budgets = {1, num_scheduled // 2, num_scheduled - 1}
    for draft_budget in sorted(b for b in budgets if 0 < b < num_scheduled):
        capacities = scheduled.clone()
        assign_draft_token_budget(
            confidence_probs, idx_mapping, capacities, draft_budget, survival
        )
        expected = global_topk_budget(
            confidence_probs, idx_mapping, scheduled, draft_budget
        )
        assert torch.equal(capacities, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_draft_budget_kernel_admits_exact_budget_on_ties():
    """Tied survivals are handed out in request order, across request blocks."""
    num_reqs, num_steps = 300, 3
    confidence_probs = torch.ones(num_reqs, num_steps, device="cuda")
    idx_mapping = torch.arange(num_reqs, device="cuda")
    capacities = torch.full((num_reqs,), num_steps, dtype=torch.int32, device="cuda")
    survival = torch.empty(num_reqs * num_steps, dtype=torch.int32, device="cuda")
    assign_draft_token_budget(confidence_probs, idx_mapping, capacities, 800, survival)
    assert capacities.tolist() == [3] * 266 + [2] + [0] * 33
