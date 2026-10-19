# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.worker.gpu.async_utils import StepTimingSample
from vllm.v1.worker.gpu.spec_decode import adaptive_verification as adaptive_module
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    maybe_create_adaptive_verification_manager,
    resolve_adaptive_cudagraph_mode,
)
from vllm.v1.worker.gpu.spec_decode.undrafted_confidence import (
    UndraftedConfidenceEstimator,
)
from vllm.v1.worker.gpu.structured_outputs import _build_grammar_mapping


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
    manager.cost_tables = (np.zeros((num_steps + 1, num_reqs + 1)), verify_cost_ms)
    manager._max_total_logits = 1 << 30
    manager.num_bonus_tokens = 1
    manager.use_adaptive_draft_depth = False
    manager._last_draft_depth = num_steps
    manager._last_num_non_draft_tokens = 0
    manager._copy_events = [SimpleNamespace(synchronize=lambda: None)] * 2
    manager._pending_resets = []
    manager.undrafted_confidence = UndraftedConfidenceEstimator(num_reqs, num_steps)
    # Every slot's last proposal drafted the full depth unless a test says otherwise.
    manager._stale_draft_depths = [np.full(num_reqs, num_steps, dtype=np.int32)] * 2
    manager._last_idx_mapping = np.zeros(0, dtype=np.int64)
    return manager


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
    manager.use_adaptive_draft_depth = False
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
            draft_depth=manager.num_speculative_steps,
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


def make_depth_manager(
    verify_cost_ms: np.ndarray, draft_ms_per_step: float
) -> AdaptiveVerificationManager:
    confidences = np.full((2, 3), 0.9, dtype=np.float32)
    manager = make_manager(confidences, verify_cost_ms)
    manager._stale_confidences = [SimpleNamespace(np=confidences)] * 2  # type: ignore[list-item]
    manager.use_adaptive_draft_depth = True
    depths = np.arange(4, dtype=np.float64)[:, None]
    manager.cost_tables = (
        np.broadcast_to(depths * draft_ms_per_step, (4, 3)).copy(),
        verify_cost_ms,
    )
    return manager


def test_draft_depth_stops_where_no_draft_would_be_verified():
    # Any draft past the two bonus tokens costs 100x to verify, so the next
    # budget admits none and every draft step past the prefill is wasted.
    manager = make_depth_manager(np.array([1.0, 1.0, 1.0] + [100.0] * 6), 1.0)

    assert manager.get_draft_depth(np.array([0, 1]), 3) == 1
    assert manager._last_draft_depth == 1


def test_draft_depth_keeps_full_depth_when_drafts_are_cheap():
    manager = make_depth_manager(np.ones(9), 0.01)

    assert manager.get_draft_depth(np.array([0, 1]), 3) == 3


def test_undrafted_steps_are_planned_on_position_estimates():
    # The last proposal stopped after one step, so the deep columns hold leftovers
    # (0.01) rather than confidences. Planning must ignore them and use the
    # per-position estimate, here confidently high, and so draft deep again.
    manager = make_depth_manager(np.ones(9), 0.01)
    leftovers = np.array([[0.9, 0.01, 0.01], [0.9, 0.01, 0.01]], dtype=np.float32)
    manager._stale_confidences = [SimpleNamespace(np=leftovers)] * 2  # type: ignore[list-item]
    manager._stale_draft_depths = [np.ones(2, dtype=np.int32)] * 2
    manager.undrafted_confidence.sums[:] = 450.0
    manager.undrafted_confidence.counts[:] = 500.0

    assert manager.get_draft_depth(np.array([0, 1]), 3) == 3


def test_position_estimates_count_only_drafted_steps():
    estimates = UndraftedConfidenceEstimator(3, 3)
    confidences = np.array([[0.8, 0.6, 0.0], [0.5, 0.5, 0.5], [0.4, 0.0, 0.0]])

    estimates.update(confidences, np.array([2, 3, 1]), np.array([2, 0]))

    # Slot 0 drafted two steps, slot 2 one; slot 1 was not in the proposal.
    np.testing.assert_allclose(
        estimates.sums, [[0.8, 0.6, 0.0], [0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]
    )
    np.testing.assert_allclose(estimates.counts, [[1, 1, 0], [0, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(estimates.population_sums, [1.2, 0.6, 0.0])
    np.testing.assert_allclose(estimates.population_counts, [2, 1, 0])


def test_position_estimates_lean_on_the_population_until_drafted():
    # A request that has not drafted a step yet is planned on what other requests
    # saw there; its own confidences take over as it drafts the step.
    estimates = UndraftedConfidenceEstimator(2, 1)
    for _ in range(1000):
        estimates.update(np.array([[0.2], [0.0]]), np.array([1, 0]), np.arange(2))
    population = estimates.mean(np.array([1]))[0, 0]
    assert population == pytest.approx(0.2, abs=1e-3)

    for _ in range(1000):
        estimates.update(np.array([[0.0], [0.9]]), np.array([0, 1]), np.arange(2))
    assert estimates.mean(np.array([1]))[0, 0] > 0.85


def test_reused_slot_does_not_inherit_position_estimates():
    manager = make_manager(np.full((2, 2), 0.5, dtype=np.float32), np.ones(7))
    manager.use_adaptive_draft_depth = True
    manager._confidence_probs = torch.ones(2, 2)
    manager._draft_depths = np.zeros(2, dtype=np.int32)
    estimates = manager.undrafted_confidence
    estimates.update(np.full((2, 2), 0.9), np.full(2, 2), np.arange(2))

    manager.add_request(0)

    np.testing.assert_array_equal(estimates.counts[0], [0.0, 0.0])
    np.testing.assert_array_equal(estimates.counts[1], [1.0, 1.0])
    # The new request starts at the population mean, not the old request's.
    population = (estimates.population_sums + 0.5) / (estimates.population_counts + 1)
    np.testing.assert_allclose(estimates.mean(np.array([0]))[0], population)


def test_budget_never_admits_slots_past_the_drafted_depth():
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9]], dtype=np.float32), np.ones(7)
    )
    manager.use_adaptive_draft_depth = True
    manager._last_draft_depth = 1

    manager.get_num_tokens({"low": 3, "high": 3}, {"low": [1, 2], "high": [3, 4]})
    assert manager._batch_budget is not None
    valid_drafts, num_non_draft_tokens, draft_budget = manager._batch_budget

    assert valid_drafts == {"low": 1, "high": 1}
    assert num_non_draft_tokens == {"low": 1, "high": 1}
    assert draft_budget == 2


def test_shallow_profile_seeds_per_depth_draft_costs(monkeypatch):
    monkeypatch.setattr(
        adaptive_module,
        "get_tp_group",
        lambda: SimpleNamespace(broadcast_object=lambda obj, src: obj),
    )
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.req_states = SimpleNamespace(max_num_batched_tokens=64, max_num_reqs=8)
    manager.num_speculative_steps = 5
    manager.num_bonus_tokens = 1
    manager.use_adaptive_draft_depth = True

    batches = list(manager.batches_to_profile([8]))
    shallow = [b for b in batches if b.get("num_speculative_tokens") == 1]
    assert shallow and len(shallow) < len(batches)

    timings = [
        StepTimingSample(
            forward_ms=1.0,
            drafter_ms=1.0 if batch.get("num_speculative_tokens") == 1 else 5.0,
            num_target_tokens=batch["num_tokens"],
            num_reqs=8,
            full_cudagraph=True,
            draft_depth=batch.get(
                "num_speculative_tokens", manager.num_speculative_steps
            ),
        )
        for batch in batches
    ]
    manager.set_initial_cost_curves(timings)

    assert manager.cost_tables is not None
    # Each decode step past the prefill-only depth adds an equal share.
    np.testing.assert_allclose(manager.cost_tables[0][1:, 8], [1, 2, 3, 4, 5])


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


def test_zero_budget_keeps_one_grammar_row_per_scheduled_draft():
    # The scheduler sizes the grammar bitmask from the *scheduled* drafts
    # (len(drafts) + 1 rows per request), but a zero budget rewrites
    # cu_num_logits_np to bonus-only. Deriving the bitmask -> logits mapping
    # from those rewritten offsets drops rows and trips the
    # `num_masks == len(mapping)` assert in apply_grammar_bitmask.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9], [1.0, 1.0]], dtype=np.float32),
        np.ones(64),
    )
    manager.req_states.req_id_to_index["prefill"] = 2
    manager.req_states.num_computed_tokens_np = np.zeros(3, dtype=np.int32)
    manager.req_states.prefill_len.np = np.array([0, 0, 60], dtype=np.int32)
    manager._max_total_logits = 2  # < 3 requests * 1 bonus token

    scheduled_spec_decode_tokens = {"low": [1, 2], "high": [3, 4]}
    manager.get_num_tokens(
        {"low": 3, "high": 3, "prefill": 40}, scheduled_spec_decode_tokens
    )
    assert manager._batch_budget is not None
    assert manager._batch_budget[2] == 0

    req_ids = ["low", "high", "prefill"]
    num_draft_tokens_per_req = np.array([2, 2, 0], dtype=np.int32)
    _, cu_num_logits_np = manager.compact_batch(
        num_draft_tokens_per_req,
        np.array([3, 3, 40], dtype=np.int32),
        np.array([0, 3, 6, 7], dtype=np.int32),
    )

    mask_stride = manager.num_speculative_steps + manager.num_bonus_tokens
    mapping = _build_grammar_mapping(
        req_ids,
        req_ids,
        cu_num_logits_np,
        num_draft_tokens_per_req,
        manager.num_bonus_tokens,
        mask_stride,
    )

    num_bitmask_rows = sum(
        len(scheduled_spec_decode_tokens.get(req_id, ())) + 1 for req_id in req_ids
    )
    assert len(mapping) == num_bitmask_rows
    # (request, position) keys, so the kernel can mask rows the compacted
    # device layout no longer has room for.
    assert mapping == [0, 1, 2, 3, 4, 5, 6]
