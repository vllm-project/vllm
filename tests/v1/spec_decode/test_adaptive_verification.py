# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np

from vllm.v1.attention.selector import _uses_device_decided_verification_lengths
from vllm.v1.worker.gpu.async_utils import StepTimingSample
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
)
from vllm.v1.worker.gpu.spec_decode.dflash.adaptive_k import (
    DFlashAdaptiveKManager,
    DFlashAdaptiveKPolicy,
    get_dflash_k_candidates,
)
from vllm.v1.worker.gpu.structured_outputs import _build_grammar_mapping


def make_manager(
    confidences: np.ndarray, verify_cost_ms: np.ndarray
) -> AdaptiveVerificationManager:
    num_reqs, num_steps = confidences.shape
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.num_speculative_steps = num_steps
    manager._stale_confidences = [SimpleNamespace(np=confidences)]
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
    valid_drafts, num_non_draft_tokens, draft_budget = manager._batch_budget

    assert draft_budget == 1
    assert valid_drafts == {"low": 2, "high": 2}
    assert num_non_draft_tokens == {"low": 1, "high": 1}


def test_profiled_batches_seed_cost_curves_via_consumer():
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.req_states = SimpleNamespace(max_num_batched_tokens=4096, max_num_reqs=64)
    manager.num_speculative_steps = 7
    manager.num_bonus_tokens = 1
    curves: dict[str, list[tuple[int, float]]] = {}
    manager.set_cost_curves = lambda draft, verify: curves.update(
        draft=draft, verify=verify
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


def _make_dflash_policy(
    draft_cost_ms: np.ndarray,
    verify_cost_ms: np.ndarray,
    max_k: int = 3,
) -> DFlashAdaptiveKPolicy:
    policy = DFlashAdaptiveKPolicy(max_k=max_k, history_weight=1.0)
    shaped_verify = np.full((len(draft_cost_ms), max_k + 1), np.inf)
    for batch_size in range(1, len(draft_cost_ms)):
        for k in policy.candidates:
            total_tokens = batch_size * (k + 1)
            if total_tokens < len(verify_cost_ms):
                shaped_verify[batch_size, k] = verify_cost_ms[total_tokens]
    policy.set_cost_tables(draft_cost_ms, shaped_verify)
    return policy


def test_dflash_adaptive_k_uses_graph_covered_verify_cost():
    draft = np.full(9, 0.2)
    verify = np.ones(33)
    policy = _make_dflash_policy(draft, verify)

    assert policy.select_k(batch_size=4) == 3


def test_dflash_adaptive_k_uses_compact_graph_buckets():
    assert get_dflash_k_candidates(15) == [0, 1, 3, 7, 15]
    assert DFlashAdaptiveKPolicy._batch_bucket(0) == 0


def test_dflash_profiles_real_batch_and_query_length_shapes():
    manager = DFlashAdaptiveKManager.__new__(DFlashAdaptiveKManager)
    manager.req_states = SimpleNamespace(max_num_batched_tokens=2048, max_num_reqs=32)
    manager.num_speculative_steps = 15
    manager.policy = SimpleNamespace(candidates=[0, 1, 3, 7, 15])

    batches = list(manager.batches_to_profile([1, 2, 4, 8, 16, 32, 64, 128, 512]))

    assert {
        "num_tokens": 32 * 16,
        "uniform_decode_query_len": 16,
        "profile_verify": True,
        "context_len": 8192,
    } in batches
    assert {
        "num_tokens": 8 * 8,
        "uniform_decode_query_len": 8,
        "profile_verify": True,
        "context_len": 8192,
    } in batches


def test_dflash_shape_costs_distinguish_equal_total_token_counts(monkeypatch):
    manager = DFlashAdaptiveKManager.__new__(DFlashAdaptiveKManager)
    manager.req_states = SimpleNamespace(max_num_batched_tokens=2048, max_num_reqs=32)
    manager.num_speculative_steps = 15
    manager._capture_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 512}
    manager._outcome_buffers = []
    manager._selected_k_by_batch = {}
    manager._selection_uses_by_batch = {}
    manager._global_k_cap = 15
    manager.current_k = 15
    captured: dict[str, np.ndarray] = {}
    manager.policy = SimpleNamespace(
        candidates=[0, 1, 3, 7, 15],
        set_cost_tables=lambda draft, verify: captured.update(
            draft=draft, verify=verify
        ),
        reset_history=lambda: None,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.dflash.adaptive_k.get_tp_group",
        lambda: SimpleNamespace(broadcast_object=lambda value, src: value),
    )
    samples = [
        # Same total token count, materially different per-request query shape.
        StepTimingSample(1.0, 0.1, 32, 32, True),
        StepTimingSample(2.0, 0.2, 2, 1, True),
        StepTimingSample(3.0, 0.3, 4, 1, True),
        StepTimingSample(4.0, 0.4, 8, 1, True),
        StepTimingSample(9.0, 0.5, 32, 2, True),
    ]

    manager.set_initial_cost_curves(samples)

    assert captured["verify"][32, 0] == 1.0
    assert captured["verify"][2, 15] == 9.0


def test_dflash_adaptive_k_preserves_full_draft_for_serial_decode():
    draft = np.full(17, 100.0)
    verify = np.ones(257)
    policy = _make_dflash_policy(draft, verify, max_k=15)

    assert policy.select_k(batch_size=1) == 15


def test_dflash_adaptive_k_falls_back_to_baseline_on_graph_miss():
    draft = np.full(65, 0.5)
    verify = np.ones(257)
    verify[33:] = 10.0
    policy = _make_dflash_policy(draft, verify)

    assert policy.select_k(batch_size=32) == 0


def test_dflash_adaptive_k_uses_observed_accepted_prefix():
    draft = np.full(9, 0.2)
    verify = np.ones(33)
    verify[5:] = 1.5
    policy = _make_dflash_policy(draft, verify)
    assert policy.select_k(batch_size=4) == 3

    policy.record_outcomes(
        num_sampled=np.ones(4, dtype=np.int32),
        num_draft_tokens=np.full(4, 3, dtype=np.int32),
    )

    assert policy.select_k(batch_size=4) == 3


def test_dflash_adaptive_k_calibrates_shared_runtime_overhead():
    draft = np.full(65, 0.2)
    verify = np.ones(1025)
    policy = _make_dflash_policy(draft, verify, max_k=15)
    assert policy.select_k(batch_size=32) == 15

    # The runtime interval also includes scheduler and sampling work that every
    # K pays. Treating it as K=15-only cost would falsely disable drafting.
    for _ in range(2):
        policy.record_runtime(batch_size=31, k=15, num_sampled=32, elapsed_ms=64.0)

    assert policy.select_k(batch_size=32) == 15


def test_dflash_adaptive_k_disables_drafting_after_rejections():
    draft = np.full(65, 0.2)
    verify = np.ones(1025)
    policy = _make_dflash_policy(draft, verify, max_k=15)
    assert policy.select_k(batch_size=32) == 15

    policy.record_outcomes(
        num_sampled=np.ones(32, dtype=np.int32),
        num_draft_tokens=np.full(32, 15, dtype=np.int32),
    )

    assert policy.select_k(batch_size=32) == 0


def test_dflash_adaptive_k_keeps_serial_friendly_small_batches():
    draft = np.full(17, 0.2)
    verify = np.ones(257)
    policy = _make_dflash_policy(draft, verify, max_k=15)
    for _ in range(4):
        policy.record_runtime(batch_size=8, k=15, num_sampled=8, elapsed_ms=64.0)

    assert policy.select_k(batch_size=8) != 0


def test_dflash_adaptive_k_never_disables_drafting_for_small_batches():
    draft = np.full(17, 100.0)
    verify = np.ones(257)
    policy = _make_dflash_policy(draft, verify, max_k=15)
    policy.record_outcomes(
        num_sampled=np.ones(8, dtype=np.int32),
        num_draft_tokens=np.full(8, 15, dtype=np.int32),
    )

    assert policy.select_k(batch_size=8) > 0


def test_dflash_profile_outcomes_do_not_seed_runtime_history():
    draft = np.full(9, 0.2)
    verify = np.ones(33)
    policy = _make_dflash_policy(draft, verify)
    policy.record_outcomes(
        num_sampled=np.ones(4, dtype=np.int32),
        num_draft_tokens=np.full(4, 3, dtype=np.int32),
    )
    assert policy.select_k(batch_size=4) == 3

    policy.reset_history()

    assert policy.select_k(batch_size=4) == 3


def test_dflash_adaptive_k_trims_current_verification_batch():
    manager = DFlashAdaptiveKManager.__new__(DFlashAdaptiveKManager)
    manager.num_speculative_steps = 15
    manager.select_k = lambda batch_size: 0

    num_tokens = manager.get_num_tokens(
        {"r0": 16, "r1": 16},
        {"r0": list(range(15)), "r1": list(range(15))},
    )

    assert num_tokens == 2
    assert manager.batch_query_len == 1
    assert manager._batch_budget == (
        {"r0": 0, "r1": 0},
        {"r0": 1, "r1": 1},
        0,
    )
    assert not manager.consume_unmodified_batch()


def test_dflash_full_k_preserves_the_original_verification_path():
    manager = DFlashAdaptiveKManager.__new__(DFlashAdaptiveKManager)
    manager.num_speculative_steps = 15
    manager.select_k = lambda batch_size: 15
    manager._write_idx = 0
    manager._runtime_start_events = [SimpleNamespace(record=lambda: None)]
    manager._pending_runtime = [None]
    manager._pending_draft_counts = [None]

    num_tokens = manager.get_num_tokens(
        {"r0": 16, "r1": 16},
        {"r0": list(range(15)), "r1": list(range(15))},
    )

    assert num_tokens == 32
    assert manager.consume_unmodified_batch()
    assert manager._batch_budget is None


def test_dflash_adaptive_k_holds_decision_between_updates():
    manager = DFlashAdaptiveKManager.__new__(DFlashAdaptiveKManager)
    manager._outcome_buffers = []
    manager._selected_k_by_batch = {}
    manager._selection_uses_by_batch = {}
    manager._global_k_cap = 15
    manager.current_k = 15
    manager.decision_interval = 2
    choices = iter((15, 0))
    manager.policy = SimpleNamespace(select_k=lambda batch_size: next(choices))

    assert manager.select_k(32) == 15
    assert manager.select_k(32) == 15
    assert manager.select_k(32) == 0


def test_dflash_empty_batch_does_not_disable_drafting():
    manager = DFlashAdaptiveKManager.__new__(DFlashAdaptiveKManager)
    manager._outcome_buffers = []
    manager._global_k_cap = 15
    manager.current_k = 15
    manager.policy = SimpleNamespace(select_k=lambda batch_size: 0)

    assert manager.select_k(0) == 15
    assert manager._global_k_cap == 15


def test_dflash_outcome_poll_does_not_synchronize_the_gpu():
    manager = DFlashAdaptiveKManager.__new__(DFlashAdaptiveKManager)
    event = SimpleNamespace(
        query=lambda: False,
        synchronize=lambda: (_ for _ in ()).throw(AssertionError("GPU sync")),
    )
    manager._copy_events = [event]
    manager._pending_draft_counts = [np.ones(2, dtype=np.int32)]
    manager._pending_runtime = [None]

    assert not manager._consume_outcomes(0, wait=False)
    assert manager._pending_draft_counts[0] is not None


def test_dflash_adaptive_k_is_monotonic_within_graph_batch_bucket():
    manager = DFlashAdaptiveKManager.__new__(DFlashAdaptiveKManager)
    manager._outcome_buffers = []
    manager._selected_k_by_batch = {}
    manager._selection_uses_by_batch = {}
    manager._global_k_cap = 15
    manager.current_k = 15
    manager.decision_interval = 1
    choices = iter((15, 3, 7, 0))
    manager.policy = SimpleNamespace(select_k=lambda batch_size: next(choices))

    assert manager.select_k(32) == 15
    assert manager.select_k(31) == 3
    assert manager.select_k(32) == 3
    assert manager.select_k(31) == 0
    assert manager.proposal_k(32) == 0
    assert manager.select_k(8) == 0


def test_only_dspark_uses_device_decided_verification_lengths():
    dspark = SimpleNamespace(method="dspark", enable_adaptive_verification=True)
    dflash = SimpleNamespace(method="dflash", enable_adaptive_verification=True)

    assert _uses_device_decided_verification_lengths(dspark)
    assert not _uses_device_decided_verification_lengths(dflash)
