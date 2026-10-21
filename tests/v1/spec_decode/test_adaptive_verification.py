# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.worker.gpu.async_utils import StepTimingSample
from vllm.v1.worker.gpu.spec_decode import adaptive_verification as adaptive_module
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    maybe_create_adaptive_verification_manager,
    resolve_adaptive_cudagraph_mode,
)


def make_manager(
    confidences: np.ndarray, verify_cost_ms: np.ndarray, slot_cost_ms: float = 0.0
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
    manager.slot_cost_table = np.full(num_reqs + 1, slot_cost_ms)
    manager.base_cost_tables = None
    manager._cudagraph_limit = 0
    manager._max_total_logits = 1 << 30
    manager.num_bonus_tokens = 1
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
    # Each slot costs 0.3 on a base of 1: the high-confidence request's drafts
    # (0.9, 0.81 expected tokens) pay for theirs, the low one's (0.1, 0.01) don't.
    manager = make_manager(
        np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32),
        np.ones(7),
        slot_cost_ms=0.3,
    )

    manager.get_num_tokens(
        {"low": 3, "high": 3},
        {"low": [1, 2], "high": [3, 4]},
    )
    assert manager._batch_budget is not None
    valid_drafts, num_non_draft_tokens, draft_budget = manager._batch_budget

    assert draft_budget == 2
    assert valid_drafts == {"low": 2, "high": 2}
    assert num_non_draft_tokens == {"low": 1, "high": 1}


def test_profiled_batches_seed_cost_curves_via_consumer(monkeypatch):
    monkeypatch.setenv("VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN", "1024")
    max_num_reqs = 64
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.req_states = SimpleNamespace(
        max_num_batched_tokens=4096, max_num_reqs=max_num_reqs
    )
    manager.num_speculative_steps = 7
    manager.num_bonus_tokens = 1
    curves: dict[str, list[tuple[int, float]]] = {}
    monkeypatch.setattr(
        manager,
        "set_cost_curves",
        lambda draft, verify, **rest: curves.update(draft=draft, verify=verify, **rest),
    )

    # A step costs 0.01 per token it runs plus 1e-4 per context token of KV
    # each request reads.
    def forward_ms(num_tokens: int, num_reqs: int, context_len: int) -> float:
        return 0.01 * num_tokens + 1e-4 * num_reqs * context_len

    timings = []
    for batch in manager.batches_to_profile([8, 1024]):
        num_tokens = batch["num_tokens"]
        num_reqs = batch.get("num_reqs") or min(num_tokens, max_num_reqs)
        # Only the captured sizes replay a graph; the tail sizes run eager.
        graphed = num_tokens <= 1024
        timings.append(
            StepTimingSample(
                forward_ms=forward_ms(num_tokens, num_reqs, batch["context_len"]),
                drafter_ms=1.0 if graphed else 50.0,
                num_target_tokens=num_tokens,
                num_reqs=num_reqs,
                full_cudagraph=graphed,
                context_len=batch["context_len"],
            )
        )
    manager.set_initial_cost_curves(timings)

    # Tail beyond the last capture size: 1.5x then doubling to the max.
    sizes = [8, 1024, 1536, 2048, 4096]
    assert [n for n, _ in curves["verify"]] == sizes
    token_curve = curves["token_curve"]
    assert [n for n, _ in token_curve] == sizes
    assert [ms for _, ms in token_curve] == pytest.approx([0.01 * n for n in sizes])
    # The verify curve prices every token as another request reading 1024
    # tokens of context; a draft slot only adds a token to a request already
    # in the batch, so it costs the per-token price alone.
    ((slope_reqs, slot_ms),) = curves["slope_curve"]
    assert (slope_reqs, slot_ms) == (8, pytest.approx(0.01))
    ((kv_reqs, kv_ms),) = curves["kv_curve"]
    assert (kv_reqs, kv_ms) == (8, pytest.approx(1e-4))
    # Eager batches must not contribute to the draft curve: keyed by request
    # count they would land inside the captured range and, once made monotonic,
    # smear that eager cost across every larger request count.
    assert curves["draft"] == [(8, 1.0), (64, 1.0)]


def test_base_charges_actual_context_and_slots_their_own_price():
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9]], dtype=np.float32),
        np.full(64, 100.0),
        slot_cost_ms=0.01,
    )
    token_cost_ms = 0.01 * np.arange(64)
    manager.base_cost_tables = (token_cost_ms, np.full(3, 1e-4))
    manager._cudagraph_limit = 64
    slots = np.array([0, 1], dtype=np.int32)

    # Two decoding requests with 100 and 300 tokens of context: the base is
    # the two tokens they run plus the 400 context tokens they read, not the
    # verify curve's long-context price.
    manager.req_states.num_computed_tokens_np = np.array([100, 300], dtype=np.int32)
    costs = manager._budget_costs_ms(2, 2, slots, 4)
    assert costs == pytest.approx(0.02 + 400e-4 + 0.01 * np.arange(5))

    # A longer context raises the base but not what each slot adds.
    manager.req_states.num_computed_tokens_np = np.array([1000, 3000], dtype=np.int32)
    longer = manager._budget_costs_ms(2, 2, slots, 4)
    assert longer[0] == pytest.approx(0.02 + 4000e-4)
    assert np.diff(longer) == pytest.approx(np.diff(costs))


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
