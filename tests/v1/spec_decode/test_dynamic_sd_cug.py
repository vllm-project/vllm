#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.spec_decode.dynamic.drafting_manager import (
    AdaptiveDraftingManager,
    DynamicSDDraftingManager,
)
from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
    allocate_draft_token_budget_from_confidence,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _mock_cudagraph_runtime(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: None,
    )


def test_flash_attention_supports_device_query_lengths():
    """FA2 and FA3 must both meet adaptive verification's target requirement."""
    assert FlashAttentionBackend.supports_device_query_lengths()


def test_adaptive_verification_supports_structured_outputs():
    """Structured decode batches can use device-side adaptive lengths."""
    req_states = _make_req_states(["req"])
    drafting_manager = _make_drafting_manager(
        [(1, 1, 2)],
        max_num_reqs=1,
        max_num_spec_tokens=2,
        req_states=req_states,
    )
    scheduler_output = _make_scheduler_output({"req": 3}, {"req": [1, 2]})

    assert isinstance(drafting_manager, AdaptiveDraftingManager)
    assert drafting_manager.plan_batch(scheduler_output) == (
        3,
        2,
    )


@pytest.mark.parametrize(
    ("scheduled_tokens_b", "draft_tokens_b"),
    ((3, [5, 6]), (5, [5, 6, -1, -1])),
)
def test_adaptive_verification_caps_irregular_draft_windows(
    scheduled_tokens_b,
    draft_tokens_b,
):
    drafting_manager = _make_drafting_manager(
        [(1, 2, 2)],
        max_num_reqs=2,
        max_num_spec_tokens=4,
        req_states=_make_req_states(["a", "b"]),
    )
    scheduler_output = _make_scheduler_output(
        {"a": 5, "b": scheduled_tokens_b},
        {
            "a": [1, 2, 3, 4],
            "b": draft_tokens_b,
        },
    )

    num_tokens, draft_token_budget = drafting_manager.plan_batch(scheduler_output)
    assert (num_tokens, draft_token_budget) == (6, 4)


@pytest.mark.parametrize(
    ("scheduled_tokens", "draft_tokens", "prefill_req_ids", "expected_num_tokens"),
    [
        (
            {"a": 5, "b": 5},
            {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]},
            set(),
            6,
        ),
        (
            {"a": 5, "b": 5},
            {"a": [1, 2, 3, 4], "b": [-1, -1, -1, -1]},
            {"b"},
            3,
        ),
        ({"a": 5, "b": 7}, {"a": [1, 2, 3, 4]}, {"b"}, 9),
        ({"a": 5}, {"a": [1, 2, -1, -1]}, set(), 4),
    ],
)
def test_adaptive_drafting_manager(
    scheduled_tokens, draft_tokens, prefill_req_ids, expected_num_tokens
):
    drafting_manager = _make_drafting_manager(
        [(1, 1, 3), (2, 2, 2), (3, 3, 1)],
        max_num_reqs=3,
        max_num_spec_tokens=4,
        req_states=_make_req_states(scheduled_tokens, prefill_req_ids),
    )
    scheduler_output = _make_scheduler_output(scheduled_tokens, draft_tokens)
    num_tokens, _ = drafting_manager.plan_batch(scheduler_output)
    assert num_tokens == expected_num_tokens


def test_adaptive_drafting_manager_builds_mixed_prefill_layout():
    req_states = _make_req_states(["decode", "prefill"], {"prefill"})
    speculator = MagicMock()
    drafting_manager = _make_drafting_manager(
        [(1, 2, 2), (3, 3, 1)],
        max_num_reqs=3,
        max_num_spec_tokens=4,
        req_states=req_states,
        speculator=speculator,
    )
    scheduler_output = _make_scheduler_output(
        {"decode": 5, "prefill": 4},
        {"decode": [1, 2, 3, 4]},
    )
    speculator.allocate_draft_token_budget.return_value = torch.tensor(
        [1, 0], dtype=torch.int32
    )
    query_start_loc = torch.empty(4, dtype=torch.int32)

    num_draft_tokens, cu_num_logits, query_start_loc_np = (
        drafting_manager.prepare_verification_layout(
            scheduler_output,
            ["decode", "prefill"],
            torch.tensor([0, 1], dtype=torch.int32),
            draft_token_budget=1,
            query_start_loc=query_start_loc,
            num_reqs_padded=2,
        )
    )

    assert num_draft_tokens.tolist() == [1, 0]
    assert query_start_loc.tolist() == [0, 2, 6, 6]
    assert query_start_loc_np.tolist() == [0, 2, 6]
    assert cu_num_logits.tolist() == [0, 2, 3]
    assert speculator.allocate_draft_token_budget.call_args.args[2].tolist() == [
        4,
        0,
    ]


def test_adaptive_drafting_manager_uses_worker_request_order():
    req_states = _make_req_states(["long", "short"])
    speculator = MagicMock()
    drafting_manager = _make_drafting_manager(
        [(1, 2, 2)],
        max_num_reqs=2,
        max_num_spec_tokens=4,
        req_states=req_states,
        speculator=speculator,
    )
    scheduler_output = _make_scheduler_output(
        {"long": 5, "short": 2},
        {"long": [1, 2, 3, 4], "short": [-1]},
    )
    speculator.allocate_draft_token_budget.return_value = torch.tensor(
        [1, 2], dtype=torch.int32
    )
    query_start_loc = torch.empty(3, dtype=torch.int32)

    drafting_manager.prepare_verification_layout(
        scheduler_output,
        ["short", "long"],
        torch.tensor([1, 0], dtype=torch.int32),
        draft_token_budget=3,
        query_start_loc=query_start_loc,
        num_reqs_padded=2,
    )

    assert speculator.allocate_draft_token_budget.call_args.args[2].tolist() == [
        1,
        4,
    ]
    assert query_start_loc.tolist() == [0, 2, 5]


def test_adaptive_verification_allocator_respects_draft_caps():
    confidence_logits = torch.zeros((2, 3))

    allocated = allocate_draft_token_budget_from_confidence(
        confidence_logits,
        draft_token_budget=3,
        draft_token_caps=torch.tensor([1, 3], dtype=torch.int32),
    )

    assert allocated.tolist() == [1, 2]


def test_adaptive_drafting_manager_treats_async_placeholders_as_drafts():
    drafting_manager = _make_drafting_manager(
        [(1, 1, 2)],
        max_num_reqs=1,
        max_num_spec_tokens=4,
        req_states=_make_req_states(["decode"]),
    )
    scheduler_output = _make_scheduler_output(
        {"decode": 5},
        {"decode": [-1, -1, -1, -1]},
    )

    assert drafting_manager.plan_batch(scheduler_output) == (3, 2)


def _make_scheduler_output(scheduled_tokens, draft_tokens):
    return SimpleNamespace(
        num_scheduled_tokens=scheduled_tokens,
        scheduled_spec_decode_tokens=draft_tokens,
        total_num_scheduled_tokens=sum(scheduled_tokens.values()),
    )


def _make_req_states(req_ids, prefill_req_ids=()):
    req_ids = list(req_ids)
    prefill_req_ids = set(prefill_req_ids)
    return SimpleNamespace(
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
        num_computed_prefill_tokens=[
            0 if req_id in prefill_req_ids else 1 for req_id in req_ids
        ],
        prefill_len=SimpleNamespace(np=[1] * len(req_ids)),
    )


def _make_drafting_manager(
    schedule,
    max_num_reqs,
    max_num_spec_tokens,
    req_states,
    speculator=None,
):
    return DynamicSDDraftingManager(
        schedule,
        max_num_reqs=max_num_reqs,
        max_num_spec_tokens=max_num_spec_tokens,
        device=torch.device("cpu"),
        speculator=speculator if speculator is not None else MagicMock(),
        req_states=req_states,
    )


def _create_vllm_config_for_dsd(
    max_num_seqs: int,
    max_spec_tokens: int,
    *,
    cudagraph_mode: str = "FULL_AND_PIECEWISE",
    use_dynamic_sd: bool = True,
    adaptive_verification: bool = False,
    num_spec_per_batch_size: list[tuple[int, int, int]] | None = None,
) -> MagicMock:
    """Create a minimal config that exercises DSD cudagraph dispatch.

    The test uses an exact capture-size grid so that every valid uniform decode
    shape has a directly matching FULL graph candidate.

    ``num_spec_per_batch_size`` lets a test supply an explicit DSD schedule of
    ``(range_start, range_end, num_speculative_tokens)`` tuples. When omitted,
    a schedule covering every query length in ``[1, max_decode_query_len]`` is
    generated.
    """

    max_decode_query_len = max_spec_tokens + 1
    max_capture_tokens = max_num_seqs * max_decode_query_len

    compilation_config = CompilationConfig(
        cudagraph_mode=cudagraph_mode,
        cudagraph_capture_sizes=list(range(1, max_capture_tokens + 1)),
    )
    compilation_config.max_cudagraph_capture_size = max_capture_tokens
    compilation_config.post_init_cudagraph_sizes()

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=max_num_seqs,
    )
    vllm_config.parallel_config = ParallelConfig()
    # num_speculative_tokens is the max K (num_speculative_steps). The manager
    # recovers num_new_sampled_tokens_per_step as
    # decode_query_len - num_speculative_tokens; with decode_query_len =
    # max_spec_tokens + 1 this yields the normal per-step bonus of 1.
    vllm_config.num_speculative_tokens = max_spec_tokens

    speculative_config = MagicMock()
    speculative_config.uses_dynamic_speculative_decoding.return_value = use_dynamic_sd
    speculative_config.adaptive_verification = adaptive_verification
    if use_dynamic_sd:
        # DSD reads the per-batch-size schedule; a schedule entry with K
        # speculative tokens maps to decode query length K + 1. By default
        # provide every query length in [1, max_decode_query_len] (i.e. K in
        # [0, max_spec_tokens]) so the manager captures a FULL decode graph for
        # each uniform shape.
        if num_spec_per_batch_size is None:
            num_spec_per_batch_size = [
                (qlen, qlen, qlen - 1) for qlen in range(1, max_decode_query_len + 1)
            ]
        speculative_config.num_speculative_tokens_per_batch_size = (
            num_spec_per_batch_size
        )
    else:
        speculative_config.num_speculative_tokens_per_batch_size = None
    vllm_config.speculative_config = speculative_config

    return vllm_config


def test_dynamic_sd_full_cudagraph_covers_all_uniform_decode_shapes(monkeypatch):
    """Dynamic SD should create FULL decode candidates for every k in [1, K+1].

    This validates the MRv2 CudaGraphManager path directly: once candidate
    shapes have been built, dispatch() should pick a FULL graph for every
    uniform decode batch shape produced by DSD up to max_num_seqs.
    """

    max_num_seqs = 512
    max_spec_tokens = 7
    max_decode_query_len = max_spec_tokens + 1

    # CudaGraphManager consults PP rank helpers during initialization even
    # though this test only exercises CPU-side candidate generation.
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    vllm_config = _create_vllm_config_for_dsd(
        max_num_seqs=max_num_seqs,
        max_spec_tokens=max_spec_tokens,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=max_decode_query_len,
    )

    # dispatch() only uses the precomputed candidate table after graphs are
    # considered captured. The actual graph objects are irrelevant here.
    manager._graphs_captured = True

    for num_reqs in range(1, max_num_seqs + 1):
        for max_query_len in range(1, max_decode_query_len + 1):
            # Uniform decode means every request contributes the same number of
            # tokens, so the total token count is exactly num_reqs * query_len.
            num_tokens = num_reqs * max_query_len
            uniform_tok_count = gpu_cudagraph_utils.get_uniform_token_count(
                num_reqs,
                num_tokens,
                max_query_len,
            )

            # The scheduler should mark every one of these shapes as a uniform
            # decode batch, which is what enables FULL decode graph selection.
            assert uniform_tok_count == max_query_len

            desc = manager.dispatch(
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                uniform_token_count=uniform_tok_count,
                num_active_loras=0,
            )

            # With DSD enabled, MRv2 should have captured a FULL candidate for
            # every k in [1, K+1], so dispatch should stay on the FULL path.
            assert desc.cg_mode == CUDAGraphMode.FULL
            assert desc.uniform_token_count == max_query_len
            assert desc.num_tokens == num_tokens
            assert desc.num_reqs == num_reqs
            assert desc.num_active_loras == 0


def test_dynamic_sd_non_uniform_batch_falls_back_to_piecewise(monkeypatch):
    """Non-varlen DSD should use PIECEWISE for a nonuniform decode batch.

    FULL DSD graphs are captured separately for each decode query length k.
    When runtime tokens are not uniform, uniform_token_count is None and those
    FULL candidates should be skipped in favor of the mixed-batch PIECEWISE
    graph under FULL_AND_PIECEWISE mode.
    """

    max_spec_tokens = 4

    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    vllm_config = _create_vllm_config_for_dsd(
        max_num_seqs=512,
        max_spec_tokens=max_spec_tokens,
        cudagraph_mode="FULL_AND_PIECEWISE",
        use_dynamic_sd=True,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=max_spec_tokens + 1,
    )
    manager._graphs_captured = True

    # This shape is intentionally non-uniform: 3 tokens across 2 requests
    # cannot correspond to a single per-request query length.
    desc = manager.dispatch(
        num_reqs=2,
        num_tokens=3,
        uniform_token_count=None,
        num_active_loras=0,
    )

    assert desc.cg_mode == CUDAGraphMode.PIECEWISE
    assert desc.uniform_token_count is None
    assert desc.num_reqs is None
    assert desc.num_tokens == 3
    assert desc.num_active_loras == 0


def test_adaptive_verification_non_uniform_batch_uses_full_cudagraph(monkeypatch):
    """Adaptive verification should select its bounded FULL descriptor.

    Both the adaptive-verification FULL descriptor and the mixed-batch
    PIECEWISE wildcard use uniform_token_count=None. max_query_len identifies
    the adaptive-verification descriptor, while cg_mode determines that replay
    remains a FULL CUDA graph.
    """

    max_num_seqs = 8
    max_spec_tokens = 7
    max_decode_query_len = max_spec_tokens + 1
    scheduled_num_spec_tokens = 3
    scheduled_query_len = scheduled_num_spec_tokens + 1

    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    vllm_config = _create_vllm_config_for_dsd(
        max_num_seqs=max_num_seqs,
        max_spec_tokens=max_spec_tokens,
        adaptive_verification=True,
        num_spec_per_batch_size=[(1, max_num_seqs, scheduled_num_spec_tokens)],
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=max_decode_query_len,
    )
    manager._graphs_captured = True

    num_reqs = 4
    num_tokens = num_reqs * scheduled_query_len
    candidates = manager._candidates[(num_tokens, 0)]
    assert any(desc.cg_mode == CUDAGraphMode.FULL for desc in candidates)
    assert any(desc.cg_mode == CUDAGraphMode.PIECEWISE for desc in candidates)

    desc = manager.dispatch(
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        uniform_token_count=None,
        num_active_loras=0,
        max_query_len=max_decode_query_len,
    )

    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.uniform_token_count is None
    assert desc.max_query_len == max_decode_query_len
    assert desc.num_tokens == num_tokens
    assert desc.num_reqs == num_reqs

    # Without the adaptive-verification max-query discriminator, the same
    # nonuniform shape follows the ordinary mixed-batch PIECEWISE path.
    fallback_desc = manager.dispatch(
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        uniform_token_count=None,
        num_active_loras=0,
    )
    assert fallback_desc.cg_mode == CUDAGraphMode.PIECEWISE


def test_basic_sd_does_not_capture_shorter_full_decode_shapes(monkeypatch):
    """Without DSD, only the max decode query length should get FULL graphs.

    Basic SD captures FULL decode graphs only for decode_query_len = K + 1.
    Uniform batches with smaller query lengths should therefore miss the FULL
    path entirely when using FULL_AND_PIECEWISE.
    """

    max_num_seqs = 512
    max_spec_tokens = 7
    max_decode_query_len = max_spec_tokens + 1

    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    vllm_config = _create_vllm_config_for_dsd(
        max_num_seqs=max_num_seqs,
        max_spec_tokens=max_spec_tokens,
        cudagraph_mode="FULL_AND_PIECEWISE",
        use_dynamic_sd=False,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=max_decode_query_len,
    )
    manager._graphs_captured = True

    for num_reqs in range(1, max_num_seqs + 1):
        for max_query_len in range(1, max_decode_query_len):
            # These are still uniform decode batches, but basic SD should only
            # have FULL graphs for query_len == max_decode_query_len.
            num_tokens = num_reqs * max_query_len
            uniform_tok_count = gpu_cudagraph_utils.get_uniform_token_count(
                num_reqs,
                num_tokens,
                max_query_len,
            )
            assert uniform_tok_count == max_query_len

            desc = manager.dispatch(
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                uniform_token_count=uniform_tok_count,
                num_active_loras=0,
            )

            assert desc.cg_mode == CUDAGraphMode.PIECEWISE
            assert desc.uniform_token_count is None
            assert desc.num_tokens == num_tokens
            assert desc.num_reqs is None
            assert desc.num_active_loras == 0


def test_dynamic_sd_only_captures_scheduled_query_lengths(monkeypatch):
    """DSD should only capture FULL graphs for query lengths in the schedule.

    With a partial schedule of ``(1, 32, 4)`` and ``(32, 128, 3)``, only the
    scheduled speculative-token counts (K = 4 and K = 3) become decode query
    lengths (K + 1 = 5 and 4). Uniform batches at those query lengths should get
    FULL graphs, while every other query length (e.g. the lower values 1, 2, 3)
    must fall back to the mixed-batch PIECEWISE graph.
    """

    max_num_seqs = 128
    max_spec_tokens = 7
    max_decode_query_len = max_spec_tokens + 1

    # (range_start, range_end, num_speculative_tokens): K = 4 and K = 3 are
    # scheduled, so FULL decode graphs should exist for query lengths K + 1,
    # i.e. exactly {5, 4}.
    num_spec_per_batch_size = [(1, 32, 4), (32, 128, 3)]
    scheduled_query_lens = {entry[2] + 1 for entry in num_spec_per_batch_size}

    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    vllm_config = _create_vllm_config_for_dsd(
        max_num_seqs=max_num_seqs,
        max_spec_tokens=max_spec_tokens,
        cudagraph_mode="FULL_AND_PIECEWISE",
        use_dynamic_sd=True,
        num_spec_per_batch_size=num_spec_per_batch_size,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=max_decode_query_len,
    )
    manager._graphs_captured = True

    for num_reqs in range(1, max_num_seqs + 1):
        for max_query_len in range(1, max_decode_query_len + 1):
            num_tokens = num_reqs * max_query_len
            uniform_tok_count = gpu_cudagraph_utils.get_uniform_token_count(
                num_reqs,
                num_tokens,
                max_query_len,
            )
            assert uniform_tok_count == max_query_len

            desc = manager.dispatch(
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                uniform_token_count=uniform_tok_count,
                num_active_loras=0,
            )

            if max_query_len in scheduled_query_lens:
                # Scheduled query lengths get a dedicated FULL decode graph.
                assert desc.cg_mode == CUDAGraphMode.FULL
                assert desc.uniform_token_count == max_query_len
                assert desc.num_tokens == num_tokens
                assert desc.num_reqs == num_reqs
            else:
                # Unscheduled query lengths (including the lower values 1 and 2)
                # have no FULL candidate and must fall back to PIECEWISE.
                assert desc.cg_mode == CUDAGraphMode.PIECEWISE
                assert desc.uniform_token_count is None
                assert desc.num_tokens == num_tokens
                assert desc.num_reqs is None
            assert desc.num_active_loras == 0
