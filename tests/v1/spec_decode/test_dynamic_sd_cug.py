#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.utils import get_uniform_decode_token_count

pytestmark = pytest.mark.cpu_test


def _create_vllm_config_for_dsd(
    max_num_seqs: int,
    max_spec_tokens: int,
    *,
    cudagraph_mode: str = "FULL_AND_PIECEWISE",
    use_dynamic_sd: bool = True,
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


@pytest.mark.parametrize(
    ("k", "mixed", "case", "expected_full"),
    [
        pytest.param(k, mixed, "padded-tail", True, id=f"tail-k{k}-mixed{mixed}")
        for k in (1, 2, 3, 5, 8)
        for mixed in (False, True)
    ]
    + [
        pytest.param(3, True, "real-prefill", False, id="same-shape-real-prefill"),
        pytest.param(3, True, "recurrent", False, id="recurrent-target"),
        pytest.param(
            3, True, "uncached-recurrent", False, id="uncached-recurrent-tail"
        ),
        pytest.param(3, True, "attention-free", False, id="attention-free-target"),
        pytest.param(3, True, "recurrent-draft", False, id="recurrent-draft"),
        pytest.param(3, True, "attention-free-draft", False, id="attention-free-draft"),
        pytest.param(3, True, "adaptive", False, id="adaptive-layout"),
        pytest.param(3, True, "pcp", False, id="pcp-layout"),
        pytest.param(3, True, "multiple-bonus", False, id="multiple-bonus-tokens"),
        pytest.param(3, True, "multimodal", True, id="multimodal-target"),
        pytest.param(3, True, "encoder-decoder", True, id="encoder-decoder-target"),
        pytest.param(3, True, "encoder-inputs", True, id="encoder-inputs-stay-eager"),
        pytest.param(3, True, "pp", True, id="pipeline-parallel-target"),
        pytest.param(3, True, "partial-tail-drafts", False, id="partial-tail-drafts"),
        pytest.param(
            3, True, "missing-decode-drafts", False, id="missing-decode-drafts"
        ),
        pytest.param(
            3, True, "partial-decode-drafts", False, id="partial-decode-drafts"
        ),
        pytest.param(3, True, "ragged", False, id="nonuniform-verifier-layout"),
        pytest.param(3, True, "async-decode", True, id="tail-with-async-decode"),
        pytest.param(
            3, True, "async-real-prefill", False, id="sentinels-are-not-padding"
        ),
        pytest.param(3, True, "uncached-tail", True, id="one-token-prompt-with-decode"),
        pytest.param(
            3, True, "uncaptured-tail", True, id="decode-without-captured-graph"
        ),
        pytest.param(0, False, "padded-tail", False, id="no-speculation"),
    ],
)
def test_model_runner_classifies_prompt_tail_after_preparing_prompt_inputs(
    monkeypatch,
    k,
    mixed,
    case,
    expected_full,
):
    """Exercise actual request gathering and graph dispatch before GPU preparation."""
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    width = k + (2 if case == "multiple-bonus" else 1)
    config = _create_vllm_config_for_dsd(
        2,
        k,
        cudagraph_mode="FULL_DECODE_ONLY",
        use_dynamic_sd=False,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        config,
        torch.device("cpu"),
        CUDAGraphMode.FULL_DECODE_ONLY,
        decode_query_len=width,
    )
    manager._graphs_captured = case != "uncaptured-tail"

    output = SchedulerOutput.make_empty()
    # The scheduler pads prompt tails to K+1 even if decode uses K+bonus.
    output.num_scheduled_tokens = {
        "tail": k + 1,
        **({"decode": width} if mixed else {}),
    }
    output.scheduled_spec_decode_tokens = {
        "tail": [-1] * k,
        **({"decode": list(range(k))} if mixed else {}),
    }
    remaining_prompt_tokens = 1
    if case in ("real-prefill", "async-real-prefill"):
        remaining_prompt_tokens = width
        del output.scheduled_spec_decode_tokens["tail"]
    if case in ("async-decode", "async-real-prefill"):
        output.scheduled_spec_decode_tokens["decode"] = [-1] * k
    elif case == "partial-tail-drafts":
        output.scheduled_spec_decode_tokens["tail"].pop()
    elif case == "missing-decode-drafts":
        del output.scheduled_spec_decode_tokens["decode"]
    elif case == "partial-decode-drafts":
        output.scheduled_spec_decode_tokens["decode"].pop()
    elif case == "ragged":
        output.num_scheduled_tokens["decode"] -= 1
        output.scheduled_spec_decode_tokens["decode"].pop()
    output.total_num_scheduled_tokens = sum(output.num_scheduled_tokens.values())
    prompt_len = 1 if case in ("uncached-tail", "uncached-recurrent") else 4096

    runner = object.__new__(GPUModelRunner)
    for method in (
        "update_pp_decode_requests",
        "finish_requests",
        "free_states",
        "add_requests",
        "update_requests",
    ):
        setattr(runner, method, MagicMock())
    runner.block_tables = MagicMock()
    runner.aux_output_connector = None
    runner.req_states = SimpleNamespace(
        req_id_to_index={"tail": 0, "decode": 1},
        prefill_len=SimpleNamespace(np=np.array([prompt_len, 100])),
        num_computed_prefill_tokens=np.array(
            [prompt_len - remaining_prompt_tokens, 100]
        ),
    )
    runner.speculative_config = SimpleNamespace(
        num_speculative_tokens=k,
        draft_model_config=SimpleNamespace(
            is_hybrid=case == "recurrent-draft",
            is_attention_free=case == "attention-free-draft",
        ),
    )
    runner.model_config = SimpleNamespace(
        is_hybrid=case in ("recurrent", "uncached-recurrent"),
        is_attention_free=case == "attention-free",
    )
    runner.model_state = SimpleNamespace(
        num_new_sampled_tokens_per_step=2 if case == "multiple-bonus" else 1
    )
    runner.parallel_config = config.parallel_config
    runner.observability_config = SimpleNamespace(cudagraph_metrics=False)
    runner.decode_query_len = width
    runner.dp_size, runner.dp_rank = 1, 0
    runner.cudagraph_manager = manager
    runner.pcp_manager = runner.lora_config = runner.adaptive_verification = None
    runner.ubatch_runner = None
    runner.is_encoder_decoder = case in ("encoder-decoder", "encoder-inputs")
    runner.supports_mm_inputs = case == "multimodal"
    if case == "encoder-inputs":
        output.scheduled_encoder_inputs = {"tail": [0]}
    elif case == "pp":
        runner.parallel_config.pipeline_parallel_size = 2
    elif case == "adaptive":
        runner.adaptive_verification = SimpleNamespace(
            get_num_tokens=lambda counts, drafts: sum(counts.values())
        )
    elif case == "pcp":
        runner.pcp_manager = SimpleNamespace(
            get_num_tokens_for_dispatch=lambda counts, prefilling: int(counts.sum())
        )
    prepared_batch = SimpleNamespace(has_prefill=True)

    def prepare_inputs(scheduled, state, desc, num_active_loras):
        assert scheduled is output
        assert state.has_prefill
        assert state.is_prefilling_np[state.req_ids.index("tail")]
        prepared_batch.req_ids = state.req_ids
        prepared_batch.is_prefilling_np = state.is_prefilling_np.copy()
        expected_mode = (
            CUDAGraphMode.FULL
            if expected_full and manager._graphs_captured and case != "encoder-inputs"
            else CUDAGraphMode.NONE
        )
        assert desc.cg_mode == expected_mode
        return prepared_batch

    class ReachedAttentionPreparation(Exception):
        pass

    def prepare_attn(batch):
        assert batch is prepared_batch
        assert batch.has_prefill is (not expected_full)
        assert batch.is_prefilling_np[batch.req_ids.index("tail")]
        raise ReachedAttentionPreparation

    runner.prepare_inputs = prepare_inputs
    runner.prepare_attn = prepare_attn
    with pytest.raises(ReachedAttentionPreparation):
        runner.execute_model(output)


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
            uniform_tok_count = get_uniform_decode_token_count(
                num_reqs,
                num_tokens,
                max_query_len,
                has_prefill=False,
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


def test_dynamic_sd_cudagraphs_use_clamped_query_length(monkeypatch):
    """Clamp configured K=5 to K=3, yielding query length 4 rather than 6."""
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: None,
    )

    vllm_config = _create_vllm_config_for_dsd(
        max_num_seqs=16,
        max_spec_tokens=3,
        num_spec_per_batch_size=[(1, 16, 5)],
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=4,
    )

    full_query_lens = {
        desc.uniform_token_count for desc in manager._capture_descs[CUDAGraphMode.FULL]
    }
    assert full_query_lens == {4}


def test_dynamic_sd_non_uniform_batch_falls_back_to_piecewise(monkeypatch):
    """DSD should use PIECEWISE when the batch is not a uniform decode batch.

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


def test_prompt_chunks_shaped_like_spec_decode_miss_the_full_graph(monkeypatch):
    """A batch holding K+1-token prompt chunks must not replay a decode graph.

    Prompt chunks of exactly K + 1 tokens give the batch a spec-decode shape
    by coincidence; replaying the FULL graph over them corrupts every request
    in the batch. See https://github.com/vllm-project/vllm/issues/49918.
    """
    max_spec_tokens = 7
    decode_query_len = max_spec_tokens + 1

    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    vllm_config = _create_vllm_config_for_dsd(
        max_num_seqs=64,
        max_spec_tokens=max_spec_tokens,
        use_dynamic_sd=False,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=decode_query_len,
    )
    manager._graphs_captured = True

    num_decodes, num_chunks = 6, 2
    num_reqs = num_decodes + num_chunks
    num_tokens = num_reqs * decode_query_len

    # The batch shape is indistinguishable from a full batch of spec decodes.
    assert (
        get_uniform_decode_token_count(
            num_reqs, num_tokens, decode_query_len, has_prefill=False
        )
        == decode_query_len
    )

    # Two of the requests are decode_query_len tokens into a longer prompt.
    uniform_tok_count = get_uniform_decode_token_count(
        num_reqs, num_tokens, decode_query_len, has_prefill=True
    )
    assert uniform_tok_count is None
    desc = manager.dispatch(
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        uniform_token_count=uniform_tok_count,
        num_active_loras=0,
    )
    assert desc.cg_mode == CUDAGraphMode.PIECEWISE
    assert desc.uniform_token_count is None

    # The same shape with every request decoding still gets its FULL graph.
    uniform_tok_count = get_uniform_decode_token_count(
        num_reqs, num_tokens, decode_query_len, has_prefill=False
    )
    assert uniform_tok_count == decode_query_len
    desc = manager.dispatch(
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        uniform_token_count=uniform_tok_count,
        num_active_loras=0,
    )
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.uniform_token_count == decode_query_len


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
            uniform_tok_count = get_uniform_decode_token_count(
                num_reqs,
                num_tokens,
                max_query_len,
                has_prefill=False,
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

    With a partial schedule of ``(1, 31, 4)`` and ``(32, 128, 3)``, only the
    scheduled speculative-token counts (K = 4 and K = 3) become decode query
    lengths (K + 1 = 5 and 4). Uniform batches at those query lengths should get
    FULL graphs, while every other query length (e.g. the lower values 1, 2, 3)
    must fall back to the mixed-batch PIECEWISE graph.
    """
    max_num_seqs = 128
    max_spec_tokens = 7
    max_decode_query_len = max_spec_tokens + 1

    # (range_start, range_end, num_speculative_tokens): K = 4 and K = 3 are
    # scheduled in non-overlapping tiers, so FULL decode graphs should exist
    # for query lengths K + 1, i.e. exactly {5, 4}.
    num_spec_per_batch_size = [(1, 31, 4), (32, 128, 3)]
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
            uniform_tok_count = get_uniform_decode_token_count(
                num_reqs,
                num_tokens,
                max_query_len,
                has_prefill=False,
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
