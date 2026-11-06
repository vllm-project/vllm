# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import random
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

import vllm.envs as envs
from vllm.config import (
    CacheConfig,
    ECTransferConfig,
    KVTransferConfig,
    LoRAConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.watermarking import WatermarkConfig
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputSchedulerConnector,
    AuxRequestOutput,
)
from vllm.distributed.ec_transfer.ec_connector.metrics import ECConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnector,
    HiSparseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.outputs import RequestError
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.utils.hashing import sha256
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.diffusion_scheduler import (
    DiffusionAsyncScheduler,
    DiffusionScheduler,
    diffusion_canvas_width,
)
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.state_scheduler import StateScheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import FinishReason
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.hisparse.coordinator import get_hisparse_coordinator
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    DraftTokenIds,
    ECConnectorOutput,
    KVConnectorOutput,
    ModelRunnerOutput,
    SamplingMaskLists,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputGrammar, StructuredOutputManager

from .utils import (
    EOS_TOKEN_ID,
    create_requests,
    create_scheduler,
    mock_kv,
    new_sliding_window_spec,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("async_scheduling", "outcome", "num_speculative_tokens"),
    [
        (False, outcome, 3)
        for outcome in ("accepted", "rejected", "finished", "preempted", "stale")
    ]
    + [(True, "accepted", 3), (False, "accepted", 0), (False, "finished", 0)],
)
def test_mamba_decode_checkpoint_publication(
    async_scheduling: bool, outcome: str, num_speculative_tokens: int
):
    """Publish accepted checkpoints before retirement; never publish rejected work."""
    block_size = 16
    scheduler = create_scheduler(
        model="Qwen/Qwen3-0.6B",
        enable_prefix_caching=True,
        async_scheduling=async_scheduling,
        num_speculative_tokens=num_speculative_tokens or None,
        speculative_method="ngram_gpu",
        block_size=block_size,
        kv_cache_spec=MambaSpec(
            block_size=block_size,
            shapes=((1, 1),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            num_speculative_blocks=int(num_speculative_tokens > 0),
        ),
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=block_size - (2 if num_speculative_tokens else 1),
        block_size=block_size,
        max_tokens=2 + num_speculative_tokens if outcome == "finished" else 32,
    )
    scheduler.add_request(request)
    prefill = scheduler.schedule()
    scheduler.update_from_output(
        prefill,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[0]],
        ),
    )
    draft_tokens = list(range(1, num_speculative_tokens + 1))
    scheduler.update_draft_token_ids(
        DraftTokenIds([request.request_id], [draft_tokens])
    )
    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens[request.request_id] == 1 + num_speculative_tokens
    assert len(decode.scheduled_spec_decode_tokens.get(request.request_id, [])) == (
        num_speculative_tokens
    )
    if outcome in ("preempted", "stale"):
        assert scheduler.reset_prefix_cache(reset_running_requests=True)
        assert request.status == RequestStatus.PREEMPTED
        assert request.num_stale_output_tokens == 4
        if outcome == "stale":
            # Resume before the old decode completes; its output must stay uncached.
            assert scheduler.schedule().num_scheduled_tokens[request.request_id] > 0
            assert request.status == RequestStatus.RUNNING
    accepted = 1 if outcome == "rejected" else 1 + num_speculative_tokens
    scheduler.update_from_output(
        decode,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[list(range(1, accepted + 1))],
        ),
    )
    if outcome == "accepted":
        # The next allocation retires the checkpoint from the request's table.
        scheduler.update_draft_token_ids(
            DraftTokenIds([request.request_id], [draft_tokens])
        )
        assert scheduler.schedule().num_scheduled_tokens[request.request_id] == (
            1 + num_speculative_tokens
        )
    elif outcome == "finished":
        assert request.is_finished()
    elif outcome in ("preempted", "stale"):
        if outcome == "preempted":
            assert request.num_computed_tokens == 0
        assert request.num_output_tokens == 1
        assert request.num_stale_output_tokens == 0
    else:
        assert request.num_computed_tokens == block_size - 1
    probe = Request(
        request_id="probe",
        prompt_token_ids=list(request.all_token_ids),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )
    _, cached_tokens, _ = scheduler.kv_cache_manager.get_computed_blocks(probe)
    assert cached_tokens == (block_size if outcome in ("accepted", "finished") else 0)


def test_make_scheduled_encoder_input_stats_output_embeddings():
    scheduler = create_scheduler()
    mm_features = [
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="image",
            identifier="image-0",
            mm_position=PlaceholderRange(offset=0, length=196),
        ),
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="video",
            identifier="video-0",
            mm_position=PlaceholderRange(offset=200, length=196),
        ),
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="audio",
            identifier="audio-0",
            mm_position=PlaceholderRange(offset=400, length=49),
        ),
    ]
    scheduler.requests["req"] = Mock(mm_features=mm_features)

    stats = scheduler._make_scheduled_encoder_input_stats({"req": [0, 1, 2]})

    assert stats is not None
    assert stats.num_inputs == 3
    assert stats.output_tokens == 441


def test_scheduled_encoder_input_stats_disabled_without_iteration_logging(
    monkeypatch: pytest.MonkeyPatch,
):
    scheduler = create_scheduler()
    make_stats = Mock(side_effect=AssertionError("stats should not be computed"))
    monkeypatch.setattr(scheduler, "_make_scheduled_encoder_input_stats", make_stats)

    scheduler_output = scheduler.schedule()

    make_stats.assert_not_called()
    assert scheduler_output.scheduled_encoder_input_stats is None


def test_scheduled_encoder_input_stats_disabled_without_log_stats(
    monkeypatch: pytest.MonkeyPatch,
):
    scheduler = create_scheduler()
    scheduler.log_stats = False
    scheduler.observability_config.enable_logging_iteration_details = True
    make_stats = Mock(side_effect=AssertionError("stats should not be computed"))
    monkeypatch.setattr(scheduler, "_make_scheduled_encoder_input_stats", make_stats)

    scheduler_output = scheduler.schedule()

    make_stats.assert_not_called()
    assert scheduler_output.scheduled_encoder_input_stats is None


def test_add_requests():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)

    for i, request in enumerate(requests):
        scheduler.add_request(request)
        assert request.request_id in scheduler.requests
        assert len(scheduler.waiting) == i + 1


@pytest.mark.parametrize("max_tokens", [1, 2])
def test_routed_experts_prompt_start_at_prompt_end(max_tokens):
    """A prompt-end offset should return empty routing data, not crash."""
    scheduler = create_scheduler()
    (request,) = create_requests(num_requests=1, max_tokens=max_tokens)
    request.sampling_params.routed_experts_prompt_start = request.num_prompt_tokens
    scheduler.aux_output_connector = AuxOutputSchedulerConnector(
        scheduler.kv_cache_manager
    )
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    assert scheduler_output.aux_output_connector_metadata.requests == {
        request.request_id: request.num_prompt_tokens
    }

    outputs = scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            aux_output_connector_output={
                request.request_id: AuxRequestOutput(
                    request.num_prompt_tokens,
                    np.empty((0, 1, 1), dtype=np.uint8),
                )
            },
        ),
    )

    output = outputs[request.client_index].outputs[0]
    assert output.new_token_ids == [0]
    assert output.routed_experts.shape == (0, 1, 1)


def test_routed_experts_hashes_follow_scheduled_tokens():
    scheduler = create_scheduler(enable_prefix_caching=True, max_num_batched_tokens=64)
    scheduler.aux_output_connector = AuxOutputSchedulerConnector(
        scheduler.kv_cache_manager
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=160, max_tokens=20, ignore_eos=True
    )
    scheduler.add_request(request)

    sent: list[bytes] = []
    num_sent_per_step = []
    while not request.is_finished():
        num_computed = request.num_computed_tokens
        scheduler_output = scheduler.schedule()
        num_scheduled = scheduler_output.num_scheduled_tokens[request.request_id]
        metadata = scheduler_output.aux_output_connector_metadata
        sent.extend(metadata.block_hashes.get(request.request_id, ()))
        num_sent_per_step.append(len(sent))

        sampled = num_computed + num_scheduled >= request.num_tokens
        scheduler.update_from_output(
            scheduler_output,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[0]] if sampled else [[]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
                aux_output_connector_output={
                    request.request_id: AuxRequestOutput(
                        0,
                        np.zeros((num_computed + num_scheduled, 1, 1), np.uint8),
                    )
                },
            ),
        )

    assert num_sent_per_step == [4, 8, 10] + [10] * 15 + [11] * 4
    assert sent == request.block_hashes[:11]


@pytest.mark.parametrize(
    "external_tokens,max_cache_hit,sliding_window,expected_hit",
    [
        (0, 79, None, 64),
        (0, 40, None, 32),
        (0, 40, 32, 32),
        (32, 79, None, 32),
        (32, 16, None, 0),
        (31, 79, None, 0),
        (32, 79, 32, 32),
        (32, 16, 32, 0),
        (32, 0, None, 0),
    ],
    ids=[
        "local",
        "local-truncated",
        "local-sliding-window-truncated",
        "external",
        "external-dropped",
        "external-mid-block",
        "external-sliding-window",
        "external-sliding-window-dropped",
        "none",
    ],
)
def test_prefix_hit_is_capped_at_retained_auxiliary_outputs(
    external_tokens, max_cache_hit, sliding_window, expected_hit
):
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        kv_cache_spec=(
            SlidingWindowSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=sliding_window,
            )
            if sliding_window
            else None
        ),
        use_kv_connector=(
            mock_kv(matched_tokens=external_tokens, is_async=False)
            if external_tokens
            else None
        ),
    )
    first, second = create_requests(num_requests=2, num_tokens=80, same_prompt=True)
    if not external_tokens:
        scheduler.add_request(first)
        scheduler.schedule()
        scheduler.finish_requests(first.request_id, RequestStatus.FINISHED_ABORTED)
    connector = AuxOutputSchedulerConnector(scheduler.kv_cache_manager, Mock())
    connector.max_cache_hit = Mock(return_value=max_cache_hit)
    scheduler.aux_output_connector = connector
    scheduler.add_request(second)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[second.request_id] == 80 - expected_hit


def test_evicted_blocks_release_their_auxiliary_outputs_past_capacity():
    scheduler = create_scheduler(enable_prefix_caching=True, num_blocks=5)
    scheduler.aux_output_connector = AuxOutputSchedulerConnector(
        scheduler.kv_cache_manager
    )
    first, second, third = create_requests(num_requests=3, num_tokens=64)
    for request in (first, second):
        scheduler.add_request(request)
        scheduler.schedule()
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)

    scheduler.add_request(third)
    metadata = scheduler.schedule().aux_output_connector_metadata

    assert list(metadata.retained_block_hashes) == second.block_hashes
    assert list(metadata.released_block_hashes) == first.block_hashes[:3]


@pytest.mark.parametrize(
    "with_config,params,expected",
    [
        (True, {"watermarking": True}, True),
        (True, {"watermarking": False}, False),
        (True, {"watermarking": True, "temperature": 0.0}, False),
        (True, {"watermarking": True, "trace_decode_token_ids": [0]}, False),
        (False, {"watermarking": True}, None),
    ],
)
def test_first_output_reports_watermarking(with_config, params, expected):
    """EngineCore sends the flag it applied on the first output only."""
    scheduler = create_scheduler()
    if with_config:
        scheduler.vllm_config.watermark_config = WatermarkConfig(key=42)
    (request,) = create_requests(num_requests=1, max_tokens=2, ignore_eos=True)
    for name, value in params.items():
        setattr(request.sampling_params, name, value)
    scheduler.add_request(request)

    reported = []
    for _ in range(2):
        scheduler_output = scheduler.schedule()
        model_output = ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        outputs = scheduler.update_from_output(scheduler_output, model_output)
        reported.append(outputs[0].outputs[0].watermarking)

    assert reported == [expected, None]


def test_finish_request():
    scheduler = create_scheduler()
    scheduler.aux_output_connector = Mock()
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    for i, request in enumerate(requests):
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        assert request.request_id not in scheduler.requests
        assert len(scheduler.waiting) == 9 - i
        scheduler.aux_output_connector.request_finished.assert_called_with(request)


def test_get_num_unfinished_requests():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    for i, request in enumerate(requests):
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_STOPPED)
        assert scheduler.get_num_unfinished_requests() == len(requests) - i - 1


def test_max_num_active_seqs_caps_admission():
    """Admission is capped by max_num_active_seqs, not max_num_seqs."""
    scheduler = create_scheduler(max_num_seqs=16, max_num_active_seqs=3)
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 3
    assert len(scheduler.running) == 3
    assert len(scheduler.waiting) == 7
    # Execution capacity is untouched.
    assert scheduler.max_num_running_reqs == 16


def _bind_hisparse_connector(scheduler):
    connector = object.__new__(HiSparseConnector)
    connector.connector_scheduler = HiSparseConnectorScheduler(async_speculative=False)
    from tests.v1.core.test_prefix_caching import make_hisparse_kv_cache_manager

    coordinator = get_hisparse_coordinator(make_hisparse_kv_cache_manager(16, 16))
    connector.connector_scheduler.bind_coordinator(coordinator)
    composite = object.__new__(MultiConnector)
    composite._connectors = (connector,)
    scheduler.connector = composite
    return coordinator


def test_pending_hisparse_spill_keeps_scheduler_alive():
    """A final host spill must complete after the last request finishes."""
    scheduler = create_scheduler()
    coordinator = _bind_hisparse_connector(scheduler)
    pending = coordinator.pending_spills
    pending[0] = Mock()
    assert scheduler.has_requests()
    pending.clear()
    assert not scheduler.has_requests()


@pytest.mark.parametrize(
    "enable_prefix_caching, prompt_logprobs",
    [
        (False, None),
        (True, 5),
    ],
)
def test_schedule(enable_prefix_caching: bool, prompt_logprobs: int | None):
    """Test scheduling.
    Two cases: default APC/no prompt logprobs; APC=True + prompt logprobs
    """
    scheduler = create_scheduler(enable_prefix_caching=enable_prefix_caching)
    requests = create_requests(num_requests=10, prompt_logprobs=prompt_logprobs)
    for request in requests:
        scheduler.add_request(request)

    # Test initial scheduling
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0
    # Verify all requests are scheduled.
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)

    # Verify requests moved from waiting to running
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == len(requests)
    for i, request in enumerate(requests):
        assert scheduler.running[i] == request


def test_pooling_chunked_prefill_can_finish_at_max_model_len():
    scheduler = create_scheduler(
        max_num_seqs=1, max_model_len=8, max_num_batched_tokens=4, runner="pooling"
    )
    assert scheduler.max_model_len == 8
    request = Request(
        request_id="pool",
        prompt_token_ids=[1] * 8,
        sampling_params=None,
        pooling_params=PoolingParams(task="embed"),
    )
    scheduler.add_request(request)

    scheduler.schedule()
    # The final chunk must reach max_model_len, not stop one token short.
    assert scheduler.schedule().num_scheduled_tokens == {"pool": 4}


def test_scheduler_stats_route_to_existing_output_client():
    scheduler = create_scheduler()
    request = create_requests(num_requests=1)[0]
    request.client_index = 1
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    model_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[1000]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    engine_core_outputs = scheduler.update_from_output(scheduler_output, model_output)

    assert 0 not in engine_core_outputs
    assert engine_core_outputs[1].scheduler_stats is not None
    assert len(engine_core_outputs[1].outputs) == 1


def test_scheduler_publishes_lookahead_blocks_at_allocation():
    """A lookahead block is published once its lookahead token exists: prompt
    blocks at their own allocation, the block ending at the first sampled
    token at the allocation after it is sampled."""
    block_size = 2
    init_none_hash(sha256)
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=block_size,
        max_num_batched_tokens=16,
    )

    def make_request(request_id: str, prompt_token_ids: list[int]) -> Request:
        return Request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=SamplingParams(max_tokens=3, ignore_eos=True),
            pooling_params=None,
            block_hasher=get_request_block_hasher(block_size, sha256, 1),
        )

    def num_cached_tokens(prompt_token_ids: list[int]) -> int:
        probe = make_request("probe", prompt_token_ids)
        return scheduler.kv_cache_manager.get_computed_blocks(probe)[1]

    request = make_request("first", [0, 1, 2, 3])
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    # Block [2, 4) waits for its lookahead token, the first sampled one.
    assert num_cached_tokens([0, 1, 2, 3, 5, 7]) == block_size

    scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[5]],
        ),
    )
    scheduler.schedule()
    assert num_cached_tokens([0, 1, 2, 3, 5, 7]) == 2 * block_size
    assert num_cached_tokens([0, 1, 2, 3, 6, 7]) == block_size


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("lookahead", [0, 1])
@pytest.mark.parametrize("ending", ["prefill", "accepted", "stop", "rejected"])
def test_finished_spec_request_caches_completed_blocks(
    async_scheduling, lookahead, ending
):
    """Final output can make a computed block's hash available before freeing it."""
    block_size = 4
    init_none_hash(sha256)
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=block_size,
        num_speculative_tokens=2,
        speculative_method="ngram_gpu",
        async_scheduling=async_scheduling,
    )
    hasher = get_request_block_hasher(block_size, sha256, lookahead)
    prompt = list(range(4 if ending == "prefill" else 2))
    request = Request(
        request_id="producer",
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(
            max_tokens={"prefill": 1, "rejected": 2}.get(ending, 4),
            ignore_eos=True,
            stop_token_ids=[4] if ending == "stop" else None,
        ),
        pooling_params=None,
        block_hasher=hasher,
    )

    def finish_step(output, tokens):
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[tokens],
            ),
        )

    scheduler.add_request(request)
    output = scheduler.schedule()
    if ending == "prefill":
        finish_step(output, [4])
    else:
        finish_step(output, [2])
        scheduler.update_draft_token_ids(
            DraftTokenIds(req_ids=[request.request_id], draft_token_ids=[[3, 4]])
        )
        output = scheduler.schedule()
        finish_step(output, [8] if ending == "rejected" else [3, 4, 5])

    assert request.is_finished()
    assert request.num_output_placeholders == 0
    probe = Request(
        request_id="probe",
        prompt_token_ids=list(request.all_token_ids) + [99] * block_size,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=hasher,
    )
    expected = 0 if ending == "rejected" else block_size
    assert scheduler.kv_cache_manager.get_computed_blocks(probe)[1] == expected


@pytest.mark.parametrize("num_spec_tokens", [None, 2])
def test_async_output_does_not_cache_inflight_tokens(num_spec_tokens):
    """A returned token ID does not prove that its next-step KV is ready."""
    init_none_hash(sha256)
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=1,
        async_scheduling=True,
        num_speculative_tokens=num_spec_tokens,
        speculative_method="ngram_gpu" if num_spec_tokens else None,
    )
    hasher = get_request_block_hasher(1, sha256)
    request = Request(
        request_id="producer",
        prompt_token_ids=[0, 1],
        sampling_params=SamplingParams(max_tokens=16, ignore_eos=True),
        pooling_params=None,
        block_hasher=hasher,
    )
    scheduler.add_request(request)
    first = scheduler.schedule()
    second = scheduler.schedule()
    assert second.num_scheduled_tokens[request.request_id] > 0
    scheduler.update_from_output(
        first,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[2]],
        ),
    )
    probe = Request(
        request_id="probe",
        prompt_token_ids=[0, 1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=hasher,
    )
    assert scheduler.kv_cache_manager.get_computed_blocks(probe)[1] == 2


@pytest.mark.parametrize("recovery", ["resident", "preempt", "reset"])
def test_streaming_lookahead_hashes_follow_recomputed_kv(recovery):
    """Recomputation uses the continuation's tokens without re-keying resident KV."""
    block_size = 4
    model_config = ModelConfig(
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        max_model_len=128,
        skip_tokenizer_init=True,
    )
    scheduler = create_scheduler(
        model=model_config.model,
        max_model_len=128,
        max_num_batched_tokens=128,
        enable_prefix_caching=True,
        block_size=block_size,
        speculative_config=SpeculativeConfig(
            method="mtp",
            num_speculative_tokens=1,
            target_model_config=model_config,
            target_parallel_config=ParallelConfig(),
        ),
    )
    assert scheduler.num_hash_lookahead_tokens == 1
    init_none_hash(sha256)
    hasher = get_request_block_hasher(block_size, sha256, 1)

    def request(request_id, tokens, max_tokens=16, resumable=False):
        return Request(
            request_id=request_id,
            prompt_token_ids=list(tokens),
            sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
            pooling_params=None,
            block_hasher=hasher,
            resumable=resumable,
        )

    session = request("session", [1, 2, 3], max_tokens=2, resumable=True)
    scheduler.add_request(session)

    def step(token):
        output = scheduler.schedule()
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=[session.request_id],
                req_id_to_index={session.request_id: 0},
                sampled_token_ids=[[token]],
            ),
        )

    step(4)
    step(5)
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    manager = scheduler.kv_cache_manager
    old_block = manager.get_blocks(session.request_id).blocks[0][0]
    old_key = old_block.block_hash
    old_hash = session.block_hashes[0]
    assert old_key is not None
    old_probe = request("old", [1, 2, 3, 4, 5, 100])
    assert manager.get_computed_blocks(old_probe)[1] == block_size

    scheduler.add_request(request("session", [42, 43, 44], resumable=True))
    assert list(session.all_token_ids) == [1, 2, 3, 4, 42, 43, 44]
    step(99)
    assert session.block_hashes[0] == old_hash
    assert old_block.block_hash == old_key
    current = request("current", session.all_token_ids)
    assert current.block_hashes[0] != old_hash
    assert manager.get_computed_blocks(current)[1] == 0
    if recovery == "resident":
        return

    if recovery == "reset":
        assert scheduler.reset_prefix_cache(reset_running_requests=True)
    else:
        scheduler.running.remove(session)
        scheduler._preempt_request(session, 0.0)
    assert session.num_computed_tokens == 0
    output = scheduler.schedule()
    assert session.block_hashes == current.block_hashes
    assert output.num_scheduled_tokens[session.request_id] == session.num_tokens
    new_blocks, hit_tokens, _ = manager.get_computed_blocks(current)
    assert hit_tokens == block_size
    old_blocks, old_hit_tokens, _ = manager.get_computed_blocks(old_probe)
    if recovery == "reset":
        assert old_hit_tokens == 0
    else:
        assert old_hit_tokens == block_size
        assert old_blocks.blocks[0][0] is old_block
        assert new_blocks.blocks[0][0] is not old_block
        assert old_block.block_hash == old_key


def test_schedule_multimodal_requests():
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    mm_positions = [[PlaceholderRange(offset=i, length=100)] for i in range(10)]
    requests = create_requests(
        num_requests=10,
        num_tokens=200,
        mm_positions=mm_positions,
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)
    assert len(output.scheduled_encoder_inputs) == 10
    for req_id, encoder_input in output.scheduled_encoder_inputs.items():
        assert len(encoder_input) == 1


def test_async_scheduling_pp_allows_rescheduling_with_output_placeholders():
    """Async scheduling + PP: allow multi-step in-flight scheduling per request."""
    scheduler = create_scheduler(async_scheduling=True, pipeline_parallel_size=2)
    (req,) = create_requests(num_requests=1, num_tokens=8)
    scheduler.add_request(req)

    _ = scheduler.schedule()
    assert req.num_output_placeholders > 0

    if scheduler.use_v2_model_runner:
        # MRV2 staggers same-request decodes by pp_size steps to match the
        # worker-side sampled-token broadcast cadence.
        output = scheduler.schedule()
        assert req.request_id not in output.num_scheduled_tokens

    # before any update_from_output, we still expect the request can be
    # scheduled again (multi-step in-flight).
    output = scheduler.schedule()
    assert req.request_id in output.num_scheduled_tokens


def test_max_loras_counts_running_request_awaiting_output():
    """PP: a running request with its prompt fully scheduled is skipped until
    its output returns, but still holds its LoRA slot, so a request with
    another adapter must not be admitted in the meantime."""
    scheduler = create_scheduler(
        pipeline_parallel_size=2, lora_config=LoRAConfig(max_loras=1)
    )
    req_a, req_b = create_requests(num_requests=2, num_tokens=8, req_ids=["a", "b"])
    req_a.lora_request = LoRARequest("lora-a", 1, "/path/a")
    req_b.lora_request = LoRARequest("lora-b", 2, "/path/b")

    scheduler.add_request(req_a)
    assert scheduler.schedule().num_scheduled_tokens == {"a": 8}

    scheduler.add_request(req_b)
    assert scheduler.schedule().num_scheduled_tokens == {}
    assert req_b.status == RequestStatus.WAITING


def test_max_loras_counts_running_request_skipped_for_encoder_budget():
    """No PP: a running request skipped because earlier requests used up the
    step's encoder compute budget still holds its LoRA slot."""
    scheduler = create_scheduler(lora_config=LoRAConfig(max_loras=2))
    scheduler.max_num_encoder_input_tokens = 100
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=1000)
    # a: two 100-token images, one per step; b: image at 50, after a's.
    req_a, req_b = create_requests(
        num_requests=2,
        num_tokens=300,
        mm_positions=[
            [
                PlaceholderRange(offset=0, length=100),
                PlaceholderRange(offset=150, length=100),
            ],
            [PlaceholderRange(offset=50, length=100)],
        ],
        req_ids=["a", "b"],
    )
    (req_c,) = create_requests(num_requests=1, num_tokens=10, req_ids=["c"])
    req_a.lora_request = LoRARequest("lora-a", 1, "/path/a")
    req_b.lora_request = LoRARequest("lora-b", 2, "/path/b")
    req_c.lora_request = LoRARequest("lora-c", 3, "/path/c")

    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"a": 150, "b": 50}
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=["a", "b"],
            req_id_to_index={"a": 0, "b": 1},
            sampled_token_ids=[[], []],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    # a's second image uses the whole encoder budget, so b is skipped; c
    # (a third adapter) must still wait.
    scheduler.add_request(req_c)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"a": 150}
    assert req_c.status == RequestStatus.WAITING


def test_cached_request_data_resumed_all_token_ids_mrv1_only():
    """all_token_ids carries a resumed request's token ids to the connector
    for the V1 model runner, but is skipped entirely for the V2 model runner.
    """
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

    scheduler = create_scheduler(use_v2_model_runner=False)
    (req,) = create_requests(num_requests=1, num_tokens=8)
    req.append_output_token_ids([101, 102, 103])

    # A resumed request was not scheduled in the previous step.
    assert req.request_id not in scheduler.prev_step_scheduled_req_ids

    empty_blocks = KVCacheBlocks(blocks=((),))

    def make_cached():
        return scheduler._make_cached_request_data(
            running_reqs=[],
            resumed_reqs=[req],
            num_scheduled_tokens={req.request_id: 1},
            spec_decode_tokens={},
            req_to_new_blocks={req.request_id: empty_blocks},
        )

    # V1 model runner: the full token id list is propagated.
    assert not scheduler.use_v2_model_runner
    cached = make_cached()
    assert req.request_id in cached.resumed_req_ids
    assert cached.all_token_ids[req.request_id] == list(req.all_token_ids)

    # V2 model runner: all_token_ids is skipped entirely.
    scheduler.use_v2_model_runner = True
    cached = make_cached()
    assert req.request_id in cached.resumed_req_ids
    assert cached.all_token_ids == {}


@pytest.mark.parametrize("long_prefill_token_threshold", [0, 128])
@pytest.mark.parametrize("cached_first_image", [False, True])
def test_encoder_cache_pressure_respects_disabled_chunked_prefill(
    long_prefill_token_threshold,
    cached_first_image,
):
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=1024,
        enable_chunked_prefill=False,
        async_scheduling=True,
        long_prefill_token_threshold=long_prefill_token_threshold,
    )
    mm_positions = [[PlaceholderRange(offset=100, length=600)] for _ in range(2)]
    mm_hashes = [["busy"], ["new"]]
    cache = scheduler.encoder_cache_manager
    if cached_first_image:
        cached_position = PlaceholderRange(offset=0, length=100)
        (primer,) = create_requests(
            num_requests=1,
            num_tokens=800,
            req_ids=["cache-primer"],
            mm_positions=[[cached_position]],
            mm_hashes_list=[["cached"]],
        )
        cache.allocate(primer, 0)
        cache.free(primer)
        mm_positions[1].insert(0, cached_position)
        mm_hashes[1].insert(0, "cached")
        assert cache.freeable["cached"] == 100

    requests = create_requests(
        num_requests=2,
        num_tokens=800,
        max_tokens=1,
        mm_positions=mm_positions,
        mm_hashes_list=mm_hashes,
    )
    for request in requests:
        scheduler.add_request(request)

    first, second = requests
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {first.request_id: 800}

    # The first image is still in flight and occupies the encoder cache.
    assert scheduler.schedule().num_scheduled_tokens == {}
    assert second.status == RequestStatus.WAITING
    assert second.num_computed_tokens == 0
    assert not cache.get_cached_input_ids(second)
    if cached_first_image:
        assert cache.cached["cached"] == set()
        assert cache.freeable["cached"] == 100

    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=[first.request_id],
            req_id_to_index={first.request_id: 0},
            sampled_token_ids=[[EOS_TOKEN_ID]],
        ),
    )
    assert scheduler.schedule().num_scheduled_tokens == {second.request_id: 800}


def test_schedule_partial_requests():
    """Test scheduling behavior with partial requests.

    This test verifies that:
    1. The scheduler can handle multiple partial requests in a single step when
       constrained by encoder budget.
    2. A request in RUNNING state may be unscheduled in subsequent steps if
       there is insufficient encoder budget.
    """
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=1024,
    )
    mm_positions = [[PlaceholderRange(offset=100, length=600)] for _ in range(3)]
    requests = create_requests(
        num_requests=3,
        num_tokens=800,
        mm_positions=mm_positions,
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 3
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0

    assert scheduler.max_num_encoder_input_tokens == 1024
    # The first request is scheduled fully.
    assert output.num_scheduled_tokens[requests[0].request_id] == 800
    # The second request is scheduled partially.
    # The <img> tokens are not scheduled because of the encoder budget.
    assert output.num_scheduled_tokens[requests[1].request_id] == 100
    # The third request is also scheduled partially.
    # The <img> tokens are not scheduled because of the encoder budget.
    assert output.num_scheduled_tokens[requests[2].request_id] == 100
    req_to_index = {request.request_id: i for i, request in enumerate(requests)}
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        # Only the first request has a sampled token id because
        # the rest requests are still being prefilled.
        sampled_token_ids=[[0], [], []],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_runner_output)

    # Schedule the next step.
    # Only the first and second requests are scheduled.
    # The third request is in the RUNNING state but not scheduled in this step
    # because of the encoder budget.
    output = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output.scheduled_new_reqs) == 0
    assert output.scheduled_cached_reqs.num_reqs == 2
    assert len(output.finished_req_ids) == 0
    assert output.num_scheduled_tokens[requests[0].request_id] == 1
    assert output.num_scheduled_tokens[requests[1].request_id] == 700
    assert requests[2].request_id not in output.num_scheduled_tokens


def test_encoder_only_prompt_longer_than_budget_is_chunked():
    """The engine switches chunked prefill off for an encoder-only instance,
    but the token budget must still split its prompt across steps instead of
    leaving the request waiting forever."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=1024,
        max_model_len=4096,
        mm_encoder_only=True,
    )
    # EngineCore turns this off after config validation whenever the instance
    # holds no KV cache, which is every encoder-only instance.
    scheduler.scheduler_config.enable_chunked_prefill = False
    (request,) = create_requests(
        num_requests=1,
        num_tokens=2500,
        mm_positions=[[PlaceholderRange(offset=100, length=600)]],
    )
    scheduler.add_request(request)

    def advance(output):
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )

    first = scheduler.schedule()
    assert first.num_scheduled_tokens[request.request_id] == 1024
    assert request.request_id in first.scheduled_encoder_inputs
    advance(first)

    second = scheduler.schedule()
    assert second.num_scheduled_tokens[request.request_id] == 1024
    advance(second)

    third = scheduler.schedule()
    assert third.num_scheduled_tokens[request.request_id] == 452


@pytest.mark.parametrize("has_running", [True, False])
def test_schedule_prefills_gating(has_running: bool):
    """DP prefill-balancing gate: when `throttle_prefills` is True, a new
    WAITING (prefill) request is deferred ONLY if this rank has running work to
    protect. With no running requests, the prefill is admitted regardless (so a
    throttled step is never wasted as a dummy), and running/decode requests are
    unaffected. Once the cadence allows prefills again, the request is admitted.
    """
    scheduler = create_scheduler(max_num_seqs=16, max_num_batched_tokens=8192)

    if has_running:
        # Establish a running (decode) request via a prefill + output step.
        (running_req,) = create_requests(num_requests=1, num_tokens=8, req_ids=["run0"])
        scheduler.add_request(running_req)
        output = scheduler.schedule()
        assert len(output.scheduled_new_reqs) == 1
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=["run0"],
                req_id_to_index={"run0": 0},
                sampled_token_ids=[[0]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        assert len(scheduler.running) == 1

    # Add a new WAITING (prefill) request, with prefills gated off.
    (new_req,) = create_requests(num_requests=1, num_tokens=8, req_ids=["new0"])
    scheduler.add_request(new_req)
    output = scheduler.schedule(throttle_prefills=True)

    if has_running:
        # There is running work to protect, so the new prefill is deferred...
        assert "new0" not in output.num_scheduled_tokens
        assert new_req.status == RequestStatus.WAITING
        # ...while the running/decode request keeps being scheduled.
        assert "run0" in output.num_scheduled_tokens
        # When the cadence allows prefills again, the request is admitted.
        output = scheduler.schedule()

    # No running work to protect (or cadence now open): the prefill is admitted.
    assert "new0" in output.num_scheduled_tokens
    assert any(r.req_id == "new0" for r in output.scheduled_new_reqs)


def _decode_one(scheduler, output, req_ids: list[str]) -> None:
    """Feed back one sampled token per request, so they keep decoding."""
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[[0] for _ in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def _start_decoding(scheduler, req_id: str = "run0"):
    """Prefill one request and take it into the decode state."""
    (request,) = create_requests(
        num_requests=1, num_tokens=8, max_tokens=100, req_ids=[req_id]
    )
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert req_id in output.num_scheduled_tokens
    _decode_one(scheduler, output, [req_id])
    assert len(scheduler.running) == 1
    return output


@pytest.mark.parametrize("interval", [2, 4])
def test_local_prefill_interval_holds_prefill_between_bursts(interval: int):
    """A single engine applies `prefill_schedule_interval` in the scheduler,
    timed from the last prefill-carrying step: a waiting prefill is held for the
    remainder of the interval while running requests keep decoding, then
    admitted once the interval has elapsed.
    """
    scheduler = create_scheduler(prefill_schedule_interval=interval)
    # This prefill starts the interval.
    _start_decoding(scheduler)
    assert scheduler.last_prefill_step == scheduler.num_executed_steps

    (new_req,) = create_requests(num_requests=1, num_tokens=8, req_ids=["new0"])
    scheduler.add_request(new_req)

    # Steps inside the interval decode alone; the waiting prefill is held.
    for _ in range(interval - 1):
        output = scheduler.schedule()
        assert "new0" not in output.num_scheduled_tokens
        assert "run0" in output.num_scheduled_tokens
        assert new_req.status == RequestStatus.WAITING
        _decode_one(scheduler, output, ["run0"])

    # The interval has now elapsed, so prefill is admitted again.
    output = scheduler.schedule()
    assert "new0" in output.num_scheduled_tokens
    assert scheduler.last_prefill_step == scheduler.num_executed_steps


def test_empty_schedule_does_not_advance_the_prefill_cadence():
    """An empty schedule() must not consume interval budget.

    `current_step` counts every schedule() call. With async scheduling,
    step_with_batch_queue() calls schedule() again whenever the batch queue has
    room and has_requests() is true -- so a call lands while the previous batch
    is still in flight and nothing new can be scheduled. An async KV connector
    makes this common, since requests sit in lookup or WAITING_FOR_REMOTE_KVS.
    Timing the cadence on current_step lets those empty calls expire the hold
    early. Reported on #54627 against the LMCache MP connector.
    """
    scheduler = create_scheduler(prefill_schedule_interval=4)
    _start_decoding(scheduler)

    # One real decode step, left in flight (no update_from_output yet).
    output = scheduler.schedule()
    assert output.total_num_scheduled_tokens > 0
    executed_before = scheduler.num_executed_steps
    step_before = scheduler.current_step

    # Re-entering schedule() while that batch is still unaccounted for
    # schedules nothing -- this is the async-scheduling shape.
    for _ in range(3):
        empty = scheduler.schedule()
        assert empty.total_num_scheduled_tokens == 0

    assert scheduler.current_step == step_before + 3, "current_step should advance"
    assert scheduler.num_executed_steps == executed_before, (
        "an empty schedule() must not count as an executed step"
    )


def test_prefill_cadence_counts_executed_steps_not_schedule_calls():
    """The hold spans `interval` EXECUTED steps, however many empty schedule()
    calls are interleaved.

    Counting schedule() calls instead lets each empty call eat interval budget,
    so with an async KV connector prefill lands on nearly every step and the
    feature stops doing anything.
    """
    interval = 4
    scheduler = create_scheduler(prefill_schedule_interval=interval)
    _start_decoding(scheduler)
    prefill_at = scheduler.num_executed_steps

    (new_req,) = create_requests(num_requests=1, num_tokens=8, req_ids=["new0"])
    scheduler.add_request(new_req)

    empty_calls = 0
    admitted_at = None
    for _ in range(50):
        output = scheduler.schedule()
        if "new0" in output.num_scheduled_tokens:
            admitted_at = scheduler.num_executed_steps
            break
        if output.total_num_scheduled_tokens == 0:
            empty_calls += 1
            continue
        # A real decode step. Leave it in flight and re-enter schedule() twice,
        # as async scheduling does while the batch queue has room.
        for _ in range(2):
            extra = scheduler.schedule()
            if "new0" in extra.num_scheduled_tokens:
                admitted_at = scheduler.num_executed_steps
                break
            empty_calls += 1
        if admitted_at is not None:
            break
        _decode_one(scheduler, output, ["run0"])

    assert admitted_at is not None, "prefill was never admitted"
    assert empty_calls > 0, "test never exercised an empty schedule() call"
    # The hold is measured in executed steps. Counting schedule() calls would
    # admit after far fewer, because the empty re-entries above would count.
    assert admitted_at - prefill_at == interval, (
        f"hold lasted {admitted_at - prefill_at} executed steps, expected "
        f"{interval}; empty schedule() calls were counted against the cadence"
    )
    assert scheduler.current_step > scheduler.num_executed_steps


def test_local_prefill_interval_defaults_to_no_gating():
    """The default interval of 1 leaves admission exactly as it was: a waiting
    prefill is admitted on the very next step.
    """
    scheduler = create_scheduler()
    assert scheduler.local_prefill_interval == 1
    _start_decoding(scheduler)

    (new_req,) = create_requests(num_requests=1, num_tokens=8, req_ids=["new0"])
    scheduler.add_request(new_req)
    output = scheduler.schedule()
    assert "new0" in output.num_scheduled_tokens


def test_local_prefill_interval_left_to_engine_under_dp():
    """Under data parallelism the cadence must stay aligned across ranks, so the
    DP engine core drives it and the scheduler must not gate on its own.
    """
    scheduler = create_scheduler(prefill_schedule_interval=4, data_parallel_size=2)
    assert scheduler.local_prefill_interval == 1
    _start_decoding(scheduler)

    (new_req,) = create_requests(num_requests=1, num_tokens=8, req_ids=["new0"])
    scheduler.add_request(new_req)
    output = scheduler.schedule()
    assert "new0" in output.num_scheduled_tokens


def test_local_prefill_interval_never_idles_the_model():
    """With nothing running to decode in prefill's place, the interval must not
    hold prefill back, or the step would be wasted entirely.
    """
    scheduler = create_scheduler(prefill_schedule_interval=8)
    _start_decoding(scheduler)

    # Finish the only running request, leaving nothing to decode.
    scheduler.finish_requests("run0", RequestStatus.FINISHED_ABORTED)
    assert not scheduler.running

    (new_req,) = create_requests(num_requests=1, num_tokens=8, req_ids=["new0"])
    scheduler.add_request(new_req)
    output = scheduler.schedule()
    assert "new0" in output.num_scheduled_tokens


@pytest.mark.parametrize("interval", [2, 3])
def test_local_prefill_interval_defers_inflight_prefill_chunk(interval: int):
    """The single-engine interval gates prefill WORK, not just new admissions.

    A request that is already partially prefilled is held for the remainder of
    the interval too, exactly as `throttle_prefills` does for DP. That is the
    point of the cadence -- a held step must carry no prefill at all, or decode
    does not get the uninterrupted stretch the interval exists to give it.
    """
    scheduler = create_scheduler(
        max_num_seqs=16,
        max_num_batched_tokens=50,
        enable_chunked_prefill=True,
        prefill_schedule_interval=interval,
    )

    # A short request that finishes prefill in one step -> a running decode.
    (decode_req,) = create_requests(num_requests=1, num_tokens=4, req_ids=["dec0"])
    scheduler.add_request(decode_req)
    output = scheduler.schedule()
    _decode_one(scheduler, output, ["dec0"])
    assert decode_req in scheduler.running and not decode_req.is_prefill_chunk

    # A long request (80 tokens, budget 50) -> prefilled in chunks. That first
    # prefill closed the interval, so walk to the next step that opens it.
    (chunk_req,) = create_requests(num_requests=1, num_tokens=80, req_ids=["chk0"])
    scheduler.add_request(chunk_req)
    for _ in range(interval - 1):
        output = scheduler.schedule()
        assert "chk0" not in output.num_scheduled_tokens
        _decode_one(scheduler, output, ["dec0"])

    output = scheduler.schedule()
    assert output.num_scheduled_tokens["chk0"] > 0
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=["dec0", "chk0"],
            req_id_to_index={"dec0": 0, "chk0": 1},
            sampled_token_ids=[[0], []],  # no token sampled for partial prefill
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert chunk_req.is_prefill_chunk  # still mid-prefill, in running

    # Held steps: the in-flight chunk is deferred along with new admissions,
    # while the decode keeps running.
    for _ in range(interval - 1):
        output = scheduler.schedule()
        assert "chk0" not in output.num_scheduled_tokens
        assert "dec0" in output.num_scheduled_tokens
        _decode_one(scheduler, output, ["dec0"])

    # Once the interval has elapsed the chunk resumes.
    output = scheduler.schedule()
    assert "chk0" in output.num_scheduled_tokens


def _setup_cache_aware(num_tokens: int = 48, **kwargs):
    """A scheduler where `warm`'s prompt is already cached and `cold`'s is not.

    Returns `(scheduler, warm, cold)`, with neither request added yet.

    The threshold is pinned rather than inherited. These fixtures sit at ~0 KV
    cache usage, so under the shipped default of 0.5 the reorder would never
    run and every test below would pass without exercising it. Tests that are
    about the threshold itself pass their own value.
    """
    kwargs.setdefault("cache_aware_admission_threshold", 0)
    scheduler = create_scheduler(enable_prefix_caching=True, **kwargs)
    seed, warm = create_requests(
        num_requests=2,
        num_tokens=num_tokens,
        same_prompt=True,
        req_ids=["seed", "warm"],
    )
    # Only index 0 of a `same_prompt` batch collides with a distinct-prompt
    # batch, so index 1 here is a prompt no other request shares.
    cold = create_requests(
        num_requests=2, num_tokens=num_tokens, req_ids=["unused", "cold"]
    )[1]

    # Run `seed` far enough to leave its prompt in the prefix cache.
    scheduler.add_request(seed)
    output = scheduler.schedule()
    assert "seed" in output.num_scheduled_tokens
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=["seed"],
            req_id_to_index={"seed": 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    scheduler.finish_requests("seed", RequestStatus.FINISHED_ABORTED)

    assert scheduler.kv_cache_manager.get_num_cached_tokens(warm) > 0
    assert scheduler.kv_cache_manager.get_num_cached_tokens(cold) == 0
    return scheduler, warm, cold


def test_cache_aware_admission_prefers_cached_prefix():
    """Within the window, a request whose prefix is already cached is admitted
    ahead of one that would have to compute its own, so the cold request cannot
    evict blocks the resident one still needs.
    """
    scheduler, warm, cold = _setup_cache_aware(cache_aware_admission_window=4)
    scheduler.add_request(cold)
    scheduler.add_request(warm)

    output = scheduler.schedule()
    assert [r.req_id for r in output.scheduled_new_reqs] == ["warm", "cold"]


@pytest.mark.parametrize("window", [0, 4])
def test_cache_aware_admission_protects_resident_prefix_from_eviction(window: int):
    """The point of the reordering, not just its ordering.

    With the cache too small to hold both prompts, whichever request is
    admitted first keeps its blocks. Arrival order puts the cold request first,
    so only the reordering can stop it evicting the prefix the warm request
    still needs. `window=0` is the control: same setup, reordering off.
    """
    # 48-token prompts at block_size 16 are 3 blocks each; 5 blocks cannot hold
    # both, so admitting one evicts the other's cached prefix.
    scheduler, warm, cold = _setup_cache_aware(
        num_tokens=48,
        num_blocks=5,
        block_size=16,
        cache_aware_admission_window=window,
    )
    cached_before = scheduler.kv_cache_manager.get_num_cached_tokens(warm)
    assert cached_before > 0

    scheduler.add_request(cold)
    scheduler.add_request(warm)
    output = scheduler.schedule()
    scheduled = [r.req_id for r in output.scheduled_new_reqs]

    if window:
        # The warm request goes first and is admitted against its cached
        # prefix, so it computes fewer tokens than its prompt.
        assert scheduled[0] == "warm"
        assert output.num_scheduled_tokens["warm"] < warm.num_prompt_tokens
    else:
        # Arrival order wins: the cold request takes the blocks it needs and
        # the warm request's resident prefix is partly evicted, so the hit it
        # would have had shrinks and those tokens must be recomputed.
        assert scheduled[0] == "cold"
        assert scheduler.kv_cache_manager.get_num_cached_tokens(warm) < cached_before


def test_cache_aware_admission_keeps_blocked_and_started_requests_in_place():
    """Reordering is confined to plain waiting peers.

    A request in a blocked status, and one that is already partially computed,
    each keep their exact index; nothing is moved across them, and the queue
    beyond the window is untouched. This is the property that makes this a
    reorder among peers rather than a priority change.
    """
    scheduler, warm, cold = _setup_cache_aware(cache_aware_admission_window=4)

    blocked, started, tail = create_requests(
        num_requests=3, num_tokens=48, req_ids=["blocked", "started", "tail"]
    )
    # Enqueue everything as a plain waiting request first: a request that is
    # already blocked on arrival is routed to `skipped_waiting` instead, and
    # the guard under test is for one whose state changed while queued.
    for request in (cold, blocked, started, warm, tail):
        scheduler.add_request(request)
    blocked.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    started.num_computed_tokens = 16

    before = [r.request_id for r in scheduler.waiting]
    assert before == ["cold", "blocked", "started", "warm", "tail"]

    scheduler._reorder_waiting_by_cached_prefix()
    after = [r.request_id for r in scheduler.waiting]

    # Only `cold` and `warm` are plain waiting peers inside the 4-slot window,
    # so only they swap. `blocked` holds index 1, `started` holds index 2, and
    # `tail` sits beyond the window untouched.
    assert after == ["warm", "blocked", "started", "cold", "tail"]


def test_cache_aware_admission_does_not_look_past_the_window():
    """The window bounds how far a cached request can be promoted from, so a
    request outside it keeps its place in arrival order.
    """
    scheduler, warm, cold = _setup_cache_aware(cache_aware_admission_window=1)
    scheduler.add_request(cold)
    scheduler.add_request(warm)

    output = scheduler.schedule()
    assert [r.req_id for r in output.scheduled_new_reqs] == ["cold", "warm"]


def test_cache_aware_admission_skipped_below_threshold():
    """Below the usage threshold there are no evictions worth avoiding, so
    admission stays in arrival order.
    """
    scheduler, warm, cold = _setup_cache_aware(
        cache_aware_admission_window=4, cache_aware_admission_threshold=0.9
    )
    assert scheduler.kv_cache_manager.usage < 0.9
    scheduler.add_request(cold)
    scheduler.add_request(warm)

    output = scheduler.schedule()
    assert [r.req_id for r in output.scheduled_new_reqs] == ["cold", "warm"]


def test_cache_aware_admission_is_safe_on_a_mamba_spec():
    """The probe must not assume full attention.

    `get_num_cached_tokens` goes through the KV cache coordinator, so a group
    whose spec is not full attention has to be handled rather than crash. A
    Mamba group keeps recurrent state rather than a reusable token prefix, so
    the probe finds nothing to reuse and the reordering degrades to a no-op
    instead of shuffling the queue on meaningless scores.
    """
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        cache_aware_admission_window=4,
        cache_aware_admission_threshold=0.0,
        kv_cache_spec=MambaSpec(
            block_size=16,
            shapes=((1, 1),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        ),
    )
    assert scheduler.cache_aware_window == 4

    requests = create_requests(num_requests=3, num_tokens=48, req_ids=["a", "b", "c"])
    for request in requests:
        scheduler.add_request(request)

    # Probing is safe, reports no reusable prefix, and leaves arrival order.
    assert all(
        scheduler.kv_cache_manager.get_num_cached_tokens(r) == 0 for r in requests
    )
    scheduler._reorder_waiting_by_cached_prefix()
    assert [r.request_id for r in scheduler.waiting] == ["a", "b", "c"]


@pytest.mark.parametrize(
    "kwargs",
    [
        # Reordering by queue position is meaningless in a priority heap.
        {"scheduling_policy": "priority", "enable_prefix_caching": True},
        # Nothing to reorder by without a prefix cache.
        {"enable_prefix_caching": False},
    ],
)
def test_cache_aware_admission_requires_fcfs_and_prefix_caching(kwargs: dict):
    scheduler = create_scheduler(cache_aware_admission_window=4, **kwargs)
    assert scheduler.cache_aware_window == 0


def _setup_remote_kv_resume(num_prompt_tokens: int, matched_tokens: int):
    """Drive a remote-KV request `r2` to the resume point (async load complete)
    while another request `r1` is already decoding, so the step is throttle-
    eligible. Returns the scheduler. The connector matches `matched_tokens` of
    `r2`'s prompt; the rest (if any) is local prefill.
    """
    from tests.v1.kv_connector.unit.utils import create_model_runner_output

    BLOCK_SIZE = 16
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(matched_tokens=matched_tokens, is_async=True),
        block_size=BLOCK_SIZE,
    )
    # Distinct prompts so r2 gets no local prefix cache hit from r1, only the
    # connector's external async load.
    r1, r2 = create_requests(
        num_requests=2,
        num_tokens=num_prompt_tokens,
        max_tokens=20,
        block_size=BLOCK_SIZE,
        req_ids=["r1", "r2"],
    )

    # r1: drive through its async KV load into the running (decode) state, so
    # self.running is non-empty (which makes the next step throttle-eligible).
    scheduler.add_request(r1)
    _step_until_kv_transfer_finished(scheduler, ["r1"])
    output = scheduler.schedule()  # promote + schedule r1
    assert "r1" in output.num_scheduled_tokens
    scheduler.update_from_output(
        output, create_model_runner_output([r1], token_id=1000)
    )
    assert scheduler.running  # r1 now decoding

    # r2: a second remote-KV request; complete its async load while r1 decodes.
    scheduler.add_request(r2)
    output = scheduler.schedule()  # r1 decodes; r2 -> WAITING_FOR_REMOTE_KVS
    assert r2.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.update_from_output(
        output, create_model_runner_output([r1], finished_recving={"r2"})
    )
    assert "r2" in scheduler.finished_recving_kv_req_ids
    return scheduler


def test_throttle_prefills_excludes_fully_transferred_remote_kv():
    """A remote-KV resume whose whole prompt was transferred (no local prefill
    left, e.g. the decode side of P/D disaggregation) must NOT be throttled by
    the DP prefill cadence -- its single-token step has no prefill compute to
    defer, so delaying it would be pointless.
    """
    block_size = 16
    num_prompt = block_size * 2
    # Fully matched: the whole prompt is loaded remotely.
    scheduler = _setup_remote_kv_resume(num_prompt, matched_tokens=num_prompt)

    output = scheduler.schedule(throttle_prefills=True)
    assert "r2" in output.num_scheduled_tokens
    assert "r1" in output.num_scheduled_tokens


def test_throttle_prefills_defers_remote_kv_resume_with_local_prefill():
    """A remote-KV resume with local prefill still to compute (the connector
    only matched part of the prompt) IS throttled by the DP prefill cadence,
    like any other request doing local prefill compute this step.
    """
    block_size = 16
    num_prompt = block_size * 4
    # Half matched: the remaining half is local prefill compute.
    scheduler = _setup_remote_kv_resume(num_prompt, matched_tokens=num_prompt // 2)

    output = scheduler.schedule(throttle_prefills=True)
    assert "r2" not in output.num_scheduled_tokens  # deferred (has local prefill)
    assert "r1" in output.num_scheduled_tokens


def test_throttle_defers_inflight_prefill_chunk():
    """DP prefill balancing throttles ALL prefill compute on a throttled step,
    not just new admissions: an in-progress (chunked) prefill already in the
    running queue is also deferred, so the step runs decode-only, while a
    separate decode keeps being scheduled."""
    scheduler = create_scheduler(
        max_num_seqs=16, max_num_batched_tokens=50, enable_chunked_prefill=True
    )

    # A short request that finishes prefill in one step -> a running decode.
    (decode_req,) = create_requests(num_requests=1, num_tokens=4, req_ids=["dec0"])
    scheduler.add_request(decode_req)
    output = scheduler.schedule()
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=["dec0"],
            req_id_to_index={"dec0": 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert decode_req in scheduler.running and not decode_req.is_prefill_chunk

    # A long request (80 tokens, budget 50) -> prefilled in chunks.
    (chunk_req,) = create_requests(num_requests=1, num_tokens=80, req_ids=["chk0"])
    scheduler.add_request(chunk_req)
    output = scheduler.schedule()  # first chunk of chk0 + decode of dec0
    assert output.num_scheduled_tokens["chk0"] > 0
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=["dec0", "chk0"],
            req_id_to_index={"dec0": 0, "chk0": 1},
            sampled_token_ids=[[0], []],  # no token sampled for partial prefill
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert chunk_req.is_prefill_chunk  # still mid-prefill, in running

    # Throttled step: the in-flight prefill chunk is deferred, the decode runs.
    output = scheduler.schedule(throttle_prefills=True)
    assert "chk0" not in output.num_scheduled_tokens
    assert "dec0" in output.num_scheduled_tokens

    # When the cadence opens again, the prefill chunk resumes.
    output = scheduler.schedule()
    assert "chk0" in output.num_scheduled_tokens


def test_throttle_capacity_bound_guard_admits():
    """Saturation guard: if a cadence-aligned release step cannot drain the
    waiting prefill queue (it ran out of token budget), the throttle backs off on
    the next step so the backlog cannot grow into a TTFT avalanche -- prefills are
    admitted even though throttle_prefills is set."""
    scheduler = create_scheduler(
        max_num_seqs=16, max_num_batched_tokens=200, enable_chunked_prefill=True
    )
    a, b = create_requests(num_requests=2, num_tokens=200, req_ids=["a", "b"])
    scheduler.add_request(a)
    scheduler.add_request(b)

    # Release step (throttle off): `a` fills the 200-token budget; `b` cannot be
    # reached, so the waiting queue is not drained -> capacity-bound.
    output = scheduler.schedule()
    assert "a" in output.num_scheduled_tokens
    assert "b" not in output.num_scheduled_tokens
    assert scheduler.prefill_capacity_bound

    # Throttle. Because the previous release was capacity-bound, the guard backs
    # off and `b` is admitted rather than stalling the backlog.
    output = scheduler.schedule(throttle_prefills=True)
    assert "b" in output.num_scheduled_tokens


def test_same_step_duplicate_encoder_input_stays_cached():
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=1024,
        max_model_len=2048,
    )
    request = create_requests(
        1,
        num_tokens=2000,
        req_ids=["repeated"],
        mm_hashes_list=[["image", "image", "image"]],
        mm_positions=[
            [PlaceholderRange(offset=offset, length=576) for offset in (0, 600, 1300)]
        ],
    )[0]
    scheduler.add_request(request)

    output = scheduler.schedule()
    assert output.scheduled_encoder_inputs == {request.request_id: [0]}
    _model_output(scheduler, output, [[]])

    # The second occurrence is partially consumed; the third is not scheduled.
    cache = scheduler.encoder_cache_manager
    assert cache.get_cached_input_ids(request) == {1}
    assert "image" not in cache.freeable


@pytest.mark.parametrize("warm_encoder_cache", [False, True])
def test_no_mm_input_chunking(warm_encoder_cache: bool):
    # Disable multimodal input chunking.
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=1024,
        disable_chunked_mm_input=True,
        max_model_len=2048,
    )
    mm_positions = [[PlaceholderRange(offset=400, length=800)]]
    requests = create_requests(
        num_requests=1, num_tokens=1200, mm_positions=mm_positions
    )
    if warm_encoder_cache:
        cached_request = create_requests(
            num_requests=1,
            num_tokens=1200,
            mm_positions=mm_positions,
            req_ids=["cache-primer"],
        )[0]
        scheduler.encoder_cache_manager.allocate(cached_request, 0)
        scheduler.encoder_cache_manager.free(cached_request)

    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0
    # We want to only see the 400 text tokens at the start scheduled
    assert output.num_scheduled_tokens[requests[0].request_id] == 400

    req_to_index = {request.request_id: i for i, request in enumerate(requests)}
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[] for _ in range(len(requests))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_runner_output)

    output = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert len(output.scheduled_new_reqs) == 0
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert len(output.finished_req_ids) == 0
    assert output.num_scheduled_tokens[requests[0].request_id] == 800
    if warm_encoder_cache:
        assert not output.scheduled_encoder_inputs

    # Test that we fail if we disable chunked mm input and use too small
    # of a max_num_batched_tokens for the mm input.
    with pytest.raises(ValueError):
        _ = create_scheduler(
            model="llava-hf/llava-1.5-7b-hf",
            max_num_batched_tokens=100,
            disable_chunked_mm_input=True,
        )


def test_no_mm_input_chunking_for_same_step_duplicate():
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=1024,
        disable_chunked_mm_input=True,
        max_model_len=2048,
    )
    request = create_requests(
        num_requests=1,
        num_tokens=1352,
        mm_hashes_list=[["image", "image"]],
        mm_positions=[
            [
                PlaceholderRange(offset=100, length=576),
                PlaceholderRange(offset=776, length=576),
            ]
        ],
    )[0]
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == 776
    assert output.scheduled_encoder_inputs == {request.request_id: [0]}


def test_no_mm_input_chunking_at_item_boundary():
    """A chunk starting exactly at a multimodal item must not split it.

    Regression test for issue #52306: the rollback guard only fired while
    ``num_computed_tokens < start_pos``, so a chunk beginning exactly at the
    item skipped the check and could cover only part of it - silently breaking
    models with bidirectional attention over the item.
    """
    mm_positions = [[PlaceholderRange(offset=400, length=800)]]

    def ask(num_computed_tokens: int, num_new_tokens: int) -> int:
        # A fresh scheduler per call: the encoder cache is stateful.
        scheduler = create_scheduler(
            model="llava-hf/llava-1.5-7b-hf",
            max_num_batched_tokens=1024,
            disable_chunked_mm_input=True,
            max_model_len=2048,
        )
        request = create_requests(
            num_requests=1, num_tokens=1200, mm_positions=mm_positions
        )[0]
        return scheduler._try_schedule_encoder_inputs(
            request, num_computed_tokens, num_new_tokens, encoder_compute_budget=100_000
        )[1]

    # Starts before the item and cannot reach past it: stop just before it.
    assert ask(0, 500) == 400
    # Starts exactly at the item and cannot cover it: schedule nothing rather
    # than half an item.
    assert ask(400, 500) == 0
    # Starts exactly at the item and covers it: schedule it.
    assert ask(400, 800) == 800
    # Starts inside the item (a prefix cache hit ended there) and cannot finish
    # it: schedule nothing, never a negative count.
    assert ask(600, 500) == 0
    # Starts inside the item and finishes it: schedule the rest.
    assert ask(600, 600) == 600


def test_no_mm_input_chunking_after_prefix_cache_hit_inside_item():
    """A prefix cache hit can end inside a multimodal item (hits are block
    aligned). The rest of the item must then be scheduled in one step too."""
    block_size = 256
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=600,
        disable_chunked_mm_input=True,
        enable_prefix_caching=True,
        block_size=block_size,
        max_model_len=2048,
    )

    def image_request(req_id: str) -> Request:
        return create_requests(
            num_requests=1,
            num_tokens=590,
            mm_hashes_list=[["image"]],
            mm_positions=[[PlaceholderRange(offset=0, length=576)]],
            block_size=block_size,
            req_ids=[req_id],
        )[0]

    def step() -> dict[str, int]:
        output = scheduler.schedule()
        scheduler.update_from_output(output, make_output(scheduler))
        return output.num_scheduled_tokens

    # Caches the first two blocks, 512 of the 576 image tokens.
    scheduler.add_request(image_request("primer"))
    assert step() == {"primer": 590}

    # The text prefill leaves 600 - 560 - 1 = 39 tokens of budget, less than the
    # 64 image tokens left after the hit.
    scheduler.add_request(
        create_requests(
            num_requests=1, num_tokens=560, block_size=block_size, req_ids=["text"]
        )[0]
    )
    scheduler.add_request(image_request("hit"))
    assert step() == {"primer": 1, "text": 560}

    # The 64 image tokens and the 14 text tokens after them, in one chunk.
    assert step() == {"primer": 1, "text": 1, "hit": 78}


@pytest.mark.parametrize("enable_prefix_caching", [True, False])
def test_schedule_concurrent_partial_requests(enable_prefix_caching: bool):
    """Test scheduling behavior with concurrent partial requests.

    This test verifies that: there are multiple long prefill requests in the
    RUNNING state, and we can schedule them together.

    """
    scheduler = create_scheduler(
        model="facebook/opt-125m",
        max_num_batched_tokens=1024,
        long_prefill_token_threshold=400,
        enable_prefix_caching=enable_prefix_caching,
    )
    requests = create_requests(
        num_requests=3,
        num_tokens=800,
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 3
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0

    # The first request is scheduled partially - 400.
    assert output.num_scheduled_tokens[requests[0].request_id] == 400
    # The second request is scheduled partially - 400.
    assert output.num_scheduled_tokens[requests[1].request_id] == 400
    # The third request is also scheduled partially - 1024 - 400 - 400 = 224.
    assert output.num_scheduled_tokens[requests[2].request_id] == 224
    req_to_index = {request.request_id: i for i, request in enumerate(requests)}
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[] for _ in range(len(requests))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_runner_output)

    # Schedule the next step. All three requests are running.
    # Processed the remaining prefills of the first and second requests.
    output1 = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output1.scheduled_new_reqs) == 0
    assert output1.scheduled_cached_reqs.num_reqs == 3
    assert len(output1.finished_req_ids) == 0
    assert output1.num_scheduled_tokens[requests[0].request_id] == 400
    assert output1.num_scheduled_tokens[requests[1].request_id] == 400
    assert output1.num_scheduled_tokens[requests[2].request_id] == 224

    # Schedule the third step. All three requests are running.
    # First and second requests are in the decode stage.
    # All the remaining tokens in the third request are processed.
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[0], [0]] + [[] for _ in range(len(requests) - 2)],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output1, model_runner_output)
    output2 = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output2.scheduled_new_reqs) == 0
    assert output2.scheduled_cached_reqs.num_reqs == 3
    assert len(output2.finished_req_ids) == 0
    assert output2.num_scheduled_tokens[requests[0].request_id] == 1
    assert output2.num_scheduled_tokens[requests[1].request_id] == 1
    assert output2.num_scheduled_tokens[requests[2].request_id] == 800 - 224 - 224


def test_long_prefill_threshold_ignored_when_alone():
    """A lone long prefill is not capped by the threshold: it has no other
    request to starve."""
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        long_prefill_token_threshold=400,
    )
    request = create_requests(num_requests=1, num_tokens=2000)[0]
    scheduler.add_request(request)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens[request.request_id] == 1024


def test_long_prefill_threshold_applies_with_other_requests():
    """The threshold caps the prefill as soon as another request is queued."""
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        long_prefill_token_threshold=400,
    )
    long_req = create_requests(num_requests=1, num_tokens=2000)[0]
    short_req = create_requests(num_requests=1, num_tokens=10, req_ids=["short"])[0]
    for request in [long_req, short_req]:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens[long_req.request_id] == 400
    assert output.num_scheduled_tokens[short_req.request_id] == 10


def test_long_prefill_threshold_floored_by_fair_share():
    """With the adaptive flag, the effective threshold never falls below the
    fair share of the token budget: max_num_batched_tokens / num queued +
    running requests."""
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        long_prefill_token_threshold=100,
        long_prefill_token_threshold_adaptive=True,
    )
    long_req = create_requests(num_requests=1, num_tokens=2000)[0]
    short_req = create_requests(num_requests=1, num_tokens=10, req_ids=["short"])[0]
    for request in [long_req, short_req]:
        scheduler.add_request(request)

    output = scheduler.schedule()
    # 100 is below the fair share (1024 // 2 = 512), so the floor binds.
    assert output.num_scheduled_tokens[long_req.request_id] == 512
    assert output.num_scheduled_tokens[short_req.request_id] == 10


def test_update_from_output_routes_sampling_masks_by_request():
    """Each request receives the sampler row at its own batch index."""
    scheduler = create_scheduler()
    scheduler.return_sampling_mask = True
    requests = create_requests(num_requests=3, max_tokens=10)
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        req.status = RequestStatus.RUNNING

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req.request_id: 1 for req in requests},
        total_num_scheduled_tokens=3,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[1], [3], [4]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        sampling_masks=SamplingMaskLists(
            np.array([1, 2, 3, 4, 5, 6], dtype=np.int32), np.array([0, 2, 3, 6])
        ),
    )

    outputs = scheduler.update_from_output(scheduler_output, model_output)[0].outputs

    assert [out.request_id for out in outputs] == [req.request_id for req in requests]
    assert [out.new_sampling_mask.token_ids.tolist() for out in outputs] == [
        [1, 2],
        [3],
        [4, 5, 6],
    ]
    assert all(out.new_sampling_mask.offsets is None for out in outputs)


def test_update_from_output_routes_multi_position_sampling_masks():
    scheduler = create_scheduler()
    scheduler.return_sampling_mask = True
    requests = create_requests(num_requests=2, max_tokens=10)
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        req.status = RequestStatus.RUNNING

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req.request_id: 1 for req in requests},
        total_num_scheduled_tokens=2,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[1, 2], [3, 4, 5]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        sampling_masks=SamplingMaskLists(
            token_ids=np.array([1, 6, 2, 3, 7, 8, 4, 5, 9], dtype=np.int32),
            offsets=np.array([0, 2, 3, 6, 7, 9]),
            cu_num_generated_tokens=[0, 2, 5],
        ),
    )

    outputs = scheduler.update_from_output(scheduler_output, model_output)[0].outputs

    assert [out.new_sampling_mask.to_nested_list() for out in outputs] == [
        [[1, 6], [2]],
        [[3, 7, 8], [4], [5, 9]],
    ]


def test_stop_via_update_from_output():
    """Test stopping behavior through update_from_output."""
    scheduler = create_scheduler(num_speculative_tokens=1)

    # Test case 1: Stop on EOS token
    requests = create_requests(num_requests=2, max_tokens=10)
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        req.status = RequestStatus.RUNNING

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={requests[0].request_id: 1, requests[1].request_id: 2},
        total_num_scheduled_tokens=3,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={
            requests[0].request_id: [],
            requests[1].request_id: [10],
        },
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[
            [EOS_TOKEN_ID],
            [10, 11],
        ],  # First request hits EOS, second continues
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify first request stopped, second continues
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[1].request_id
    assert requests[0].status == RequestStatus.FINISHED_STOPPED
    assert requests[0].request_id in scheduler.finished_req_ids
    assert list(requests[0].output_token_ids) == [EOS_TOKEN_ID]
    assert list(requests[1].output_token_ids) == [10, 11]

    # Test case 2: Stop on custom stop token
    scheduler = create_scheduler(num_speculative_tokens=2)
    requests = create_requests(num_requests=2, max_tokens=10, stop_token_ids=[42, 43])
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        req.status = RequestStatus.RUNNING

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={requests[0].request_id: 3, requests[1].request_id: 2},
        total_num_scheduled_tokens=5,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={
            requests[0].request_id: [10, 42],
            requests[1].request_id: [13],
        },
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[10, 42, 12], [13, 14]],  # First request hits stop token
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify first request stopped on custom token
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[1].request_id
    assert requests[0].status == RequestStatus.FINISHED_STOPPED
    assert requests[0].stop_reason == 42
    assert requests[0].request_id in scheduler.finished_req_ids
    assert list(requests[0].output_token_ids) == [10, 42]
    assert list(requests[1].output_token_ids) == [13, 14]

    # Test case 3: Stop on max tokens
    scheduler = create_scheduler(num_speculative_tokens=2)
    requests = create_requests(num_requests=2, max_tokens=2)
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        req.status = RequestStatus.RUNNING

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={requests[0].request_id: 3, requests[1].request_id: 1},
        total_num_scheduled_tokens=4,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={
            requests[0].request_id: [10, 11],
            requests[1].request_id: [],
        },
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[10, 11, 12], [13]],  # First request exceeds max_tokens
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify first request stopped due to length
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[1].request_id
    assert requests[0].status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert requests[0].request_id in scheduler.finished_req_ids
    assert list(requests[0].output_token_ids) == [10, 11]  # Truncated to max_tokens
    assert list(requests[1].output_token_ids) == [13]

    # Test case 4: Ignore EOS flag
    scheduler = create_scheduler(num_speculative_tokens=2)
    requests = create_requests(num_requests=1, max_tokens=10, ignore_eos=True)
    requests[0].num_computed_tokens = requests[0].num_tokens
    scheduler.requests[requests[0].request_id] = requests[0]
    scheduler.running.append(requests[0])

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={requests[0].request_id: 3},
        total_num_scheduled_tokens=3,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={requests[0].request_id: [EOS_TOKEN_ID, 10]},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[EOS_TOKEN_ID, 10, 11]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify request continues past EOS
    assert len(scheduler.running) == 1
    assert not requests[0].is_finished()
    assert list(requests[0].output_token_ids) == [EOS_TOKEN_ID, 10, 11]


def test_check_stop_min_tokens():
    """Test that sampled stop tokens still finish under min_tokens."""
    from vllm.v1.core.sched.utils import check_stop

    # Test case 1: regular tokens below min_tokens should keep running.
    sampling_params = SamplingParams(
        ignore_eos=False,
        max_tokens=20,
        min_tokens=5,
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id="0",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params,
        pooling_params=None,
    )
    request.append_output_token_ids([10, 11, 12])

    result = check_stop(request, max_model_len=100)
    assert result is False, "Should not stop when num_output_tokens<min_tokens"

    # Test case 2: if EOS is sampled, it should stop even below min_tokens.
    request_eos = Request(
        request_id="eos",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params,
        pooling_params=None,
    )
    request_eos.append_output_token_ids([10, 11, EOS_TOKEN_ID])
    result = check_stop(request_eos, max_model_len=100)
    assert result is True, "Sampled EOS should stop even below min_tokens"
    assert request_eos.status == RequestStatus.FINISHED_STOPPED

    # Test case 3: ignore_eos should still prevent EOS-based stopping.
    sampling_params_ignore_eos = SamplingParams(
        ignore_eos=True,
        max_tokens=20,
        min_tokens=5,
    )
    sampling_params_ignore_eos.update_from_generation_config({}, EOS_TOKEN_ID)
    request_ignore_eos = Request(
        request_id="ignore-eos",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params_ignore_eos,
        pooling_params=None,
    )
    request_ignore_eos.append_output_token_ids([EOS_TOKEN_ID])
    result = check_stop(request_ignore_eos, max_model_len=100)
    assert result is False, "ignore_eos should still suppress EOS stopping"

    # Test case 4: min_tokens = 0 should follow normal stopping logic.
    sampling_params_no_min = SamplingParams(
        ignore_eos=False,
        max_tokens=20,
        min_tokens=0,
    )
    sampling_params_no_min.update_from_generation_config({}, EOS_TOKEN_ID)
    request_no_min = Request(
        request_id="1",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params_no_min,
        pooling_params=None,
    )
    request_no_min.append_output_token_ids([10, EOS_TOKEN_ID])

    result = check_stop(request_no_min, max_model_len=100)
    assert result is True, "Should stop on EOS when min_tokens=0"
    assert request_no_min.status == RequestStatus.FINISHED_STOPPED

    # Test case 5: if a stop token is sampled, it should stop immediately.
    sampling_params_stop = SamplingParams(
        ignore_eos=False,
        max_tokens=20,
        min_tokens=5,
        stop_token_ids=[42],
    )
    sampling_params_stop.update_from_generation_config({}, EOS_TOKEN_ID)
    request_stop = Request(
        request_id="2",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params_stop,
        pooling_params=None,
    )
    request_stop.append_output_token_ids([10, 11, 42])
    result = check_stop(request_stop, max_model_len=100)
    assert result is True, "Sampled stop tokens should stop even below min_tokens"
    assert request_stop.status == RequestStatus.FINISHED_STOPPED
    assert request_stop.stop_reason == 42


def test_check_stop_length_cap_below_min_tokens():
    """Length caps must finish a request even below min_tokens.

    Guards against requests hanging forever when min_tokens cannot be
    satisfied: min_tokens only defers EOS/stop tokens, so a request that
    reaches max_model_len (or a max_tokens clamped below min_tokens) must
    still finish as FINISHED_LENGTH_CAPPED instead of spinning in the
    scheduler with zero new tokens to schedule.
    """
    from vllm.v1.core.sched.utils import check_stop

    # Reaching max_model_len below min_tokens must stop the request.
    sampling_params = SamplingParams(
        ignore_eos=True,
        max_tokens=500,
        min_tokens=500,
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id="0",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params,
        pooling_params=None,
    )
    # 7 output tokens: num_tokens == 10 == max_model_len, min_tokens unmet.
    request.append_output_token_ids([10, 11, 12, 13, 14, 15, 16])

    result = check_stop(request, max_model_len=10)
    assert result is True, "Should stop at max_model_len even below min_tokens"
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED

    # Reaching max_tokens below min_tokens must stop the request. This can
    # happen when max_tokens is clamped to the context window after
    # SamplingParams validation (e.g. max_tokens=None on a long prompt).
    sampling_params_capped = SamplingParams(
        ignore_eos=True,
        max_tokens=500,
        min_tokens=500,
    )
    sampling_params_capped.max_tokens = 3
    sampling_params_capped.update_from_generation_config({}, EOS_TOKEN_ID)
    request_capped = Request(
        request_id="1",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params_capped,
        pooling_params=None,
    )
    request_capped.append_output_token_ids([10, 11, 12])

    result = check_stop(request_capped, max_model_len=100)
    assert result is True, "Should stop at max_tokens even below min_tokens"
    assert request_capped.status == RequestStatus.FINISHED_LENGTH_CAPPED


@pytest.mark.parametrize(
    "enable_prefix_caching, prompt_logprobs",
    [
        (False, None),
        (True, 5),
    ],
)
def test_schedule_concurrent_batches(
    enable_prefix_caching: bool, prompt_logprobs: int | None
):
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=2,
        enable_prefix_caching=enable_prefix_caching,
    )
    requests = create_requests(
        num_requests=2,
        num_tokens=512,
        prompt_logprobs=prompt_logprobs,
    )

    # Schedule the first request.
    scheduler.add_request(requests[0])
    scheduler_output0 = scheduler.schedule()
    assert len(scheduler_output0.scheduled_new_reqs) == 1
    assert scheduler_output0.num_scheduled_tokens[requests[0].request_id] == 512

    # The first request is still running, so only schedule the second request.
    scheduler.add_request(requests[1])
    scheduler_output1 = scheduler.schedule()
    assert len(scheduler_output1.scheduled_new_reqs) == 1
    assert scheduler_output1.num_scheduled_tokens[requests[1].request_id] == 512

    # Model output of the first request.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(scheduler_output0, model_runner_output)

    # Schedule the next step.
    # The first request can be scheduled again while the second
    # request is still running.
    scheduler_output2 = scheduler.schedule()
    assert scheduler_output2.num_scheduled_tokens[requests[0].request_id] == 1

    # Model output of the second request.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[1].request_id],
        req_id_to_index={requests[1].request_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(scheduler_output1, model_runner_output)


@pytest.mark.parametrize("enable_chunked_prefill", [True, False])
def test_schedule_order(enable_chunked_prefill: bool):
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=3,
        enable_chunked_prefill=enable_chunked_prefill,
    )

    # long requests
    requests = create_requests(num_requests=2, num_tokens=800, req_ids=["1", "2"])
    # short requests
    requests += create_requests(num_requests=2, num_tokens=10, req_ids=["3", "4"])

    for request in requests:
        scheduler.add_request(request)

    scheduler_output1 = scheduler.schedule()

    if enable_chunked_prefill:
        # When enable chunked prefill, long requests will be chunked.
        assert len(scheduler_output1.scheduled_new_reqs) == 2
    else:
        # When disable chunked prefill, should not skip the long requests,
        # and scheduling subsequent short requests in advance,
        # even though there is still token budgets remaining.
        assert len(scheduler_output1.scheduled_new_reqs) == 1


def test_preempt_during_execution():
    # NOTE(woosuk): The actual number of available blocks is 10 instead of 11
    # because block 0 is reserved as the null block.
    scheduler = create_scheduler(
        max_num_batched_tokens=100,
        block_size=16,
        num_blocks=11,
        enable_prefix_caching=False,
    )
    requests = create_requests(num_requests=2, num_tokens=80, block_size=16)

    # Schedule the first request.
    scheduler.add_request(requests[0])
    scheduler_output0 = scheduler.schedule()
    assert len(scheduler_output0.num_scheduled_tokens) == 1
    assert len(scheduler_output0.scheduled_new_reqs[0].block_ids[0]) == 5

    # Schedule the second request while the first request is still running.
    # This scenario can occur in certain cases, when max_concurrent_batches > 1
    # (e.g., when pipeline parallelism is used).
    scheduler.add_request(requests[1])
    scheduler_output1 = scheduler.schedule()
    assert len(scheduler_output1.num_scheduled_tokens) == 1
    assert len(scheduler_output1.scheduled_new_reqs[0].block_ids[0]) == 5

    # Get the output of the first request.
    model_runner_output0 = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(scheduler_output0, model_runner_output0)

    # Schedule the first request again. This will cause the preemption
    # of the second request because the KV cache is full.
    _ = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert scheduler.running[0] == requests[0]
    assert requests[1].status == RequestStatus.PREEMPTED

    model_runner_output1 = ModelRunnerOutput(
        req_ids=[requests[1].request_id],
        req_id_to_index={requests[1].request_id: 0},
        sampled_token_ids=[[42]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(scheduler_output1, model_runner_output1)

    # The second request (that is preempted) should be updated with the
    # sampled token id.
    assert len(requests[1].output_token_ids) == 1
    assert requests[1].output_token_ids[0] == 42


def test_pending_hisparse_reclamation_defers_preemption(monkeypatch):
    """Pending reclamation must stall rather than preempt its owning request."""
    scheduler = create_scheduler(enable_prefix_caching=False)
    request = create_requests(num_requests=1, num_tokens=16, block_size=16)[0]
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    monkeypatch.setattr(
        scheduler.kv_cache_manager, "allocate_slots", Mock(return_value=None)
    )
    coordinator = _bind_hisparse_connector(scheduler)
    monkeypatch.setattr(
        scheduler.connector, "build_connector_meta", Mock(return_value=None)
    )
    monkeypatch.setattr(
        coordinator,
        "has_pending_reclamation",
        Mock(return_value=True),
    )

    deferred_output = scheduler.schedule()

    assert not deferred_output.num_scheduled_tokens
    assert scheduler.running == [request]
    assert request.status == RequestStatus.RUNNING
    assert request.num_preemptions == 0


def test_prefix_cache_query_not_inflated_by_connector_defer():
    """The GPU prefix-cache query is recorded at admission, so a request the
    connector defers several times is counted once, not once per retry."""
    num_defers_before_matching = 3
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(
            matched_tokens=0,
            is_async=False,
            num_defers_before_matching=num_defers_before_matching,
        ),
    )
    request = create_requests(num_requests=1, num_tokens=32, block_size=16)[0]
    scheduler.add_request(request)

    # Each deferred step re-runs the lookup but records nothing.
    for _ in range(num_defers_before_matching):
        assert not scheduler.schedule().scheduled_new_reqs

    output = scheduler.schedule()
    assert any(r.req_id == request.request_id for r in output.scheduled_new_reqs)

    stats = scheduler.kv_cache_manager.prefix_cache_stats
    assert stats is not None
    assert stats.requests == 1
    assert stats.queries == request.num_tokens


def test_preemption_re_records_prefix_cache_query():
    """A preempted request re-enters the lookup on resume, so its recomputation
    is counted again into the preempted stats."""
    scheduler = create_scheduler(enable_prefix_caching=True)
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    stats = scheduler.kv_cache_manager.prefix_cache_stats
    assert stats is not None
    assert (stats.requests, stats.preempted_requests) == (1, 0)

    scheduler.running.remove(request)
    scheduler._preempt_request(request, 0.0)
    assert request.status == RequestStatus.PREEMPTED

    scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[1000]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert request.num_stale_output_tokens == 0
    stats = scheduler.kv_cache_manager.prefix_cache_stats
    assert stats is not None

    scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    assert stats.preempted_requests == 1


def test_preemption_processes_stale_aux_output():
    scheduler = create_scheduler(enable_prefix_caching=True)
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    scheduler.running.remove(request)
    scheduler.aux_output_connector = Mock()
    scheduler._preempt_request(request, 0.0)
    scheduler.aux_output_connector.request_finished.assert_called_once_with(request)

    scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[1000]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    scheduler.aux_output_connector.take_output.assert_called_once_with(request, None)


def test_prefix_cache_stats_not_recorded_when_caching_disabled():
    """With prefix caching off there is no local lookup, so admitting a request
    records no phantom miss."""
    scheduler = create_scheduler(enable_prefix_caching=False)
    for request in create_requests(num_requests=2):
        scheduler.add_request(request)

    scheduler.schedule()

    stats = scheduler.kv_cache_manager.prefix_cache_stats
    assert stats is not None
    assert (stats.requests, stats.queries, stats.hits) == (0, 0, 0)


def test_prefix_cache_stats_counted_once_for_retried_then_scheduled_request():
    """A real cache hit rejected once by allocate_slots and admitted on the next
    step is counted exactly once, hits included."""
    block_size = 16
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        block_size=block_size,
    )

    # Seed the cache so the next request with the same prompt hits it.
    seed = create_requests(
        num_requests=1,
        num_tokens=block_size * 2,
        max_tokens=2,
        same_prompt=True,
        block_size=block_size,
        req_ids=["seed"],
    )[0]
    scheduler.add_request(seed)
    _step_until_done(
        scheduler,
        scheduler.schedule(),
        ModelRunnerOutput(
            req_ids=["seed"],
            req_id_to_index={"seed": 0},
            sampled_token_ids=[[1000]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    # The seeding step swapped in a fresh accumulator, so re-read it; the
    # retried request must be the only thing recorded from here on.
    stats = scheduler.kv_cache_manager.prefix_cache_stats
    assert stats is not None
    assert (stats.requests, stats.queries, stats.hits) == (0, 0, 0)

    retried = create_requests(
        num_requests=1,
        num_tokens=block_size * 3,
        max_tokens=1,
        same_prompt=True,
        block_size=block_size,
        req_ids=["retried"],
    )[0]
    scheduler.add_request(retried)

    # Reject the first allocation attempt, then delegate to the real one.
    orig_allocate_slots = scheduler.kv_cache_manager.allocate_slots
    allocate_results: list = []

    def spy_allocate_slots(*args, **kwargs):
        result = None if not allocate_results else orig_allocate_slots(*args, **kwargs)
        allocate_results.append(result)
        return result

    scheduler.kv_cache_manager.allocate_slots = spy_allocate_slots

    assert not scheduler.schedule().scheduled_new_reqs
    assert (stats.requests, stats.queries, stats.hits) == (0, 0, 0)

    assert "retried" in scheduler.schedule().num_scheduled_tokens
    assert allocate_results[0] is None and allocate_results[1] is not None
    assert (stats.requests, stats.queries, stats.hits) == (
        1,
        retried.num_tokens,
        block_size * 2,
    )


def test_scheduler_reset_prefix_cache():
    scheduler = create_scheduler(enable_prefix_caching=True)
    scheduler.aux_output_connector = Mock(
        wraps=AuxOutputSchedulerConnector(scheduler.kv_cache_manager)
    )
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    # Initial scheduling, requests should be at the running state now
    _ = scheduler.schedule()

    # Verify requests moved from waiting to running
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == len(requests)
    for i, request in enumerate(requests):
        assert scheduler.running[i] == request

    # Reset prefix cache should fail since there are still running requests
    # and they are taking KV cache
    assert not scheduler.reset_prefix_cache()
    scheduler.aux_output_connector.reset.assert_not_called()

    # Pause completes pending model outputs before the caller resets the scheduler.
    for request in requests:
        request.num_in_flight_tokens = 0

    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    scheduler.aux_output_connector.reset.assert_called_once_with()

    # Verify requests moved from running to waiting
    assert len(scheduler.waiting) == len(requests)
    assert len(scheduler.running) == 0

    for i, request in enumerate(requests):
        assert scheduler.waiting[i] == request


@pytest.mark.parametrize("reset_successful", [False, True])
def test_aux_output_reset_follows_kv_reset_result(reset_successful: bool):
    scheduler = create_scheduler(enable_prefix_caching=True)
    scheduler.aux_output_connector = Mock()
    scheduler.kv_cache_manager.reset_prefix_cache = Mock(return_value=reset_successful)

    assert scheduler.reset_prefix_cache() is reset_successful
    assert scheduler.aux_output_connector.reset.call_count == int(reset_successful)


def test_kv_cache_release_after_keep_pause_preserves_requests():
    """A completed keep pause permits release and recomputation of live requests."""
    scheduler = create_scheduler(max_num_seqs=1, enable_prefix_caching=True)
    running, waiting = create_requests(num_requests=2)
    for request in (running, waiting):
        scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=[running.request_id],
            req_id_to_index={running.request_id: 0},
            sampled_token_ids=[[1]],
        ),
    )
    assert scheduler.kv_cache_manager.get_block_ids(running.request_id)[0]

    core = object.__new__(EngineCore)
    core.scheduler = scheduler
    core.model_executor = Mock(is_sleeping=False)
    core.mm_receiver_cache = None
    core.batch_queue = None

    assert core.pause_scheduler(mode="keep", clear_cache=False) is None
    assert scheduler.running == [running]
    assert list(scheduler.waiting) == [waiting]
    assert not scheduler.has_requests()

    core.release_kv_cache_memory()

    core.model_executor.discard.assert_called_once_with(("kv_cache",))
    assert scheduler.requests == {
        request.request_id: request for request in (running, waiting)
    }
    assert not scheduler.running
    assert list(scheduler.waiting) == [running, waiting]
    assert running.status == RequestStatus.PREEMPTED
    assert running.num_computed_tokens == 0
    assert list(running.output_token_ids) == [1]
    assert not scheduler.kv_cache_manager.get_block_ids(running.request_id)[0]
    assert scheduler.schedule().total_num_scheduled_tokens == 0

    assert core.wake_up(tags=["kv_cache"])
    resumed = scheduler.schedule()
    assert resumed.num_scheduled_tokens == {running.request_id: running.num_tokens}


@pytest.mark.parametrize("extensible", [True, False])
def test_pause_that_clears_the_kv_cache_lets_weight_updates_borrow_it(extensible):
    """Workers learn that their extensible KV cache holds nothing to keep from
    a pause that cleared it until the resume, and only then."""
    from types import SimpleNamespace

    core = object.__new__(EngineCore)
    core.scheduler = create_scheduler()
    core.mm_receiver_cache = None
    core.batch_queue = None
    core.model_executor = Mock(
        is_sleeping=False,
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(enable_extensible_kv_cache=extensible)
        ),
    )
    rpc = core.model_executor.collective_rpc

    def reclaimable_calls() -> list[bool]:
        return [
            call.kwargs["args"][0]
            for call in rpc.call_args_list
            if call.args == ("set_kv_cache_reclaimable",)
        ]

    core.pause_scheduler(mode="keep", clear_cache=False)
    core.resume_scheduler()
    assert reclaimable_calls() == []

    core.pause_scheduler(mode="abort", clear_cache=True)
    core.pause_scheduler(mode="abort", clear_cache=True)
    core.resume_scheduler()
    assert reclaimable_calls() == ([True, False] if extensible else [])


@pytest.mark.parametrize(
    "sleeping_tags",
    [{"weights", "kv_cache"}, {"weights"}, {"kv_cache"}],
    ids=["full-sleep", "weights-only", "kv-only-or-repeated-release"],
)
def test_kv_cache_release_rejects_nonresident_memory(sleeping_tags):
    """Reject release without resetting caches or discarding more memory."""
    core = object.__new__(EngineCore)
    core.scheduler = create_scheduler()
    core.scheduler.set_pause_state(PauseState.PAUSED_ALL)
    core.batch_queue = None
    core._reset_caches = Mock()
    core.model_executor = object.__new__(UniProcExecutor)
    core.model_executor.sleeping_tags = sleeping_tags.copy()
    core.model_executor.collective_rpc = Mock()

    with pytest.raises(
        RuntimeError, match="requires all executor memory to be resident"
    ):
        core.release_kv_cache_memory()

    core._reset_caches.assert_not_called()
    core.model_executor.collective_rpc.assert_not_called()
    assert core.model_executor.sleeping_tags == sleeping_tags


def test_reset_connector_cache_no_connector_is_no_op_success():
    """``reset_connector_cache`` must return True when no connector is
    configured.

    Without this, ``reset_prefix_cache(reset_connector=True)`` returns
    ``False`` on every engine that doesn't have a KV connector configured —
    even when the local prefix cache reset succeeded — and any caller that
    interprets the return value as "did the reset I asked for succeed?"
    sees a spurious failure.
    """
    scheduler = create_scheduler(enable_prefix_caching=True)
    assert scheduler.connector is None

    # No-connector reset is treated as success.
    assert scheduler.reset_connector_cache() is True

    # End-to-end: reset_prefix_cache(reset_connector=True) on an idle
    # scheduler succeeds with or without a connector.
    assert scheduler.reset_prefix_cache(reset_connector=True) is True


def test_draft_slots_budgeted_per_scheduled_request(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    (tmp_path / "config.json").write_text(
        '{"architectures": ["OPTForCausalLM"], "model_type": "opt"}'
    )
    base = create_scheduler(
        model=str(tmp_path),
        max_num_seqs=16,
        max_num_batched_tokens=20,
        num_speculative_tokens=4,
        parallel_drafting=True,
        skip_tokenizer_init=True,
    )
    # P-EAGLE needs a resolvable draft model, so switch the ngram config in place
    # and redo the config step.
    vllm_config = base.vllm_config
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    object.__setattr__(speculative_config, "method", "eagle3")
    object.__setattr__(
        speculative_config,
        "draft_model_config",
        SimpleNamespace(hf_config=SimpleNamespace(), architectures=[]),
    )
    vllm_config._set_max_num_scheduled_tokens()
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=base.kv_cache_config,
        structured_output_manager=base.structured_output_manager,
        block_size=base.block_size,
    )
    assert scheduler.max_num_scheduled_tokens == 20
    assert speculative_config.max_num_new_slots_for_drafting == 3

    for request in create_requests(num_requests=2, num_tokens=10):
        scheduler.add_request(request)

    assert scheduler.schedule().num_scheduled_tokens == {"0": 10, "1": 4}


@pytest.mark.parametrize(("method", "max_num_reqs"), [("dflash", 4), ("dspark", 5)])
def test_block_drafters_cap_requests_instead_of_draft_slots(
    tmp_path, method, max_num_reqs
):
    (tmp_path / "config.json").write_text(
        '{"architectures": ["OPTForCausalLM"], "model_type": "opt"}'
    )
    base = create_scheduler(
        model=str(tmp_path),
        max_num_batched_tokens=20,
        num_speculative_tokens=4,
        parallel_drafting=True,
        skip_tokenizer_init=True,
    )
    # DFlash/DSpark need a resolvable draft model, so switch the ngram config in
    # place and redo the config step. DSpark then has 4 lookahead tokens, not 5.
    vllm_config = base.vllm_config
    object.__setattr__(vllm_config.speculative_config, "method", method)
    object.__setattr__(
        vllm_config.speculative_config,
        "draft_model_config",
        SimpleNamespace(hf_config=SimpleNamespace(), architectures=[]),
    )
    vllm_config._set_max_num_scheduled_tokens()
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=base.kv_cache_config,
        structured_output_manager=base.structured_output_manager,
        block_size=base.block_size,
    )
    for request in create_requests(num_requests=8, num_tokens=2):
        scheduler.add_request(request)

    # Admission stops at 20 // num_lookahead_tokens, and no draft slots are
    # charged: each admitted prompt is scheduled in full.
    assert scheduler.schedule().num_scheduled_tokens == {
        str(i): 2 for i in range(max_num_reqs)
    }


# Note - these test cases mirror some of those in test_rejection_sampler.py
@pytest.mark.parametrize(
    "spec_tokens,output_tokens,expected,expected_per_req",
    [
        ([[1, 2, 3]], [[1, 2, 3, 4]], (1, 3, 3, [1, 1, 1]), [[0, 0, 0, 1]]),  # perfect
        ([[1, 2, 3]], [[1, 5]], (1, 3, 1, [1, 0, 0]), [[0, 1, 0, 0]]),  # early mismatch
        (
            [[1, 2], [3]],
            [[1, 2, 5], [3, 4]],
            (2, 3, 3, [2, 1]),
            [[0, 0, 1], [0, 1, 0]],
        ),  # multiple sequences
        ([[1]], [[1, 2]], (1, 1, 1, [1]), [[0, 1]]),  # single token sequence
        ([[]], [[5]], (0, 0, 0, [0]), [[0, 0]]),  # empty sequence -> empty accumulator
        (
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 7], [4, 8]],
            (2, 6, 3, [2, 1, 0]),
            [[0, 0, 1, 0], [0, 1, 0, 0]],
        ),  # multiple mismatches
    ],
)
def test_schedule_spec_decoding_stats(
    spec_tokens, output_tokens, expected, expected_per_req
):
    """Test scheduling behavior with speculative decoding.

    This test verifies that:
    1. Speculated tokens get scheduled correctly
    2. The aggregate SpecDecodingStats count draft and accepted tokens
    3. The per-request accumulator (enabled via per_request_spec_decode_metrics)
       buckets the same acceptance by accepted draft count (j)
    """
    num_spec_tokens = max(1, max(len(t) for t in spec_tokens))
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec_tokens,
        per_request_spec_decode_metrics="summary",
    )
    requests = create_requests(num_requests=len(spec_tokens), num_tokens=1)
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    # Schedule a decode, which will also draft speculative tokens
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert output.total_num_scheduled_tokens == len(requests)
    for i in range(len(requests)):
        req_id = requests[i].request_id
        assert output.num_scheduled_tokens[req_id] == 1
        assert req_id not in output.scheduled_spec_decode_tokens

    model_runner_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[0] for _ in range(len(requests))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    engine_core_outputs = scheduler.update_from_output(output, model_runner_output)
    draft_token_ids = DraftTokenIds(req_ids, spec_tokens)
    scheduler.update_draft_token_ids(draft_token_ids)

    for i in range(len(requests)):
        running_req = scheduler.running[i]
        # The prompt token
        assert running_req.num_computed_tokens == 1
        # The prompt token and the sampled token
        assert running_req.num_tokens == 2
        # The prompt token, the sampled token, and the speculated tokens
        assert running_req.num_tokens_with_spec == 2 + len(spec_tokens[i])

    # No draft or accepted tokens counted yet
    assert not engine_core_outputs or (
        engine_core_outputs[0].scheduler_stats.spec_decoding_stats is None
    )

    # Schedule the speculated tokens for validation
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 0
    # The sampled token and speculated tokens
    assert output.total_num_scheduled_tokens == len(requests) + sum(
        len(ids) for ids in spec_tokens
    )
    for i in range(len(requests)):
        req_id = requests[i].request_id
        assert output.num_scheduled_tokens[req_id] == 1 + len(spec_tokens[i])
        if spec_tokens[i]:
            assert len(output.scheduled_spec_decode_tokens[req_id]) == len(
                spec_tokens[i]
            )
        else:
            assert req_id not in output.scheduled_spec_decode_tokens

    model_runner_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=output_tokens,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    engine_core_outputs = scheduler.update_from_output(output, model_runner_output)

    scheduler_stats = (
        engine_core_outputs[0].scheduler_stats if engine_core_outputs else None
    )
    if expected[0] == 0:
        assert scheduler_stats is not None
        assert scheduler_stats.spec_decoding_stats is None
    else:
        assert scheduler_stats is not None
        assert scheduler_stats.spec_decoding_stats is not None
        stats = scheduler_stats.spec_decoding_stats
        assert stats.num_drafts == expected[0]
        assert stats.num_draft_tokens == expected[1]
        assert stats.num_accepted_tokens == expected[2]
        assert stats.num_accepted_tokens_per_pos == expected[3]

    # Per-request accumulator: the same acceptance, bucketed by accepted draft
    # count (j) on each request rather than summed across the batch. The
    # accumulator is created eagerly on add_request, so every request has one
    # (an empty histogram when it drafted nothing).
    for i, req_id in enumerate(req_ids):
        payload = scheduler.requests[req_id].spec_decode_metrics.to_dict()
        assert payload["acceptance_histogram"] == expected_per_req[i]
        assert payload["num_draft_tokens"] == len(spec_tokens[i])
        assert "per_step_accepted" not in payload  # summary level


def _run_spec_verify_steps(scheduler, rounds, num_invalid_per_round=None):
    """Drive prefill + one draft/verify step per round for a single request.

    ``rounds`` is a list of ``(spec_token_ids, output_token_ids)`` -- one verify
    step each. ``num_invalid_per_round`` optionally injects the grammar-invalid
    draft count structured-output decoding would set on the verify output.
    Returns the request.
    """
    [req] = create_requests(num_requests=1, num_tokens=1)
    scheduler.add_request(req)
    rid = req.request_id
    req_to_index = {rid: 0}

    def _mk_output(sampled):
        return ModelRunnerOutput(
            req_ids=[rid],
            req_id_to_index=req_to_index,
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    # Prefill: sample one token.
    scheduler.update_from_output(scheduler.schedule(), _mk_output([[0]]))
    for i, (spec, out) in enumerate(rounds):
        scheduler.update_draft_token_ids(DraftTokenIds([rid], [spec]))
        output = scheduler.schedule()
        if num_invalid_per_round is not None and num_invalid_per_round[i]:
            output.num_invalid_spec_tokens = {rid: num_invalid_per_round[i]}
        scheduler.update_from_output(output, _mk_output([out]))
    return req


def test_per_request_spec_decode_detailed_records_per_step():
    scheduler = create_scheduler(
        num_speculative_tokens=3,
        per_request_spec_decode_metrics="detailed",
    )
    # Three verify steps for one request; num_accepted = len(output) - 1, so the
    # outputs below accept 3, 1, then 0 drafts across the steps.
    req = _run_spec_verify_steps(
        scheduler,
        [
            ([1, 2, 3], [1, 2, 3, 4]),  # accept 3
            ([5, 6, 7], [5, 8]),  # accept 1
            ([9, 10, 11], [12]),  # accept 0
        ],
    )
    payload = scheduler.requests[req.request_id].spec_decode_metrics.to_dict()
    assert payload["per_step_accepted"] == [3, 1, 0]
    assert payload["per_step_drafted"] == [3, 3, 3]
    assert payload["num_spec_steps"] == 3


def test_per_request_spec_decode_subtracts_invalid_drafts():
    # Grammar-invalidated drafts (num_invalid_spec_tokens, set by structured
    # output) are excluded from the proposed count, mirroring the aggregate.
    scheduler = create_scheduler(
        num_speculative_tokens=3,
        per_request_spec_decode_metrics="summary",
    )
    # One verify step: 3 drafted, 1 grammar-invalid, output accepts 2.
    req = _run_spec_verify_steps(
        scheduler,
        [([1, 2, 3], [1, 2, 5])],
        num_invalid_per_round=[1],
    )
    payload = scheduler.requests[req.request_id].spec_decode_metrics.to_dict()
    assert payload["num_draft_tokens"] == 2  # 3 drafted - 1 invalid
    assert payload["num_accepted_draft_tokens"] == 2  # len([1,2,5]) - 1
    assert payload["acceptance_histogram"] == [0, 0, 1, 0]


def test_per_request_spec_decode_acceptance_disabled_by_default():
    scheduler = create_scheduler(num_speculative_tokens=3)
    assert scheduler.spec_decode_metrics_level == "none"
    req = _run_spec_verify_steps(scheduler, [([1, 2, 3], [1, 2, 3, 4])])
    assert scheduler.requests[req.request_id].spec_decode_metrics is None


def test_spec_decoding_stats_empty_output():
    """Test that spec decoding stats handle empty output tokens gracefully.

    This is a regression test for a bug where empty sampled_token_ids
    would cause num_accepted = len([]) - 1 = -1, leading to a
    ValueError when incrementing a Prometheus counter with a negative value.
    """
    num_spec_tokens = 3
    scheduler = create_scheduler(num_speculative_tokens=num_spec_tokens)
    requests = create_requests(num_requests=1, num_tokens=1)
    request = requests[0]
    req_id = request.request_id

    scheduler.add_request(request)

    # Initial schedule (prefill)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1

    # Complete the prefill with a sampled token
    model_runner_output = ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_runner_output)

    # Add draft tokens for speculation
    draft_token_ids = DraftTokenIds([req_id], [[1, 2, 3]])
    scheduler.update_draft_token_ids(draft_token_ids)

    # Schedule the speculated tokens for validation
    output = scheduler.schedule()
    assert req_id in output.scheduled_spec_decode_tokens
    assert len(output.scheduled_spec_decode_tokens[req_id]) == 3

    # Simulate empty output tokens (e.g., due to request abortion or error)
    # This would previously cause num_accepted = -1 and crash
    model_runner_output = ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[[]],  # Empty output tokens
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # This should not raise an error
    engine_core_outputs = scheduler.update_from_output(output, model_runner_output)

    # Spec decoding stats should be None since no tokens were generated
    scheduler_stats = (
        engine_core_outputs[0].scheduler_stats if engine_core_outputs else None
    )
    assert scheduler_stats is None or scheduler_stats.spec_decoding_stats is None


def test_no_spec_tokens_scheduled_for_prefill_chunks():
    """Test that draft tokens are ignored for prefill chunk requests.

    When a request is being prefilled in chunks (chunked prefill), draft tokens
    from `update_draft_token_ids` should be ignored until the prefill is complete.

    The bug manifests when:
    - A prefill chunk is scheduled
    - Draft tokens are provided via update_draft_token_ids
    - The next schedule has enough budget to include spec tokens

    Without the fix, spec tokens would incorrectly be scheduled with the
    remaining prefill tokens. With the fix, draft tokens are ignored for
    prefill chunks.
    """
    num_spec_tokens = 3
    # Use budget of 50, with 80 token prompt:
    # - First chunk: 50 tokens
    # - Second chunk: 30 remaining + potentially 3 spec tokens = 33
    # Without fix: num_scheduled_spec_tokens = 33 + 50 - 80 = 3 (BUG!)
    # With fix: spec_token_ids cleared, so no spec tokens scheduled
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec_tokens,
        max_num_batched_tokens=50,
        enable_chunked_prefill=True,
    )
    requests = create_requests(num_requests=1, num_tokens=80)
    req = requests[0]
    scheduler.add_request(req)

    # First schedule - prefill chunk (50 of 80 tokens)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.num_scheduled_tokens[req.request_id] == 50

    # Update from output (no sampled token since still prefilling)
    req_to_index = {req.request_id: 0}
    model_runner_output = ModelRunnerOutput(
        req_ids=[req.request_id],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_runner_output)

    # Provide draft tokens while request is still in prefill.
    # The fix ensures these are ignored for prefill chunks.
    draft_token_ids = DraftTokenIds([req.request_id], [[1, 2, 3]])
    scheduler.update_draft_token_ids(draft_token_ids)

    # Second schedule - remaining 30 tokens of prefill
    output = scheduler.schedule()
    # KEY ASSERTION: Should schedule exactly the remaining 30 prefill tokens,
    # NOT 33 (30 + 3 spec). Without the fix, this would be 33.
    assert output.num_scheduled_tokens[req.request_id] == 30, (
        f"Expected 30 tokens (remaining prefill only), "
        f"got {output.num_scheduled_tokens[req.request_id]}. "
        "Spec tokens should not be scheduled with prefill chunks."
    )
    # No spec tokens should be in the output
    assert req.request_id not in output.scheduled_spec_decode_tokens, (
        "Spec tokens should not be scheduled with prefill chunks"
    )

    # Update from output with a sampled token (prefill complete)
    model_runner_output = ModelRunnerOutput(
        req_ids=[req.request_id],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[42]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_runner_output)

    # Now provide draft tokens - should be accepted since prefill is complete
    draft_token_ids = DraftTokenIds([req.request_id], [[1, 2, 3]])
    scheduler.update_draft_token_ids(draft_token_ids)

    # spec_token_ids SHOULD be set after prefill is complete
    assert req.spec_token_ids == [1, 2, 3], (
        f"spec_token_ids should be set after prefill, got {req.spec_token_ids}"
    )

    # Third schedule - decode phase with spec tokens
    output = scheduler.schedule()
    # 1 new token + 3 spec tokens = 4
    assert output.num_scheduled_tokens[req.request_id] == 4
    assert req.request_id in output.scheduled_spec_decode_tokens
    assert len(output.scheduled_spec_decode_tokens[req.request_id]) == num_spec_tokens


def _model_output(scheduler, output, sampled):
    """Feed `sampled` (per-request list) back to the scheduler."""
    req_ids = list(output.num_scheduled_tokens.keys())
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def test_spec_decode_padding_first_decode_step():
    """A request taking its first decode step (whole prompt already computed via
    a prefix-cache hit) is padded with placeholder (-1) spec tokens so it enters
    the worker with the same 1 + num_spec_tokens shape as the other speculative
    decodes, keeping the batch uniform.
    """
    num_spec = 3
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec,
        enable_prefix_caching=True,
        block_size=16,
    )
    # Two identical 33-token prompts: 2 full blocks (32 tokens) get cached, so a
    # second identical request hits num_computed == num_prompt_tokens - 1.
    r1, r2 = create_requests(
        num_requests=2, num_tokens=33, same_prompt=True, max_tokens=16
    )

    # Drive r1 through prefill so its prompt blocks are cached, then give it real
    # drafts so it is a running speculative decode (1 + num_spec shape).
    scheduler.add_request(r1)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[r1.request_id] == 33
    _model_output(scheduler, out, [[100]])
    scheduler.update_draft_token_ids(DraftTokenIds([r1.request_id], [[1, 2, 3]]))

    # r2 arrives; its whole prompt is a prefix-cache hit -> first decode step.
    scheduler.add_request(r2)
    out = scheduler.schedule()

    # r1 verifies its real drafts.
    assert out.scheduled_spec_decode_tokens[r1.request_id] == [1, 2, 3]
    # r2 is padded to the 1 + num_spec shape with placeholder (-1) drafts.
    assert out.num_scheduled_tokens[r2.request_id] == 1 + num_spec
    assert out.scheduled_spec_decode_tokens[r2.request_id] == [-1] * num_spec


def test_spec_decode_padding_resumed_request_without_running_requests():
    """Pad a synchronously resumed request even without local running work."""
    num_spec = 3
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec,
        use_kv_connector=mock_kv(matched_tokens=32, is_async=False),
    )
    (resumed,) = create_requests(num_requests=1, num_tokens=33, max_tokens=16)

    scheduler.add_request(resumed)
    out = scheduler.schedule()

    assert out.num_scheduled_tokens[resumed.request_id] == 1 + num_spec
    assert out.scheduled_spec_decode_tokens[resumed.request_id] == [-1] * num_spec


def test_spec_decode_padding_skipped_for_diffusion():
    """Diffusion spec tokens are the fixed-size denoising canvas, not
    rejectable drafts: a first-decode-step request must keep its 1-token span
    instead of being padded to 1 + num_spec_tokens, which would overflow the
    canvas.
    """
    num_spec = 3
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec,
        enable_prefix_caching=True,
        block_size=16,
    )
    # Diffusion schedulers initialize this to 0 (model_config.is_diffusion).
    scheduler.num_sampled_tokens_per_step = 0
    r1, r2 = create_requests(
        num_requests=2, num_tokens=33, same_prompt=True, max_tokens=16
    )

    scheduler.add_request(r1)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[r1.request_id] == 33
    _model_output(scheduler, out, [[100]])
    scheduler.update_draft_token_ids(DraftTokenIds([r1.request_id], [[1, 2, 3]]))

    # r2 arrives; its whole prompt is a prefix-cache hit -> needs exactly
    # 1 token while r1 is a running speculative decode.
    scheduler.add_request(r2)
    out = scheduler.schedule()

    assert out.scheduled_spec_decode_tokens[r1.request_id] == [1, 2, 3]
    # r2 keeps its true 1-token span; no placeholder drafts are attached.
    assert out.num_scheduled_tokens[r2.request_id] == 1
    assert r2.request_id not in out.scheduled_spec_decode_tokens


@pytest.mark.parametrize("aligned_tokens", [1, 2, 3])
def test_spec_decode_padding_dropped_when_recurrent_alignment_clips(
    aligned_tokens: int,
):
    """A clipped padded request must drop its padding, not shorten it.

    The sampler derives a request's row window from its draft count, so a
    padded request that keeps fewer than 1 + num_spec query rows would select
    the preceding request's hidden states.
    """
    num_spec = 3
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec,
        enable_prefix_caching=True,
        block_size=16,
    )
    r1, r2 = create_requests(
        num_requests=2, num_tokens=33, same_prompt=True, max_tokens=16
    )

    scheduler.add_request(r1)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[r1.request_id] == 33
    _model_output(scheduler, out, [[100]])
    scheduler.update_draft_token_ids(DraftTokenIds([r1.request_id], [[1, 2, 3]]))

    # Model the recurrent align split that clips a padded 1+3 prompt tail at
    # the next cache boundary. The real QSA block size produces all three cases
    # depending on prompt length modulo four.
    scheduler.need_mamba_block_aligned_split = True
    original_split = scheduler._mamba_block_aligned_split

    def align_prompt_tail(request, num_new_tokens, *args):
        if request is r2:
            return aligned_tokens
        return original_split(request, num_new_tokens, *args)

    scheduler._mamba_block_aligned_split = align_prompt_tail
    scheduler.add_request(r2)
    out = scheduler.schedule()

    assert out.scheduled_spec_decode_tokens[r1.request_id] == [1, 2, 3]
    assert out.num_scheduled_tokens[r2.request_id] == 1
    assert r2.request_id not in out.scheduled_spec_decode_tokens


def test_spec_decode_padding_skipped_with_prefill_in_batch():
    """Padding is skipped when the batch contains a prefill chunk: the batch is
    already mixed/non-uniform, so padding a new decode request buys nothing.
    """
    num_spec = 3
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec,
        enable_prefix_caching=True,
        block_size=16,
        max_num_batched_tokens=64,
    )
    # r_warm + r_candidate share a prompt so r_candidate gets a full prefix hit.
    r_warm, r_candidate = create_requests(
        num_requests=2, num_tokens=33, same_prompt=True, max_tokens=1
    )
    # r_long has a different, long prompt that prefills over multiple chunks.
    (r_long,) = create_requests(num_requests=1, num_tokens=100, max_tokens=16)

    # Warm the prefix cache with r_warm's prompt (it finishes; blocks stay cached).
    scheduler.add_request(r_warm)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[r_warm.request_id] == 33
    _model_output(scheduler, out, [[100]])
    assert r_warm.request_id in scheduler.finished_req_ids

    # Start r_long; after one chunk it remains a prefill chunk in the running queue.
    scheduler.add_request(r_long)
    out = scheduler.schedule()
    _model_output(scheduler, out, [[]])  # still prefilling, no sampled token
    assert r_long.is_prefill_chunk

    # r_candidate arrives (prefix-cache hit -> first decode step) alongside the
    # in-flight prefill chunk.
    scheduler.add_request(r_candidate)
    out = scheduler.schedule()

    # The batch has a prefill chunk, so r_candidate is NOT padded.
    assert r_long.request_id in out.num_scheduled_tokens
    assert out.num_scheduled_tokens[r_candidate.request_id] == 1
    assert r_candidate.request_id not in out.scheduled_spec_decode_tokens


def test_scheduler_stats_waiting_queues():
    """Test that scheduler stats correctly report capacity-bound and blocked
    waiting requests."""
    # Create scheduler with limited capacity so we can have waiting requests
    scheduler = create_scheduler(max_num_batched_tokens=100)

    # Create requests: some will be scheduled, some will wait on capacity,
    # and some will be blocked by constraints
    all_requests = create_requests(num_requests=5, num_tokens=50)

    # Add 3 requests - only 2 can be scheduled (2 * 50 = 100 tokens)
    # The 3rd will remain in waiting queue (capacity constraint)
    for request in all_requests[:3]:
        scheduler.add_request(request)

    # Add 2 more that are blocked on grammar compilation
    for request in all_requests[3:]:
        request.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
        scheduler.add_request(request)

    # Schedule - this will schedule 2 requests, leaving 1 in waiting
    output = scheduler.schedule()

    # Verify: 2 scheduled, 1 still waiting on capacity, 2 blocked by constraints
    assert len(output.scheduled_new_reqs) == 2
    assert len(scheduler.waiting) == 3
    assert len(scheduler.deferred_waiting) == 2

    # Call update_from_output() to get frontend-facing stat
    scheduled_req_ids = list(output.num_scheduled_tokens.keys())
    model_runner_output = ModelRunnerOutput(
        req_ids=scheduled_req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(scheduled_req_ids)},
        sampled_token_ids=[[1]] * len(scheduled_req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    engine_core_outputs = scheduler.update_from_output(output, model_runner_output)
    assert engine_core_outputs and len(engine_core_outputs) > 0
    stats = engine_core_outputs[0].scheduler_stats
    assert stats is not None

    # Verify stats match queue lengths after scheduling
    assert stats.num_running_reqs == 2  # 2 were scheduled
    assert stats.num_waiting_reqs == 1  # 1 waiting on capacity
    assert stats.num_skipped_waiting_reqs == 2  # 2 blocked by constraints


def _assert_right_scheduler_output(
    output: SchedulerOutput,
    num_requests: int,
    expected_num_scheduled_tokens: int,
):
    """Check if SchedulerOutput is correct after remote KV cache hit."""
    # We should inject the kv_connector_metadata.
    assert len(output.kv_connector_metadata.requests) == num_requests

    # Only num_tokens - matched_num_new_tokens should be scheduled.
    for _, num_scheduled_tokens in output.num_scheduled_tokens.items():
        assert num_scheduled_tokens == expected_num_scheduled_tokens


def _assert_right_kv_cache_manager(
    scheduler: Scheduler,
    requests: list[Request],
    num_tokens: int,
    block_size: int,
    num_requests: int,
    num_total_blocks: int,
):
    """Check whether KVCacheManager is correct after allocate."""
    # Make sure the request stats are right.
    EXPECTED_TOTAL_BLOCKS = num_tokens // block_size
    for req in requests:
        blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
            0
        ].req_to_blocks[req.request_id]
        hashes = req.block_hashes
        assert (
            scheduler.kv_cache_manager.coordinator.single_type_managers[
                0
            ].num_cached_block[req.request_id]
            == EXPECTED_TOTAL_BLOCKS
        )
        assert len(blocks) == EXPECTED_TOTAL_BLOCKS
        assert len(hashes) == EXPECTED_TOTAL_BLOCKS

    # Make sure we actually touched all the blocks.
    BLOCKS_PER_REQ = num_tokens / block_size
    assert (
        scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        == num_total_blocks - num_requests * BLOCKS_PER_REQ
    )


def _step_until_done(
    scheduler: Scheduler,
    output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
):
    """Loop over schedule(), update_from_output() until finished."""
    all_finished = False
    _ = scheduler.update_from_output(output, model_runner_output)
    while not all_finished:
        # Schedule + a few iterations until stopping.
        output = scheduler.schedule()
        assert len(scheduler.running)
        for _, num_scheduled_tokens in output.num_scheduled_tokens.items():
            # We should be in the decode phase now.
            assert num_scheduled_tokens == 1
        if scheduler.connector is not None:
            assert len(output.kv_connector_metadata.requests) == 0
        if scheduler.ec_connector is not None:
            assert len(output.ec_connector_metadata.mm_datas) == 0
        ecos = scheduler.update_from_output(output, model_runner_output)[0]
        all_done = True
        for eco in ecos.outputs:
            if eco.finish_reason is None:
                all_done = False
        all_finished = all_done


def _num_waiting_requests(scheduler: Scheduler) -> int:
    return len(scheduler.waiting) + len(scheduler.kv_holding_waiting)


def _step_until_kv_transfer_finished(scheduler: Scheduler, req_ids: list[str]):
    """Cycle requests through a KV transfer cycle."""
    # Requests should first transition to WAITING_FOR_REMOTE_KVS
    output = scheduler.schedule()
    assert _num_waiting_requests(scheduler) == len(req_ids)
    assert len(scheduler.running) == 0
    assert len(output.scheduled_new_reqs) == 0
    for req in scheduler.requests.values():
        assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    # No model execution yet
    EMPTY_OUTPUT = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    initial_ecos = scheduler.update_from_output(output, EMPTY_OUTPUT)

    # Simulate KV transfer completion using KVConnectorOutput.finished_recving
    output = scheduler.schedule()
    assert _num_waiting_requests(scheduler) == len(req_ids)
    assert len(scheduler.running) == 0

    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=KVConnectorOutput(finished_recving=req_ids),
    )
    scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    for req_id in req_ids:
        assert req_id in scheduler.finished_recving_kv_req_ids

    return initial_ecos


def _runner_output(
    req_ids: list[str], finished_recving: set[str] | None = None
) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[1] for _ in req_ids],
        kv_connector_output=KVConnectorOutput(finished_recving=finished_recving),
    )


@pytest.mark.parametrize(
    "load_done_before_pause", [False, True], ids=["load-in-flight", "load-done"]
)
def test_pause_new_drains_async_kv_loads(load_done_before_pause: bool):
    """A PAUSED_NEW (pause mode="wait") drain must finish requests that hold KV
    blocks for an async load, keep blockless queued requests queued, and leave
    no blocks behind, so the post-drain cache reset succeeds."""
    block_size = 16
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(matched_tokens=2 * block_size, is_async=True),
        block_size=block_size,
    )
    loading, queued = create_requests(
        num_requests=2,
        num_tokens=3 * block_size,
        max_tokens=2,
        block_size=block_size,
        req_ids=["loading", "queued"],
    )
    scheduler.add_request(loading)
    output = scheduler.schedule()
    assert loading.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    pending_recv = {"loading"}
    if load_done_before_pause:
        scheduler.update_from_output(output, _runner_output([], pending_recv))
        pending_recv = None
    else:
        scheduler.update_from_output(output, _runner_output([]))

    scheduler.set_pause_state(PauseState.PAUSED_NEW)
    scheduler.add_request(queued)

    num_steps = 0
    while scheduler.has_requests():
        num_steps += 1
        assert num_steps <= 4, "pause drain did not finish"
        output = scheduler.schedule()
        assert set(output.num_scheduled_tokens) <= {"loading"}
        scheduler.update_from_output(
            output, _runner_output(list(output.num_scheduled_tokens), pending_recv)
        )
        pending_recv = None

    # What pause runs once drained; it raises if the load still holds blocks.
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert loading.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert loading.request_id not in scheduler.requests
    assert list(scheduler.waiting) == [queued]
    assert queued.status == RequestStatus.WAITING

    scheduler.set_pause_state(PauseState.UNPAUSED)
    scheduler.schedule()
    assert queued.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_pause_new_ignores_streaming_session_waiting_for_input():
    """A session idle in WAITING_FOR_STREAMING_REQ cannot progress, so it must
    not keep a PAUSED_NEW drain from completing."""
    stop_token = 7
    scheduler = create_scheduler()
    (session,) = create_requests(
        num_requests=1, num_tokens=4, stop_token_ids=[stop_token], req_ids=["s"]
    )
    session.resumable = True
    scheduler.add_request(session)
    output = scheduler.schedule()
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=["s"], req_id_to_index={"s": 0}, sampled_token_ids=[[stop_token]]
        ),
    )
    assert session in scheduler.kv_holding_waiting

    scheduler.set_pause_state(PauseState.PAUSED_NEW)
    assert scheduler.get_num_unfinished_requests() == 0


@pytest.mark.parametrize(
    ("load_modes", "expected_has_sync_loads"),
    [
        ((True,), False),
        ((False,), True),
        ((True, False), True),
    ],
)
@pytest.mark.skip_global_cleanup
def test_has_sync_kv_loads(
    load_modes: tuple[bool, ...],
    expected_has_sync_loads: bool,
    tmp_path,
):
    (tmp_path / "config.json").write_text(
        '{"architectures": ["OPTForCausalLM"], "model_type": "opt"}'
    )
    block_size = 16
    scheduler = create_scheduler(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        use_kv_connector=mock_kv(matched_tokens=block_size, is_async=False),
        block_size=block_size,
    )
    requests = create_requests(
        num_requests=len(load_modes),
        num_tokens=block_size * 2,
        block_size=block_size,
    )
    for request in requests:
        scheduler.add_request(request)

    scheduler.connector.get_num_new_matched_tokens = Mock(
        side_effect=[(block_size, is_async) for is_async in load_modes]
    )

    output = scheduler.schedule()

    assert output.has_sync_kv_loads is expected_has_sync_loads


@pytest.mark.parametrize("suffix_scoring", [False, True])
def test_kv_connector_honors_skip_reading_prefix_cache(suffix_scoring):
    """External hits cannot skip rows required by prompt scoring."""
    BLOCK_SIZE = 16
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(matched_tokens=BLOCK_SIZE * 2, is_async=False),
        block_size=BLOCK_SIZE,
    )
    scoring_params = (
        SamplingParams(
            prompt_logprob_token_ids=[[1]] * (BLOCK_SIZE * 3 - 1),
            prompt_logprob_start=BLOCK_SIZE,
            skip_reading_prefix_cache=False,
        )
        if suffix_scoring
        else SamplingParams(prompt_logprobs=1)
    )
    (plain,) = create_requests(1, num_tokens=BLOCK_SIZE * 4, req_ids=["plain"])
    (scoring,) = create_requests(
        1,
        num_tokens=BLOCK_SIZE * 4,
        req_ids=["scoring"],
        sampling_params=scoring_params,
    )
    admission = []
    scheduler.connector.on_new_request = lambda req: admission.append(
        req.skip_reading_external_kv
    )
    scheduler.add_request(plain)
    scheduler.add_request(scoring)
    assert admission == [False, True]

    output = scheduler.schedule()
    assert output.num_scheduled_tokens[plain.request_id] == BLOCK_SIZE * 2
    # No external hit; suffix scoring still reuses the local blocks plain just
    # loaded, up to prompt_logprob_start.
    expected = BLOCK_SIZE * 3 if suffix_scoring else BLOCK_SIZE * 4
    assert output.num_scheduled_tokens[scoring.request_id] == expected


@pytest.mark.parametrize("is_async", [False, True])
def test_kv_connector_records_external_cache_hit_sources(monkeypatch, is_async):
    block_size = 16
    num_matched_tokens = 2 * block_size
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(
            matched_tokens=num_matched_tokens,
            is_async=is_async,
        ),
        block_size=block_size,
    )
    request = create_requests(
        num_requests=1,
        num_tokens=2 * num_matched_tokens,
        block_size=block_size,
    )[0]
    assert scheduler.connector is not None
    get_sources = Mock(
        return_value={CacheHitSource.P2P: block_size, CacheHitSource.HOST: block_size}
    )
    update_state = Mock(wraps=scheduler.connector.update_state_after_alloc)
    calls = Mock()
    calls.attach_mock(update_state, "allocated")
    calls.attach_mock(get_sources, "sources")
    monkeypatch.setattr(scheduler.connector, "update_state_after_alloc", update_state)
    monkeypatch.setattr(
        scheduler.connector, "get_external_cache_hit_sources", get_sources
    )

    scheduler.add_request(request)
    # A failed allocation is not an admission: nothing is attributed.
    with patch.object(scheduler.kv_cache_manager, "allocate_slots", return_value=None):
        scheduler.schedule()
    get_sources.assert_not_called()
    update_state.assert_not_called()

    scheduler.schedule()
    # Attribution runs once, after the connector has built its load plan.
    get_sources.assert_called_once_with(request, num_matched_tokens)
    assert [c[0] for c in calls.mock_calls] == ["allocated", "sources"]
    connector_stats = scheduler.connector_prefix_cache_stats
    assert connector_stats is not None
    assert connector_stats.hits == num_matched_tokens
    assert connector_stats.hits_by_source == {
        CacheHitSource.P2P: block_size,
        CacheHitSource.HOST: block_size,
    }

    if is_async:
        # Re-admission after the async load completes is not a new hit.
        scheduler.make_stats()
        _step_until_kv_transfer_finished(scheduler, [request.request_id])
        get_sources.assert_called_once()
        connector_stats = scheduler.connector_prefix_cache_stats
        assert connector_stats is not None
        assert connector_stats.hits_by_source == {}


@pytest.mark.parametrize("is_async", [False, True])
def test_kv_connector_basic(is_async: bool):
    """Test whether Scheduler with KVConnector schedules tokens, allocates
    memory, and cleans up requests as expected under normal operation.
    """
    # Setup Scheduler.
    BLOCK_SIZE = 16
    NUM_MATCHED_NEW_TOKENS = BLOCK_SIZE * 2
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(
            matched_tokens=NUM_MATCHED_NEW_TOKENS, is_async=is_async
        ),
        block_size=BLOCK_SIZE,
    )
    NUM_TOTAL_BLOCKS = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()

    ######################################################
    # FIRST SET OF REQUESTS - External Hit Only
    NUM_REQUESTS = 2
    NUM_TOKENS = NUM_MATCHED_NEW_TOKENS * 2
    MAX_TOKENS = 3
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
        max_tokens=MAX_TOKENS,
        block_size=BLOCK_SIZE,
    )
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    if is_async:
        _step_until_kv_transfer_finished(scheduler, req_ids)

    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[1000]] * len(req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # Ensure ScheduleOutput is correct.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output=output,
        num_requests=NUM_REQUESTS,
        # Just the incremental tokens should be scheduled.
        expected_num_scheduled_tokens=NUM_TOKENS - NUM_MATCHED_NEW_TOKENS,
    )

    # Ensure KVCacheManager is correct.
    _assert_right_kv_cache_manager(
        scheduler, requests, NUM_TOKENS, BLOCK_SIZE, NUM_REQUESTS, NUM_TOTAL_BLOCKS
    )

    # Continue Generation until done.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    _ = scheduler.schedule()
    # Confirm we clean up the memory properly.
    assert (
        scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_TOTAL_BLOCKS
    )

    ######################################################
    # SECOND SET OF REQUESTS - Local And External Hit
    NUM_TOKENS_PREFIX = NUM_TOKENS
    # We will get a local prefix cache hit for the first
    # NUM_TOKENS_PREFIX tokens since they are used above.
    NUM_TOKENS = NUM_TOKENS_PREFIX * 2
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
        max_tokens=MAX_TOKENS,
        block_size=BLOCK_SIZE,
    )
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    if is_async:
        _step_until_kv_transfer_finished(scheduler, req_ids)

    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[1000]] * len(req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # We should get a local cache hit of NUM_TOKENS_PREFIX and
    # a remote KV cache hit of NUM_MATCHED_NEW_TOKENS.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output=output,
        num_requests=NUM_REQUESTS,
        # Just the incremental tokens after local + remote cache hit.
        expected_num_scheduled_tokens=(
            NUM_TOKENS - NUM_TOKENS_PREFIX - NUM_MATCHED_NEW_TOKENS
        ),
    )

    # Ensure KVCacheManager is correct.
    _assert_right_kv_cache_manager(
        scheduler, requests, NUM_TOKENS, BLOCK_SIZE, NUM_REQUESTS, NUM_TOTAL_BLOCKS
    )

    # Continue Generation until done.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    _ = scheduler.schedule()
    # Confirm we clean up the memory properly.
    assert (
        scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_TOTAL_BLOCKS
    )


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("local_cache_hits", [False, True])
def test_external_prefix_cache_metrics(is_async: bool, local_cache_hits: bool):
    """Verify connector prefix cache metrics are updated
    correctly when the scheduler processes requests with KV connector hits.
    """
    BLOCK_SIZE = 16
    if local_cache_hits:
        NUM_MATCHED_NEW_TOKENS = BLOCK_SIZE * 2  # 32 tokens
        NUM_LOCAL_HITS = NUM_MATCHED_NEW_TOKENS * 2  # 64 tokens
        NUM_REQUESTS = 1
        NUM_TOKENS = NUM_LOCAL_HITS * 2  # 128 tokens
    else:
        NUM_MATCHED_NEW_TOKENS = 4
        NUM_LOCAL_HITS = 0
        NUM_REQUESTS = 2
        NUM_TOKENS = 8  # 8 tokens

    # Setup Scheduler.
    scheduler = create_scheduler(
        enable_prefix_caching=local_cache_hits,
        use_kv_connector=mock_kv(
            matched_tokens=NUM_MATCHED_NEW_TOKENS, is_async=is_async
        ),
        block_size=BLOCK_SIZE,
    )

    if local_cache_hits:
        # First, establish local cache by running a request to completion
        requests = create_requests(
            num_requests=1,
            num_tokens=NUM_LOCAL_HITS,
            max_tokens=2,
            block_size=BLOCK_SIZE,
        )
        req_ids = []
        req_to_index = {}
        for i, request in enumerate(requests):
            scheduler.add_request(request)
            req_ids.append(request.request_id)
            req_to_index[request.request_id] = i

        if is_async:
            _step_until_kv_transfer_finished(scheduler, req_ids)

        # Run first request to completion to establish local cache
        output = scheduler.schedule()
        MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_to_index,
            sampled_token_ids=[[1000]] * len(req_ids),
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
        _ = scheduler.schedule()

    # --- Prepare test requests ---
    MAX_TOKENS = 2
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
        max_tokens=MAX_TOKENS,
        block_size=BLOCK_SIZE,
    )
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    initial_ecos = None
    if is_async:
        initial_ecos = _step_until_kv_transfer_finished(scheduler, req_ids)

    # --- Trigger scheduling and simulate model output ---
    output = scheduler.schedule()
    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=[r.request_id for r in requests],
        req_id_to_index={r.request_id: i for i, r in enumerate(requests)},
        sampled_token_ids=[[1000]] * NUM_REQUESTS,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # Update scheduler stats
    ecos = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)

    # --- Assertions ---
    assert ecos is not None and len(ecos) > 0
    assert ecos[0].scheduler_stats is not None

    if local_cache_hits:
        # For async, local cache stats come from the first step
        if initial_ecos:
            local_stats = initial_ecos[0].scheduler_stats.prefix_cache_stats
        else:
            local_stats = ecos[0].scheduler_stats.prefix_cache_stats
        assert local_stats is not None
        assert local_stats.queries == NUM_TOKENS * NUM_REQUESTS
        assert local_stats.hits == NUM_LOCAL_HITS * NUM_REQUESTS

    if initial_ecos:
        external_stats = initial_ecos[0].scheduler_stats.connector_prefix_cache_stats
    else:
        external_stats = ecos[0].scheduler_stats.connector_prefix_cache_stats
    assert external_stats is not None

    assert external_stats.queries == (NUM_TOKENS - NUM_LOCAL_HITS) * NUM_REQUESTS
    assert external_stats.hits == NUM_MATCHED_NEW_TOKENS * NUM_REQUESTS
    assert external_stats.requests == NUM_REQUESTS
    assert external_stats.preempted_requests == 0


@pytest.mark.parametrize(
    "use_ec_connector, ec_role", [(False, None), (True, "ec_consumer")]
)
def test_kv_connector_unable_to_allocate(use_ec_connector, ec_role):
    """Test whether scheduler with KVConnector is able to handle
    unable to allocate (run out of blocks in allocate_slots().
    """
    # Setup Scheduler With Mock External Cache Hit.
    BLOCK_SIZE = 4
    NUM_BLOCKS = 10
    NUM_MATCHED_NEW_TOKENS = BLOCK_SIZE * 2
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(matched_tokens=NUM_MATCHED_NEW_TOKENS, is_async=False),
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
        # encoder connector should not affect test results
        use_ec_connector=use_ec_connector,
        ec_role=ec_role,
    )

    # Create two requests. The second request will not be able to
    # allocate slots because it will not have enough blocks.
    NUM_REQUESTS = 2
    NUM_TOKENS = (NUM_BLOCKS // 2 + 1) * BLOCK_SIZE
    MAX_TOKENS = 2
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
        max_tokens=MAX_TOKENS,
        block_size=BLOCK_SIZE,
    )
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[1000]] * len(req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # Just one request should be running.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        num_requests=1,
        expected_num_scheduled_tokens=NUM_TOKENS - NUM_MATCHED_NEW_TOKENS,
    )
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1

    # All memory should be freed, with one request waiting.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 1

    # Just one request should be running.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        num_requests=1,
        expected_num_scheduled_tokens=NUM_TOKENS - NUM_MATCHED_NEW_TOKENS,
    )
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0

    # All memory should be freed, with no requests waiting / running.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 0


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "use_ec_connector, ec_role", [(False, None), (True, "ec_consumer")]
)
def test_kv_connector_handles_preemption(
    is_async, use_ec_connector, ec_role, use_v2_model_runner
):
    """Test whether scheduler with KVConnector is able to handle
    unable to allocate (run out of blocks in allocate_slots().
    """
    # Setup Scheduler With Mock External Cache Hit.
    BLOCK_SIZE = 2
    # NOTE: there is 1 null block, so this is 6 blocks.
    NUM_BLOCKS = 7
    NUM_MATCHED_NEW_TOKENS = BLOCK_SIZE
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(
            matched_tokens=NUM_MATCHED_NEW_TOKENS, is_async=is_async
        ),
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
        # encoder connector should not affect test results
        use_ec_connector=use_ec_connector,
        ec_role=ec_role,
        use_v2_model_runner=use_v2_model_runner,
    )

    # Create two requests.
    # Both can be scheduled at first, but the second request
    # will be preempted and re-scheduled.
    NUM_REQUESTS = 2
    NUM_TOKENS = BLOCK_SIZE * 2 + 1
    MAX_TOKENS = BLOCK_SIZE * 2
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
        max_tokens=MAX_TOKENS,
        block_size=BLOCK_SIZE,
    )
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[1000]] * len(req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # All can be scheduled - 1st token.
    output = scheduler.schedule()
    if is_async:
        assert _num_waiting_requests(scheduler) == 2
        assert scheduler.running == []
        _step_until_kv_transfer_finished(scheduler, req_ids)
        output = scheduler.schedule()

    _assert_right_scheduler_output(
        output,
        # 2 remote kv cache hits.
        num_requests=2,
        expected_num_scheduled_tokens=NUM_TOKENS - NUM_MATCHED_NEW_TOKENS,
    )
    assert len(scheduler.running) == 2
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)

    # All can be scheduled - 2nd token.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        # no connector_metadata
        num_requests=0,
        expected_num_scheduled_tokens=1,
    )
    assert len(scheduler.running) == 2
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)

    # This will generate a new block and cause a preemption - 3rd token.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        # no connector_metadata
        num_requests=0,
        expected_num_scheduled_tokens=1,
    )
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1

    # Only 1 can be scheduled - 4th (and last token).
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        # no connector_metadata
        num_requests=0,
        expected_num_scheduled_tokens=1,
    )
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 1
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    assert len(scheduler.running) == 0
    # All memory should be freed since nothing is running.
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1

    # Restarts the preempted request - generate 3rd token.
    # This will have a local and remote cache hit.
    output = scheduler.schedule()
    if is_async:
        waiting_req_ids = [
            req.request_id
            for req in scheduler.kv_holding_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        ]
        assert len(waiting_req_ids) == 1
        _step_until_kv_transfer_finished(scheduler, waiting_req_ids)
        output = scheduler.schedule()

    _assert_right_scheduler_output(
        output,
        # 1 remote kv_cache hit!
        num_requests=1,
        # Only 1 block was preempted and there is a single
        # remote hit. So only single new token is scheduled.
        expected_num_scheduled_tokens=1,
    )
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0
    if use_v2_model_runner:
        # V2 emits a resumed (previously preempted) request as a
        # NewRequestData rather than a cached request.
        assert output.scheduled_cached_reqs.num_reqs == 0
        assert len(output.scheduled_new_reqs) == 1
    else:
        assert output.scheduled_cached_reqs.num_reqs == 1
        assert output.scheduled_new_reqs == []
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0

    # Only 1 can be scheduled - 4th (and last token).
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        # no connector_metadata
        num_requests=0,
        expected_num_scheduled_tokens=1,
    )
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert output.scheduled_new_reqs == []
    assert len(scheduler.running) == 1
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    assert len(scheduler.running) == 0
    # All memory should be freed since nothing is running.
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1


def make_output(scheduler: Scheduler):
    return ModelRunnerOutput(
        req_ids=[req.request_id for req in scheduler.running],
        req_id_to_index={req.request_id: i for i, req in enumerate(scheduler.running)},
        sampled_token_ids=[[1000]] * len(scheduler.running),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def assert_scheduler_empty(scheduler: Scheduler):
    """Confirm the scheduler is "empty" - i.e. no leaks."""
    # Scheduler Metadata.
    assert len(scheduler.requests) == 0
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 0
    assert len(scheduler.finished_req_ids) == 0

    # EncoderCacheManager.
    assert len(scheduler.encoder_cache_manager.freed) == 0
    assert len(scheduler.encoder_cache_manager.cached) == 0

    # KVCache Manager.
    assert (
        len(
            scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks
        )
        == 0
    )
    assert (
        len(
            scheduler.kv_cache_manager.coordinator.single_type_managers[
                0
            ].num_cached_block
        )
        == 0
    )
    num_free_blocks = (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
    )
    assert num_free_blocks == (scheduler.kv_cache_manager.block_pool.num_gpu_blocks - 1)

    # NOTE(rob): just the ref count on blocks will be 0. The hash
    # value, etc will remain since we lazily evict for prefix cache.
    for block in scheduler.kv_cache_manager.block_pool.blocks:
        assert block.ref_cnt == 0
        # assert block.block_hash is None
    # assert (
    #     len(scheduler.kv_cache_manager.block_pool.cached_block_hash_to_block
    #           ) == 0)


def test_memory_leak():
    """Test that we do not have a memory leak."""
    scheduler = create_scheduler(enable_prefix_caching=True)

    NUM_REQUESTS = 5
    NUM_TOKENS = 10
    MAX_TOKENS = 10
    requests = create_requests(
        num_requests=NUM_REQUESTS, num_tokens=NUM_TOKENS, max_tokens=MAX_TOKENS
    )

    # Add each request.
    for request in requests:
        scheduler.add_request(request)
        scheduler_output = scheduler.schedule()
        model_runner_output = make_output(scheduler)
        scheduler.update_from_output(scheduler_output, model_runner_output)

    # Iterate until done.
    while True:
        scheduler_output = scheduler.schedule()
        if len(scheduler.running) == 0:
            break
        model_runner_output = make_output(scheduler)
        scheduler.update_from_output(scheduler_output, model_runner_output)

    # Confirm no memory leak.
    assert_scheduler_empty(scheduler)


def create_scheduler_with_priority(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    enable_prefix_caching: bool = False,
    long_prefill_token_threshold: int = 0,
    disable_chunked_mm_input: bool = False,
    use_kv_connector: bool = False,
    num_blocks: int = 10000,
    block_size: int = 16,
    max_model_len: int | None = None,
    num_speculative_tokens: int | None = None,
    use_ec_connector: bool = False,
    ec_role: str | None = None,
    use_v2_model_runner: bool | None = None,
) -> Scheduler:
    """Create scheduler with priority policy enabled.

    Args:
      model: model under test
      max_num_seqs: max sequences to schedule
      max_num_batch_tokens: max num tokens to batch
      enable_prefix_caching: optionally force APC config
                             (True/False) or use default
                             (False)

    Returns:
      {class}`Scheduler` instance with priority scheduling

    """
    model_config = ModelConfig(
        model=model,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    if max_model_len is None:
        max_model_len = max_num_batched_tokens
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        long_prefill_token_threshold=long_prefill_token_threshold,
        disable_chunked_mm_input=disable_chunked_mm_input,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
        policy="priority",  # Enable priority scheduling
        # Ensure admission/preemption mechanics are deterministic
        watermark=0.0,
    )
    # Cache config, optionally force APC
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    kv_transfer_config = (
        KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": "local_storage"},
        )
        if use_kv_connector
        else None
    )

    speculative_config: SpeculativeConfig | None = None
    if num_speculative_tokens is not None:
        speculative_config = SpeculativeConfig(
            model="ngram", num_speculative_tokens=num_speculative_tokens
        )

    ec_transfer_config = (
        ECTransferConfig(
            ec_connector="ECExampleConnector",
            ec_role=ec_role,
            ec_connector_extra_config={"shared_storage_path": "/tmp/ec_test"},
        )
        if use_ec_connector
        else None
    )

    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        speculative_config=speculative_config,
        ec_transfer_config=ec_transfer_config,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,  # A large number of blocks to hold all requests
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
        hash_block_size=block_size,
    )
    if use_v2_model_runner is None:
        use_v2_model_runner = bool(envs.VLLM_USE_V2_MODEL_RUNNER)
    scheduler.use_v2_model_runner = use_v2_model_runner
    return scheduler


_none_hash_initialized = False


def create_requests_with_priority(
    num_requests: int,
    priorities: list[int],
    arrival_times: list[float] | None = None,
    num_tokens: int = 10,
    mm_hashes_list: list[list[str]] | None = None,
    mm_positions: list[list[PlaceholderRange]] | None = None,
    max_tokens: int = 16,
    stop_token_ids: list[int] | None = None,
    prompt_logprobs: int | None = None,
    starting_idx: int = 0,
    same_prompt: bool = False,
    block_size: int = 16,
    req_ids: list[str] | None = None,
):
    """Create requests with specified priorities and arrival times."""
    assert len(priorities) == num_requests
    if arrival_times is not None:
        assert len(arrival_times) == num_requests
    else:
        arrival_times = [float(i) for i in range(num_requests)]

    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True

    block_hasher = get_request_block_hasher(block_size, sha256)
    sampling_params = SamplingParams(
        ignore_eos=False,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
        prompt_logprobs=prompt_logprobs,
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    requests = []

    if mm_hashes_list is not None:
        # NOTE: allow manual input; some mm items can have the same identifier
        # no. of mm_hashes and mm_positions for each request should be identical
        assert mm_positions is not None, (
            "mm_positions must be provided when mm_hashes_list is provided"
        )
        assert len(mm_hashes_list) == len(mm_positions) == num_requests
        assert [len(h) for h in mm_hashes_list] == [len(p) for p in mm_positions]

        # Since same identifier would imply they are identical encoder output
        # Verify mm items with identical identifier are having mm_position.length
        seen_hashes: dict[str, int] = {}

    if req_ids:
        assert len(req_ids) == num_requests
    else:
        req_ids = [f"{i + starting_idx}" for i in range(num_requests)]

    for i in range(num_requests):
        mm_features = []

        for j, position in enumerate(
            mm_positions[i] if mm_positions is not None else []
        ):
            if mm_hashes_list is not None:
                identifier = mm_hashes_list[i][j]

                # Verify if position length is identical
                position_length = position.length
                if identifier in seen_hashes:
                    assert seen_hashes[identifier] == position_length, (
                        f"mm_hash '{identifier}' has inconsistent position lengths: "
                        f"previously {seen_hashes[identifier]}, now {position_length} "
                        f"at request {i}, position {j}"
                    )
                else:
                    seen_hashes[identifier] = position_length
            else:
                # Unique dummy hash for each mm item
                identifier = f"hash{i}_{j}"
            mm_feature = MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                mm_position=position,
                identifier=identifier,
                modality="image",
            )
            mm_features.append(mm_feature)

        prompt_token_ids = (
            [starting_idx] * num_tokens
            if same_prompt
            else [i + starting_idx] * num_tokens
        )
        request = Request(
            request_id=req_ids[i],
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            pooling_params=None,
            mm_features=mm_features if mm_features else None,
            arrival_time=arrival_times[i],
            priority=priorities[i],
            block_hasher=block_hasher,
        )
        requests.append(request)
    return requests


def test_priority_scheduling_basic_ordering():
    """Test that requests are scheduled in priority order
    (lower value = higher priority)."""
    scheduler = create_scheduler_with_priority()

    # Create requests with different priorities
    # Priority 0 (highest), 1, 2 (lowest)
    priorities = [2, 0, 1]  # Add in non-priority order
    arrival_times = [1.0, 2.0, 3.0]  # All different arrival times
    requests = create_requests_with_priority(
        num_requests=3, priorities=priorities, arrival_times=arrival_times
    )

    # Add requests in non-priority order
    for request in requests:
        scheduler.add_request(request)

    # Schedule and verify priority order
    output = scheduler.schedule()

    # Should schedule all requests since they fit in budget
    assert len(output.scheduled_new_reqs) == 3

    # Verify they are scheduled in priority order:
    # req_1 (priority 0), req_2 (priority 1), req_0 (priority 2)
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled_req_ids == ["1", "2", "0"]


def test_priority_scheduling_arrival_time_tiebreaker():
    """Test that arrival time is used
    as tiebreaker when priorities are equal."""
    scheduler = create_scheduler_with_priority()

    # Create requests with same priority but different arrival times
    priorities = [1, 1, 1]  # All same priority
    arrival_times = [3.0, 1.0, 2.0]  # Different arrival times
    requests = create_requests_with_priority(
        num_requests=3, priorities=priorities, arrival_times=arrival_times
    )

    # Add requests in non-arrival order
    for request in requests:
        scheduler.add_request(request)

    # Schedule and verify arrival time order
    output = scheduler.schedule()

    # Should schedule all requests since they fit in budget
    assert len(output.scheduled_new_reqs) == 3

    # Verify they are scheduled in arrival time order:
    # req_1 (1.0), req_2 (2.0), req_0 (3.0)
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled_req_ids == ["1", "2", "0"]


def test_priority_scheduling_mixed_priority_and_arrival():
    """Test priority scheduling with mixed priorities and arrival times."""
    scheduler = create_scheduler_with_priority()

    # Create requests with mixed priorities and arrival times
    priorities = [2, 1, 1, 0]  # Mixed priorities
    arrival_times = [1.0, 3.0, 2.0, 4.0]  # Mixed arrival times
    requests = create_requests_with_priority(
        num_requests=4, priorities=priorities, arrival_times=arrival_times
    )

    # Add requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule and verify order
    output = scheduler.schedule()

    # Should schedule all requests since they fit in budget
    assert len(output.scheduled_new_reqs) == 4

    # Expected order:
    # 1. req_3 (priority 0, arrival 4.0)
    # 2. req_2 (priority 1, arrival 2.0) - earlier arrival than req_1
    # 3. req_1 (priority 1, arrival 3.0)
    # 4. req_0 (priority 2, arrival 1.0)
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled_req_ids == ["3", "2", "1", "0"]


def test_priority_scheduling_preemption():
    """Test that under KV block pressure the scheduler preempts the
    lowest-priority *running* request, not the highest-priority one.

    A low-priority request starts running first. Then a high-priority
    request arrives and is admitted to running.  When block pressure
    builds, the scheduler preempts the low-priority running request
    while keeping the high-priority one.

    Block math
    ----------
    block_size = 16, num_blocks = 6 (1 null → 5 usable).

    Phase 1: lo1 (priority 5, 32 tokens) → 2 blocks.  3 free.
             Decode → lo1 has 33 tokens (needs 3rd block on next schedule).
    Phase 2: hi1 (priority 0, 32 tokens) arrives.
             schedule() allocates lo1's 3rd block (3 used) and admits
             hi1 (2 blocks) → 5 used, 0 free. Both running.
             Decode → lo1 34 tokens, hi1 33 tokens.
    Phase 3: schedule() → hi1 needs 3rd block, 0 free → preemption.
             lo1 (priority 5) is preempted, hi1 (priority 0) survives.
    """
    block_size = 16
    num_blocks = 6  # 1 null block → 5 usable
    num_tokens = block_size * 2  # 32 tokens = exactly 2 blocks

    scheduler = create_scheduler_with_priority(
        max_num_seqs=3,
        max_num_batched_tokens=200,
        num_blocks=num_blocks,
        block_size=block_size,
    )

    # --- Phase 1: low-priority request starts running ---
    lo1 = create_requests_with_priority(
        num_requests=1,
        priorities=[5],
        arrival_times=[1.0],
        num_tokens=num_tokens,
        req_ids=["lo1"],
    )[0]
    scheduler.add_request(lo1)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1

    # Decode: lo1 now has 33 tokens (crosses 32-token boundary).
    model_output = ModelRunnerOutput(
        req_ids=["lo1"],
        req_id_to_index={"lo1": 0},
        sampled_token_ids=[[100]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # --- Phase 2: high-priority request arrives AFTER lo1 is running ---
    hi1 = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[2.0],
        num_tokens=num_tokens,
        req_ids=["hi1"],
    )[0]
    scheduler.add_request(hi1)

    # schedule(): lo1 gets its 3rd block (3 used), hi1 admitted (5 used,
    # 0 free).  Both are now running.
    output = scheduler.schedule()
    assert any(r.req_id == "hi1" for r in output.scheduled_new_reqs)
    assert len(scheduler.running) == 2

    # Decode: lo1 → 34 tokens, hi1 → 33 tokens.
    model_output = ModelRunnerOutput(
        req_ids=["lo1", "hi1"],
        req_id_to_index={"lo1": 0, "hi1": 1},
        sampled_token_ids=[[101], [100]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # --- Phase 3: preemption with mixed-priority running requests ---
    # hi1 needs a 3rd block but 0 are free.  The scheduler picks the
    # lowest-priority running request to preempt:
    #   max(running, key=(priority, arrival_time)) → lo1 (5 > 0).
    output = scheduler.schedule()

    lo1_req = scheduler.requests["lo1"]
    assert lo1_req.status == RequestStatus.PREEMPTED, (
        "Expected low-priority 'lo1' to be preempted"
    )
    assert any(req.request_id == "hi1" for req in scheduler.running), (
        "High-priority 'hi1' should still be running"
    )


def test_priority_scheduling_no_preemption_when_space_available():
    """Test that preemption doesn't happen
    when there's space for new requests."""
    scheduler = create_scheduler_with_priority(
        max_num_seqs=3,  # Allow 3 concurrent requests
        max_num_batched_tokens=200,  # Sufficient token budget
    )

    # Add two low-priority running requests
    low_priority_requests = create_requests_with_priority(
        num_requests=2,
        priorities=[5, 5],
        arrival_times=[1.0, 2.0],
        num_tokens=30,
        req_ids=["lo1", "lo2"],
    )

    for request in low_priority_requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in low_priority_requests],
        req_id_to_index={
            req.request_id: i for i, req in enumerate(low_priority_requests)
        },
        sampled_token_ids=[[100] for _ in low_priority_requests],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # Add high-priority request
    high_priority_request = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[3.0],
        num_tokens=30,
        req_ids=["hi1"],
    )[0]

    scheduler.add_request(high_priority_request)

    # Schedule - should not preempt since there's space
    output = scheduler.schedule()

    # Should schedule the new request without preemption
    assert len(output.scheduled_new_reqs) == 1
    assert len(scheduler.running) == 3  # All three requests running
    assert len(scheduler.waiting) == 0  # No requests waiting


def test_priority_scheduling_preemption_victim_selection():
    """Test that the correct victim is selected for
    preemption based on priority and arrival time."""
    # This test verifies the priority-based victim selection logic
    # by checking the waiting queue order after adding requests with different
    # priorities
    scheduler = create_scheduler_with_priority(
        max_num_seqs=1,  # Force sequential processing to test priority order
    )

    # Create requests with different priorities
    requests = create_requests_with_priority(
        num_requests=3,
        priorities=[3, 2, 0],  # Different priorities: low, medium, high
        arrival_times=[1.0, 2.0, 3.0],
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule - should only schedule the highest priority request
    # (req_2, priority 0)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_new_reqs[0].req_id == "2"  # Highest priority

    # Verify the waiting queue has the remaining requests in priority order
    assert len(scheduler.waiting) == 2

    # Extract waiting requests and verify priority order
    waiting_requests = list(scheduler.waiting)

    waiting_priorities = [req.priority for req in waiting_requests]
    waiting_req_ids = [req.request_id for req in waiting_requests]

    # Should be req_1 (priority 2) then req_0 (priority 3)
    assert waiting_priorities == [2, 3]
    assert waiting_req_ids == ["1", "0"]


def test_priority_scheduling_equal_priority_preemption():
    """Test arrival time tiebreaker when requests have equal priority."""
    # This test verifies that arrival time is used as a tiebreaker for equal
    # priorities
    scheduler = create_scheduler_with_priority(
        max_num_seqs=1,  # Force sequential processing
    )

    # Create requests with same priority but different arrival times
    requests = create_requests_with_priority(
        num_requests=3,
        priorities=[2, 2, 2],  # Same priority
        arrival_times=[3.0, 1.0, 2.0],  # Different arrival times
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule - should schedule the request with earliest arrival time
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_new_reqs[0].req_id == "1"  # Earliest arrival (1.0)

    # Verify the waiting queue has remaining requests in arrival time order
    assert len(scheduler.waiting) == 2

    # Extract waiting requests and verify arrival time order
    waiting_requests = list(scheduler.waiting)

    waiting_arrival_times = [req.arrival_time for req in waiting_requests]
    waiting_req_ids = [req.request_id for req in waiting_requests]

    # Should be req_2 (arrival 2.0) then req_0 (arrival 3.0)
    assert waiting_arrival_times == [2.0, 3.0]
    assert waiting_req_ids == ["2", "0"]


def test_priority_scheduling_waiting_queue_order():
    """Test that the waiting queue maintains priority order."""
    scheduler = create_scheduler_with_priority(
        max_num_seqs=1,  # Only one request can run at a time
    )

    # Create multiple requests with different priorities
    requests = create_requests_with_priority(
        num_requests=4,
        priorities=[3, 1, 2, 0],  # Mixed priorities
        arrival_times=[1.0, 2.0, 3.0, 4.0],
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule - should only schedule the highest priority request
    # (req_3, priority 0)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_new_reqs[0].req_id == "3"

    # Verify waiting queue has remaining requests in priority order
    assert len(scheduler.waiting) == 3

    # Extract requests from waiting queue
    # (it's a heap, so we need to pop to see order)
    waiting_requests = list(scheduler.waiting)

    waiting_priorities = [req.priority for req in waiting_requests]
    waiting_req_ids = [req.request_id for req in waiting_requests]

    # Should be ordered by priority: req_1 (1), req_2 (2), req_0 (3)
    assert waiting_req_ids == ["1", "2", "0"]
    assert waiting_priorities == [1, 2, 3]


def test_priority_scheduling_fcfs_fallback():
    """Test that FCFS behavior is maintained when all
    requests have same priority."""
    scheduler = create_scheduler_with_priority()

    # Create requests with same priority but different arrival times
    priorities = [1, 1, 1, 1]  # All same priority
    arrival_times = [4.0, 1.0, 3.0, 2.0]  # Different arrival times
    requests = create_requests_with_priority(
        num_requests=4, priorities=priorities, arrival_times=arrival_times
    )

    # Add requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule
    output = scheduler.schedule()

    # Should schedule all requests in arrival time order
    assert len(output.scheduled_new_reqs) == 4
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]

    # Expected order by arrival time:
    # req_1 (1.0), req_3 (2.0), req_2 (3.0), req_0 (4.0)
    assert scheduled_req_ids == ["1", "3", "2", "0"]


def test_priority_scheduling_with_limited_slots():
    """Test priority scheduling when max_num_seqs limits concurrent requests."""
    scheduler = create_scheduler_with_priority(
        max_num_seqs=2,  # Only allow 2 concurrent requests
        max_num_batched_tokens=1000,  # Plenty of token budget
    )

    # Create requests with different priorities
    requests = create_requests_with_priority(
        num_requests=4,
        priorities=[3, 1, 2, 0],  # Mixed priorities
        arrival_times=[1.0, 2.0, 3.0, 4.0],
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule - should only schedule the 2 highest priority requests
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 2

    # Should schedule req_3 (priority 0) and req_1 (priority 1)
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert "3" in scheduled_req_ids  # Priority 0
    assert "1" in scheduled_req_ids  # Priority 1

    # Remaining requests should be in waiting queue in priority order
    assert len(scheduler.waiting) == 2

    # Extract waiting requests and verify order
    waiting_requests = list(scheduler.waiting)
    waiting_priorities = [req.priority for req in waiting_requests]
    waiting_req_ids = [req.request_id for req in waiting_requests]

    # Should be req_2 (priority 2) then req_0 (priority 3)
    assert waiting_priorities == [2, 3]
    assert waiting_req_ids == ["2", "0"]


def test_priority_scheduling_heap_property():
    """Test that the waiting queue maintains heap
    property for priority scheduling."""
    scheduler = create_scheduler_with_priority(
        max_num_seqs=1,  # Only one request can run at a time
    )

    # Add requests in random priority order
    priorities = [5, 1, 8, 3, 2, 7, 4, 6]
    arrival_times = [float(i) for i in range(len(priorities))]
    requests = create_requests_with_priority(
        num_requests=len(priorities),
        priorities=priorities,
        arrival_times=arrival_times,
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule one request at a time and verify priority order
    scheduled_priorities = []

    while scheduler.waiting:
        output = scheduler.schedule()
        if output.scheduled_new_reqs:
            req = output.scheduled_new_reqs[0]
            scheduled_priorities.append(requests[int(req.req_id)].priority)

            # Simulate completion to make room for next request
            model_output = ModelRunnerOutput(
                req_ids=[req.req_id],
                req_id_to_index={req.req_id: 0},
                sampled_token_ids=[[100]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )
            scheduler.update_from_output(output, model_output)

            # Finish the request to make room for the next one
            scheduler.finish_requests(req.req_id, RequestStatus.FINISHED_STOPPED)

    # Verify requests were scheduled in priority order (lowest value first)
    expected_priorities = sorted(priorities)
    assert scheduled_priorities == expected_priorities


def test_schedule_skip_tokenizer_init():
    scheduler = create_scheduler(skip_tokenizer_init=True)
    requests = create_requests(num_requests=5)
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)


def test_schedule_skip_tokenizer_init_structured_output_request():
    scheduler = create_scheduler(skip_tokenizer_init=True)
    structured_outputs_params = StructuredOutputsParams(regex="[0-9]+")
    sampling_params = SamplingParams(
        ignore_eos=False,
        max_tokens=16,
        structured_outputs=structured_outputs_params,
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id="0",
        prompt_token_ids=[0, 1],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
    )
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 0
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 1
    assert len(scheduler.deferred_waiting) == 1


@pytest.mark.parametrize("async_grammar", [True, False])
def test_grammar_compile_error_finishes_only_request(async_grammar: bool):
    scheduler = create_scheduler()
    manager = scheduler.structured_output_manager
    manager.backend = Mock()
    manager.backend_name = "xgrammar"
    manager.backend.compile_grammar.side_effect = RuntimeError(
        "forced FSM compilation error"
    )
    manager._use_async_grammar_compilation = async_grammar

    structured_outputs = StructuredOutputsParams(json='{"type": "object"}')
    structured_outputs._backend = "xgrammar"
    sampling_params = SamplingParams(
        max_tokens=16, structured_outputs=structured_outputs
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id="grammar-error",
        prompt_token_ids=[0, 1],
        sampling_params=sampling_params,
        pooling_params=None,
    )

    manager.grammar_init(request)
    assert request.structured_output_request is not None
    grammar_future = request.structured_output_request._grammar
    assert isinstance(grammar_future, Future)
    assert isinstance(grammar_future.exception(timeout=5), RuntimeError)

    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    assert not scheduler_output.num_scheduled_tokens

    engine_core_outputs = scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(req_ids=[], req_id_to_index={}),
    )

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    output = engine_core_outputs[0].outputs[0]
    assert output.request_id == request.request_id
    assert output.finish_reason == FinishReason.ERROR
    assert output.stop_reason is None
    assert output.error == RequestError(
        code="validation_error",
        message="Invalid structured outputs: forced FSM compilation error",
    )

    healthy_request = create_requests(num_requests=1, req_ids=["healthy-request"])[0]
    scheduler.add_request(healthy_request)
    next_output = scheduler.schedule()
    assert [req.req_id for req in next_output.scheduled_new_reqs] == [
        healthy_request.request_id
    ]


def test_abort_request_when_structured_output_fsm_cannot_advance():
    scheduler = object.__new__(Scheduler)
    sampling_params = SamplingParams(ignore_eos=True, max_tokens=4)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)

    request = Request(
        request_id="0",
        prompt_token_ids=[0, 1],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
    )
    request.structured_output_request = Mock()
    request.structured_output_request.grammar = Mock(spec=StructuredOutputGrammar)
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens

    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.accept_tokens.return_value = False
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = Mock()
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.kv_cache_manager.estimate_cached_tokens.return_value = 0
    scheduler.kv_event_publisher = Mock()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.aux_output_connector = None
    scheduler.grammar_compile_error_reqs = {}
    scheduler.encoder_cache_mismatch_reqs = set()
    scheduler.vllm_config = Mock()
    scheduler.return_sampling_mask = False
    scheduler.recompute_kv_load_failures = False
    scheduler.defer_block_free = False
    scheduler.num_spec_tokens = 0
    scheduler.has_mamba_layers = False
    scheduler.make_stats = Mock(return_value=None)
    scheduler.max_model_len = 128

    def free_request(req: Request, delay_free_blocks: bool = False):
        scheduler.finished_req_ids.add(req.request_id)
        scheduler.requests.pop(req.request_id, None)
        return None, None

    scheduler._free_request = Mock(side_effect=free_request)

    output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[123]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    engine_core_outputs = scheduler.update_from_output(output, model_runner_output)

    scheduler.structured_output_manager.accept_tokens.assert_called_once_with(
        request, [123]
    )
    assert request.resumable is False
    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    assert not scheduler.running
    scheduler._free_request.assert_called_once_with(request)
    assert len(engine_core_outputs[0].outputs) == 1
    engine_core_output = engine_core_outputs[0].outputs[0]
    assert engine_core_output.request_id == request.request_id
    assert engine_core_output.new_token_ids == [123]
    assert engine_core_output.finish_reason == FinishReason.ERROR


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
@pytest.mark.parametrize(
    "use_ec_connector, ec_role", [(False, None), (True, "ec_consumer")]
)
def test_priority_scheduling_preemption_and_resumption_when_out_of_kv(
    use_ec_connector, ec_role, use_v2_model_runner
):
    """Test that priority scheduling preempts lower priority requests
    when out of KV cache space."""
    # Create scheduler with very limited memory to force preemption
    scheduler = create_scheduler_with_priority(
        max_num_seqs=2,  # Allow multiple requests
        max_num_batched_tokens=200,
        num_blocks=5,  # Can hold 64 tokens (first block is null)
        block_size=16,  # Standard block size
        use_kv_connector=True,
        # encoder connector should not affect test results
        use_ec_connector=use_ec_connector,
        ec_role=ec_role,
        use_v2_model_runner=use_v2_model_runner,
    )

    # Create a request and schedule it
    request_low = create_requests_with_priority(
        num_requests=1,
        priorities=[1],
        arrival_times=[0.0],
        num_tokens=30,
        starting_idx=0,
    )[0]
    scheduler.add_request(request_low)
    # 1st schedule
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 1

    # Simulate model execution - 1st decode
    model_output = ModelRunnerOutput(
        req_ids=[request_low.request_id],
        req_id_to_index={request_low.request_id: 0},
        sampled_token_ids=[[100]],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # Create a high priority request and schedule it
    request_high = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[1.0],
        num_tokens=32,
        starting_idx=1,
    )[0]
    scheduler.add_request(request_high)
    # 2nd schedule
    output = scheduler.schedule()
    # KV cache should be full at this point
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 0
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 2

    # Simulate model execution - 2nd decode
    requests = [request_low, request_high]
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[100] for _ in requests],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # 3rd schedule - this should trigger preemption
    # req_low needs 32 tokens = 2 blocks
    # req_high needs 33 tokens = 3 blocks
    # so doesn't fit in 4 blocks.
    output = scheduler.schedule()

    # Should have preempted req_low
    assert len(output.scheduled_new_reqs) == 0
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert output.scheduled_cached_reqs.req_ids[0] == request_high.request_id
    assert scheduler.requests[request_low.request_id].status == RequestStatus.PREEMPTED
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 1

    # Simulate model execution - 3rd decode
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[], [100]],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    # Finish the requests to make room for the preempted requests to resume
    scheduler.update_from_output(output, model_output)
    scheduler.finish_requests(request_high.request_id, RequestStatus.FINISHED_STOPPED)

    # 4th Schedule - this should trigger the resumption
    output = scheduler.schedule()
    scheduled_cached_reqs = output.scheduled_cached_reqs

    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 1

    if use_v2_model_runner:
        # V2 emits the resumed request as a NewRequestData, carrying its full
        # token ids in prefill_token_ids (instead of cached all_token_ids).
        assert scheduled_cached_reqs.num_reqs == 0
        assert len(output.scheduled_new_reqs) == 1
        new_req = output.scheduled_new_reqs[0]
        assert new_req.req_id == request_low.request_id
        # Resumed tokens include 30 prompt tokens and 2 decoded tokens.
        assert len(new_req.prefill_token_ids) == 32
        assert new_req.prefill_token_ids[31] == 100
    else:
        assert len(output.scheduled_new_reqs) == 0
        assert scheduled_cached_reqs.num_reqs == 1
        # Preempted request resumed in scheduled_cached_reqs
        assert len(scheduled_cached_reqs.resumed_req_ids) == 1
        assert len(scheduled_cached_reqs.all_token_ids) == 1
        assert scheduled_cached_reqs.req_ids[0] == request_low.request_id
        assert request_low.request_id in scheduled_cached_reqs.resumed_req_ids
        assert request_low.request_id in scheduled_cached_reqs.all_token_ids
        # Resumed tokens include 30 prompt tokens and 2 decoded tokens
        assert len(scheduled_cached_reqs.all_token_ids[request_low.request_id]) == 32
        assert scheduled_cached_reqs.all_token_ids[request_low.request_id][31] == 100


@pytest.mark.parametrize(
    ("enable_chunked_prefill", "is_encoder_decoder", "expect_enabled"),
    [
        (True, False, True),
        (False, False, False),
        # Encoder-decoder models should always have it disabled
        (False, True, False),
        (True, True, False),
    ],
)
def test_chunked_prefill_disabled_for_encoder_decoder(
    enable_chunked_prefill: bool, is_encoder_decoder: bool, expect_enabled: bool
) -> None:
    """Validate that chunked prefill is appropriately disabled for
    encoder-decoder models."""
    scheduler_config = SchedulerConfig(
        enable_chunked_prefill=enable_chunked_prefill,
        is_encoder_decoder=is_encoder_decoder,
        # Must <= max_num_batched_tokens if chunked prefill is disabled
        max_model_len=SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS,
    )

    # `is_encoder_decoder` should only be used during construction
    # of the config, and otherwise stored in the model config.
    assert "is_encoder_decoder" not in vars(scheduler_config)
    assert "is_encoder_decoder" not in [
        f.name for f in dataclasses.fields(scheduler_config)
    ]
    _validate_chunked_prefill_settings_for_encoder_decoder(
        scheduler_config, is_encoder_decoder, expect_enabled
    )

    # Ensure it is retained in VllmConfig, even after its post-init.
    vllm_config = VllmConfig(scheduler_config=scheduler_config)
    _validate_chunked_prefill_settings_for_encoder_decoder(
        vllm_config.scheduler_config, is_encoder_decoder, expect_enabled
    )


def _validate_chunked_prefill_settings_for_encoder_decoder(
    scheduler_config: SchedulerConfig, is_encoder_decoder: bool, expect_enabled: bool
) -> None:
    """Validate chunked prefill settings in the scheduler config for
    encoder-decoder models."""
    assert scheduler_config.enable_chunked_prefill is expect_enabled
    if is_encoder_decoder:
        # Encoder-decoder models should automatically disable chunked multimodal
        # inputs as well
        assert scheduler_config.disable_chunked_mm_input is not expect_enabled
    if is_encoder_decoder and not expect_enabled:
        assert scheduler_config.long_prefill_token_threshold == 0


# ==============================================================================
# EPD (Encoder-Prefill-Decode) Encoder-cache-specific tests start
# NOTE: In E->P->D disagg case, both KV and EC Connector works in P instance
# Unless specify, the existence of KV Connector should not affect any test results
# ==============================================================================


def _assert_right_encoder_cache_allocated(
    scheduler: Scheduler,
    hashes_to_check: list[str] | None = None,
    requests: list[Request] | None = None,
    expected_total_allocated: int | None = None,
):
    """Check whether encoder cache is allocated correctly."""
    encoder_cache_manager = scheduler.encoder_cache_manager

    # Verify encoder cache manager exists
    assert encoder_cache_manager is not None, "Encoder cache manager should exist"

    # Verify number of cache
    if expected_total_allocated is not None:
        assert len(encoder_cache_manager.cached) == expected_total_allocated
        if expected_total_allocated == 0:
            return

    # Verify each request with MM data is in cache
    cached_hashes = set(encoder_cache_manager.cached.keys())

    if hashes_to_check:
        missed_hashes = set(hashes_to_check) - cached_hashes
        assert not missed_hashes, (
            f"Miss hashes: {missed_hashes} "
            f"Existing encoder cache: {encoder_cache_manager.cached}"
        )

    for req in requests if requests is not None else []:
        if req.mm_features:
            mm_hashes = [f.identifier for f in req.mm_features]
            req_hashes = set(mm_hashes)  # unique hashes set
            missed_hashes = req_hashes - cached_hashes
            assert not missed_hashes, (
                f"Miss hashes in cache for request {req.request_id}: {missed_hashes} "
                f"Existing encoder cache: {encoder_cache_manager.cached}"
            )


def _assert_right_ec_connector_metadata(
    output: SchedulerOutput,
    mm_features_list: list[MultiModalFeatureSpec],
):
    """Verify that ECConnector metadata EXACTLY matches the input MM data."""
    # Get the connector metadata
    metadata = output.ec_connector_metadata

    # Create lookup dictionaries for efficient access
    metadata_dict = {mm_data.mm_hash: mm_data for mm_data in metadata.mm_datas}

    # Check all required identifiers exist in metadata; and no extra
    # In ECExampleConnector format
    # NOTE: even having same identifier, the mm_features can be different
    # since their mm_position can be in different offsets, etc
    identifiers_dict = {f.identifier for f in mm_features_list}
    assert set(metadata_dict.keys()) == identifiers_dict

    # Verify the info matches
    for i, mm_feature in enumerate(mm_features_list):
        identifier = mm_feature.identifier
        assert metadata_dict[identifier].mm_hash == identifier
        assert metadata_dict[identifier].num_token == mm_feature.mm_position.length


def _assert_right_encoder_inputs(
    output: SchedulerOutput,
    check_exist: bool | None = True,
    requests: list[Request] | None = None,
    expected_encoder_inputs: list[list[int]] | None = None,
    expected_total_reqs: int | None = None,
):
    """Verify that requests/mm_hashes should (not) in scheduled encoder input
    If check_exist is False, this function returns True
    if requests are NOT in encoder inputs"""
    # Get the scheduled encoder inputs
    # NOTE: scheduled_encoder_inputs is a dictionary with request id as key
    scheduled_encoder_inputs = output.scheduled_encoder_inputs

    # Check if scheduled_encoder_inputs is empty as expected
    if expected_total_reqs is not None:
        assert len(scheduled_encoder_inputs) == expected_total_reqs
        if expected_total_reqs == 0:
            return

    # Number of expected encoder inputs should match number of requests
    if expected_encoder_inputs:
        assert check_exist and requests is not None  # only support expect input exist
        assert len(requests) == len(expected_encoder_inputs)

    # Check request (not) exist as expected
    for i, request in enumerate(requests if requests is not None else []):
        assert (request.request_id in scheduled_encoder_inputs) is check_exist, (
            f"Request {request.id} presence mismatch: expected {check_exist}, "
            f"got {request.id in scheduled_encoder_inputs}"
        )
        if expected_encoder_inputs:
            scheduled_encoder_input = scheduled_encoder_inputs[request.request_id]
            assert scheduled_encoder_input == expected_encoder_inputs[i]


def test_scheduler_no_ec_connector_by_default():
    """Test scheduler doesn't have EC connector by default."""
    scheduler = create_scheduler()
    assert scheduler.ec_connector is None


def test_mamba_align_encoder_cache_cap_makes_progress():
    """Two individually cacheable images must not deadlock Mamba alignment."""
    block_size = 768
    encoder_cache_size = 600
    scheduler = create_scheduler(
        max_num_batched_tokens=8192,
        block_size=block_size,
        enable_prefix_caching=True,
    )
    scheduler.need_mamba_block_aligned_split = True
    scheduler.max_num_encoder_input_tokens = encoder_cache_size
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=encoder_cache_size)

    first_image_end = 510
    request = create_requests(
        num_requests=1,
        num_tokens=1010,
        mm_positions=[
            [
                PlaceholderRange(offset=0, length=500),
                PlaceholderRange(offset=first_image_end, length=500),
            ]
        ],
        max_tokens=1,
        block_size=block_size,
        req_ids=["req"],
    )[0]
    scheduler.add_request(request)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens[request.request_id] == first_image_end
    assert output.scheduled_encoder_inputs[request.request_id] == [0]

    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    output = scheduler.schedule()
    next_block_boundary = block_size
    assert output.num_scheduled_tokens[request.request_id] == (
        next_block_boundary - first_image_end
    )
    assert output.scheduled_encoder_inputs[request.request_id] == [1]


def test_mamba_align_eagle_schedules_encoder_at_boundary():
    """EAGLE lookahead at an aligned MM boundary requires encoder cache."""
    block_size = 512
    scheduler = create_scheduler(
        max_num_batched_tokens=700,
        max_model_len=2048,
        block_size=block_size,
        enable_prefix_caching=True,
    )
    scheduler.need_mamba_block_aligned_split = True
    scheduler.use_eagle = True
    scheduler.use_eagle_block_drop = True
    scheduler.num_prefill_lookahead = 1
    scheduler.max_num_encoder_input_tokens = 2048
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=2048)

    request = create_requests(
        num_requests=1,
        num_tokens=1200,
        mm_positions=[[PlaceholderRange(offset=block_size, length=100)]],
        max_tokens=1,
        block_size=block_size,
        req_ids=["req"],
    )[0]
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == block_size
    assert output.scheduled_encoder_inputs[request.request_id] == [0]


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_text_only_request(use_kv_connector):
    """Test text-only requests don't allocate encoder cache."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        use_kv_connector=use_kv_connector,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    NUM_PROMPT_TOKENS = 100

    # Create text-only request (no mm_positions)
    requests = create_requests(
        num_requests=1,
        num_tokens=NUM_PROMPT_TOKENS,
    )
    assert not requests[0].mm_features  # No MM data

    scheduler.add_request(requests[0])
    output = scheduler.schedule()

    # Should schedule
    assert len(output.scheduled_new_reqs) == 1

    # Scheduled tokens should equal prompt tokens exactly
    scheduled = output.num_scheduled_tokens[requests[0].request_id]
    assert scheduled == NUM_PROMPT_TOKENS, (
        f"Text-only should schedule {NUM_PROMPT_TOKENS}, got {scheduled}"
    )

    # Encoder cache should be empty
    _assert_right_encoder_cache_allocated(scheduler, expected_total_allocated=0)

    # ECConnector should carry no metadata
    _assert_right_ec_connector_metadata(output, mm_features_list=[])

    # Scheduled encoder input should be empty; no mm to compute
    _assert_right_encoder_inputs(output, expected_total_reqs=0)


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_cache_hit_external_load(use_kv_connector):
    """Test ec_consumer loads from external cache when hit.
    A normal basic operation for EPD disaggrgation"""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        enable_prefix_caching=True,
        # kv connector should not effect test results
        use_kv_connector=use_kv_connector,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    # Create MM request
    NUM_TOKENS = 200  # NOTE: includes mm tokens
    NUM_ENCODER_TOKENS = 100
    mm_hashes_list = [["hash_test1"]]
    mm_positions = [[PlaceholderRange(offset=0, length=NUM_ENCODER_TOKENS)]]

    request = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS,
        mm_hashes_list=mm_hashes_list,
        mm_positions=mm_positions,
    )[0]

    # Mock cache hit - encoder cache has_exists externally
    scheduler.ec_connector.has_cache_item = Mock(return_value=True)
    scheduler.ec_connector.update_state_after_alloc = Mock(
        wraps=scheduler.ec_connector.update_state_after_alloc
    )

    scheduler.add_request(request)
    output = scheduler.schedule()
    # Should schedule prompt tokens
    scheduled_tokens = output.num_scheduled_tokens[request.request_id]
    assert scheduled_tokens == NUM_TOKENS

    # Should called update_state_after_alloc for external load
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(request, 0)

    # Encoder cache should contain mm items from request
    _assert_right_encoder_cache_allocated(scheduler, requests=[request])

    # ECConnector should carry metadata of request
    _assert_right_ec_connector_metadata(output, mm_features_list=request.mm_features)

    # Scheduled encoder input should be empty; no mm to compute
    _assert_right_encoder_inputs(output, expected_total_reqs=0)


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_cache_miss_computes_locally(use_kv_connector):
    """Test consumer can compute encoder locally when cache miss (fallback)."""
    # encoder cache itself if it doesn't receive it from external storage

    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        enable_prefix_caching=True,
        use_kv_connector=use_kv_connector,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    # Verify consumer role
    assert scheduler.ec_connector is not None
    assert not scheduler.ec_connector.is_producer

    # Create MM request
    request_mm_missed = create_requests(
        num_requests=1,
        num_tokens=200,  # Total (including 100 MM)
        mm_positions=[[PlaceholderRange(offset=0, length=100)]],  # 100 MM tokens
    )[0]

    # Mock cache miss - encoder cache doesn't exist externally
    scheduler.ec_connector.has_cache_item = Mock(return_value=False)

    scheduler.add_request(request_mm_missed)
    output = scheduler.schedule()

    # SCHEDULER should decide to compute encoder locally (fallback)
    assert len(output.scheduled_new_reqs) == 1

    # Should schedule full prompt tokens
    scheduled_tokens = output.num_scheduled_tokens[request_mm_missed.request_id]
    assert scheduled_tokens == 200, (
        f"Expected 200 tokens on cache miss, got {scheduled_tokens}"
    )

    # Encoder cache should contain mm items from request
    _assert_right_encoder_cache_allocated(scheduler, requests=[request_mm_missed])

    # ECConnector should carry no metadata (missed cache)
    _assert_right_ec_connector_metadata(output, mm_features_list=[])

    # Scheduled encoder input contain mm for request_mm_missed
    _assert_right_encoder_inputs(
        output,
        requests=[request_mm_missed],
        expected_encoder_inputs=[[0]],  # index 0 of the mm item
        expected_total_reqs=1,
    )

    # Then MODEL_RUNNER will execute the encoder and cache the result


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_with_partial_cache_hit_multi_round(use_kv_connector):
    """Test consumer with partial cache hit (local & connector) with 2 requests."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        enable_prefix_caching=True,
        use_kv_connector=use_kv_connector,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    # Create MM request
    NUM_TOKENS_1 = 300  # NOTE: includes mm tokens
    NUM_ENCODER_TOKENS_1 = 50
    mm_hashes_list_1 = [["hash1_A", "hash1_B", "hash1_A", "hash1_F"]]
    mm_positions_1 = [
        [
            PlaceholderRange(offset=0, length=NUM_ENCODER_TOKENS_1),
            PlaceholderRange(offset=100, length=NUM_ENCODER_TOKENS_1),
            PlaceholderRange(offset=200, length=NUM_ENCODER_TOKENS_1),
            PlaceholderRange(offset=250, length=NUM_ENCODER_TOKENS_1),
        ]
    ]
    has_cache_item_result_map_1 = {"hash1_A": False, "hash1_B": True, "hash1_F": True}
    # Create request with 4 MM items, with 2 identical items
    request1 = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS_1,
        mm_hashes_list=mm_hashes_list_1,
        mm_positions=mm_positions_1,
        max_tokens=1,  # For simplicity
    )[0]

    # Mock partial cache hit: 1st and 3rd missing, 2nd and 4th exist
    scheduler.ec_connector.has_cache_item = Mock(
        side_effect=lambda hash_val: has_cache_item_result_map_1[hash_val]
    )
    scheduler.ec_connector.update_state_after_alloc = Mock(
        wraps=scheduler.ec_connector.update_state_after_alloc
    )

    scheduler.add_request(request1)
    output = scheduler.schedule()

    # Should schedule all tokens
    scheduled_tokens = output.num_scheduled_tokens[request1.request_id]
    assert scheduled_tokens == NUM_TOKENS_1

    # Encoder cache should contain all mm items from request
    _assert_right_encoder_cache_allocated(scheduler, requests=[request1])

    # Should have called update_state_after_alloc for external load
    scheduler.ec_connector.update_state_after_alloc.assert_called()
    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # ECConnector should carry metadata for 2nd and 4th mm item
    _assert_right_ec_connector_metadata(
        output, mm_features_list=[request1.mm_features[1], request1.mm_features[3]]
    )

    # Should schedule ONLY 1 encoder input (index 0), no repeat for identical items
    _assert_right_encoder_inputs(
        output,
        requests=[request1],
        expected_encoder_inputs=[[0]],  # index 0 of the mm item ONLY
        expected_total_reqs=1,
    )

    # Simulate model execution 1 step
    model_output = ModelRunnerOutput(
        req_ids=[request1.request_id],
        req_id_to_index={request1.request_id: 0},
        sampled_token_ids=[[100]],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # request1 is finished after outputting 1 token
    # Finish request
    scheduler.finish_requests(request1.request_id, RequestStatus.FINISHED_LENGTH_CAPPED)

    # Create another request with 4 MM items
    NUM_TOKENS_2 = 400
    NUM_ENCODER_TOKENS_2 = 50
    mm_hashes_list_2 = [["hash1_C", "hash1_D", "hash1_E", "hash1_A"]]
    mm_positions_2 = [
        [
            PlaceholderRange(offset=0, length=NUM_ENCODER_TOKENS_2),
            PlaceholderRange(offset=100, length=NUM_ENCODER_TOKENS_2),
            PlaceholderRange(offset=200, length=NUM_ENCODER_TOKENS_2),
            PlaceholderRange(offset=250, length=NUM_ENCODER_TOKENS_2),
        ]
    ]
    has_cache_item_result_map_2 = {
        "hash1_C": True,
        "hash1_D": False,
        "hash1_E": False,
        "hash1_A": True,
    }
    request2 = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS_2,
        mm_hashes_list=mm_hashes_list_2,
        mm_positions=mm_positions_2,
        max_tokens=1,  # For simplicity
    )[0]

    # Mock partial cache hit: only hash1_A and hash1_C exist in connector
    scheduler.ec_connector.has_cache_item = Mock(
        side_effect=lambda hash_val: has_cache_item_result_map_2[hash_val]
    )

    scheduler.add_request(request2)
    output = scheduler.schedule()

    # Check
    # Should schedule all tokens
    scheduled_tokens = output.num_scheduled_tokens[request2.request_id]
    assert scheduled_tokens == 400

    # Encoder cache should contain all mm items from request2
    _assert_right_encoder_cache_allocated(scheduler, requests=[request2])

    # hash1_A should not be loaded from connector
    # since it's computed in last request & exist in local cache
    # Order of getting encoder cache should be: local cache -> connector-> compute
    # update_state_after_alloc is called for all paths:
    #   index 0 (hash1_C): connector hit → queued for load
    #   index 1 (hash1_D): cache miss → no-op inside connector
    #   index 2 (hash1_E): cache miss → no-op inside connector
    scheduler.ec_connector.update_state_after_alloc.assert_any_call(request2, 0)
    scheduler.ec_connector.update_state_after_alloc.assert_any_call(request2, 1)
    scheduler.ec_connector.update_state_after_alloc.assert_any_call(request2, 2)

    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # ECConnector should carry metadata for hash1_C only (index 0)
    _assert_right_ec_connector_metadata(
        output, mm_features_list=[request2.mm_features[0]]
    )

    # Should schedule 2 encoder input hash1_D and hash1_E (index 1, 2)
    _assert_right_encoder_inputs(
        output,
        requests=[request2],
        expected_encoder_inputs=[[1, 2]],
        expected_total_reqs=1,
    )


@pytest.mark.parametrize("cache_exist", ["local", "connector_only", "no_where"])
@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_schedule_multiple_requests(cache_exist, use_kv_connector):
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_seqs=10,  # allow multiple requests
        max_num_batched_tokens=2048,
        enable_prefix_caching=True,
        use_kv_connector=use_kv_connector,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )
    mm_hashes_list = [[f"hash_{i}"] for i in range(10)]
    mm_positions = [[PlaceholderRange(offset=i, length=100)] for i in range(10)]
    requests = create_requests(
        num_requests=10,
        num_tokens=200,
        mm_hashes_list=mm_hashes_list,
        mm_positions=mm_positions,
    )
    for request in requests:
        scheduler.add_request(request)

    # Set up to test different encoder cache existence scenario after preemption
    # Order of getting encoder cache should be: local cache -> connector-> compute
    scheduler.ec_connector.update_state_after_alloc = Mock(
        wraps=scheduler.ec_connector.update_state_after_alloc
    )

    if cache_exist == "local":
        # Allocate cache to cache manager manually to mimic
        for req in requests:
            scheduler.encoder_cache_manager.allocate(req, 0)
    else:
        # Make sure local encoder cache empty
        scheduler.encoder_cache_manager.cached = {}

    if cache_exist == "connector_only":
        # Cache exist in ec_connector
        scheduler.ec_connector.has_cache_item = Mock(return_value=True)
    elif cache_exist == "no_where":
        scheduler.ec_connector.has_cache_item = Mock(return_value=False)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)

    ## Encoder-cache-specific checks:
    # mm_hashes of requests exist in cache after scheduling for all scenario
    _assert_right_encoder_cache_allocated(scheduler, requests=requests)

    if cache_exist == "connector_only":
        scheduler.ec_connector.update_state_after_alloc.assert_called_with(
            requests[-1], 0
        )

        # Concat mm_features for the 10 requests together
        mm_features_list = [feature for req in requests for feature in req.mm_features]

        # Check metadata should contain mm data for all 10 requests
        _assert_right_ec_connector_metadata(output, mm_features_list=mm_features_list)
    elif cache_exist == "local":
        # Local cache hit: items never reach update_state_after_alloc
        scheduler.ec_connector.update_state_after_alloc.assert_not_called()
        _assert_right_ec_connector_metadata(output, mm_features_list=[])
    else:
        # no_where: called from encoder_inputs_to_schedule but no-op
        # inside connector (has_cache_item returns False)
        assert cache_exist == "no_where"
        scheduler.ec_connector.update_state_after_alloc.assert_called()
        _assert_right_ec_connector_metadata(output, mm_features_list=[])

    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # Should only schedule encoder input when cache is not found anywhere
    if cache_exist == "no_where":
        _assert_right_encoder_inputs(
            output,
            requests=requests,
            expected_encoder_inputs=[[0] for _ in range(10)],
            expected_total_reqs=10,
        )
    else:
        _assert_right_encoder_inputs(output, expected_total_reqs=0)


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_unable_to_allocate(use_kv_connector):
    """Test whether scheduler with ECConnector is able to handle
    unable to allocate (run out of blocks).
    """
    # Setup Scheduler With Mock External Cache Hit.
    BLOCK_SIZE = 4
    NUM_BLOCKS = 10
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        enable_prefix_caching=True,
        use_kv_connector=use_kv_connector,
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    # Mock ec_connector load external cache behavior
    scheduler.ec_connector.has_cache_item = Mock(return_value=True)
    scheduler.ec_connector.update_state_after_alloc = Mock(
        wraps=scheduler.ec_connector.update_state_after_alloc
    )

    # Create two requests. The second request will not be able to
    # allocate slots because it will not have enough blocks.
    NUM_REQUESTS = 2
    NUM_TOKENS = (NUM_BLOCKS // 2 + 1) * BLOCK_SIZE
    MAX_TOKENS = 2
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
        mm_hashes_list=[["hash_1"], ["hash_2"]],
        mm_positions=[
            [PlaceholderRange(offset=1, length=10)] for _ in range(NUM_REQUESTS)
        ],
        max_tokens=MAX_TOKENS,
        block_size=BLOCK_SIZE,
    )
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    # Setup MODEL_RUNNER_OUTPUT to be run in _step_until_done later
    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[1000]] * len(req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # Just one request should be running.
    output = scheduler.schedule()
    scheduled_tokens = output.num_scheduled_tokens[scheduler.running[0].request_id]
    assert scheduled_tokens == NUM_TOKENS
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1

    # Should have called update_state_after_alloc for external load
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(
        scheduler.running[0], 0
    )
    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # All memory should be freed, with one request waiting.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 1

    # Just one request should be running.
    output = scheduler.schedule()
    scheduled_tokens = output.num_scheduled_tokens[scheduler.running[0].request_id]
    assert scheduled_tokens == NUM_TOKENS
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0

    # update_state_after_alloc should be called for loading external cache
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(
        scheduler.running[0], 0
    )
    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # All memory should be freed, with no requests waiting / running.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 0


@pytest.mark.parametrize("cache_exist", ["local", "connector_only", "no_where"])
@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_priority_scheduling_ec_connector_preemption_and_resumption(
    cache_exist, use_kv_connector
):
    """Test that priority scheduling preempts lower priority requests
    when out of KV cache space."""
    # Create scheduler with very limited memory to force preemption
    scheduler = create_scheduler_with_priority(
        model="llava-hf/llava-1.5-7b-hf",
        enable_prefix_caching=True,
        max_num_seqs=2,  # allow multiple requests
        # kv connector should not effect test results
        use_kv_connector=use_kv_connector,
        num_blocks=15,  # can hold 244 tokens with 14 blocks (first block is null)
        block_size=16,  # standard block size
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    # Mock cache hit: Both cache exist in connector (at E->PD initially)
    scheduler.ec_connector.has_cache_item = Mock(return_value=True)
    scheduler.ec_connector.update_state_after_alloc = Mock(
        wraps=scheduler.ec_connector.update_state_after_alloc
    )

    # Create a request and schedule it (and to be preempted)
    request_low = create_requests_with_priority(
        num_requests=1,
        priorities=[1],
        arrival_times=[0.0],
        num_tokens=94,
        mm_hashes_list=[["hash_low"]],
        # NOTE: this test only preempt the last block.
        # Setting mm_position at the last block can force to recompute encoding
        mm_positions=[[PlaceholderRange(offset=82, length=10)]],
        starting_idx=0,
    )[0]
    scheduler.add_request(request_low)
    # 1st schedule
    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 1
    scheduled_tokens = output.num_scheduled_tokens[request_low.request_id]
    assert scheduled_tokens == 94
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 1

    ## Encoder-cache-specific checks:
    # Encoder cache should contain mm items from request
    _assert_right_encoder_cache_allocated(scheduler, requests=[request_low])

    # Verify update_state_after_alloc called (external load)
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(request_low, 0)
    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # ECConnector should carry metadata of request
    _assert_right_ec_connector_metadata(
        output, mm_features_list=request_low.mm_features
    )

    # Scheduled encoder input should be empty; no mm to compute
    _assert_right_encoder_inputs(output, expected_total_reqs=0)

    # Simulate model execution - 1st decode
    model_output = ModelRunnerOutput(
        req_ids=[request_low.request_id],
        req_id_to_index={request_low.request_id: 0},
        sampled_token_ids=[[100]],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # Create a high priority request and schedule it
    request_high = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[1.0],
        num_tokens=128,
        mm_hashes_list=[["hash_high"]],
        mm_positions=[[PlaceholderRange(offset=1, length=10)]],
        max_tokens=2,
        starting_idx=1,
    )[0]
    scheduler.add_request(request_high)
    # 2nd schedule
    output = scheduler.schedule()

    # KV cache should be full at this point
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 0
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 2

    ## Encoder-cache-specific checks:
    # Encoder cache should contain mm items from request
    _assert_right_encoder_cache_allocated(scheduler, requests=[request_high])

    # Verify update_state_after_alloc called (external load)
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(request_high, 0)
    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # ECConnector should carry metadata of request
    _assert_right_ec_connector_metadata(
        output, mm_features_list=request_high.mm_features
    )

    # Scheduled encoder input should be empty; no mm to compute
    _assert_right_encoder_inputs(output, expected_total_reqs=0)

    # Simulate model execution - 2nd decode
    requests = [request_low, request_high]
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[100] for _ in requests],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # 3rd schedule - - this should trigger preemption
    # req_low needs 96 tokens = 6 blocks
    # req_high needs 129 tokens = 9 blocks
    # so doesn't fit in 14 blocks.
    output = scheduler.schedule()

    # Should have preempted req_low
    assert len(output.scheduled_new_reqs) == 0
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert output.scheduled_cached_reqs.req_ids[0] == request_high.request_id
    assert scheduler.requests[request_low.request_id].status == RequestStatus.PREEMPTED
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 1

    ## Encoder-cache-specific checks:
    # request_high is in decode phase now
    # ECConnector should carry no metadata
    _assert_right_ec_connector_metadata(output, mm_features_list=[])

    # Scheduled encoder input should be empty; no mm to compute
    _assert_right_encoder_inputs(output, expected_total_reqs=0)

    # Simulate model execution - 3rd decode, after req_low was preempted
    requests = [request_low, request_high]
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[100], [100, 200]],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    # Finish the requests to make room for the preempted requests to resume
    # req_high is finished after outputting 2 tokens
    scheduler.update_from_output(output, model_output)
    scheduler.finish_requests(
        request_high.request_id, RequestStatus.FINISHED_LENGTH_CAPPED
    )

    # Set up to test different encoder cache existence scenario after preemption
    # Order of getting encoder cache should be: local cache -> connector-> compute
    # By default, the cache should still exist in local in this test case
    if cache_exist != "local":
        # Make local encoder cache empty
        scheduler.encoder_cache_manager.cached = {}

    if cache_exist == "connector_only":
        # Cache exist in ec_connector
        scheduler.ec_connector.has_cache_item = Mock(return_value=True)
    elif cache_exist == "no_where":
        scheduler.ec_connector.has_cache_item = Mock(return_value=False)

    # 4th Schedule - this should trigger req_low resumption from waiting
    output = scheduler.schedule()
    scheduled_cached_reqs = output.scheduled_cached_reqs

    assert len(output.scheduled_new_reqs) == 0
    assert scheduled_cached_reqs.num_reqs == 1
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 1

    # Preempted request resumed in scheduled_cached_reqs
    assert len(scheduled_cached_reqs.resumed_req_ids) == 1
    assert len(scheduled_cached_reqs.all_token_ids) == 1
    assert scheduled_cached_reqs.req_ids[0] == request_low.request_id
    assert request_low.request_id in scheduled_cached_reqs.resumed_req_ids
    assert request_low.request_id in scheduled_cached_reqs.all_token_ids
    ## Resumed tokens include 94 prompt tokens and 2 decoded tokens
    assert len(scheduled_cached_reqs.all_token_ids[request_low.request_id]) == 96
    assert scheduled_cached_reqs.all_token_ids[request_low.request_id][95] == 100
    assert scheduler.running[0].request_id == request_low.request_id
    assert request_high.request_id in output.finished_req_ids

    ## Encoder-cache-specific checks:
    # mm_hash of request_low exists in cache after scheduling for all scenario
    _assert_right_encoder_cache_allocated(scheduler, requests=[request_low])

    if cache_exist == "connector_only":
        scheduler.ec_connector.update_state_after_alloc.assert_called_with(
            request_low, 0
        )
        _assert_right_ec_connector_metadata(
            output, mm_features_list=request_low.mm_features
        )
    elif cache_exist == "local":
        scheduler.ec_connector.update_state_after_alloc.assert_not_called()
        _assert_right_ec_connector_metadata(output, mm_features_list=[])
    else:
        assert cache_exist == "no_where"
        scheduler.ec_connector.update_state_after_alloc.assert_called_with(
            request_low, 0
        )
        _assert_right_ec_connector_metadata(output, mm_features_list=[])

    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # Should only schedule encoder input when cache is not found anywhere
    if cache_exist == "no_where":
        _assert_right_encoder_inputs(
            output,
            requests=[request_low],
            expected_encoder_inputs=[[0]],
            expected_total_reqs=1,
        )
    else:
        _assert_right_encoder_inputs(output, expected_total_reqs=0)


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_allocate_encoder_tokens_with_external_load(use_kv_connector):
    """Scenario:
      - Encoder cache size: 32
      - Request A: 1 feature (12 tokens) → NOT cached remotely.
      - Request B: 3 features (3 x 10 tokens) → ALL cached remotely.

    Steps:
      1. Schedule Request A (locally uses 12 tokens).
      2. Schedule Request B (remote cache) - only schedule 1st and 2nd
      3. Free A's cache, then schedule B again (continuation) - schedule 3rd image
    """
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=1024,
        enable_prefix_caching=True,
        use_kv_connector=use_kv_connector,
        block_size=16,
        num_blocks=11,  # Can hold 160 tokens (first block is null)
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    # Limit the number of available slots of EncoderCacheManager
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=32)

    # Create MM request1
    NUM_TOKENS_1 = 50  # NOTE: includes mm tokens
    NUM_ENCODER_TOKENS_1 = 12
    mm_hashes_list_1 = [["hash1_1"]]
    mm_positions_1 = [[PlaceholderRange(offset=0, length=NUM_ENCODER_TOKENS_1)]]

    request1 = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS_1,
        mm_hashes_list=mm_hashes_list_1,
        mm_positions=mm_positions_1,
        max_tokens=1,  # For simplicity
        req_ids=["req1"],
    )[0]

    # Create MM request1 with 3 MM items
    NUM_TOKENS_2 = 40
    NUM_ENCODER_TOKENS_2 = 10
    mm_hashes_list_2 = [["hash2_1", "hash2_2", "hash2_3"]]
    mm_positions_2 = [
        [
            PlaceholderRange(offset=0, length=NUM_ENCODER_TOKENS_2),
            PlaceholderRange(offset=12, length=NUM_ENCODER_TOKENS_2),
            PlaceholderRange(offset=24, length=NUM_ENCODER_TOKENS_2),
        ]
    ]

    request2 = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS_2,
        mm_hashes_list=mm_hashes_list_2,
        mm_positions=mm_positions_2,
        max_tokens=10,
        req_ids=["req2"],
    )[0]

    # Mock cache hit: MM of request1 NOT cached remotely, request2 cached remotely
    scheduler.ec_connector.has_cache_item = Mock(
        side_effect=lambda hash_value: hash_value in mm_hashes_list_2[0]
    )
    scheduler.ec_connector.update_state_after_alloc = Mock(
        wraps=scheduler.ec_connector.update_state_after_alloc
    )

    scheduler.add_request(request1)
    scheduler.add_request(request2)
    output = scheduler.schedule()

    # Now, since encoder cache manager can only store 32 tokens
    # It should allocated mm item hash1_1, hash2_1 and hash2_2
    scheduled_tokens = output.num_scheduled_tokens[request1.request_id]
    assert scheduled_tokens == NUM_TOKENS_1
    assert scheduler.get_num_unfinished_requests() == 2

    # Encoder cache should contain mm item from request1
    _assert_right_encoder_cache_allocated(
        scheduler, hashes_to_check=["hash1_1", "hash2_1", "hash2_2"]
    )

    # request2's 2nd mm item is the last call of update_state_after_alloc
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(request2, 1)
    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # ECConnector should carry metadata of hash2_1 and hash2_2 ONLY
    _assert_right_ec_connector_metadata(
        output, mm_features_list=[request2.mm_features[0], request2.mm_features[1]]
    )

    # Should schedule ONLY 1 encoder input
    _assert_right_encoder_inputs(
        output,
        requests=[request1],
        expected_encoder_inputs=[[0]],  # index 0 of the mm item of request1
        expected_total_reqs=1,
    )

    # Simulate model execution 1 step
    model_output = ModelRunnerOutput(
        req_ids=[request1.request_id, request2.request_id],
        req_id_to_index={request1.request_id: 0, request2.request_id: 1},
        sampled_token_ids=[[100], [121]],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # request1 is finished after outputting 1 token
    # Finish request
    scheduler.finish_requests(request1.request_id, RequestStatus.FINISHED_LENGTH_CAPPED)
    assert scheduler.get_num_unfinished_requests() == 1

    # Schedule again; Now request1's encoder cache should be freed
    # -> hash2_3 can be scheduled and allocated
    output = scheduler.schedule()

    # Check
    # Should schedule all tokens
    scheduled_tokens = output.num_scheduled_tokens[request2.request_id]
    print(f"Hero: scheduled_tokens for req2: {scheduled_tokens}")
    print(f"hero: num_scheduled_tokens 2: {output.num_scheduled_tokens}")

    # Encoder cache should contain all mm items from request2
    _assert_right_encoder_cache_allocated(scheduler, requests=[request2])

    # request2's 3rd mm item is the ONLY call of update_state_after_alloc
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(request2, 2)
    scheduler.ec_connector.update_state_after_alloc.assert_called_once()

    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # ECConnector should carry metadata for hash2_3 ONLY
    _assert_right_ec_connector_metadata(
        output, mm_features_list=[request2.mm_features[2]]
    )

    # Should schedule no encoder input
    _assert_right_encoder_inputs(
        output,
        expected_total_reqs=0,
    )


# ==============================================================================
# EPD (Encoder-Prefill-Decode) Encoder-cache-specific tests end
# ==============================================================================


def test_prepend_skipped_requests_order():
    scheduler = create_scheduler(max_num_seqs=1, use_kv_connector=True)
    requests = create_requests(num_requests=4)
    for request in requests:
        scheduler.add_request(request)

    # 4 requests waiting, capture their order
    expected_waiting_reqs = list(scheduler.waiting)

    # simulate first 2 waiting requests are waiting for remote KVs
    for req in expected_waiting_reqs[:2]:
        req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.waiting.remove_requests(expected_waiting_reqs[:2])
    for req in expected_waiting_reqs[:2]:
        scheduler.kv_holding_waiting.add_request(req)

    # schedule step
    # expect the first 2 waiting to be skipped, the third running,
    # and the fourth waiting
    scheduler.schedule()

    # pop the third request which is expected to be running
    expected_waiting_reqs.pop(2)

    # verify waiting order is preserved, with the KV-holding requests first
    waiting_reqs = list(scheduler.kv_holding_waiting) + list(scheduler.waiting)
    assert waiting_reqs == expected_waiting_reqs


def test_remote_kv_promotion_drains_before_grammar_prefix():
    scheduler = create_scheduler(max_num_seqs=1)
    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.return_value = (0, False)

    requests = create_requests(num_requests=4)
    req_grammar_1, req_grammar_2, req_remote, req_tail = requests

    # simulate two structured-output grammar requests at the waiting head
    # that become ready now.
    req_grammar_1.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    req_grammar_1.structured_output_request = Mock(grammar=object())
    req_grammar_2.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    req_grammar_2.structured_output_request = Mock(grammar=object())

    # simulate a remote-KV request that is ready to be promoted now. Parked
    # loads hold KV blocks, so they drain ahead of the grammar requests.
    req_remote.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    for request in requests:
        scheduler.add_request(request)
    assert list(scheduler.kv_holding_waiting) == [req_remote]
    scheduler.finished_recving_kv_req_ids.add(req_remote.request_id)
    scheduler._update_waiting_for_remote_kv = Mock()

    output = scheduler.schedule()

    assert output.scheduled_new_reqs
    assert output.scheduled_new_reqs[0].req_id == req_remote.request_id
    waiting_req_ids = [req.request_id for req in scheduler.waiting]
    assert waiting_req_ids == [
        req_grammar_1.request_id,
        req_grammar_2.request_id,
        req_tail.request_id,
    ]


def test_fcfs_mixed_blocked_types_remote_load_drains_first():
    scheduler = create_scheduler(max_num_batched_tokens=20)
    scheduler._update_waiting_for_remote_kv = Mock()

    mk_req = lambda req_id, num_tokens=1: create_requests(  # noqa: E731
        num_requests=1, num_tokens=num_tokens, req_ids=[req_id]
    )[0]
    req_grammar, req_remote, req_stream = (
        mk_req("grammar"),
        mk_req("remote"),
        mk_req("stream"),
    )
    req_regular, req_tail = mk_req("regular", 20), mk_req("tail")
    req_grammar.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    req_grammar.structured_output_request = Mock(grammar=None)
    req_remote.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    req_stream.status = RequestStatus.WAITING_FOR_STREAMING_REQ

    for req in (req_grammar, req_remote, req_stream, req_regular, req_tail):
        scheduler.add_request(req)
    scheduler.schedule()
    assert list(scheduler.waiting) == [req_grammar, req_stream, req_tail]
    assert list(scheduler.kv_holding_waiting) == [req_remote]

    scheduler.finish_requests(req_regular.request_id, RequestStatus.FINISHED_ABORTED)
    assert not scheduler.running

    req_grammar.structured_output_request = Mock(grammar=object())
    scheduler.finished_recving_kv_req_ids.add(req_remote.request_id)
    req_stream.status = RequestStatus.WAITING

    second_output = scheduler.schedule()
    expected_order = [
        req_remote.request_id,
        req_grammar.request_id,
        req_stream.request_id,
        req_tail.request_id,
    ]
    assert [req.req_id for req in second_output.scheduled_new_reqs] == expected_order
    assert [req.request_id for req in scheduler.running] == expected_order
    scheduler._update_waiting_for_remote_kv.assert_called_once_with(req_remote)


def test_abort_request_waiting_for_remote_kvs():
    scheduler = create_scheduler(use_kv_connector=True)

    # add a single request
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)

    # set request to waiting for remote KVs, and abort it
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.finish_requests((request.request_id,), RequestStatus.FINISHED_ABORTED)
    assert request.status == RequestStatus.FINISHED_ABORTED

    # verify request is not deleted
    assert request.request_id in scheduler.requests

    # finish recving request
    scheduler_output = scheduler.schedule()
    model_runner_output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        kv_connector_output=KVConnectorOutput(finished_recving={request.request_id}),
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # assert request is deleted
    assert request.request_id not in scheduler.requests
    assert not scheduler.finished_recving_kv_req_ids


def test_abort_request_finished_recving():
    scheduler = create_scheduler(use_kv_connector=True)

    # add a single request
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)

    # set request to waiting for remote KVs, finished but not yet updated
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.finished_recving_kv_req_ids.add(request.request_id)

    # abort request
    scheduler.finish_requests((request.request_id,), RequestStatus.FINISHED_ABORTED)
    assert request.status == RequestStatus.FINISHED_ABORTED

    # verify request is deleted
    assert request.request_id not in scheduler.requests
    assert not scheduler.finished_recving_kv_req_ids


def test_delayed_kv_connector_free_keeps_scheduler_active():
    scheduler = create_scheduler(use_kv_connector=True)
    queued_request, request = create_requests(
        num_requests=2, req_ids=["queued", "finished"]
    )
    scheduler.add_request(queued_request)

    assert not scheduler.has_finished_requests()

    request.status = RequestStatus.FINISHED_STOPPED
    scheduler.requests[request.request_id] = request
    scheduler.finished_req_ids = set()

    assert scheduler.has_finished_requests()
    assert scheduler.has_requests()

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_runner_output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        kv_connector_output=KVConnectorOutput(finished_sending={request.request_id}),
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    assert request.request_id not in scheduler.requests
    assert not scheduler.has_finished_requests()


def test_scheduler_kv_connector_stats():
    """Test worker-side, scheduler-side, and combined KV connector stats."""

    class GenericKVConnectorStats(KVConnectorStats):
        def reset(self):
            self.data = {}

        def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
            self.data.update(other.data)
            return self

        def reduce(self) -> dict[str, int | float]:
            return {}

        def is_empty(self) -> bool:
            return not self.data

    test_cases = (
        ({"worker": 1}, None, {"worker": 1}),
        (None, {"scheduler": 2}, {"scheduler": 2}),
        ({"worker": 1}, {"scheduler": 2}, {"worker": 1, "scheduler": 2}),
    )

    for worker_data, scheduler_data, expected_data in test_cases:
        scheduler = create_scheduler()
        worker_stats = (
            GenericKVConnectorStats(data=worker_data) if worker_data else None
        )
        scheduler_stats = (
            GenericKVConnectorStats(data=scheduler_data) if scheduler_data else None
        )
        scheduler.connector = Mock()
        scheduler.connector.get_kv_connector_stats.return_value = (
            scheduler_stats if worker_stats is None else None
        )
        scheduler.connector.take_events.return_value = []

        def update_connector_output(
            kv_connector_output: KVConnectorOutput,
            scheduler=scheduler,
            scheduler_stats=scheduler_stats,
        ):
            scheduler.connector.get_kv_connector_stats.return_value = scheduler_stats

        scheduler.connector.update_connector_output.side_effect = (
            update_connector_output
        )

        model_output = ModelRunnerOutput(
            req_ids=["req_0"],
            req_id_to_index={"req_0": 0},
            sampled_token_ids=[[123]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None],
            kv_connector_output=KVConnectorOutput(kv_connector_stats=worker_stats)
            if worker_stats
            else None,
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=None,
            num_scheduled_tokens={"req_0": 1},
            total_num_scheduled_tokens=1,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )

        engine_core_outputs = scheduler.update_from_output(
            scheduler_output, model_output
        )

        final_stats = next(
            iter(engine_core_outputs.values())
        ).scheduler_stats.kv_connector_stats
        assert final_stats == expected_data


def test_scheduler_ec_connector_stats():
    """Test worker-side, scheduler-side, and combined EC connector stats."""

    class GenericECConnectorStats(ECConnectorStats):
        def reset(self):
            self.data = {}

        def aggregate(self, other: ECConnectorStats) -> ECConnectorStats:
            self.data.update(other.data)
            return self

        def reduce(self) -> dict[str, int | float]:
            return {}

        def is_empty(self) -> bool:
            return not self.data

    test_cases = (
        ({"worker": 1}, None, {"worker": 1}),
        (None, {"scheduler": 2}, {"scheduler": 2}),
        ({"worker": 1}, {"scheduler": 2}, {"worker": 1, "scheduler": 2}),
    )

    for worker_data, scheduler_data, expected_data in test_cases:
        scheduler = create_scheduler()
        worker_stats = (
            GenericECConnectorStats(data=worker_data) if worker_data else None
        )
        scheduler_stats = (
            GenericECConnectorStats(data=scheduler_data) if scheduler_data else None
        )
        scheduler.ec_connector = Mock()
        scheduler.ec_connector.take_unavailable_requests.return_value = set()
        scheduler.ec_connector.get_ec_connector_stats.return_value = (
            scheduler_stats if worker_stats is None else None
        )

        def update_connector_output(
            ec_connector_output: ECConnectorOutput,
            scheduler=scheduler,
            scheduler_stats=scheduler_stats,
        ):
            scheduler.ec_connector.get_ec_connector_stats.return_value = scheduler_stats

        scheduler.ec_connector.update_connector_output.side_effect = (
            update_connector_output
        )

        model_output = ModelRunnerOutput(
            req_ids=["req_0"],
            req_id_to_index={"req_0": 0},
            sampled_token_ids=[[123]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None],
            ec_connector_output=ECConnectorOutput(ec_connector_stats=worker_stats)
            if worker_stats
            else None,
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=None,
            num_scheduled_tokens={"req_0": 1},
            total_num_scheduled_tokens=1,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )

        engine_core_outputs = scheduler.update_from_output(
            scheduler_output, model_output
        )

        final_stats = next(
            iter(engine_core_outputs.values())
        ).scheduler_stats.ec_connector_stats
        assert final_stats == expected_data


def test_ec_connector_update_connector_output_called():
    """Test that worker-side EC connector output is forwarded to the
    EC connector's update_connector_output hook."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        use_ec_connector=True,
        ec_role="ec_consumer",
    )
    scheduler.ec_connector.update_connector_output = Mock()

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    ec_connector_output = ECConnectorOutput(finished_sending={"hash_test1"})
    model_runner_output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        ec_connector_output=ec_connector_output,
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    scheduler.ec_connector.update_connector_output.assert_called_once_with(
        ec_connector_output
    )


# ==============================================================================
# Variable-length encoder cross-attention block allocation tests
# ==============================================================================


def _create_encoder_decoder_scheduler(
    block_size: int = 16,
    num_blocks: int = 10000,
    max_num_batched_tokens: int = 8192,
    max_num_seqs: int = 16,
) -> Scheduler:
    """Create a scheduler configured for encoder-decoder cross-attention
    block allocation testing.

    Constructs a scheduler with both FullAttentionSpec (self-attention) and
    CrossAttentionSpec (cross-attention) KV cache groups, then patches it
    to behave as an encoder-decoder model.
    """
    from vllm.v1.core.encoder_cache_manager import EncoderDecoderCacheManager
    from vllm.v1.kv_cache_interface import CrossAttentionSpec

    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_num_batched_tokens,
        # is_encoder_decoder disables chunked prefill and prefix caching
        is_encoder_decoder=True,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    cache_config.num_gpu_blocks = num_blocks

    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
    )

    # KV cache config with both self-attention and cross-attention groups,
    # mirroring an encoder-decoder model like Whisper.
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["self_attn_layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["cross_attn_layer"],
                CrossAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
        ],
    )

    # Construct the scheduler. Since opt-125m is not truly encoder-decoder,
    # the __init__ won't set up encoder-decoder internals. We patch them
    # after construction.
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        hash_block_size=block_size,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )

    # Patch to enable encoder-decoder behavior in the scheduling loop.
    scheduler.is_encoder_decoder = True
    scheduler.max_num_encoder_input_tokens = max_num_batched_tokens
    scheduler.encoder_cache_manager = EncoderDecoderCacheManager(
        cache_size=max_num_batched_tokens
    )

    return scheduler


def _get_num_cross_attn_blocks(scheduler: Scheduler, request_id: str) -> int:
    """Get the number of cross-attention blocks allocated for a request."""
    from vllm.v1.core.single_type_kv_cache_manager import CrossAttentionManager

    coordinator = scheduler.kv_cache_manager.coordinator
    for manager in coordinator.single_type_managers:
        if isinstance(manager, CrossAttentionManager):
            blocks = manager.req_to_blocks.get(request_id, [])
            return len(blocks)
    raise AssertionError("No CrossAttentionManager found in coordinator")


def test_variable_length_cross_attn_block_allocation():
    """Test that cross-attention blocks are allocated per-request based on
    actual encoder input length, not a fixed maximum.

    Fixed max-encoder-length allocation would assign
    `ceil(max_encoder_tokens / block_size)` blocks to
    every request whereas with dynamic allocation, exactly
    `ceil(actual_encoder_tokens / block_size)` blocks are assigned
    to each request.
    """
    block_size = 16
    scheduler = _create_encoder_decoder_scheduler(block_size=block_size)

    # Create requests with distinctly different encoder input lengths,
    # simulating variable-length audio inputs to a model like Whisper.
    encoder_lengths = [500, 1000, 200]
    num_prompt_tokens = 100  # Decoder prompt tokens

    requests = []
    for i, enc_len in enumerate(encoder_lengths):
        req = create_requests(
            num_requests=1,
            num_tokens=num_prompt_tokens,
            mm_hashes_list=[[f"enc_hash_{i}"]],
            mm_positions=[[PlaceholderRange(offset=0, length=enc_len)]],
            req_ids=[f"req_{i}"],
        )[0]
        requests.append(req)

    # Add and schedule all requests.
    for req in requests:
        scheduler.add_request(req)

    output = scheduler.schedule()

    # All requests should be scheduled.
    assert len(output.scheduled_new_reqs) == len(requests)

    # Verify cross-attention blocks per request match the actual encoder length.
    from math import ceil

    for req, enc_len in zip(requests, encoder_lengths):
        expected_blocks = ceil(enc_len / block_size)
        actual_blocks = _get_num_cross_attn_blocks(scheduler, req.request_id)

        assert actual_blocks == expected_blocks, (
            f"Request {req.request_id} with {enc_len} encoder tokens: "
            f"expected {expected_blocks} cross-attn blocks, "
            f"got {actual_blocks}"
        )

    # Verify that different encoder lengths produce different block counts,
    # confirming variable-length (not fixed-max) allocation.
    block_counts = [
        _get_num_cross_attn_blocks(scheduler, req.request_id) for req in requests
    ]
    assert len(set(block_counts)) > 1, (
        "All requests have the same number of cross-attn blocks, "
        "suggesting static max-based allocation instead of per-request"
    )


def test_cross_attn_blocks_not_over_allocated():
    """Test that cross-attention blocks are not over-allocated compared to
    what each request actually needs."""
    from math import ceil

    block_size = 16
    max_encoder_tokens = 1500  # e.g., Whisper's max mel-spectrogram length
    scheduler = _create_encoder_decoder_scheduler(block_size=block_size)

    # Request with a small encoder input (much less than the max).
    small_enc_len = 200
    request = create_requests(
        num_requests=1,
        num_tokens=100,
        mm_hashes_list=[["enc_small"]],
        mm_positions=[[PlaceholderRange(offset=0, length=small_enc_len)]],
        req_ids=["req_small"],
    )[0]

    scheduler.add_request(request)
    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 1

    actual_blocks = _get_num_cross_attn_blocks(scheduler, request.request_id)
    expected_blocks = ceil(small_enc_len / block_size)
    max_blocks = ceil(max_encoder_tokens / block_size)

    # Blocks should match the actual encoder length.
    assert actual_blocks == expected_blocks, (
        f"Expected {expected_blocks} blocks for {small_enc_len} encoder tokens, "
        f"got {actual_blocks}"
    )

    # Blocks should be strictly less than what max-based allocation would give.
    assert actual_blocks < max_blocks, (
        f"Cross-attn blocks ({actual_blocks}) should be less than max "
        f"({max_blocks}), indicating no over-allocation"
    )


def test_cross_attn_blocks_not_under_allocated():
    """Test that cross-attention blocks are sufficient for each request's
    actual encoder input length. Every encoder token must have a slot.

    Tests various edge cases including exact block boundaries, off-by-one,
    and the minimum/maximum encoder input sizes.
    """
    from math import ceil

    block_size = 16

    # Test various encoder lengths including edge cases around block boundaries.
    test_cases = [
        1,  # Minimum: single encoder token
        block_size - 1,  # Just under one full block
        block_size,  # Exactly one full block
        block_size + 1,  # Just over one block (needs 2 blocks)
        block_size * 10,  # Exact multiple of block size
        block_size * 10 + 1,  # One over exact multiple
        1500,  # Whisper's typical max
    ]

    for enc_len in test_cases:
        scheduler = _create_encoder_decoder_scheduler(block_size=block_size)

        request = create_requests(
            num_requests=1,
            num_tokens=100,
            mm_hashes_list=[[f"enc_{enc_len}"]],
            mm_positions=[[PlaceholderRange(offset=0, length=enc_len)]],
            req_ids=[f"req_{enc_len}"],
        )[0]

        scheduler.add_request(request)
        output = scheduler.schedule()

        assert len(output.scheduled_new_reqs) == 1

        actual_blocks = _get_num_cross_attn_blocks(scheduler, request.request_id)
        expected_blocks = ceil(enc_len / block_size)

        # Number of blocks must be exactly ceil(enc_len / block_size).
        assert actual_blocks == expected_blocks, (
            f"Encoder length {enc_len}: expected {expected_blocks} blocks, "
            f"got {actual_blocks}"
        )

        # Total available slots must be >= encoder tokens (no under-allocation).
        total_slots = actual_blocks * block_size
        assert total_slots >= enc_len, (
            f"Encoder length {enc_len}: total slots {total_slots} < "
            f"needed {enc_len} (under-allocation)"
        )


def test_cross_attn_zero_blocks_without_encoder_inputs():
    """Test that requests without encoder inputs get zero cross-attention
    blocks, even when the scheduler is configured for encoder-decoder."""
    block_size = 16
    scheduler = _create_encoder_decoder_scheduler(block_size=block_size)

    # Create a text-only request (no mm_features).
    request = create_requests(
        num_requests=1,
        num_tokens=100,
        req_ids=["req_text_only"],
    )[0]

    # Text-only request has no encoder inputs.
    assert not request.has_encoder_inputs

    scheduler.add_request(request)
    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 1

    # No cross-attention blocks should be allocated.
    actual_blocks = _get_num_cross_attn_blocks(scheduler, request.request_id)
    assert actual_blocks == 0, (
        f"Text-only request should have 0 cross-attn blocks, got {actual_blocks}"
    )


def test_eagle3_mm_encoder_cache_with_shift():
    """Test EAGLE3 encoder scheduling accounts for shift_computed_tokens.

    Regression test for issue #32469: When EAGLE3 is enabled with
    disable_chunked_mm_input=True, ensure encoder inputs are scheduled
    when tokens overlap the MM range, properly accounting for
    shift_computed_tokens in the boundary calculation.

    Without the fix, the scheduler would fail to schedule encoder inputs
    at the boundary, causing "Encoder cache miss" errors.
    """
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=1024,
        disable_chunked_mm_input=True,
        max_model_len=2048,
        num_speculative_tokens=4,  # This enables EAGLE with shift=1
    )

    mm_start_pos = 100
    mm_length = 576

    mm_positions = [
        [PlaceholderRange(offset=mm_start_pos, length=mm_length)],
    ]

    requests = create_requests(
        num_requests=1,
        num_tokens=mm_start_pos + mm_length + 100,
        mm_positions=mm_positions,
    )

    # Start with some tokens already computed to simulate decoding
    request = requests[0]
    request.num_computed_tokens = 0

    scheduler.add_request(request)
    output = scheduler.schedule()

    assert output is not None
    shift_computed_tokens = 1
    req_id = request.request_id

    assert req_id in output.num_scheduled_tokens
    num_scheduled = output.num_scheduled_tokens[req_id]

    mm_feature = request.mm_features[0]
    start_pos = mm_feature.mm_position.offset
    tokens_end = request.num_computed_tokens + num_scheduled
    scheduled_end_with_shift = tokens_end + shift_computed_tokens

    # Assert that we scheduled into the MM range (test setup verification)
    assert scheduled_end_with_shift > start_pos, (
        f"Test setup error: expected to schedule into MM range. "
        f"scheduled_end_with_shift={scheduled_end_with_shift}, "
        f"start_pos={start_pos}"
    )

    # The key assertion: when scheduled tokens overlap MM range
    # (accounting for EAGLE's shift), encoder MUST be scheduled.
    # Without the fix, this would fail at the boundary case.
    assert req_id in output.scheduled_encoder_inputs, (
        f"Encoder input missing: scheduled {num_scheduled} tokens "
        f"(computed={request.num_computed_tokens}, end={tokens_end}, "
        f"shifted_end={scheduled_end_with_shift}) overlapping MM at "
        f"{start_pos}. The fix must schedule encoder inputs."
    )


def test_free_encoder_inputs_respects_unconfirmed_placeholders():
    """Regression test for issue #38551 (rollback path): under async
    scheduling with speculative decoding, num_computed_tokens is advanced
    optimistically and can be rolled back when in-flight draft tokens are
    rejected. Freeing an encoder input as soon as num_computed_tokens passes
    the end of its placeholder range allows a later rollback to rewind back
    into the range, after which the worker's MM-embedding gather reads an
    evicted entry and crashes the engine with "Encoder cache miss". The
    scheduler must retain the input until the *confirmed* progress
    (num_computed_tokens - num_output_placeholders) passes the range end, so
    that no pending rejection can rewind into the range."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        num_speculative_tokens=3,
    )
    mm_start_pos = 50
    mm_length = 100
    mm_positions = [
        [PlaceholderRange(offset=mm_start_pos, length=mm_length)],
    ]
    request = create_requests(
        num_requests=1,
        num_tokens=mm_start_pos + mm_length + 100,
        mm_positions=mm_positions,
    )[0]
    manager = scheduler.encoder_cache_manager
    manager.allocate(request, 0)
    mm_end = mm_start_pos + mm_length

    # One optimistically-scheduled in-flight step advanced num_computed_tokens
    # by 1 sampled + 3 draft tokens; none are confirmed yet, so all 4 are
    # still output placeholders that a rejection could rewind.
    request.num_output_placeholders = 4

    # Optimistic progress reaches the end of the MM range, but the confirmed
    # position (mm_end + 1 - 4) is still inside it: a rejection could rewind
    # back into the range, so the entry must be retained.
    request.num_computed_tokens = mm_end + 1
    scheduler._free_encoder_inputs(request)
    assert manager.get_cached_input_ids(request) == {0}

    # Confirmed position still inside the range.
    request.num_computed_tokens = mm_end + 3
    scheduler._free_encoder_inputs(request)
    assert manager.get_cached_input_ids(request) == {0}

    # Confirmed position (mm_end + 4 - 4) now reaches the range end: even if
    # every unconfirmed token is rejected, progress cannot rewind into the
    # range, so the entry is freed.
    request.num_computed_tokens = mm_end + 4
    scheduler._free_encoder_inputs(request)
    assert manager.get_cached_input_ids(request) == set()


def test_free_encoder_inputs_notifies_the_ec_connector():
    """The connector learns the item is consumed, not just that the request ended.

    Connectors hold per-item transfer state (remote buffers, reservations);
    waiting for `request_finished` pins it for the whole generation.
    """
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    mm_start_pos, mm_length = 50, 100
    request = create_requests(
        num_requests=1,
        num_tokens=mm_start_pos + mm_length + 10,
        mm_positions=[[PlaceholderRange(offset=mm_start_pos, length=mm_length)]],
    )[0]
    scheduler.encoder_cache_manager.allocate(request, 0)
    scheduler.ec_connector = Mock()

    request.num_computed_tokens = mm_start_pos + mm_length - 1
    scheduler._free_encoder_inputs(request)
    scheduler.ec_connector.update_state_after_free.assert_not_called()

    request.num_computed_tokens = mm_start_pos + mm_length
    scheduler._free_encoder_inputs(request)
    scheduler.ec_connector.update_state_after_free.assert_called_once_with(request, 0)


def test_unavailable_encoder_input_fails_the_request_as_retryable():
    """An encoder input the connector gave up on must end the request.

    `FinishReason.ERROR` is the retryable channel the KV connector already uses
    for load failures, so the caller can re-issue; deferring instead parked the
    request until the client timed out.
    """
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    request = create_requests(
        num_requests=1,
        num_tokens=160,
        mm_positions=[[PlaceholderRange(offset=50, length=100)]],
    )[0]
    scheduler.add_request(request)
    scheduler.ec_connector = Mock()
    scheduler.ec_connector.take_unavailable_requests.return_value = {request.request_id}
    # `request_finished` reports (delay_free, params) for the request teardown.
    scheduler.ec_connector.request_finished.return_value = (False, None)

    scheduler_output = scheduler.schedule()
    outputs = scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    assert request.status == RequestStatus.FINISHED_ERROR
    engine_outputs = outputs[0].outputs
    assert [o.finish_reason for o in engine_outputs] == [FinishReason.ERROR]


def _create_mm_request_with_embedding_positions(
    request_id: str, identifier: str, is_embed: list[bool]
) -> Request:
    return create_requests(
        num_requests=1,
        num_tokens=len(is_embed),
        mm_hashes_list=[[identifier]],
        mm_positions=[
            [
                PlaceholderRange(
                    offset=0,
                    length=len(is_embed),
                    is_embed=torch.tensor(is_embed),
                )
            ]
        ],
        req_ids=[request_id],
    )[0]


def test_encoder_cache_accepts_matching_embed_count():
    """An actively referenced entry is reused when embedding counts match."""
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    identifier = "reused-client-uuid"
    owner = _create_mm_request_with_embedding_positions(
        "owner", identifier, [True, True, True, False, False, False]
    )
    compatible = _create_mm_request_with_embedding_positions(
        "compatible", identifier, [True, False, True, False, True, False, False]
    )
    scheduler.encoder_cache_manager.allocate(owner, 0)
    scheduler.add_request(compatible)

    scheduler_output = scheduler.schedule()

    assert compatible.request_id in scheduler_output.num_scheduled_tokens
    assert compatible.request_id not in scheduler.encoder_cache_mismatch_reqs
    assert compatible.request_id not in scheduler_output.scheduled_encoder_inputs


def test_encoder_cache_rejects_mismatched_embed_count(caplog_vllm):
    """A completed request's freeable entry rejects an incompatible reuse."""
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    identifier = "reused-client-uuid"
    owner = _create_mm_request_with_embedding_positions(
        "owner", identifier, [True, True, True, False, False, False]
    )
    mismatched = _create_mm_request_with_embedding_positions(
        "mismatched", identifier, [True, True, True, True, False, False]
    )
    scheduler.encoder_cache_manager.allocate(owner, 0)
    scheduler.encoder_cache_manager.free(owner)
    freeable_slots_before = scheduler.encoder_cache_manager.num_freeable_slots
    scheduler.add_request(mismatched)

    scheduler_output = scheduler.schedule()
    assert not scheduler_output.num_scheduled_tokens
    assert mismatched.request_id in scheduler.encoder_cache_mismatch_reqs
    assert (
        "multimodal identifier 'reused-client-uuid' holds 3 embeddings"
        in caplog_vllm.text
        and "expects 4" in caplog_vllm.text
    )

    outputs = scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(req_ids=[], req_id_to_index={}),
    )

    assert mismatched.status == RequestStatus.FINISHED_ERROR
    assert not scheduler.encoder_cache_mismatch_reqs
    assert mismatched.request_id not in scheduler.requests
    assert scheduler.encoder_cache_manager.cached[identifier] == set()
    assert (
        mismatched.request_id not in scheduler.encoder_cache_manager.request_cached_ids
    )
    assert scheduler.encoder_cache_manager.freeable[identifier] == 3
    assert scheduler.encoder_cache_manager.num_freeable_slots == freeable_slots_before
    output = outputs[mismatched.client_index].outputs[0]
    assert output.request_id == mismatched.request_id
    assert output.finish_reason == FinishReason.ERROR

    healthy = create_requests(num_requests=1, req_ids=["healthy"], num_tokens=6)[0]
    scheduler.add_request(healthy)
    next_output = scheduler.schedule()
    assert healthy.request_id in next_output.num_scheduled_tokens


def test_encoder_cache_embed_count_mismatch_restores_prior_hit_state():
    """A prior hit is released when a later mismatch fails the request."""
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    manager = scheduler.encoder_cache_manager
    valid_identifier = "valid-reuse"
    mismatch_identifier = "mismatched-reuse"
    valid_owner = _create_mm_request_with_embedding_positions(
        "valid-owner", valid_identifier, [True, True, False]
    )
    mismatch_owner = _create_mm_request_with_embedding_positions(
        "mismatch-owner", mismatch_identifier, [True, True, True, False]
    )
    manager.allocate(valid_owner, 0)
    manager.free(valid_owner)
    manager.allocate(mismatch_owner, 0)
    manager.free(mismatch_owner)
    freeable_slots_before = manager.num_freeable_slots

    request = create_requests(
        num_requests=1,
        num_tokens=7,
        mm_hashes_list=[[valid_identifier, mismatch_identifier]],
        mm_positions=[
            [
                PlaceholderRange(
                    offset=0,
                    length=3,
                    is_embed=torch.tensor([True, True, False]),
                ),
                PlaceholderRange(
                    offset=3,
                    length=4,
                    is_embed=torch.tensor([True, True, True, True]),
                ),
            ]
        ],
        req_ids=["partially-mutated"],
    )[0]
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    assert not scheduler_output.num_scheduled_tokens
    assert request.request_id in scheduler.encoder_cache_mismatch_reqs
    assert request.request_id in manager.cached[valid_identifier]
    assert valid_identifier not in manager.freeable
    assert manager.num_freeable_slots == freeable_slots_before - 2

    outputs = scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(req_ids=[], req_id_to_index={}),
    )

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    assert request.request_id not in manager.request_cached_ids
    assert manager.cached[valid_identifier] == set()
    assert manager.freeable[valid_identifier] == 2
    assert manager.freeable[mismatch_identifier] == 3
    assert manager.num_freeable_slots == freeable_slots_before
    output = outputs[request.client_index].outputs[0]
    assert output.finish_reason == FinishReason.ERROR


def test_encoder_cache_rejects_mismatched_embed_count_within_request():
    """One request cannot reuse an identifier with a different embed count."""
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    identifier = "reused-within-request"
    request = create_requests(
        num_requests=1,
        num_tokens=8,
        mm_hashes_list=[[identifier, identifier]],
        mm_positions=[
            [
                PlaceholderRange(
                    offset=0,
                    length=4,
                    is_embed=torch.tensor([True, True, False, False]),
                ),
                PlaceholderRange(
                    offset=4,
                    length=4,
                    is_embed=torch.tensor([True, True, True, False]),
                ),
            ]
        ],
        req_ids=["mismatched-within-request"],
    )[0]
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    assert not scheduler_output.num_scheduled_tokens
    assert request.request_id in scheduler.encoder_cache_mismatch_reqs
    assert identifier not in scheduler.encoder_cache_manager.cached

    outputs = scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(req_ids=[], req_id_to_index={}),
    )

    assert request.status == RequestStatus.FINISHED_ERROR
    assert not scheduler.encoder_cache_mismatch_reqs
    assert request.request_id not in scheduler.requests
    output = outputs[request.client_index].outputs[0]
    assert output.request_id == request.request_id
    assert output.finish_reason == FinishReason.ERROR


@pytest.mark.parametrize("prefill_lookahead", [1, 3])
def test_drafter_lookahead_encoder_cache_cap_makes_progress(prefill_lookahead):
    """Regression test for #40707: with a drafter look-ahead, two adjacent
    images that only fit in the encoder cache one at a time must not deadlock.
    Stopping short of the second image by the look-ahead used to leave the
    first image unfreed, so the second could never be allocated."""
    image_len = 600
    scheduler = create_scheduler(max_num_batched_tokens=900, max_model_len=2048)
    scheduler.num_prefill_lookahead = prefill_lookahead
    scheduler.max_num_encoder_input_tokens = 900
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=900)

    num_tokens = 2 * image_len + 20
    request = create_requests(
        num_requests=1,
        num_tokens=num_tokens,
        mm_positions=[
            [
                PlaceholderRange(offset=10, length=image_len),
                PlaceholderRange(offset=10 + image_len, length=image_len),
            ]
        ],
        max_tokens=1,
        req_ids=["req"],
    )[0]
    scheduler.add_request(request)

    scheduled_inputs = []
    while request.num_computed_tokens < num_tokens:
        output = scheduler.schedule()
        assert output.num_scheduled_tokens.get("req", 0) > 0
        scheduled_inputs += output.scheduled_encoder_inputs.get("req", [])
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=["req"],
                req_id_to_index={"req": 0},
                sampled_token_ids=[[]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
    assert scheduled_inputs == [0, 1]


@pytest.mark.parametrize("prefill_lookahead", [1, 3])
@pytest.mark.parametrize("past_chunk_end", [0, 1, 2])
def test_drafter_lookahead_encoder_cache_cap_respects_token_budget(
    prefill_lookahead, past_chunk_end
):
    """An unallocatable input reached only by look-ahead must not grow the chunk."""
    budget = 100
    scheduler = create_scheduler(max_num_batched_tokens=budget, max_model_len=2048)
    scheduler.num_prefill_lookahead = prefill_lookahead
    scheduler.max_num_encoder_input_tokens = 1000
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=300)

    request = create_requests(
        num_requests=1,
        num_tokens=budget + past_chunk_end + 320,
        mm_positions=[
            [
                PlaceholderRange(offset=10, length=80),
                PlaceholderRange(offset=budget + past_chunk_end, length=300),
            ]
        ],
        max_tokens=1,
        req_ids=["req"],
    )[0]
    scheduler.add_request(request)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens["req"] == budget
    assert output.scheduled_encoder_inputs["req"] == [0]


def test_free_encoder_inputs_unchanged_without_spec_decode():
    """Without speculative decoding, encoder inputs are freed as soon as
    num_computed_tokens passes the placeholder range, as before."""
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    mm_positions = [[PlaceholderRange(offset=50, length=100)]]
    request = create_requests(
        num_requests=1,
        num_tokens=250,
        mm_positions=mm_positions,
    )[0]
    manager = scheduler.encoder_cache_manager
    manager.allocate(request, 0)

    request.num_computed_tokens = 149
    scheduler._free_encoder_inputs(request)
    assert manager.get_cached_input_ids(request) == {0}

    request.num_computed_tokens = 150
    scheduler._free_encoder_inputs(request)
    assert manager.get_cached_input_ids(request) == set()


def test_waiting_encoder_stall_does_not_starve_later_requests():
    """A waiting request stalled by the encoder cache must not end the pass.

    Regression test: the zero-token branch used to `break`, letting the first
    encoder-stalled request starve every request queued behind it.
    """
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    scheduler.max_num_encoder_input_tokens = 0
    stalled, ready = create_requests(
        num_requests=2,
        num_tokens=100,
        mm_positions=[[PlaceholderRange(offset=0, length=50)], []],
        req_ids=["stalled", "ready"],
    )
    scheduler.add_request(stalled)
    scheduler.add_request(ready)

    output = scheduler.schedule()

    assert "ready" in output.num_scheduled_tokens


def test_waiting_encoder_stall_is_counted_as_deferred():
    """The requeue must go through skip_request's deferred_waiting accounting.

    A hand-rolled pop-and-prepend left the request out of `deferred_waiting`,
    skewing the DP prefill-capacity signal.
    """
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    scheduler.max_num_encoder_input_tokens = 0
    stalled = create_requests(
        num_requests=1,
        num_tokens=100,
        mm_positions=[[PlaceholderRange(offset=0, length=50)]],
        req_ids=["stalled"],
    )[0]
    scheduler.add_request(stalled)

    scheduler.schedule()

    assert stalled in scheduler.deferred_waiting


def test_waiting_encoder_stall_kv_holder_keeps_drain_priority():
    """A stalled request holding KV blocks must stay on the drain-first queue.

    Async KV-load resumes hold blocks and are drained before plain waiting
    requests; a hand-rolled requeue sent them to the plain queue.
    """
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    scheduler.max_num_encoder_input_tokens = 0
    stalled = create_requests(
        num_requests=1,
        num_tokens=100,
        mm_positions=[[PlaceholderRange(offset=0, length=50)]],
        req_ids=["stalled"],
    )[0]
    stalled.num_computed_tokens = 10
    scheduler.add_request(stalled)

    scheduler.schedule()

    assert stalled in scheduler.kv_holding_waiting


def test_encoder_cache_retained_across_preemption_and_resume():
    """Regression guard for issue #38551 (preemption path).

    A request preempted under KV pressure resets num_computed_tokens to 0
    and drops its encoder references (scheduler._preempt_request calls
    encoder_cache_manager.free). Because that only moves the entry into
    `freeable` (it is not evicted), the worker still holds it: the scheduler
    must NOT report the mm_hash as freed. On resume, re-requesting the
    encoder input must pull the still-cached entry back out of `freeable`
    without scheduling a recompute, keeping the scheduler and worker
    consistent. The spec-rollback retention margin does not gate this path,
    so it is covered separately here."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        num_speculative_tokens=3,
    )
    mm_positions = [[PlaceholderRange(offset=50, length=100)]]
    request = create_requests(
        num_requests=1,
        num_tokens=250,
        mm_hashes_list=[["img_a"]],
        mm_positions=mm_positions,
    )[0]
    manager = scheduler.encoder_cache_manager
    mm_hash = request.mm_features[0].identifier

    # Prefill scheduled and computed the encoder input; it is pinned.
    manager.allocate(request, 0)
    assert manager.get_cached_input_ids(request) == {0}

    # Preemption drops the request's encoder references (scheduler.py:
    # _preempt_request -> encoder_cache_manager.free) and resets progress.
    manager.free(request)
    request.num_computed_tokens = 0
    # The entry is now ref-free but only `freeable` (not evicted): the
    # worker still holds it, so nothing must be reported as freed.
    assert mm_hash in manager.cached
    assert mm_hash in manager.freeable
    assert manager.get_freed_mm_hashes() == []

    # Resume re-requests the encoder output. The still-cached entry is pulled
    # back out of `freeable` with no recompute and no worker-side free.
    assert manager.check_and_update_cache(request, 0) is True
    assert mm_hash not in manager.freeable
    assert manager.get_cached_input_ids(request) == {0}
    assert manager.get_freed_mm_hashes() == []


def test_encoder_cache_recomputed_when_evicted_during_preemption():
    """Companion to the retention case (issue #38551, preemption path).

    If a preempted request's retained encoder entry IS evicted under memory
    pressure before it resumes, the scheduler reports the mm_hash as freed
    (so the worker drops it) and a resume must schedule a recompute rather
    than assume the worker still holds it. check_and_update_cache must
    return False so the encoder input is re-scheduled."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        num_speculative_tokens=3,
    )
    mm_positions = [[PlaceholderRange(offset=50, length=100)]]
    request = create_requests(
        num_requests=1,
        num_tokens=250,
        mm_hashes_list=[["img_a"]],
        mm_positions=mm_positions,
    )[0]
    manager = scheduler.encoder_cache_manager
    mm_hash = request.mm_features[0].identifier

    manager.allocate(request, 0)
    # Preemption drops references; the entry becomes freeable.
    manager.free(request)
    request.num_computed_tokens = 0
    assert mm_hash in manager.freeable

    # A new request with a different image hits memory pressure and evicts
    # the freeable entry to make room.
    other = create_requests(
        num_requests=1,
        num_tokens=250,
        mm_hashes_list=[["img_b"]],
        mm_positions=mm_positions,
        req_ids=["1"],
    )[0]
    manager.num_free_slots = 50  # force eviction of the freeable entry
    assert manager.can_allocate(
        other, 0, encoder_compute_budget=10_000, num_embeds_to_schedule=0
    )

    # The evicted entry is reported to the worker, which drops it.
    assert mm_hash not in manager.cached
    assert manager.get_freed_mm_hashes() == [mm_hash]

    # On resume the original request must recompute (cache miss is correct).
    assert manager.check_and_update_cache(request, 0) is False


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_ensure_cache_available_defers_request(use_kv_connector):
    """Test that ensure_cache_available() returning False defers the request.

    When the EC connector signals a prefetch is in progress (returns False),
    the scheduler should:
    1. Not schedule the request (no KV cache or encoder cache allocated)
    2. Still schedule other requests behind the deferred one
    3. Schedule the deferred request on the next step when ensure_cache_available
       returns True and has_cache_item returns True
    """
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        enable_prefix_caching=True,
        use_kv_connector=use_kv_connector,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    NUM_TOKENS = 200
    NUM_ENCODER_TOKENS = 100

    request_deferred = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS,
        mm_positions=[[PlaceholderRange(offset=0, length=NUM_ENCODER_TOKENS)]],
        req_ids=["deferred"],
    )[0]

    request_behind = create_requests(
        num_requests=1,
        num_tokens=20,
        req_ids=["behind"],
    )[0]

    # --- Step 1: ensure_cache_available returns False → request deferred ---
    scheduler.ec_connector.ensure_cache_available = Mock(return_value=False)

    scheduler.add_request(request_deferred)
    scheduler.add_request(request_behind)
    output = scheduler.schedule()

    # ensure_cache_available must have been called with (request, num_computed_tokens=0)
    # for a brand-new request that has no cached tokens yet.
    ensure_call = scheduler.ec_connector.ensure_cache_available.call_args
    assert ensure_call.args == (request_deferred, 0)
    # Deferred request must NOT be scheduled
    assert request_deferred.request_id not in output.num_scheduled_tokens
    _assert_right_encoder_cache_allocated(scheduler, expected_total_allocated=0)
    # No KV blocks allocated for the deferred request
    for mgr in scheduler.kv_cache_manager.coordinator.single_type_managers:
        assert request_deferred.request_id not in mgr.req_to_blocks

    # The text-only request behind the deferred one MUST still be scheduled
    assert request_behind.request_id in output.num_scheduled_tokens
    assert output.num_scheduled_tokens[request_behind.request_id] == 20

    # --- Step 2: prefetch done, cache exists → request scheduled ---
    # has_cache_item is called inside _try_schedule_encoder_inputs (not during
    # deferral), so it is only relevant here in step 2.
    scheduler.ec_connector.ensure_cache_available = Mock(return_value=True)
    scheduler.ec_connector.has_cache_item = Mock(return_value=True)

    output = scheduler.schedule()

    # Now the deferred request should be scheduled
    assert request_deferred.request_id in output.num_scheduled_tokens
    assert output.num_scheduled_tokens[request_deferred.request_id] == NUM_TOKENS
    _assert_right_encoder_cache_allocated(scheduler, requests=[request_deferred])
    # EC connector metadata should carry the deferred request's MM data
    _assert_right_ec_connector_metadata(
        output, mm_features_list=request_deferred.mm_features
    )
    # No local encoder compute — all loaded externally
    _assert_right_encoder_inputs(output, expected_total_reqs=0)


@pytest.mark.parametrize(
    "replay_tokens,window_loaded,local_hit",
    [(128, False, True), (128, False, False), (0, False, False), (128, True, False)],
)
def test_ec_connector_waits_for_embeddings_in_prefix_replay(
    replay_tokens, window_loaded, local_hit
):
    """A replayed image must finish loading before its request is admitted."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        enable_prefix_caching=True,
        use_kv_connector=None if local_hit else mock_kv(640, is_async=False),
        use_ec_connector=True,
        ec_role="ec_consumer",
    )
    scheduler.prefix_replay_tokens = replay_tokens
    scheduler.prefix_replay_group_ids = (0,)
    if not local_hit:
        scheduler.connector.get_loaded_kv_cache_group_ids = Mock(
            return_value=(0,) if window_loaded else ()
        )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=819,
        mm_positions=[[PlaceholderRange(offset=383, length=184)]],
    )
    if local_hit:
        (seed,) = create_requests(
            num_requests=1,
            num_tokens=819,
            mm_positions=[[PlaceholderRange(offset=383, length=184)]],
            req_ids=["seed"],
        )
        scheduler.kv_cache_manager.allocate_slots(seed, 640)
        scheduler.kv_cache_manager.cache_blocks(seed, 640)
        scheduler.kv_cache_manager.free(seed)
    loaded = set()

    def ensure_cache_available(request, num_computed_tokens):
        return all(
            feature.mm_position.offset + feature.mm_position.length
            <= num_computed_tokens
            or feature.identifier in loaded
            for feature in request.mm_features
        )

    scheduler.ec_connector.ensure_cache_available = ensure_cache_available
    scheduler.ec_connector.has_cache_item = Mock(return_value=True)
    scheduler.add_request(request)
    output = scheduler.schedule()

    if replay_tokens and not window_loaded:
        assert request.request_id not in output.num_scheduled_tokens
        assert not scheduler.encoder_cache_manager.cached
        loaded.add(request.mm_features[0].identifier)
        output = scheduler.schedule()

    (admitted,) = output.scheduled_new_reqs
    expected_start = 640 - (0 if window_loaded else replay_tokens)
    assert admitted.num_computed_tokens == expected_start
    assert output.num_scheduled_tokens[request.request_id] == 819 - expected_start
    assert not output.scheduled_encoder_inputs
    if replay_tokens and not window_loaded:
        _assert_right_ec_connector_metadata(output, request.mm_features)


def test_ec_connector_defers_running_request_for_async_reload():
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=32,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )
    request = create_requests(
        num_requests=1,
        num_tokens=128,
        mm_positions=[[PlaceholderRange(offset=48, length=32)]],
        req_ids=["request"],
    )[0]
    scheduler.ec_connector.ensure_cache_available = Mock(side_effect=[True, False])

    scheduler.add_request(request)
    first_output = scheduler.schedule()
    assert first_output.num_scheduled_tokens[request.request_id] == 32

    second_output = scheduler.schedule()
    assert request.request_id not in second_output.num_scheduled_tokens
    ensure_call = scheduler.ec_connector.ensure_cache_available.call_args
    assert ensure_call.args[:2] == (request, 32)


def test_ec_connector_pending_prefetch_only_checks_future_mm_features():
    """Test that future mm feature filtering only yields features beyond
    the computed token frontier.

    Features already within num_computed_tokens (past/boundary) must be
    filtered out; only features that extend beyond the frontier (future) should
    be yielded so that connector implementations know which items to prefetch.

    Filter cases:
      "past":     end = 0  + 16 = 16 <  32 → filtered OUT
      "boundary": end = 16 + 16 = 32 == 32 → filtered OUT (condition is >, not >=)
      "future":   end = 48 + 32 = 80 >  32 → yielded
    """
    BLOCK_SIZE = 16
    NUM_COMPUTED_TOKENS = BLOCK_SIZE * 2  # 32
    NUM_TOKENS = BLOCK_SIZE * 8  # 128

    HASH_PAST = "hash_past"
    HASH_BOUNDARY = "hash_boundary"
    HASH_FUTURE = "hash_future"

    request = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS,
        mm_hashes_list=[[HASH_PAST, HASH_BOUNDARY, HASH_FUTURE]],
        mm_positions=[
            [
                PlaceholderRange(offset=0, length=BLOCK_SIZE),  # end=16 (past)
                PlaceholderRange(offset=16, length=BLOCK_SIZE),  # end=32 (boundary)
                PlaceholderRange(offset=48, length=BLOCK_SIZE * 2),  # end=80 (future)
            ]
        ],
        block_size=BLOCK_SIZE,
    )[0]

    future_hashes = [
        f.identifier
        for f in request.mm_features
        if f.mm_position.offset + f.mm_position.length > NUM_COMPUTED_TOKENS
    ]

    assert future_hashes == [HASH_FUTURE], (
        f"Expected only {HASH_FUTURE!r} from future mm feature filtering, "
        f"got {future_hashes!r}. Past/boundary features must be filtered out."
    )


def test_async_load_reservation_prevents_wedge_e2e():
    """Same wedge scenario as PR #40968's lateral-preemption e2e test, but
    resolved by reservation-based admission control instead of preemption.

    A (8 blocks) and B (5 blocks) both want an async KV load, sharing a 4-block
    prefix, in a 10-block pool (9 usable). Admitting both loads would wedge:
    once their recvs finish neither can complete its local prefill (8+5 > 9).

    Here the reservation gate refuses to admit B's load while A's full sequence
    is still reserved, so B never holds blocks and A is free to complete - no
    deadlock, and (unlike lateral preemption) B is never preempted.
    """
    BLOCK_SIZE = 16
    A_TOKENS = BLOCK_SIZE * 8  # bigger request
    B_TOKENS = BLOCK_SIZE * 5  # smaller request
    MATCHED_TOKENS = BLOCK_SIZE * 4  # 4-block prefix loaded for both
    NUM_BLOCKS = 10  # 9 usable; both prefixes fit, but not both full sequences

    scheduler = create_scheduler(
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
        max_num_seqs=4,
        max_num_batched_tokens=A_TOKENS * 2,
        use_kv_connector=mock_kv(matched_tokens=MATCHED_TOKENS, is_async=True),
    )

    [a] = create_requests(
        num_requests=1, num_tokens=A_TOKENS, block_size=BLOCK_SIZE, req_ids=["a"]
    )
    [b] = create_requests(
        num_requests=1, num_tokens=B_TOKENS, block_size=BLOCK_SIZE, req_ids=["b"]
    )
    scheduler.add_request(a)
    scheduler.add_request(b)

    EMPTY_OUTPUT = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    req_to_blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0
    ].req_to_blocks

    # Step 1: A's load is admitted; B's is held back by the reservation (B never
    # holds blocks, so the wedge precondition - both holding prefixes - is gone).
    out1 = scheduler.schedule()
    assert a.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert a.num_computed_tokens == MATCHED_TOKENS
    assert b.status == RequestStatus.WAITING
    assert b.request_id not in req_to_blocks
    assert len(scheduler.running) == 0
    scheduler.update_from_output(out1, EMPTY_OUTPUT)

    # Step 2: nothing changes until A's recv lands.
    out2 = scheduler.schedule()
    assert len(scheduler.running) == 0
    a_finished = dataclasses.replace(
        EMPTY_OUTPUT,
        kv_connector_output=KVConnectorOutput(finished_recving=[a.request_id]),
    )
    scheduler.update_from_output(out2, a_finished)

    # Step 3: A makes forward progress straight to RUNNING - no preemption was
    # needed because B never wedged it.
    out3 = scheduler.schedule()
    assert a.status == RequestStatus.RUNNING
    assert a in scheduler.running
    assert a.request_id in {req.req_id for req in out3.scheduled_new_reqs}
    assert b.status == RequestStatus.WAITING
    assert b.num_preemptions == 0
    assert b.request_id not in req_to_blocks


def test_async_load_admission_reads_parked_reservations_from_cache(monkeypatch):
    """Admitting an async KV load computes only its own reservation, not that of
    every load already parked, which made admissions quadratic in a burst."""
    scheduler = create_scheduler(
        max_num_seqs=128, use_kv_connector=mock_kv(matched_tokens=64, is_async=True)
    )
    for request in create_requests(num_requests=64, num_tokens=80):
        scheduler.add_request(request)
    scheduler.update_from_output(scheduler.schedule(), EMPTY_MODEL_RUNNER_OUTPUT)

    remaining = Mock(wraps=scheduler._request_remaining_blocks)
    monkeypatch.setattr(scheduler, "_request_remaining_blocks", remaining)
    new_ids = [f"new{i}" for i in range(8)]
    for request in create_requests(num_requests=8, num_tokens=80, req_ids=new_ids):
        scheduler.add_request(request)
    scheduler.schedule()

    assert [call.args[0].request_id for call in remaining.call_args_list] == new_ids
    parked = scheduler._parked_reservations
    assert scheduler._parked_reserved_blocks == sum(map(remaining, parked)) > 0


def test_failed_load_reservation_keeps_new_async_loads_out():
    """Promotion frees a failed load's blocks, but it still needs its full sequence.
    A stale reservation would admit a new load into them and overcommit the pool."""
    from tests.v1.kv_connector.unit.utils import create_model_runner_output

    scheduler = create_scheduler(
        num_blocks=16, use_kv_connector=mock_kv(matched_tokens=96, is_async=True)
    )
    scheduler.recompute_kv_load_failures = True
    failed, new = create_requests(2, num_tokens=160, req_ids=["failed", "new"])
    scheduler.add_request(failed)
    output = scheduler.schedule()
    ids = {failed.request_id}
    scheduler.update_from_output(
        output, create_model_runner_output([], finished_recving=ids, failed_recving=ids)
    )

    # The retry waits on a deferred lookup, so the promoted load holds no blocks.
    lookups = {"failed": (None, False), "new": (96, True)}
    scheduler.connector.get_num_new_matched_tokens = lambda r, _: lookups[r.request_id]
    scheduler.schedule()
    assert failed.status == RequestStatus.WAITING
    scheduler.add_request(new)
    scheduler.schedule()
    assert new.status == RequestStatus.WAITING

    lookups["failed"] = (0, False)
    scheduler.schedule()
    assert failed.status == RequestStatus.RUNNING


def _run_async_load_events(spec, num_spec_tokens, seed, on_demand):
    """Random receives, failed loads, invalid blocks and aborts on a tight pool.
    Returns every step's decisions and the events that occurred."""
    from tests.v1.kv_connector.unit.utils import create_model_runner_output

    rng = random.Random(seed)
    s = create_scheduler(
        num_blocks=64,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=True),
        kv_cache_spec=spec,
        num_speculative_tokens=num_spec_tokens,
    )
    s.recompute_kv_load_failures = True
    if on_demand:
        s._reserve_parked = s._inflight_prefills.add
        s._unreserve_parked = lambda _: None

    def match(request, computed):
        if rng.random() < 0.1:
            return None, False
        tokens = min(rng.randrange(12), (request.num_tokens - computed) // 16) * 16
        return (tokens, True) if tokens else (0, False)

    def check_reservations():
        parked = s._parked_reservations
        assert parked == {r: s._request_remaining_blocks(r) for r in parked}
        assert s._parked_reserved_blocks == sum(parked.values())
        if any(r.status != RequestStatus.WAITING_FOR_REMOTE_KVS for r in parked):
            events.add("promoted_load_held")

    s.connector.get_num_new_matched_tokens = match
    group = s.kv_cache_manager.coordinator.single_type_managers[0]
    decisions, events = [], set()
    for step in range(60):
        for i in range(rng.choice((0, 1, 2, 6))):
            tokens, max_tokens = rng.randrange(16, 400), rng.randrange(1, 8)
            ids = [f"{step}.{i}"]
            s.add_request(
                create_requests(
                    1, tokens, max_tokens=max_tokens, req_ids=ids, ignore_eos=True
                )[0]
            )
        output = s.schedule()
        check_reservations()
        requests = sorted(s.requests.items())
        states = [(k, r.status, r.num_computed_tokens) for k, r in requests]
        decisions.append((sorted(output.num_scheduled_tokens.items()), states))
        finished: set[str] = set()
        failed: set[str] = set()
        invalid: set[int] = set()
        for req_id, r in requests:
            if (
                r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
                and req_id not in s.finished_recving_kv_req_ids
            ):
                x, blocks = rng.random(), group.req_to_blocks.get(req_id, ())
                if x < 0.4:
                    (failed if x < 0.15 else finished).add(req_id)
                elif x < 0.5 and blocks and not blocks[0].is_null:
                    invalid.add(blocks[0].block_id)
        events |= {"received"} if finished else set()
        events |= {"failed"} if failed else set()
        events |= {"invalid_blocks"} if invalid else set()
        s.update_from_output(
            output,
            create_model_runner_output(
                [s.requests[i] for i in sorted(output.num_scheduled_tokens)],
                finished_recving=finished | failed,
                failed_recving=failed,
                invalid_block_ids=invalid,
            ),
        )
        check_reservations()
        for req_id in sorted(s.requests):
            if rng.random() < 0.02:
                s.finish_requests(req_id, RequestStatus.FINISHED_ABORTED)
                events.add("aborted")
        check_reservations()
    return decisions, events


@pytest.mark.parametrize(
    "spec,num_spec_tokens",
    [
        (None, None),
        (None, 2),
        (new_sliding_window_spec(sliding_window=64), None),
    ],
    ids=["full", "full_spec_decode", "sliding_window"],
)
def test_cached_reservations_admit_like_on_demand_sum(spec, num_spec_tokens):
    """Caching must not change admission through any async load transition, including
    a scheduled load's leftover draft blocks, which count until the step ends."""
    seen = set()
    for seed in range(8):
        decisions, events = _run_async_load_events(spec, num_spec_tokens, seed, False)
        assert decisions == _run_async_load_events(spec, num_spec_tokens, seed, True)[0]
        seen |= events
    assert seen == {
        "received",
        "failed",
        "invalid_blocks",
        "aborted",
        "promoted_load_held",
    }


def _create_hybrid_mamba_connector_scheduler(
    matched_tokens: int,
    block_size: int = 16,
    num_blocks: int = 100,
    supports_divergent_hits: bool = True,
) -> Scheduler:
    """FA + Mamba ("align" cache mode) scheduler with a MockKVConnector."""
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=8192,
            max_model_len=8192,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=block_size,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="MockKVConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "matched_tokens": matched_tokens,
                "is_async": False,
                "supports_divergent_local_hybrid_hits": supports_divergent_hits,
            },
        ),
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["fa"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    register_all_kvcache_specs(vllm_config)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
        hash_block_size=block_size,
        log_stats=True,
    )


def _seed_hybrid_prefix(
    manager: KVCacheManager, num_blocks: int, block_size: int
) -> tuple[list[int], list[int]]:
    """Prefill a prefix one block per step so that align-mode Mamba caches
    the state at every block boundary. Returns the FA and Mamba block ids."""
    [fill] = create_requests(
        num_requests=1,
        num_tokens=num_blocks * block_size,
        max_tokens=1,
        same_prompt=True,
        block_size=block_size,
        req_ids=["fill"],
    )
    computed_blocks, num_computed, _ = manager.get_computed_blocks(fill)
    assert num_computed == 0
    mamba_ids = []
    for i in range(num_blocks):
        manager.allocate_slots(
            fill, block_size, new_computed_blocks=computed_blocks if i == 0 else None
        )
        fill.num_computed_tokens += block_size
        fa_ids, group_mamba_ids = manager.get_block_ids(fill.request_id)
        mamba_ids.append(group_mamba_ids[i])
    manager.free(fill)
    return fa_ids, mamba_ids


@pytest.mark.parametrize(
    "matched_tokens,expected_num_computed",
    [
        # No external hit: resume on the deepest locally-consistent boundary
        # (block 0's state survives for both groups).
        (0, 16),
        # One external block on top of the reconciled local boundary.
        (16, 32),
    ],
)
def test_hybrid_per_group_hit_divergence_with_connector(
    matched_tokens: int, expected_num_computed: int
):
    """Per-group prefix hits can diverge for hybrid models with a connector
    (#46453): under block pressure the FA prefix tail is evicted while a
    deeper Mamba state block survives. The scheduler must not report the
    deeper hit as locally computed (evicted FA blocks are not resident ->
    engine crash / dirty KV); it falls back to the reconciled boundary that
    every group is consistent at.
    """
    block_size = 16
    scheduler = _create_hybrid_mamba_connector_scheduler(matched_tokens)
    manager = scheduler.kv_cache_manager
    assert isinstance(manager.coordinator, HybridKVCacheCoordinator)

    # Seed a 4-block prefix so both groups cache all four boundaries.
    fa_ids, mamba_ids = _seed_hybrid_prefix(manager, 4, block_size)

    # Evict the FA tail and the middle mamba states; block 0 (both groups)
    # and the deep mamba state at block 3 survive.
    manager.block_pool.evict_blocks({fa_ids[2], fa_ids[3], mamba_ids[1], mamba_ids[2]})

    # A replay of the prefix plus one extra block now sees diverged
    # per-group hits: FA stops at the evicted tail, while the mamba lookup
    # finds the deeper surviving state.
    [replay] = create_requests(
        num_requests=1,
        num_tokens=5 * block_size,
        max_tokens=1,
        same_prompt=True,
        block_size=block_size,
        req_ids=["replay"],
    )
    _, per_group_hits = manager.coordinator.find_longest_cache_hit_per_group(
        replay.block_hashes, replay.num_tokens - 1
    )
    assert per_group_hits == (2 * block_size, 4 * block_size)  # diverged

    scheduler.add_request(replay)
    output = scheduler.schedule()
    [new_req] = output.scheduled_new_reqs
    assert new_req.num_computed_tokens == expected_num_computed


@pytest.mark.parametrize(
    "max_cache_hit,expected_num_computed", [(95, 80), (0, 0)], ids=["kept", "capped"]
)
def test_hybrid_divergent_hit_fallback_is_capped_at_retained_auxiliary_outputs(
    max_cache_hit, expected_num_computed
):
    block_size = 16
    scheduler = _create_hybrid_mamba_connector_scheduler(16)
    manager = scheduler.kv_cache_manager
    _, mamba_ids = _seed_hybrid_prefix(manager, 4, block_size)
    manager.block_pool.evict_blocks({mamba_ids[1], mamba_ids[2], mamba_ids[3]})
    connector = AuxOutputSchedulerConnector(manager, scheduler.connector)
    connector.max_cache_hit = Mock(return_value=max_cache_hit)
    scheduler.aux_output_connector = connector
    [replay] = create_requests(
        num_requests=1,
        num_tokens=6 * block_size,
        max_tokens=1,
        same_prompt=True,
        block_size=block_size,
        req_ids=["replay"],
    )
    scheduler.add_request(replay)

    [new_req] = scheduler.schedule().scheduled_new_reqs

    assert new_req.num_computed_tokens == expected_num_computed


@pytest.mark.parametrize(
    (
        "supports_divergent_local_hybrid_hits",
        "matched_tokens",
        "replay_blocks",
        "expected_num_computed",
    ),
    [
        (True, 0, 5, 16),
        (True, 16, 6, 80),
        (False, 16, 6, 32),
    ],
)
def test_hybrid_fa_deeper_hit_respects_connector_lookup_policy(
    supports_divergent_local_hybrid_hits: bool,
    matched_tokens: int,
    replay_blocks: int,
    expected_num_computed: int,
):
    """A capable connector may restore missing Mamba state at the deeper FA
    boundary. An external miss or an incapable connector uses a locally
    consistent boundary instead.
    """
    block_size = 16
    scheduler = _create_hybrid_mamba_connector_scheduler(
        matched_tokens,
        supports_divergent_hits=supports_divergent_local_hybrid_hits,
    )
    manager = scheduler.kv_cache_manager
    assert isinstance(manager.coordinator, HybridKVCacheCoordinator)

    # Seed a 4-block prefix in both groups.
    _, mamba_ids = _seed_hybrid_prefix(manager, 4, block_size)

    # Keep all FA blocks; evict every mamba state but block 0. FA reaches 4
    # blocks, the mamba hit only reaches 1 -> diverged (FA > Mamba).
    manager.block_pool.evict_blocks({mamba_ids[1], mamba_ids[2], mamba_ids[3]})

    [replay] = create_requests(
        num_requests=1,
        num_tokens=replay_blocks * block_size,
        max_tokens=1,
        same_prompt=True,
        block_size=block_size,
        req_ids=["replay"],
    )
    _, per_group_hits = manager.coordinator.find_longest_cache_hit_per_group(
        replay.block_hashes, replay.num_tokens - 1
    )
    assert per_group_hits == (4 * block_size, 1 * block_size)  # FA deeper

    scheduler.add_request(replay)
    output = scheduler.schedule()
    [new_req] = output.scheduled_new_reqs
    assert new_req.num_computed_tokens == expected_num_computed


def _make_encoder_instance_request(scheduler, text_prefix=8, image_tokens=16):
    """One request whose image sits after some text, as an EPD proxy sends it.

    `create_scheduler` builds a text-only model, so `compute_mm_encoder_budget`
    leaves the encoder cache at zero capacity; give it room so the admitted path
    is reachable.
    """
    ecm = scheduler.encoder_cache_manager
    ecm.cache_size = ecm.num_free_slots = ecm.num_freeable_slots = 8 * image_tokens
    scheduler.max_num_encoder_input_tokens = 8 * image_tokens

    (request,) = create_requests(
        num_requests=1,
        num_tokens=text_prefix + image_tokens + 4,
        # Completion must hinge on the prompt being encoded, not on max_tokens.
        max_tokens=16,
        mm_hashes_list=[["img-hash-0"]],
        mm_positions=[[PlaceholderRange(offset=text_prefix, length=image_tokens)]],
    )
    scheduler.add_request(request)
    return request


def test_encoder_instance_defers_stop_until_prompt_is_consumed():
    """An encoder instance must not finish a request that has not encoded yet.

    When the encoder cache cannot admit the multi-modal item, the scheduler
    schedules only the tokens before it. Finishing that step would hand the
    client a successful completion for a request whose image was never
    encoded, so no embedding is ever published to the EC connector.
    """
    scheduler = create_scheduler(
        max_num_seqs=8,
        max_num_batched_tokens=1024,
        use_ec_connector=True,
        ec_role="ec_producer",
    )
    request = _make_encoder_instance_request(scheduler)
    req_id = request.request_id

    # Encoder budget exhausted: the item cannot be admitted this step.
    scheduler.encoder_cache_manager.can_allocate = lambda *a, **k: False

    output = scheduler.schedule()
    assert output.num_scheduled_tokens[req_id] > 0
    assert not output.scheduled_encoder_inputs.get(req_id)
    assert output.num_scheduled_tokens[req_id] < request.num_prompt_tokens

    scheduler.update_from_output(
        output,
        make_empty_encoder_model_runner_output(output),
    )

    assert not request.is_finished(), (
        "request finished before its image was encoded, so no embedding was "
        "published to the EC connector"
    )
    assert request.num_output_tokens == 0

    # With the budget restored the item is encoded and only then does the
    # request finish.
    del scheduler.encoder_cache_manager.can_allocate  # unshadow the real method

    output = scheduler.schedule()
    assert output.scheduled_encoder_inputs.get(req_id) == [0]

    scheduler.update_from_output(
        output,
        make_empty_encoder_model_runner_output(output),
    )
    assert request.is_finished()


def test_encoder_instance_finishes_request_once_prompt_is_consumed():
    """The unblocked path completes in a single step, without sampling."""
    scheduler = create_scheduler(
        max_num_seqs=8,
        max_num_batched_tokens=1024,
        use_ec_connector=True,
        ec_role="ec_producer",
    )
    request = _make_encoder_instance_request(scheduler)
    req_id = request.request_id

    output = scheduler.schedule()
    assert output.scheduled_encoder_inputs.get(req_id) == [0]
    assert output.num_scheduled_tokens[req_id] == request.num_prompt_tokens

    scheduler.update_from_output(
        output,
        make_empty_encoder_model_runner_output(output),
    )
    assert request.status == RequestStatus.FINISHED_STOPPED
    # The encoder instance publishes an embedding, not tokens.
    assert request.num_output_tokens == 0


@pytest.mark.parametrize("ec_role", ["ec_producer", "ec_consumer"])
def test_encoder_input_skipped_when_connector_already_has_the_item(ec_role: str):
    """Neither role re-encodes what the connector already holds.

    For a consumer the item is loaded; for a producer there is nothing left to
    do at all -- it published that embedding earlier (or a sibling encoder did),
    so a second ViT pass would be pure waste. Reached on any repeat: a second
    chat turn re-sending its image, a sibling encoder behind the proxy's
    round-robin, or a restart that kept the shared storage.

    Pinned because the obvious "fix" for the encoder-instance crash this used to
    cause is to make the producer encode anyway; the crash belongs to the worker
    (an encoder instance must not gather embeddings it never needed), and paying
    for it here would cost every deployment a redundant encode.
    """
    scheduler = create_scheduler(
        max_num_seqs=8,
        max_num_batched_tokens=1024,
        use_ec_connector=True,
        ec_role=ec_role,
    )
    request = _make_encoder_instance_request(scheduler)
    req_id = request.request_id
    scheduler.ec_connector.has_cache_item = lambda *a, **k: True

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[req_id] > 0
    assert not output.scheduled_encoder_inputs.get(req_id)


def test_diffusion_canvas_width_defaults_to_the_served_canvas():
    def req(extra):
        return SimpleNamespace(sampling_params=SimpleNamespace(extra_args=extra))

    assert diffusion_canvas_width(req(None), 64) == 64
    assert diffusion_canvas_width(req({}), 64) == 64
    assert diffusion_canvas_width(req({"diffusion_canvas_length": 16}), 64) == 16
    assert diffusion_canvas_width(SimpleNamespace(sampling_params=None), 64) == 64


def _diffusion_request(req_id: str, extra_args: dict) -> Request:
    (request,) = create_requests(
        num_requests=1, num_tokens=8, max_tokens=64, req_ids=[req_id]
    )
    request.sampling_params.extra_args = extra_args
    return request


@pytest.fixture
def diffusion_model_runner(monkeypatch):
    # These CPU tests only exercise scheduling, not Triton kernels.
    monkeypatch.setattr("vllm.config.vllm.HAS_TRITON", True)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")


def _diffusion_scheduler(async_scheduling: bool = True, **kwargs) -> DiffusionScheduler:
    scheduler_cls = DiffusionAsyncScheduler if async_scheduling else DiffusionScheduler
    scheduler = create_scheduler(
        async_scheduling=async_scheduling,
        diffusion_canvas_length=8,
        scheduler_cls=scheduler_cls,
        **kwargs,
    )
    assert isinstance(scheduler, scheduler_cls)
    return scheduler


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.usefixtures("diffusion_model_runner")
def test_diffusion_scheduler_is_selected_by_default(async_scheduling):
    config = create_scheduler(
        async_scheduling=async_scheduling, diffusion_canvas_length=8
    ).vllm_config.scheduler_config
    assert config.get_scheduler_cls() is (
        DiffusionAsyncScheduler if config.async_scheduling else DiffusionScheduler
    )


@pytest.mark.usefixtures("diffusion_model_runner")
def test_diffusion_scheduler_narrows_the_canvas_per_request():
    scheduler = _diffusion_scheduler()
    wide = _diffusion_request("wide", {})
    narrow = _diffusion_request("narrow", {"diffusion_canvas_length": 4})
    scheduler.add_request(wide)
    scheduler.add_request(narrow)

    # The prefill step lays down the served canvas as placeholders.
    scheduler.schedule()
    output = scheduler.schedule()
    assert output.scheduled_spec_decode_tokens["wide"] == [-1] * 8
    assert output.scheduled_spec_decode_tokens["narrow"] == [-1] * 4
    assert output.num_scheduled_tokens["narrow"] == 4
    assert narrow.num_output_placeholders == 4


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.usefixtures("diffusion_model_runner")
def test_diffusion_scheduler_trims_full_width_worker_drafts(async_scheduling):
    """Padded worker drafts must be narrowed before scheduling."""
    scheduler = _diffusion_scheduler(async_scheduling=async_scheduling)
    wide = _diffusion_request("wide", {})
    narrow = _diffusion_request("narrow", {"diffusion_canvas_length": 4})
    for request in (wide, narrow):
        scheduler.add_request(request)
    prefill = scheduler.schedule()
    _model_output(scheduler, prefill, [[], []])

    tokens = [-1] * 8
    drafts = DraftTokenIds(["wide", "narrow"], [tokens.copy(), tokens.copy()])
    if async_scheduling:
        output = scheduler.schedule()
        scheduler.update_draft_token_ids_in_output(drafts, output)
    else:
        scheduler.update_draft_token_ids(drafts)
        output = scheduler.schedule()

    assert output.scheduled_spec_decode_tokens == {
        "wide": tokens,
        "narrow": tokens[:4],
    }
    assert output.num_scheduled_tokens == {"wide": 8, "narrow": 4}


@pytest.mark.parametrize(
    "sampled, expected_status",
    [
        pytest.param([], RequestStatus.RUNNING, id="denoising"),
        pytest.param([1] * 4, RequestStatus.FINISHED_LENGTH_CAPPED, id="final"),
    ],
)
@pytest.mark.usefixtures("diffusion_model_runner")
def test_sync_diffusion_step_commits_only_emitted_tokens(sampled, expected_status):
    """An empty denoising step reuses the canvas; a completed read emits it."""
    scheduler = _diffusion_scheduler(async_scheduling=False)
    request = _diffusion_request(
        "read", {"diffusion_canvas_length": 4, "diffusion_read_only": True}
    )
    request.max_tokens = request.sampling_params.max_tokens = 4
    scheduler.add_request(request)
    _model_output(scheduler, scheduler.schedule(), [[]])

    scheduler.update_draft_token_ids(
        DraftTokenIds([request.request_id], [list(range(8))])
    )
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {request.request_id: 4}
    _model_output(scheduler, output, [sampled])

    assert request.num_computed_tokens == request.num_prompt_tokens + len(sampled)
    assert list(request.output_token_ids) == sampled
    assert request.status == expected_status


@pytest.mark.usefixtures("diffusion_model_runner")
def test_diffusion_scheduler_defers_a_read_with_every_step_in_flight():
    scheduler = _diffusion_scheduler()
    one = _diffusion_request(
        "one", {"diffusion_read_only": True, "diffusion_max_steps": 1}
    )
    two = _diffusion_request(
        "two", {"diffusion_read_only": True, "diffusion_max_steps": 2}
    )
    # Only reads are deferred. A capped generation keeps its extra async step.
    gen = _diffusion_request("gen", {"diffusion_max_steps": 1})
    for request in (one, two, gen):
        scheduler.add_request(request)

    scheduler.schedule()
    assert set(scheduler.schedule().num_scheduled_tokens) == {"one", "two", "gen"}
    assert one.num_output_placeholders == 8
    assert set(scheduler.schedule().num_scheduled_tokens) == {"two", "gen"}
    assert set(scheduler.schedule().num_scheduled_tokens) == {"gen"}
    # The deferral is renewed every call.
    assert set(scheduler.schedule().num_scheduled_tokens) == {"gen"}


@pytest.mark.usefixtures("diffusion_model_runner")
def test_diffusion_read_deferral_keeps_a_longer_pp_wait():
    scheduler = _diffusion_scheduler(pipeline_parallel_size=3, use_v2_model_runner=True)
    read = _diffusion_request(
        "read", {"diffusion_read_only": True, "diffusion_max_steps": 1}
    )
    scheduler.add_request(read)

    scheduler.schedule()
    assert read.next_decode_eligible_step == 4
    for _ in range(2):
        assert "read" not in scheduler.schedule().num_scheduled_tokens
    assert "read" in scheduler.schedule().num_scheduled_tokens
    assert read.next_decode_eligible_step == 7
    # Deferring this step alone would ask for 6. The PP wait to 7 stands.
    assert "read" not in scheduler.schedule().num_scheduled_tokens
    assert read.next_decode_eligible_step == 7


def _skip_drafts_per_step(
    num_requests: int, num_tokens: int, max_tokens: int, num_steps: int = 2
) -> list[bool]:
    """SchedulerOutput.skip_drafts for each step, with 64-token chunks."""
    scheduler = create_scheduler(num_speculative_tokens=2, max_num_batched_tokens=64)
    # As for a drafter on the V2 model runner (the ngram config reserves no
    # lookahead slots, which disables the check).
    scheduler.can_skip_drafts = True
    for request in create_requests(num_requests, num_tokens, max_tokens=max_tokens):
        scheduler.add_request(request)
    skip_drafts = []
    for _ in range(num_steps):
        output = scheduler.schedule()
        skip_drafts.append(output.skip_drafts)
        req_ids = list(output.num_scheduled_tokens)
        requests = [scheduler.requests[req_id] for req_id in req_ids]
        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[
                [] if r.num_computed_tokens < r.num_tokens else [100] for r in requests
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        scheduler.update_from_output(output, model_runner_output)
    return skip_drafts


def test_skip_drafts_only_when_no_request_can_use_them():
    # A non-final chunk drafts nothing; a final chunk that samples the last of
    # max_tokens finishes the request.
    assert _skip_drafts_per_step(1, num_tokens=100, max_tokens=1) == [True, True]
    # The final chunk's drafts are verified by the next decode step.
    assert _skip_drafts_per_step(1, num_tokens=100, max_tokens=2) == [True, False]
    # One request finishes its prompt next to the other's non-final chunk.
    assert _skip_drafts_per_step(2, num_tokens=40, max_tokens=2, num_steps=1) == [False]


def test_drafts_unused_with_async_placeholders():
    """The predicate must assume that all in-flight drafts are rejected."""
    scheduler = create_scheduler(num_speculative_tokens=2)
    (request,) = create_requests(num_requests=1, num_tokens=16, max_tokens=3)

    # Non-final prefill chunk: no sample, drafts unusable.
    assert scheduler._drafts_unused(request, 0, 8)
    # Final prefill chunk with max_tokens=3: drafts usable.
    assert not scheduler._drafts_unused(request, 0, 16)

    # Sync decode: 2 of 3 output tokens verified; this step must finish.
    request.append_output_token_ids(100)
    request.append_output_token_ids(101)
    assert scheduler._drafts_unused(request, 17, 1)
    # Only 1 verified output: this step may or may not finish.
    (request,) = create_requests(num_requests=1, num_tokens=16, max_tokens=3)
    request.append_output_token_ids(100)
    assert not scheduler._drafts_unused(request, 16, 1)

    # Async decode, 1 verified output and one in-flight step (1 sample +
    # 1 draft placeholder): worst case still reaches 3 = max_tokens.
    request.num_output_placeholders = 2
    assert scheduler._drafts_unused(request, 18, 3)
    # 0 verified outputs: worst case gives only 2 tokens, not guaranteed.
    (request,) = create_requests(num_requests=1, num_tokens=16, max_tokens=3)
    request.num_output_placeholders = 2
    assert not scheduler._drafts_unused(request, 17, 3)


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_cpu_encoder_input_updates_wait_for_completed_prefill(async_scheduling):
    """In-flight chunks cannot release pixels or another occurrence of a hash."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        use_v2_model_runner=True,
        async_scheduling=async_scheduling,
        max_num_batched_tokens=8,
        max_num_seqs=1,
        max_model_len=128,
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=64,
        mm_hashes_list=[["same", "same"]],
        mm_positions=[
            [
                PlaceholderRange(offset=0, length=8),
                PlaceholderRange(offset=32, length=8),
            ]
        ],
    )
    scheduler.add_request(request)
    first = scheduler.schedule()
    second = scheduler.schedule()
    assert first.encoder_input_updates == second.encoder_input_updates == {}
    scheduler.update_from_output(
        first,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    third = scheduler.schedule()
    assert third.encoder_input_updates == {request.request_id: {0: None}}
    assert all(feature.data is not None for feature in request.mm_features)
    assert scheduler.schedule().encoder_input_updates == {}


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_cpu_encoder_input_updates_resume_and_abort(async_scheduling):
    """A real forced preemption reinstalls replay inputs and abort clears state."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        use_v2_model_runner=True,
        async_scheduling=async_scheduling,
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=200,
        mm_positions=[[PlaceholderRange(offset=0, length=100)]],
    )
    scheduler.add_request(request)
    first = scheduler.schedule()
    model_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[1]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(first, model_output)
    second = scheduler.schedule()
    assert second.encoder_input_updates == {request.request_id: {0: None}}
    scheduler.update_from_output(second, model_output)
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    resumed = scheduler.schedule()
    assert resumed.scheduled_new_reqs[0].mm_features[0].data is not None
    assert resumed.encoder_input_updates == {}
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert request.request_id not in scheduler.requests
    assert scheduler.schedule().encoder_input_updates == {}


@pytest.mark.parametrize("prefix_hit", [False, True])
def test_cpu_encoder_input_updates_restore_after_kv_load_failure(prefix_hit):
    """A connector failure rewinds a running request and requires its pixels again."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        use_v2_model_runner=True,
        enable_prefix_caching=prefix_hit,
    )
    warmup, request = create_requests(
        num_requests=2,
        num_tokens=64,
        same_prompt=True,
        mm_hashes_list=[["image"], ["image"]],
        mm_positions=[[PlaceholderRange(offset=0, length=8)]] * 2,
    )
    if prefix_hit:
        scheduler.add_request(warmup)
        warmup_output = scheduler.schedule()
        scheduler.update_from_output(
            warmup_output,
            ModelRunnerOutput(
                req_ids=[warmup.request_id],
                req_id_to_index={warmup.request_id: 0},
                sampled_token_ids=[[1]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        scheduler.finish_requests(warmup.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.add_request(request)
    first = scheduler.schedule()
    assert (first.scheduled_new_reqs[0].mm_features[0].data is None) == prefix_hit
    model_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[1]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(first, model_output)
    second = scheduler.schedule()
    assert second.encoder_input_updates == (
        {} if prefix_hit else {request.request_id: {0: None}}
    )
    # Evict the GPU embedding so the replay must encode, not hit that cache.
    scheduler.reset_encoder_cache()
    invalid_block = scheduler.kv_cache_manager.get_block_ids(request.request_id)[0][0]
    scheduler.update_from_output(
        second,
        dataclasses.replace(
            model_output,
            kv_connector_output=KVConnectorOutput(invalid_block_ids={invalid_block}),
        ),
    )
    assert request.num_computed_tokens == 0
    replay = scheduler.schedule()
    assert replay.scheduled_new_reqs == []
    assert replay.scheduled_encoder_inputs == {request.request_id: [0]}
    assert replay.encoder_input_updates == {
        request.request_id: {0: request.mm_features[0].data},
    }
    scheduler.update_from_output(replay, model_output)
    assert scheduler.schedule().encoder_input_updates == {request.request_id: {0: None}}


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
def test_cpu_encoder_input_updates_preserve_shm_handles(use_v2_model_runner):
    """Only MRV2 tensor payloads participate; SHM ACK descriptors are untouched."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        use_v2_model_runner=use_v2_model_runner,
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=64,
        mm_positions=[
            [
                PlaceholderRange(offset=0, length=8),
                PlaceholderRange(offset=16, length=8),
            ]
        ],
    )
    handle = MultiModalKwargsItem(
        {
            key: MultiModalFieldElem(data=value, field=MultiModalBatchedField())
            for key, value in {
                "address": 0,
                "monotonic_id": 0,
                "signature": [0],
            }.items()
        }
    )
    request.mm_features[0].data = handle
    scheduler.add_request(request)
    first = scheduler.schedule()
    scheduler.update_from_output(
        first,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[1]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert scheduler.schedule().encoder_input_updates == (
        {request.request_id: {1: None}} if use_v2_model_runner else {}
    )
    assert request.mm_features[0].data is handle


def _state_scheduler(*, log_stats=False, num_slots=2):
    from tests.v1.core.utils import create_state_spec, get_kv_cache_configs

    base = create_scheduler(max_num_seqs=2, max_model_len=32)
    base.vllm_config.cache_config.kv_cache_layout = "LBNHC"
    spec = create_state_spec()
    # One page is the null block; the rest are usable slots.
    config = get_kv_cache_configs(
        base.vllm_config, [{"state": spec}], [(num_slots + 1) * spec.page_size_bytes]
    )[0]
    return StateScheduler(
        vllm_config=base.vllm_config,
        kv_cache_config=config,
        block_size=1,
        log_stats=log_stats,
        structured_output_manager=base.structured_output_manager,
    )


def _state_output(step, *, acknowledge=True):
    from copy import deepcopy

    from vllm.v1.kv_cache_interface import StateCompletion

    ids = list(step.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(ids)},
        sampled_token_ids=[[42] for _ in ids],
        state_completion=(
            StateCompletion(
                step.state_step_seq,
                deepcopy(step.state_slot_handles),
                deepcopy(step.state_cleanup_handles),
            )
            if acknowledge
            else None
        ),
    )


def _stream_core(*, num_slots=2):
    core = object.__new__(EngineCore)
    core.scheduler = _state_scheduler(num_slots=num_slots)
    core.request_block_hasher = None
    core.vllm_config = core.scheduler.vllm_config
    core.scheduler.stream_admissions.clock = lambda: 100.0
    return core


def _stream_request(request_id="stream"):
    from vllm.v1.engine import EngineCoreRequest

    # Token 7 is the park token, which state models declare as EOS.
    params = SamplingParams(max_tokens=4)
    params.update_from_generation_config({}, 7)
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[],
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=True,
    )


def _admit_stream(core, request=None):
    return core.admit_stream(request or _stream_request())


def _stream_chunk(core=None, *, closed=False, request_id="stream", tokens=(1, 2, 3)):
    from vllm.v1.engine import StreamAdmissionContext

    payload = _stream_request(request_id)
    payload.prompt_token_ids = list(tokens)
    payload.stream_admission = StreamAdmissionContext(closed)
    return Request.from_engine_core_request(payload, None)


def _pausable_stream_core():
    import queue

    from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState

    core = object.__new__(EngineCoreProc)
    core.__dict__.update(_stream_core().__dict__)
    core.model_executor = Mock()
    core.batch_queue = None
    core.engines_running = False
    core.shutdown_state = EngineShutdownState.RUNNING
    core.reject_new_requests = False
    core.output_queue = queue.Queue()
    core._idle_state_callbacks = []
    return core


@pytest.mark.parametrize("mode", ["abort", "wait"])
def test_stream_admission_pause_refuses_without_slots(mode):
    """A paused engine refuses new streams without remembering the refusal."""
    core = _pausable_stream_core()
    admissions = core.scheduler.stream_admissions
    free_slots = admissions.manager.num_free_slots
    core.pause_generation(mode=mode, clear_cache=False)
    for _ in range(2):
        refused = _admit_stream(core)
        assert not refused.accepted
        assert refused.reason == "engine_paused"
        assert refused.status == "refused"
        assert admissions.manager.num_free_slots == free_slots
        assert not core.scheduler.requests and not core.scheduler.waiting
        assert not admissions.sessions
    assert admissions.refusals["engine_paused"] == 2

    core.resume_scheduler()
    assert _admit_stream(core).accepted
    assert admissions.manager.num_free_slots == free_slots - 1


def test_stream_admission_accepted_replay_survives_wait_pause():
    from vllm.v1.engine import EngineCoreRequestType

    core = _pausable_stream_core()
    admissions = core.scheduler.stream_admissions
    accepted = _admit_stream(core)
    handle = admissions.manager.get("stream")
    deadline = admissions.next_deadline()
    core.pause_generation(mode="wait", clear_cache=False)
    assert _admit_stream(core) == accepted
    assert admissions.manager.get("stream") == handle
    assert admissions.next_deadline() == deadline
    assert not admissions.refusals
    core._handle_client_request(EngineCoreRequestType.ADD, (_stream_chunk(core), 0))
    assert admissions.sessions["stream"].status == "live"
    assert core.output_queue.empty()


def test_stream_admission_paused_add_rejection_preserves_incoming_identity():
    """A tagged ADD absent from the scheduler is rejected without finishing its ID."""
    from vllm.v1.engine import EngineCoreRequestType

    core = _pausable_stream_core()
    admissions = core.scheduler.stream_admissions
    free_slots = admissions.manager.num_free_slots
    core.pause_generation(mode="wait", clear_cache=False)
    request = _stream_chunk(core)
    request.client_index = 7
    core._handle_client_request(EngineCoreRequestType.ADD, (request, 0))

    client, batch = core.output_queue.get_nowait()
    (output,) = batch.outputs
    assert client == 7
    assert output.request_id == request.request_id
    assert output.finish_reason == FinishReason.ERROR
    assert output.error.code == "engine_paused" and output.error.retryable
    assert output.stream_output.context == request.stream_admission
    assert output.stream_output.rejection == "engine_paused"
    assert not batch.finished_requests
    assert core.output_queue.empty()
    assert not core.scheduler.requests and not core.scheduler.waiting
    assert not admissions.sessions
    assert admissions.manager.num_free_slots == free_slots


@pytest.mark.parametrize(
    "fault",
    [
        "allocate",
        "pool",
        "free_queue",
        "enqueue",
        "waiting_queue",
        "insert",
        "request_handle",
        "slot_handle",
        "request_blocks",
        "generation",
        "completion",
        "request_table",
        "session_table",
    ],
)
def test_failed_utility_body_rolls_back_admission(monkeypatch, fault):
    """Every transaction boundary unwinds before utility failure is returned."""
    from vllm.v1.engine import UtilityOutput
    from vllm.v1.engine.core import EngineCoreProc

    core = _stream_core()
    scheduler = core.scheduler
    admissions = scheduler.stream_admissions
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    boundaries = {
        "allocate": (manager, "allocate"),
        "pool": (manager.block_pool, "get_new_blocks"),
        "free_queue": (manager.block_pool.free_block_queue, "popleft_n"),
        "enqueue": (scheduler, "_enqueue_waiting_request"),
        "waiting_queue": (scheduler.kv_holding_waiting, "add_request"),
        "insert": (admissions, "_insert_parked_request"),
    }
    tables = {
        "request_handle": (manager, "_requests"),
        "slot_handle": (manager, "_handles"),
        "request_blocks": (manager, "req_to_blocks"),
        "generation": (manager, "_generations"),
        "completion": (manager, "_completed"),
        "request_table": (scheduler, "requests"),
        "session_table": (admissions, "sessions"),
    }
    before_free = manager.block_pool.free_block_queue.get_all_free_blocks()
    before_generations = manager._generations.copy()

    with monkeypatch.context() as m:
        if fault in boundaries:
            target, name = boundaries[fault]
            original = getattr(target, name)

            def fail_after(*args, **kwargs):
                original(*args, **kwargs)
                raise RuntimeError("injected admission failure")

            m.setattr(target, name, fail_after)
        else:
            target, name = tables[fault]
            original_table = getattr(target, name)

            class FailAfterSet(type(original_table)):  # type: ignore[misc]
                failed = False

                def __setitem__(self, key, value):
                    super().__setitem__(key, value)
                    if not self.failed:
                        self.failed = True
                        raise RuntimeError("injected admission failure")

            table = FailAfterSet()
            if isinstance(original_table, list):
                table.extend(original_table)
            else:
                table.update(original_table)
            m.setattr(target, name, table)
        replies: list[UtilityOutput] = []
        EngineCoreProc._invoke_utility_method(
            "admit_stream",
            lambda: _admit_stream(core),
            UtilityOutput(1),
            replies.append,
        )
        assert not manager._requests and not manager._handles
        assert not manager.req_to_blocks
        assert manager._generations == before_generations
        assert manager.block_pool.free_block_queue.get_all_free_blocks() == before_free
    assert "injected admission failure" in replies[0].failure_message
    assert manager.num_free_slots == 2
    assert manager.get("stream") is None
    assert not scheduler.requests and not scheduler.waiting
    assert not scheduler.kv_holding_waiting and not scheduler.deferred_waiting
    assert scheduler.num_waiting_for_streaming_input == 0
    assert not scheduler.state_retirements and not scheduler.finished_req_ids
    assert not admissions.sessions
    assert _admit_stream(core).accepted


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt_token_ids", [1]),
        ("resumable", False),
        ("stream_admission", "invalid"),
        ("abort_immediately", True),
        ("data_parallel_rank", 1),
    ],
)
def test_stream_admission_validation_recovers(field, value):
    core = _stream_core()
    invalid = _stream_request()
    setattr(invalid, field, value)
    with pytest.raises(ValueError):
        _admit_stream(core, invalid)
    assert not core.scheduler.requests
    assert not core.scheduler.stream_admissions.sessions
    assert _admit_stream(core).accepted


@pytest.mark.parametrize("params", [SamplingParams(n=2), SamplingParams(stop=["end"])])
def test_stream_admission_rejects_unsupported_streaming_params(params):
    core = _stream_core()
    invalid = _stream_request()
    invalid.sampling_params = params
    with pytest.raises(ValueError):
        _admit_stream(core, invalid)
    assert not core.scheduler.requests
    assert _admit_stream(core).accepted


@pytest.mark.parametrize("unsupported", ["structured_outputs", "pooling"])
def test_stream_admission_refuses_structured_outputs_and_pooling(unsupported):
    """Admission requires a plain sampling request."""
    core = _stream_core()
    invalid = _stream_request()
    if unsupported == "pooling":
        invalid.sampling_params = None
        invalid.pooling_params = PoolingParams(task="embed")
    else:
        invalid.sampling_params = SamplingParams(
            structured_outputs=StructuredOutputsParams(regex="[0-9]+")
        )
    free_slots = core.scheduler.stream_admissions.manager.num_free_slots
    with pytest.raises(ValueError):
        _admit_stream(core, invalid)
    assert not core.scheduler.requests
    assert not core.scheduler.stream_admissions.sessions
    assert core.scheduler.stream_admissions.manager.num_free_slots == free_slots
    assert _admit_stream(core).accepted


def test_lost_utility_reply_reclaims_admission():
    """A retry keeps the original deadline; abandonment needs no API release."""
    core = _stream_core()
    scheduler = core.scheduler
    admissions = scheduler.stream_admissions
    first = _admit_stream(core)
    deadline = admissions.next_deadline()
    admissions.clock = lambda: deadline - 1
    assert _admit_stream(core) == first
    assert admissions.next_deadline() == deadline
    admissions.clock = lambda: deadline
    admissions.expire()
    assert admissions.expiries == {"never_fed": 1}
    assert not scheduler.requests and not admissions.sessions
    assert _admit_stream(core, _stream_request("replacement")).accepted


@pytest.mark.parametrize("change", ["structured", "embeds", "mm_offset", "mm_overrun"])
def test_unsupported_stream_input_refused_before_preprocessing(change):
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.v1.engine import StreamAdmissionContext
    from vllm.v1.structured_output import StructuredOutputManager

    core = _stream_core()
    core.mm_receiver_cache = None
    core.prefix_hash_config = None
    manager = StructuredOutputManager(core.scheduler.vllm_config)
    core.structured_output_manager = manager
    assert _admit_stream(core).accepted
    payload = _stream_request()
    payload.prompt_token_ids = [1, 2, 3]
    payload.stream_admission = StreamAdmissionContext()
    if change == "structured":
        payload.sampling_params = SamplingParams(
            structured_outputs=StructuredOutputsParams(regex="[0-9]+")
        )
    elif change == "embeds":
        payload.prompt_embeds = torch.zeros(3, 4)
    elif change == "mm_offset":
        payload.mm_features = _stream_mm_chunk(offset=3).mm_features
    else:
        payload.mm_features = _stream_mm_chunk(offset=0, length=4).mm_features
    with pytest.raises(VLLMValidationError, match="stream input"):
        core.preprocess_add_request(payload)
    if change == "structured":
        # A refused update must not select or initialize a grammar backend.
        assert manager.backend is None and manager.backend_name is None
    assert core.scheduler.stream_admissions.sessions["stream"].status == "never_fed"


def test_admission_refuses_without_mutation_when_full():
    core = _stream_core()
    scheduler = core.scheduler
    assert _admit_stream(core).accepted
    assert _admit_stream(core, _stream_request("second")).accepted
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    before = dict(scheduler.requests), list(scheduler.waiting), manager.num_free_slots
    refused = _admit_stream(core, _stream_request("third"))
    assert not refused.accepted and refused.reason == "capacity"
    assert before == (
        dict(scheduler.requests),
        list(scheduler.waiting),
        manager.num_free_slots,
    )
    assert _admit_stream(core, _stream_request("third")) == refused
    # Never scheduled, so its slot needs no fence and frees at once.
    scheduler.finish_requests("stream", RequestStatus.FINISHED_ABORTED)
    assert _admit_stream(core, _stream_request("third")).accepted


def test_stream_admission_respects_active_limit_with_spare_physical_slots():
    core = _stream_core()
    scheduler = core.scheduler
    scheduler.max_num_active_reqs = 1
    assert _admit_stream(core).accepted
    assert _admit_stream(core, _stream_request("second")).reason == "active_limit"
    scheduler.add_request(_stream_chunk(core))
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"stream": 3}
    scheduler.update_from_output(step, _state_output(step))
    scheduler.finish_requests("stream", RequestStatus.FINISHED_ABORTED)
    cleanup = scheduler.schedule()
    scheduler.update_from_output(cleanup, _state_output(cleanup))
    assert _admit_stream(core, _stream_request("second")).accepted


def test_stream_admission_request_id_conflicts():
    """Only an identical retry of a live session is idempotent."""
    core = _stream_core()
    scheduler = core.scheduler
    admissions = scheduler.stream_admissions
    original = _stream_request()
    accepted = _admit_stream(core, original)
    assert accepted.accepted
    for field, value in [("priority", 1), ("client_index", 1), ("arrival_time", 2.0)]:
        changed = _stream_request()
        setattr(changed, field, value)
        assert _admit_stream(core, changed).reason == "request_id_in_use"
    assert _admit_stream(core, original) == accepted
    assert admissions.refusals == {"request_id_in_use": 3}

    # An ordinary request with the ID is not an admission to replay.
    ordinary = Request.from_engine_core_request(_stream_request("ordinary"), None)
    scheduler.add_request(ordinary)
    assert (
        _admit_stream(core, _stream_request("ordinary")).reason == "request_id_in_use"
    )
    assert scheduler.requests["ordinary"] is ordinary
    assert "ordinary" not in admissions.sessions


def test_stream_closing_input_ends_at_eos_and_retains_slot_until_fence():
    """A resumable input that closes input ends its session at EOS instead of
    parking; the retiring session keeps its ID and slot until the fence."""
    core = _stream_core()
    scheduler = core.scheduler
    admissions = scheduler.stream_admissions
    assert _admit_stream(core).accepted
    scheduler.add_request(_stream_chunk(core, closed=True))
    step = scheduler.schedule()
    output = _state_output(step)
    output.sampled_token_ids = [[7]]
    result = scheduler.update_from_output(step, output)
    (terminal,) = [o for batch in result.values() for o in batch.outputs]
    assert terminal.finish_reason == FinishReason.STOP
    assert terminal.stream_output.context.input_closed
    assert "stream" not in scheduler.requests
    assert not admissions.sessions
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    # The retiring session still owns its ID and its slot.
    assert _admit_stream(core).reason == "request_id_in_use"
    assert manager.num_free_slots == 1
    cleanup = scheduler.schedule()
    scheduler.update_from_output(cleanup, _state_output(cleanup))
    assert manager.num_free_slots == 2
    assert not admissions.sessions and not scheduler.state_retirements
    assert not admissions.expiries
    assert _admit_stream(core, _stream_request("replacement")).accepted


def test_parked_idle_deadline_recovers_and_feed_cancels_never_fed():
    core = _stream_core()
    scheduler = core.scheduler
    admissions = scheduler.stream_admissions
    _admit_stream(core)
    chunk = _stream_chunk(core)
    scheduler.add_request(chunk)
    assert admissions.next_deadline() is None
    step = scheduler.schedule()
    output = _state_output(step)
    output.sampled_token_ids = [[7]]
    scheduler.update_from_output(step, output)
    assert _admit_stream(core).status == "parked"
    deadline = admissions.next_deadline()
    admissions.clock = lambda: deadline
    admissions.expire()
    cleanup = scheduler.schedule()
    scheduler.update_from_output(cleanup, _state_output(cleanup))
    assert admissions.expiries == {"parked_idle": 1}
    assert _admit_stream(core, _stream_request("replacement")).accepted


def test_stream_late_chunk_after_expiry_is_refused():
    """A chunk for an expired stream cannot create a request or touch another."""
    core = _stream_core()
    scheduler = core.scheduler
    admissions = scheduler.stream_admissions
    _admit_stream(core)
    stale = _stream_chunk()
    admissions.clock = lambda: 115.0
    scheduler.add_request(stale)
    assert not scheduler.requests and not admissions.sessions
    _, rejection = scheduler.state_error_outputs[-1]
    assert rejection.request_id == "stream"
    assert rejection.stream_output.rejection == "unknown_admission"
    assert _admit_stream(core, _stream_request("next")).accepted
    scheduler.add_request(stale)
    core.abort_requests(["stream"])
    assert admissions.sessions["next"].status == "never_fed"
    scheduler.add_request(_stream_chunk(request_id="next"))
    assert admissions.sessions["next"].status == "live"
    assert scheduler.schedule().num_scheduled_tokens == {"next": 3}


def test_stream_queued_final_add_and_nonrenewable_drain_deadline():
    core = _stream_core()
    scheduler = core.scheduler
    admissions = scheduler.stream_admissions
    _admit_stream(core)
    scheduler.add_request(_stream_chunk(core, closed=True))
    deadline = admissions.next_deadline()
    admissions.clock = lambda: deadline - 1
    scheduler.add_request(_stream_chunk(core, closed=True))
    final = _stream_chunk(core)
    final.resumable = False
    scheduler.add_request(final)
    assert admissions.next_deadline() == deadline
    step = scheduler.schedule()
    output = _state_output(step)
    output.sampled_token_ids = [[7]]
    scheduler.update_from_output(step, output)
    assert not scheduler.requests and not admissions.sessions
    # The retiring session still owns its ID.
    assert _admit_stream(core).reason == "request_id_in_use"
    cleanup = scheduler.schedule()
    scheduler.update_from_output(cleanup, _state_output(cleanup))
    assert not scheduler.state_retirements
    assert not admissions.expiries


def test_stream_deadline_scheduler_abort_output_and_occupied_counts():
    core = _stream_core()
    core.scheduler = _state_scheduler(log_stats=True)
    scheduler = core.scheduler
    admissions = scheduler.stream_admissions
    admissions.clock = lambda: 100.0
    _admit_stream(core)
    assert admissions.occupied() == {
        "never_fed": 1,
        "live": 0,
        "parked": 0,
        "draining": 0,
        "retiring": 0,
    }
    assert scheduler.make_stats().stream_occupied == admissions.occupied()
    scheduler.add_request(_stream_chunk(core, closed=True))
    assert admissions.occupied()["draining"] == 1
    step = scheduler.schedule()
    scheduler.update_from_output(step, _state_output(step))
    admissions.clock = lambda: 140.0
    cleanup = scheduler.schedule()
    assert admissions.occupied()["retiring"] == 1
    outputs = scheduler.update_from_output(cleanup, _state_output(cleanup))
    assert outputs[0].outputs[0].finish_reason == FinishReason.ABORT
    assert not any(admissions.occupied().values())
    assert scheduler.make_stats().stream_expiries == {"draining": 1}
    assert _admit_stream(core, _stream_request("replacement")).accepted


def test_stream_admission_utility_wire_round_trip(monkeypatch):
    import queue

    import msgspec

    from vllm.v1.engine import (
        EngineCoreOutputs,
        EngineCoreRequestType,
        StreamAdmissionResult,
    )
    from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0")
    core = _stream_core()
    proc = object.__new__(EngineCoreProc)
    proc.scheduler = core.scheduler
    proc.shutdown_state = EngineShutdownState.RUNNING
    proc.output_queue = queue.Queue()
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder()
    for sequence in range(1, 4):
        # As the client sends it: no prompt.
        args = (
            msgspec.structs.replace(
                _stream_request(str(sequence)), prompt_token_ids=None
            ),
        )
        proc._handle_client_request(
            EngineCoreRequestType.UTILITY,
            (0, sequence, "admit_stream", decoder.decode(encoder.encode(args))),
        )
        _, output = proc.output_queue.get_nowait()
        decoded = MsgpackDecoder(EngineCoreOutputs).decode(encoder.encode(output))
        assert decoded.utility_output.failure_message is None
        result = msgspec.convert(
            decoded.utility_output.result.result, type=StreamAdmissionResult
        )
        assert result.accepted == (sequence < 3)
    assert result.reason == "capacity"


@pytest.mark.parametrize("state", [False, True, "overdue"])
def test_idle_engine_wakes_for_stream_deadline_only(state):
    import queue
    from collections import deque

    from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState

    core = _stream_core()
    if state:
        _admit_stream(core)
        admissions = core.scheduler.stream_admissions
        if state == "overdue":
            admissions.clock = lambda: 116.0
    else:
        core.scheduler = create_scheduler(max_num_seqs=2, max_model_len=32)
    proc = object.__new__(EngineCoreProc)
    proc.scheduler = core.scheduler
    proc.shutdown_state = EngineShutdownState.RUNNING
    proc.engines_running = False
    proc.batch_queue = None
    proc.process_input_queue_block = True
    proc._idle_state_callbacks = []
    proc._deferred_inputs = deque()
    proc.aborts_queue = queue.Queue()
    proc.input_queue = Mock()
    proc.input_queue.empty.return_value = True

    def wait_for_input(**kwargs):
        if state:
            assert kwargs["timeout"] == (0.0 if state == "overdue" else 15.0)
            admissions.clock = lambda: 115.0
        else:
            assert kwargs == {"block": True}
        raise queue.Empty

    proc.input_queue.get.side_effect = wait_for_input
    proc._process_input_queue()
    proc.input_queue.get.assert_called_once()
    if state:
        _step_stream_core(core)
        assert not proc.scheduler.requests
        assert _admit_stream(core, _stream_request("replacement")).accepted


def _step_stream_core(core):
    from contextlib import nullcontext

    core.model_executor = Mock()
    core.model_executor.execute_model.side_effect = lambda step, **kwargs: Mock(
        result=lambda: _state_output(step)
    )
    core.capture_iteration_details = lambda step: nullcontext()
    core.log_error_detail = lambda step: nullcontext()
    core._process_aborts_queue = lambda: None
    core._attach_iteration_details = lambda *args: None
    core._weight_version = None
    return core.step()


def test_stream_deadline_inprocess_idle_step_expires_session():
    """An in-process caller has no input-queue timer to expire a parked stream."""
    core = _stream_core()
    assert _admit_stream(core).accepted
    assert not core.scheduler.has_requests()
    core.scheduler.stream_admissions.clock = lambda: 115.0
    outputs, executed = _step_stream_core(core)
    assert not executed
    (output,) = outputs[0].outputs
    assert output.finish_reason == FinishReason.ABORT
    # The session-ending output closes input; nothing was applied.
    assert output.stream_output.context == _seq_context(0, True)
    assert output.stream_output.rejection is None
    assert not core.scheduler.requests
    assert core.scheduler.stream_admissions.expiries == {"never_fed": 1}


def test_abort_inflight_reclaims_slot_and_reopens_admission():
    """Abort fences the slot until state completion, without API release."""
    scheduler = _state_scheduler()
    a, b, c = create_requests(num_requests=3, num_tokens=4)
    for req in (a, b):
        scheduler.add_request(req)
    step = scheduler.schedule()
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    old = manager.get(a.request_id)
    scheduler.finish_requests(a.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.add_request(c)
    assert manager.allocate(c.request_id) is None
    scheduler.update_from_output(step, _state_output(step))
    cleanup = scheduler.schedule()
    assert c.request_id not in cleanup.num_scheduled_tokens
    scheduler.update_from_output(cleanup, _state_output(cleanup))
    replacement = scheduler.schedule()
    assert c.request_id in replacement.num_scheduled_tokens
    new = manager.get(c.request_id)
    assert new.slot == old.slot and new.generation > old.generation
    assert manager.get(b.request_id) == step.state_slot_handles[b.request_id]["state"]


def test_immediate_replacement_without_external_release():
    """A completed request releases through the same scheduler fence."""
    scheduler = _state_scheduler()
    a, b, c = create_requests(num_requests=3, num_tokens=4)
    for req in (a, b):
        scheduler.add_request(req)
    step = scheduler.schedule()
    scheduler.update_from_output(step, _state_output(step))
    scheduler.finish_requests(a.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.add_request(c)
    cleanup = scheduler.schedule()
    assert c.request_id not in cleanup.num_scheduled_tokens
    scheduler.update_from_output(cleanup, _state_output(cleanup))
    assert c.request_id in scheduler.schedule().num_scheduled_tokens


def test_state_preemption_ends_session():
    """State cannot be reconstructed by replaying prompt tokens."""
    scheduler = _state_scheduler()
    (req,) = create_requests(num_requests=1, num_tokens=4)
    scheduler.add_request(req)
    step = scheduler.schedule()
    scheduler.running.remove(req)
    scheduler._preempt_request(req, 0.0)
    assert req.status == RequestStatus.FINISHED_ERROR
    assert req not in scheduler.waiting
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    assert manager.num_free_slots == 1
    scheduler.update_from_output(step, _state_output(step))
    assert manager.num_free_slots == 1
    cleanup = scheduler.schedule()
    scheduler.update_from_output(cleanup, _state_output(cleanup))
    assert manager.num_free_slots == 2


def test_missing_state_ack_quarantines_only_affected_slots():
    """An unrelated later completion cannot release a missing write fence."""
    scheduler = _state_scheduler()
    a, b = create_requests(num_requests=2, num_tokens=4)
    scheduler.add_request(a)
    step = scheduler.schedule()
    scheduler.finish_requests(a.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.update_from_output(step, _state_output(step, acknowledge=False))
    scheduler.add_request(b)
    next_step = scheduler.schedule()
    output = _state_output(next_step)
    # Acknowledging only B does not acknowledge A's cleanup.
    output.state_completion.cleanup_handles.clear()
    scheduler.update_from_output(next_step, output)
    scheduler.finish_requests(b.request_id, RequestStatus.FINISHED_ABORTED)
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    assert manager.num_free_slots == 0
    cleanup = scheduler.schedule()
    output = _state_output(cleanup)
    # Acknowledging only A does not acknowledge B's cleanup.
    del output.state_completion.cleanup_handles[b.request_id]
    scheduler.update_from_output(cleanup, output)
    assert manager.num_free_slots == 1
    assert len(scheduler.state_retirements) == 1
    retry = scheduler.schedule()
    scheduler.update_from_output(retry, _state_output(retry))
    assert manager.num_free_slots == 2


def test_parked_state_retains_slot_until_terminal():
    """Parking preserves ownership; only terminal retirement frees it."""
    scheduler = _state_scheduler()
    (req,) = create_requests(num_requests=1, num_tokens=4)
    req.resumable = True
    scheduler.add_request(req)
    step = scheduler.schedule()
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    handle = manager.get(req.request_id)
    output = _state_output(step)
    output.sampled_token_ids = [[EOS_TOKEN_ID]]
    scheduler.update_from_output(step, output)
    assert req.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert manager.get(req.request_id) == handle
    assert manager.num_free_slots == 1
    scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)
    assert manager.num_free_slots == 1
    cleanup = scheduler.schedule()
    scheduler.update_from_output(cleanup, _state_output(cleanup))
    assert manager.num_free_slots == 2


def test_stale_state_completion_cannot_update_reused_request_id():
    """Late progress from a retired generation cannot change its replacement."""
    scheduler = _state_scheduler()
    (req,) = create_requests(num_requests=1, num_tokens=4)
    scheduler.add_request(req)
    step = scheduler.schedule()
    scheduler.update_from_output(step, _state_output(step))
    scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)
    (replacement,) = create_requests(num_requests=1, num_tokens=4)
    scheduler.add_request(replacement)
    cleanup = scheduler.schedule()
    assert replacement.request_id not in cleanup.num_scheduled_tokens
    scheduler.update_from_output(cleanup, _state_output(cleanup))
    scheduler.schedule()
    scheduler.update_from_output(step, _state_output(step))
    assert replacement.status == RequestStatus.RUNNING
    assert replacement.num_output_tokens == 0


def test_state_stop_output_reclaims_without_external_release():
    scheduler = _state_scheduler()
    (req,) = create_requests(num_requests=1, num_tokens=4, stop_token_ids=[42])
    scheduler.add_request(req)
    step = scheduler.schedule()
    scheduler.update_from_output(step, _state_output(step))
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    assert req.status == RequestStatus.FINISHED_STOPPED
    assert manager.get(req.request_id) is None
    assert manager.num_free_slots == 1
    cleanup = scheduler.schedule()
    scheduler.update_from_output(cleanup, _state_output(cleanup))
    assert manager.num_free_slots == 2


def test_partial_state_ack_quarantines_only_missing_request():
    scheduler = _state_scheduler()
    a, b = create_requests(num_requests=2, num_tokens=4)
    scheduler.add_request(a)
    scheduler.add_request(b)
    step = scheduler.schedule()
    from vllm.v1.kv_cache_interface import StateCompletion

    output = _state_output(step)
    output.state_completion = StateCompletion(
        step.state_step_seq, {b.request_id: step.state_slot_handles[b.request_id]}
    )
    results = scheduler.update_from_output(step, output)
    assert a.status == RequestStatus.FINISHED_ERROR
    assert a.num_output_tokens == 0
    assert b.status == RequestStatus.RUNNING
    assert b.num_output_tokens == 1
    assert any(
        o.request_id == a.request_id and o.finish_reason is not None
        for batch in results.values()
        for o in batch.outputs
    )
    scheduler.finish_requests(b.request_id, RequestStatus.FINISHED_ABORTED)
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    assert manager.num_free_slots == 0
    assert len(scheduler.state_retirements) == 2
    assert not scheduler.deferred_frees


@pytest.mark.parametrize("wrong_ack", ["missing", "generation", "session", "step"])
def test_state_cleanup_recovers_without_new_client_request(wrong_ack):
    from dataclasses import replace

    from vllm.v1.kv_cache_interface import StateCompletion

    scheduler = _state_scheduler()
    (req,) = create_requests(num_requests=1, num_tokens=4)
    scheduler.add_request(req)
    step = scheduler.schedule()
    scheduler.update_from_output(step, _state_output(step))
    scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    assert manager.num_free_slots == 1
    cleanup = scheduler.schedule()
    handle = cleanup.state_cleanup_handles[req.request_id]["state"]
    output = _state_output(cleanup, acknowledge=False)
    handles = {req.request_id: {"state": handle}}
    seq = cleanup.state_step_seq
    if wrong_ack == "generation":
        handles[req.request_id]["state"] = replace(handle, generation=99)
    elif wrong_ack == "session":
        handles = {"other": {"state": handle}}
    elif wrong_ack == "step":
        seq -= 1
    if wrong_ack != "missing":
        output.state_completion = StateCompletion(seq, {}, handles)
    scheduler.update_from_output(cleanup, output)
    assert manager.num_free_slots == 1
    assert scheduler.has_requests()
    retry = scheduler.schedule()
    assert not retry.finished_req_ids
    assert not retry.num_scheduled_tokens
    assert retry.state_cleanup_handles == cleanup.state_cleanup_handles
    scheduler.update_from_output(retry, _state_output(retry))
    assert manager.num_free_slots == 2
    assert not scheduler.has_requests()


def test_state_cleanup_missing_ack_after_retry_fails_without_reuse():
    """A worker that never fences cleanup stops the engine, not polls it."""
    scheduler = _state_scheduler()
    (req,) = create_requests(num_requests=1, num_tokens=4)
    scheduler.add_request(req)
    step = scheduler.schedule()
    scheduler.update_from_output(step, _state_output(step))
    scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)
    cleanup = scheduler.schedule()
    scheduler.update_from_output(cleanup, _state_output(cleanup, acknowledge=False))
    retry = scheduler.schedule()
    with pytest.raises(RuntimeError, match="not acknowledged after cleanup"):
        scheduler.update_from_output(retry, _state_output(retry, acknowledge=False))
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    assert manager.num_free_slots == 1
    assert len(scheduler.state_retirements) == 1


def test_state_cleanup_late_ack_releases_without_token_replay():
    scheduler = _state_scheduler()
    (req,) = create_requests(num_requests=1, num_tokens=4)
    scheduler.add_request(req)
    step = scheduler.schedule()
    scheduler.update_from_output(step, _state_output(step))
    scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)
    cleanup = scheduler.schedule()
    scheduler.update_from_output(cleanup, _state_output(cleanup, acknowledge=False))
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    assert manager.num_free_slots == 1
    scheduler._complete_state_slots(cleanup, _state_output(cleanup))
    assert manager.num_free_slots == 2
    assert req.num_output_tokens == 1


def test_state_slot_never_dispatched_releases_immediately():
    scheduler = _state_scheduler()
    (req,) = create_requests(num_requests=1, num_tokens=4)
    scheduler.add_request(req)
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    assert manager.allocate(req.request_id) is not None
    scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)
    assert manager.num_free_slots == 2
    assert not scheduler.state_retirements


@pytest.mark.parametrize("change", ["untagged", "stop", "structured", "embeds"])
def test_stream_admission_rejected_add_cannot_abort_session(change):
    """A rejected update leaves the session and its deadline untouched."""
    from vllm.sampling_params import StructuredOutputsParams

    core = _stream_core()
    admissions = core.scheduler.stream_admissions
    _admit_stream(core)
    deadline = admissions.next_deadline()
    rejected = _stream_chunk(core)
    rejected.abort_immediately = True
    expected = "unsupported_update"
    if change == "untagged":
        rejected.stream_admission = None
        expected = "identity_mismatch"
    elif change == "stop":
        rejected.sampling_params = SamplingParams(stop=["end"])
    elif change == "embeds":
        rejected.prompt_embeds = torch.zeros(3, 4)
        rejected.prompt_is_token_ids = [True, False, True]
    else:
        rejected.sampling_params = SamplingParams(
            structured_outputs=StructuredOutputsParams(regex="[0-9]+")
        )
    before = core.scheduler.requests["stream"]
    core.add_request(rejected)
    assert core.scheduler.requests["stream"] is before
    assert before.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert not before.streaming_queue
    assert admissions.sessions["stream"].status == "never_fed"
    assert admissions.next_deadline() == deadline
    _, output = core.scheduler.state_error_outputs[-1]
    assert output.stream_output.context == rejected.stream_admission
    assert output.stream_output.rejection == expected


def test_stream_admission_unknown_update_delivers_correlated_error():
    from vllm.v1.engine import EngineCoreOutputs, FinishReason
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    core = _stream_core()
    unknown = _stream_chunk(core)
    unknown.client_index = 7
    core.add_request(unknown)
    assert not core.scheduler.requests
    assert core.scheduler.has_requests()
    step = core.scheduler.schedule()
    outputs = core.scheduler.update_from_output(step, _state_output(step))
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(
        MsgpackEncoder().encode(outputs[7])
    )
    (error,) = decoded.outputs
    assert error.request_id == unknown.request_id
    assert error.stream_output.context == unknown.stream_admission
    assert error.stream_output.rejection == "unknown_admission"
    assert error.finish_reason == FinishReason.ERROR
    assert not core.scheduler.has_requests()


def test_stream_admission_ordinary_preaborted_add_still_aborts():
    core = _stream_core()
    request = create_requests(num_requests=1, num_tokens=4)[0]
    request.abort_immediately = True
    core.add_request(request)
    assert request.request_id not in core.scheduler.requests
    assert request.status == RequestStatus.FINISHED_ABORTED


@pytest.mark.parametrize("identity", ["tagged", "untagged"])
@pytest.mark.parametrize(
    "cause", ["preprocessing_error", "shutdown", "mm_cache_miss", "validation_error"]
)
def test_stream_admission_ingress_rejection_preserves_identity(
    identity, cause, monkeypatch
):
    """A rejected ADD cannot report completion of a reused request ID."""
    import queue
    from copy import deepcopy

    from vllm.exceptions import VLLMValidationError
    from vllm.multimodal.cache import MultiModalCacheMissError
    from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest
    from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
    from vllm.v1.engine.core_client import _with_packed_prompt
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    core = _stream_core()
    context = _stream_chunk(core).stream_admission
    assert _admit_stream(core).accepted
    scheduler = core.scheduler
    replacement = scheduler.requests["stream"]
    sessions_before = deepcopy(scheduler.stream_admissions.sessions)
    finished_before = scheduler.finished_req_ids.copy()
    handle_before = scheduler.stream_admissions.manager.get("stream")

    proc = object.__new__(EngineCoreProc)
    proc.scheduler = scheduler
    proc.vllm_config = scheduler.vllm_config
    proc.output_queue = queue.Queue()
    proc.mm_receiver_cache = None
    proc.request_block_hasher = None
    proc.prefix_hash_config = None
    proc.engine_index = 2
    proc.shutdown_state = EngineShutdownState.REQUESTED
    payload = _stream_request()
    payload.prompt_token_ids = [1]
    payload.client_index = 7
    payload.stream_admission = context if identity != "untagged" else None
    # As a frontend sends it: the prompt packed as an int32 ndarray.
    payload = MsgpackDecoder(EngineCoreRequest).decode(
        MsgpackEncoder().encode(_with_packed_prompt(payload))
    )
    if cause == "preprocessing_error":

        def fail(*args, **kwargs):
            raise ValueError("preprocessing failed")

        monkeypatch.setattr(Request, "from_engine_core_request", fail)
        with pytest.raises(ValueError):
            proc.preprocess_add_request(payload)
        proc._handle_request_preproc_error(payload)
    elif cause == "shutdown":
        request, _ = proc.preprocess_add_request(payload)
        assert proc._reject_add_in_shutdown(request)
    elif cause == "validation_error":
        proc._handle_request_validation_error(
            payload, VLLMValidationError("invalid stream update")
        )
    else:
        proc._handle_mm_cache_miss(payload, MultiModalCacheMissError(["missing"]))

    client, batch = proc.output_queue.get_nowait()
    batch = MsgpackDecoder(EngineCoreOutputs).decode(MsgpackEncoder().encode(batch))
    (output,) = batch.outputs
    assert client == 7
    assert output.request_id == "stream"
    assert output.finish_reason == (
        FinishReason.ABORT if cause == "shutdown" else FinishReason.ERROR
    )
    if identity != "untagged":
        assert output.stream_output.context == context
        assert output.stream_output.rejection == cause
        assert not batch.finished_requests
    else:
        assert output.stream_output is None
        assert set(batch.finished_requests) == {"stream"}
    assert output.mm_cache_miss_hashes == (
        ["missing"] if cause == "mm_cache_miss" else None
    )
    if cause == "mm_cache_miss":
        assert batch.engine_index == 2
    if cause == "validation_error":
        assert output.error.code == "validation_error"
        assert output.error.message == "invalid stream update"
    assert proc.output_queue.empty()
    assert scheduler.requests["stream"] is replacement
    assert replacement.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert not replacement.streaming_queue
    assert scheduler.stream_admissions.sessions == sessions_before
    assert scheduler.stream_admissions.manager.get("stream") == handle_before
    assert scheduler.finished_req_ids == finished_before


def test_stream_admission_abort_outputs_keep_admitted_contexts():
    """Actual terminal outputs retain each session identity and finished ID."""
    import queue

    from vllm.v1.engine.core import EngineCoreProc

    core = _stream_core()
    for request_id, client in [("1", 7), ("2", 8)]:
        payload = _stream_request(request_id)
        payload.client_index = client
        assert _admit_stream(core, payload).accepted
    ordinary = create_requests(num_requests=1, num_tokens=4)[0]
    ordinary.client_index = 7
    core.add_request(ordinary)
    requests = list(core.scheduler.requests.values())
    contexts = {r.request_id: r.stream_admission for r in requests}
    clients = {r.request_id: r.client_index for r in requests}
    proc = object.__new__(EngineCoreProc)
    proc.output_queue = queue.Queue()
    proc._send_abort_outputs(
        core.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
    )
    seen = set()
    while not proc.output_queue.empty():
        client, batch = proc.output_queue.get_nowait()
        assert set(batch.finished_requests) == {o.request_id for o in batch.outputs}
        for output in batch.outputs:
            assert client == clients[output.request_id]
            context = contexts[output.request_id]
            if context is None:
                assert output.stream_output is None
            else:
                # A session-ending output closes input; nothing was applied.
                assert output.stream_output.context == _seq_context(0, True)
                assert output.stream_output.rejection is None
            assert output.finish_reason == FinishReason.ABORT
            seen.add(output.request_id)
    assert seen == set(contexts)


@pytest.mark.parametrize("queued", [False, True])
def test_state_stream_replaces_history_only_at_legal_park(queued):
    core = _stream_core()
    assert _admit_stream(core).accepted
    scheduler = core.scheduler
    chunk = _stream_chunk(core, tokens=[3])
    scheduler.add_request(chunk)
    step = scheduler.schedule()
    handle = step.state_slot_handles["stream"]["state"]
    output = _state_output(step)
    output.sampled_token_ids = [[6]]
    scheduler.update_from_output(step, output)
    step = scheduler.schedule()
    replacement = _stream_chunk(core)
    if queued:
        scheduler.add_request(replacement)
    output = _state_output(step)
    output.sampled_token_ids = [[7]]
    scheduler.update_from_output(step, output)
    if not queued:
        scheduler.add_request(replacement)
    session = scheduler.requests["stream"]
    assert session.prompt_token_ids == [1, 2, 3]
    assert list(session.all_token_ids) == [1, 2, 3]
    assert session.num_computed_tokens == 0
    assert not session.output_token_ids
    assert (
        scheduler.kv_cache_manager.coordinator.state_managers["state"].get("stream")
        == handle
    )
    assert scheduler.num_waiting_for_streaming_input == 0


@pytest.mark.parametrize("queued", [False, True])
def test_state_stream_stop_token_terminates_instead_of_parking(queued):
    """Only EOS parks a state stream; a configured stop token ends it."""
    core = _stream_core()
    assert _admit_stream(core).accepted
    scheduler = core.scheduler
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    chunk = _stream_chunk(core, tokens=[3])
    scheduler.add_request(chunk)
    session = scheduler.requests["stream"]
    session.sampling_params.stop_token_ids = [6]
    free_while_held = manager.num_free_slots
    step = scheduler.schedule()
    old = step.state_slot_handles["stream"]["state"]
    replacement = _stream_chunk(core)
    if queued:
        scheduler.add_request(replacement)
    output = _state_output(step)
    output.sampled_token_ids = [[6]]
    result = scheduler.update_from_output(step, output)
    assert session.status == RequestStatus.FINISHED_STOPPED
    assert session.stop_reason == 6
    assert not session.resumable
    assert list(session.all_token_ids) == [3, 6]
    assert "stream" not in scheduler.requests
    outputs = [o for group in result.values() for o in group.outputs]
    assert [(o.finish_reason, o.stop_reason) for o in outputs] == [
        (FinishReason.STOP, 6)
    ]
    if not queued:
        scheduler.add_request(replacement)
        assert "stream" not in scheduler.requests
    cleanup = scheduler.schedule()
    assert cleanup.state_cleanup_handles["stream"]["state"] == old
    assert manager.num_free_slots == free_while_held
    scheduler.update_from_output(cleanup, _state_output(cleanup))
    assert manager.num_free_slots == free_while_held + 1


@pytest.mark.parametrize("queued", [False, True])
def test_state_stream_replaces_only_after_park_not_length_stop(queued):
    core = _stream_core()
    assert _admit_stream(core).accepted
    assert _admit_stream(core, _stream_request("peer")).accepted
    scheduler = core.scheduler
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    peer_handle = manager.get("peer")
    chunk = _stream_chunk(core, tokens=[3])
    chunk.max_tokens = chunk.sampling_params.max_tokens = 1
    scheduler.add_request(chunk)
    session = scheduler.requests["stream"]
    step = scheduler.schedule()
    old = step.state_slot_handles["stream"]["state"]
    replacement = _stream_chunk(core)
    if queued:
        scheduler.add_request(replacement)
    output = _state_output(step)
    output.sampled_token_ids = [[6]]
    result = scheduler.update_from_output(step, output)
    assert session.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert not session.resumable
    assert "stream" not in scheduler.requests
    assert scheduler.num_waiting_for_streaming_input == 1  # The admitted peer.
    assert [o.finish_reason for group in result.values() for o in group.outputs] == [
        FinishReason.LENGTH
    ]
    if not queued:
        scheduler.add_request(replacement)
        assert "stream" not in scheduler.requests
    assert manager.get("peer") == peer_handle
    assert manager.num_free_slots == 0
    assert not _admit_stream(core, _stream_request("next")).accepted
    cleanup = scheduler.schedule()
    assert cleanup.state_cleanup_handles["stream"]["state"] == old
    assert manager.num_free_slots == 0
    scheduler.update_from_output(cleanup, _state_output(cleanup))
    assert _admit_stream(core, _stream_request("next")).accepted
    new = manager.get("next")
    assert new.slot == old.slot and new.generation > old.generation
    assert manager.get("peer") == peer_handle


class _CustomStateScheduler(StateScheduler):
    pass


class _AsyncStateScheduler(StateScheduler, AsyncScheduler):
    pass


@pytest.mark.parametrize(
    ("scheduler_cls", "expected"),
    [
        (None, StateScheduler),
        (_CustomStateScheduler, _CustomStateScheduler),
        (Scheduler, None),
        (DiffusionScheduler, None),
        (_AsyncStateScheduler, None),
    ],
)
def test_state_layout_selects_state_scheduler(scheduler_cls, expected):
    from vllm.v1.core.sched.state_scheduler import get_scheduler_cls

    scheduler = _state_scheduler()
    scheduler_config = scheduler.vllm_config.scheduler_config
    scheduler_config.scheduler_cls = scheduler_cls
    if expected is None:
        with pytest.raises(ValueError, match="StateScheduler subclass"):
            get_scheduler_cls(scheduler_config, scheduler.kv_cache_config)
    else:
        assert (
            get_scheduler_cls(scheduler_config, scheduler.kv_cache_config) is expected
        )


def test_attention_layout_keeps_configured_scheduler():
    from vllm.v1.core.sched.state_scheduler import get_scheduler_cls

    scheduler = create_scheduler()
    assert (
        get_scheduler_cls(
            scheduler.vllm_config.scheduler_config, scheduler.kv_cache_config
        )
        is Scheduler
    )
    with pytest.raises(ValueError, match="persistent-state KV cache"):
        StateScheduler(
            vllm_config=scheduler.vllm_config,
            kv_cache_config=scheduler.kv_cache_config,
            block_size=16,
            structured_output_manager=scheduler.structured_output_manager,
        )


def test_state_error_only_step_keeps_client_batch_and_stats():
    from vllm.v1.core.sched.interface import AddRequestDisposition
    from vllm.v1.engine import StreamAdmissionContext

    scheduler = _state_scheduler(log_stats=True)
    payload = _stream_request("unknown")
    payload.stream_admission = StreamAdmissionContext()
    payload.client_index = 7
    request = Request.from_engine_core_request(payload, None)
    assert scheduler.add_request(request) is AddRequestDisposition.REJECTED
    step = scheduler.schedule()
    outputs = scheduler.update_from_output(step, _state_output(step))
    assert list(outputs) == [7]
    assert outputs[7].outputs[0].request_id == "unknown"
    assert outputs[7].outputs[0].finish_reason == FinishReason.ERROR
    assert outputs[7].scheduler_stats is not None


def test_tagged_add_cannot_touch_ordinary_request():
    from vllm.v1.engine import StreamAdmissionContext

    core = object.__new__(EngineCore)
    core.scheduler = create_scheduler()
    existing = Request.from_engine_core_request(_stream_request(), None)
    core.add_request(existing)
    payload = _stream_request()
    payload.stream_admission = StreamAdmissionContext()
    tagged = Request.from_engine_core_request(payload, None)
    tagged.abort_immediately = True
    with pytest.raises(ValueError, match="requires persistent state"):
        core.add_request(tagged)
    assert core.scheduler.requests["stream"] is existing
    assert existing.status == RequestStatus.WAITING
    assert not existing.streaming_queue


def test_tagged_update_refused_without_state_scheduler():
    from vllm.exceptions import VLLMValidationError
    from vllm.v1.engine import StreamAdmissionContext

    core = object.__new__(EngineCore)
    core.scheduler = create_scheduler()
    payload = _stream_request()
    payload.stream_admission = StreamAdmissionContext()
    with pytest.raises(VLLMValidationError, match="requires persistent state"):
        core.preprocess_add_request(payload)


def test_state_model_failure_retires_only_matching_live_handle():
    from dataclasses import replace

    scheduler = _state_scheduler()
    requests = create_requests(num_requests=2, num_tokens=1)
    for request in requests:
        scheduler.add_request(request)
    step = scheduler.schedule()
    output = _state_output(step)
    failed_id, survivor_id = [r.request_id for r in requests]
    output.state_completion = replace(
        output.state_completion, failed_req_ids=frozenset({failed_id, "unknown"})
    )
    result = scheduler.update_from_output(step, output)
    assert failed_id not in scheduler.requests
    assert scheduler.requests[survivor_id].output_token_ids
    failures = [
        o
        for values in result.values()
        for o in values.outputs
        if o.request_id == failed_id
    ]
    assert len(failures) == 1 and failures[0].finish_reason == FinishReason.ERROR
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    old = step.state_slot_handles[failed_id]["state"]
    cleanup = scheduler.schedule()
    scheduler.update_from_output(cleanup, _state_output(cleanup))
    replacement = manager.allocate(failed_id)
    assert replacement.generation > old.generation
    scheduler._complete_state_slots(step, output)
    assert manager.get(failed_id) == replacement


def test_admitted_stream_is_not_starved_by_a_request_waiting_for_a_slot():
    """An admitted session already owns its slot, so an older queued request
    that cannot get one must not stall the waiting loop ahead of its input."""
    core = _stream_core(num_slots=1)
    scheduler = core.scheduler
    (blocked,) = create_requests(num_requests=1, num_tokens=4)
    scheduler.add_request(blocked)
    assert _admit_stream(core).accepted
    scheduler.add_request(_stream_chunk(core))
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"stream": 3}
    assert blocked.status == RequestStatus.WAITING


def test_first_stream_input_drains_under_pause_new():
    """A never-fed admission holds a slot, so its first input drains while the
    scheduler admits only requests that already hold KV."""
    core = _stream_core()
    scheduler = core.scheduler
    assert _admit_stream(core).accepted
    scheduler.set_pause_state(PauseState.PAUSED_NEW)
    assert scheduler.get_num_unfinished_requests() == 0
    scheduler.add_request(_stream_chunk(core))
    assert scheduler.get_num_unfinished_requests() == 1
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"stream": 3}


def test_stream_continuation_queued_while_running_keeps_slot_priority():
    """A continuation that arrives mid-turn replaces the prompt at EOS; the
    session must stay ahead of requests that still need a slot."""
    core = _stream_core(num_slots=1)
    scheduler = core.scheduler
    assert _admit_stream(core).accepted
    scheduler.add_request(_stream_chunk(core))
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"stream": 3}
    scheduler.add_request(_stream_chunk(core))
    (blocked,) = create_requests(num_requests=1, num_tokens=4)
    scheduler.add_request(blocked)
    output = _state_output(step)
    output.sampled_token_ids = [[7]]
    scheduler.update_from_output(step, output)
    assert scheduler.requests["stream"].status == RequestStatus.WAITING
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"stream": 3}
    assert blocked.status == RequestStatus.WAITING


def _stream_mm_chunk(offset, length=1):
    chunk = _stream_chunk()
    chunk.mm_features = [
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            mm_position=PlaceholderRange(offset=offset, length=length),
            identifier="image",
            modality="image",
        )
    ]
    return chunk


# Placements a three-token replacement prompt cannot hold: one starts past its
# end, the other starts inside it but runs past the end.
_MISPLACED_MM = pytest.mark.parametrize(
    ("offset", "length"), [(5, 1), (0, 4)], ids=["starts_outside", "overruns_end"]
)


@_MISPLACED_MM
@pytest.mark.parametrize("while_running", [False, True])
def test_invalid_multimodal_update_is_rejected_without_ending_session(
    while_running, offset, length
):
    """A multimodal position outside the replacement prompt is a correlated
    rejection; the session, its slot and its deadline are untouched."""
    from collections import deque

    core = _stream_core()
    scheduler = core.scheduler
    admissions = scheduler.stream_admissions
    assert _admit_stream(core).accepted
    if while_running:
        scheduler.add_request(_stream_chunk(core))
        step = scheduler.schedule()
    record = dataclasses.replace(admissions.sessions["stream"])
    deadline = admissions.next_deadline()
    bad = _stream_mm_chunk(offset, length)
    bad.client_index = 3
    scheduler.add_request(bad)
    client, rejection = scheduler.state_error_outputs[-1]
    assert client == 3
    assert rejection.request_id == "stream"
    assert rejection.finish_reason == FinishReason.ERROR
    assert rejection.stream_output.rejection == "validation_error"
    assert admissions.sessions["stream"] == record
    assert admissions.next_deadline() == deadline
    if while_running:
        assert scheduler.requests["stream"].streaming_queue == deque()
        output = _state_output(step)
        output.sampled_token_ids = [[7]]
        scheduler.update_from_output(step, output)
    session = scheduler.requests["stream"]
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    scheduler.add_request(_stream_chunk(core))
    assert scheduler.schedule().num_scheduled_tokens == {"stream": 3}


def test_encoder_decoder_stream_input_may_outrun_its_prompt():
    """An encoder-decoder item is encoder input at position 0, not a span of
    the decoder prompt, so its length is not bounded by the prompt; only the
    start position is checked."""
    core = _stream_core()
    scheduler = core.scheduler
    scheduler.is_encoder_decoder = True
    assert _admit_stream(core).accepted
    scheduler.add_request(_stream_mm_chunk(offset=0, length=4))
    assert not scheduler.state_error_outputs
    session = scheduler.requests["stream"]
    assert session.status == RequestStatus.WAITING
    assert session.mm_features[0].mm_position.length == 4
    assert scheduler.stream_admissions.sessions["stream"].status == "live"
    bad = _stream_mm_chunk(offset=5)
    scheduler.add_request(bad)
    assert scheduler.state_error_outputs[-1][1].stream_output.rejection == (
        "validation_error"
    )


@_MISPLACED_MM
@pytest.mark.parametrize("while_running", [False, True])
def test_invalid_multimodal_continuation_of_ordinary_stream_is_rejected(
    while_running, offset, length
):
    """A resumable request that was never admitted replaces its prompt the
    same way; a misplaced position is refused before replacement or queueing."""
    scheduler = _state_scheduler()
    (req,) = create_requests(num_requests=1, num_tokens=4)
    req.resumable = True
    scheduler.add_request(req)
    step = scheduler.schedule()
    output = _state_output(step)
    output.sampled_token_ids = [[EOS_TOKEN_ID]]
    if not while_running:
        scheduler.update_from_output(step, output)
        assert req.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    bad = Request(
        request_id=req.request_id,
        prompt_token_ids=[1, 2, 3],
        sampling_params=req.sampling_params,
        pooling_params=None,
        mm_features=_stream_mm_chunk(offset, length).mm_features,
        resumable=True,
    )
    bad.client_index = 3
    scheduler.add_request(bad)
    client, rejection = scheduler.state_error_outputs[-1]
    assert client == 3
    assert rejection.request_id == req.request_id
    assert rejection.finish_reason == FinishReason.ERROR
    assert rejection.stream_output.rejection == "validation_error"
    assert not req.streaming_queue
    if while_running:
        scheduler.update_from_output(step, output)
    assert req.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    good = Request(
        request_id=req.request_id,
        prompt_token_ids=[1, 2, 3],
        sampling_params=req.sampling_params,
        pooling_params=None,
        resumable=True,
    )
    scheduler.add_request(good)
    assert scheduler.schedule().num_scheduled_tokens == {req.request_id: 3}


@_MISPLACED_MM
def test_ordinary_stream_invalid_continuation_rejected_through_engine_core(
    offset, length
):
    """Through preprocessing and EngineCore.add_request, a never-admitted
    stream's misplaced continuation is a correlated rejection: no terminal
    output, the parked session stays live, and the next valid input runs."""
    core = _stream_core()
    core.mm_receiver_cache = None
    core.prefix_hash_config = None
    scheduler = core.scheduler

    def send(payload):
        request, _ = core.preprocess_add_request(payload)
        core.add_request(request)
        return request

    first = _stream_request("plain")
    first.prompt_token_ids = [1, 2, 3]
    first.client_index = 3
    session = send(first)
    assert session.status == RequestStatus.WAITING
    step = scheduler.schedule()
    output = _state_output(step)
    output.sampled_token_ids = [[7]]
    scheduler.update_from_output(step, output)
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ

    bad = _stream_request("plain")
    bad.prompt_token_ids = [1, 2, 3]
    bad.mm_features = _stream_mm_chunk(offset, length).mm_features
    bad.client_index = 3
    send(bad)
    client, rejection = scheduler.state_error_outputs[-1]
    assert client == 3
    assert rejection.request_id == "plain"
    assert rejection.finish_reason == FinishReason.ERROR
    assert rejection.stream_output.rejection == "validation_error"
    assert scheduler.requests["plain"] is session
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert not session.streaming_queue
    assert not scheduler.finished_req_ids

    good = _stream_request("plain")
    good.prompt_token_ids = [4, 5, 6]
    send(good)
    assert scheduler.schedule().num_scheduled_tokens == {"plain": 3}


def _seq_context(update_seq, closed=False):
    from vllm.v1.engine import StreamAdmissionContext

    return StreamAdmissionContext(update_seq=update_seq, input_closed=closed)


def _seq_chunk(update_seq, *, closed=False, resumable=True, tokens=(1, 2, 3)):
    """One numbered input of the admitted stream "stream"; a close when not
    resumable (its placeholder prompt is ignored)."""
    payload = _stream_request()
    payload.prompt_token_ids = list(tokens) if resumable else [0]
    payload.resumable = resumable
    payload.stream_admission = _seq_context(update_seq, closed)
    return Request.from_engine_core_request(payload, None)


def _close_chunk(update_seq):
    return _seq_chunk(update_seq, closed=True, resumable=False)


def _install_state_executor(core, sampled=42, *, acknowledge=True, failed=()):
    """A worker stub: every scheduled request samples `sampled` and the state
    completion reports `failed` requests."""
    from contextlib import nullcontext

    def execute(step, **kwargs):
        output = _state_output(step, acknowledge=acknowledge)
        output.sampled_token_ids = [[sampled] for _ in output.req_ids]
        if failed:
            output.state_completion = dataclasses.replace(
                output.state_completion, failed_req_ids=frozenset(failed)
            )
        return Mock(result=lambda: output)

    core.model_executor = Mock()
    core.model_executor.execute_model.side_effect = execute
    core.capture_iteration_details = lambda step: nullcontext()
    core.log_error_detail = lambda step: nullcontext()
    core._process_aborts_queue = lambda: None
    core._attach_iteration_details = lambda *args: None
    core._weight_version = None


def _step_state_core(core, sampled=42, *, acknowledge=True, failed=()):
    """One EngineCore.step() over `_install_state_executor`."""
    _install_state_executor(core, sampled, acknowledge=acknowledge, failed=failed)
    outputs, _ = core.step()
    return outputs


def _stream_outputs(batches, request_id="stream"):
    return [
        output
        for batch in batches.values()
        for output in batch.outputs
        if output.request_id == request_id
    ]


def _retired_by_client(batches):
    return {
        client: list(batch.retired_streams)
        for client, batch in batches.items()
        if batch.retired_streams
    }


def test_admitted_stream_update_seq_follows_applied_input():
    """An input's context is copied onto the session when it is applied, at
    once or from the queue, and each output carries the context captured
    before stop handling applies the next queued input."""
    core = _stream_core()
    assert _admit_stream(core).accepted
    core.add_request(_seq_chunk(1))
    (output,) = _stream_outputs(_step_state_core(core, 42))
    assert output.stream_output.context == _seq_context(1)
    core.add_request(_seq_chunk(2))
    (eos,) = _stream_outputs(_step_state_core(core, 7))
    assert eos.finish_reason == FinishReason.STOP
    assert eos.stream_output.context == _seq_context(1)
    (output,) = _stream_outputs(_step_state_core(core, 42))
    assert output.stream_output.context == _seq_context(2)
    (eos,) = _stream_outputs(_step_state_core(core, 7))
    assert eos.stream_output.context == _seq_context(2)
    session = core.scheduler.requests["stream"]
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    core.add_request(_seq_chunk(3))
    (output,) = _stream_outputs(_step_state_core(core, 42))
    assert output.stream_output.context == _seq_context(3)


def test_admitted_stream_rejection_carries_incoming_context():
    core = _stream_core()
    assert _admit_stream(core).accepted
    core.add_request(_seq_chunk(1))
    core.add_request(_seq_chunk(2, tokens=()))
    outputs = _stream_outputs(_step_state_core(core, 42))
    rejections = [o for o in outputs if o.stream_output.rejection is not None]
    (rejection,) = rejections
    assert rejection.stream_output.rejection == "empty_update"
    assert rejection.stream_output.context == _seq_context(2)
    (ran,) = [o for o in outputs if o.stream_output.rejection is None]
    assert ran.stream_output.context == _seq_context(1)
    assert core.scheduler.requests["stream"].stream_admission == _seq_context(1)


@pytest.mark.parametrize("state", ["never_fed", "parked"])
def test_admitted_stream_close_on_idle_session_has_one_terminal_output(state):
    """A close with nothing queued ends a parked or never-fed session at once
    with a terminal STOP output that carries no tokens and the close's seq."""
    core = _stream_core()
    assert _admit_stream(core).accepted
    close_seq = 1
    if state == "parked":
        core.add_request(_seq_chunk(1))
        _step_state_core(core, 7)
        assert (
            core.scheduler.requests["stream"].status
            == RequestStatus.WAITING_FOR_STREAMING_REQ
        )
        close_seq = 2
    core.add_request(_close_chunk(close_seq))
    assert "stream" not in core.scheduler.requests
    outputs = []
    for _ in range(3):
        outputs += _stream_outputs(_step_state_core(core))
    (terminal,) = outputs
    assert terminal.finish_reason == FinishReason.STOP
    assert terminal.new_token_ids == []
    assert terminal.stream_output.context == _seq_context(close_seq, True)
    assert terminal.stream_output.rejection is None


def test_admitted_stream_queued_close_drains_accepted_inputs_in_order():
    """A close queued behind a running input runs every accepted input; the
    ending output carries the last input's seq and input_closed=True."""
    core = _stream_core()
    assert _admit_stream(core).accepted
    core.add_request(_seq_chunk(1))
    first = _stream_outputs(_step_state_core(core, 42))
    core.add_request(_seq_chunk(2, tokens=(4, 5)))
    core.add_request(_close_chunk(3))
    boundary = _stream_outputs(_step_state_core(core, 7))
    session = core.scheduler.requests["stream"]
    assert session.prompt_token_ids == [4, 5]
    final = _stream_outputs(_step_state_core(core, 7))
    contexts = [o.stream_output.context for o in first + boundary + final]
    assert contexts == [_seq_context(1), _seq_context(1), _seq_context(2, True)]
    assert boundary[0].finish_reason == FinishReason.STOP
    assert final[0].finish_reason == FinishReason.STOP
    assert final[0].new_token_ids == [7]
    assert "stream" not in core.scheduler.requests
    assert not _stream_outputs(_step_state_core(core))


@pytest.mark.parametrize("queued", [False, True])
def test_admitted_stream_resumable_closing_input_runs_then_ends(queued):
    """A resumable input with input_closed=True runs before the session ends.
    Its outputs without a finish reason carry its context unchanged; an earlier
    input's EOS output does not close; the closing input's EOS ends the session."""
    core = _stream_core()
    assert _admit_stream(core).accepted
    outputs = []
    expected = []
    if queued:
        core.add_request(_seq_chunk(1))
        outputs += _stream_outputs(_step_state_core(core, 42))
        core.add_request(_seq_chunk(2, closed=True))
        outputs += _stream_outputs(_step_state_core(core, 7))
        expected += [(None, _seq_context(1)), (FinishReason.STOP, _seq_context(1))]
        last = 2
    else:
        core.add_request(_seq_chunk(1, closed=True))
        last = 1
    outputs += _stream_outputs(_step_state_core(core, 42))
    assert "stream" in core.scheduler.requests
    outputs += _stream_outputs(_step_state_core(core, 7))
    expected += [
        (None, _seq_context(last, True)),
        (FinishReason.STOP, _seq_context(last, True)),
    ]
    assert [(o.finish_reason, o.stream_output.context) for o in outputs] == expected
    assert "stream" not in core.scheduler.requests


@pytest.mark.parametrize(
    "cause",
    [
        "abort",
        "never_fed_abort",
        "parked_expiry",
        "never_fed_expiry",
        "preemption",
        "stop_token",
        "length",
        "failed_step",
        "worker_failure",
        "draining_expiry",
    ],
)
def test_admitted_stream_every_terminal_output_carries_input_closed(cause):
    """Abort, expiry, preemption, a failed state step and a non-EOS stop all
    end the session with a finish reason and input_closed=True."""
    import queue

    from vllm.v1.engine.core import EngineCoreProc

    core = _stream_core()
    scheduler = core.scheduler
    assert _admit_stream(core).accepted
    seq = 0
    outputs = []
    if not cause.startswith("never_fed"):
        seq = 1
        chunk = _seq_chunk(1, closed=cause == "draining_expiry")
        if cause == "length":
            chunk.max_tokens = chunk.sampling_params.max_tokens = 1
        core.add_request(chunk)
        if cause == "stop_token":
            scheduler.requests["stream"].sampling_params.stop_token_ids = [6]
    if cause in ("abort", "never_fed_abort"):
        if cause == "abort":
            outputs += _stream_outputs(_step_state_core(core, 42))
        proc = object.__new__(EngineCoreProc)
        proc.output_queue = queue.Queue()
        proc._send_abort_outputs(
            scheduler.finish_requests("stream", RequestStatus.FINISHED_ABORTED)
        )
        _, batch = proc.output_queue.get_nowait()
        outputs += batch.outputs
        expected = FinishReason.ABORT
    elif cause.endswith("expiry"):
        if cause == "parked_expiry":
            outputs += _stream_outputs(_step_state_core(core, 7))
        elif cause == "draining_expiry":
            outputs += _stream_outputs(_step_state_core(core, 42))
            assert scheduler.stream_admissions.sessions["stream"].status == "draining"
        scheduler.stream_admissions.clock = lambda: 1000.0
        outputs += _stream_outputs(_step_state_core(core))
        expected = FinishReason.ABORT
    elif cause == "preemption":
        step = scheduler.schedule()
        request = scheduler.requests["stream"]
        scheduler.running.remove(request)
        scheduler._preempt_request(request, 0.0)
        outputs += _stream_outputs(
            scheduler.update_from_output(step, _state_output(step))
        )
        expected = FinishReason.ERROR
    elif cause == "failed_step":
        outputs += _stream_outputs(_step_state_core(core, acknowledge=False))
        expected = FinishReason.ERROR
    elif cause == "worker_failure":
        outputs += _stream_outputs(_step_state_core(core, failed={"stream"}))
        expected = FinishReason.ERROR
    elif cause == "stop_token":
        outputs += _stream_outputs(_step_state_core(core, 6))
        expected = FinishReason.STOP
    else:
        outputs += _stream_outputs(_step_state_core(core, 42))
        expected = FinishReason.LENGTH
    assert "stream" not in scheduler.requests
    *earlier, terminal = outputs
    # Earlier outputs keep their input's context unchanged, including the EOS
    # of a non-closing input; only a closing input's context closes input.
    closing = _seq_context(seq, cause == "draining_expiry")
    assert [o.stream_output.context for o in earlier] == [closing] * len(earlier)
    assert terminal.finish_reason == expected
    assert terminal.stream_output.context == _seq_context(seq, True)


def test_stream_retirement_signal_never_scheduled_release_reported_once():
    """A never-scheduled slot released at once (fence 0) names the stream once,
    for the client that admitted it, even in a batch with no other output."""
    core = _stream_core()
    payload = _stream_request()
    payload.client_index = 3
    assert _admit_stream(core, payload).accepted
    core.abort_requests(["stream"])
    reported = [_retired_by_client(_step_state_core(core)) for _ in range(3)]
    assert [r for r in reported if r] == [{3: ["stream"]}]


def test_stream_retirement_signal_waits_for_fenced_release():
    """A scheduled slot is reported only once the worker completion releases
    it, and an ordinary state request is never reported."""
    core = _stream_core()
    scheduler = core.scheduler
    payload = _stream_request()
    payload.client_index = 3
    assert _admit_stream(core, payload).accepted
    core.add_request(_seq_chunk(1))
    (ordinary,) = create_requests(num_requests=1, num_tokens=4)
    ordinary.client_index = 3
    scheduler.add_request(ordinary)
    step = scheduler.schedule()
    assert set(step.num_scheduled_tokens) == {"stream", ordinary.request_id}
    core.abort_requests(["stream", ordinary.request_id])
    in_flight = scheduler.update_from_output(step, _state_output(step))
    assert not _retired_by_client(in_flight)
    assert scheduler.state_retirements
    reported = [_retired_by_client(_step_state_core(core)) for _ in range(3)]
    assert not scheduler.state_retirements
    assert [r for r in reported if r] == [{3: ["stream"]}]


def test_stream_retirement_signal_quarantined_slot_reported_after_release():
    core = _stream_core()
    scheduler = core.scheduler
    assert _admit_stream(core).accepted
    core.add_request(_seq_chunk(1))
    _step_state_core(core, 42)
    core.abort_requests(["stream"])
    missed = _step_state_core(core, acknowledge=False)
    assert not _retired_by_client(missed)
    assert scheduler.state_retirements
    reported = [_retired_by_client(_step_state_core(core)) for _ in range(3)]
    assert not scheduler.state_retirements
    assert [r for r in reported if r] == [{0: ["stream"]}]


def test_stream_retirement_only_batch_reaches_output_queue():
    """A retirement with nothing else to output is put on EngineCoreProc's
    output queue, for the admitting client, by the ordinary busy-loop steps."""
    import queue
    from collections import deque

    from vllm.v1.engine import EngineCoreRequestType

    proc = _pausable_stream_core()
    payload = _stream_request()
    payload.client_index = 5
    assert _admit_stream(proc, payload).accepted
    _install_state_executor(proc)
    proc.step_fn = proc.step
    proc.check_for_draft_tokens = False
    proc.input_queue = queue.Queue()
    proc.aborts_queue = queue.Queue()
    proc._deferred_inputs = deque()
    proc.process_input_queue_block = False
    proc.input_queue.put_nowait((EngineCoreRequestType.ABORT, ["stream"]))
    for _ in range(3):
        proc._process_input_queue()
        proc._process_engine_step()
    batches = []
    while not proc.output_queue.empty():
        batches.append(proc.output_queue.get_nowait())
    retired = [(client, batch) for client, batch in batches if batch.retired_streams]
    assert [(client, list(b.retired_streams)) for client, b in retired] == [
        (5, ["stream"])
    ]
    ((_, batch),) = retired
    assert batch.outputs == []


@pytest.mark.parametrize("tokens", [[4, 5, 6], [0]], ids=["multi_token", "one_token"])
def test_admitted_stream_frontend_packed_input_is_checked_and_applied(tokens):
    """Frontends send an input's prompt as an int32 ndarray: it passes the
    multimodal placement check and replaces the session prompt, whether
    applied at once to a parked session or dequeued after an EOS."""
    core = _stream_core()
    core.mm_receiver_cache = None
    core.prefix_hash_config = None
    scheduler = core.scheduler
    assert _admit_stream(core).accepted

    def send(update_seq, prompt, mm_features=None):
        payload = _stream_request()
        payload.prompt_token_ids = np.array(prompt, dtype=np.int32)
        payload.mm_features = mm_features
        payload.stream_admission = _seq_context(update_seq)
        request, _ = core.preprocess_add_request(payload)
        core.add_request(request)

    send(1, tokens, _stream_mm_chunk(0).mm_features)
    session = scheduler.requests["stream"]
    assert not scheduler.state_error_outputs
    assert session.prompt_token_ids == tokens
    assert list(session.all_token_ids) == tokens
    assert session.num_prompt_tokens == len(tokens)
    assert session.stream_admission == _seq_context(1)

    session.mm_features = []
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"stream": len(tokens)}
    send(2, tokens[::-1] + [9])
    output = _state_output(step)
    output.sampled_token_ids = [[7]]
    scheduler.update_from_output(step, output)
    assert scheduler.requests["stream"] is session
    assert session.prompt_token_ids == tokens[::-1] + [9]
    assert list(session.all_token_ids) == tokens[::-1] + [9]
    assert session.stream_admission == _seq_context(2)
