# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise bounded sessions through the real scheduler admission/output loop."""

import json

import numpy as np
import pytest
import torch

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.engine import FinishReason
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheGroup
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.streaming.retention import StreamingRetentionParams
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def scheduler(tmp_path):
    # Scheduler construction needs a real model config, but never loads weights.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "opt",
                "architectures": ["OPTForCausalLM"],
                "hidden_size": 32,
                "word_embed_proj_dim": 32,
                "ffn_dim": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "vocab_size": 128,
                "max_position_embeddings": 256,
            }
        )
    )
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            enforce_eager=True,
            dtype="float32",
        ),
        device_config=DeviceConfig("cpu"),
        cache_config=CacheConfig(block_size=16, enable_prefix_caching=True),
        scheduler_config=SchedulerConfig(
            max_model_len=256,
            is_encoder_decoder=False,
            max_num_batched_tokens=256,
            max_num_seqs=4,
            async_scheduling=False,
        ),
    )
    config.cache_config.num_gpu_blocks = 128
    init_none_hash(sha256)
    return Scheduler(
        vllm_config=config,
        kv_cache_config=KVCacheConfig(
            num_blocks=128,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroup.from_layers(
                    {
                        "layer": FullAttentionManager.layer_config(
                            AttentionSpec(
                                block_size=16,
                                num_kv_heads=1,
                                head_size=8,
                                dtype=torch.float32,
                            )
                        )
                    }
                )
            ],
        ),
        structured_output_manager=StructuredOutputManager(config),
        block_size=16,
        hash_block_size=16,
    )


def chunk(
    index,
    length,
    *,
    bounded=False,
    max_tokens=1,
    mm_features=None,
    reprefill_threshold=0.7,
    replay_token=None,
    **sampling_kwargs,
):
    extra_args = {}
    if bounded:
        extra_args["streaming_retention"] = StreamingRetentionParams(
            max_session_tokens=80, reprefill_threshold=reprefill_threshold
        )
    if replay_token is not None:
        extra_args["streaming_frame_replay_tokens"] = 1
    return Request(
        request_id="session",
        prompt_token_ids=([] if replay_token is None else [replay_token])
        + [index] * length,
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            output_kind=RequestOutputKind.DELTA,
            extra_args=extra_args or None,
            **sampling_kwargs,
        ),
        pooling_params=None,
        mm_features=mm_features,
        resumable=True,
    )


def finish_step(scheduler, scheduled, *, max_cached_positions=None):
    ids = list(scheduled.num_scheduled_tokens)
    return scheduler.update_from_output(
        scheduled,
        ModelRunnerOutput(
            req_ids=ids,
            sampled_token_ids=[[100] for _ in ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            max_cached_positions=max_cached_positions or {},
        ),
    )


def test_repeated_chunks_stay_bounded_and_consolidation_answers_the_new_chunk(
    scheduler,
):
    session = chunk(1, 16, bounded=True)
    scheduler.add_request(session)
    saw_eviction = saw_reprefill = False
    for index in range(1, 11):
        if index > 1:
            # Continuations may replace sampling params without repeating retention.
            scheduler.add_request(chunk(index, 32, replay_token=100))
        scheduled = scheduler.schedule()
        [new] = scheduled.scheduled_new_reqs
        assert new.is_bounded_session
        saw_eviction |= bool(new.evicted_token_ranges)
        saw_reprefill |= new.is_reprefill
        if new.is_reprefill:
            assert new.num_computed_tokens == 0
            assert all(segment.complete for segment in session.session_history)
            for prior, segment in zip(
                session.session_history, session.session_history[1:]
            ):
                if segment.segment_type == "assistant_text":
                    assert prior.segment_type != "assistant_text"
        [output] = finish_step(scheduler, scheduled)[0].outputs
        assert output.new_token_ids == [100]
        assert output.finish_reason == FinishReason.LENGTH
        assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
        assert session.num_prompt_tokens <= 80
        assert session.prompt_token_ids[:16] == [1] * 16
        assert session.prompt_token_ids[-(16 if index == 1 else 32) :] == [index] * (
            16 if index == 1 else 32
        )
        if index > 1:
            user = session.session_history[-1]
            assert user.segment_type == "user_text"
            assert user.token_range[1] - user.token_range[0] == 32
            if not new.is_reprefill:
                assistant = session.session_history[-2]
                assert assistant.segment_type == "assistant_text"
                assert list(session.all_token_ids[slice(*assistant.token_range)]) == [
                    100
                ]
    assert saw_eviction and saw_reprefill
    scheduler.finish_requests("session", RequestStatus.FINISHED_ABORTED)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 127


@pytest.mark.parametrize("watermark, reprefill", [(110, False), (111, True)])
def test_reprefill_reserves_positions_for_the_next_input_and_reply(
    scheduler, watermark, reprefill
):
    session = chunk(1, 16, bounded=True, reprefill_threshold=0.5)
    scheduler.add_request(session)
    finish_step(
        scheduler,
        scheduler.schedule(),
        max_cached_positions={session.request_id: watermark},
    )
    assert session.max_cached_position == watermark

    # The threshold is 128 positions, including 16 input tokens and one reply.
    scheduler.add_request(chunk(2, 16))
    scheduled = scheduler.schedule()
    [new] = scheduled.scheduled_new_reqs
    assert new.is_reprefill is reprefill
    assert new.num_computed_tokens == (0 if reprefill else 16)
    [output] = finish_step(scheduler, scheduled)[0].outputs
    assert output.new_token_ids == [100]
    scheduler.finish_requests(session.request_id, RequestStatus.FINISHED_ABORTED)


@pytest.mark.parametrize("queued", [False, True])
def test_oversized_chunk_terminates_only_its_session(scheduler, queued):
    scheduler.add_request(chunk(1, 16, bounded=True))
    scheduled = scheduler.schedule()
    if queued:
        scheduler.add_request(chunk(2, 128))
    outputs = finish_step(scheduler, scheduled)[0].outputs
    if not queued:
        scheduler.add_request(chunk(2, 128))
        outputs = finish_step(scheduler, scheduler.schedule())[0].outputs
    assert outputs[-1].finish_reason == FinishReason.ERROR
    assert not scheduler.has_request("session")
    assert scheduler.num_waiting_for_streaming_input == 0
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 127


def test_invalid_initial_budget_returns_a_request_error_and_leaves_engine_usable(
    scheduler,
):
    scheduler.add_request(chunk(1, 128, bounded=True))
    [output] = finish_step(scheduler, scheduler.schedule())[0].outputs
    assert output.finish_reason == FinishReason.ERROR
    assert not scheduler.has_request("session")
    scheduler.add_request(chunk(2, 16, bounded=True))
    [output] = finish_step(scheduler, scheduler.schedule())[0].outputs
    assert output.new_token_ids == [100]


def test_changed_reply_budget_is_rechecked_before_reprefill(scheduler):
    scheduler.add_request(chunk(1, 16, bounded=True))
    finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(chunk(2, 32, max_tokens=120))
    [output] = finish_step(scheduler, scheduler.schedule())[0].outputs
    assert output.finish_reason == FinishReason.ERROR
    assert not scheduler.has_request("session")


@pytest.mark.parametrize("initial", [True, False])
@pytest.mark.parametrize(
    "scoring",
    [
        {"prompt_logprobs": 1},
        {"prompt_logprob_token_ids": np.array([[1], [2]], dtype=np.int32)},
    ],
)
def test_prompt_scoring_returns_a_request_error(scheduler, initial, scoring):
    if not initial:
        scheduler.add_request(chunk(1, 16, bounded=True))
        finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(chunk(2, 16, bounded=initial, **scoring))
    [output] = finish_step(scheduler, scheduler.schedule())[0].outputs
    assert output.finish_reason == FinishReason.ERROR
    assert output.error is not None
    assert output.error.code == "validation_error"
    assert output.error.message == (
        "Bounded streaming sessions do not support prompt scoring"
    )
    assert not scheduler.has_request("session")
    assert scheduler.num_waiting_for_streaming_input == 0
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 127

    scheduler.add_request(chunk(3, 16, bounded=True, logprobs=1))
    [output] = finish_step(scheduler, scheduler.schedule())[0].outputs
    assert output.new_token_ids == [100]
    scheduler.finish_requests("session", RequestStatus.FINISHED_ABORTED)


def test_partial_block_history_is_dropped_by_reprefill_when_the_new_turn_fits(
    scheduler,
):
    session = chunk(1, 16, bounded=True)
    scheduler.add_request(session)
    finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(chunk(2, 7))
    finish_step(scheduler, scheduler.schedule())

    # Seven old tokens contain no complete KV block. The pinned prompt plus
    # this new turn fits exactly, so rebuilding that window can still proceed.
    scheduler.add_request(chunk(3, 64))
    scheduled = scheduler.schedule()
    [new] = scheduled.scheduled_new_reqs
    assert new.is_reprefill
    assert new.num_computed_tokens == 0
    assert not new.evicted_token_ranges
    assert new.position_offset == 0
    assert session.prompt_token_ids == [1] * 16 + [3] * 64
    [output] = finish_step(scheduler, scheduled)[0].outputs
    assert output.new_token_ids == [100]
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert session.num_prompt_tokens == 80


def test_natural_stream_finish_accepts_the_original_retention_sampling_params(
    scheduler,
):
    session = chunk(1, 16, bounded=True)
    scheduler.add_request(session)
    finish_step(scheduler, scheduler.schedule())
    sentinel = Request(
        request_id="session",
        prompt_token_ids=[0],
        sampling_params=session.sampling_params,
        pooling_params=None,
        resumable=False,
    )
    scheduler.add_request(sentinel)
    assert not scheduler.has_request("session")
    assert scheduler.num_waiting_for_streaming_input == 0
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 127


@pytest.mark.parametrize("initial", [True, False])
@pytest.mark.parametrize("shm_descriptor", [False, True])
def test_nonreplayable_images_return_a_session_error(
    scheduler, initial, shm_descriptor
):
    data = None
    if shm_descriptor:
        data = MultiModalKwargsItem(
            {
                name: MultiModalFieldElem(data=value, field=MultiModalBatchedField())
                for name, value in {
                    "address": 1,
                    "monotonic_id": 1,
                    "signature": [1],
                }.items()
            }
        )
    feature = MultiModalFeatureSpec(
        data=data,
        modality="image",
        identifier="frame",
        mm_position=PlaceholderRange(offset=0, length=16),
    )
    if not initial:
        scheduler.add_request(chunk(1, 16, bounded=True))
        finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(chunk(2, 16, bounded=initial, mm_features=[feature]))
    [output] = finish_step(scheduler, scheduler.schedule())[0].outputs
    assert output.finish_reason == FinishReason.ERROR
    assert not scheduler.has_request("session")
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 127
