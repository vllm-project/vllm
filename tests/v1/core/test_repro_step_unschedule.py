# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.core.test_scheduler import create_requests_with_priority
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.multimodal.inputs import PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test


def _out(so, sampled=None):
    ids = list(so.num_scheduled_tokens)
    sampled = sampled or {}
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={r: i for i, r in enumerate(ids)},
        sampled_token_ids=[sampled.get(r, []) for r in ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _req(rid, n, prio, mm=None, max_tokens=16):
    kw = {}
    if mm is not None:
        kw = dict(mm_hashes_list=[[mm[0]]], mm_positions=[[mm[1]]])
    (request,) = create_requests(
        num_requests=1,
        num_tokens=n,
        req_ids=[rid],
        max_tokens=max_tokens,
        ignore_eos=True,
        **kw,
    )
    request.priority = prio
    return request


class WorkerEncoderCache:
    """Model-runner encoder cache for the scheduler-level reproducer."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.present = set()

    def apply(self, scheduler_output):
        self.present -= set(scheduler_output.free_encoder_mm_hashes)
        for request_id, inputs in scheduler_output.scheduled_encoder_inputs.items():
            for i in inputs:
                request = self.scheduler.requests[request_id]
                self.present.add(request.mm_features[i].identifier)
        for request_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            request = self.scheduler.requests[request_id]
            start = request.num_computed_tokens - num_tokens
            for feature in request.mm_features:
                lo = feature.mm_position.offset
                hi = lo + feature.mm_position.length
                if (
                    lo < start + num_tokens
                    and start < hi
                    and feature.identifier not in self.present
                ):
                    raise RuntimeError(
                        f"Encoder cache miss for {feature.identifier} "
                        f"(req {request_id}, tokens [{start},{start + num_tokens}))"
                    )


def test_encoder_cache_is_not_evicted_for_a_scheduled_request():
    scheduler = create_scheduler(
        scheduling_policy="priority",
        max_num_seqs=4,
        max_num_batched_tokens=64,
        max_model_len=256,
        long_prefill_token_threshold=32,
        num_blocks=9,
    )
    scheduler.max_num_encoder_input_tokens = 64
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=64)
    worker_cache = WorkerEncoderCache(scheduler)

    def step(sampled=None):
        scheduler_output = scheduler.schedule()
        worker_cache.apply(scheduler_output)
        scheduler.update_from_output(scheduler_output, _out(scheduler_output, sampled))
        return scheduler_output

    scheduler.add_request(_req("low", 112, prio=10, mm=("H", PlaceholderRange(96, 16))))
    step()
    scheduler.add_request(_req("high", 16, prio=0, max_tokens=4))
    step({"high": [7]})
    scheduler_output = step({"high": [7]})
    assert "low" in scheduler_output.preempted_req_ids

    scheduler.add_request(_req("other", 32, prio=0, mm=("H", PlaceholderRange(0, 16))))
    step()


def test_encoder_cache_is_not_evicted_when_shared_in_one_step():
    scheduler = create_scheduler(
        scheduling_policy="priority",
        max_num_seqs=8,
        max_num_batched_tokens=96,
        max_model_len=256,
        long_prefill_token_threshold=32,
    )
    scheduler.max_num_encoder_input_tokens = 64
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=64)
    worker_cache = WorkerEncoderCache(scheduler)

    def step(sampled=None):
        scheduler_output = scheduler.schedule()
        worker_cache.apply(scheduler_output)
        scheduler.update_from_output(scheduler_output, _out(scheduler_output, sampled))

    scheduler.add_request(_req("d", 16, prio=1, max_tokens=1))
    scheduler.add_request(_req("low", 128, prio=10, mm=("H", PlaceholderRange(64, 16))))
    step({"d": [7]})
    scheduler.add_request(_req("mid", 48, prio=5, mm=("H", PlaceholderRange(32, 16))))
    scheduler.add_request(_req("r3", 16, prio=7))
    step({"r3": [7]})

    original_allocate_slots = scheduler.kv_cache_manager.allocate_slots
    failed = []

    def allocate_slots(request, *args, **kwargs):
        if request.request_id == "r3" and not failed:
            failed.append(True)
            return None
        return original_allocate_slots(request, *args, **kwargs)

    scheduler.kv_cache_manager.allocate_slots = allocate_slots
    scheduler_output = scheduler.schedule()
    assert "low" in scheduler_output.preempted_req_ids
    worker_cache.apply(scheduler_output)


def test_prefix_cache_does_not_hash_unwritten_blocks():
    scheduler = create_scheduler(
        scheduling_policy="priority",
        max_num_seqs=4,
        max_num_batched_tokens=64,
        long_prefill_token_threshold=32,
        enable_prefix_caching=True,
        num_blocks=11,
        block_size=16,
    )
    (first,) = create_requests_with_priority(1, [9], [0.0], num_tokens=160)
    scheduler.add_request(first)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, _out(scheduler_output))
    (second,) = create_requests_with_priority(
        1, [0], [1.0], num_tokens=32, starting_idx=1
    )
    scheduler.add_request(second)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output, _out(scheduler_output, {second.request_id: [100]})
    )
    scheduler_output = scheduler.schedule()
    assert first.request_id in scheduler_output.preempted_req_ids

    (sibling,) = create_requests_with_priority(1, [5], [2.0], num_tokens=160)
    hit = scheduler.kv_cache_manager.get_computed_blocks(sibling)[1]
    assert hit <= 96, f"sibling hits {hit} tokens, only 96 were ever computed"


def test_mamba_checkpoint_does_not_include_unwritten_tokens():
    base_scheduler = create_scheduler(
        scheduling_policy="priority",
        max_num_seqs=4,
        max_num_batched_tokens=64,
        max_model_len=512,
        long_prefill_token_threshold=16,
        enable_prefix_caching=True,
        num_blocks=14,
    )
    config = base_scheduler.vllm_config
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.mamba_block_size = 16
    kv_cache_config = KVCacheConfig(
        num_blocks=14,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attn"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=2,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["ssm"],
                MambaSpec(
                    block_size=16,
                    shapes=((4, 4),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_prefill_checkpoint_blocks=0,
                ),
            ),
        ],
    )
    scheduler = type(base_scheduler)(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=16,
        hash_block_size=16,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )
    init_none_hash(sha256)

    def request(request_id, num_tokens, offset=0, priority=0):
        return Request(
            request_id=request_id,
            prompt_token_ids=list(range(offset, offset + num_tokens)),
            sampling_params=SamplingParams(max_tokens=128, ignore_eos=True),
            pooling_params=None,
            priority=priority,
            block_hasher=get_request_block_hasher(16, sha256),
        )

    assert scheduler.kv_cache_manager.allocate_slots(request("owner", 65), 64)
    scheduler.add_request(request("low", 49, priority=10))
    scheduler.max_num_scheduled_tokens = 16
    scheduler_output = scheduler.schedule()
    scheduler.max_num_scheduled_tokens = 64
    scheduler.update_from_output(scheduler_output, _out(scheduler_output))
    scheduler.add_request(request("high", 33, offset=1000, priority=0))
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, _out(scheduler_output))
    scheduler_output = scheduler.schedule()
    assert "low" in scheduler_output.preempted_req_ids

    hit = scheduler.kv_cache_manager.get_computed_blocks(request("consumer", 49))[1]
    assert hit <= 32, f"consumer hits {hit} tokens, low only computed 32"


def test_priority_running_requests_are_scheduled_in_order():
    scheduler = create_scheduler(
        scheduling_policy="priority",
        max_num_seqs=4,
        max_num_batched_tokens=64,
        max_model_len=1024,
        long_prefill_token_threshold=48,
    )
    scheduler.add_request(_req("low", 512, prio=10))
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, _out(scheduler_output))
    scheduler.add_request(_req("high", 512, prio=0))
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, _out(scheduler_output))

    for _ in range(3):
        scheduler_output = scheduler.schedule()
        scheduler.update_from_output(scheduler_output, _out(scheduler_output))
        scheduled_tokens = scheduler_output.num_scheduled_tokens
        assert scheduled_tokens["high"] >= scheduled_tokens["low"], (
            f"high (prio 0) got {scheduled_tokens['high']} tokens, "
            f"low (prio 10) got {scheduled_tokens['low']}"
        )
