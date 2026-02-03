# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
import uuid

import pytest

from vllm.config import VllmConfig
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request

from .test_scheduler import create_scheduler_with_priority
from .utils import EOS_TOKEN_ID

pytestmark = pytest.mark.cpu_test


def _create_random_request(
    max_tokens_range: tuple[int, int],
    num_tokens_range: tuple[int, int],
    arrival_time_range: tuple[float, float],
    priority_range: tuple[int, int],
    num_mm_item_range: tuple[int, int],
    vllm_config: VllmConfig,
):
    max_tokens = random.randint(*max_tokens_range)
    num_tokens = random.randint(*num_tokens_range)
    priority = random.randint(*priority_range)
    arrival_time = random.uniform(*arrival_time_range)
    num_mm_item = random.randint(*num_mm_item_range)

    mm_positions: list[PlaceholderRange] = []
    for mm_start in sorted(
        random.sample(range(num_tokens), min(num_mm_item, num_tokens))
    ):
        if mm_start + 10 > num_tokens:
            continue
        mm_positions.append(PlaceholderRange(offset=mm_start, length=10))

    request_id = uuid.uuid4().hex

    sampling_params = SamplingParams(
        ignore_eos=False,
        max_tokens=max_tokens,
    )
    mm_features = []
    for j, position in enumerate(mm_positions):
        identifier = f"{request_id}_hash_{j}"
        mm_feature = MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            mm_position=position,
            identifier=identifier,
            modality="image",
        )
        mm_features.append(mm_feature)

    prompt_token_ids = random.choices(range(100), k=num_tokens)

    caching_hash_fn = get_hash_fn_by_name(
        vllm_config.cache_config.prefix_caching_hash_algo
    )
    init_none_hash(caching_hash_fn)
    block_hasher = get_request_block_hasher(
        vllm_config.cache_config.block_size, caching_hash_fn
    )

    request = Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=mm_features if mm_features else None,
        eos_token_id=EOS_TOKEN_ID,
        arrival_time=arrival_time,
        priority=priority,
        block_hasher=block_hasher,
    )
    return request


def _mock_execute_model(
    scheduler_output: SchedulerOutput, num_output_tokens_range: tuple[int, int]
) -> ModelRunnerOutput:
    request_ids: list[str] = []
    request_ids.extend(req.req_id for req in scheduler_output.scheduled_new_reqs)
    request_ids.extend(scheduler_output.scheduled_cached_reqs.req_ids)
    random.shuffle(request_ids)

    num_output_tokens = [
        random.randint(*num_output_tokens_range) for _ in range(len(request_ids))
    ]
    sampled_token_ids = [
        [random.randint(0, 100) for _ in range(num_tokens)]
        for num_tokens in num_output_tokens
    ]

    return ModelRunnerOutput(
        req_ids=request_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(request_ids)},
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _mock_draft_token_ids(
    scheduler_output: SchedulerOutput,
    num_output_tokens_range: tuple[int, int],
    seen_request_prompt_length: dict[str, int],
) -> DraftTokenIds:
    request_ids: list[str] = []
    sampled_token_ids: list[list[int]] = []
    for request in scheduler_output.scheduled_new_reqs:
        assert request.req_id not in seen_request_prompt_length
        seen_request_prompt_length[request.req_id] = len(request.prompt_token_ids or [])
        if request.num_computed_tokens >= seen_request_prompt_length[request.req_id]:
            num_tokens = random.randint(*num_output_tokens_range)
            request_ids.append(request.req_id)
            sampled_token_ids.append(
                [random.randint(0, 100) for _ in range(num_tokens)]
            )
    for req_id, num_computed_tokens in zip(
        scheduler_output.scheduled_cached_reqs.req_ids,
        scheduler_output.scheduled_cached_reqs.num_computed_tokens,
    ):
        if num_computed_tokens >= seen_request_prompt_length[req_id]:
            num_tokens = random.randint(*num_output_tokens_range)
            request_ids.append(req_id)
            sampled_token_ids.append(
                [random.randint(0, 100) for _ in range(num_tokens)]
            )
    return DraftTokenIds(req_ids=request_ids, draft_token_ids=sampled_token_ids)


def _chech_valid_scheduler_output(
    scheduler_output: SchedulerOutput,
    seen_request_ids: set[str],
    seen_mm_hashes: set[str],
):
    for req in scheduler_output.scheduled_new_reqs:
        assert req.req_id not in seen_request_ids
        seen_request_ids.add(req.req_id)
    for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
        assert req_id in seen_request_ids

    req_ids = set[str]()
    req_ids.update(req.req_id for req in scheduler_output.scheduled_new_reqs)
    req_ids.update(scheduler_output.scheduled_cached_reqs.req_ids)

    assert set(scheduler_output.num_scheduled_tokens.keys()) == req_ids
    assert (
        sum(scheduler_output.num_scheduled_tokens.values())
        == scheduler_output.total_num_scheduled_tokens
    )

    assert set(scheduler_output.scheduled_spec_decode_tokens.keys()) <= req_ids
    assert set(scheduler_output.scheduled_encoder_inputs.keys()) <= req_ids

    for req in scheduler_output.scheduled_new_reqs:
        for mm_feature in req.mm_features:
            seen_mm_hashes.add(mm_feature.identifier)
    for mm_hash in scheduler_output.free_encoder_mm_hashes:
        assert mm_hash in seen_mm_hashes

    assert scheduler_output.finished_req_ids <= seen_request_ids


@pytest.mark.parametrize("enable_prefix_caching", [True, False])
@pytest.mark.parametrize("num_speculative_tokens", [None, 1, 5])
@pytest.mark.parametrize(
    ("max_input_tokens", "max_output_tokens", "max_num_seqs", "num_blocks"),
    [
        # Standard profile
        (5000, 500, 256, 10000),
        # Generation heavy + high max_num_seqs + low num_blocks -> Many preemptions
        (500, 5000, 1024, 1000),
    ],
    ids=["standard", "preemption"],
)
def test_priority_scheduling_blast(
    enable_prefix_caching: bool,
    num_speculative_tokens: int | None,
    max_input_tokens: int,
    max_output_tokens: int,
    max_num_seqs: int,
    num_blocks: int,
):
    random.seed(42)
    seen_request_prompt_length = dict[str, int]()
    seen_request_ids = set[str]()
    seen_mm_hashes = set[str]()

    scheduler = create_scheduler_with_priority(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=enable_prefix_caching,
        num_blocks=num_blocks,
        num_speculative_tokens=num_speculative_tokens,
    )

    num_initial_requests = 10
    for _ in range(num_initial_requests):
        req = _create_random_request(
            max_tokens_range=(1, max_output_tokens),
            num_tokens_range=(1, max_input_tokens),
            arrival_time_range=(0, 1),
            priority_range=(-3, 3),
            num_mm_item_range=(0, 2),
            vllm_config=scheduler.vllm_config,
        )
        scheduler.add_request(req)
    num_initial_requests = 2
    for _ in range(num_initial_requests):
        req = _create_random_request(
            max_tokens_range=(1, max_output_tokens),
            num_tokens_range=(1, max_input_tokens),
            arrival_time_range=(0, 0),
            priority_range=(4, 4),
            num_mm_item_range=(0, 2),
            vllm_config=scheduler.vllm_config,
        )
        scheduler.add_request(req)
    for _ in range(20000):
        if len(scheduler.waiting) == 0:
            num_new_requests = random.randint(0, 2)
            for _ in range(num_new_requests):
                req = _create_random_request(
                    max_tokens_range=(1, max_output_tokens),
                    num_tokens_range=(1, max_input_tokens),
                    arrival_time_range=(0, 1),
                    priority_range=(-3, 3),
                    num_mm_item_range=(0, 2),
                    vllm_config=scheduler.vllm_config,
                )
                scheduler.add_request(req)
        scheduler_output = scheduler.schedule()
        _chech_valid_scheduler_output(
            scheduler_output, seen_request_ids, seen_mm_hashes
        )
        model_output = _mock_execute_model(
            scheduler_output,
            num_output_tokens_range=(1, 1 + (num_speculative_tokens or 0)),
        )
        scheduler.update_from_output(scheduler_output, model_output)
        if num_speculative_tokens is not None:
            scheduler.update_draft_token_ids(
                _mock_draft_token_ids(
                    scheduler_output,
                    (0, num_speculative_tokens),
                    seen_request_prompt_length,
                )
            )
