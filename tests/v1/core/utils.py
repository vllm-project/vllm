# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from tests.v1.kv_connector.unit.utils import MockKVConfig
from vllm.config import (
    CacheConfig,
    ECTransferConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

EOS_TOKEN_ID = 50256


def mock_kv(matched_tokens: int, is_async: bool):
    return MockKVConfig(matched_tokens=matched_tokens, is_async=is_async)


def create_scheduler(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    enable_chunked_prefill: bool = True,
    enable_prefix_caching: bool = False,
    long_prefill_token_threshold: int = 0,
    disable_chunked_mm_input: bool = False,
    use_kv_connector: None | bool | MockKVConfig = None,
    num_blocks: int = 10000,
    block_size: int = 16,
    max_model_len: int | None = None,
    num_speculative_tokens: int | None = None,
    skip_tokenizer_init: bool = False,
    async_scheduling: bool = False,
    pipeline_parallel_size: int = 1,
    use_ec_connector: bool = False,
    ec_role: str | None = None,
) -> Scheduler | AsyncScheduler:
    """Create scheduler under test.

    Args:
      model: model under test
      max_num_seqs: max sequences to schedule
      max_num_batch_tokens: max num tokens to batch
      enable_prefix_caching: optionally force APC config
                             (True/False) or use default
                             (False)

    Returns:
      {class}`Scheduler` instance
    """
    model_config = ModelConfig(
        model=model,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=skip_tokenizer_init,
    )
    if max_model_len is None:
        max_model_len = max_num_batched_tokens
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        long_prefill_token_threshold=long_prefill_token_threshold,
        disable_chunked_mm_input=disable_chunked_mm_input,
        enable_chunked_prefill=enable_chunked_prefill,
        async_scheduling=async_scheduling,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    # Cache config, optionally force APC
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    kv_transfer_config = None
    if isinstance(use_kv_connector, MockKVConfig):
        kv_transfer_config = KVTransferConfig(
            kv_connector="MockKVConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "matched_tokens": use_kv_connector.matched_tokens,
                "is_async": use_kv_connector.is_async,
            },
        )
    elif use_kv_connector:
        kv_transfer_config = KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": "local_storage"},
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
        parallel_config=ParallelConfig(pipeline_parallel_size=pipeline_parallel_size),
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
    scheduler_cls = AsyncScheduler if async_scheduling else Scheduler
    return scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


_none_hash_initialized = False


def create_requests(
    num_requests: int,
    num_tokens: int = 10,
    mm_hashes_list: list[list[str]] | None = None,
    mm_positions: list[list[PlaceholderRange]] | None = None,
    max_tokens: int = 16,
    stop_token_ids: list[int] | None = None,
    prompt_logprobs: int | None = None,
    same_prompt: bool = False,
    block_size: int = 16,
    req_ids: list[str] | None = None,
) -> list[Request]:
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
        req_ids = [f"{i}" for i in range(num_requests)]

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

        prompt_token_ids = [0] * num_tokens if same_prompt else [i] * num_tokens
        request = Request(
            request_id=req_ids[i],
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            pooling_params=None,
            mm_features=mm_features if mm_features else None,
            eos_token_id=EOS_TOKEN_ID,
            block_hasher=block_hasher,
        )
        requests.append(request)
    return requests
