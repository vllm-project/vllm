# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import tempfile
from collections import defaultdict
from collections.abc import Callable
from itertools import count
from typing import Any

import torch

from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector import (  # noqa
    SharedStorageConnector,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

EOS_TOKEN_ID = 50256


def assert_scheduler_empty(scheduler: Scheduler):
    """Confirm the scheduler is "empty" - i.e. no leaks."""
    # Scheduler Metadata.
    assert len(scheduler.requests) == 0
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 0
    assert len(scheduler.finished_req_ids) == 0
    assert len(scheduler.finished_recving_kv_req_ids) == 0

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


def create_vllm_config(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 64,
    block_size: int = 16,
    max_model_len: int = 10000,
    enable_chunked_prefill: bool = True,
    enable_permute_local_kv: bool = False,
) -> VllmConfig:
    """Initialize VllmConfig For Testing."""
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        enable_chunked_prefill=enable_chunked_prefill,
        # Disable hybrid KV cache manager for testing
        # Should be removed after we support hybrid KV cache manager-based testing.
        disable_hybrid_kv_cache_manager=True,
    )
    model_config = ModelConfig(
        model=model,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    # Cache config, optionally force APC
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_both",
        enable_permute_local_kv=enable_permute_local_kv,
    )
    return VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu"),
    )


def create_scheduler(
    vllm_config: VllmConfig,
    num_blocks: int = 10000,
) -> Scheduler:
    """Initialize Scheduler For Testing."""
    block_size = vllm_config.cache_config.block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,  # A large number of blocks to hold all requests
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"], FullAttentionSpec(block_size, 1, 1, torch.float32, False)
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
    )


_request_count = count(1)
_none_hash_initialized = False


def create_request(
    request_id: int | None = None,
    num_tokens: int = 10,
    common_prefix_len=0,
    max_tokens: int = 16,
    do_remote_decode: bool = False,
    do_remote_prefill: bool = False,
    num_remote_blocks: int = 3,
    block_size: int = 16,
    hash_fn: Callable = sha256,
) -> Request:
    """Make dummy request for testing."""
    assert num_tokens >= common_prefix_len >= 0

    if request_id is None:
        request_id = next(_request_count)

    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(hash_fn)
        _none_hash_initialized = True

    kv_transfer_params: dict[str, Any] | None = None

    if do_remote_decode:
        assert not do_remote_prefill
        kv_transfer_params = dict(do_remote_prefill=False, do_remote_decode=True)
    elif do_remote_prefill:
        kv_transfer_params = dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_engine_id="my-engine-id",
            remote_block_ids=list(range(num_remote_blocks)),
            remote_host="my-host",
            remote_port=1234,
        )

    max_tokens = 1 if do_remote_decode else max_tokens
    sampling_params = SamplingParams(max_tokens=max_tokens)

    common_prefix = [1] * common_prefix_len if common_prefix_len > 0 else []
    suffix = [i * request_id for i in range(num_tokens - common_prefix_len)]
    prompt_token_ids = common_prefix + suffix

    req = Request(
        request_id=f"id-{request_id}",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=None,
        eos_token_id=EOS_TOKEN_ID,
        block_hasher=get_request_block_hasher(block_size, hash_fn),
    )
    req.kv_transfer_params = kv_transfer_params
    return req


def create_model_runner_output(
    reqs: list[Request],
    finished_sending: set[str] | None = None,
    finished_recving: set[str] | None = None,
    invalid_block_ids: set[int] | None = None,
    use_eos: bool = False,
    token_id: int = 0,
) -> ModelRunnerOutput:
    """Make dummy model runner output for testing."""

    # Make request data.
    req_ids = [req.request_id for req in reqs]
    req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

    # Make sampled tokens.
    sampled_token = EOS_TOKEN_ID if use_eos else token_id
    sampled_token_ids = [[sampled_token] for _ in req_ids]

    kv_connector_output = (
        None
        if (
            finished_sending is None
            and finished_recving is None
            and invalid_block_ids is None
        )
        else KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            invalid_block_ids=invalid_block_ids or set(),
        )
    )

    # Make output data structure.
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        kv_connector_output=kv_connector_output,
    )


class TestSharedStorageConnector(SharedStorageConnector):
    def __init__(self, config: VllmConfig, role, kv_cache_config):
        self.name = config.kv_transfer_config.kv_connector_extra_config["name"]
        self._connector = SharedStorageConnector(config, role)
        self.call_record: dict[str, int] = defaultdict(int)
        # Use a unique temp file per connector
        self._event_file = (
            tempfile.gettempdir()
            + f"/connector_{self.name}-{self.role.name}_events.log"
        )
        # Start with an empty file
        with open(self._event_file, "w") as _:
            pass

    def __getattribute__(self, name):
        if name in (
            "_connector",
            "call_record",
            "name",
            "_event_file",
            "__class__",
            "__dict__",
            "__getattribute__",
            "__init__",
        ):  # avoid recursion
            return object.__getattribute__(self, name)
        if not hasattr(self._connector, name):
            return object.__getattribute__(self, name)
        attr = getattr(self._connector, name)

        # Intercept calls to the connector interface and write an event
        # for each one to a file, which can be read back in the main test proc.
        if callable(attr):

            def wrapper(*args, **kwargs):
                self.call_record[name] += 1

                # Include args that we're interested in
                to_log = [name]
                for arg in args:
                    if isinstance(arg, int):
                        to_log.append(str(arg))
                    elif isinstance(arg, KVCacheBlocks):
                        to_log.append(f"num_blocks={[len(b) for b in arg.blocks]}")

                # Log the event as a line to the file
                try:
                    with open(self._event_file, "a") as f:
                        f.write(" ".join(to_log) + "\n")
                except Exception as e:
                    print(f"[ERROR] Could not log event {name} for {self.name}: {e}")
                return attr(*args, **kwargs)

            return wrapper
        return attr


KVConnectorFactory.register_connector(
    "TestSharedStorageConnector", __name__, TestSharedStorageConnector.__name__
)
