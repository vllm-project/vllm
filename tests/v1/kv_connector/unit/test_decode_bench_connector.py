# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for DecodeBenchConnector.

Tests the functionality of the DecodeBenchConnector which fills KV cache
with dummy values for decode performance benchmarking.
"""

import pytest
import torch

from vllm import SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole

# ruff: noqa: E501
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnector,
    DecodeBenchConnectorMetadata,
)
from vllm.forward_context import ForwardContext
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

from .utils import (
    EOS_TOKEN_ID,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)


class DecodeBenchTestRunner:
    """Test runner for DecodeBenchConnector."""

    def __init__(self, block_size: int, num_gpu_blocks: int):
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks

        self.req_id = -1

        # Create vllm config with DecodeBenchConnector
        vllm_config = create_vllm_config(
            block_size=block_size,
            max_num_batched_tokens=1000,
            kv_connector="DecodeBenchConnector",
        )

        self.vllm_config = vllm_config
        self.scheduler: Scheduler = create_scheduler(
            vllm_config, num_blocks=num_gpu_blocks
        )

        # Create worker-side connector
        self.worker_connector = DecodeBenchConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            self.scheduler.kv_cache_config,
        )

        # Create dummy KV caches for testing
        # Shape: [num_blocks, 2, num_heads, block_size, head_dim]
        # Using simplified shape for testing
        num_heads = 4
        head_dim = 64
        layer_names = self.scheduler.kv_cache_config.kv_cache_groups[0].layer_names
        self.kv_caches = {
            layer_name: torch.zeros(num_gpu_blocks, 2, num_heads, block_size, head_dim)
            for layer_name in layer_names
        }

        # Register KV caches with worker connector
        self.worker_connector.register_kv_caches(self.kv_caches)

        # Extract scheduler-side connector
        scheduler_connector = self.scheduler.connector
        assert scheduler_connector is not None
        assert isinstance(scheduler_connector, DecodeBenchConnector)
        self.scheduler_connector: DecodeBenchConnector = scheduler_connector

        init_none_hash(sha256)
        self._block_hasher = get_request_block_hasher(block_size, sha256)

        self._dummy_ctx: ForwardContext = ForwardContext(
            no_compile_layers={}, attn_metadata={}, slot_mapping={}
        )

    def new_request(self, token_ids: list[int]) -> Request:
        """Create a new request with given token IDs."""
        self.req_id += 1

        sampling_params = SamplingParams(max_tokens=100)
        sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)

        req = Request(
            request_id=str(self.req_id),
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=self._block_hasher,
        )

        self.scheduler.add_request(req)
        return req

    def run_single_step(self, token_id: int = 0):
        """Run a single scheduler + worker step."""
        scheduler_output = self.scheduler.schedule()

        # Get connector metadata
        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        assert isinstance(kv_connector_metadata, DecodeBenchConnectorMetadata)

        # Bind metadata and load KV
        self.worker_connector.bind_connector_metadata(kv_connector_metadata)
        self.worker_connector.start_load_kv(self._dummy_ctx)

        if scheduler_output.total_num_scheduled_tokens > 0:
            self.worker_connector.wait_for_save()

        self.worker_connector.clear_connector_metadata()

        # Create model runner output
        model_runner_output = create_model_runner_output(
            reqs=self.scheduler.running,
            token_id=token_id,
        )

        self.scheduler.update_from_output(scheduler_output, model_runner_output)

        return scheduler_output, kv_connector_metadata


def test_decode_bench_connector_basic():
    """Test basic functionality of DecodeBenchConnector."""
    block_size = 16
    num_gpu_blocks = 100

    runner = DecodeBenchTestRunner(block_size=block_size, num_gpu_blocks=num_gpu_blocks)

    # Create a request with multiple blocks worth of tokens
    num_tokens = block_size * 3  # 3 blocks
    token_ids = [1] * num_tokens

    req = runner.new_request(token_ids)

    # Run first step - should fill KV cache with dummy values
    scheduler_output, metadata = runner.run_single_step()

    # Check that get_num_new_matched_tokens returned correct value
    # Should be num_tokens - 1 (all except the last token for decode)
    expected_fill_tokens = num_tokens - 1

    # Check metadata has the request to fill
    assert len(metadata.reqs_to_fill) == 1
    assert req.request_id in metadata.reqs_to_fill

    block_ids_per_group, num_tokens_to_fill = metadata.reqs_to_fill[req.request_id]
    assert num_tokens_to_fill == expected_fill_tokens

    # For standard attention, there's only one group
    assert len(block_ids_per_group) == 1
    block_ids = block_ids_per_group[0]

    # Calculate expected number of blocks
    expected_num_blocks = (expected_fill_tokens + block_size - 1) // block_size
    assert len(block_ids) == expected_num_blocks

    # Verify KV caches were filled with constant value
    for layer_name, kv_cache in runner.kv_caches.items():
        for block_id in block_ids:
            # Check that the block was filled
            block_data = kv_cache[block_id]
            # Should be filled with constant value 0.015
            assert torch.allclose(block_data, torch.tensor(0.015))


def test_decode_bench_connector_prefix_hit_fills_only_suffix_blocks():
    """A local prefix-cache hit shifts the fill to the suffix blocks."""
    block_size = 16
    runner = DecodeBenchTestRunner(block_size=block_size, num_gpu_blocks=100)

    prefix = list(range(1, 1 + block_size * 3))
    req1 = runner.new_request(prefix)
    _, metadata = runner.run_single_step()
    (prefix_block_ids,), _ = metadata.reqs_to_fill[req1.request_id]
    assert len(prefix_block_ids) == 3

    scheduler_output = runner.scheduler.schedule()
    runner.scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            reqs=runner.scheduler.running,
            token_id=EOS_TOKEN_ID,
            use_eos=True,
        ),
    )
    assert not runner.scheduler.running

    suffix = list(range(1000, 1000 + block_size * 2))
    req2 = runner.new_request(prefix + suffix)
    _, metadata = runner.run_single_step()
    (fill_block_ids,), num_tokens_to_fill = metadata.reqs_to_fill[req2.request_id]

    req2_block_ids = runner.scheduler.kv_cache_manager.get_block_ids(req2.request_id)[0]
    assert req2_block_ids[:3] == prefix_block_ids
    assert num_tokens_to_fill == len(suffix) - 1
    assert fill_block_ids == req2_block_ids[3:5]
    assert set(fill_block_ids).isdisjoint(prefix_block_ids)
    for kv_cache in runner.kv_caches.values():
        for block_id in fill_block_ids:
            assert torch.allclose(kv_cache[block_id], torch.tensor(0.015))


def test_decode_bench_connector_fills_each_hma_group():
    """Each cache group is filled using its own block IDs."""
    block_size = 16
    num_gpu_blocks = 8
    vllm_config = create_vllm_config(
        block_size=block_size,
        max_num_batched_tokens=1000,
        kv_connector="DecodeBenchConnector",
    )
    kv_cache_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_gpu_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["group_0_layer"], kv_cache_spec),
            KVCacheGroupSpec(["group_1_layer"], kv_cache_spec),
        ],
    )
    connector = DecodeBenchConnector(
        vllm_config,
        KVConnectorRole.WORKER,
        kv_cache_config,
    )
    group_0_cache = torch.zeros(num_gpu_blocks, 2)
    group_1_cache = torch.zeros(num_gpu_blocks, 2)
    connector.register_kv_caches(
        {
            "group_0_layer": group_0_cache,
            "group_1_layer": group_1_cache,
        }
    )
    connector.bind_connector_metadata(
        DecodeBenchConnectorMetadata(reqs_to_fill={"request": (([1], [5]), block_size)})
    )

    connector.start_load_kv(
        ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
    )

    assert torch.allclose(group_0_cache[1], torch.tensor(0.015))
    assert torch.count_nonzero(group_0_cache[5]) == 0
    assert torch.count_nonzero(group_1_cache[1]) == 0
    assert torch.allclose(group_1_cache[5], torch.tensor(0.015))


def test_decode_bench_connector_zero_fills_circular_buffer_groups():
    """Circular buffers are zeroed to preserve packed non-floating metadata."""
    num_gpu_blocks = 8
    vllm_config = create_vllm_config(
        block_size=16,
        max_num_batched_tokens=1000,
        kv_connector="DecodeBenchConnector",
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_gpu_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["circular_layer"],
                CircularBufferSpec(
                    block_size=2,
                    num_kv_heads=1,
                    head_size=16,
                    head_size_v=0,
                    dtype=torch.bfloat16,
                ),
            ),
        ],
    )
    connector = DecodeBenchConnector(
        vllm_config,
        KVConnectorRole.WORKER,
        kv_cache_config,
    )
    circular_cache = torch.ones(num_gpu_blocks, 16, dtype=torch.bfloat16)
    connector.register_kv_caches({"circular_layer": circular_cache})
    connector.bind_connector_metadata(
        DecodeBenchConnectorMetadata(reqs_to_fill={"request": (([5],), 1)})
    )

    connector.start_load_kv(
        ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
    )

    expected_circular = torch.ones_like(circular_cache)
    expected_circular[5] = 0
    assert torch.equal(circular_cache, expected_circular)


def test_decode_bench_connector_selects_external_blocks_per_group():
    """External fills follow the local prefix using each group's block size."""
    vllm_config = create_vllm_config(
        block_size=16,
        max_num_batched_tokens=1000,
        kv_connector="DecodeBenchConnector",
    )
    kv_cache_groups = [
        KVCacheGroupSpec(
            ["full_attention"],
            FullAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
        KVCacheGroupSpec(
            ["compressor_ring"],
            CircularBufferSpec(
                block_size=32,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
    ]
    connector = DecodeBenchConnector(
        vllm_config,
        KVConnectorRole.SCHEDULER,
        KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        ),
    )
    num_local_tokens = 32
    num_external_tokens = 17
    request = create_request(
        request_id=1,
        num_tokens=num_local_tokens + num_external_tokens + 1,
        block_size=16,
    )
    blocks = KVCacheBlocks(
        (
            [
                KVCacheBlock(block_id=0),
                KVCacheBlock(block_id=1),
                # Full external blocks may already be hashed by allocate_slots().
                KVCacheBlock(
                    block_id=2,
                    _block_hash=make_block_hash_with_group_id(
                        BlockHash(b"external-full-block"), 0
                    ),
                    _block_hash_num_tokens=48,
                ),
                KVCacheBlock(block_id=3),
            ],
            [KVCacheBlock(block_id=4)],
        )
    )

    connector.update_state_after_alloc(request, blocks, num_external_tokens)
    metadata = connector.build_connector_meta(SchedulerOutput.make_empty())

    assert isinstance(metadata, DecodeBenchConnectorMetadata)
    block_ids_per_group, num_tokens = metadata.reqs_to_fill[request.request_id]
    assert block_ids_per_group == ([2, 3], [4])
    assert num_tokens == num_external_tokens


def test_decode_bench_connector_selects_unaligned_external_block():
    """An external interval includes the block containing its first token."""
    block_size = 32
    vllm_config = create_vllm_config(
        block_size=block_size,
        max_num_batched_tokens=1000,
        kv_connector="DecodeBenchConnector",
    )
    kv_cache_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    connector = DecodeBenchConnector(
        vllm_config,
        KVConnectorRole.SCHEDULER,
        KVCacheConfig(
            num_blocks=4,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
        ),
    )
    num_local_tokens = 23
    num_external_tokens = 17
    request = create_request(
        request_id=1,
        num_tokens=num_local_tokens + num_external_tokens + 1,
        block_size=block_size,
    )
    blocks = KVCacheBlocks(([KVCacheBlock(block_id=1), KVCacheBlock(block_id=2)],))

    connector.update_state_after_alloc(request, blocks, num_external_tokens)
    metadata = connector.build_connector_meta(SchedulerOutput.make_empty())

    assert isinstance(metadata, DecodeBenchConnectorMetadata)
    assert metadata.reqs_to_fill[request.request_id] == (
        ([1, 2],),
        num_external_tokens,
    )


def test_decode_bench_connector_skips_null_padding():
    """External fills exclude null padding from a sliding-window group."""
    block_size = 16
    vllm_config = create_vllm_config(
        block_size=block_size,
        max_num_batched_tokens=1000,
        kv_connector="DecodeBenchConnector",
    )
    kv_cache_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=2 * block_size,
    )
    connector = DecodeBenchConnector(
        vllm_config,
        KVConnectorRole.SCHEDULER,
        KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
        ),
    )
    num_external_tokens = 4 * block_size
    request = create_request(
        request_id=1,
        num_tokens=num_external_tokens + 1,
        block_size=block_size,
    )
    null_block = KVCacheBlock(block_id=0, is_null=True)
    blocks = KVCacheBlocks(
        (
            [
                null_block,
                null_block,
                KVCacheBlock(block_id=1),
                KVCacheBlock(block_id=2),
                KVCacheBlock(block_id=3),
            ],
        )
    )

    connector.update_state_after_alloc(request, blocks, num_external_tokens)
    metadata = connector.build_connector_meta(SchedulerOutput.make_empty())

    assert isinstance(metadata, DecodeBenchConnectorMetadata)
    assert metadata.reqs_to_fill[request.request_id] == (
        ([1, 2],),
        num_external_tokens,
    )


def test_decode_bench_connector_uses_per_group_dcp_block_sizes():
    """DCP scales full-attention blocks but not sliding-window blocks."""
    block_size = 16
    dcp_world_size = 2
    vllm_config = create_vllm_config(
        block_size=block_size,
        kv_connector="DecodeBenchConnector",
    )
    vllm_config.parallel_config.decode_context_parallel_size = dcp_world_size
    full_attention = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    sliding_window = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=2 * block_size,
    )
    connector = DecodeBenchConnector(
        vllm_config,
        KVConnectorRole.SCHEDULER,
        KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(["full_attention"], full_attention),
                KVCacheGroupSpec(["sliding_window"], sliding_window),
            ],
        ),
    )
    num_external_tokens = block_size * dcp_world_size + 1
    request = create_request(
        request_id=1,
        num_tokens=num_external_tokens + 1,
        block_size=block_size,
    )
    blocks = KVCacheBlocks(
        (
            [KVCacheBlock(block_id=i) for i in (1, 2, 3)],
            [KVCacheBlock(block_id=i) for i in (5, 6, 7, 8)],
        )
    )

    connector.update_state_after_alloc(request, blocks, num_external_tokens)
    metadata = connector.build_connector_meta(SchedulerOutput.make_empty())

    assert isinstance(metadata, DecodeBenchConnectorMetadata)
    assert metadata.reqs_to_fill[request.request_id] == (
        ([1, 2], [5, 6, 7]),
        num_external_tokens,
    )


def test_decode_bench_connector_no_refill():
    """Test that DecodeBenchConnector only fills once per request."""
    block_size = 16
    num_gpu_blocks = 100

    runner = DecodeBenchTestRunner(block_size=block_size, num_gpu_blocks=num_gpu_blocks)

    # Create a request
    num_tokens = block_size * 2
    token_ids = [1] * num_tokens

    runner.new_request(token_ids)

    # Run first step - should fill KV cache
    _, metadata1 = runner.run_single_step()
    assert len(metadata1.reqs_to_fill) == 1

    # Run second step - should NOT fill again (already filled)
    _, metadata2 = runner.run_single_step()
    assert len(metadata2.reqs_to_fill) == 0


def test_decode_bench_connector_single_token():
    """Test DecodeBenchConnector with single token request."""
    block_size = 16
    num_gpu_blocks = 100

    runner = DecodeBenchTestRunner(block_size=block_size, num_gpu_blocks=num_gpu_blocks)

    # Create a request with just 1 token
    # Should not fill anything (need at least 2 tokens: 1 to fill, 1 to decode)
    token_ids = [1]

    runner.new_request(token_ids)

    # Run step - should NOT fill KV cache
    _, metadata = runner.run_single_step()
    assert len(metadata.reqs_to_fill) == 0


def test_decode_bench_connector_two_tokens():
    """Test DecodeBenchConnector with two token request."""
    block_size = 16
    num_gpu_blocks = 100

    runner = DecodeBenchTestRunner(block_size=block_size, num_gpu_blocks=num_gpu_blocks)

    # Create a request with 2 tokens
    # Should fill 1 token (first token), decode the second
    token_ids = [1, 2]

    req = runner.new_request(token_ids)

    # Run step
    _, metadata = runner.run_single_step()

    assert len(metadata.reqs_to_fill) == 1
    assert req.request_id in metadata.reqs_to_fill

    block_ids_per_group, num_tokens_to_fill = metadata.reqs_to_fill[req.request_id]
    assert num_tokens_to_fill == 1
    # For standard attention, there's only one group
    assert len(block_ids_per_group) == 1
    assert len(block_ids_per_group[0]) == 1  # 1 token needs 1 block


def test_decode_bench_connector_large_context():
    """Test DecodeBenchConnector with large context size."""
    block_size = 16
    num_gpu_blocks = 1000

    runner = DecodeBenchTestRunner(block_size=block_size, num_gpu_blocks=num_gpu_blocks)

    # Create a request with many blocks
    num_blocks = 20
    num_tokens = block_size * num_blocks
    token_ids = list(range(num_tokens))

    req = runner.new_request(token_ids)

    # Run step
    _, metadata = runner.run_single_step()

    assert len(metadata.reqs_to_fill) == 1
    assert req.request_id in metadata.reqs_to_fill

    block_ids_per_group, num_tokens_to_fill = metadata.reqs_to_fill[req.request_id]

    # Should fill all tokens except the last one
    expected_fill_tokens = num_tokens - 1
    assert num_tokens_to_fill == expected_fill_tokens

    # For standard attention, there's only one group
    assert len(block_ids_per_group) == 1
    block_ids = block_ids_per_group[0]

    # Calculate expected number of blocks
    expected_num_blocks = (expected_fill_tokens + block_size - 1) // block_size
    assert len(block_ids) == expected_num_blocks

    # Verify blocks were filled
    for layer_name, kv_cache in runner.kv_caches.items():
        for block_id in block_ids:
            block_data = kv_cache[block_id]
            assert torch.allclose(block_data, torch.tensor(0.015))


def test_decode_bench_connector_multiple_requests():
    """Test DecodeBenchConnector with multiple sequential requests."""
    block_size = 16
    num_gpu_blocks = 100

    runner = DecodeBenchTestRunner(block_size=block_size, num_gpu_blocks=num_gpu_blocks)

    # First request
    req1 = runner.new_request([1] * (block_size * 2))
    _, metadata1 = runner.run_single_step()

    assert len(metadata1.reqs_to_fill) == 1
    assert req1.request_id in metadata1.reqs_to_fill

    # Complete first request
    while runner.scheduler.running:
        runner.run_single_step()

    # Add EOS to finish
    scheduler_output = runner.scheduler.schedule()
    model_runner_output = create_model_runner_output(
        reqs=runner.scheduler.running,
        token_id=EOS_TOKEN_ID,
        use_eos=True,
    )
    runner.scheduler.update_from_output(scheduler_output, model_runner_output)

    # Second request - should also get filled
    req2 = runner.new_request([2] * (block_size * 3))
    _, metadata2 = runner.run_single_step()

    assert len(metadata2.reqs_to_fill) == 1
    assert req2.request_id in metadata2.reqs_to_fill

    # Different request should have different metadata
    _, num_tokens1 = metadata1.reqs_to_fill[req1.request_id]
    _, num_tokens2 = metadata2.reqs_to_fill[req2.request_id]

    assert num_tokens1 == block_size * 2 - 1
    assert num_tokens2 == block_size * 3 - 1


def test_decode_bench_connector_partial_block():
    """Test DecodeBenchConnector with partial block filling."""
    block_size = 16
    num_gpu_blocks = 100

    runner = DecodeBenchTestRunner(block_size=block_size, num_gpu_blocks=num_gpu_blocks)

    # Create a request that doesn't align to block boundaries
    # e.g., 2.5 blocks worth of tokens
    num_tokens = block_size * 2 + block_size // 2
    token_ids = [1] * num_tokens

    req = runner.new_request(token_ids)

    # Run step
    _, metadata = runner.run_single_step()

    assert len(metadata.reqs_to_fill) == 1
    assert req.request_id in metadata.reqs_to_fill

    block_ids_per_group, num_tokens_to_fill = metadata.reqs_to_fill[req.request_id]

    # Should fill all tokens except the last one
    expected_fill_tokens = num_tokens - 1
    assert num_tokens_to_fill == expected_fill_tokens

    # For standard attention, there's only one group
    assert len(block_ids_per_group) == 1
    block_ids = block_ids_per_group[0]

    # Should allocate 3 blocks to hold the partial data
    expected_num_blocks = 3
    assert len(block_ids) == expected_num_blocks


def test_decode_bench_connector_concurrent_requests():
    """Test DecodeBenchConnector with multiple concurrent requests in the same batch."""
    block_size = 16
    num_gpu_blocks = 1000

    runner = DecodeBenchTestRunner(block_size=block_size, num_gpu_blocks=num_gpu_blocks)

    # Create multiple requests that will be batched together
    req1 = runner.new_request([1] * (block_size * 2))
    req2 = runner.new_request([2] * (block_size * 3))
    req3 = runner.new_request([3] * (block_size * 1))

    # Run first step - all requests should be filled concurrently
    _, metadata = runner.run_single_step()

    # All three requests should be in the metadata
    assert len(metadata.reqs_to_fill) == 3
    assert req1.request_id in metadata.reqs_to_fill
    assert req2.request_id in metadata.reqs_to_fill
    assert req3.request_id in metadata.reqs_to_fill

    # Verify each request has correct fill info
    block_ids_per_group1, num_tokens1 = metadata.reqs_to_fill[req1.request_id]
    block_ids_per_group2, num_tokens2 = metadata.reqs_to_fill[req2.request_id]
    block_ids_per_group3, num_tokens3 = metadata.reqs_to_fill[req3.request_id]

    # Verify token counts (all tokens except last one)
    assert num_tokens1 == block_size * 2 - 1
    assert num_tokens2 == block_size * 3 - 1
    assert num_tokens3 == block_size * 1 - 1

    # Verify block counts for each request
    assert len(block_ids_per_group1[0]) == 2  # 2 blocks
    assert len(block_ids_per_group2[0]) == 3  # 3 blocks
    assert len(block_ids_per_group3[0]) == 1  # 1 block

    # Verify all blocks are filled in KV cache
    for req_id, (block_ids_per_group, _) in metadata.reqs_to_fill.items():
        block_ids = block_ids_per_group[0]
        for layer_name, kv_cache in runner.kv_caches.items():
            for block_id in block_ids:
                block_data = kv_cache[block_id]
                assert torch.allclose(block_data, torch.tensor(0.015))

    # Run second step - should NOT fill again (already filled)
    _, metadata2 = runner.run_single_step()
    assert len(metadata2.reqs_to_fill) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
