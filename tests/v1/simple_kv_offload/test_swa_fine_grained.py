# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-tier fine-grained hits with a sliding-window group (eager offload).

A producer prefills, decodes and finishes; its GPU prefix cache is then
dropped. A follow-up turn must get the fine-grained hit at the producer's
prompt boundary from the CPU tier, including the SWA window and the SWA tail
block that ends inside a block.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.output import (
    CachedRequestData,
    KVConnectorBlockState,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadWorkerMetadata

pytestmark = pytest.mark.skip_global_cleanup

UNIT = 16
BLOCK = 32
WINDOW = 56
MAMBA_BLOCK = 256
NUM_GPU_BLOCKS = 400
NUM_CPU_BLOCKS = 400

init_none_hash(sha256)


def _config() -> KVCacheConfig:
    specs = [
        FullAttentionSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=16, dtype=torch.float16
        ),
        MambaSpec(
            block_size=MAMBA_BLOCK,
            shapes=((16, 16),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
        ),
        SlidingWindowSpec(
            block_size=BLOCK,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.float16,
            sliding_window=WINDOW,
        ),
    ]
    names = ["full", "mamba", "swa"]
    return KVCacheConfig(
        num_blocks=NUM_GPU_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(
                size=spec.page_size_bytes * NUM_GPU_BLOCKS,
                layers=[name],
                layer_stride=spec.page_size_bytes * NUM_GPU_BLOCKS,
                block_stride=spec.page_size_bytes,
            )
            for name, spec in zip(names, specs)
        ],
        kv_cache_groups=[KVCacheGroupSpec([n], s) for n, s in zip(names, specs)],
        prefix_cache_retention_interval=0,
    )


def _setup(monkeypatch, fine_grained: bool = True):
    monkeypatch.setattr(
        SlidingWindowManager, "supports_fine_grained_hash_lookup", fine_grained
    )
    config = _config()
    gpu = make_kv_cache_manager(
        config, max_model_len=8192, enable_caching=True, hash_block_size=UNIT
    )
    vllm_config = SimpleNamespace(
        kv_events_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, world_size=1),
        speculative_config=None,
        model_config=SimpleNamespace(max_model_len=8192),
        max_in_flight_tokens=8192,
        num_prefill_lookahead_tokens=0,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024),
    )
    cpu = SimpleCPUOffloadScheduler(
        vllm_config=vllm_config,  # type: ignore[arg-type]
        kv_cache_config=config,
        cpu_capacity_bytes=config.kv_cache_tensors[0].size
        // NUM_GPU_BLOCKS
        * NUM_CPU_BLOCKS,
        scheduler_block_size=MAMBA_BLOCK,
        hash_block_size=UNIT,
        num_cpu_blocks=NUM_CPU_BLOCKS,
    )
    cpu.bind_gpu_block_pool(gpu.block_pool)
    return gpu, cpu


def _step(gpu, cpu, req, n: int, first: bool) -> None:
    """allocate -> build connector meta -> advance, as the scheduler does."""
    new_blocks = gpu.allocate_slots(req, n)
    assert new_blocks is not None
    if first:
        cpu.update_state_after_alloc(req, gpu.get_blocks(req.request_id), 0)
    table = gpu.get_block_ids(req.request_id)
    out = SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id=req.request_id,
                prompt_token_ids=None,
                mm_features=[],
                sampling_params=None,
                pooling_params=None,
                block_ids=table,
                num_computed_tokens=req.num_computed_tokens,
                lora_request=None,
            )
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req.request_id: n},
        total_num_scheduled_tokens=n,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        preempted_req_ids=set(),
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    out.kv_connector_block_state = KVConnectorBlockState(
        req_ids={req.request_id},
        resolve_block_ids=gpu.get_block_ids,
        boundary_state_offloads=gpu.take_boundary_state_offloads(),
    )
    _, cow_retained = gpu.take_kv_cache_block_copies()
    _complete(cpu, cpu.build_connector_meta(out))
    req.num_computed_tokens += n
    # The step (and its CoW copies) ran; release the copy endpoints.
    gpu.block_pool.free_blocks(cow_retained)
    gpu.new_step_starts()


def _complete(cpu, meta) -> None:
    if meta.store_event >= 0:
        cpu.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=SimpleCPUOffloadWorkerMetadata(
                    completed_store_events={meta.store_event: 1}
                )
            )
        )


def _empty_output() -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        preempted_req_ids=set(),
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _produce_and_evict(gpu, cpu, prompt_len: int, num_output: int):
    tokens = list(range(prompt_len))
    req = make_request("producer", tokens, UNIT, sha256)
    boundary = (prompt_len - 1) // UNIT * UNIT
    ends = sorted({*range(MAMBA_BLOCK, boundary, MAMBA_BLOCK), boundary, prompt_len})
    for i, end in enumerate(ends):
        _step(gpu, cpu, req, end - req.num_computed_tokens, first=i == 0)
    for i in range(num_output):
        req.append_output_token_ids(50_000 + i)
        _step(gpu, cpu, req, 1, first=False)
    # Finish: the scheduler nulls out-of-window pages, then hands the table over.
    gpu.remove_skipped_blocks(req.request_id, req.num_computed_tokens)
    cpu.request_finished_all_groups(req, gpu.get_block_ids(req.request_id))
    gpu.free(req)
    gpu.new_step_starts()
    _complete(cpu, cpu.build_connector_meta(_empty_output()))
    pool = gpu.block_pool
    held = [b.block_id for b in pool.blocks if b.ref_cnt > 0 and not b.is_null]
    assert not held, f"blocks still referenced after finish: {held}"
    pool.evict_blocks({b.block_id for b in pool.blocks if not b.is_null})
    return tokens + list(req.output_token_ids), boundary


@pytest.mark.parametrize("num_output", [4, 40, 80])
def test_cpu_tier_serves_fine_grained_swa_hit(num_output, monkeypatch):
    gpu, cpu = _setup(monkeypatch)
    assert cpu.cpu_coordinator.enable_partial_hash_hits
    history, boundary = _produce_and_evict(gpu, cpu, 601, num_output)
    assert boundary % BLOCK and boundary % MAMBA_BLOCK

    consumer = make_request(
        "consumer", history + list(range(70_000, 70_100)), UNIT, sha256
    )
    _, local, _ = gpu.get_computed_blocks(consumer)
    assert local == 0
    hit, is_async = cpu.get_num_new_matched_tokens(consumer, 0)
    assert (hit, is_async) == (boundary, True)

    # The load covers the whole SWA window, tail block included.
    blocks = gpu.allocate_slots(
        consumer,
        len(consumer.all_token_ids) - hit,
        num_external_computed_tokens=hit,
    )
    assert blocks is not None
    cpu.update_state_after_alloc(consumer, gpu.get_blocks(consumer.request_id), hit)
    load = cpu._reqs_to_load[consumer.request_id].transfer_meta
    swa_table = gpu.get_block_ids(consumer.request_id)[2]
    first = (boundary - (WINDOW - 1)) // BLOCK
    window_gpu_ids = set(swa_table[first : boundary // BLOCK + 1])
    assert window_gpu_ids <= set(load.gpu_block_ids)


def test_cpu_tier_block_aligned_without_fine_grained_swa(monkeypatch):
    gpu, cpu = _setup(monkeypatch, fine_grained=False)
    assert not cpu.cpu_coordinator.enable_partial_hash_hits
    history, boundary = _produce_and_evict(gpu, cpu, 601, 4)
    consumer = make_request(
        "consumer", history + list(range(70_000, 70_100)), UNIT, sha256
    )
    hit, _ = cpu.get_num_new_matched_tokens(consumer, 0)
    assert hit == boundary // MAMBA_BLOCK * MAMBA_BLOCK
