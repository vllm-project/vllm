# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PD replay storage follows exact positions and allocation lifetimes."""

from copy import copy, deepcopy
from dataclasses import replace

import msgspec
import pytest
import torch
from transformers import OPTConfig

from tests.v1.core.test_prefix_caching import make_request
from tests.v1.core.utils import new_kv_cache_group
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_pd_replay import (
    PDReplayRetention,
    pd_replay_retention_enabled,
)
from vllm.exceptions import VLLMValidationError
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy, init_none_hash
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import (
    DFlashSlidingWindowManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def new_manager(lookahead=0, *, planned=True, extra_retained_tokens=0):
    groups = []
    for i, (size, window) in enumerate(
        ((64, 128), (64, 128), (256, 0), (8, 128), (4, 8))
    ):
        spec = AttentionSpec(
            block_size=size, num_kv_heads=1, head_size=8, dtype=torch.float32
        )
        policy = (
            SlidingWindowManager.layer_config(
                spec,
                sliding_window=window,
                extra_retained_tokens=extra_retained_tokens,
                storage_retention_tokens=max(1, lookahead) + 255 if planned else 0,
            )
            if window
            else spec
        )
        groups.append(new_kv_cache_group([f"layer{i}"], policy))
    cache = KVCacheConfig(
        6000,
        [],
        groups,
        hash_block_size=4,
        cache_hit_alignment_tokens=256,
        prefix_cache_retention_interval=0,
    )
    init_none_hash(sha256)
    return KVCacheManager(
        cache,
        max_model_len=32768,
        max_in_flight_tokens=8192,
        scheduler_block_size=256,
        hash_block_size=4,
        num_hash_lookahead_tokens=lookahead,
    )


def request_of(name, length, lookahead=0, *, max_tokens=1):
    return make_request(
        name,
        list(range(length)),
        4,
        sha256,
        num_hash_lookahead_tokens=lookahead,
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def step_for(request_id):
    return SchedulerOutput(
        [], CachedRequestData.make_empty(), {request_id: 1}, 1, {}, {}, set(), []
    )


@pytest.fixture(scope="module")
def pd_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("pd-model")
    OPTConfig(
        hidden_size=64,
        num_attention_heads=2,
        ffn_dim=128,
        num_hidden_layers=1,
        vocab_size=128,
        max_position_embeddings=32768,
    ).save_pretrained(path)
    return str(path)


def connector_for(manager, model, *, producer, enabled=True):
    options = {
        "read_mode": True,
        "host": "127.0.0.1",
        "handshake_port": 32500,
        "notify_port": 32502,
    }
    if enabled is not None:
        options["pd_replay_retention"] = enabled
    config = VllmConfig(
        model_config=ModelConfig(
            model=model, skip_tokenizer_init=True, dtype="float32", max_model_len=32768
        ),
        cache_config=CacheConfig(block_size=256, enable_prefix_caching=True),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=32768,
            max_num_seqs=2,
            max_num_batched_tokens=8192,
        ),
        device_config=DeviceConfig("cpu"),
        kv_transfer_config=KVTransferConfig(
            kv_connector="MoRIIOConnector",
            kv_role="kv_producer" if producer else "kv_consumer",
            engine_id="pd-test",
            kv_connector_extra_config=options,
        ),
    )
    connector = MoRIIOConnector(
        config, KVConnectorRole.SCHEDULER, manager.kv_cache_config
    )
    connector.bind_kv_cache_manager(manager)
    return connector


@pytest.mark.parametrize(
    "lookahead,length,expected",
    [
        (0, 1024, ((10, 16), (10, 16), (0, 4), (80, 128), (190, 256))),
        (1, 1025, ((14, 17), (14, 17), (0, 5), (112, 129), (254, 257))),
        (0, 1224, ((14, 20), (14, 20), (0, 5), (112, 153), (254, 306))),
        (257, 1024, ((6, 16), (6, 16), (0, 4), (48, 128), (126, 256))),
        (257, 1025, ((10, 17), (10, 17), (0, 5), (80, 129), (190, 257))),
        (257, 1224, ((10, 20), (10, 20), (0, 5), (80, 153), (190, 306))),
    ],
)
def test_pd_ranges_cover_only_required_contiguous_tails(lookahead, length, expected):
    """Aligned boundaries 512/768/1024 need tails of 2/16/2 group blocks."""
    policy = PDReplayRetention(new_manager(lookahead), lookahead, producer=True)
    assert policy.ranges(length, length) == expected


def test_pd_retention_survives_chunks_until_completion_ack(pd_model):
    manager = new_manager()
    connector = connector_for(manager, pd_model, producer=True)
    request = request_of("producer", 16484)
    request.kv_transfer_params = {"do_remote_decode": True, "transfer_id": "transfer"}
    try:
        for tokens in (8192, 8192, 100):
            assert manager.allocate_slots(request, tokens) is not None
            request.num_computed_tokens += tokens
            manager.remove_skipped_blocks(
                request.request_id,
                request.num_computed_tokens,
                request.num_prompt_tokens,
            )
        policy = connector.connector_scheduler.pd_replay
        assert policy is not None
        ranges = policy.ranges(request.num_prompt_tokens, request.num_computed_tokens)
        blocks = manager.get_block_ids(request.request_id)
        assert all(all(group[start:end]) for group, (start, end) in zip(blocks, ranges))
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        delay, params = connector.request_finished_all_groups(request, blocks)
        assert delay and params["pd_replay"]["ranges"] == [list(r) for r in ranges]
        assert request.request_id in policy.allocations
        connector.update_connector_output(
            KVConnectorOutput(finished_sending={request.request_id})
        )
        assert request.request_id not in policy.allocations
        manager.free(request)
        repeat = request_of("repeat", request.num_prompt_tokens)
        assert manager.get_computed_blocks(repeat)[1] == 16384
    finally:
        connector.shutdown()


@pytest.mark.parametrize("length,local_hit", [(1025, 0), (1124, 0), (1124, 1024)])
@pytest.mark.parametrize("read_failed", [False, True])
def test_pd_read_pairs_absolute_positions_and_releases_after_result(
    pd_model, length, local_hit, read_failed
):
    producer, consumer = new_manager(), new_manager()
    p = connector_for(producer, pd_model, producer=True)
    d = connector_for(consumer, pd_model, producer=False)
    source = request_of("source", length)
    source.kv_transfer_params = {"do_remote_decode": True, "transfer_id": "read"}
    target = request_of("target", length)
    try:
        assert producer.allocate_slots(source, length) is not None
        source.num_computed_tokens = length
        source.status = RequestStatus.FINISHED_LENGTH_CAPPED
        _, params = p.request_finished_all_groups(
            source, producer.get_block_ids(source.request_id)
        )
        target.kv_transfer_params = deepcopy(params)
        if local_hit:
            warm = request_of("warm", length)
            assert consumer.allocate_slots(warm, 1024) is not None
            warm.num_computed_tokens = 1024
            consumer.free(warm)
        hits, local_tokens, _ = consumer.get_computed_blocks(target)
        assert local_tokens == local_hit
        external, asynchronous = d.get_num_new_matched_tokens(target, local_tokens)
        assert external == length - 1 - local_tokens and not asynchronous
        assert (
            consumer.allocate_slots(
                target,
                1,
                num_new_computed_tokens=local_tokens,
                new_computed_blocks=hits,
                num_external_computed_tokens=external,
                num_lookahead_tokens=256,
            )
            is not None
        )
        allocated = consumer.get_blocks(target.request_id)
        d.update_state_after_alloc(target, allocated, external)
        step = step_for(target.request_id)
        step.has_sync_kv_loads = external > 0
        all_blocks = {b for group in allocated.get_block_ids() for b in group if b}
        step.new_block_ids_to_zero = sorted(all_blocks)
        step.kv_cache_block_copies = [
            KVCacheBlockCopy(0, allocated.get_block_ids()[2][local_tokens // 256])
        ]
        meta = d.build_connector_meta(step)
        assert meta.order_reads_after_compute_stream
        recv = meta.reqs_to_recv[target.request_id]
        read_blocks = {b for group in recv.local_block_ids for b in group}
        assert meta.unzeroed_read_block_ids == read_blocks
        assert step.new_block_ids_to_zero == sorted(all_blocks - read_blocks)
        for i, ((start, end), local_ids, remote_ids) in enumerate(
            zip(
                params["pd_replay"]["ranges"],
                recv.local_block_ids,
                recv.remote_block_ids,
            )
        ):
            size = consumer.coordinator.single_type_managers[i].block_size
            lo, hi = max(start, local_tokens // size), min(end, cdiv(length - 1, size))
            assert local_ids == allocated.get_block_ids()[i][lo:hi]
            assert remote_ids == params["remote_block_ids"][i][lo - start : hi - start]
            assert all(local_ids) and len(local_ids) == len(remote_ids)
        policy = d.connector_scheduler.pd_replay
        assert target.request_id in policy.allocations
        d.bind_connector_metadata(meta)
        completion = d.build_connector_worker_meta()
        assert isinstance(completion, MoRIIOConnectorWorkerMetadata)
        assert completion.pd_replay_generations == meta.pd_replay_generations
        failed_blocks = read_blocks if read_failed else set()
        completion = completion.aggregate(
            MoRIIOConnectorWorkerMetadata(
                failed_blocks, dict(completion.pd_replay_generations)
            )
        )
        d.clear_connector_metadata()
        d.update_connector_output(
            KVConnectorOutput(
                invalid_block_ids=failed_blocks, kv_connector_worker_meta=completion
            )
        )
        assert target.request_id not in policy.allocations
        next_step = SchedulerOutput.make_empty()
        d.build_connector_meta(next_step)
        assert (next_step.new_block_ids_to_zero or []) == sorted(failed_blocks)
    finally:
        p.shutdown()
        d.shutdown()


@pytest.mark.parametrize("pop", [False, True])
def test_pd_old_result_cannot_release_a_new_allocation(pop):
    manager = new_manager()
    policy = PDReplayRetention(manager, 0, producer=True)
    request = request_of("generation", 1124)
    request.kv_transfer_params = {"do_remote_decode": True}
    assert manager.allocate_slots(request, 512) is not None
    old = policy.allocations[request.request_id].generation
    generations = policy.step_generations([request.request_id])
    if pop:
        blocks = manager.pop_blocks_for_free(request)
        assert blocks and all(b.ref_cnt == 1 for b in blocks)
        manager.block_pool.free_blocks(reversed(blocks))
    else:
        manager.free(request)
    assert manager.allocate_slots(request, 512) is not None
    assert policy.allocations[request.request_id].generation > old
    policy.complete_step(generations)
    assert request.request_id in policy.allocations


def test_pd_release_restores_moving_window_for_long_decode():
    producer = new_manager()
    source_policy = PDReplayRetention(producer, 0, producer=True)
    source = request_of("long-source", 16484)
    source.kv_transfer_params = {"do_remote_decode": True}
    for tokens in (8192, 8192, 100):
        assert producer.allocate_slots(source, tokens) is not None
        source.num_computed_tokens += tokens
    ids, params = source_policy.export(
        source, producer.get_block_ids(source.request_id)
    )
    manager = new_manager()
    policy = PDReplayRetention(manager, 0, producer=False)
    request = request_of("long-output", 16484, max_tokens=1024)
    request.kv_transfer_params = {
        "do_remote_prefill": True,
        "pd_replay": params,
        "remote_block_ids": ids,
    }
    assert (
        manager.allocate_slots(request, 1, num_external_computed_tokens=16483)
        is not None
    )
    generations = policy.step_generations([request.request_id])
    policy.complete_step(generations)
    request.num_computed_tokens = request.num_prompt_tokens
    for _ in range(512):
        request.append_output_token_ids(1)
        assert manager.allocate_slots(request, 1) is not None
        request.num_computed_tokens += 1
    for group in policy.managers:
        if isinstance(group, SlidingWindowManager):
            held = sum(not b.is_null for b in group.req_to_blocks[request.request_id])
            assert held <= cdiv(group.num_retained_tokens + 1, group.block_size) + 1


@pytest.mark.parametrize("kind", ["write", "other", "dflash", "role"])
def test_pd_replay_rejects_unsupported_configs(kind):
    if kind == "dflash":
        manager = new_manager()
        group = manager.kv_cache_config.kv_cache_groups[0]
        changed = replace(
            group,
            manager_config=DFlashSlidingWindowManager.config(
                sliding_window=128, storage_retention_tokens=256
            ),
        )
        cache = replace(
            manager.kv_cache_config,
            kv_cache_groups=[changed, *manager.kv_cache_config.kv_cache_groups[1:]],
        )
        manager = KVCacheManager(
            cache, max_model_len=32768, scheduler_block_size=256, hash_block_size=4
        )
        with pytest.raises(ValueError, match="plain SlidingWindowManager"):
            PDReplayRetention(manager, 0, producer=True)
    else:
        config = KVTransferConfig(
            kv_connector="NixlConnector" if kind == "other" else "MoRIIOConnector",
            kv_role="kv_both" if kind == "role" else "kv_producer",
            kv_connector_extra_config={
                "pd_replay_retention": True,
                "read_mode": kind == "role",
            },
        )
        with pytest.raises(
            ValueError,
            match="producer or consumer role" if kind == "role" else "READ mode",
        ):
            pd_replay_retention_enabled(config)


def test_pd_opt_out_preserves_returned_wire_bytes(pd_model):
    wires = []
    for enabled in (None, False):
        manager = new_manager(planned=False)
        connector = connector_for(manager, pd_model, producer=True, enabled=enabled)
        request = request_of("opt-out", 1124)
        request.kv_transfer_params = {
            "do_remote_decode": True,
            "transfer_id": "unchanged",
        }
        try:
            assert manager.allocate_slots(request, 1124) is not None
            request.num_computed_tokens = 1124
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            delay, params = connector.request_finished_all_groups(
                request, manager.get_block_ids(request.request_id)
            )
            assert (
                connector.connector_scheduler.pd_replay is None
                and "pd_replay" not in params
            )
            wires.append(msgspec.json.encode((delay, params)))
        finally:
            connector.shutdown()
    assert wires[0] == wires[1]


@pytest.mark.parametrize(
    "kind",
    [
        "single",
        "no_connector",
        "multi",
        "multi_disabled",
        "multi_duplicate",
        "multi_duplicate_full",
        "unsupported",
    ],
)
def test_pd_planning_margin_does_not_change_hits_or_attention(pd_model, kind):
    from vllm.v1.core.kv_cache_planning import get_global_kv_cache_groups
    from vllm.v1.kv_cache_interface import KVCacheLayerConfig

    manager = new_manager()
    connector = connector_for(manager, pd_model, producer=True)
    try:
        config = copy(connector.connector_scheduler.vllm_config)
        config.cache_config = copy(config.cache_config)
        config.cache_config.kv_cache_layout = "LBHNC"
        if kind == "no_connector":
            config.kv_transfer_config = KVTransferConfig()
        if kind.startswith("multi"):
            child = {
                "kv_connector": "MoRIIOConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "read_mode": True,
                    "pd_replay_retention": True,
                },
            }
            disabled = {
                **child,
                "kv_connector_extra_config": {
                    "read_mode": True,
                    "pd_replay_retention": False,
                },
            }
            config.kv_transfer_config = KVTransferConfig(
                kv_connector="MultiConnector",
                kv_role="kv_producer",
                kv_connector_extra_config={
                    "connectors": [
                        child if kind.startswith("multi_duplicate") else disabled,
                        disabled if kind == "multi_disabled" else child,
                    ]
                },
            )
        layers = {
            name: KVCacheLayerConfig(
                group.manager_config.replace_params(storage_retention_tokens=0)
                if group.manager_cls is SlidingWindowManager
                else group.manager_config,
                spec,
            )
            for group in manager.kv_cache_config.kv_cache_groups
            for name, spec in group.kv_cache_specs.items()
        }
        if kind == "multi_duplicate_full":
            layers = {"layer2": layers["layer2"]}
        if kind.startswith("multi_duplicate"):
            with pytest.raises(ValueError, match="Only one .* connector"):
                get_global_kv_cache_groups(config, [layers])
            return
        if kind == "unsupported":
            layers["layer0"] = replace(
                layers["layer0"],
                manager_config=DFlashSlidingWindowManager.config(sliding_window=128),
            )
            with pytest.raises(ValueError, match="plain SlidingWindowManager"):
                get_global_kv_cache_groups(config, [layers])
            return
        planned = get_global_kv_cache_groups(config, [layers])
        margin = 0 if kind in ("no_connector", "multi_disabled") else 256
        for group in planned:
            if group.manager_cls is not SlidingWindowManager:
                continue
            policy = group.manager_config
            baseline = policy.replace_params(storage_retention_tokens=0)
            assert isinstance(policy.params, SlidingWindowManager.Params)
            assert policy.params.storage_retention_tokens == margin
            assert repr(policy) == repr(baseline)
            assert SlidingWindowManager.sliding_window_reach(
                policy
            ) == SlidingWindowManager.sliding_window_reach(baseline)
            cap = SlidingWindowManager.max_admission_blocks(
                policy, group.block_size, 8192, 32768
            )
            old = SlidingWindowManager.max_admission_blocks(
                baseline, group.block_size, 8192, 32768
            )
            assert cap is not None and old is not None
            assert cap == old + margin // group.block_size
            mask_args = (0, 300, 256, group.block_size)
            assert SlidingWindowManager.reachable_block_mask(
                *mask_args, policy
            ) == SlidingWindowManager.reachable_block_mask(*mask_args, baseline)
    finally:
        connector.shutdown()


def test_real_scheduler_confirms_pd_load_before_releasing_retention(pd_model):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.structured_output import StructuredOutputManager

    producer, template = new_manager(), new_manager()
    p = connector_for(producer, pd_model, producer=True)
    d = connector_for(template, pd_model, producer=False)
    source = request_of("core-source", 1124)
    source.kv_transfer_params = {"do_remote_decode": True, "transfer_id": "core-read"}
    assert producer.allocate_slots(source, 1124) is not None
    source.num_computed_tokens = 1124
    source.status = RequestStatus.FINISHED_LENGTH_CAPPED
    _, params = p.request_finished_all_groups(
        source, producer.get_block_ids(source.request_id)
    )
    config = d.connector_scheduler.vllm_config
    config.cache_config.num_gpu_blocks = 6000
    scheduler = Scheduler(
        config,
        template.kv_cache_config,
        StructuredOutputManager(config),
        block_size=256,
        hash_block_size=4,
    )
    try:
        target = request_of("core-target", 1124, max_tokens=2)
        target.kv_transfer_params = params
        scheduler.add_request(target)
        scheduled = scheduler.schedule()
        connector = scheduler.connector
        assert isinstance(connector, MoRIIOConnector)
        assert connector.connector_scheduler is not None
        policy = connector.connector_scheduler.pd_replay
        assert policy is not None
        assert target.request_id in policy.allocations
        assert scheduled.kv_connector_metadata is not None
        connector.bind_connector_metadata(scheduled.kv_connector_metadata)
        completion = connector.build_connector_worker_meta()
        connector.clear_connector_metadata()
        assert target.request_id in policy.allocations
        scheduler.update_from_output(
            scheduled,
            ModelRunnerOutput(
                req_ids=[target.request_id],
                sampled_token_ids=[[1]],
                kv_connector_output=KVConnectorOutput(
                    kv_connector_worker_meta=completion
                ),
            ),
        )
        assert target.status == RequestStatus.RUNNING
        assert scheduler.has_request(target.request_id)
        assert target.request_id not in policy.allocations
    finally:
        scheduler.shutdown()
        p.shutdown()
        d.shutdown()


@pytest.mark.parametrize("invalid", ["prompt", "target_null"])
def test_pd_validates_peer_settings_and_real_block_ranges(invalid):
    producer, consumer = new_manager(), new_manager()
    p = PDReplayRetention(producer, 0, producer=True)
    d = PDReplayRetention(consumer, 0, producer=False)
    source = request_of("validation-source", 1124)
    source.kv_transfer_params = {"do_remote_decode": True}
    assert producer.allocate_slots(source, 1124) is not None
    source.num_computed_tokens = 1124
    source_ids, params = p.export(source, producer.get_block_ids(source.request_id))
    target = request_of("validation-target", 1124)
    target.kv_transfer_params = {
        "do_remote_prefill": True,
        "pd_replay": params,
        "remote_block_ids": source_ids,
    }
    try:
        if invalid == "prompt":
            params["prompt_tokens"] -= 1
            with pytest.raises(VLLMValidationError, match="prompt"):
                consumer.allocate_slots(target, 1, num_external_computed_tokens=1123)
        else:
            assert (
                consumer.allocate_slots(target, 1, num_external_computed_tokens=1123)
                is not None
            )
            local = [list(g) for g in consumer.get_block_ids(target.request_id)]
            local[0][params["ranges"][0][0]] = 0
            with pytest.raises(RuntimeError, match="null or missing blocks"):
                d.pair(target, local, 1123)
    finally:
        producer.free(source)
        consumer.free(target)


def test_bounded_replay_group_keeps_only_the_normal_pd_window():
    manager = new_manager()
    groups = list(manager.kv_cache_config.kv_cache_groups)
    groups[0] = replace(
        groups[0],
        manager_config=groups[0].manager_config.replace_params(
            bounded_replay=True, storage_retention_tokens=1
        ),
    )
    cache = replace(manager.kv_cache_config, kv_cache_groups=groups)
    manager = KVCacheManager(
        cache, max_model_len=32768, scheduler_block_size=256, hash_block_size=4
    )
    policy = PDReplayRetention(manager, 0, producer=True)
    assert policy.ranges(1023, 1023)[:2] == ((13, 16), (10, 16))


@pytest.mark.parametrize("first_hit,next_hit", [(512, 1024), (1024, 512)])
def test_pd_failed_allocation_retries_pair_from_the_new_local_hit(first_hit, next_hit):
    producer, consumer = new_manager(), new_manager()
    p = PDReplayRetention(producer, 0, producer=True)
    d = PDReplayRetention(consumer, 0, producer=False)
    source = request_of("retry-source", 1124)
    source.kv_transfer_params = {"do_remote_decode": True}
    assert producer.allocate_slots(source, 1124) is not None
    source.num_computed_tokens = 1124
    remote, bundle = p.export(source, producer.get_block_ids(source.request_id))
    target = request_of("retry-target", 1124)
    target.kv_transfer_params = dict(
        do_remote_prefill=True, pd_replay=bundle, remote_block_ids=remote
    )
    for attempt, hit in enumerate((first_hit, next_hit)):
        warm = request_of(f"warm-{attempt}", hit + 1)
        assert consumer.allocate_slots(warm, hit) is not None
        consumer.free(warm)
        hits, local, _ = consumer.get_computed_blocks(target)
        assert local == hit
        allocated = consumer.allocate_slots(
            target,
            1,
            new_computed_blocks=hits,
            num_new_computed_tokens=local,
            num_external_computed_tokens=1123 - local,
            reserved_blocks=6001 if attempt == 0 else 0,
        )
        if attempt == 0:
            assert allocated is None
            assert target.request_id not in d.allocations
            assert all(
                target.request_id not in group.storage_retention_starts
                for group in d.managers
            )
            assert consumer.reset_prefix_cache()
        else:
            assert allocated is not None
            assert target.request_id in d.allocations
            assert all(
                target.request_id in group.storage_retention_starts
                for group in d.managers
                if isinstance(group, SlidingWindowManager)
            )
            dst, src = d.pair(
                target, consumer.get_block_ids(target.request_id), 1123 - local
            )
            for g, ((start, end), size) in enumerate(
                zip(bundle["ranges"], (64, 64, 256, 8, 4))
            ):
                lo, hi = max(start, local // size), min(end, cdiv(1123, size))
                assert dst[g] == consumer.get_block_ids(target.request_id)[g][lo:hi]
                assert src[g] == remote[g][lo - start : hi - start]
    consumer.free(target)
    producer.free(source)


@pytest.mark.parametrize("lookahead", [0, 3, 5])
def test_pd_reprefill_storage_does_not_require_null_blocks_from_a_local_hit(lookahead):
    producer = new_manager(lookahead, extra_retained_tokens=16)
    consumer = new_manager(lookahead, extra_retained_tokens=16)
    length = 1024 + max(1, lookahead)
    for manager in (producer, consumer):
        warm = request_of("warm", length, lookahead)
        assert manager.allocate_slots(warm, length) is not None
        manager.free(warm)
    p = PDReplayRetention(producer, lookahead, producer=True)
    d = PDReplayRetention(consumer, lookahead, producer=False)
    source = request_of("hit-source", length, lookahead)
    source.kv_transfer_params = {"do_remote_decode": True}
    hits, local, _ = producer.get_computed_blocks(source)
    assert local == 1024
    assert (
        producer.allocate_slots(
            source,
            length - local,
            new_computed_blocks=hits,
            num_new_computed_tokens=local,
        )
        is not None
    )
    source.num_computed_tokens = length
    remote, bundle = p.export(source, producer.get_block_ids(source.request_id))
    for group, (start, _) in zip(p.managers, bundle["ranges"]):
        if isinstance(group, SlidingWindowManager):
            params = group.manager_config.params
            assert isinstance(params, SlidingWindowManager.Params)
            assert start >= local // group.block_size - cdiv(
                params.sliding_window - 1, group.block_size
            )
    target = request_of("hit-target", length, lookahead)
    target.kv_transfer_params = dict(
        do_remote_prefill=True, pd_replay=bundle, remote_block_ids=remote
    )
    hits, local, _ = consumer.get_computed_blocks(target)
    assert local == 1024
    assert (
        consumer.allocate_slots(
            target,
            1,
            new_computed_blocks=hits,
            num_new_computed_tokens=local,
            num_external_computed_tokens=length - 1 - local,
        )
        is not None
    )
    if length - 1 - local:
        dst, src = d.pair(
            target, consumer.get_block_ids(target.request_id), length - 1 - local
        )
        assert all(
            len(a) == len(b) and all(x > 0 for x in a + b) for a, b in zip(dst, src)
        )
    consumer.free(target)
    producer.free(source)


@pytest.mark.parametrize(
    "invalid", ["max_tokens", "signature", "ranges", "source_null"]
)
def test_pd_invalid_request_is_rejected_before_scheduler_enqueue(pd_model, invalid):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.structured_output import StructuredOutputManager

    producer, template = new_manager(), new_manager()
    policy = PDReplayRetention(producer, 0, producer=True)
    source = request_of("validation-source", 1124)
    source.kv_transfer_params = {"do_remote_decode": True}
    assert producer.allocate_slots(source, 1124) is not None
    source.num_computed_tokens = 1124
    remote, bundle = policy.export(source, producer.get_block_ids(source.request_id))
    connector = connector_for(template, pd_model, producer=invalid == "max_tokens")
    config = connector.connector_scheduler.vllm_config
    config.cache_config.num_gpu_blocks = 6000
    scheduler = Scheduler(
        config,
        template.kv_cache_config,
        StructuredOutputManager(config),
        block_size=256,
        hash_block_size=4,
    )
    try:
        bad = request_of("bad", 1124, max_tokens=2 if invalid == "max_tokens" else 1)
        if invalid == "max_tokens":
            bad.kv_transfer_params = {"do_remote_decode": True}
        else:
            bad.kv_transfer_params = dict(
                do_remote_prefill=True,
                remote_engine_id="source",
                pd_replay=bundle,
                remote_block_ids=remote,
            )
            if invalid == "source_null":
                remote[0][0] = 0
            else:
                bundle[invalid][0] = -1
        with pytest.raises(VLLMValidationError):
            scheduler.add_request(bad)
        assert not scheduler.has_request(bad.request_id)
        good = request_of("good", 1124)
        scheduler.add_request(good)
        step = scheduler.schedule()
        assert step.num_scheduled_tokens[good.request_id] == 1124
    finally:
        scheduler.shutdown()
        connector.shutdown()
        producer.free(source)


def test_pd_read_missing_engine_releases_retention_and_recomputes(pd_model):
    producer, consumer = new_manager(), new_manager()
    p = PDReplayRetention(producer, 0, producer=True)
    connector = connector_for(consumer, pd_model, producer=False)
    source, target = request_of("source", 1124), request_of("target", 1124)
    source.kv_transfer_params = {"do_remote_decode": True}
    assert producer.allocate_slots(source, 1124) is not None
    source.num_computed_tokens = 1124
    remote, bundle = p.export(source, producer.get_block_ids(source.request_id))
    target.kv_transfer_params = dict(
        do_remote_prefill=True, pd_replay=bundle, remote_block_ids=remote
    )
    try:
        assert (
            consumer.allocate_slots(target, 1, num_external_computed_tokens=1123)
            is not None
        )
        connector.update_state_after_alloc(
            target, consumer.get_blocks(target.request_id), 1123
        )
        assert (
            target.request_id not in connector.connector_scheduler.pd_replay.allocations
        )
        assert connector.get_num_new_matched_tokens(target, 0) == (0, False)
    finally:
        consumer.free(target)
        producer.free(source)
        connector.shutdown()


def test_pd_export_without_a_complete_tail_disables_transfer_for_that_request(pd_model):
    manager = new_manager()
    connector = connector_for(manager, pd_model, producer=True)
    request = request_of("unavailable-export", 1124)
    request.kv_transfer_params = {"do_remote_decode": True}
    try:
        assert manager.allocate_slots(request, 1124) is not None
        request.num_computed_tokens = 1124
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        ids = [list(group) for group in manager.get_block_ids(request.request_id)]
        ids[0][-1] = 0
        assert connector.request_finished_all_groups(request, ids) == (False, None)
        assert (
            request.request_id
            not in connector.connector_scheduler.pd_replay.allocations
        )
    finally:
        manager.free(request)
        connector.shutdown()
