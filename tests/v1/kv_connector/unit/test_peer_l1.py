# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native authorization protects block contents without scheduler RPC work."""

import select
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from uuid import uuid4

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.native import (
    CleanupQueue,
    NativeOwner,
    PlacementReplica,
)
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
    Grant,
    NativeOwnerChannel,
    OwnerBusy,
    peer_key,
)

pytestmark = pytest.mark.cpu_test


def wait_until(condition, timeout=5):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            pytest.fail("Peer L1 condition did not become true before deadline")
        threading.Event().wait(0.001)


class ManualClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


@pytest.fixture
def clock():
    return ManualClock()


@pytest.fixture
def owner_resources(clock):
    with ExitStack() as stack:
        owner = NativeOwner(
            f"vllm-peer-l1/test/{uuid4().hex}",
            17,
            128,
            window=10,
            pin_cap=64,
            log_capacity=16,
            clock=clock,
        )
        stack.callback(owner.close)
        cleanup = CleanupQueue()
        stack.callback(cleanup.close)
        channel = NativeOwnerChannel(owner.endpoint, owner.epoch, 1, cleanup)
        yield owner, channel, cleanup


def test_group_keys_aliases_and_duplicate_rows(owner_resources):
    owner, channel, _ = owner_resources
    a, b = [peer_key(b"x" * 32 + group.to_bytes(4, "little")) for group in (0, 1)]
    assert a != b
    owner.publish([a, b, a], [1, 1, 2])
    grant = channel.acquire_exact([a, b])
    assert grant.rows == (1, 1)
    assert owner.refs == 1
    assert owner.absorb() == ([1], [], [])
    channel.cancel_unsubmitted(grant.sequence)
    assert owner.absorb() == ([], [1], [])
    assert not owner.retire(1, withhold=False)
    assert channel.acquire_exact([a]) is None
    assert channel.acquire_exact([b]) is None


def test_window_absorb_release_and_wakeup(owner_resources):
    owner, channel, cleanup = owner_resources
    keys = [peer_key(bytes([i])) for i in range(11)]
    owner.publish(keys, list(range(1, 12)))
    grant = channel.acquire_exact(keys[:10])
    assert select.select([owner.wake_fd], [], [], 1)[0]
    with pytest.raises(OwnerBusy) as busy:
        channel.acquire_exact(keys[10:])
    assert busy.value.generation >= 0
    assert busy.value.window == 10
    for row in grant.rows:
        assert owner.retire(row, withhold=True)
    assert owner.admission_reserve() == 0
    assert owner.absorb() == ([], [], [])
    assert not owner.pending()
    # A stable pin does not force endless empty engine steps.
    cleanup.enqueue(
        cleanup.create(grant.endpoint, grant.epoch, grant.reader, grant.sequence)
    )
    wait_until(owner.pending)
    touch, free, returned = owner.absorb()
    assert touch == free == []
    assert set(returned) == set(grant.rows)
    assert owner.refs == 0
    assert owner.admission_reserve() == 10


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("before_absorb", [False, True])
def test_absorb_handoff_cannot_lose_a_release(owner_resources, queued, before_absorb):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
        OwnerDeferred,
    )

    owner, channel, _ = owner_resources
    key = peer_key(b"release")
    owner.publish([key], [1])
    if queued:
        owner.absorb(free_blocks=0)
        with pytest.raises(OwnerDeferred) as pending:
            channel.acquire_exact([key], defer=True)
        sequence = pending.value.sequence
    else:
        sequence = channel.acquire_exact([key]).sequence
    proceed = threading.Event()

    def release():
        assert proceed.wait(5)
        channel.cancel_unsubmitted(sequence)

    with ThreadPoolExecutor(1) as executor:
        released = executor.submit(release)
        if before_absorb:
            proceed.set()
            released.result(timeout=5)
        touched, freed, withheld = owner.absorb(free_blocks=100 if queued else None)
        proceed.set()
        released.result(timeout=5)
    touched2, freed2, withheld2 = owner.absorb()
    assert touched + touched2 == freed + freed2
    assert not withheld + withheld2
    assert not owner.refs


def test_freeze_unfreezes_after_busy_and_success(owner_resources):
    owner, channel, _ = owner_resources
    key = peer_key(b"freeze")
    owner.publish([key], [1])
    grant = channel.acquire_exact([key])
    assert not owner.freeze()
    owner.thaw()
    owner.absorb()
    channel.cancel_unsubmitted(grant.sequence)
    assert owner.absorb() == ([], [1], [])
    assert owner.freeze()
    assert channel.acquire_exact([key]) is None
    owner.thaw(clear=True)
    assert channel.acquire_exact([key]) is None
    owner.publish([key], [1])
    grant = channel.acquire_exact([key])
    assert grant is not None
    channel.cancel_unsubmitted(grant.sequence)


def test_replica_snapshot_deltas_and_authoritative_full_key(owner_resources):
    owner, channel, _ = owner_resources
    key = peer_key(b"replica")
    owner.publish([key], [1])
    replica = PlacementReplica([(owner.endpoint, owner.epoch)])
    try:
        wait_until(lambda: replica.query([key])[0, 0] == 1)
        collision = key[:8] + bytes(x ^ 0xFF for x in key[8:])
        assert replica.query([collision])[0, 0]
        assert channel.acquire_exact([collision]) is None
        owner.retire(1, withhold=False)
        wait_until(lambda: replica.query([key])[0, 0] == 0)
        owner.publish([key], [2])
        wait_until(lambda: replica.query([key])[0, 0] == 1)
        grant = channel.acquire_exact([key])
        assert grant.rows == (2,)
        channel.cancel_unsubmitted(grant.sequence)
    finally:
        replica.close()


def test_completion_watermark_waits_for_delayed_drafter_and_survives_empty_output():
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.worker import (
        CompletionWatermark,
    )
    from vllm.v1.outputs import KVConnectorOutput

    watermark = CompletionWatermark()
    first, second = threading.Event(), threading.Event()
    # post_forward can run before the drafter's final fence even exists.
    assert watermark.report() is None
    watermark.record(1, first.is_set)
    watermark.record(2, second.is_set)
    second.set()
    assert watermark.report() is None
    first.set()
    meta = watermark.report()
    assert meta.completed_step == 2
    output = KVConnectorOutput(kv_connector_worker_meta=meta)
    assert not output.is_empty()
    assert watermark.report() is None


@pytest.mark.parametrize("before_absorb", [False, True])
def test_absorb_handoff_cannot_lose_a_new_acquire(owner_resources, before_absorb):
    owner, channel, _ = owner_resources
    key = peer_key(b"new-pin")
    owner.publish([key], [1])
    proceed = threading.Event()

    def acquire():
        assert proceed.wait(5)
        return channel.acquire_exact([key])

    with ThreadPoolExecutor(1) as executor:
        acquired = executor.submit(acquire)
        if before_absorb:
            proceed.set()
            grant = acquired.result(timeout=5)
        touched, freed, withheld = owner.absorb()
        proceed.set()
        grant = acquired.result(timeout=5)
    touched2, freed2, withheld2 = owner.absorb()
    assert touched + touched2 == [1]
    assert not freed + freed2 + withheld + withheld2
    channel.cancel_unsubmitted(grant.sequence)
    assert owner.absorb() == ([], [1], [])


@pytest.fixture
def scheduler_resources(tmp_path, request):
    import torch
    from transformers import DeepseekV3Config, OPTConfig

    from tests.v1.core.utils import new_kv_cache_group
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        KVTransferConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        VllmConfig,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import (
        PeerHandshake,
        PeerMemory,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.scheduler import (
        PeerScheduler,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
    from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig

    settings = getattr(request, "param", {})
    large = bool(settings)
    flash = settings.get("flash", False)
    mtp = settings.get("mtp", False)
    max_length = settings.get("max_model_len", 33024 if large else 256)
    OPTConfig(
        hidden_size=64,
        num_attention_heads=2,
        ffn_dim=128,
        num_hidden_layers=1,
        vocab_size=128,
        max_position_embeddings=max_length,
    ).save_pretrained(tmp_path)
    if mtp:
        DeepseekV3Config(
            architectures=["DeepseekV3ForCausalLM"],
            hidden_size=64,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_hidden_layers=1,
            intermediate_size=128,
            moe_intermediate_size=32,
            n_routed_experts=4,
            num_experts_per_tok=2,
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
            num_nextn_predict_layers=2,
            max_position_embeddings=max_length,
        ).save_pretrained(tmp_path)
    model = ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=max_length,
    )
    parallel = ParallelConfig()
    config = VllmConfig(
        model_config=model,
        parallel_config=parallel,
        speculative_config=SpeculativeConfig(
            method="mtp",
            num_speculative_tokens=2,
            target_model_config=model,
            target_parallel_config=parallel,
        )
        if mtp
        else None,
        device_config=DeviceConfig("cpu"),
        cache_config=CacheConfig(block_size=16, enable_prefix_caching=True),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=max_length,
            max_num_seqs=2,
            max_num_batched_tokens=max_length,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="PeerL1Connector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "pin_window": settings.get("pin_window", 512 if large else 2),
                "pin_cap": settings.get("pin_cap", 4096),
            },
        ),
    )
    spec = AttentionSpec(
        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
    )
    cache = KVCacheConfig(
        32,
        [],
        [
            new_kv_cache_group(["full"], spec),
            new_kv_cache_group(
                ["fine"],
                AttentionSpec(
                    block_size=4, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
                ),
            ),
        ],
        hash_block_size=4,
        cache_hit_alignment_tokens=16,
    )
    if large:
        groups = []
        for group, size in enumerate(
            (32, 32, 32, 32, 128) if flash else (64, 64, 256, 8, 4)
        ):
            spec = AttentionSpec(
                block_size=size, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
            )
            policy = (
                spec
                if group == (4 if flash else 2)
                else SlidingWindowManager.layer_config(
                    spec,
                    sliding_window=8 if group == 4 else 128,
                    bounded_replay=flash,
                    extra_retained_tokens=config.num_reprefillable_tokens,
                )
            )
            cache_group = new_kv_cache_group([f"layer{group}"], policy)
            groups.append(cache_group)
        cache = KVCacheConfig(
            20000 if mtp or flash else 4096,
            [],
            groups,
            hash_block_size=128 if flash else 4,
            cache_hit_alignment_tokens=128 if flash else 256,
            prefix_cache_retention_interval=0,
        )
    manager = KVCacheManager(
        cache,
        max_model_len=max_length,
        enable_caching=True,
        scheduler_block_size=128 if flash else 256 if large else 16,
        hash_block_size=128 if flash else 4,
        num_hash_lookahead_tokens=config.num_hash_lookahead_tokens,
    )
    scheduler = PeerScheduler(
        config, cache, (4,) if flash else tuple(range(5)) if large else (0, 1)
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import init_none_hash

    init_none_hash(sha256)
    scheduler.bind(manager)
    endpoint = f"vllm-peer-l1/scheduler/{uuid4().hex}"
    member = PeerMemory(
        0, endpoint, 101, "test", 0, "same", cache.num_blocks, ((),), {}
    )
    owners = []
    members = [member]
    if large:
        for rank in range(1, 8):
            owner = NativeOwner(
                f"vllm-peer-l1/large/{uuid4().hex}",
                rank,
                cache.num_blocks,
                window=scheduler.window,
                pin_cap=scheduler.pin_cap,
            )
            owners.append(owner)
            members.append(
                PeerMemory(
                    rank,
                    owner.endpoint,
                    rank,
                    "test",
                    rank,
                    "same",
                    cache.num_blocks,
                    ((),),
                    {},
                )
            )
    scheduler.handshake(PeerHandshake(members))
    cleanup = CleanupQueue()
    channel = NativeOwnerChannel(endpoint, 101, 9, cleanup)
    try:
        yield scheduler, manager, owners if large else channel
    finally:
        cleanup.close()
        scheduler.close()
        for owner in owners:
            owner.close()


def test_publication_waits_for_gpu_watermark_and_ignores_partial_keys(
    scheduler_resources,
):
    from tests.v1.core.test_prefix_caching import make_request
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import (
        PeerWorkerMetadata,
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import (
        init_none_hash,
        make_block_hash_with_group_id,
    )
    from vllm.v1.outputs import KVConnectorOutput

    scheduler, manager, channel = scheduler_resources
    init_none_hash(sha256)
    request = make_request("publish", list(range(32)), 4, sha256)
    pool = manager.block_pool
    block = pool.get_new_blocks(1)[0]
    pool.cache_partial_block(request, block, 12, 0, 16)
    partial = peer_key(make_block_hash_with_group_id(request.block_hashes[2], 0))
    scheduler.build()
    scheduler.update(KVConnectorOutput(kv_connector_worker_meta=PeerWorkerMetadata(1)))
    assert channel.acquire_exact([partial]) is None
    pool.cache_full_blocks(request, [block], 0, 1, 16, 0)
    full = peer_key(make_block_hash_with_group_id(request.block_hashes[3], 0))
    scheduler.build()
    scheduler.update(KVConnectorOutput(kv_connector_worker_meta=PeerWorkerMetadata(1)))
    assert channel.acquire_exact([partial]) is None
    assert channel.acquire_exact([full]) is None
    scheduler.update(KVConnectorOutput(kv_connector_worker_meta=PeerWorkerMetadata(2)))
    assert channel.acquire_exact([partial]) is None
    grant = channel.acquire_exact([full])
    assert grant is not None
    channel.cancel_unsubmitted(grant.sequence)
    scheduler.build()
    pool.free_blocks([block])
    assert pool.reset_prefix_cache()
    assert channel.acquire_exact([full]) is None


@pytest.mark.parametrize("block_first", [True, False])
def test_group_byte_regions_exclude_overlaid_group_padding(block_first):
    import torch

    from tests.v1.core.utils import new_kv_cache_group
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.worker import (
        cache_segments,
    )
    from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheTensor

    spec = AttentionSpec(
        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
    )
    page = spec.page_size_bytes
    layer_stride, block_stride = (page, page * 3) if block_first else (page * 8, page)
    tensors = [
        KVCacheTensor(page * 24, ["a", "b"], layer_stride, block_stride),
        KVCacheTensor(page * 24, ["c"], layer_stride, block_stride),
    ]
    cache = KVCacheConfig(
        8,
        tensors,
        [new_kv_cache_group(["a", "b"], spec), new_kv_cache_group(["c"], spec)],
    )
    assert cache_segments(cache) == (
        ((0, block_stride, page), (layer_stride, block_stride, page)),
        ((0, block_stride, page),),
    )


@pytest.mark.parametrize("queued", [False, True])
def test_owner_restart_clears_old_replica_epoch_and_rejects_old_grants(
    owner_resources, queued
):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
        OwnerDeferred,
    )

    owner, channel, cleanup = owner_resources
    key = peer_key(b"incarnation")
    owner.publish([key], [1])
    replica = PlacementReplica([(owner.endpoint, owner.epoch)])
    try:
        wait_until(lambda: replica.query([key])[0, 0])
        if queued:
            owner.absorb(free_blocks=0)
            with pytest.raises(OwnerDeferred) as pending:
                channel.acquire_exact([key], defer=True)
            assert owner.pending() and not owner.refs
        owner.close()
        replacement = NativeOwner(owner.endpoint, owner.epoch + 1, 128, window=10)
        try:
            replacement.publish([key], [2])
            wait_until(lambda: not replica.query([key])[0, 0])
            assert channel.acquire_exact([key]) is None
            current = NativeOwnerChannel(
                replacement.endpoint, replacement.epoch, channel.reader, cleanup
            )
            grant = current.acquire_exact([key])
            if queued:
                assert (
                    channel.acquire_exact(
                        [key], defer=True, sequence=pending.value.sequence
                    )
                    is None
                )
                channel.cancel_unsubmitted(pending.value.sequence)
            assert replacement.refs == 1
            current.cancel_unsubmitted(grant.sequence)
        finally:
            replacement.close()
    finally:
        replica.close()


@pytest.mark.parametrize("withhold", [None, "pop", "evict"])
def test_idle_owner_applies_late_cleanup_to_real_pool(scheduler_resources, withhold):
    from tests.v1.core.test_prefix_caching import make_request
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import (
        PeerWorkerMetadata,
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import (
        init_none_hash,
        make_block_hash_with_group_id,
    )
    from vllm.v1.outputs import KVConnectorOutput

    scheduler, manager, channel = scheduler_resources
    init_none_hash(sha256)
    request = make_request("idle", list(range(32)), 4, sha256)
    pool = manager.block_pool
    block = pool.get_new_blocks(1)[0]
    pool.cache_full_blocks(request, [block], 0, 1, 16, 0)
    scheduler.build()
    scheduler.update(KVConnectorOutput(kv_connector_worker_meta=PeerWorkerMetadata(1)))
    pool.free_blocks([block], evict_first=True)
    key = peer_key(make_block_hash_with_group_id(request.block_hashes[3], 0))
    woke = threading.Event()
    scheduler.set_engine_wakeup(woke.set)
    grant = channel.acquire_exact([key])
    assert woke.wait(timeout=5)
    allocated = []
    if withhold == "evict":
        pool.evict_blocks({block.block_id})
        assert block.block_hash is None
        assert pool.free_block_queue.num_free_blocks == 30
    elif withhold:
        allocated = pool.get_new_blocks(1)
        assert allocated[0] is not block
        assert block.ref_cnt == 0
    scheduler.build()
    assert not scheduler.pending()
    assert block.ref_cnt == (0 if withhold else 1)
    woke.clear()
    # No local requests are running when the remote completion arrives.
    channel.cancel_unsubmitted(grant.sequence)
    assert woke.wait(timeout=5)
    scheduler.build()
    assert block.ref_cnt == 0
    assert not scheduler.pending()
    if withhold:
        assert block.block_hash is None
    else:
        assert pool.get_cached_block(request.block_hashes[3], [0]) == [block]
    pool.free_blocks(allocated)
    assert pool.reset_prefix_cache()


@pytest.mark.parametrize("peer_count", [2, 64, 65])
def test_replica_owner_membership_survives_retirement_and_log_rotation(peer_count):
    """Bitmap bit 63 and list fallback retain other owners of a shared key."""
    key = peer_key(b"shared")
    with ExitStack() as stack:
        owners = []
        for index in range(2):
            owner = NativeOwner(
                f"vllm-peer-l1/membership/{uuid4().hex}",
                index + 1,
                128,
                window=10,
                log_capacity=16,
            )
            stack.callback(owner.close)
            owner.publish([key], [1])
            owners.append(owner)
        # Repeated subscriptions exercise the highest owner ID with two servers.
        peers = [(owners[0].endpoint, 1)] * (peer_count - 1) + [(owners[1].endpoint, 2)]
        replica = PlacementReplica(peers)
        stack.callback(replica.close)
        wait_until(lambda: replica.query([key]).all())
        owners[0].retire(1, withhold=False)
        wait_until(lambda: not replica.query([key])[0, :-1].any())
        assert replica.query([key])[0, -1]
        for index in range(96):
            owners[1].publish([peer_key(index.to_bytes(4, "little"))], [2])
            owners[1].retire(2, withhold=False)
        owners[1].retire(1, withhold=False)
        wait_until(lambda: not replica.query([key]).any())
        owners[1].publish([key], [3])
        wait_until(lambda: replica.query([key])[0, -1])


@pytest.mark.parametrize(
    ("ignore_priority", "interposition", "disable_tools", "valid"),
    [
        (None, "1", None, False),
        ("1", "1", None, False),
        ("0", "0", None, False),
        ("0", "1", None, True),
        ("0", "0", "1", True),
    ],
)
def test_rocm_copy_checks_priority_preserving_environment(
    monkeypatch, ignore_priority, interposition, disable_tools, valid
):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.worker import (
        check_rocm_copy_environment,
    )

    for name, value in (
        ("DEBUG_HIP_IGNORE_STREAM_PRIORITY", ignore_priority),
        ("ROCPROFILER_QUEUE_INTERPOSITION", interposition),
        ("HSA_TOOLS_DISABLE_REGISTER", disable_tools),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert check_rocm_copy_environment() is valid


@pytest.mark.parametrize("stalled", [False, True])
def test_fan_in_waits_for_owner_progress_with_a_fixed_deadline(
    scheduler_resources, monkeypatch, stalled
):
    """Seven readers survive >2 local steps, or fall back at a wall deadline."""
    from copy import deepcopy

    from tests.v1.core.test_prefix_caching import make_request
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1 import scheduler as policy
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import (
        PeerHandshake,
        PeerMemory,
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        init_none_hash,
        make_block_hash_with_group_id,
    )

    base, _, _ = scheduler_resources
    init_none_hash(sha256)
    now = [0.0]
    # Only the scheduler's external clock is controlled; native RPCs are real.
    monkeypatch.setattr(policy, "monotonic", lambda: now[0], raising=False)
    with ExitStack() as stack:
        owner = NativeOwner(f"vllm-peer-l1/fan-in/{uuid4().hex}", 99, 512, window=10)
        stack.callback(owner.close)
        owner_member = PeerMemory(
            99, owner.endpoint, 99, "test", 0, "same", 512, ((),), {}
        )
        readers = []
        for rank in range(7):
            config = deepcopy(base.config)
            config.kv_transfer_config.kv_connector_extra_config.update(
                pin_window=10, busy_timeout_ms=1000
            )
            reader = policy.PeerScheduler(config, base.cache, (0, 1))
            stack.callback(reader.close)
            manager = KVCacheManager(
                base.cache,
                max_model_len=256,
                enable_caching=True,
                scheduler_block_size=16,
                hash_block_size=4,
            )
            reader.bind(manager)
            member = PeerMemory(
                0,
                f"vllm-peer-l1/reader/{uuid4().hex}",
                rank + 1,
                "test",
                0,
                "same",
                32,
                ((),),
                {},
            )
            reader.handshake(PeerHandshake([member, owner_member]))
            readers.append(reader)
        pending, keys = [], []
        for index in range(28):
            request = make_request(
                f"burst-{index}", list(range(index * 100, index * 100 + 33)), 4, sha256
            )
            pending.append((readers[index % 7], request))
            for group, size in ((0, 16), (1, 4)):
                for end in range(size, 33, size):
                    keys.append(
                        peer_key(
                            make_block_hash_with_group_id(
                                request.block_hashes[end // 4 - 1], group
                            )
                        )
                    )
        owner.publish(keys, list(range(1, len(keys) + 1)))
        for reader in readers:
            wait_until(lambda r=reader: r._replica.query(keys).all())
        first_reader, first_request = pending.pop(0)
        assert first_reader.match(first_request, 0) == (32, False)
        first_reader.finished(first_request.request_id)
        for _ in range(2):
            for reader, request in pending:
                assert reader.match(request, 0) == (None, False)
        for reader, request in pending:
            assert reader.match(request, 32) == (0, False)
        if stalled:
            now[0] = 1.001
            for reader, request in pending:
                assert reader.match(request, 0) == (0, False)
                assert reader.match(request, 0) == (0, False)
        else:
            while pending:
                generations = [r._replica.generation(0) for r in readers]
                owner.absorb()
                now[0] += 0.01
                for reader, generation in zip(readers, generations):
                    wait_until(
                        lambda r=reader, g=generation: r._replica.generation(0) > g
                    )
                remaining = []
                for reader, request in pending:
                    hit, _ = reader.match(request, 0)
                    if hit is None:
                        remaining.append((reader, request))
                    else:
                        assert hit == 32
                        reader.finished(request.request_id)
                assert len(remaining) < len(pending)
                pending = remaining
        assert owner.refs == 0
        owner.absorb()
        assert owner.admission_reserve() == 10


@pytest.mark.parametrize("scheduler_resources", [{"mtp": True}], indirect=True)
def test_peer_connector_keeps_mtp_hash_lookahead(scheduler_resources):
    scheduler, _, _ = scheduler_resources
    config = scheduler.config
    assert config.num_prefill_lookahead_tokens == 2
    assert config.num_reprefillable_tokens == 1
    assert config.num_hash_lookahead_tokens == 2


@pytest.mark.parametrize("scheduler_resources", [{"mtp": True}], indirect=True)
@pytest.mark.parametrize("prompt_length", [257, 258, 9967, 10240])
def test_mtp_lookahead_peer_hit_matches_local_five_group_hit(
    scheduler_resources, prompt_length
):
    """Lookahead keeps the core MTP hit length, including sparse replay tails."""
    from tests.v1.core.test_prefix_caching import make_request
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import (
        PeerWorkerMetadata,
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_group_id, init_none_hash
    from vllm.v1.outputs import KVConnectorOutput

    scheduler, manager, owners = scheduler_resources
    assert scheduler.config.num_hash_lookahead_tokens == 2
    init_none_hash(sha256)
    request = make_request(
        "mtp-prefix",
        list(range(prompt_length)),
        4,
        sha256,
        num_hash_lookahead_tokens=2,
    )
    assert manager.allocate_slots(request, prompt_length) is not None
    _, local_tokens, _ = manager.get_computed_blocks(request)
    assert local_tokens == (prompt_length - 2) // 256 * 256
    keys, rows = [], []
    for block in manager.block_pool.blocks:
        if block.block_hash is None or block.block_hash_num_tokens is None:
            continue
        group = scheduler.cache.kv_cache_groups[get_group_id(block.block_hash)]
        if block.block_hash_num_tokens % group.block_size == 0:
            keys.append(peer_key(block.block_hash))
            rows.append(block.block_id)
    channel = NativeOwnerChannel(
        scheduler._owner.endpoint, scheduler._owner.epoch, 901, scheduler._cleanup
    )
    assert all(channel.acquire_exact([key]) is None for key in keys)
    scheduler.build()
    scheduler.update(KVConnectorOutput(kv_connector_worker_meta=PeerWorkerMetadata(0)))
    assert all(channel.acquire_exact([key]) is None for key in keys)
    scheduler.update(KVConnectorOutput(kv_connector_worker_meta=PeerWorkerMetadata(1)))
    grant = channel.acquire_exact(keys)
    assert grant is not None
    channel.cancel_unsubmitted(grant.sequence)
    owners[0].publish(keys, rows)
    wait_until(lambda: scheduler._replica.query(keys)[:, 0].all())
    assert scheduler.match(request, 0) == (local_tokens, False)
    scheduler.finished(request.request_id)
    sibling = make_request(
        "mtp-different-lookahead",
        request.prompt_token_ids[:-2] + [prompt_length + 1, prompt_length + 2],
        4,
        sha256,
        num_hash_lookahead_tokens=2,
    )
    _, expected, _ = manager.get_computed_blocks(sibling)
    assert scheduler.match(sibling, 0) == (expected, False)
    if prompt_length == 258:
        assert expected == 0 < local_tokens
    scheduler.finished(sibling.request_id)


@pytest.mark.parametrize("scheduler_resources", [{"mtp": True}], indirect=True)
def test_peer_stats_cross_real_scheduler_and_prometheus(scheduler_resources):
    from multiprocessing import Pipe

    from prometheus_client import REGISTRY

    from tests.v1.core.test_prefix_caching import make_request
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorProm
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.connector import (
        PeerL1Connector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import (
        PeerHandshake,
        PeerMemory,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.stats import PeerL1Stats
    from vllm.utils.hashing import sha256
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.metrics.prometheus import unregister_vllm_metrics
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
    from vllm.v1.structured_output import StructuredOutputManager

    base, _, _ = scheduler_resources
    base.config.cache_config.num_gpu_blocks = base.cache.num_blocks
    scheduler = Scheduler(
        base.config,
        base.cache,
        StructuredOutputManager(base.config),
        block_size=256,
        hash_block_size=4,
        log_stats=True,
    )
    unregister_vllm_metrics()
    try:
        member = PeerMemory(
            0,
            f"vllm-peer-l1/stats-engine/{uuid4().hex}",
            991,
            "test",
            0,
            "same",
            base.cache.num_blocks,
            ((),),
            {},
        )
        scheduler.connector.set_xfer_handshake_metadata({0: PeerHandshake([member])})
        request = make_request(
            "stats-local", list(range(1026)), 4, sha256, num_hash_lookahead_tokens=2
        )
        assert scheduler.connector.get_num_new_matched_tokens(request, 0) == (0, False)
        scheduled = scheduler.schedule()
        worker = PeerL1Stats()
        worker.inc("worker_copy_bytes", 3072)
        worker.observe("worker_copy_gpu_seconds", 0.002)
        output = ModelRunnerOutput(
            req_ids=[],
            kv_connector_output=KVConnectorOutput(
                kv_connector_stats=worker.clone_and_reset()
            ),
        )
        receiver, sender = Pipe(duplex=False)
        with receiver, sender:
            sender.send(output)
            received = receiver.recv()
        assert worker.is_empty()
        result = scheduler.update_from_output(scheduled, received)[0]
        wire = MsgpackEncoder().encode(result)
        decoded = MsgpackDecoder(EngineCoreOutputs).decode(wire)
        data = decoded.scheduler_stats.kv_connector_stats
        rebuilt = PeerL1Connector.build_kv_connector_stats(data)
        assert rebuilt.reduce()["worker_copy_bytes"] > 0
        prom = KVConnectorProm(
            base.config, ["model_name", "engine"], {0: ["test", "0"], 1: ["test", "1"]}
        )
        prom.observe(data, engine_idx=1)
        labels = {"model_name": "test", "engine": "1"}
        name = "vllm:peer_l1_"
        for metric in ("worker_copy_bytes_total", "worker_copy_gpu_seconds_count"):
            value = REGISTRY.get_sample_value(name + metric, labels)
            assert value is not None and value > 0
        assert REGISTRY.get_sample_value(name + "owner_reserve", labels) is not None
    finally:
        scheduler.shutdown()
        unregister_vllm_metrics()


@pytest.fixture
def adaptive_peer(scheduler_resources):
    from copy import deepcopy

    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import (
        PeerHandshake,
        PeerMemory,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.scheduler import (
        PeerScheduler,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import KVCacheConfig

    base, _, _ = scheduler_resources
    with ExitStack() as stack:

        def create(free, maximum):
            config = deepcopy(base.config)
            config.kv_transfer_config.kv_connector_extra_config["pin_window"] = maximum
            cache = KVCacheConfig(
                free + 1,
                [],
                [base.cache.kv_cache_groups[0]],
                hash_block_size=16,
                cache_hit_alignment_tokens=16,
            )
            manager = KVCacheManager(
                cache,
                max_model_len=1024,
                enable_caching=True,
                scheduler_block_size=16,
                hash_block_size=16,
            )
            peer = PeerScheduler(config, cache, (0,))
            stack.callback(peer.close)
            peer.bind(manager)
            member = PeerMemory(
                0,
                f"vllm-peer-l1/adaptive/{uuid4().hex}",
                19,
                "test",
                0,
                "same",
                free + 1,
                ((),),
                {},
            )
            peer.handshake(PeerHandshake([member]))
            channel = NativeOwnerChannel(member.endpoint, 19, 91, peer._cleanup)
            return peer, manager.block_pool, channel

        yield create


def _publish_blocks(peer, pool, count):
    from tests.v1.core.test_prefix_caching import make_request
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import (
        PeerWorkerMetadata,
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import (
        init_none_hash,
        make_block_hash_with_group_id,
    )
    from vllm.v1.outputs import KVConnectorOutput

    init_none_hash(sha256)
    request = make_request(
        f"publish-{uuid4().hex}", list(range(count * 16 + 1)), 16, sha256
    )
    blocks = pool.get_new_blocks(count)
    pool.cache_full_blocks(request, blocks, 0, count, 16, 0)
    peer.build()
    peer.update(
        KVConnectorOutput(kv_connector_worker_meta=PeerWorkerMetadata(peer.step))
    )
    keys = [
        peer_key(make_block_hash_with_group_id(request.block_hashes[i], 0))
        for i in range(count)
    ]
    return blocks, keys


@pytest.mark.parametrize(
    ("free", "maximum"), [(100, 10), (100, 100), (10, 10), (8, 10)]
)
def test_adaptive_window_preserves_multigroup_allocation(adaptive_peer, free, maximum):
    peer, pool, channel = adaptive_peer(free, maximum)
    effective = min(maximum, free // 9)
    assert peer.admission_reserve() == effective
    # Publish before the test window, then return rows to the queue head.
    blocks, keys = _publish_blocks(peer, pool, effective)
    pool.free_blocks(blocks, evict_first=True)
    peer.build()
    effective = min(maximum, free // 9)
    grant = channel.acquire_exact(keys) if keys else None
    available = free - effective
    assert pool.get_num_free_blocks() == available
    first = pool.get_new_blocks(available // 2)
    assert not {b.block_id for b in blocks}.intersection(b.block_id for b in first)
    assert peer.admission_reserve() == peer._owner.admission_reserve() == 0
    second = pool.get_new_blocks(available - len(first))
    assert len(first) + len(second) == available
    assert pool.get_num_free_blocks() == 0
    pool.free_blocks(first + second)
    if grant is not None:
        channel.cancel_unsubmitted(grant.sequence)
    peer.build()
    assert pool.free_block_queue.num_free_blocks == free
    assert peer._owner.absorb_stats()["owner_effective_window"] == min(
        maximum, free // 9
    )
    assert pool.reset_prefix_cache()


def test_adaptive_budget_completes_in_use_touches_before_allocation(adaptive_peer):
    peer, pool, channel = adaptive_peer(100, 100)
    blocks, keys = _publish_blocks(peer, pool, 9)
    # Nine new pins are all in-use, so absorb must not charge nine free rows.
    assert peer.admission_reserve() == 10
    grant = channel.acquire_exact(keys)
    local = pool.get_new_blocks(pool.get_num_free_blocks())
    assert pool.free_block_queue.num_free_blocks == 10
    peer.build()
    assert all(block.ref_cnt == 2 for block in blocks)
    assert peer.admission_reserve() == 1
    assert pool.get_num_free_blocks() == 9
    channel.cancel_unsubmitted(grant.sequence)
    peer.build()
    assert all(block.ref_cnt == 1 for block in blocks)
    pool.free_blocks(blocks + local)
    peer.build()
    assert pool.reset_prefix_cache()


def test_pinned_in_use_invalidation_adopts_a_pool_lease(adaptive_peer):
    peer, pool, channel = adaptive_peer(100, 10)
    blocks, keys = _publish_blocks(peer, pool, 1)
    block = blocks[0]
    grant = channel.acquire_exact(keys)
    pool.evict_blocks({block.block_id})
    assert block.ref_cnt == 2
    pool.free_blocks([block], evict_first=True)
    allocated = pool.get_new_blocks(pool.get_num_free_blocks())
    assert block not in allocated
    peer.build()
    assert block.ref_cnt == 1
    channel.cancel_unsubmitted(grant.sequence)
    peer.build()
    assert block.ref_cnt == 0
    pool.free_blocks(allocated)
    peer.build()
    assert pool.reset_prefix_cache()


@pytest.mark.parametrize(
    ("free", "maximum", "fast_rows"),
    [(100, 10, 0), (18, 10, 0), (8, 10, 0), (100, 10, 10)],
)
def test_deferred_free_rows_keep_half_the_owner_headroom(
    adaptive_peer, free, maximum, fast_rows
):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
        OwnerDeferred,
    )

    source, pool, channel = adaptive_peer(free, maximum)
    count = (free - fast_rows) // 2
    blocks, keys = _publish_blocks(source, pool, fast_rows + count + 1)
    pool.free_blocks(blocks)
    source.build()
    immediate = channel.acquire_exact(keys[:fast_rows]) if fast_rows else None
    keys = keys[fast_rows:]
    with pytest.raises(OwnerDeferred) as denied:
        channel.acquire_exact(keys, defer=True)
    with pytest.raises(OwnerDeferred) as pending:
        channel.acquire_exact(keys[:count], defer=True)
    source.build()
    grant = channel.acquire_exact(
        keys[:count], defer=True, sequence=pending.value.sequence
    )
    assert len(grant.rows) == count
    assert (
        channel.acquire_exact(keys, defer=True, sequence=denied.value.sequence) is None
    )
    assert pool.free_block_queue.num_free_blocks == free - count - fast_rows
    assert source.admission_reserve() == min(maximum, (free - count - fast_rows) // 9)
    channel.release(grant)
    if immediate is not None:
        channel.release(immediate)
    wait_until(lambda: source._owner.refs == 0)
    source.build()
    assert pool.free_block_queue.num_free_blocks == free
    assert all(block.ref_cnt == 0 for block in blocks)


def test_deferred_queue_blocks_reset_until_cancelled(adaptive_peer):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
        OwnerDeferred,
    )

    source, pool, channel = adaptive_peer(100, 10)
    blocks, keys = _publish_blocks(source, pool, 11)
    pool.free_blocks(blocks)
    source.build()
    with pytest.raises(OwnerDeferred) as pending:
        channel.acquire_exact(keys, defer=True)
    assert all(block.ref_cnt == 0 for block in blocks)
    assert not pool.reset_prefix_cache()
    assert all(block.block_hash is not None for block in blocks)
    channel.cancel_unsubmitted(pending.value.sequence)
    source.build()
    assert pool.reset_prefix_cache()
    assert all(block.block_hash is None for block in blocks)
    assert channel.acquire_exact(keys) is None
    assert (
        channel.acquire_exact(keys, defer=True, sequence=pending.value.sequence) is None
    )


def test_deferred_acquire_revalidates_keys_at_absorb(adaptive_peer):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
        OwnerDeferred,
    )

    source, pool, channel = adaptive_peer(100, 10)
    blocks, keys = _publish_blocks(source, pool, 11)
    with pytest.raises(OwnerDeferred) as pending:
        channel.acquire_exact(keys, defer=True)
    pool.evict_blocks([blocks[0].block_id])
    source.build()
    assert (
        channel.acquire_exact(keys, defer=True, sequence=pending.value.sequence) is None
    )
    assert not source._owner.refs and all(block.ref_cnt == 1 for block in blocks)


def test_absorbed_long_hits_keep_the_immediate_native_path(adaptive_peer):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
        OwnerDeferred,
    )

    source, pool, channel = adaptive_peer(100, 10)
    _, keys = _publish_blocks(source, pool, 11)
    with pytest.raises(OwnerDeferred) as pending:
        channel.acquire_exact(keys, defer=True)
    source.build()
    first = channel.acquire_exact(keys, defer=True, sequence=pending.value.sequence)
    second = channel.acquire_exact(keys, defer=True)
    assert first.rows == second.rows
    assert source._owner.refs == 22
    source.get_stats()
    source.build()
    assert source.get_stats().reduce()["owner_deferred_grants"] == 0
    channel.cancel_unsubmitted(first.sequence)
    channel.cancel_unsubmitted(second.sequence)
    source.build()
    assert not source._owner.refs


def test_deferred_enqueue_wakes_idle_owner_maintenance(adaptive_peer):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
        OwnerDeferred,
    )

    source, pool, channel = adaptive_peer(100, 10)
    blocks, keys = _publish_blocks(source, pool, 11)
    wake = threading.Event()
    source.set_engine_wakeup(wake.set)
    assert not source.pending()
    with pytest.raises(OwnerDeferred) as pending:
        channel.acquire_exact(keys, defer=True)
    assert wake.wait(5)
    assert source.pending() and source._owner.refs == 0
    source.build()  # The owner's scheduler thread handles the maintenance wake.
    grant = channel.acquire_exact(keys, defer=True, sequence=pending.value.sequence)
    assert len(grant.rows) == 11
    assert all(block.ref_cnt == 2 for block in blocks)
    assert not source.pending()
    channel.cancel_unsubmitted(grant.sequence)
    source.build()
    assert all(block.ref_cnt == 1 for block in blocks)


def test_acquire_after_refcount_snapshot_waits_for_the_next_absorb(adaptive_peer):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
        OwnerDeferred,
    )

    source, pool, channel = adaptive_peer(100, 10)
    blocks, keys = _publish_blocks(source, pool, 11)
    assert source._owner.deferred_rows() == []
    with pytest.raises(OwnerDeferred) as pending:
        channel.acquire_exact(keys, defer=True)
    # Interleave the RPC after the scheduler snapshot, before its native absorb.
    assert source._owner.absorb(
        free_blocks=pool.free_block_queue.num_free_blocks, in_use_rows=[]
    ) == ([], [], [])
    assert source._owner.pending() and source._owner.refs == 0
    source.build()
    grant = channel.acquire_exact(keys, defer=True, sequence=pending.value.sequence)
    assert len(grant.rows) == 11
    assert all(block.ref_cnt == 2 for block in blocks)
    channel.cancel_unsubmitted(grant.sequence)
    source.build()
    assert all(block.ref_cnt == 1 for block in blocks)


@pytest.mark.parametrize("outcome", ["success", "before", "late", "abort", "step"])
def test_async_receives_keep_uncertain_dma_targets_and_grants(
    owner_resources, monkeypatch, outcome
):
    from concurrent.futures import Future

    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1 import async_load
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import CopyJob

    owner, _, cleanup = owner_resources
    channel = NativeOwnerChannel(owner.endpoint, owner.epoch, 71, cleanup, lease_s=1.0)
    key = peer_key(b"async")
    owner.publish([key], [1])
    grant = channel.acquire_exact([key])
    now = [time.monotonic()]
    monkeypatch.setattr(async_load, "monotonic", lambda: now[0])
    receives = async_load.AsyncReceives(channel.release)
    read = receives.add(
        CopyJob(0, grant, ((0, 1, 17),), "async-rid"),
        0.1 if outcome == "step" else 1.0,
    )
    assert read.deadline == pytest.approx(grant.acquired_before + 0.5)
    late_dma: Future[bytes] = Future()
    if outcome == "step":
        now[0] += 0.2
        assert not receives.poll().finished_recving
    if outcome in ("success", "late", "step"):
        assert read.begin()
    if outcome in ("success", "step"):
        read.complete(True, drained=True)
    elif outcome == "late":
        read.complete(False, drained=False, drain_query=late_dma.done)
    elif outcome == "before":
        now[0] += 1.0
    result = receives.poll({"async-rid"} if outcome == "abort" else set())
    assert result.finished_recving == {"async-rid"}
    assert result.failed_recving == (
        set() if outcome in ("success", "step") else {"async-rid"}
    )
    assert not receives.poll().finished_recving
    meta = receives.metadata(0)
    if outcome == "late":
        assert meta.quarantined == {read.identity: (17,)}
        assert owner.refs == 1
        second = channel.acquire_exact([key])
        retry = receives.add(CopyJob(0, second, ((0, 1, 18),), "async-rid"), 1.0)
        assert retry.begin()
        retry.complete(True, drained=True)
        assert receives.poll().finished_recving == {"async-rid"}
        assert read.identity in receives.reads
        late_dma.set_result(b"late bytes")
        assert not receives.poll().finished_recving
        assert receives.metadata(0).drained == {read.identity}
    else:
        assert not meta.quarantined
    assert not read.begin()
    wait_until(lambda: owner.refs == 0)


@pytest.mark.parametrize("reason", ["deadline_before_post", "deadline_after_post"])
def test_receive_failure_reason_is_counted_once_and_info_is_rate_limited(
    owner_resources, monkeypatch, caplog, reason
):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1 import async_load
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import CopyJob
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.stats import PeerL1Stats

    owner, _, cleanup = owner_resources
    channel = NativeOwnerChannel(owner.endpoint, owner.epoch, 76, cleanup, lease_s=1.0)
    key = peer_key(b"failure-reason")
    owner.publish([key], [1])
    stats = PeerL1Stats()
    receives = async_load.AsyncReceives(channel.release, stats)
    now = [time.monotonic()]
    monkeypatch.setattr(async_load, "monotonic", lambda: now[0])
    caplog.set_level("INFO", logger=async_load.__name__)
    for index in range(2):
        grant = channel.acquire_exact([key])
        now[0] = grant.acquired_before
        rid = f"failure-{index}"
        read = receives.add(CopyJob(0, grant, ((0, 1, 17),), rid), 0.01)
        if reason == "deadline_before_post":
            now[0] += 1.0
        elif reason == "deadline_after_post":
            assert read.begin()
            now[0] += 1.0
            read.complete(True, drained=True)
        elif reason in ("queue_full", "cancelled"):
            read.cancel_unsubmitted(reason)
        else:
            read.complete(False, drained=True, reason=reason)
        result = receives.poll()
        assert result.failed_recving == result.finished_recving == {rid}
        assert not receives.poll().finished_recving
        wait_until(lambda: owner.refs == 0)
    assert stats.reduce()[f"worker_rdma_failure_{reason}"] > 0
    assert stats.reduce().get("worker_rdma_successes", 0) == 0
    assert [r.getMessage() for r in caplog.records] == [
        (
            f"Peer L1 RDMA receive failed owner={owner.endpoint} reason={reason}; "
            "recomputing request"
        )
    ]


def test_leased_deferred_grant_does_not_renew_on_retry(owner_resources, clock):
    owner, _, cleanup = owner_resources
    key = peer_key(b"leased-queue")
    owner.publish([key], [1])
    owner.absorb(free_blocks=0)
    channel = NativeOwnerChannel(owner.endpoint, owner.epoch, 82, cleanup, lease_s=1.0)
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
        OwnerDeferred,
    )

    with pytest.raises(OwnerDeferred) as queued:
        channel.acquire_exact([key], defer=True)
    owner.absorb(free_blocks=100)
    grant = channel.acquire_exact(
        [key],
        defer=True,
        sequence=queued.value.sequence,
        acquired_before=queued.value.acquired_before,
    )
    assert grant.acquired_before == queued.value.acquired_before
    assert grant.lease_s == pytest.approx(1.0) and owner.refs == 1
    clock.advance(1.1)
    wait_until(lambda: owner.refs == 0)
    owner.absorb()
    assert owner.freeze()
    owner.thaw()


@pytest.mark.parametrize("leased", [False, True])
def test_deferred_decision_uses_the_original_lease_clock(
    owner_resources, clock, leased
):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
        OwnerDeferred,
    )

    _, _, cleanup = owner_resources
    owner = NativeOwner(
        f"vllm-peer-l1/clock/{uuid4().hex}",
        73,
        128,
        window=10,
        initial_window=0,
        lease_ms=20,
        clock=clock,
    )
    try:
        key = peer_key(b"pending")
        owner.publish([key], [1])
        channel = NativeOwnerChannel(
            owner.endpoint, owner.epoch, 93, cleanup, lease_s=0.02 if leased else None
        )
        with pytest.raises(OwnerDeferred) as queued:
            channel.acquire_exact([key], defer=True)
        clock.advance(0.04)
        assert owner.pending() and owner.refs == 0
        assert not owner.freeze()
        owner.thaw()
        owner.absorb(free_blocks=100)
        grant = channel.acquire_exact(
            [key],
            defer=True,
            sequence=queued.value.sequence,
            acquired_before=queued.value.acquired_before,
        )
        if leased:
            assert grant is None and owner.refs == 0
        else:
            assert grant.rows == (1,)
            clock.advance(10)
            owner.absorb(free_blocks=100)
            assert owner.refs == 1
            channel.cancel_unsubmitted(grant.sequence)
        owner.absorb()
        assert owner.freeze()
        owner.thaw(clear=True)
    finally:
        owner.close()


def test_scheduler_quarantine_reference_survives_request_free(scheduler_resources):
    from tests.v1.core.test_prefix_caching import make_request
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import (
        PeerWorkerMetadata,
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.outputs import KVConnectorOutput

    peer, manager, _ = scheduler_resources
    request = make_request("quarantine", list(range(33)), 4, sha256)
    blocks = manager.allocate_slots(request, 32, delay_cache_blocks=True)
    rows = tuple(row for group in blocks.get_block_ids() for row in group if row)
    output = KVConnectorOutput(
        kv_connector_worker_meta=PeerWorkerMetadata(0, {"dma": rows})
    )
    peer.update(output)
    peer.update(output)
    manager.free(request)
    assert all(manager.block_pool.blocks[row].ref_cnt == 1 for row in rows)
    reused = manager.block_pool.get_new_blocks(manager.block_pool.get_num_free_blocks())
    assert not set(rows).intersection(block.block_id for block in reused)
    peer.update(
        KVConnectorOutput(
            kv_connector_worker_meta=PeerWorkerMetadata(0, drained={"dma"})
        )
    )
    assert all(manager.block_pool.blocks[row].ref_cnt == 0 for row in rows)


@pytest.fixture
def tcp_members(unused_tcp_port_factory, clock):
    import hashlib
    from dataclasses import replace

    import msgspec

    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import PeerMemory
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.registry import Manifest

    auth = hashlib.sha256(b"deployment secret").digest()
    with ExitStack() as stack:
        cleanup = CleanupQueue(credential=auth)
        stack.callback(cleanup.close)

        def create(rank=0, epoch=19, endpoint=None, window=10, lease_ms=1000):
            for _ in range(100):
                selected = endpoint or f"tcp://127.0.0.1:{unused_tcp_port_factory()}"
                if selected.endswith(":65535"):
                    continue
                try:
                    owner = NativeOwner(
                        selected,
                        epoch,
                        128,
                        window=window,
                        credential=auth,
                        lease_ms=lease_ms,
                        clock=clock,
                    )
                    break
                except RuntimeError as error:
                    if endpoint is not None or "Address already in use" not in str(
                        error
                    ):
                        raise
            else:
                pytest.fail("Could not bind a control/subscription port pair")
            endpoint = selected
            stack.callback(owner.close)
            member = PeerMemory(
                rank,
                endpoint,
                epoch,
                "owner",
                0,
                "geometry",
                128,
                (((0, 1, 1),),),
                {},
                instance_id="source",
                kv_compatibility_id="weights",
                hash_settings="sha256-seed-fixed",
                restart_group="source",
                session_kinds=("rdma",),
                rdma=b"registered memory",
                lease_s=lease_ms / 1000,
            )
            publish([(owner, member)])
            return owner, member

        def publish(entries):
            manifest = Manifest(
                1, "source", tuple(member for _, member in entries), len(entries)
            )
            for owner, member in entries:
                owner.describe(
                    msgspec.json.encode(member), msgspec.json.encode(manifest)
                )

        yield create, publish, cleanup, auth, replace


def test_members_snapshot_hello_authentication_and_tcp_subscription(tcp_members):
    import msgspec

    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.native import description
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.registry import (
        Domain,
        Resolver,
    )

    create, publish, cleanup, auth, replace = tcp_members
    first, a = create()
    second, b = create(rank=1, epoch=20)
    publish([(first, a), (second, b)])
    key = peer_key(b"remote prefix")
    first.publish([key], [1])
    local = replace(a, instance_id="reader")
    resolver = Resolver(
        Domain(instance_id="reader", seeds=(a.endpoint,), credential=auth), local
    )
    try:
        resolver.refresh()
        snapshot = resolver.snapshot()
        assert snapshot.members == (a, b)
        wait_until(lambda: snapshot.replica.query([key])[0, 0])
        assert not snapshot.replica.query([key])[0, 1]
        assert (
            msgspec.json.decode(description(a.endpoint, a.epoch, credential=auth))[
                "epoch"
            ]
            == a.epoch
        )
        with pytest.raises(ConnectionError):
            description(a.endpoint, 0, members=True)
        unauthorized = NativeOwnerChannel(a.endpoint, a.epoch, 999, CleanupQueue())
        try:
            assert unauthorized.acquire_exact([key]) is None
        finally:
            unauthorized.cleanup.close()
        grant = NativeOwnerChannel(
            a.endpoint, a.epoch, 7, cleanup, lease_s=1.0
        ).acquire_exact([key])
        assert grant.rows == (1,) and first.refs == 1
        cleanup.enqueue(
            cleanup.create(a.endpoint, a.epoch, grant.reader, grant.sequence)
        )
        wait_until(lambda: first.refs == 0)
    finally:
        resolver.close()


@pytest.mark.parametrize(
    "difference",
    [
        "kv_compatibility_id",
        "hash_settings",
        "compatibility",
        "schema",
        "hello",
        "partial",
        "sessions",
    ],
)
def test_resolver_rejects_incompatible_or_inconsistent_members(tcp_members, difference):
    import msgspec

    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.registry import (
        Domain,
        Manifest,
        Resolver,
    )

    create, _, _, auth, replace = tcp_members
    owner, member = create()
    local = replace(member, instance_id="reader")
    if difference == "sessions":
        member = replace(member, session_kinds=("ipc",), rdma=b"")
    if difference in ("schema", "hello", "partial", "sessions"):
        owner.describe(
            msgspec.json.encode(member),
            msgspec.json.encode(
                Manifest(
                    2 if difference == "schema" else 1,
                    "source",
                    (replace(member, epoch=77) if difference == "hello" else member,),
                    2 if difference == "partial" else 1,
                )
            ),
        )
    else:
        local = replace(local, **{difference: "incompatible"})
    resolver = Resolver(
        Domain(instance_id="reader", seeds=(member.endpoint,), credential=auth), local
    )
    try:
        resolver.refresh()
        assert resolver.snapshot() is None
        assert owner.refs == 0
    finally:
        resolver.close()


def test_resolver_replaces_epoch_without_releasing_the_new_owner_grant(
    tcp_members, monkeypatch
):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1 import registry

    create, _, cleanup, auth, replace = tcp_members
    old, member = create()
    now = [0.0]
    monkeypatch.setattr(registry, "monotonic", lambda: now[0])
    resolver = registry.Resolver(
        registry.Domain(
            instance_id="reader", seeds=(member.endpoint,), credential=auth
        ),
        replace(member, instance_id="reader"),
    )
    try:
        resolver.refresh()
        assert resolver.snapshot().members == (member,)
        key = peer_key(b"restart-outstanding")
        old.publish([key], [1])
        old_channel = NativeOwnerChannel(
            member.endpoint, member.epoch, 9, cleanup, lease_s=1.0
        )
        outstanding = old_channel.acquire_exact([key])
        old.close()
        new, replacement = create(epoch=member.epoch + 1, endpoint=member.endpoint)
        new.publish([key], [2])
        now[0] = 2.0
        resolver.refresh()
        # The first MEMBERS call consumes the old cached TCP connection's
        # failure. A refresh inside backoff must retain the last snapshot.
        assert resolver.snapshot().members == (member,)
        now[0] = 2.01
        resolver.refresh()
        assert resolver.snapshot().members == (member,)
        now[0] = 2.1
        resolver.refresh()
        assert resolver.snapshot().members == (replacement,)
        current = NativeOwnerChannel(
            replacement.endpoint, replacement.epoch, 9, cleanup, lease_s=1.0
        )
        grant = current.acquire_exact([key])
        old_channel.release(outstanding)
        wait_until(lambda: not cleanup.pending)
        assert new.refs == 1
        current.cancel_unsubmitted(grant.sequence)
        assert new.refs == 0
    finally:
        resolver.close()


@pytest.fixture
def async_scheduler(scheduler_resources, tcp_members):
    from copy import deepcopy

    from tests.v1.core.test_prefix_caching import make_request
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import PeerHandshake
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import (
        init_none_hash,
        make_block_hash_with_group_id,
    )
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.structured_output import StructuredOutputManager

    base, _, _ = scheduler_resources
    create, publish, _, _, replace_member = tcp_members
    with ExitStack() as stack:

        def start(*, flash=False, ipc_tokens=0, min_gain=1, quarantine_fraction=1 / 16):
            config = deepcopy(base.config)
            if flash:
                config = VllmConfig(
                    model_config=config.model_config,
                    cache_config=config.cache_config,
                    parallel_config=config.parallel_config,
                    device_config=config.device_config,
                    scheduler_config=config.scheduler_config,
                    kv_transfer_config=config.kv_transfer_config,
                )
            remote, member = create(window=512, lease_ms=30000)
            prepared, local = create(epoch=911, window=512, lease_ms=30000)
            prepared.close()
            local = replace_member(
                local, instance_id="reader", restart_group="reader", host="reader"
            )
            config.kv_transfer_config.kv_load_failure_policy = "recompute"
            config.kv_transfer_config.kv_connector_extra_config.update(
                instance_id="reader",
                kv_compatibility_id="weights",
                control_host="127.0.0.1",
                control_port=int(local.endpoint.rsplit(":", 1)[1]),
                peer_seeds=[member.endpoint],
                shared_secret="deployment secret",
                min_gain_pages=min_gain,
                rdma_quarantine_fraction=quarantine_fraction,
            )
            config.cache_config.num_gpu_blocks = base.cache.num_blocks
            scheduler = Scheduler(
                config,
                base.cache,
                StructuredOutputManager(config),
                block_size=128 if flash else 256,
                hash_block_size=128 if flash else 4,
                log_stats=True,
            )
            stack.callback(scheduler.shutdown)
            members = [local]
            ipc = None
            if ipc_tokens:
                ipc, ipc_member = create(rank=1, window=512, lease_ms=30000)
                members.append(
                    replace_member(ipc_member, instance_id="reader", host="reader")
                )
            scheduler.connector.set_xfer_handshake_metadata({0: PeerHandshake(members)})
            init_none_hash(sha256)
            request = make_request(
                "real-async",
                list(range(1027)),
                128 if flash else 4,
                sha256,
                num_hash_lookahead_tokens=0 if flash else 2,
            )
            policy = scheduler.connector._scheduler

            def seed(owner, length):
                length, hashes = policy._lookup.find_longest_cache_hit(
                    request.block_hashes, length, lambda h, g: True
                )
                keys = [
                    peer_key(make_block_hash_with_group_id(h, g))
                    for g, group in enumerate(hashes)
                    for h in group
                    if h is not None
                ]
                owner.publish(keys, list(range(1, len(keys) + 1)))
                return keys

            keys = seed(remote, 1024)
            if ipc is not None:
                ipc_keys = seed(ipc, ipc_tokens)
                wait_until(lambda: policy._replica.query(ipc_keys)[:, 0].all())
            wait_until(
                lambda: (
                    policy._resolver.snapshot() is not None
                    and policy._resolver.snapshot().replica.query(keys).all()
                )
            )
            scheduler.add_request(request)
            expected = (
                (ipc_tokens, False)
                if ipc_tokens and 1024 - ipc_tokens < min_gain * policy.alignment
                else (1024, True)
            )
            wait_until(
                lambda: (
                    scheduler.connector.get_num_new_matched_tokens(request, 0)
                    == expected
                )
            )
            scheduled = scheduler.schedule()
            return scheduler, request, scheduled, remote, ipc

        yield start


@pytest.mark.parametrize(
    "scheduler_resources", [{"mtp": True}, {"mtp": True, "flash": True}], indirect=True
)
def test_real_hma_scheduler_resumes_async_pro_and_flash(
    async_scheduler, scheduler_resources
):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.async_load import (
        AsyncReceives,
    )
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
    from vllm.v1.request import RequestStatus

    base, _, _ = scheduler_resources
    flash = base.supported == (4,)
    scheduler, request, scheduled, owner, _ = async_scheduler(flash=flash)
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.connector.get_loaded_kv_cache_group_ids(request) == (
        (4,) if flash else (0, 1, 2, 3, 4)
    )
    job = scheduled.kv_connector_metadata.jobs[0]
    released: list[Grant] = []
    receives = AsyncReceives(released.append)
    read = receives.add(job, 1.0)
    assert read.begin()
    read.complete(True, drained=True)
    results = receives.poll()
    scheduler.update_from_output(
        scheduled,
        ModelRunnerOutput(
            req_ids=[],
            kv_connector_output=KVConnectorOutput(
                finished_recving=results.finished_recving,
                failed_recving=results.failed_recving,
                kv_connector_worker_meta=receives.metadata(0),
            ),
        ),
    )
    resumed = scheduler.schedule()
    assert request.request_id in resumed.num_scheduled_tokens
    assert request.status == RequestStatus.RUNNING
    assert request.replay_start > 0 if flash else request.replay_start == 0
    assert released == [job.grant]


@pytest.mark.parametrize("scheduler_resources", [{"mtp": True}], indirect=True)
@pytest.mark.parametrize("finish", ["recompute", "abort", "finish"])
def test_real_scheduler_quarantines_late_dma_across_request_lifecycles(
    async_scheduler, finish
):
    from concurrent.futures import Future

    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.async_load import (
        AsyncReceives,
    )
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
    from vllm.v1.request import RequestStatus

    scheduler, request, scheduled, _, _ = async_scheduler()
    job = scheduled.kv_connector_metadata.jobs[0]
    targets = {target for _, _, target in job.blocks}
    released: list[Grant] = []
    receives = AsyncReceives(released.append)
    read = receives.add(job, 1.0)
    assert read.begin()
    late: Future[bool] = Future()
    read.complete(False, drained=False, drain_query=late.done)
    if finish != "recompute":
        scheduler.finish_requests(
            [request.request_id],
            RequestStatus.FINISHED_ABORTED
            if finish == "abort"
            else RequestStatus.FINISHED_STOPPED,
        )
    results = receives.poll()
    assert results.failed_recving == results.finished_recving == {request.request_id}
    scheduler.update_from_output(
        scheduled,
        ModelRunnerOutput(
            req_ids=[],
            kv_connector_output=KVConnectorOutput(
                finished_recving=results.finished_recving,
                failed_recving=results.failed_recving,
                kv_connector_worker_meta=receives.metadata(0),
            ),
        ),
    )
    if finish == "recompute":
        scheduler.schedule()
        recompute_rows = {
            row
            for group in scheduler.kv_cache_manager.get_blocks(
                request.request_id
            ).get_block_ids()
            for row in group
        }
        assert not targets.intersection(recompute_rows)
    else:
        assert request.request_id not in scheduler.requests
    assert all(
        scheduler.kv_cache_manager.block_pool.blocks[row].ref_cnt == 1
        for row in targets
    )
    assert not released
    late.set_result(True)
    receives.poll()
    scheduler.connector.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=receives.metadata(0))
    )
    assert all(
        scheduler.kv_cache_manager.block_pool.blocks[row].ref_cnt == 0
        for row in targets
    )
    assert released == [job.grant]


@pytest.mark.parametrize("scheduler_resources", [{"mtp": True}], indirect=True)
@pytest.mark.parametrize(
    ("ipc_tokens", "gain", "asynchronous"),
    [(1024, 1, False), (768, 2, False), (768, 1, True)],
)
def test_source_tiering_requires_a_remote_gain(
    async_scheduler, ipc_tokens, gain, asynchronous
):
    _, _, scheduled, remote, ipc = async_scheduler(ipc_tokens=ipc_tokens, min_gain=gain)
    job = scheduled.kv_connector_metadata.jobs[0]
    assert (job.grant.lease_s is not None) is asynchronous
    assert (remote.refs > 0) is asynchronous
    assert (ipc.refs > 0) is not asynchronous


@pytest.mark.parametrize("scheduler_resources", [{"mtp": True}], indirect=True)
def test_pro_async_read_starts_transfer_budget_after_the_step_fence(
    async_scheduler, tcp_members, monkeypatch
):
    """A five-group MTP load succeeds after a step longer than its read timeout."""
    from concurrent.futures import Future
    from dataclasses import replace

    from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
        MoRIIOOwnedRead,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1 import async_load, rdma
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

    scheduler, request, scheduled, owner, _ = async_scheduler()
    _, publish, cleanup, credential, _ = tcp_members
    job = scheduled.kv_connector_metadata.jobs[0]
    assert {group for group, _, _ in job.blocks} == set(range(5))
    rows = scheduler.kv_cache_config.num_blocks
    segments = tuple(((group * rows, 1, 1),) for group in range(5))
    memory = replace(job.memory, segments=segments)
    publish([(owner, memory)])
    source = bytes(i % 251 for i in range(5 * rows))
    target = bytearray(5 * rows)
    completed: Future[bool] = Future()

    class Status:
        def Succeeded(self):
            return completed.done()

        def Failed(self):
            return False

    class Transport:
        def prepare_owned_read(self, sizes, local, remote, memory):
            return MoRIIOOwnedRead(Status(), (sizes, local, remote))

        def submit_owned_read(self, submission):
            sizes, local, remote = submission.arguments
            for size, dst, src in zip(sizes, local, remote):
                target[dst : dst + size] = source[src : src + size]
            completed.set_result(True)

    channel = NativeOwnerChannel(
        memory.endpoint, memory.epoch, job.grant.reader, cleanup, lease_s=30.0
    )
    receives = async_load.AsyncReceives(channel.release)
    now = [job.grant.acquired_before]
    monkeypatch.setattr(async_load, "monotonic", lambda: now[0])
    monkeypatch.setattr(rdma, "monotonic", lambda: now[0])
    # save() hands this real admission and an incomplete main-stream fence to
    # the receive machinery. CPU time is controlled instead of running a GPU step.
    read = receives.add(replace(job, memory=memory), 1.0)
    ready, queried = threading.Event(), threading.Event()

    def query():
        queried.set()
        return ready.is_set()

    session = rdma.RdmaReadSession(
        Transport(), object(), rows, segments, memory, credential
    )
    thread = threading.Thread(target=session.run, args=(read, query, threading.Event()))
    thread.start()
    assert queried.wait(5)
    now[0] += 2.0
    try:
        assert not receives.poll().finished_recving
    finally:
        ready.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    result = receives.poll()
    assert result.finished_recving == {request.request_id} and not result.failed_recving
    assert read.deadline == pytest.approx(job.grant.acquired_before + 3.0)
    for group, src, dst in job.blocks:
        assert target[group * rows + dst] == source[group * rows + src]
    scheduler.update_from_output(
        scheduled,
        ModelRunnerOutput(
            req_ids=[],
            kv_connector_output=KVConnectorOutput(
                finished_recving=result.finished_recving
            ),
        ),
    )
    assert request.request_id in scheduler.schedule().num_scheduled_tokens
    wait_until(lambda: owner.refs == 0)


@pytest.mark.parametrize("mode", ["late", "partial"])
def test_rdma_session_uses_deadlines_and_never_releases_unknown_dma(
    tcp_members, monkeypatch, mode
):
    from concurrent.futures import Future

    from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_engine
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1 import async_load, rdma
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import CopyJob

    create, _, cleanup, auth, _ = tcp_members
    owner, member = create(lease_ms=30000)
    channel = NativeOwnerChannel(member.endpoint, member.epoch, 81, cleanup, lease_s=30)
    key = peer_key(b"read")
    owner.publish([key], [1])
    grant = channel.acquire_exact([key])
    now = [grant.acquired_before]
    monkeypatch.setattr(async_load, "monotonic", lambda: now[0])
    monkeypatch.setattr(rdma, "monotonic", lambda: now[0])
    receives = async_load.AsyncReceives(channel.release)
    read = receives.add(CopyJob("source", grant, ((0, 1, 17),), "read", member), 1)
    posted = threading.Event()
    future: Future[bool] = Future()

    class Status:
        def Succeeded(self):
            return future.done()

        def Failed(self):
            return False

    class Transport:
        def prepare_owned_read(self, sizes, local, remote, memory):
            return moriio_engine.MoRIIOOwnedRead(Status(), (sizes, local, remote))

        def submit_owned_read(self, submission):
            assert submission.arguments == ([1], [17], [1])
            posted.set()
            if mode == "partial":
                raise RuntimeError("partial native post")

    session = rdma.RdmaReadSession(
        Transport(), object(), 128, (((0, 1, 1),),), member, auth
    )
    thread = threading.Thread(
        target=session.run, args=(read, lambda: True, threading.Event())
    )
    thread.start()
    assert posted.wait(5)
    now[0] += 2
    thread.join(timeout=5)
    assert not thread.is_alive()
    result = receives.poll()
    assert result.failed_recving == result.finished_recving == {"read"}
    assert receives.metadata(0).quarantined == {read.identity: (17,)}
    assert any(isinstance(r, moriio_engine.MoRIIOOwnedRead) for r in read.resources)
    assert owner.refs == 1
    future.set_result(True)
    receives.poll()
    if mode == "late":
        assert receives.metadata(0).drained == {read.identity}
        wait_until(lambda: owner.refs == 0)
    else:
        assert not receives.metadata(0).drained
        assert owner.refs == 1


@pytest.mark.parametrize("scheduler_resources", [{"mtp": True}], indirect=True)
def test_membership_refresh_keeps_grant_sequences_for_unchanged_owners(
    async_scheduler, tcp_members
):
    from tests.v1.core.test_prefix_caching import make_request
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import (
        PeerWorkerMetadata,
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.outputs import KVConnectorOutput

    scheduler, request, scheduled, owner, _ = async_scheduler()
    original = scheduled.kv_connector_metadata.jobs[0]
    create, publish, _, _, _ = tcp_members
    added, member = create(rank=1, epoch=20, window=512, lease_ms=30000)
    publish([(owner, original.memory), (added, member)])
    scheduler.connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=PeerWorkerMetadata(
                0, failed_peers={original.memory.endpoint}
            )
        )
    )
    policy = scheduler.connector._scheduler
    wait_until(lambda: len(policy._resolver.snapshot().members) == 2)
    newer = make_request(
        "second-outstanding",
        request.prompt_token_ids,
        4,
        sha256,
        num_hash_lookahead_tokens=2,
    )
    scheduler.add_request(newer)
    wait_until(
        lambda: scheduler.connector.get_num_new_matched_tokens(newer, 0) == (1024, True)
    )
    following = scheduler.schedule().kv_connector_metadata.jobs[0]
    assert following.grant.epoch == original.grant.epoch
    assert following.grant.sequence > original.grant.sequence
    assert owner.refs == 2 * len(set(original.grant.rows))


@pytest.mark.parametrize("scheduler_resources", [{"mtp": True}], indirect=True)
@pytest.mark.parametrize("ipc_tokens", [0, 768])
def test_failed_rdma_admission_retries_only_ipc_or_recompute(
    async_scheduler, monkeypatch, ipc_tokens
):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1 import scheduler as policy
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.async_load import (
        AsyncReceives,
    )
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
    from vllm.v1.request import RequestStatus

    scheduler, request, scheduled, _, _ = async_scheduler(ipc_tokens=ipc_tokens)
    now = [time.monotonic()]
    monkeypatch.setattr(policy, "monotonic", lambda: now[0])
    first_job = scheduled.kv_connector_metadata.jobs[0]
    released: list[Grant] = []
    receives = AsyncReceives(released.append)
    first = receives.add(first_job, 1.0)
    assert first.begin()
    first.complete(False, drained=False)
    result = receives.poll()
    scheduler.update_from_output(
        scheduled,
        ModelRunnerOutput(
            req_ids=[],
            kv_connector_output=KVConnectorOutput(
                finished_recving=result.finished_recving,
                failed_recving=result.failed_recving,
                kv_connector_worker_meta=receives.metadata(0),
            ),
        ),
    )
    now[0] += 2.0
    retry = scheduler.schedule()
    assert bool(retry.kv_connector_metadata.jobs) is bool(ipc_tokens)
    for job in retry.kv_connector_metadata.jobs:
        assert job.grant.lease_s is None
        assert not {t for _, _, t in first_job.blocks}.intersection(
            t for _, _, t in job.blocks
        )
    assert request.request_id in retry.num_scheduled_tokens
    assert request.status == RequestStatus.RUNNING
    assert first.identity in receives.reads and not released
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    # Admission memory is cleared on finish; late failure feedback cannot add it.
    scheduler.connector.update_connector_output(
        KVConnectorOutput(failed_recving={request.request_id})
    )
    now[0] += 2.0
    assert scheduler.connector.get_num_new_matched_tokens(request, 0) == (1024, True)
    scheduler.connector.request_finished(request, [])


@pytest.mark.parametrize("scheduler_resources", [{"mtp": True}], indirect=True)
def test_rdma_failures_back_off_and_a_new_success_resets_the_owner(
    async_scheduler, monkeypatch
):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1 import scheduler as policy
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import (
        PeerWorkerMetadata,
    )
    from vllm.v1.outputs import KVConnectorOutput

    scheduler, request, scheduled, _, _ = async_scheduler()
    endpoint = scheduled.kv_connector_metadata.jobs[0].grant.endpoint
    now = [0.0]
    monkeypatch.setattr(policy, "monotonic", lambda: now[0])

    def outcome(*, failed=False, success=None):
        scheduler.connector.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=PeerWorkerMetadata(
                    0,
                    failed_peers={endpoint} if failed else set(),
                    successful_peers={} if success is None else {endpoint: success},
                )
            )
        )

    outcome(failed=True)  # Session setup/before-submit failures use the same feedback.
    assert scheduler.connector.get_num_new_matched_tokens(request, 0) == (0, False)
    now[0] = 1.1
    assert scheduler.connector.get_num_new_matched_tokens(request, 0) == (1024, True)
    scheduler.connector.request_finished(request, [])
    outcome(failed=True)
    now[0] = 2.2
    outcome(success=0.0)  # A success acquired before the failure cannot heal it.
    assert scheduler.connector.get_num_new_matched_tokens(request, 0) == (0, False)
    now[0] = 3.2
    assert scheduler.connector.get_num_new_matched_tokens(request, 0) == (1024, True)
    scheduler.connector.request_finished(request, [])
    outcome(success=3.2)
    outcome(failed=True)
    now[0] = 4.3
    assert scheduler.connector.get_num_new_matched_tokens(request, 0) == (1024, True)
    scheduler.connector.request_finished(request, [])


@pytest.mark.parametrize("scheduler_resources", [{"mtp": True}], indirect=True)
def test_quarantine_budget_stops_rdma_but_preserves_ipc_and_recovers_on_drain(
    async_scheduler, monkeypatch, caplog
):
    from concurrent.futures import Future

    from tests.v1.core.test_prefix_caching import make_request
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1 import scheduler as policy
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.async_load import (
        AsyncReceives,
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

    scheduler, request, scheduled, _, _ = async_scheduler(
        ipc_tokens=768, quarantine_fraction=0.0001
    )
    now = [0.0]
    monkeypatch.setattr(policy, "monotonic", lambda: now[0])
    receives = AsyncReceives(lambda grant: None)
    read = receives.add(scheduled.kv_connector_metadata.jobs[0], 1.0)
    assert read.begin()
    late: Future[bool] = Future()
    read.complete(False, drained=False, drain_query=late.done)
    results = receives.poll()
    meta = receives.metadata(0)
    output = scheduler.update_from_output(
        scheduled,
        ModelRunnerOutput(
            req_ids=[],
            kv_connector_output=KVConnectorOutput(
                failed_recving=results.failed_recving,
                finished_recving=results.finished_recving,
                kv_connector_worker_meta=meta,
            ),
        ),
    )[0]
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.stats import PeerL1Stats

    assert (
        PeerL1Stats(output.scheduler_stats.kv_connector_stats).reduce()[
            "reader_rdma_quarantine_stops"
        ]
        == 1
    )
    scheduler.connector.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=meta)
    )
    values = scheduler.connector.get_kv_connector_stats().reduce()
    assert values["reader_rdma_quarantine_disabled"] == 1
    assert values.get("reader_rdma_quarantine_stops", 0) == 0
    assert sum("Peer L1 RDMA disabled:" in r.message for r in caplog.records) == 1
    assert all(
        job.grant.lease_s is None
        for job in scheduler.schedule().kv_connector_metadata.jobs
    )
    now[0] = 2.0
    probe = make_request(
        "budget-probe", request.prompt_token_ids, 4, sha256, num_hash_lookahead_tokens=2
    )
    assert scheduler.connector.get_num_new_matched_tokens(probe, 0) == (768, False)
    scheduler.connector.request_finished(probe, [])
    late.set_result(True)
    receives.poll()
    scheduler.connector.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=receives.metadata(0))
    )
    assert (
        scheduler.connector.get_kv_connector_stats().reduce()[
            "reader_rdma_quarantine_disabled"
        ]
        == 0
    )
    assert scheduler.connector.get_num_new_matched_tokens(probe, 0) == (1024, True)
    scheduler.connector.request_finished(probe, [])


def test_receive_feedback_survives_worker_serialization_and_aggregation():
    from multiprocessing import Pipe

    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import (
        PeerWorkerMetadata,
    )

    first = PeerWorkerMetadata(
        3, failed_peers={"failed"}, successful_peers={"healthy": 4.0}
    )
    second = PeerWorkerMetadata(2, successful_peers={"healthy": 5.0, "other": 1.0})
    receiver, sender = Pipe(duplex=False)
    with receiver, sender:
        sender.send(first.aggregate(second))
        combined = receiver.recv()
    assert combined.completed_step == 2
    assert combined.failed_peers == {"failed"}
    assert combined.successful_peers == {"healthy": 5.0, "other": 1.0}


def test_resolver_fetches_one_snapshot_per_instance_and_only_changed_hellos(
    tcp_members, monkeypatch
):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1 import registry

    create, publish, _, auth, replace = tcp_members
    a, first = create()
    b, second = create(rank=1, epoch=20)
    publish([(a, first), (b, second)])
    calls = []
    real = registry.description

    def observed(endpoint, epoch, **kwargs):
        calls.append((endpoint, kwargs.get("members", False)))
        return real(endpoint, epoch, **kwargs)

    monkeypatch.setattr(registry, "description", observed)
    now = [0.0]
    monkeypatch.setattr(registry, "monotonic", lambda: now[0])
    resolver = registry.Resolver(
        registry.Domain(
            instance_id="reader",
            seeds=(first.endpoint, second.endpoint),
            credential=auth,
        ),
        replace(first, instance_id="reader"),
    )
    try:
        resolver.refresh()
        assert len(resolver.snapshot().members) == 2
        assert sum(members for _, members in calls) == 1
        assert sum(not members for _, members in calls) == 2
        calls.clear()
        now[0] = 1.1
        resolver.refresh()
        assert calls == [(first.endpoint, True)]
        a.close()
        calls.clear()
        now[0] = 2.2
        resolver.refresh()
        now[0] = 2.21
        resolver.refresh()
        assert calls == [(first.endpoint, True)]
        now[0] = 2.3
        resolver.refresh()
        assert calls[-1] == (second.endpoint, True)
        assert sum(not members for _, members in calls) == 0
    finally:
        resolver.close()


@pytest.mark.parametrize(
    "scheduler_resources", [{"mtp": True}, {"flash": True}], indirect=True
)
@pytest.mark.parametrize("explicit", [False, True])
def test_dp_identity_agrees_with_rank_suffixed_engine_ids(
    scheduler_resources, explicit, monkeypatch
):
    from copy import deepcopy

    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1 import worker
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.registry import Domain
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.worker import (
        memory_description,
        validated_members,
    )
    from vllm.v1.engine.utils import _apply_dp_identity_suffix

    base, _, _ = scheduler_resources
    epochs = iter((100, 101))
    monkeypatch.setattr(worker.secrets, "randbits", lambda bits: next(epochs))
    members = []
    for rank in range(2):
        config = deepcopy(base.config)
        config.parallel_config.data_parallel_size = 2
        config.parallel_config.data_parallel_rank = rank
        config.kv_transfer_config.engine_id = "engine_dp7"
        _apply_dp_identity_suffix(config, rank)
        assert config.kv_transfer_config.engine_id == f"engine_dp7_dp{rank}"
        options = (
            {}
            if not explicit
            else dict(
                instance_id="deployment_dp7",
                kv_compatibility_id="weights",
                control_host="127.0.0.1",
                control_port=28500,
            )
        )
        members.append(
            memory_description(config, base.cache, Domain.from_options(options), rank)
        )
    assert members[0].endpoint != members[1].endpoint
    assert members[0].epoch != members[1].epoch
    assert members[0].compatibility == members[1].compatibility
    assert members[0].hash_settings == members[1].hash_settings
    for rank in range(2):
        agreed = validated_members(members, rank=rank, implicit=not explicit)
        assert {m.instance_id for m in agreed} == {
            "deployment_dp7" if explicit else members[0].instance_id
        }
        assert {m.restart_group for m in agreed} == {agreed[0].instance_id}
        assert [m.endpoint for m in agreed] == [m.endpoint for m in members]


@pytest.mark.parametrize(
    ("field", "implicit"),
    [
        ("hash_settings", True),
        ("compatibility", True),
        ("instance_id", False),
        ("restart_group", False),
        ("kv_compatibility_id", True),
    ],
)
def test_dp_identity_agreement_keeps_compatibility_checks(
    scheduler_resources, field, implicit
):
    from copy import deepcopy
    from dataclasses import replace

    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.registry import Domain
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.worker import (
        memory_description,
        validated_members,
    )

    base, _, _ = scheduler_resources
    members = []
    for rank in range(2):
        config = deepcopy(base.config)
        config.parallel_config.data_parallel_rank = rank
        config.kv_transfer_config.engine_id = f"engine_dp{rank}"
        domain = (
            Domain()
            if implicit
            else Domain(
                instance_id="explicit",
                compatibility_id="weights",
                host="127.0.0.1",
                port=28500,
            )
        )
        members.append(memory_description(config, base.cache, domain, rank))
    members[1] = replace(members[1], **{field: "different"})
    for rank in range(2):
        with pytest.raises(ValueError):
            validated_members(members, rank=rank, implicit=implicit)
