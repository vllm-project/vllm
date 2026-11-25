# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native authorization protects block contents without scheduler RPC work."""

import os
import select
import shutil
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, suppress
from pathlib import Path
from uuid import uuid4

import pytest
import regex as re

from tests.v1.kv_connector.unit.peer_l1_utils import (
    _publish_blocks,
    wait_until,
)
from tests.v1.kv_connector.unit.peer_l1_utils import (
    adaptive_peer as adaptive_peer,
)
from tests.v1.kv_connector.unit.peer_l1_utils import (
    async_scheduler as async_scheduler,
)
from tests.v1.kv_connector.unit.peer_l1_utils import (
    clock as clock,
)
from tests.v1.kv_connector.unit.peer_l1_utils import (
    mori_sdk as mori_sdk,
)
from tests.v1.kv_connector.unit.peer_l1_utils import (
    scheduler_resources as scheduler_resources,
)
from tests.v1.kv_connector.unit.peer_l1_utils import (
    tcp_members as tcp_members,
)
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1 import native
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
    release_grant,
)

pytestmark = pytest.mark.cpu_test


def test_native_build_loads_changed_sources_with_older_mtimes(tmp_path, monkeypatch):
    """A fresh process must load current source bytes, even with older mtimes."""
    module = tmp_path / "native.py"
    shutil.copyfile(native.__file__, module)
    sources = tmp_path / "native"
    sources.mkdir()
    exports = set(
        re.findall(
            r"\b(l1_\w+)\(",
            (Path(native.__file__).with_suffix("") / "owner.cpp").read_text(),
        )
    )
    stubs = "\n".join(
        f'extern "C" long long {name}() {{ return 0; }}' for name in sorted(exports)
    )
    source = sources / "owner.cpp"
    header = sources / "version.h"
    source.write_text(
        '#include "version.h"\n'
        'extern "C" int peer_l1_test_value() { return VALUE; }\n' + stubs
    )
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm_cache"))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "torch_cache"))
    for value in (1, 2, 3):
        header.write_text(f"#define VALUE {min(value, 2)}\n")
        if value == 3:
            source.write_text(source.read_text().replace("return VALUE;", "return 3;"))
        if value > 1:
            for path in (source, header):
                os.utime(path, (1, 1))
        output = subprocess.check_output(
            [
                sys.executable,
                "-c",
                (
                    "import runpy, sys; "
                    "lib = runpy.run_path(sys.argv[1])['library'](); "
                    "print(lib.peer_l1_test_value())"
                ),
                str(module),
            ],
            text=True,
        )
        assert int(output.strip().splitlines()[-1]) == value


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


@pytest.mark.parametrize("action", ["acquire", "release", "cancel-queued"])
@pytest.mark.parametrize("before_absorb", [False, True])
def test_absorb_preserves_acquires_and_releases(owner_resources, action, before_absorb):
    """Absorb must account for RPCs on either side of its atomic handoff."""
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
        OwnerDeferred,
    )

    owner, channel, _ = owner_resources
    key = peer_key(b"handoff")
    owner.publish([key], [1])
    if action == "cancel-queued":
        owner.absorb(free_blocks=0)
        with pytest.raises(OwnerDeferred) as queued:
            channel.acquire_exact([key], defer=True)
        sequence = queued.value.sequence
    elif action == "release":
        sequence = channel.acquire_exact([key]).sequence

    def rpc():
        if action == "acquire":
            return channel.acquire_exact([key])
        channel.cancel_unsubmitted(sequence)

    if before_absorb:
        grant = rpc()
    first = owner.absorb(free_blocks=100)
    if not before_absorb:
        grant = rpc()
    second = owner.absorb()
    touched, freed, withheld = [a + b for a, b in zip(first, second)]
    assert not withheld
    if action == "acquire":
        assert touched == [1] and not freed
        channel.cancel_unsubmitted(grant.sequence)
        assert owner.absorb() == ([], [1], [])
    else:
        assert touched == freed
    assert not owner.refs


def test_replica_snapshot_deltas_and_authoritative_full_key(owner_resources):
    owner, channel, _ = owner_resources
    key = peer_key(b"replica")
    owner.publish([key], [1])
    replica = PlacementReplica([(owner.endpoint, owner.epoch, 128)])
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
    assert meta is not None
    assert meta.completed_step == 2
    output = KVConnectorOutput(kv_connector_worker_meta=meta)
    assert not output.is_empty()
    assert watermark.report() is None


@pytest.mark.parametrize("pd_tokens,busy", [(0, False), (16, False), (16, True)])
@pytest.mark.parametrize("scheduler_resources", [{"moe": True}], indirect=True)
def test_multi_connector_peer_grant_selection(scheduler_resources, pd_tokens, busy):
    """An unselected peer cancels its grant; any waiting child still blocks."""
    from copy import deepcopy
    from dataclasses import replace

    from tests.v1.core.test_prefix_caching import make_request
    from vllm.config import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
        MultiKVConnectorHandshakeMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.connector import (
        PeerL1Connector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import (
        PeerHandshake,
        PeerMemory,
        PeerMetadata,
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    base, _, _ = scheduler_resources
    config = deepcopy(base.config)
    config.scheduler_config.disable_hybrid_kv_cache_manager = True
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "MockKVConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": "tests.v1.kv_connector.unit.utils",
                    "kv_connector_extra_config": {
                        "matched_tokens": pd_tokens,
                        "is_async": False,
                    },
                },
                {"kv_connector": "PeerL1Connector", "kv_role": "kv_both"},
            ]
        },
    )
    cache = replace(
        base.cache, kv_cache_groups=base.cache.kv_cache_groups[:1], hash_block_size=16
    )
    manager = KVCacheManager(
        cache,
        max_model_len=256,
        enable_caching=True,
        scheduler_block_size=16,
        hash_block_size=16,
    )
    connector = MultiConnector(config, KVConnectorRole.SCHEDULER, cache)
    connector.bind_kv_cache_manager(manager)
    credential = base.domain.credential
    with ExitStack() as stack:
        owner = NativeOwner(
            f"composed/{uuid4().hex}",
            7,
            cache.num_blocks,
            window=10,
            credential=credential,
        )
        stack.callback(owner.close)
        local = PeerMemory(
            0,
            f"composed/{uuid4().hex}",
            88,
            "test",
            0,
            "same",
            cache.num_blocks,
            ((),),
            {},
        )
        remote = replace(
            local, rank=1, endpoint=owner.endpoint, epoch=owner.epoch, device=1
        )
        connector.set_xfer_handshake_metadata(
            {
                0: MultiKVConnectorHandshakeMetadata(
                    (None, PeerHandshake([local, remote], credential=credential))
                )
            }
        )
        peer = connector.sub_connectors[1]
        assert isinstance(peer, PeerL1Connector)
        assert peer._scheduler is not None
        assert peer._scheduler._replica is not None
        request = make_request("composed", list(range(67)), 16, sha256)
        keys = [
            peer_key(make_block_hash_with_group_id(h, 0)) for h in request.block_hashes
        ]
        owner.publish(keys, list(range(1, len(keys) + 1)))
        wait_until(lambda: peer._scheduler._replica.query(keys)[:, 0].all())
        held = None
        try:
            if busy:
                other = [peer_key(f"other-{i}".encode()) for i in range(10)]
                owner.publish(other, list(range(20, 30)))
                held = peer._scheduler._channels[0].acquire_exact(other)
            assert connector.get_num_new_matched_tokens(request, 0) == (
                (None, False) if busy else (pd_tokens or 64, False)
            )
            if busy:
                return
            assert owner.refs > 0
            blocks = manager.allocate_slots(request, 67, delay_cache_blocks=True)
            assert blocks is not None
            connector.update_state_after_alloc(request, blocks, pd_tokens or 64)
            metadata = peer.build_connector_meta(None)
            assert isinstance(metadata, PeerMetadata)
            assert bool(metadata.jobs) == (pd_tokens == 0)
            if pd_tokens:
                assert owner.refs == 0
                assert peer.get_num_new_matched_tokens(request, 0) == (64, False)
                with pytest.raises(RuntimeError, match="allocation differs"):
                    peer.update_state_after_alloc(request, blocks, 16)
        finally:
            if held is not None:
                peer._scheduler._channels[0].cancel_unsubmitted(held.sequence)
            peer._scheduler.finished(request.request_id)
            connector.shutdown()


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
    replica = PlacementReplica([(owner.endpoint, owner.epoch, 128)])
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
            assert grant is not None
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
        peers = [(owners[0].endpoint, 1, 128)] * (peer_count - 1) + [
            (owners[1].endpoint, 2, 128)
        ]
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
        owner = NativeOwner(
            f"vllm-peer-l1/fan-in/{uuid4().hex}",
            99,
            512,
            window=10,
            credential=base.domain.credential,
        )
        stack.callback(owner.close)
        owner_member = PeerMemory(
            99, owner.endpoint, 99, "test", 0, "same", 512, ((),), {}
        )
        readers = []
        for rank in range(7):
            config = deepcopy(base.config)
            reader = policy.PeerScheduler(config, base.cache, (0, 1), window=10)
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
            reader.handshake(
                PeerHandshake([member, owner_member], credential=base.domain.credential)
            )
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
                generations = []
                for reader in readers:
                    assert reader._replica is not None
                    generations.append(reader._replica.generation(0))
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
@pytest.mark.parametrize("prompt_length", [258, 10240])
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


def test_peer_stats_survive_msgpack_and_reach_prometheus():
    from prometheus_client import REGISTRY

    from vllm.config import KVTransferConfig, VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorProm
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.stats import PeerL1Stats
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.metrics.prometheus import unregister_vllm_metrics
    from vllm.v1.metrics.stats import SchedulerStats
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    stats = PeerL1Stats()
    stats.inc("worker_copy_bytes", 3072, transport="ipc")
    output = EngineCoreOutputs(
        scheduler_stats=SchedulerStats(kv_connector_stats=stats.clone_and_reset().data)
    )
    data = (
        MsgpackDecoder(EngineCoreOutputs)
        .decode(MsgpackEncoder().encode(output))
        .scheduler_stats.kv_connector_stats
    )
    assert stats.is_empty()
    config = VllmConfig(
        kv_transfer_config=KVTransferConfig(
            kv_connector="PeerL1Connector", kv_role="kv_both"
        )
    )
    unregister_vllm_metrics()
    try:
        prom = KVConnectorProm(config, ["model_name", "engine"], {0: ["test", "0"]})
        prom.observe(data)
        assert (
            REGISTRY.get_sample_value(
                "vllm:peer_l1_worker_copy_bytes_total",
                {
                    "model_name": "test",
                    "engine": "0",
                    "transport": "ipc",
                },
            )
            == 3072
        )
    finally:
        unregister_vllm_metrics()


@pytest.mark.parametrize(
    "free,maximum,in_use", [(100, 100, False), (100, 100, True), (8, 10, False)]
)
def test_adaptive_window_preserves_multigroup_allocation(
    adaptive_peer, free, maximum, in_use
):
    peer, pool, channel = adaptive_peer(free, maximum)
    effective = min(maximum, free // 9)
    assert peer.admission_reserve() == effective
    # Publish before the test window, then return rows to the queue head.
    blocks, keys = _publish_blocks(
        peer, pool, min(effective, 9) if in_use else effective
    )
    if not in_use:
        pool.free_blocks(blocks, evict_first=True)
    peer.build()
    grant = channel.acquire_exact(keys) if keys else None
    if grant is not None:
        assert not pool.reset_prefix_cache()
    available = pool.get_num_free_blocks()
    first = pool.get_new_blocks(available // 2)
    assert not {b.block_id for b in blocks}.intersection(b.block_id for b in first)
    if not in_use:
        assert peer.admission_reserve() == peer._owner.admission_reserve() == 0
    second = pool.get_new_blocks(available - len(first))
    assert len(first) + len(second) == available
    assert pool.get_num_free_blocks() == 0
    peer.build()
    assert all(block.ref_cnt == (2 if in_use else 0) for block in blocks)
    if in_use and blocks:
        assert peer.admission_reserve() == 1
        assert pool.get_num_free_blocks() == 9
    pool.free_blocks(first + second)
    if in_use:
        pool.free_blocks(blocks)
    if grant is not None:
        channel.cancel_unsubmitted(grant.sequence)
    peer.build()
    assert pool.free_block_queue.num_free_blocks == free
    assert peer._owner.absorb_stats()["owner_effective_window"] == min(
        maximum, free // 9
    )
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
    [(8, 10, 0), (100, 10, 10)],
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
    release_grant(channel.cleanup, grant)
    if immediate is not None:
        release_grant(channel.cleanup, immediate)
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


def test_deferred_enqueue_wakes_and_revalidates_after_snapshot(adaptive_peer):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
        OwnerDeferred,
    )

    source, pool, channel = adaptive_peer(100, 10)
    blocks, keys = _publish_blocks(source, pool, 11)
    wake = threading.Event()
    source.set_engine_wakeup(wake.set)
    assert not source.pending()
    assert source._owner.deferred_rows() == []
    with pytest.raises(OwnerDeferred) as pending:
        channel.acquire_exact(keys, defer=True)
    assert wake.wait(5)
    # Interleave the RPC after the scheduler snapshot, before its native absorb.
    assert source._owner.absorb(
        free_blocks=pool.free_block_queue.num_free_blocks, in_use_rows=[]
    ) == ([], [], [])
    assert source._owner.pending() and source._owner.refs == 0
    source.build()
    grant = channel.acquire_exact(keys, defer=True, sequence=pending.value.sequence)
    assert len(grant.rows) == 11
    assert all(block.ref_cnt == 2 for block in blocks)
    assert not source.pending()
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
    assert grant is not None
    monkeypatch.setattr(async_load, "monotonic", lambda: now[0])
    receives = async_load.AsyncReceives(
        lambda grant: release_grant(channel.cleanup, grant)
    )
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


@pytest.mark.parametrize("timing", ["before", "after", "unleased"])
def test_deferred_decision_uses_the_original_lease_clock(
    owner_resources, clock, timing
):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
        OwnerDeferred,
    )

    leased = timing != "unleased"
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
        if timing != "after":
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
        if timing == "before":
            assert grant is None and owner.refs == 0
        else:
            assert grant is not None
            assert grant.rows == (1,)
            assert grant.acquired_before == queued.value.acquired_before
            clock.advance(10)
            owner.absorb(free_blocks=100)
            if leased:
                wait_until(lambda: owner.refs == 0)
            else:
                assert owner.refs == 1
                channel.cancel_unsubmitted(grant.sequence)
        owner.absorb()
        assert owner.freeze()
        owner.thaw(clear=True)
    finally:
        owner.close()


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
        assert snapshot is not None
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
        assert grant is not None
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
        "partial",
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
        snapshot = resolver.snapshot()
        assert snapshot is not None and snapshot.members == (member,)
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
        snapshot = resolver.snapshot()
        assert snapshot is not None and snapshot.members == (member,)
        now[0] = 2.01
        resolver.refresh()
        snapshot = resolver.snapshot()
        assert snapshot is not None and snapshot.members == (member,)
        now[0] = 2.1
        resolver.refresh()
        snapshot = resolver.snapshot()
        assert snapshot is not None and snapshot.members == (replacement,)
        current = NativeOwnerChannel(
            replacement.endpoint, replacement.epoch, 9, cleanup, lease_s=1.0
        )
        grant = current.acquire_exact([key])
        assert grant is not None
        release_grant(cleanup, outstanding)
        wait_until(lambda: not cleanup.pending)
        assert new.refs == 1
        current.cancel_unsubmitted(grant.sequence)
        assert new.refs == 0
    finally:
        resolver.close()


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
    ("ipc_tokens", "remote_tokens", "asynchronous"),
    [(1024, 1024, False), (256, 1024, False), (256, 1280, True)],
)
def test_source_tiering_requires_a_remote_gain(
    async_scheduler, ipc_tokens, remote_tokens, asynchronous
):
    _, _, scheduled, remote, ipc = async_scheduler(
        ipc_tokens=ipc_tokens, remote_tokens=remote_tokens, prompt_tokens=remote_tokens
    )
    job = scheduled.kv_connector_metadata.jobs[0]
    assert (job.grant.lease_s is not None) is asynchronous
    assert (remote.refs > 0) is asynchronous
    assert (ipc.refs > 0) is not asynchronous


@pytest.mark.parametrize("scheduler_resources", [{"mtp": True}], indirect=True)
def test_pro_async_read_starts_transfer_budget_after_the_step_fence(
    async_scheduler, tcp_members, monkeypatch, mori_sdk
):
    """A five-group MTP load succeeds after a step longer than its read timeout."""
    from dataclasses import replace

    import torch

    from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
        MoRIIOWrapper,
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
    engine = mori_sdk.IOEngine()
    wrapper = MoRIIOWrapper(engine)
    wrapper.register_local_tensor(torch.frombuffer(target, dtype=torch.uint8))
    remote = mori_sdk.MemoryDesc(torch.tensor(list(source), dtype=torch.uint8))

    channel = NativeOwnerChannel(
        memory.endpoint, memory.epoch, job.grant.reader, cleanup, lease_s=30.0
    )
    receives = async_load.AsyncReceives(
        lambda grant: release_grant(channel.cleanup, grant)
    )
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

    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.transports.mori import (
        MoRISession,
    )

    session = rdma.RdmaReadSession(
        MoRISession(
            wrapper,
            engine.create_session(wrapper.local_memory_metadata, remote),
            remote,
        ),
        rows,
        segments,
        memory,
        credential,
        max_ranges_per_post=64,
        max_range_bytes=16 << 20,
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
def test_failed_rdma_admission_retries_ipc_without_reusing_quarantined_rows(
    async_scheduler, monkeypatch
):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1 import scheduler as policy
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.async_load import (
        AsyncReceives,
    )
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
    from vllm.v1.request import RequestStatus

    scheduler, request, scheduled, _, _ = async_scheduler(
        ipc_tokens=256, remote_tokens=1280, prompt_tokens=1280
    )
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
    assert retry.kv_connector_metadata.jobs
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
    assert scheduler.connector.get_num_new_matched_tokens(request, 0) == (1280, True)
    scheduler.connector.request_finished(request, [])


@pytest.mark.parametrize("scheduler_resources", [{"mtp": True}], indirect=True)
def test_quarantine_budget_stops_rdma_but_preserves_ipc_and_recovers_on_drain(
    async_scheduler, monkeypatch
):
    from concurrent.futures import Future

    from tests.v1.core.test_prefix_caching import make_request
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1 import scheduler as policy
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.async_load import (
        AsyncReceives,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import (
        PeerWorkerMetadata,
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

    scheduler, request, scheduled, _, _ = async_scheduler(
        ipc_tokens=256, remote_tokens=1280, prompt_tokens=1280, num_blocks=128
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
            "reader_rdma_quarantine_disabled"
        ]
        == 1
    )
    scheduler.connector.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=meta)
    )
    values = scheduler.connector.get_kv_connector_stats().reduce()
    assert values["reader_rdma_quarantine_disabled"] == 1
    assert all(
        job.grant.lease_s is None
        for job in scheduler.schedule().kv_connector_metadata.jobs
    )
    now[0] = 2.0
    probe = make_request(
        "budget-probe", request.prompt_token_ids, 4, sha256, num_hash_lookahead_tokens=2
    )
    assert scheduler.connector.get_num_new_matched_tokens(probe, 0) == (256, False)
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
    assert scheduler.connector.get_num_new_matched_tokens(probe, 0) == (1280, True)
    scheduler.connector.request_finished(probe, [])

    endpoint = read.job.grant.endpoint

    def feedback(*, failed=False, success=None):
        scheduler.connector.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=PeerWorkerMetadata(
                    0,
                    failed_peers={endpoint} if failed else set(),
                    successful_peers={} if success is None else {endpoint: success},
                )
            )
        )

    feedback(success=2.0)
    for failed_at, retry_at, stale in (
        (2.0, 3.1, None),
        (3.1, 5.2, 2.0),
        (5.2, 6.3, None),
    ):
        now[0] = failed_at
        if failed_at == 5.2:
            feedback(success=failed_at)
        feedback(failed=True)
        assert scheduler.connector.get_num_new_matched_tokens(probe, 0) == (256, False)
        scheduler.connector.request_finished(probe, [])
        if stale is not None:
            now[0] = 4.2
            feedback(success=stale)
            assert scheduler.connector.get_num_new_matched_tokens(probe, 0) == (
                256,
                False,
            )
            scheduler.connector.request_finished(probe, [])
        now[0] = retry_at
        assert scheduler.connector.get_num_new_matched_tokens(probe, 0) == (1280, True)
        scheduler.connector.request_finished(probe, [])


@pytest.mark.parametrize("scheduler_resources", [{"mtp": True}], indirect=True)
@pytest.mark.parametrize(
    "explicit,field",
    [
        (False, None),
        (True, None),
        (False, "hash_settings"),
        (False, "compatibility"),
        (True, "instance_id"),
        (True, "restart_group"),
        (False, "kv_compatibility_id"),
    ],
)
def test_dp_identity_agrees_with_rank_suffixed_engine_ids(
    scheduler_resources, explicit, field, monkeypatch
):
    from copy import deepcopy
    from dataclasses import replace

    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.registry import Domain
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.worker import (
        memory_description,
        validated_members,
    )
    from vllm.v1.engine.utils import _apply_dp_identity_suffix

    base, _, _ = scheduler_resources
    members = []
    monkeypatch.setenv("PEER_L1_TEST_SECRET", "identity test secret")
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
                shared_secret_env="PEER_L1_TEST_SECRET",
                kv_compatibility_id="weights",
                control_host="127.0.0.1",
                control_port=28500,
            )
        )
        members.append(
            memory_description(config, base.cache, Domain.from_options(options), rank)
        )
    if field is not None:
        members[1] = replace(members[1], **{field: "different"})
        with pytest.raises(ValueError):
            validated_members(members, rank=0, implicit=not explicit)
        return
    assert members[0].endpoint != members[1].endpoint
    assert members[0].compatibility == members[1].compatibility
    assert members[0].hash_settings == members[1].hash_settings
    for rank in range(2):
        agreed = validated_members(members, rank=rank, implicit=not explicit)
        assert {m.instance_id for m in agreed} == {
            "deployment_dp7" if explicit else members[0].instance_id
        }
        assert {m.restart_group for m in agreed} == {agreed[0].instance_id}
        assert [m.endpoint for m in agreed] == [m.endpoint for m in members]


# Native control-plane hardening regressions.
@pytest.mark.parametrize(
    "operation", ["owner.freeze()", "cleanup.pending", "replica.query([])"]
)
def test_native_closed_handles_raise_python_errors(operation):
    """Run in a child so an ABI regression reports a failure, not a pytest crash."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from uuid import uuid4\n"
                "from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.native "
                "import NativeOwner, CleanupQueue, PlacementReplica\n"
                "owner = NativeOwner('closed/' + uuid4().hex, 1, 16)\n"
                "cleanup = CleanupQueue()\n"
                "replica = PlacementReplica([])\n"
                "owner.close(); cleanup.close(); replica.close()\n"
                "try:\n"
                f"    {operation}\n"
                "except RuntimeError as exc:\n"
                "    assert 'handle is closed' in str(exc), str(exc)\n"
                "else:\n"
                "    raise AssertionError('closed handle accepted')\n"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("failure", ["maintenance", "bad-alloc", "unknown"])
def test_native_exception_containment(native_allocation_probe, failure):
    import ctypes

    lib = native_allocation_probe
    if failure != "maintenance":
        lib.fail_connection(1 if failure == "bad-alloc" else 2)
        try:
            assert (
                lib.l1_owner_rpc(
                    b"tcp://127.0.0.1:1", 3, 37, 1, 1, None, None, 0, 50, None
                )
                == -1
            )
            error = lib.l1_owner_error().decode()
            assert (
                "bad_alloc" in error
                if failure == "bad-alloc"
                else error == "unknown peer L1 exception"
            )
        finally:
            lib.fail_connection(0)
        return
    published = (ctypes.c_uint8 * 17)()
    handle = lib.l1_owner_start(
        f"maintenance/{uuid4().hex}".encode(),
        1,
        16,
        8,
        4,
        1000,
        16,
        4,
        ctypes.addressof(published),
        None,
        ctypes.cast(lib.failing_clock, ctypes.c_void_p),
        (1 << 64) - 1,
    )
    assert handle > 0
    try:

        def failed():
            result = lib.l1_owner_pending(handle)
            if result < 0:
                assert lib.l1_owner_error() == b"maintenance clock failed"
                return True
            return False

        wait_until(failed)
    finally:
        lib.l1_owner_stop(handle)


@pytest.fixture
def native_control_owner():
    with ExitStack() as stack:
        auth = peer_key(b"native control plane credential")
        owner = NativeOwner(
            f"control/{uuid4().hex}",
            37,
            128,
            pin_cap=64,
            window=10,
            credential=auth,
        )
        stack.callback(owner.close)
        cleanup = CleanupQueue(credential=auth)
        stack.callback(cleanup.close)
        channel = NativeOwnerChannel(owner.endpoint, owner.epoch, 91, cleanup)
        owner.describe(b"hello state", b"member state")
        key = peer_key(b"native-control-row")
        owner.publish([key], [1])
        grant = channel.acquire_exact([key])
        assert grant is not None
        owner.absorb()
        yield owner, channel, cleanup, key, grant


def control_header(op, epoch, reader=0, sequence=0, n=0, credential=b"", magic=None):
    import struct

    return struct.pack(
        "<6Q32s",
        0x4C314F574E455232 if magic is None else magic,
        op,
        epoch,
        reader,
        sequence,
        n,
        credential,
    )


def control_socket(endpoint):
    import socket

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(2)
    sock.connect("\0" + endpoint)
    return sock


def assert_control_closed(sock):
    try:
        reply = sock.recv(1)
    except ConnectionResetError:
        reply = b""
    assert reply == b"", f"Rejected request disclosed reply bytes: {reply!r}"


@pytest.mark.parametrize("op", [3, 7, 8, 10])
def test_native_unauthenticated_operations_disclose_nothing(native_control_owner, op):
    """Neither owner state nor a grant can be read or changed without the secret."""
    owner, channel, _, key, grant = native_control_owner
    n = 16 if op in (8, 13) else 0 if op in (3, 4, 10) else 1
    body = key if op == 7 else b""
    endpoint = owner.endpoint + (".sub" if op == 10 else "")
    with control_socket(endpoint) as sock:
        sock.sendall(
            control_header(op, owner.epoch, grant.reader, grant.sequence, n) + body
        )
        assert_control_closed(sock)
    assert owner.refs == 1
    assert not owner.pending()
    # The authenticated grant survives an attempted forged release/cancel.
    retry = channel.acquire_exact([key], sequence=grant.sequence)
    assert retry is not None and retry.rows == grant.rows


@pytest.mark.parametrize("case", ["wrong_magic", "truncated_header", "truncated_body"])
def test_native_rejects_malformed_or_idle_connections(native_control_owner, case):
    owner, channel, cleanup, key, grant = native_control_owner
    n = 2 if case == "truncated_body" else 1
    header = control_header(
        7,
        owner.epoch,
        grant.reader,
        grant.sequence,
        n,
        cleanup.credential,
        magic=0 if case == "wrong_magic" else None,
    )
    data = header + key
    if case == "truncated_header":
        data = header[:17]
    with control_socket(owner.endpoint) as sock:
        sock.sendall(data)
        assert_control_closed(sock)
    assert owner.refs == 1
    assert channel.acquire_exact([key], sequence=grant.sequence).rows == (1,)


def test_native_accepts_largest_acquire_body_and_cleanup_batch():
    """The DEP8 pin cap can occupy a whole body, including alias rows."""
    import struct

    with ExitStack() as stack:
        owner = NativeOwner(
            f"limits/{uuid4().hex}",
            37,
            128,
            pin_cap=65536,
            window=16384,
            credential=b"a" * 32,
        )
        stack.callback(owner.close)
        cleanup = CleanupQueue(credential=b"a" * 32)
        stack.callback(cleanup.close)
        key = peer_key(b"largest-request")
        owner.publish([key], [1])
        channel = NativeOwnerChannel(
            owner.endpoint, owner.epoch, 91, cleanup, timeout_ms=1000
        )
        grant = channel.acquire_exact([key] * 65536)
        assert grant is not None and grant.rows == (1,) * 65536
        assert owner.refs == 1
        with control_socket(owner.endpoint) as sock:
            sock.sendall(
                control_header(6, owner.epoch, n=64, credential=cleanup.credential)
                + struct.pack("<4Q", grant.reader, grant.sequence, 0, 0) * 64
            )
            reply = bytearray()
            while len(reply) < 16:
                chunk = sock.recv(16 - len(reply))
                assert chunk
                reply.extend(chunk)
            assert struct.unpack("<2Q", reply) == (owner.epoch, 0)
        assert owner.refs == 0


@pytest.mark.parametrize("snapshot", [True, False])
def test_native_replica_rejects_frame_above_peer_bound(snapshot):
    """A peer header cannot expand a snapshot beyond its memory advertisement."""
    import socket
    import struct

    endpoint = f"frame/{uuid4().hex}"
    key = peer_key(b"frame-key")
    short_key = int.from_bytes(key[:8], "little")
    with ExitStack() as stack:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stack.callback(listener.close)
        listener.settimeout(2)
        listener.bind("\0" + endpoint + ".sub")
        listener.listen()
        replica = PlacementReplica([(endpoint, 37, 128)])
        stack.callback(replica.close)
        peer, _ = listener.accept()
        stack.callback(peer.close)
        peer.settimeout(2)
        request = bytearray()
        while len(request) < 80:
            request.extend(peer.recv(80 - len(request)))
        if not snapshot:
            peer.sendall(struct.pack("<7Q", 0x4C31535542534336, 37, 0, 0, 1, 1, 10))
            wait_until(lambda: replica.generation(0) == 1)
        n = 129 if snapshot else 4097
        frame = struct.pack(
            "<7Q",
            0x4C31535542534336,
            37,
            0 if snapshot else n,
            n,
            int(snapshot),
            2,
            10,
        )
        events = b"".join(struct.pack("<3Q", i + 1, short_key, 1) for i in range(n))
        with suppress(BrokenPipeError):
            peer.sendall(frame + events)
        assert_control_closed(peer)
        assert replica.generation(0) == (0 if snapshot else 1)
        assert not replica.query([key]).any()
        # A rejected stream must not prevent a legal boundary snapshot on reconnect.
        peer, _ = listener.accept()
        stack.callback(peer.close)
        peer.settimeout(2)
        request = bytearray()
        while len(request) < 80:
            request.extend(peer.recv(80 - len(request)))
        peer.sendall(
            struct.pack("<7Q", 0x4C31535542534336, 37, 0, 128, 1, 3, 10)
            + struct.pack("<3Q", 0, short_key, 1) * 128
        )
        wait_until(lambda: replica.generation(0) == 3)
        assert replica.query([key]).all()


@pytest.fixture(scope="module")
def native_allocation_probe(tmp_path_factory):
    """Link the real server with an allocator probe, without replacing its classes."""
    import ctypes
    import shlex

    from torch.utils.cpp_extension import get_cxx_compiler

    directory = tmp_path_factory.mktemp("native-allocation")
    probe = directory / "allocation.cpp"
    probe.write_text(
        '#include "rpc.h"\n'
        "#include <atomic>\n#include <cstdlib>\n#include <new>\n#include <stdexcept>\n"
        "static std::atomic<size_t> largest{0};\n"
        "static std::atomic<int> failure{0};\n"
        "void* operator new(size_t n) {\n"
        "  if (n == sizeof(vllm::peer_l1::TcpConnection)) {\n"
        "    if (failure.load() == 1) throw std::bad_alloc();\n"
        "    if (failure.load() == 2) throw 7;\n"
        "  }\n"
        "  size_t old = largest.load();\n"
        "  while (old < n && !largest.compare_exchange_weak(old, n)) {}\n"
        "  if (auto* p = std::malloc(n ? n : 1)) return p;\n"
        "  throw std::bad_alloc();\n}\n"
        "void operator delete(void* p) noexcept { std::free(p); }\n"
        "void operator delete(void* p, size_t) noexcept { std::free(p); }\n"
        'extern "C" int64_t failing_clock() {\n'
        '  throw std::logic_error("maintenance clock failed");\n}\n'
        'extern "C" void reset_allocation() { largest.store(0); }\n'
        'extern "C" size_t largest_allocation() { return largest.load(); }\n'
        'extern "C" void fail_connection(int mode) { failure.store(mode); }\n'
    )
    target = directory / "probe.so"
    source = Path(native.__file__).with_suffix("") / "owner.cpp"
    subprocess.run(
        [
            *shlex.split(get_cxx_compiler()),
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-pthread",
            "-O1",
            "-Wl,-Bsymbolic",
            "-I",
            str(source.parent),
            str(source),
            str(probe),
            "-o",
            str(target),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lib = ctypes.CDLL(str(target))
    for name in (
        "l1_owner_start",
        "l1_owner_stop",
        "l1_owner_pending",
        "l1_owner_rpc",
    ):
        function = getattr(lib, name)
        function.argtypes = getattr(native.library(), name).argtypes
        function.restype = ctypes.c_int64
    lib.largest_allocation.restype = ctypes.c_size_t
    lib.l1_owner_error.restype = ctypes.c_char_p
    return lib


@pytest.mark.parametrize(
    "op,n,authenticated",
    [(7, 64, False), (6, 65, True), (7, 65, True), (8, 65537, True), (3, 1, True)],
    ids=[
        "unauthenticated",
        "cleanup-cap",
        "pin-cap",
        "description-cap",
        "release-body",
    ],
)
def test_native_rejects_unsafe_counts_before_body_allocation(
    native_allocation_probe, op, n, authenticated
):
    import ctypes

    lib = native_allocation_probe
    endpoint = f"allocation/{uuid4().hex}"
    published = (ctypes.c_uint8 * 129)()
    auth = native.credential_array(b"a" * 32)
    handle = lib.l1_owner_start(
        endpoint.encode(),
        37,
        128,
        64,
        10,
        1000,
        16,
        10,
        ctypes.addressof(published),
        auth.ctypes.data,
        None,
        (1 << 64) - 1,
    )
    assert handle > 0
    try:
        lib.reset_allocation()
        with control_socket(endpoint) as sock:
            sock.sendall(
                control_header(
                    op,
                    37,
                    n=n,
                    credential=b"a" * 32 if authenticated else b"",
                )
                + (bytes(32) if op == 3 else b"")
            )
            assert_control_closed(sock)
        # Poll vectors and a client node fit; even 64 Key entries need 2048 bytes.
        assert lib.largest_allocation() < 1024
    finally:
        assert lib.l1_owner_stop(handle) == 0


def native_dp_credential_rank(rank, rendezvous, endpoint_prefix):
    import datetime
    import socket

    import msgspec
    import torch.distributed as dist

    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import PeerMemory
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.worker import (
        exchange_handshake,
    )

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        member = PeerMemory(
            rank,
            f"{endpoint_prefix}/{rank}",
            rank + 37,
            socket.gethostname(),
            rank,
            "geometry",
            128,
            ((),),
            {},
            instance_id=f"engine_dp{rank}",
        )
        handshake = exchange_handshake(
            member,
            b"",
            implicit=True,
            group=dist.group.WORLD,
        )
        assert len(handshake.credential) == 32
        assert "credential=" not in repr(handshake)
        with ExitStack() as stack:
            owner = NativeOwner(
                member.endpoint,
                member.epoch,
                128,
                window=10,
                credential=handshake.credential,
            )
            stack.callback(owner.close)
            cleanup = CleanupQueue(credential=handshake.credential)
            stack.callback(cleanup.close)
            key = peer_key(b"default DP credential")
            owner.publish([key], [1])
            owner.describe(msgspec.json.encode(member), b"members")
            dist.barrier()
            remote = handshake.members[1 - rank]
            with pytest.raises(ConnectionError, match="unauthorized"):
                native.description(remote.endpoint, remote.epoch)
            public = native.description(
                remote.endpoint,
                remote.epoch,
                credential=handshake.credential,
                timeout_ms=1000,
            )
            assert "credential" not in msgspec.json.decode(public)
            channel = NativeOwnerChannel(
                remote.endpoint,
                remote.epoch,
                rank + 91,
                cleanup,
                timeout_ms=1000,
            )
            grant = channel.acquire_exact([key])
            assert grant is not None and grant.rows == (1,)
            dist.barrier()
            channel.cancel_unsubmitted(grant.sequence)
            wait_until(lambda: owner.refs == 0)
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "case",
    ["dp", "fresh", "cross-instance", "scheduler-missing", "scheduler-installed"],
)
def test_native_credential_bootstrap(case, request, tmp_path):
    """Bootstrap authorizes peers without publishing an instance credential."""
    if case == "dp":
        import torch.multiprocessing as mp

        mp.spawn(
            native_dp_credential_rank,
            args=(f"file://{tmp_path}/gloo", f"dp-credential/{uuid4().hex}"),
            nprocs=2,
            join=True,
        )
    elif case == "fresh":
        from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import PeerMemory
        from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.worker import (
            exchange_handshake,
        )

        member = PeerMemory(0, "bootstrap", 37, "host", 0, "geometry", 128, ((),), {})
        first = exchange_handshake(member, b"", implicit=True)
        second = exchange_handshake(member, b"", implicit=True)
        assert len(first.credential) == 32 and first.credential != second.credential
        configured = peer_key(b"configured secret")
        assert (
            exchange_handshake(member, configured, implicit=False).credential
            == configured
        )
    elif case == "cross-instance":
        from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.registry import Domain

        with pytest.raises(ValueError, match="control_host requires shared_secret_env"):
            Domain.from_options(
                {
                    "instance_id": "instance",
                    "kv_compatibility_id": "geometry",
                    "control_host": "127.0.0.1",
                    "control_port": 28500,
                }
            )
    elif case.startswith("scheduler"):
        scheduler_resources = request.getfixturevalue("scheduler_resources")
        with_credential = case == "scheduler-installed"
        from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import (
            PeerHandshake,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.registry import Domain
        from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.scheduler import (
            PeerScheduler,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.worker import (
            exchange_handshake,
            memory_description,
        )
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

        base, _, _ = scheduler_resources
        block_size, hash_size = resolve_kv_cache_block_sizes(base.cache, base.config)
        manager = KVCacheManager(
            base.cache,
            max_model_len=base.config.model_config.max_model_len,
            enable_caching=True,
            scheduler_block_size=block_size,
            hash_block_size=hash_size,
        )
        peer = PeerScheduler(base.config, base.cache, base.supported)
        peer.bind(manager)
        local = memory_description(base.config, base.cache, Domain(), 0)
        try:
            if not with_credential:
                with pytest.raises(
                    ValueError, match="credential from its memory handshake"
                ):
                    peer.handshake(PeerHandshake([local]))
                return
            metadata = exchange_handshake(local, b"", implicit=True)
            peer.handshake(metadata)
            assert peer.domain.credential == metadata.credential
            with pytest.raises(ConnectionError, match="unauthorized"):
                native.description(local.endpoint, local.epoch, members=True)
            public = native.description(
                local.endpoint,
                local.epoch,
                credential=metadata.credential,
            )
            assert public and metadata.credential not in public
            # Completion messages must carry the installed credential as well.
            assert peer._cleanup is not None
            assert (
                native.description(
                    local.endpoint, local.epoch, credential=peer._cleanup.credential
                )
                == public
            )
            channel = NativeOwnerChannel(local.endpoint, local.epoch, 91, peer._cleanup)
            channel.cancel_unsubmitted(1)
            wait_until(lambda: peer._cleanup.pending == 0)
        finally:
            peer.close()
