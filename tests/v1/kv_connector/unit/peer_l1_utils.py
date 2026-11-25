# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real owners and schedulers shared by the peer L1 component suites."""

import threading
import time
from contextlib import ExitStack
from uuid import uuid4

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.connector import (
    PeerL1Connector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.native import (
    CleanupQueue,
    NativeOwner,
)
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
    NativeOwnerChannel,
    peer_key,
)


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
    flash = settings.get("flash", False)
    mtp = settings.get("mtp", False)
    large = flash or mtp
    max_length = settings.get("max_model_len", 33024 if large else 256)
    OPTConfig(
        hidden_size=64,
        num_attention_heads=2,
        ffn_dim=128,
        num_hidden_layers=1,
        vocab_size=128,
        max_position_embeddings=max_length,
    ).save_pretrained(tmp_path)
    if mtp or settings.get("moe", False):
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
        config,
        cache,
        (4,) if flash else tuple(range(5)) if large else (0, 1),
        window=settings.get("pin_window", 512 if large else 2),
        pin_cap=settings.get("pin_cap", 4096),
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
        rank = 1
        owner = NativeOwner(
            f"vllm-peer-l1/large/{uuid4().hex}",
            rank,
            cache.num_blocks,
            window=scheduler.window,
            pin_cap=scheduler.pin_cap,
            credential=peer_key(b"scheduler"),
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
    scheduler.handshake(PeerHandshake(members, credential=peer_key(b"scheduler")))
    cleanup = CleanupQueue(credential=peer_key(b"scheduler"))
    channel = NativeOwnerChannel(endpoint, 101, 9, cleanup)
    try:
        yield scheduler, manager, owners if large else channel
    finally:
        cleanup.close()
        scheduler.close()
        for owner in owners:
            owner.close()


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

        def create(free, maximum, **options):
            config = deepcopy(base.config)
            config.kv_transfer_config.kv_connector_extra_config.update(options)
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
            peer = PeerScheduler(config, cache, (0,), window=maximum)
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
            peer.handshake(PeerHandshake([member], credential=base.domain.credential))
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

        def create(
            rank=0,
            epoch=19,
            endpoint=None,
            window=10,
            lease_ms=1000,
        ):
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


@pytest.fixture
def async_scheduler(scheduler_resources, tcp_members, monkeypatch):
    monkeypatch.setenv("PEER_L1_TEST_SECRET", "deployment secret")
    from copy import deepcopy
    from dataclasses import replace

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

        def start(
            *,
            flash=False,
            ipc_tokens=0,
            remote_tokens=1024,
            prompt_tokens=1024,
            num_blocks=None,
        ):
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
            publish([(remote, member)])
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
                shared_secret_env="PEER_L1_TEST_SECRET",
            )
            cache = (
                base.cache
                if num_blocks is None
                else replace(base.cache, num_blocks=num_blocks)
            )
            config.cache_config.num_gpu_blocks = cache.num_blocks
            scheduler = Scheduler(
                config,
                cache,
                StructuredOutputManager(config),
                block_size=128 if flash else 256,
                hash_block_size=128 if flash else 4,
                collect_stats=True,
            )
            stack.callback(scheduler.shutdown)
            members = [local]
            ipc = None
            if ipc_tokens:
                ipc, ipc_member = create(rank=1, window=512, lease_ms=30000)
                members.append(
                    replace_member(ipc_member, instance_id="reader", host="reader")
                )
            connector = scheduler.connector
            assert isinstance(connector, PeerL1Connector)
            connector.set_xfer_handshake_metadata({0: PeerHandshake(members)})
            init_none_hash(sha256)
            request = make_request(
                "real-async",
                list(range(prompt_tokens + 3)),
                128 if flash else 4,
                sha256,
                num_hash_lookahead_tokens=0 if flash else 2,
            )
            policy = connector._scheduler
            assert policy is not None

            def seed(owner, length):
                keys: list[bytes] = []
                lengths = (length, remote_tokens) if owner is remote else (length,)
                for span in dict.fromkeys(lengths):
                    _, hashes = policy._lookup.find_longest_cache_hit(
                        request.block_hashes, span, lambda h, g: True
                    )
                    keys.extend(
                        peer_key(make_block_hash_with_group_id(h, g))
                        for g, group in enumerate(hashes)
                        for h in group
                        if h is not None
                    )
                keys = list(dict.fromkeys(keys))
                owner.publish(keys, list(range(1, len(keys) + 1)))
                return keys

            keys = seed(remote, prompt_tokens)
            if ipc is not None:
                ipc_keys = seed(ipc, ipc_tokens)
                replica = policy._replica
                assert replica is not None
                wait_until(lambda: replica.query(ipc_keys)[:, 0].all())
            resolver = policy._resolver
            assert resolver is not None

            def remote_ready():
                snapshot = resolver.snapshot()
                return snapshot is not None and snapshot.replica.query(keys).all()

            wait_until(remote_ready)
            scheduler.add_request(request)
            expected = (
                (ipc_tokens, False)
                if ipc_tokens
                and remote_tokens - ipc_tokens
                < policy.domain.min_gain_pages * policy.alignment
                else (remote_tokens, True)
            )

            def matched():
                return connector.get_num_new_matched_tokens(request, 0) == expected

            wait_until(matched)
            scheduled = scheduler.schedule()
            return scheduler, request, scheduled, remote, ipc

        yield start


@pytest.fixture
def mori_sdk(monkeypatch):
    """An in-memory external MoRI SDK; vLLM's wrapper and transport stay real."""
    import sys
    from types import ModuleType

    from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_engine

    class Descriptor:
        key = "engine"
        registered = {}

        def __init__(self, tensor=None):
            self.tensor = tensor
            self.packed = str(id(self)).encode()
            self.registered[self.packed] = self

        def pack(self):
            return self.packed

        @classmethod
        def unpack(cls, packed):
            return cls.registered[packed]

    class Status:
        result = "pending"

        def Succeeded(self):
            return self.result == "success"

        def Failed(self):
            return self.result == "failed"

    class Engine:
        def __init__(self, *args):
            self._engine = self
            self.descriptor = Descriptor()
            self.posted = threading.Event()
            self.statuses = []
            self.registered = []

        def create_backend(self, backend, config):
            pass

        def get_engine_desc(self):
            return self.descriptor

        def register_remote_engine(self, descriptor):
            pass

        def register_torch_tensor(self, tensor):
            self.registered.append(tensor)
            return Descriptor(tensor)

        def create_session(self, local, remote):
            return (local, remote)

        def allocate_transfer_uid(self):
            return len(self.statuses)

        def BatchRead(
            self, locals, local_offsets, remotes, remote_offsets, sizes, statuses, uids
        ):
            self.statuses.extend(statuses)
            for local, dsts, remote, srcs, widths in zip(
                locals, local_offsets, remotes, remote_offsets, sizes
            ):
                for dst, src, width in zip(dsts, srcs, widths):
                    local.tensor[dst : dst + width].copy_(
                        remote.tensor[src : src + width].clone()
                    )
            for status in statuses:
                status.result = "success"
            self.posted.set()

    io, cpp, package = ModuleType("mori.io"), ModuleType("mori.cpp"), ModuleType("mori")
    io.__dict__.update(
        BackendType=type("BackendType", (), {"RDMA": "rdma", "XGMI": "xgmi"}),
        PollCqMode=type("PollCqMode", (), {"POLLING": 0}),
        RdmaBackendConfig=lambda *args, **kwargs: None,
        IOEngineConfig=lambda **kwargs: None,
        EngineDesc=Descriptor,
        MemoryDesc=Descriptor,
        IOEngine=Engine,
    )
    cpp.__dict__["TransferStatus"] = Status
    package.__dict__.update(io=io, cpp=cpp)
    for name, module in (("mori", package), ("mori.io", io), ("mori.cpp", cpp)):
        monkeypatch.setitem(sys.modules, name, module)
    # These are SDK imports already bound when moriio_engine was imported.
    for name in (
        "BackendType",
        "PollCqMode",
        "RdmaBackendConfig",
        "EngineDesc",
        "MemoryDesc",
    ):
        monkeypatch.setattr(moriio_engine, name, getattr(io, name), raising=False)
    return io
