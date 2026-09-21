# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real pull adapter without registering GPU memory.

The adapter must preserve descriptor selection and peer validation, while its
worker hooks stay usable during a blocked native post. The recording wrapper
replaces only the native boundary; the production mapping and hooks run here.
"""

import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import Future
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlHandshakePayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_receiver import (
    NixlPullReceiver,
    ReadJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
    _PullReceiverBackend,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import NixlKVConnectorStats


class RecordingNixl:
    def __init__(self):
        self.prepared = []
        self.calls = []
        self.post_entered = threading.Event()
        self.allow_post = threading.Event()
        self.allow_post.set()

    def make_prepped_xfer(self, operation, local, local_ids, remote, remote_ids, **kw):
        self.calls.append(("prepare", threading.get_ident()))
        self.prepared.append(
            (operation, local, tuple(local_ids), remote, tuple(remote_ids), kw)
        )
        return len(self.prepared)

    def transfer(self, handle):
        self.calls.append(("post", threading.get_ident()))
        self.post_entered.set()
        assert self.allow_post.wait(10), "test did not unblock native post"
        return "DONE"

    def check_xfer_state(self, handle):
        self.calls.append(("poll", threading.get_ident()))
        return "DONE"

    def get_xfer_telemetry(self, handle):
        raise RuntimeError("deliberately unavailable optional telemetry")

    def release_xfer_handle(self, handle):
        self.calls.append(("release", threading.get_ident()))

    def send_notif(self, name, notif_msg):
        self.calls.append(("notify", threading.get_ident()))


def _worker(rank=0, physical_ratio=1):
    worker = object.__new__(NixlPullConnectorWorker)
    worker.kv_transfer_config = SimpleNamespace(
        get_from_extra_config=lambda name, default: default
    )
    worker.engine_id = "decode"
    worker.tp_rank = rank
    worker.device_id = rank
    worker._background_receiver_enabled = True
    worker.world_size = 2
    worker.dcp_size = worker.pcp_size = 1
    worker._mixed_mem_types = worker._is_csa_linear = False
    worker._transfer_layer_names = ()
    worker._transfer_layer_group_ids = ()
    worker.region_group_ids = [0, 0, 0]
    worker.region_members = []
    worker._uses_region_group_mapping = False
    worker.dst_region_group_ids = {"prefill": [0, 0, 0]}
    worker.dst_region_num_blocks = {}
    worker.dst_uses_region_group_mapping = {"prefill": False}
    worker.block_size = 128
    worker._has_mamba = False
    worker._mamba_ssm_size = []
    worker._physical_blocks_per_logical_kv_block = physical_ratio
    worker._bidirectional_kv_xfer_enabled = False
    worker.use_mla = False
    worker.kv_cache_layout = "HND"
    worker.backend_name = "FLASHINFER"
    worker.block_len_per_layer = [4096, 4096, 4096]
    worker.block_stride_per_layer = [4096, 4096, 4096]
    worker.num_regions = 3
    worker.kv_cache_config = SimpleNamespace(
        transfer_groups=[SimpleNamespace(kv_cache_spec=None) for _ in range(3)]
    )
    info = SimpleNamespace(
        remote_block_size=128,
        remote_tp_size=2,
        remote_dcp_size=1,
        remote_physical_blocks_per_logical=physical_ratio,
    )
    worker.transfer_topo = SimpleNamespace(
        get_engine_info=lambda engine: info,
        tp_ratio=lambda tp: 1,
        block_size_ratio=lambda size: 1,
        handshake_target_ranks=lambda tp, dcp=1: [rank],
    )
    worker.tp_mappings = {
        "prefill": SimpleNamespace(
            all_source_ranks=[rank],
            source_ranks_per_group=[{rank}] * 3,
            local_consumers=1,
        )
    }
    worker.src_xfer_handles_by_block_size = {128: "local-descriptors"}
    worker.dst_xfer_side_handles = {"prefill": {rank: "remote-descriptors"}}
    worker.dst_num_blocks = {
        "decode": 128 * physical_ratio,
        "prefill": 256 * physical_ratio,
    }
    worker._remote_agents = {"prefill": {(0, rank): "producer-native"}}
    worker._engine_last_active = {}
    worker._engine_clock_offset = {}
    worker._recving_transfers = defaultdict(list)
    worker._receiver_keys = {}
    worker._receiver_stats = NixlKVConnectorStats()
    worker._receiver_published_sequence = 0
    worker.xfer_stats = NixlKVConnectorStats()
    worker.nixl_wrapper = RecordingNixl()
    return worker


def _metadata():
    return SimpleNamespace(
        local_block_ids=([1, 2], [5, 6, 7], [8]),
        local_physical_block_ids=(),
        remote=SimpleNamespace(
            engine_id="prefill",
            request_id="producer-request",
            host="host",
            port=1,
            block_ids=([20, 21, 22], [23, 24, 25, 26], [27, 28]),
        ),
        tp_size=2,
        dcp_size=1,
        pp_size=1,
        awaiting_kvs=True,
        receiver_generation=1,
        receiver_is_async=True,
    )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("physical_ratio", [1, 2, 4])
def test_adapter_preserves_legacy_grouped_attention_descriptors(rank, physical_ratio):
    worker = _worker(rank, physical_ratio)
    meta = _metadata()
    legacy = deepcopy(meta)
    legacy.local_physical_block_ids = worker._logical_to_kernel_block_ids(
        legacy.local_block_ids, physical_ratio
    )
    worker._read_blocks_for_req("request", legacy)
    expected = worker.nixl_wrapper.prepared.copy()
    worker.nixl_wrapper.prepared.clear()
    adapter = _PullReceiverBackend(worker)
    transfers = list(adapter.transfers(ReadJob(("request", 1), meta)))
    assert len(transfers) == 1
    assert worker.nixl_wrapper.prepared == expected
    assert meta.remote.block_ids == _metadata().remote.block_ids


def test_peer_rejects_later_rank_geometry_before_any_native_registration():
    worker = _worker()
    worker._remote_agents.clear()
    worker.add_remote_agent = MagicMock()
    worker.transfer_topo.handshake_target_ranks = lambda tp: [0, 1]
    good = SimpleNamespace(
        engine_id="prefill",
        block_size=128,
        physical_blocks_per_logical_kv_block=1,
        kv_cache_layout="HND",
        attn_backend_name="FLASHINFER",
        block_lens=[4096, 4096, 4096],
        kv_caches_base_addr=[100, 200, 300],
        ssm_sizes=[],
        dcp_size=1,
        pcp_size=1,
        block_strides=[4096] * 3,
        region_group_ids=[0, 0, 0],
        region_members=[],
        region_mem_types=["VRAM"] * 3,
    )
    bad = deepcopy(good)
    bad.block_size = 64
    future: Future[tuple[dict[tuple[int, int], SimpleNamespace], float]] = Future()
    future.set_result(({(0, 0): good, (0, 1): bad}, 0.0))
    adapter = _PullReceiverBackend(worker)
    adapter.handshakes["prefill"] = (future, 2)
    with pytest.raises(ValueError, match="geometry"):
        adapter._peer("prefill", "host", 1, 2)
    worker.add_remote_agent.assert_not_called()
    assert not worker._remote_agents


@pytest.mark.parametrize("matching_hash", [True, False])
def test_fetch_only_handshake_never_touches_native_agent(monkeypatch, matching_hash):
    worker = _worker()
    worker.use_host_buffer = False
    worker.device_id = 0
    worker.compat_hash = "same"
    worker.enforce_compat_hash = False
    worker.add_remote_agent = MagicMock()
    worker._add_notif_only_remote_agent = MagicMock()
    platform = MagicMock()
    monkeypatch.setattr(base_worker, "current_platform", platform)
    meta = NixlAgentMetadata(
        engine_id="prefill",
        agent_metadata=b"native-metadata",
        kv_caches_base_addr=[100, 200, 300],
        device_id=0,
        num_blocks=256,
        block_lens=[4096] * 3,
        block_strides=[4096] * 3,
        kv_cache_layout="HND",
        block_size=128,
        ssm_sizes=(0, 0),
        attn_backend_name="FLASHINFER",
        physical_blocks_per_logical_kv_block=1,
    )
    payload = NixlHandshakePayload(
        compatibility_hash="same" if matching_hash else "different",
        agent_metadata_bytes=msgspec.msgpack.encode(meta),
    )
    socket = MagicMock()
    socket.recv_multipart.return_value = [
        msgspec.msgpack.encode(payload),
        msgspec.msgpack.encode(0.0),
    ]
    monkeypatch.setattr(base_worker, "zmq_ctx", lambda *args: nullcontext(socket))
    if matching_hash:
        result, offset = worker._nixl_handshake(
            "host", 1, 2, "prefill", fetch_only=True
        )
        assert result == {(0, 0): meta}
        assert isinstance(offset, float)
    else:
        with pytest.raises(RuntimeError, match="compatibility hash mismatch"):
            worker._nixl_handshake("host", 1, 2, "prefill", fetch_only=True)
    worker.add_remote_agent.assert_not_called()
    worker._add_notif_only_remote_agent.assert_not_called()
    platform.set_device.assert_not_called()
    assert worker.nixl_wrapper.calls == []


def test_empty_group_mismatch_is_rejected_before_native_preparation():
    worker = _worker()
    meta = _metadata()
    meta.local_block_ids = ([], [5, 6, 7], [8])
    adapter = _PullReceiverBackend(worker)
    with pytest.raises(ValueError, match="matching nonempty descriptors"):
        list(adapter.transfers(ReadJob(("request", 1), meta)))
    assert worker.nixl_wrapper.prepared == []


class QuietAdapter(_PullReceiverBackend):
    """Leave native prepare/post/poll/release real, suppress unrelated IO."""

    def initialize(self):
        pass

    def tick(self, active_jobs):
        return ()

    def shutdown(self):
        pass


def test_real_model_hooks_remain_usable_while_native_post_is_blocked():
    worker = _worker()
    wrapper = worker.nixl_wrapper
    wrapper.allow_post.clear()
    receiver = NixlPullReceiver(QuietAdapter(worker))
    worker._receiver = receiver
    receiver.start()
    assert receiver.initialized.wait(10)
    metadata = NixlConnectorMetadata()
    metadata.reqs_to_recv["request"] = _metadata()
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
        NixlPullConnector,
    )

    connector = object.__new__(NixlPullConnector)
    connector.connector_worker = worker
    connector.kv_transfer_config = SimpleNamespace(kv_role="kv_consumer")

    def finish(ids=None):
        result = connector.get_transfer_results(ids or set())
        assert not result.failed_recving
        return result.finished_sending, result.finished_recving

    try:
        worker.start_load_kv(metadata)
        assert wrapper.post_entered.wait(10)
        assert finish({"unrelated"}) == (set(), set())
        assert worker.get_block_ids_with_load_errors() == set()
        assert worker.get_kv_connector_stats() is None
        wrapper.allow_post.set()
        assert receiver.results_available.wait(10)
        assert finish() == (set(), set())
        assert receiver.results_available.wait(10)
        assert finish() == (set(), {"request"})
        assert finish() == (set(), set())
        assert worker._receiver_keys == {}
        # A stale scheduler publication cannot resurrect already retired state.
        worker.start_load_kv(metadata)
        assert finish() == (set(), set())
        assert worker._receiver_keys == {}
        assert len(wrapper.prepared) == 1
        assert {tid for _, tid in wrapper.calls} == {receiver._thread.ident}
    finally:
        wrapper.allow_post.set()
        receiver.shutdown(wait=True, timeout=10)


@pytest.mark.parametrize(
    "diagnostic_failure", ["none", "closed_stderr", "error_string"]
)
def test_native_fatal_handler_exits_without_running_python_finalizers(
    diagnostic_failure,
):
    script = """
import atexit
import os
import threading
from types import SimpleNamespace
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import pull_worker
from tests.v1.kv_connector.unit.test_nixl_receiver_worker import _worker, _metadata
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
atexit.register(lambda: print('UNSAFE_NORMAL_FINALIZATION', flush=True))
pull_worker.current_platform = SimpleNamespace(set_device=lambda device: None)
pull_worker._PullReceiverBackend.tick = lambda self, active: ()
worker = _worker()
worker._receiver = None
diagnostic_failure = DIAGNOSTIC_FAILURE
class NativeFailure(RuntimeError):
    def __str__(self):
        if diagnostic_failure == 'error_string':
            raise RuntimeError('error formatting failed')
        return super().__str__()
def fail(handle):
    raise NativeFailure('injected uncertain DMA')
worker.nixl_wrapper.transfer = fail
if diagnostic_failure == 'closed_stderr':
    os.close(2)
worker._start_receiver()
assert worker._receiver.initialized.wait(10)
metadata = NixlConnectorMetadata()
metadata.reqs_to_recv['request'] = _metadata()
worker.start_load_kv(metadata)
threading.Event().wait(10)
raise SystemExit(2)
""".replace("DIAGNOSTIC_FAILURE", repr(diagnostic_failure))
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 1
    if diagnostic_failure == "none":
        assert "injected uncertain DMA" in result.stderr
    assert "UNSAFE_NORMAL_FINALIZATION" not in result.stdout


def test_adapter_preserves_nonuniform_region_descriptor_offsets():
    """Equal descriptor counts cannot detect an incorrect region base offset."""
    worker = _worker()
    worker.region_group_ids = [0, 1, 2]
    worker.dst_region_group_ids = {"prefill": [0, 1, 2]}
    worker._uses_region_group_mapping = True
    worker.dst_uses_region_group_mapping = {"prefill": True}
    worker.dst_region_num_blocks = {"decode": [32, 64, 128], "prefill": [64, 128, 256]}
    meta = _metadata()
    legacy = deepcopy(meta)
    legacy.local_physical_block_ids = legacy.local_block_ids
    worker._read_blocks_for_req("request", legacy)
    expected = worker.nixl_wrapper.prepared.copy()
    worker.nixl_wrapper.prepared.clear()
    list(_PullReceiverBackend(worker).transfers(ReadJob(("request", 1), meta)))
    assert worker.nixl_wrapper.prepared == expected
    assert expected[0][2] == (1, 2, 37, 38, 39, 104)
    assert expected[0][4] == (21, 22, 88, 89, 90, 220)


def test_notify_burst_keeps_all_obligations_without_receive_completion(monkeypatch):
    """Prefix hits and pre-admission aborts bypass receive credits legitimately."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import pull_worker

    monkeypatch.setattr(pull_worker, "_PullReceiverBackend", QuietAdapter)
    worker = _worker()
    worker._receiver = None
    worker._start_receiver()
    receiver = worker._receiver
    assert receiver is not None and receiver.initialized.wait(10)
    metadata = NixlConnectorMetadata()
    for index in range(300):
        meta = _metadata()
        meta.receiver_is_async = False
        meta.local_block_ids = ()
        meta.remote.request_id = f"producer-{index}"
        metadata.reqs_to_recv[f"local-{index}"] = meta
    try:
        # Publish the whole burst before returning any mailbox credit.
        worker.start_load_kv(metadata)
        while worker._receiver_keys:
            assert receiver.results_available.wait(10)
            assert worker.get_finished() == (set(), set())
        assert sum(call[0] == "notify" for call in worker.nixl_wrapper.calls) == 300
        assert not worker.nixl_wrapper.prepared
    finally:
        receiver.shutdown(wait=True, timeout=10)


def test_known_producer_notifications_do_not_spend_unmatched_capacity():
    worker = _worker()
    worker._receiver_published_sequence = 1
    worker._reqs_to_process = {f"producer-{index}" for index in range(300)}
    worker._reqs_to_send = {}
    worker._engine_ttl = 0
    worker.nixl_wrapper.get_new_notifs = lambda: {
        "consumer": [f"producer-{index}:1".encode() for index in range(300)]
    }
    backend = _PullReceiverBackend(worker)
    backend.applied_sequence = 1
    backend.birth_sequence = {req_id: 1 for req_id in worker._reqs_to_process}
    events = list(backend.tick(()))
    assert len(events) == 1 and len(events[0][0]) == 300
    assert not worker._reqs_to_process
    assert not backend.pending_notifications
