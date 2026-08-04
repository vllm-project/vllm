# SPDX-License-Identifier: Apache-2.0
"""
Tests for the P2P protocol and P2PController: enum registration, protocol
definitions, MemoryLayoutDesc wire serialization, and server handlers.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import httpx
import torch

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey, TrimPolicy
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import P2PL2AdapterConfig
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.multiprocess.config import CoordinatorConfig, P2PConfig
from lmcache.v1.multiprocess.modules.p2p_controller import (
    _MAX_MISSES,
    P2PController,
    _P2PState,
    _PeerInstance,
)
from lmcache.v1.multiprocess.mq import msgspec_decode, msgspec_encode
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
    get_response_class,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType


def _make_key(i: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=f"hash{i}".encode(),
        model_name="test_model",
        kv_rank=1,
    )


def _make_layout_desc() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[torch.Size([2, 3]), torch.Size([4])],
        dtypes=[torch.float16, torch.bfloat16],
    )


# ============================================================================
# Protocol definition tests
# ============================================================================


def test_p2p_request_types_registered():
    """The three P2P request types should be members of RequestType."""
    for name in (
        "P2P_LOOKUP_AND_LOCK",
        "P2P_QUERY_LOOKUP_RESULTS",
        "P2P_UNLOCK_OBJECTS",
    ):
        assert hasattr(RequestType, name)
        assert isinstance(getattr(RequestType, name), RequestType)


def test_p2p_lookup_and_lock_protocol():
    """P2P_LOOKUP_AND_LOCK payload is [list[ObjectKey], MemoryLayoutDesc],
    returns int, and is BLOCKING."""
    payload_classes = get_payload_classes(RequestType.P2P_LOOKUP_AND_LOCK)
    assert payload_classes == [list[ObjectKey], MemoryLayoutDesc]
    assert get_response_class(RequestType.P2P_LOOKUP_AND_LOCK) is int
    assert get_handler_type(RequestType.P2P_LOOKUP_AND_LOCK) == HandlerType.BLOCKING


def test_p2p_query_lookup_results_protocol():
    """P2P_QUERY_LOOKUP_RESULTS payload is [int], returns the optional address
    list, and is BLOCKING."""
    assert get_payload_classes(RequestType.P2P_QUERY_LOOKUP_RESULTS) == [int]
    assert (
        get_response_class(RequestType.P2P_QUERY_LOOKUP_RESULTS)
        == list[TransferChannelAddress] | None
    )
    assert (
        get_handler_type(RequestType.P2P_QUERY_LOOKUP_RESULTS) == HandlerType.BLOCKING
    )


def test_p2p_unlock_objects_protocol():
    """P2P_UNLOCK_OBJECTS payload is [list[ObjectKey]], returns None, BLOCKING."""
    assert get_payload_classes(RequestType.P2P_UNLOCK_OBJECTS) == [list[ObjectKey]]
    assert get_response_class(RequestType.P2P_UNLOCK_OBJECTS) is None
    assert get_handler_type(RequestType.P2P_UNLOCK_OBJECTS) == HandlerType.BLOCKING


# ============================================================================
# MemoryLayoutDesc serialization tests
# ============================================================================


def test_memory_layout_desc_mq_roundtrip():
    """The mq encode/decode dispatch must round-trip MemoryLayoutDesc, whose
    torch.Size / torch.dtype fields ride the customized enc_hook / dec_hook."""
    desc = MemoryLayoutDesc(
        shapes=[torch.Size([8, 16, 128]), torch.Size([4])],
        dtypes=[torch.float32, torch.int8],
    )
    decoded = msgspec_decode(
        msgspec_encode(desc, cls=MemoryLayoutDesc), cls=MemoryLayoutDesc
    )
    assert decoded == desc
    assert all(isinstance(s, torch.Size) for s in decoded.shapes)
    assert all(isinstance(d, torch.dtype) for d in decoded.dtypes)


def test_memory_layout_desc_empty_mq_roundtrip():
    """An empty layout descriptor must round-trip through the mq dispatch."""
    desc = MemoryLayoutDesc(shapes=[], dtypes=[])
    decoded = msgspec_decode(
        msgspec_encode(desc, cls=MemoryLayoutDesc), cls=MemoryLayoutDesc
    )
    assert decoded == desc


# ============================================================================
# TransferChannelAddress tests
# ============================================================================


def test_transfer_channel_address_validity():
    """A non-negative offset is valid; a negative one is not."""
    assert TransferChannelAddress(offset=128, size=64).is_valid()
    assert not TransferChannelAddress(offset=-1, size=0).is_valid()


# ============================================================================
# Server handler tests
# ============================================================================


def _make_controller() -> tuple[P2PController, MagicMock]:
    """Build a P2P-disabled controller (no thread / transfer channel)."""
    ctx = MagicMock()
    controller = P2PController(
        ctx,
        P2PConfig(),
        CoordinatorConfig(),
        instance_id="self",
    )
    return controller, ctx


def test_lookup_and_lock_submits_skip_l2_and_returns_task_id():
    """p2p_lookup_and_lock submits a sparse skip_l2 prefetch and returns a
    fresh id."""
    controller, ctx = _make_controller()
    handle = MagicMock(l1_found_indices=(0, 1))
    ctx.storage_manager.submit_prefetch_task.return_value = handle

    keys = [_make_key(0), _make_key(1)]
    layout_desc = _make_layout_desc()
    task_id = controller.p2p_lookup_and_lock(keys, layout_desc)

    assert task_id == 0
    (spec,), kwargs = ctx.storage_manager.submit_prefetch_task.call_args
    assert spec.keys == keys
    assert spec.layout_desc is layout_desc
    assert kwargs["skip_l2"] is True
    assert spec.policy is TrimPolicy.SPARSE
    # A second call gets a distinct id.
    assert controller.p2p_lookup_and_lock(keys, layout_desc) == 1


def test_query_lookup_results_builds_addresses_for_prefix():
    """A completed lookup returns one address per key: real offsets for the
    found prefix, invalid offsets for the rest."""
    controller, ctx = _make_controller()
    handle = MagicMock(l1_found_indices=(0, 1))
    ctx.storage_manager.submit_prefetch_task.return_value = handle

    keys = [_make_key(0), _make_key(1), _make_key(2)]
    task_id = controller.p2p_lookup_and_lock(keys, _make_layout_desc())

    ctx.storage_manager.query_prefetch_status.return_value = Bitmap(3, 2)
    obj0 = MagicMock(shm_offset=100, shm_byte_length=10)
    obj1 = MagicMock(shm_offset=200, shm_byte_length=20)
    ctx.storage_manager.unsafe_read.return_value = ([keys[0], keys[1]], [obj0, obj1])

    addresses = controller.p2p_query_lookup_results(task_id)
    assert addresses == [
        TransferChannelAddress(offset=100, size=10),
        TransferChannelAddress(offset=200, size=20),
        TransferChannelAddress(offset=-1, size=0),
    ]


def test_query_lookup_results_builds_addresses_for_sparse_hits():
    """A lookup with a mid-sequence L1 gap returns real offsets at the found
    indices and an invalid offset at the gap."""
    controller, ctx = _make_controller()
    handle = MagicMock(l1_found_indices=(0, 2))
    ctx.storage_manager.submit_prefetch_task.return_value = handle

    keys = [_make_key(0), _make_key(1), _make_key(2)]
    task_id = controller.p2p_lookup_and_lock(keys, _make_layout_desc())

    found = Bitmap(3)
    found.batched_set([0, 2])
    ctx.storage_manager.query_prefetch_status.return_value = found
    obj0 = MagicMock(shm_offset=100, shm_byte_length=10)
    obj2 = MagicMock(shm_offset=300, shm_byte_length=30)
    ctx.storage_manager.unsafe_read.return_value = ([keys[0], keys[2]], [obj0, obj2])

    addresses = controller.p2p_query_lookup_results(task_id)
    ctx.storage_manager.unsafe_read.assert_called_once_with([keys[0], keys[2]])
    assert addresses == [
        TransferChannelAddress(offset=100, size=10),
        TransferChannelAddress(offset=-1, size=0),
        TransferChannelAddress(offset=300, size=30),
    ]


def test_query_lookup_results_exactly_once():
    """Re-querying a completed task returns None (the job is consumed)."""
    controller, ctx = _make_controller()
    ctx.storage_manager.submit_prefetch_task.return_value = MagicMock(
        l1_found_indices=()
    )
    task_id = controller.p2p_lookup_and_lock([_make_key(0)], _make_layout_desc())

    ctx.storage_manager.query_prefetch_status.return_value = Bitmap(1)

    assert controller.p2p_query_lookup_results(task_id) == [
        TransferChannelAddress(offset=-1, size=0)
    ]
    assert controller.p2p_query_lookup_results(task_id) is None


def test_query_lookup_results_unknown_task():
    """Querying an unknown task id returns None."""
    controller, _ = _make_controller()
    assert controller.p2p_query_lookup_results(999) is None


def test_query_lookup_results_in_progress():
    """A lookup whose prefetch is not done yet returns None without consuming
    the job."""
    controller, ctx = _make_controller()
    ctx.storage_manager.submit_prefetch_task.return_value = MagicMock(
        l1_found_indices=()
    )
    task_id = controller.p2p_lookup_and_lock([_make_key(0)], _make_layout_desc())

    ctx.storage_manager.query_prefetch_status.return_value = None
    assert controller.p2p_query_lookup_results(task_id) is None
    # Job is still alive; status flips to done on the next poll.
    ctx.storage_manager.query_prefetch_status.return_value = Bitmap(1)
    assert controller.p2p_query_lookup_results(task_id) is not None


def test_unlock_objects_calls_finish_read_prefetched():
    """p2p_unlock_objects forwards the keys to finish_read_prefetched."""
    controller, ctx = _make_controller()
    keys = [_make_key(0), _make_key(1)]
    controller.p2p_unlock_objects(keys)
    ctx.storage_manager.finish_read_prefetched.assert_called_once_with(keys)


def test_unlock_objects_empty_is_noop():
    """Unlocking an empty key list does nothing."""
    controller, ctx = _make_controller()
    controller.p2p_unlock_objects([])
    ctx.storage_manager.finish_read_prefetched.assert_not_called()


def test_report_status_counts_active_jobs():
    """report_status reflects the number of in-flight lookup jobs."""
    controller, ctx = _make_controller()
    ctx.storage_manager.submit_prefetch_task.return_value = MagicMock(
        l1_found_indices=()
    )
    assert controller.report_status()["active_p2p_lookup_jobs"] == 0
    controller.p2p_lookup_and_lock([_make_key(0)], _make_layout_desc())
    status = controller.report_status()
    assert status["active_p2p_lookup_jobs"] == 1
    assert status["p2p_enabled"] is False
    assert status["p2p_state"] == _P2PState.UNREGISTERED.value


def test_get_handlers_covers_all_p2p_request_types():
    """get_handlers wires exactly the three P2P request types."""
    controller, _ = _make_controller()
    request_types = {spec.request_type for spec in controller.get_handlers()}
    assert request_types == {
        RequestType.P2P_LOOKUP_AND_LOCK,
        RequestType.P2P_QUERY_LOOKUP_RESULTS,
        RequestType.P2P_UNLOCK_OBJECTS,
    }


# ============================================================================
# Orchestration: adapter reconcile
# ============================================================================


def _peer(
    instance_id: str,
    url: str = "tc-host:9",
    ip: str = "10.0.0.2",
    mq_port: int = 5555,
) -> _PeerInstance:
    return _PeerInstance(
        instance_id=instance_id,
        ip=ip,
        p2p_advertised_url=url,
        mq_port=mq_port,
    )


def test_reconcile_adds_new_peer():
    """A newly discovered peer gets one P2P L2 adapter with the right urls."""
    controller, ctx = _make_controller()
    ctx.storage_manager.add_l2_adapter.return_value = 7

    controller._apply_state(_P2PState.REGISTERED, {"peerA": _peer("peerA")})

    ctx.storage_manager.add_l2_adapter.assert_called_once()
    config = ctx.storage_manager.add_l2_adapter.call_args.args[0]
    assert isinstance(config, P2PL2AdapterConfig)
    assert config.peer_mq_server_url == "tcp://10.0.0.2:5555"
    assert config.peer_transfer_channel_server_url == "tc-host:9"
    assert controller.report_status()["p2p_peers"] == ["peerA"]


def test_reconcile_no_op_when_peer_unchanged():
    """A still-present, unchanged peer is not re-added on the next cycle."""
    controller, ctx = _make_controller()
    ctx.storage_manager.add_l2_adapter.return_value = 7

    controller._apply_state(_P2PState.REGISTERED, {"peerA": _peer("peerA")})
    controller._apply_state(_P2PState.REGISTERED, {"peerA": _peer("peerA")})

    ctx.storage_manager.add_l2_adapter.assert_called_once()
    ctx.storage_manager.delete_l2_adapter.assert_not_called()


def test_reconcile_keeps_peer_within_grace():
    """A peer absent for up to _MAX_MISSES cycles keeps its adapter."""
    controller, ctx = _make_controller()
    ctx.storage_manager.add_l2_adapter.return_value = 7

    controller._apply_state(_P2PState.REGISTERED, {"peerA": _peer("peerA")})
    for _ in range(_MAX_MISSES):
        controller._apply_state(_P2PState.REGISTERED, {})

    ctx.storage_manager.delete_l2_adapter.assert_not_called()
    assert controller.report_status()["p2p_peers"] == ["peerA"]


def test_reconcile_removes_peer_after_grace():
    """A peer absent beyond _MAX_MISSES cycles has its adapter removed."""
    controller, ctx = _make_controller()
    ctx.storage_manager.add_l2_adapter.return_value = 7

    controller._apply_state(_P2PState.REGISTERED, {"peerA": _peer("peerA")})
    for _ in range(_MAX_MISSES + 1):
        controller._apply_state(_P2PState.REGISTERED, {})

    ctx.storage_manager.delete_l2_adapter.assert_called_once_with(7)
    assert controller.report_status()["p2p_peers"] == []


def test_reconcile_readds_on_url_change():
    """A peer whose advertised url changes is removed and re-added."""
    controller, ctx = _make_controller()
    ctx.storage_manager.add_l2_adapter.side_effect = [7, 8]

    controller._apply_state(_P2PState.REGISTERED, {"peerA": _peer("peerA", url="a:1")})
    controller._apply_state(_P2PState.REGISTERED, {"peerA": _peer("peerA", url="b:2")})

    ctx.storage_manager.delete_l2_adapter.assert_called_once_with(7)
    assert ctx.storage_manager.add_l2_adapter.call_count == 2
    latest = ctx.storage_manager.add_l2_adapter.call_args.args[0]
    assert latest.peer_transfer_channel_server_url == "b:2"


def test_close_removes_all_adapters():
    """close drains every tracked peer adapter."""
    controller, ctx = _make_controller()
    ctx.storage_manager.add_l2_adapter.side_effect = [7, 8]
    controller._apply_state(
        _P2PState.REGISTERED,
        {"peerA": _peer("peerA"), "peerB": _peer("peerB", mq_port=5556)},
    )

    controller.close()

    removed = {c.args[0] for c in ctx.storage_manager.delete_l2_adapter.call_args_list}
    assert removed == {7, 8}


# ============================================================================
# Orchestration: poll-cycle state machine
# ============================================================================


def _enable_polling(
    controller: P2PController, advertise_url: str = "me:1"
) -> MagicMock:
    """Point a controller's poll loop at a mock coordinator client."""
    controller._p2p_config = P2PConfig(advertise_url=advertise_url)
    controller._instances_url = "http://coordinator/instances"
    client = MagicMock()
    controller._http_client = client
    return client


def _instances_response(instances: list[dict]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"instances": instances}
    return response


def _entry(
    instance_id: str,
    url: str,
    ip: str = "10.0.0.9",
    mq_port: int = 5555,
) -> dict:
    return {
        "instance_id": instance_id,
        "ip": ip,
        "p2p_advertised_url": url,
        "mq_port": mq_port,
    }


def test_poll_cycle_timeout_is_disconnected():
    """A request timeout maps to the Disconnected state."""
    controller, _ = _make_controller()
    client = _enable_polling(controller)
    client.get.side_effect = httpx.TimeoutException("boom")

    controller._poll_cycle()
    assert controller.report_status()["p2p_state"] == _P2PState.DISCONNECTED.value


def test_poll_cycle_http_error_is_unregistered():
    """A non-timeout HTTP error maps to the Unregistered state."""
    controller, _ = _make_controller()
    client = _enable_polling(controller)
    client.get.side_effect = httpx.HTTPError("boom")

    controller._poll_cycle()
    assert controller.report_status()["p2p_state"] == _P2PState.UNREGISTERED.value


def test_poll_cycle_self_absent_is_unregistered():
    """Not finding our own instance in the listing is Unregistered."""
    controller, _ = _make_controller()
    client = _enable_polling(controller)
    client.get.return_value = _instances_response([_entry("other", "tc:1")])

    controller._poll_cycle()
    assert controller.report_status()["p2p_state"] == _P2PState.UNREGISTERED.value


def test_poll_cycle_registered_excludes_self_and_adds_peer():
    """When registered, peers (but not self) get adapters."""
    controller, ctx = _make_controller()
    ctx.storage_manager.add_l2_adapter.return_value = 7
    client = _enable_polling(controller, advertise_url="me:1")
    client.get.return_value = _instances_response(
        [_entry("self", "me:1"), _entry("peerA", "tc:1")]
    )

    controller._poll_cycle()

    status = controller.report_status()
    assert status["p2p_state"] == _P2PState.REGISTERED.value
    assert status["p2p_peers"] == ["peerA"]


def test_poll_cycle_unexpected_error_does_not_crash():
    """An unexpected parsing error is caught and maps to Unregistered."""
    controller, _ = _make_controller()
    client = _enable_polling(controller)
    # A non-dict entry makes parsing raise AttributeError (raw.get on a str).
    client.get.return_value = _instances_response(["not-a-dict"])

    summary = controller._poll_cycle()  # must not raise
    assert summary.success is True
    assert controller.report_status()["p2p_state"] == _P2PState.UNREGISTERED.value


def test_poll_cycle_skips_peer_without_p2p_url():
    """A peer that advertises no p2p url is not adapted."""
    controller, ctx = _make_controller()
    client = _enable_polling(controller, advertise_url="me:1")
    client.get.return_value = _instances_response(
        [_entry("self", "me:1"), _entry("peerA", "")]
    )

    controller._poll_cycle()

    ctx.storage_manager.add_l2_adapter.assert_not_called()
    assert controller.report_status()["p2p_peers"] == []
