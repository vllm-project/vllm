# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import threading
from collections import OrderedDict, defaultdict
from queue import Queue
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

aiter_available = importlib.util.find_spec("aiter") is not None
mori_available = importlib.util.find_spec("mori") is not None

if not (current_platform.is_rocm() and mori_available):
    pytest.skip(
        "MoRIIOs are only available on ROCm with mori package installed",
        allow_module_level=True,
    )

moriio_common = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common"
)
moriio_engine = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine"
)
moriio_layout = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_layout"
)
msgpack = importlib.import_module("msgpack")
ssm_conv_transfer_utils = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils"
)
MambaConvSplitInfo = ssm_conv_transfer_utils.MambaConvSplitInfo

ROLE = moriio_common.ROLE
MoRIIOError = moriio_common.MoRIIOError
MoRIIOTransferAck = moriio_common.MoRIIOTransferAck
RemoteAllocInfo = moriio_common.RemoteAllocInfo
WriteTask = moriio_common.WriteTask
set_role = moriio_common.set_role
MoRIIOWrapper = moriio_engine.MoRIIOWrapper
MoRIIOWriter = moriio_engine.MoRIIOWriter


def _full_spec(
    block_size: int = 4, num_kv_heads: int = 2, head_size: int = 3
) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
    )


def _mla_spec(block_size: int = 4) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=3,
        dtype=torch.bfloat16,
    )


def _worker(
    kv_caches: dict[str, torch.Tensor],
    layer_to_spec: dict[str, object],
    num_blocks: int = 8,
) -> SimpleNamespace:
    return SimpleNamespace(
        kv_caches=kv_caches,
        layer_to_spec=layer_to_spec,
        num_blocks=num_blocks,
        block_size=4,
    )


def _remote_meta(num_blocks: int = 16) -> SimpleNamespace:
    return SimpleNamespace(num_blocks=num_blocks)


def _writer_with_fake_worker(fake_worker: Any) -> Any:
    writer = MoRIIOWriter.__new__(MoRIIOWriter)
    writer._worker_ref = lambda: fake_worker
    writer._write_task_q = Queue()
    writer._write_state_lock = threading.Lock()
    writer._scheduled_writes = defaultdict(int)
    writer._scheduled_layers = defaultdict(set)
    writer._sealed_writes = {}
    writer.ensure_worker_started = lambda: None
    return writer


def _wrapper_for_messages() -> Any:
    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper.done_remote_allocate_req_dict = {}
    wrapper.done_req_ids = []
    wrapper.done_write_cache_req_ids = []
    wrapper._terminal_transfer_ids = OrderedDict()
    return wrapper


def _write_task(layer_name: str, transfer_id: str = "xfer") -> Any:
    return WriteTask(
        request_id="req",
        transfer_id=transfer_id,
        dst_engine_id="remote-engine",
        local_block_ids=[1, 3],
        remote_block_ids_hint=None,
        layer_name=layer_name,
        event=None,
        remote_notify_port=7000,
        remote_ip="127.0.0.1",
    )


@pytest.mark.parametrize(
    ("shape", "spec", "remote_num_blocks", "expected_geometry", "expected_offsets"),
    [
        pytest.param(
            (2, 8, 4, 2, 3),
            _full_spec(),
            16,
            {
                "block_stride": 24,
                "local_kv_stride": 192,
                "remote_kv_stride": 384,
                "split_kv_regions": True,
            },
            ([48, 144, 432, 528], [192, 240, 960, 1008], [48, 48, 48, 48]),
            id="separated",
        ),
        pytest.param(
            (8, 2, 4, 2, 3),
            _full_spec(),
            16,
            {
                "block_stride": 48,
                "local_kv_stride": 24,
                "remote_kv_stride": 24,
                "split_kv_regions": False,
            },
            ([96, 288], [384, 480], [96, 96]),
            id="interleaved",
        ),
        pytest.param(
            (2, 8, 2, 4, 3),
            _full_spec(),
            16,
            {
                "block_size": 4,
                "block_stride": 24,
                "local_kv_stride": 192,
                "remote_kv_stride": 384,
                "split_kv_regions": True,
            },
            ([48, 144, 432, 528], [192, 240, 960, 1008], [48, 48, 48, 48]),
            id="shuffled-separated",
        ),
        pytest.param(
            (8, 2, 2, 4, 3),
            _full_spec(),
            16,
            {
                "block_size": 4,
                "block_stride": 48,
                "local_kv_stride": 24,
                "remote_kv_stride": 24,
                "split_kv_regions": False,
            },
            ([96, 288], [384, 480], [96, 96]),
            id="shuffled-interleaved",
        ),
        pytest.param(
            (2, 16, 2, 2, 3),
            _full_spec(),
            16,
            {
                "num_blocks": 8,
                "block_size": 4,
                "block_stride": 24,
                "local_kv_stride": 192,
                "remote_kv_stride": 384,
                "split_kv_regions": True,
            },
            ([48, 144, 432, 528], [192, 240, 960, 1008], [48, 48, 48, 48]),
            id="separated-kernel-blocks",
        ),
        pytest.param(
            (16, 2, 2, 2, 3),
            _full_spec(),
            16,
            {
                "num_blocks": 8,
                "block_size": 4,
                "block_stride": 48,
                "local_kv_stride": None,
                "remote_kv_stride": None,
                "transfers_per_block": 1,
            },
            ([96, 288], [384, 480], [96, 96]),
            id="interleaved-kernel-blocks",
        ),
        pytest.param(
            (2, 32, 8, 2, 3),
            _full_spec(block_size=16, num_kv_heads=8),
            8,
            {
                "num_blocks": 4,
                "block_size": 16,
                "block_len": 768,
                "block_stride": 384,
                "local_kv_stride": 1536,
                "remote_kv_stride": 3072,
                "split_kv_regions": True,
            },
            (
                [768, 2304, 3840, 5376],
                [3072, 3840, 9216, 9984],
                [768, 768, 768, 768],
            ),
            id="separated-kernel-axis-from-spec",
        ),
        pytest.param(
            (32, 2, 8, 2, 3),
            _full_spec(block_size=16, num_kv_heads=8),
            8,
            {
                "num_blocks": 4,
                "block_size": 16,
                "block_len": 1536,
                "block_stride": 768,
                "local_kv_stride": None,
                "remote_kv_stride": None,
                "transfers_per_block": 1,
            },
            ([1536, 4608], [6144, 7680], [1536, 1536]),
            id="interleaved-kernel-axis-from-spec",
        ),
        pytest.param(
            (8, 4, 3),
            _mla_spec(),
            16,
            {
                "block_stride": 12,
                "local_kv_stride": None,
                "remote_kv_stride": None,
                "transfers_per_block": 1,
            },
            ([24, 72], [96, 120], [24, 24]),
            id="mla-key-only",
        ),
    ],
)
def test_supported_layouts_compute_expected_geometry_and_offsets(
    shape, spec, remote_num_blocks, expected_geometry, expected_offsets
):
    cache = torch.empty(shape, dtype=torch.bfloat16)
    worker = _worker({"layer": cache}, {"layer": spec})

    geometry = moriio_layout.get_layer_transfer_geometry(
        "layer", cache, worker.layer_to_spec, remote_num_blocks=remote_num_blocks
    )
    for field, expected in expected_geometry.items():
        assert getattr(geometry, field) == expected

    assert (
        moriio_layout.compute_block_transfer_offsets(
            "layer",
            cache,
            worker.layer_to_spec,
            [1, 3],
            [4, 5],
            remote_num_blocks,
        )
        == expected_offsets
    )


def test_kernel_block_layout_without_spec_dimensions_rejects_ambiguous_axes():
    cache = torch.empty((2, 32, 8, 2, 3), dtype=torch.bfloat16)
    worker = _worker(
        {"layer": cache},
        {"layer": SimpleNamespace(block_size=16)},
    )

    with pytest.raises(ValueError, match="Ambiguous MoRIIO kernel-block"):
        moriio_layout.get_layer_transfer_geometry(
            "layer", cache, worker.layer_to_spec, remote_num_blocks=8
        )


def test_mixed_layers_compute_distinct_offsets_per_layer():
    kv_caches = {
        "separated": torch.empty((2, 8, 4, 2, 3), dtype=torch.bfloat16),
        "interleaved": torch.empty((8, 2, 4, 2, 3), dtype=torch.bfloat16),
        "indexer": torch.empty((8, 4, 3), dtype=torch.bfloat16),
    }
    worker = _worker(
        kv_caches,
        {
            "separated": _full_spec(),
            "interleaved": _full_spec(),
            "indexer": _mla_spec(),
        },
    )

    separated = moriio_layout.compute_block_transfer_offsets(
        "separated",
        kv_caches["separated"],
        worker.layer_to_spec,
        [1, 3],
        [4, 5],
        _remote_meta().num_blocks,
    )
    interleaved = moriio_layout.compute_block_transfer_offsets(
        "interleaved",
        kv_caches["interleaved"],
        worker.layer_to_spec,
        [1, 3],
        [4, 5],
        _remote_meta().num_blocks,
    )
    indexer = moriio_layout.compute_block_transfer_offsets(
        "indexer",
        kv_caches["indexer"],
        worker.layer_to_spec,
        [1, 3],
        [4, 5],
        _remote_meta().num_blocks,
    )

    assert separated != interleaved
    assert separated != indexer
    assert interleaved != indexer


def test_write_transfer_plan_caches_offsets_per_geometry():
    kv_caches = {
        "dense0": torch.empty((8, 2, 4, 2, 3), dtype=torch.bfloat16),
        "dense1": torch.empty((8, 2, 4, 2, 3), dtype=torch.bfloat16),
        "indexer": torch.empty((8, 4, 3), dtype=torch.bfloat16),
    }
    calls: list[str] = []

    class FakeWorker:
        kv_caches: dict[str, torch.Tensor]
        layer_name_to_local_kv_cache_metadata: dict[str, list[Any]]

        def _is_mamba_layer(self, layer_name):
            return False

        def _region_session_indices(self, layer_name):
            return [list(self.kv_caches).index(layer_name)]

        def _compute_block_transfer_offsets(
            self, layer_name, local_block_ids, remote_block_ids, remote_moriio_meta
        ):
            calls.append(layer_name)
            call_id = len(calls)
            return ([call_id], [call_id + 10], [call_id + 20])

    fake_worker = FakeWorker()
    fake_worker.kv_caches = kv_caches
    fake_worker.layer_name_to_local_kv_cache_metadata = {name: [] for name in kv_caches}
    writer = MoRIIOWriter.__new__(MoRIIOWriter)
    writer._worker_ref = lambda: fake_worker
    request_info = RemoteAllocInfo(block_ids=[4, 5])
    remote_meta = _remote_meta()

    dense0_plan = writer._prepare_transfer_plan(
        SimpleNamespace(
            layer_name="dense0",
            local_block_ids=[1, 3],
            request_id="req",
            transfer_id="xfer",
        ),
        request_info,
        remote_meta,
    )
    dense1_plan = writer._prepare_transfer_plan(
        SimpleNamespace(
            layer_name="dense1",
            local_block_ids=[1, 3],
            request_id="req",
            transfer_id="xfer",
        ),
        request_info,
        remote_meta,
    )
    indexer_plan = writer._prepare_transfer_plan(
        SimpleNamespace(
            layer_name="indexer",
            local_block_ids=[1, 3],
            request_id="req",
            transfer_id="xfer",
        ),
        request_info,
        remote_meta,
    )

    assert calls == ["dense0", "indexer"]
    assert dense0_plan.transfer_local_offsets == [1]
    assert dense1_plan.transfer_local_offsets == [1]
    assert indexer_plan.transfer_local_offsets == [2]
    assert len(request_info.transfer_offsets) == 2


def test_write_scheduler_deduplicates_layers_and_seals_expected_count():
    request_info = RemoteAllocInfo(block_ids=[4, 5])
    wrapper = _wrapper_for_messages()
    wrapper.done_remote_allocate_req_dict["xfer"] = request_info
    writer = _writer_with_fake_worker(SimpleNamespace(moriio_wrapper=wrapper))

    assert writer.schedule_write(_write_task("dense0"))
    assert not writer.schedule_write(_write_task("dense0"))
    assert writer.schedule_write(_write_task("indexer"))

    assert writer._write_task_q.qsize() == 2
    writer.seal_pending_transfers()

    assert request_info.writes_expected == 2
    assert writer._sealed_writes["xfer"] == 2


def test_write_completion_notifies_once_after_all_sealed_writes_finish():
    class FakeWrapper:
        def __init__(self):
            self.done_remote_allocate_req_dict = {}
            self.done_req_ids = []
            self.lock = threading.Lock()
            self.notifications = []
            self.wait_count = 0
            self.waited_statuses = []
            self._terminal_transfer_ids = OrderedDict()

        def waiting_for_transfer_complete(self, transfer_statuses=None):
            self.wait_count += 1
            self.waited_statuses.append(list(transfer_statuses or []))

        def _is_transfer_terminal_locked(self, transfer_id):
            return transfer_id in self._terminal_transfer_ids

        def _mark_transfer_terminal_locked(self, transfer_id):
            self._terminal_transfer_ids[transfer_id] = None

        def send_notify(
            self,
            transfer_id,
            remote_ip,
            remote_port,
            message_type=None,
            message_fields=None,
        ):
            self.notifications.append(
                (transfer_id, remote_ip, remote_port, message_type, message_fields)
            )

    wrapper = FakeWrapper()
    request_info = RemoteAllocInfo(block_ids=[4, 5], writes_expected=2)
    request_info.transfer_statuses.extend(["status-a", "status-b"])
    request_info.completion_request_id = "req"
    request_info.completion_remote_notify_port = 7000
    request_info.completion_remote_ip = "127.0.0.1"
    wrapper.done_remote_allocate_req_dict["xfer"] = request_info
    writer = _writer_with_fake_worker(
        SimpleNamespace(moriio_wrapper=wrapper, tp_rank=2)
    )
    writer._scheduled_writes["xfer"] = 2
    writer._scheduled_layers["xfer"] = {"dense0", "indexer"}
    writer._sealed_writes["xfer"] = 2

    writer._mark_write_done("xfer", request_info)
    assert wrapper.notifications == []
    writer._mark_write_done("xfer", request_info)
    writer._finalize_if_complete("xfer", request_info)

    assert wrapper.notifications == [("xfer", "127.0.0.1", 7002, "write_done", None)]
    assert wrapper.done_req_ids == [MoRIIOTransferAck("xfer")]
    assert wrapper.done_remote_allocate_req_dict == {}
    assert wrapper.wait_count == 1
    assert wrapper.waited_statuses == [["status-a", "status-b"]]
    assert request_info.transfer_statuses == []
    assert wrapper._is_transfer_terminal_locked("xfer")


def test_moriio_wrapper_waits_scoped_statuses_without_global_drain():
    class FakeStatus:
        def __init__(self):
            self.checked = 0

        def Succeeded(self):
            self.checked += 1
            return True

        def Failed(self):
            return False

    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper._transfer_timeout = 1
    global_status = FakeStatus()
    scoped_status = FakeStatus()
    wrapper.transfer_status = [global_status]

    wrapper.waiting_for_transfer_complete([scoped_status])

    assert scoped_status.checked == 1
    assert global_status.checked == 0
    assert wrapper.transfer_status == [global_status]


def test_write_failure_marks_terminal_and_clears_scheduled_state():
    wrapper = _wrapper_for_messages()
    wrapper.done_remote_allocate_req_dict["xfer"] = RemoteAllocInfo(block_ids=[4, 5])
    writer = _writer_with_fake_worker(SimpleNamespace(moriio_wrapper=wrapper))
    writer._scheduled_writes["xfer"] = 2
    writer._scheduled_layers["xfer"] = {"dense0", "indexer"}
    writer._sealed_writes["xfer"] = 2

    writer._mark_request_done("xfer")

    assert wrapper.done_req_ids == [MoRIIOTransferAck("xfer")]
    assert wrapper.done_remote_allocate_req_dict == {}
    assert wrapper._is_transfer_terminal_locked("xfer")
    assert "xfer" not in writer._scheduled_writes
    assert "xfer" not in writer._scheduled_layers
    assert "xfer" not in writer._sealed_writes


def test_schedule_write_rejects_terminal_transfer_without_recreating_state():
    wrapper = _wrapper_for_messages()
    wrapper.done_remote_allocate_req_dict["xfer"] = RemoteAllocInfo(block_ids=[4, 5])
    writer = _writer_with_fake_worker(SimpleNamespace(moriio_wrapper=wrapper))
    writer._scheduled_writes["xfer"] = 1
    writer._scheduled_layers["xfer"] = {"dense0"}
    writer._sealed_writes["xfer"] = 1

    writer._mark_request_done("xfer")

    assert not writer.schedule_write(_write_task("indexer"))
    assert writer._write_task_q.empty()
    assert "xfer" not in writer._scheduled_writes
    assert "xfer" not in writer._scheduled_layers
    assert "xfer" not in writer._sealed_writes


def test_late_remote_blocks_message_is_ignored_after_transfer_done():
    set_role(ROLE.PRODUCER)
    wrapper = _wrapper_for_messages()
    with wrapper.lock:
        wrapper._mark_transfer_terminal_locked("xfer")

    wrapper._handle_message(
        msgpack.dumps(
            {
                "type": "remote_blocks",
                "req_id": "req",
                "transfer_id": "xfer",
                "block_notify_list": [4, 5],
                "decode_rank": 3,
            }
        )
    )

    assert "xfer" not in wrapper.done_remote_allocate_req_dict


@pytest.mark.parametrize(
    ("role", "payload", "expected"),
    [
        pytest.param(
            ROLE.PRODUCER,
            msgpack.dumps(
                {
                    "type": "remote_blocks",
                    "req_id": "req",
                    "transfer_id": "xfer",
                    "block_notify_list": [4, 5],
                    "decode_rank": 3,
                }
            ),
            "remote_blocks",
            id="remote-blocks",
        ),
        pytest.param(
            ROLE.CONSUMER,
            msgpack.dumps({"type": "write_done", "transfer_id": "xfer"}),
            "write_done",
            id="write-done",
        ),
        pytest.param(
            ROLE.PRODUCER,
            msgpack.dumps({"type": "release", "transfer_id": "xfer"}),
            MoRIIOTransferAck("xfer"),
            id="release",
        ),
        pytest.param(
            ROLE.PRODUCER,
            msgpack.dumps(
                {
                    "type": "release",
                    "transfer_id": "xfer",
                    "consumer_tp_size": 8,
                }
            ),
            MoRIIOTransferAck("xfer", 8),
            id="release-consumer-tp-size",
        ),
        pytest.param(None, b"xfer", "plain", id="plain-string"),
    ],
)
def test_moriio_wrapper_routes_valid_messages(role, payload, expected):
    wrapper = _wrapper_for_messages()
    completions: list[str] = []
    if role is not None:
        set_role(role)
    if expected == "plain":
        wrapper._handle_completion_message = completions.append

    wrapper._handle_message(payload)

    if expected == "remote_blocks":
        request_info = wrapper.done_remote_allocate_req_dict["xfer"]
        assert request_info.block_ids == [4, 5]
        assert request_info.decode_dp_rank == 3
    elif expected == "write_done":
        assert wrapper.done_write_cache_req_ids == ["xfer"]
    elif expected == "plain":
        assert completions == ["xfer"]
    else:
        assert wrapper.done_req_ids == [expected]
        assert wrapper._is_transfer_terminal_locked("xfer")


@pytest.mark.parametrize(
    ("role", "payload", "match"),
    [
        pytest.param(
            None,
            msgpack.dumps({"type": "unknown", "transfer_id": "xfer"}),
            "Unhandled structured message type",
            id="unknown-structured-type",
        ),
        pytest.param(
            ROLE.PRODUCER,
            msgpack.dumps(
                {
                    "type": "remote_blocks",
                    "req_id": "req",
                    "transfer_id": "xfer",
                    "block_notify_list": [],
                }
            ),
            "block_notify_list cannot be empty",
            id="empty-remote-blocks",
        ),
        pytest.param(
            None,
            b"",
            "Unhandled message format",
            id="empty-completion",
        ),
    ],
)
def test_moriio_wrapper_rejects_invalid_messages(role, payload, match):
    wrapper = _wrapper_for_messages()
    if role is not None:
        set_role(role)
    wrapper._handle_completion_message = lambda msg: None

    with pytest.raises(MoRIIOError, match=match):
        wrapper._handle_message(payload)


def test_local_block_ids_longer_than_remote_raises_value_error():
    cache = torch.empty((8, 2, 4, 2, 3), dtype=torch.bfloat16)
    worker = _worker({"layer": cache}, {"layer": _full_spec()})

    with pytest.raises(ValueError, match="longer than remote_block_ids"):
        moriio_layout.compute_block_transfer_offsets(
            "layer", cache, worker.layer_to_spec, [1, 3], [4], _remote_meta().num_blocks
        )


def test_empty_local_block_ids_is_free_only_noop():
    cache = torch.empty((8, 2, 4, 2, 3), dtype=torch.bfloat16)
    worker = _worker({"layer": cache}, {"layer": _full_spec()})

    assert moriio_layout.compute_block_transfer_offsets(
        "layer", cache, worker.layer_to_spec, [], [4, 5], _remote_meta().num_blocks
    ) == ([], [], [])


def test_registration_regions_do_not_split_interleaved_or_mla_cache():
    separated = torch.empty((2, 8, 4, 2, 3), dtype=torch.bfloat16)
    interleaved = torch.empty((8, 2, 4, 2, 3), dtype=torch.bfloat16)
    indexer = torch.empty((8, 4, 3), dtype=torch.bfloat16)
    worker = _worker(
        {
            "separated": separated,
            "interleaved": interleaved,
            "indexer": indexer,
        },
        {
            "separated": _full_spec(),
            "interleaved": _full_spec(),
            "indexer": _mla_spec(),
        },
    )

    separated_regions = moriio_layout.iter_layer_registration_regions(
        "separated", separated, worker.layer_to_spec
    )
    interleaved_regions = moriio_layout.iter_layer_registration_regions(
        "interleaved", interleaved, worker.layer_to_spec
    )
    indexer_regions = moriio_layout.iter_layer_registration_regions(
        "indexer", indexer, worker.layer_to_spec
    )

    assert [region[0].data_ptr() for region in separated_regions] == [
        separated[0].data_ptr(),
        separated[1].data_ptr(),
    ]
    assert separated_regions[0][1] == 8 * 48
    assert separated_regions[1][1] == 8 * 48

    assert len(interleaved_regions) == 1
    assert interleaved_regions[0][0].data_ptr() == interleaved.data_ptr()
    assert interleaved_regions[0][1] == 8 * 2 * 48

    assert len(indexer_regions) == 1
    assert indexer_regions[0][0].data_ptr() == indexer.data_ptr()
    assert indexer_regions[0][1] == 8 * 24


def test_registration_regions_use_layer_num_blocks():
    cache = torch.empty((4, 2, 4, 2, 3), dtype=torch.bfloat16)
    worker = _worker({"layer": cache}, {"layer": _full_spec()}, num_blocks=8)

    regions = moriio_layout.iter_layer_registration_regions(
        "layer", cache, worker.layer_to_spec
    )

    assert len(regions) == 1
    assert regions[0][1] == 4 * 2 * 48


def test_unsupported_shape_raises_value_error():
    cache = torch.empty((8, 4, 2, 3), dtype=torch.bfloat16)
    worker = _worker({"layer": cache}, {"layer": _full_spec()})

    with pytest.raises(ValueError, match="Unsupported MoRIIO K/V cache shape"):
        moriio_layout.get_layer_transfer_geometry("layer", cache, worker.layer_to_spec)


# ---------------------------------------------------------------------------
# Mamba/KDA (conv + ssm) transfer geometry
# ---------------------------------------------------------------------------


def _slot_strided(num_slots, per_slot_shape, slot_stride, dtype=torch.bfloat16):
    """Build a slot-strided view: shape (num_slots, *per_slot_shape) whose slot
    stride (dim 0) exceeds the per-slot element count, mimicking the K3 KDA
    conv/ssm cache tensors."""
    inner = 1
    for d in per_slot_shape:
        inner *= d
    assert slot_stride >= inner
    backing = torch.zeros(num_slots * slot_stride, dtype=dtype)
    inner_strides = []
    acc = 1
    for d in reversed(per_slot_shape):
        inner_strides.insert(0, acc)
        acc *= d
    return backing.as_strided((num_slots, *per_slot_shape), (slot_stride, *inner_strides))


def _gdn_split_info(conv_rows=3, key_dim=4, value_dim=8, dtype_size=2):
    """A GDN-style conv split: sub-projections (key, key, value)."""
    conv_dim = 2 * key_dim + value_dim
    conv_state_bytes = conv_dim * conv_rows * dtype_size
    ssm_state_bytes = 64  # opaque here; ssm size is taken from the tensor
    return MambaConvSplitInfo(
        conv_rows=conv_rows,
        local_proj_dims=(key_dim, key_dim, value_dim),
        conv_dtype_size=dtype_size,
        ssm_sizes=(conv_state_bytes, ssm_state_bytes),
    )


def test_mamba_transfer_geometry_is_slot_strided():
    split = _gdn_split_info()
    conv_dim = sum(split.local_proj_dims)
    conv = _slot_strided(5, (conv_dim, split.conv_rows), slot_stride=200)
    ssm = _slot_strided(5, (2, 4, 4), slot_stride=64)

    geom = moriio_layout.get_mamba_transfer_geometry("kda.0", conv, ssm, {})

    assert geom.num_slots == 5
    assert geom.conv_slot_stride == 200
    assert geom.ssm_slot_stride == 64
    assert geom.conv_slot_bytes == conv_dim * split.conv_rows * conv.element_size()
    assert geom.ssm_slot_bytes == (2 * 4 * 4) * ssm.element_size()
    # Region spans the full slot-strided extent (shape[0] * stride(0)), not just
    # numel(), so the highest-slot transfers stay in-bounds.
    assert geom.conv_region_len == conv.shape[0] * conv.stride(0) * conv.element_size()
    assert geom.ssm_region_len == ssm.shape[0] * ssm.stride(0) * ssm.element_size()


def test_mamba_conv_ssm_offsets_homogeneous_tp():
    split = _gdn_split_info()
    conv_dim = sum(split.local_proj_dims)
    conv = _slot_strided(6, (conv_dim, split.conv_rows), slot_stride=200)
    ssm = _slot_strided(6, (2, 4, 4), slot_stride=64)
    geom = moriio_layout.get_mamba_transfer_geometry("kda.0", conv, ssm, {})

    local_slots = [1, 2]
    remote_slots = [1, 2]
    local_offs, remote_offs, sizes = moriio_layout.compute_mamba_conv_ssm_offsets(
        "kda.0", conv, ssm, {}, local_slots, remote_slots, split,
        tp_ratio=1, tp_rank=0, world_size=1,
    )

    n_conv = moriio_layout.compute_mamba_conv_split_count(local_slots, split)
    assert n_conv == len(local_slots) * len(split.local_proj_dims)
    assert len(local_offs) == n_conv + len(local_slots)

    # Homogeneous TP: remote conv slices equal local slices; since local==remote
    # slots, conv local and remote offsets coincide.
    loc_co = split.local_conv_offsets
    rem_co = split.remote_conv_offsets(0, 1)
    k = 0
    esz = conv.element_size()
    for ls, rs in zip(local_slots, remote_slots):
        lbase = ls * geom.conv_slot_stride * esz
        rbase = rs * geom.conv_slot_stride * esz
        for (loff, _), (roff, rsz) in zip(loc_co, rem_co):
            assert local_offs[k] == lbase + loff
            assert remote_offs[k] == rbase + roff
            assert sizes[k] == rsz
            k += 1

    # SSM: whole per-slot block, no head sharding at tp_ratio == 1.
    essz = ssm.element_size()
    for j, (ls, rs) in enumerate(zip(local_slots, remote_slots)):
        assert local_offs[n_conv + j] == ls * geom.ssm_slot_stride * essz
        assert remote_offs[n_conv + j] == rs * geom.ssm_slot_stride * essz
        assert sizes[n_conv + j] == geom.ssm_slot_bytes


def test_mamba_conv_ssm_offsets_heterogeneous_tp_is_gated():
    # Heterogeneous-TP mamba transfer is a documented follow-up (remote slot
    # stride + multi-rank fan-in gather). Until then the offset helper must fail
    # loudly rather than silently transfer wrong bytes, since the mamba path
    # bypasses the attention KV-head validator.
    split = _gdn_split_info()
    conv_dim = sum(split.local_proj_dims)
    conv = _slot_strided(4, (conv_dim, split.conv_rows), slot_stride=200)
    ssm = _slot_strided(4, (2, 4, 4), slot_stride=64)
    with pytest.raises(NotImplementedError):
        moriio_layout.compute_mamba_conv_ssm_offsets(
            "kda.0", conv, ssm, {}, [0], [3], split,
            tp_ratio=2, tp_rank=1, world_size=1,
        )
