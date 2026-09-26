# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    TransferPriority,
    parse_transfer_priority,
    transfer_ordered,
    transfer_priority_from_extra_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheConfig


class _StubConnector(KVConnectorBase_V1):
    def start_load_kv(self, forward_context, **kwargs):
        return None

    def wait_for_layer_load(self, layer_name):
        return None

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        return None

    def wait_for_save(self):
        return None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return (0, False)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        return None

    def build_connector_meta(self, scheduler_output):
        return None


def _recv_params() -> dict:
    return {
        "remote_block_ids": ([],),
        "remote_engine_id": "engine",
        "remote_request_id": "remote-req",
        "remote_host": "127.0.0.1",
        "remote_port": 5555,
    }


def test_req_meta_priority_defaults_to_zero():
    metadata = NixlConnectorMetadata()
    metadata.add_new_req_to_save("save-req", ([],), {})
    metadata.add_new_req_to_recv("recv-req", ([],), _recv_params())
    assert metadata.reqs_to_save["save-req"].priority == 0
    assert metadata.reqs_to_recv["recv-req"].priority == 0


def test_req_meta_priority_is_copied_and_orders_save_queue():
    metadata = NixlConnectorMetadata()
    metadata.add_new_req_to_save("low", ([1],), {}, priority=5)
    metadata.add_new_req_to_save("high", ([2],), {}, priority=0)
    metadata.add_new_req_to_save("mid", ([3],), {}, priority=2)

    ordered = [
        req_id for req_id, _ in metadata.iter_reqs_by_priority(metadata.reqs_to_save)
    ]
    assert ordered == ["high", "mid", "low"]
    assert metadata.reqs_to_save["low"].priority == 5


def test_req_meta_priority_orders_recv_queue():
    metadata = NixlConnectorMetadata()
    metadata.add_new_req_to_recv("low", ([],), _recv_params(), priority=7)
    metadata.add_new_req_to_recv("high", ([],), _recv_params(), priority=1)

    ordered = [
        req_id for req_id, _ in metadata.iter_reqs_by_priority(metadata.reqs_to_recv)
    ]
    assert ordered == ["high", "low"]


def test_parse_transfer_priority_aliases():
    assert parse_transfer_priority("critical") is TransferPriority.CRITICAL
    assert parse_transfer_priority("p2d") is TransferPriority.CRITICAL
    assert parse_transfer_priority("store") is TransferPriority.BACKGROUND
    assert parse_transfer_priority("BACKGROUND") is TransferPriority.BACKGROUND
    assert parse_transfer_priority(2) is TransferPriority.CRITICAL
    assert parse_transfer_priority("not-a-role") is None
    assert parse_transfer_priority(True) is None


def test_transfer_priority_from_extra_config():
    assert (
        transfer_priority_from_extra_config({"role": "p2d"})
        is TransferPriority.CRITICAL
    )
    assert (
        transfer_priority_from_extra_config({"default_transfer_priority": 1})
        is TransferPriority.BACKGROUND
    )
    assert (
        transfer_priority_from_extra_config(
            {"default_transfer_priority": "CRITICAL", "role": "store"}
        )
        is TransferPriority.CRITICAL
    )


def _make_stub(extra_config: dict | None = None) -> _StubConnector:
    vllm_config = MagicMock()
    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="ExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config or {},
    )
    return _StubConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.WORKER,
        kv_cache_config=KVCacheConfig(
            num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
        ),
    )


def test_connector_defaults_to_background():
    assert _make_stub().default_transfer_priority is TransferPriority.BACKGROUND


def test_nixl_connector_defaults_to_critical():
    assert NixlBaseConnector._default_transfer_priority is TransferPriority.CRITICAL


def test_extra_config_overrides_default_transfer_priority():
    connector = _make_stub({"role": "critical"})
    assert connector.default_transfer_priority is TransferPriority.CRITICAL


def _mock_sub_connector(priority: TransferPriority) -> MagicMock:
    sub = MagicMock(spec_set=KVConnectorBase_V1)
    sub.default_transfer_priority = priority
    return sub


def _multi_connector_of(*subs: MagicMock) -> MultiConnector:
    """Build a MultiConnector around mocks, mirroring what __init__ sets up."""
    connector = object.__new__(MultiConnector)
    connector._connectors = list(subs)
    connector._transfer_ordered_connectors = transfer_ordered(connector._connectors)
    return connector


def test_multi_connector_save_orders_critical_before_background():
    background = _mock_sub_connector(TransferPriority.BACKGROUND)
    critical = _mock_sub_connector(TransferPriority.CRITICAL)
    # Config order is Store then P→D; save fan-out must still run P→D first.
    connector = _multi_connector_of(background, critical)

    order: list[str] = []
    background.save_kv_layer.side_effect = lambda *args, **kwargs: order.append("bg")
    critical.save_kv_layer.side_effect = lambda *args, **kwargs: order.append("crit")

    connector.save_kv_layer("layer", MagicMock(), MagicMock())
    assert order == ["crit", "bg"]


def test_multi_connector_equal_priority_keeps_config_order():
    first = _mock_sub_connector(TransferPriority.BACKGROUND)
    second = _mock_sub_connector(TransferPriority.BACKGROUND)
    connector = _multi_connector_of(first, second)

    order: list[str] = []
    first.save_kv_layer.side_effect = lambda *args, **kwargs: order.append("first")
    second.save_kv_layer.side_effect = lambda *args, **kwargs: order.append("second")

    connector.save_kv_layer("layer", MagicMock(), MagicMock())
    assert order == ["first", "second"]


def test_multi_connector_start_load_orders_critical_first():
    background = _mock_sub_connector(TransferPriority.BACKGROUND)
    critical = _mock_sub_connector(TransferPriority.CRITICAL)
    connector = _multi_connector_of(background, critical)

    order: list[str] = []
    background.start_load_kv.side_effect = lambda *args, **kwargs: order.append("bg")
    critical.start_load_kv.side_effect = lambda *args, **kwargs: order.append("crit")

    connector.start_load_kv(MagicMock())
    assert order == ["crit", "bg"]
