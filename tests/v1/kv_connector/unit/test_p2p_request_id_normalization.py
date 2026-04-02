import threading
from types import SimpleNamespace

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    P2pNcclConnector,
    P2pNcclConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    P2pNcclEngine,
)
from vllm.utils.request_id import normalize_request_id

TRANSPORT_REQUEST_ID = (
    "___prefill_addr_10.0.0.1:31000___decode_addr_10.0.0.2:32000_deadbeef"
)
RAW_REQUEST_ID = f"cmpl-{TRANSPORT_REQUEST_ID}-0-a1b2c3d4"
NORMALIZED_REQUEST_ID = normalize_request_id(RAW_REQUEST_ID)


class FakeP2pNcclEngine:
    def __init__(self, recv_tensor: torch.Tensor):
        self._recv_tensor = recv_tensor
        self.recv_calls: list[tuple[str, str]] = []
        self.send_calls: list[tuple[str, torch.Tensor, str]] = []

    def recv_tensor(self, tensor_id: str, remote_address: str) -> torch.Tensor:
        self.recv_calls.append((tensor_id, remote_address))
        return self._recv_tensor.clone()

    def send_tensor(
        self, tensor_id: str, tensor: torch.Tensor, remote_address: str
    ) -> bool:
        self.send_calls.append((tensor_id, tensor.clone(), remote_address))
        return True


def _make_metadata(request_id: str) -> P2pNcclConnectorMetadata:
    metadata = P2pNcclConnectorMetadata()
    metadata.add_request(
        request_id=request_id,
        token_ids=[1, 2, 3],
        block_ids=[0],
        block_size=16,
    )
    return metadata


def test_start_load_kv_uses_normalized_tensor_id() -> None:
    recv_tensor = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
    fake_engine = FakeP2pNcclEngine(recv_tensor)
    connector = object.__new__(P2pNcclConnector)
    connector.is_producer = False
    connector.p2p_nccl_engine = fake_engine
    connector._rank = 0
    connector._get_connector_metadata = lambda: _make_metadata(RAW_REQUEST_ID)

    layer = SimpleNamespace(kv_cache=torch.zeros_like(recv_tensor))
    forward_context = SimpleNamespace(
        attn_metadata=object(),
        no_compile_layers={"layer0": layer},
    )

    connector.start_load_kv(forward_context)

    assert fake_engine.recv_calls == [
        (f"{NORMALIZED_REQUEST_ID}#layer0", "10.0.0.1:31000")
    ]
    assert torch.equal(layer.kv_cache, recv_tensor)


def test_save_kv_layer_uses_normalized_tensor_id() -> None:
    kv_layer = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
    fake_engine = FakeP2pNcclEngine(kv_layer)
    connector = object.__new__(P2pNcclConnector)
    connector.is_producer = True
    connector.p2p_nccl_engine = fake_engine
    connector._rank = 0
    connector._get_connector_metadata = lambda: _make_metadata(RAW_REQUEST_ID)

    connector.save_kv_layer(
        layer_name="layer0",
        kv_layer=kv_layer,
        attn_metadata=object(),
    )

    assert len(fake_engine.send_calls) == 1
    tensor_id, tensor, remote_address = fake_engine.send_calls[0]
    assert tensor_id == f"{NORMALIZED_REQUEST_ID}#layer0"
    assert remote_address == "10.0.0.2:32000"
    assert torch.equal(tensor, kv_layer)


def test_get_finished_cleans_up_normalized_request_state() -> None:
    engine = object.__new__(P2pNcclEngine)
    engine.recv_store_cv = threading.Condition()
    engine.send_store_cv = threading.Condition()
    engine.pool = SimpleNamespace(free=lambda *_args, **_kwargs: None)
    engine.send_type = "GET"

    normalized_tensor_id = f"{NORMALIZED_REQUEST_ID}#layer0"
    send_tensor = torch.ones(2, dtype=torch.float32)

    engine.recv_store = {normalized_tensor_id: torch.ones(1, dtype=torch.float32)}
    engine.send_store = {normalized_tensor_id: send_tensor}
    engine.recv_request_id_to_tensor_ids = {NORMALIZED_REQUEST_ID: {normalized_tensor_id}}
    engine.send_request_id_to_tensor_ids = {NORMALIZED_REQUEST_ID: {normalized_tensor_id}}
    engine.buffer_size = send_tensor.element_size() * send_tensor.numel()

    engine.have_sent_tensor_id(f"{RAW_REQUEST_ID}#layer1")
    engine.have_received_tensor_id(f"{RAW_REQUEST_ID}#layer1")

    assert NORMALIZED_REQUEST_ID in engine.send_request_id_to_tensor_ids
    assert NORMALIZED_REQUEST_ID in engine.recv_request_id_to_tensor_ids

    finished = engine.get_finished({RAW_REQUEST_ID}, ["layer0"])

    assert finished == (None, None)
    assert normalized_tensor_id not in engine.recv_store
    assert normalized_tensor_id not in engine.send_store
    assert NORMALIZED_REQUEST_ID not in engine.recv_request_id_to_tensor_ids
    assert NORMALIZED_REQUEST_ID not in engine.send_request_id_to_tensor_ids
    assert engine.buffer_size == 0
