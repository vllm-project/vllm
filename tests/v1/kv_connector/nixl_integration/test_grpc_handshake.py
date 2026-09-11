# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL handshake metadata served over the Rust frontend's gRPC control plane.

Expects a prefill and a decode instance started with ``vllm-rs serve
--grpc-port`` and a toy proxy in front of them; see
run_grpc_handshake_test.sh.
"""

import os
import sys
from pathlib import Path

import msgspec
import openai
import pytest
import requests
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    NixlAgentMetadata,
    NixlHandshakePayload,
)

grpc = pytest.importorskip("grpc")

PREFILL_PORT = int(os.environ["PREFILL_PORT"])
DECODE_PORT = int(os.environ["DECODE_PORT"])
PROXY_PORT = int(os.environ["PROXY_PORT"])
PREFILL_GRPC_PORT = int(os.environ["PREFILL_GRPC_PORT"])
DECODE_GRPC_PORT = int(os.environ["DECODE_GRPC_PORT"])
PREFILL_SIDE_CHANNEL_PORT = int(os.environ["PREFILL_SIDE_CHANNEL_PORT"])
# Stubs generated from rust/proto/control.proto by the launcher script. protoc must
# run in a separate process: grpc_tools' bundled libprotobuf crashes on dlopen once
# torch is loaded in the same interpreter.
CONTROL_PB_DIR = Path(os.environ["CONTROL_PB_DIR"])

PROMPT = (
    "Prefill and decode disaggregation splits an inference request across two "
    "engines: the prefill engine processes the prompt and hands its KV cache to "
    "the decode engine, which then generates the completion. As a result,"
)


@pytest.fixture(scope="module")
def control_pb():
    sys.path.insert(0, str(CONTROL_PB_DIR))
    import control_pb2
    import control_pb2_grpc

    return control_pb2, control_pb2_grpc


@pytest.fixture(scope="module")
def stubs(control_pb):
    _, control_pb2_grpc = control_pb
    channels = [
        grpc.insecure_channel(f"127.0.0.1:{PREFILL_GRPC_PORT}"),
        grpc.insecure_channel(f"127.0.0.1:{DECODE_GRPC_PORT}"),
    ]
    yield tuple(control_pb2_grpc.KvTransferStub(channel) for channel in channels)
    for channel in channels:
        channel.close()


def _single_engine(info):
    assert len(info.engines) == 1, info
    engine = info.engines[0]
    assert engine.connector == "NixlConnector"
    assert engine.role == "kv_both"
    assert engine.tensor_parallel_size == 1
    assert engine.pipeline_parallel_size == 1
    assert engine.kv_block_size > 0
    assert len(engine.compatibility_hash) == 64
    int(engine.compatibility_hash, 16)
    return engine


def _side_channel_payload(port: int, pp_rank: int, tp_rank: int) -> bytes:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    try:
        sock.connect(f"tcp://127.0.0.1:{port}")
        sock.send(msgspec.msgpack.encode((GET_META_MSG, pp_rank, tp_rank)))
        payload, _perf_counter = sock.recv_multipart()
        return payload
    finally:
        sock.close()
        ctx.term()


def test_kv_transfer_info_matches_across_prefill_and_decode(control_pb, stubs):
    control_pb2, _ = control_pb
    prefill, decode = stubs
    p_engine = _single_engine(
        prefill.GetKvTransferInfo(control_pb2.GetKvTransferInfoRequest())
    )
    d_engine = _single_engine(
        decode.GetKvTransferInfo(control_pb2.GetKvTransferInfoRequest())
    )
    assert p_engine.engine_id != d_engine.engine_id
    assert p_engine.compatibility_hash == d_engine.compatibility_hash
    assert p_engine.kv_block_size == d_engine.kv_block_size


def test_handshake_metadata_matches_side_channel(control_pb, stubs):
    control_pb2, _ = control_pb
    prefill, _ = stubs
    engine = _single_engine(
        prefill.GetKvTransferInfo(control_pb2.GetKvTransferInfoRequest())
    )

    response = prefill.GetKvHandshakeMetadata(
        control_pb2.GetKvHandshakeMetadataRequest(engine_id=engine.engine_id)
    )
    assert [(rank.pp_rank, rank.tp_rank) for rank in response.ranks] == [(0, 0)]
    rank = response.ranks[0]
    assert rank.encoding == "msgpack"
    assert rank.compatibility_hash == engine.compatibility_hash

    handshake = msgspec.msgpack.Decoder(NixlHandshakePayload).decode(rank.payload)
    assert handshake.compatibility_hash == engine.compatibility_hash
    agent = msgspec.msgpack.Decoder(NixlAgentMetadata).decode(
        handshake.agent_metadata_bytes
    )
    assert agent.engine_id == engine.engine_id
    assert agent.block_size == engine.kv_block_size
    assert agent.num_blocks > 0

    assert rank.payload == _side_channel_payload(PREFILL_SIDE_CHANNEL_PORT, 0, 0)

    again = prefill.GetKvHandshakeMetadata(
        control_pb2.GetKvHandshakeMetadataRequest(engine_id=engine.engine_id)
    )
    assert again.ranks[0].payload == rank.payload


def test_unknown_engine_id_is_not_found(control_pb, stubs):
    control_pb2, _ = control_pb
    prefill, _ = stubs
    with pytest.raises(grpc.RpcError) as excinfo:
        prefill.GetKvHandshakeMetadata(
            control_pb2.GetKvHandshakeMetadataRequest(engine_id="no-such-engine")
        )
    assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND


def test_kv_transfer_params_point_at_discovered_engine(control_pb, stubs):
    """What P hands a router in kv_transfer_params agrees with discovery."""
    control_pb2, _ = control_pb
    prefill, _ = stubs
    engine = _single_engine(
        prefill.GetKvTransferInfo(control_pb2.GetKvTransferInfoRequest())
    )
    model = (
        openai.OpenAI(api_key="x", base_url=f"http://127.0.0.1:{PREFILL_PORT}/v1")
        .models.list()
        .data[0]
        .id
    )

    response = requests.post(
        f"http://127.0.0.1:{PREFILL_PORT}/v1/completions",
        json={
            "model": model,
            "prompt": PROMPT,
            "max_tokens": 1,
            "temperature": 0,
            "kv_transfer_params": {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "remote_engine_id": None,
                "remote_block_ids": None,
                "remote_host": None,
                "remote_port": None,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    params = response.json()["kv_transfer_params"]
    assert params["remote_engine_id"] == engine.engine_id
    assert params["remote_port"] == PREFILL_SIDE_CHANNEL_PORT
    assert params["remote_control_port"] == PREFILL_GRPC_PORT
    assert params["tp_size"] == engine.tensor_parallel_size
    assert params["remote_block_ids"]


def test_disaggregated_completion_matches_prefill_only():
    """The decoder runs with handshake_transport=grpc, so this completion can
    only succeed if its frontend pushed the prefiller's handshake metadata."""
    proxy = openai.OpenAI(api_key="x", base_url=f"http://127.0.0.1:{PROXY_PORT}/v1")
    prefill = openai.OpenAI(api_key="x", base_url=f"http://127.0.0.1:{PREFILL_PORT}/v1")
    model = prefill.models.list().data[0].id
    kwargs = dict(model=model, prompt=PROMPT, temperature=0, max_tokens=32)
    via_proxy = proxy.completions.create(**kwargs).choices[0].text
    prefill_only = prefill.completions.create(**kwargs).choices[0].text
    assert via_proxy == prefill_only
