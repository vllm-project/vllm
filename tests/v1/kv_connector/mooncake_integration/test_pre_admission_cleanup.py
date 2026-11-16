# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise a pre-admission rejection through a real OpenAI server."""

import json

import httpx
import msgspec
import zmq

from tests.utils import RemoteOpenAIServer
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def test_empty_chat_rejection_sends_prefill_cleanup():
    """A rejected decode leg must notify the producer holding its KV blocks."""
    bootstrap = MooncakeBootstrapServer("127.0.0.1", 0)
    context = zmq.Context()
    producer = context.socket(zmq.ROUTER)
    producer_port = producer.bind_to_random_port("tcp://127.0.0.1")
    try:
        bootstrap.start()
        bootstrap_url = f"http://127.0.0.1:{bootstrap.port}"
        registration = httpx.post(
            f"{bootstrap_url}/register",
            json={
                "engine_id": "prefill-engine",
                "dp_rank": 0,
                "tp_rank": 0,
                "pp_rank": 0,
                "addr": f"tcp://127.0.0.1:{producer_port}",
            },
            timeout=10,
        )
        registration.raise_for_status()

        transfer_config = {
            "kv_connector": "MooncakeConnector",
            "kv_role": "kv_consumer",
            "kv_connector_extra_config": {"mooncake_protocol": "tcp"},
        }
        args = [
            "--enforce-eager",
            "--max-model-len",
            "512",
            "--kv-cache-memory-bytes",
            "67108864",
            "--gpu-memory-utilization",
            "0.35",
            "--disable-flashinfer-autotune",
            "--kv-transfer-config",
            json.dumps(transfer_config),
        ]
        env = {
            "MC_FORCE_TCP": "1",
            "VLLM_HOST_IP": "127.0.0.1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
        with RemoteOpenAIServer(MODEL, args, env_dict=env) as decoder:
            response = httpx.post(
                decoder.url_for("v1/chat/completions"),
                json={
                    "model": MODEL,
                    "messages": [],
                    "request_id": "rejected-decode",
                    "kv_transfer_params": {
                        "do_remote_prefill": True,
                        "do_remote_decode": False,
                        "remote_engine_id": "prefill-engine",
                        "remote_bootstrap_addr": bootstrap_url,
                        "transfer_id": "pending-transfer",
                    },
                },
                timeout=30,
            )
            assert response.status_code == 400
            assert response.json()["error"]["param"] == "messages"
            assert producer.poll(timeout=20_000), "no cleanup pull reached the producer"
            identity, payload = producer.recv_multipart()
            metadata = msgspec.msgpack.decode(payload, type=MooncakeXferMetadata)
            assert list(metadata.req_blocks.values()) == [("pending-transfer", [])]
            reply = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.FINISH,
                ok_reqs=list(metadata.req_blocks),
            )
            producer.send_multipart([identity, msgspec.msgpack.encode(reply)])
    finally:
        bootstrap.shutdown()
        producer.close(linger=0)
        context.term()
