# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL request failure isolation with real E/PD servers and an injected NACK.

Requires two CUDA GPUs and NIXL. Set MODEL to override the small default model.
"""

import asyncio
import copy
import json
import os
import uuid
from pathlib import Path

import aiohttp
import msgspec
import pytest
import zmq
import zmq.asyncio
from PIL import Image

from examples.disaggregated.disaggregated_encoder.disagg_epd_proxy import (
    rewrite_for_decode,
)
from tests.utils import RemoteOpenAIServer
from vllm.distributed.ec_transfer.ec_connector.cpu.protocol import (
    XferAck,
    XferReq,
    XferStatus,
)
from vllm.multimodal.utils import encode_image_url
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")


@pytest.fixture(scope="module")
def servers():
    pytest.importorskip("nixl")
    if not current_platform.is_cuda() or current_platform.device_count() < 2:
        pytest.skip("Requires two CUDA GPUs")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    common = [
        "--host",
        "127.0.0.1",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.7",
        "--no-enable-prefix-caching",
    ]

    def config(role):
        return json.dumps(
            {
                "ec_connector": "ECCPUConnector",
                "ec_role": role,
                "ec_connector_extra_config": {
                    "ec_enable_nixl": True,
                    "ec_cpu_bytes": 64 << 20,
                },
            }
        )

    with (
        RemoteOpenAIServer(
            MODEL,
            common
            + [
                "--mm-encoder-only",
                "--disable-hybrid-kv-cache-manager",
                "--ec-transfer-config",
                config("ec_producer"),
            ],
            env_dict={
                "CUDA_VISIBLE_DEVICES": devices[0],
                "VLLM_EC_SIDE_CHANNEL_HOST": "127.0.0.1",
                "VLLM_EC_SIDE_CHANNEL_PORT": str(get_open_port()),
            },
        ) as encoder,
        RemoteOpenAIServer(
            MODEL,
            common
            + ["--enable-mm-embeds", "--ec-transfer-config", config("ec_consumer")],
            env_dict={"CUDA_VISIBLE_DEVICES": devices[1]},
        ) as consumer,
    ):
        yield encoder, consumer


async def _post(session, server, body):
    async with session.post(server.url_for("v1/chat/completions"), json=body) as r:
        return r.status, await r.json()


async def _encode(session, encoder):
    image = Image.open(Path(__file__).with_name("hato.jpg")).convert("RGB")
    image.thumbnail((224, 224))
    item_uuid = uuid.uuid4().hex  # Avoid both EC and processor cache hits.
    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_url(image)},
                        "uuid": item_uuid,
                    },
                    {"type": "text", "text": "Describe the image briefly."},
                ],
            }
        ],
        "max_tokens": 16,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    status, result = await _post(session, encoder, body)
    assert status == 200, result
    params = result["ec_transfer_params"]
    assert len(params) == 1, params
    ((mm_hash, handle),) = params.items()
    assert handle["metadata"] and handle["peer_host"] and handle["size_bytes"] > 0
    rewritten = rewrite_for_decode(
        body,
        {
            0: {
                **handle["metadata"],
                "mm_hash": item_uuid,
                "ec_mm_hash": mm_hash,
            }
        },
    )
    assert rewritten["messages"][0]["content"][0]["type"] == "image_embeds"
    for request in (body, rewritten):
        request["ec_transfer_params"] = {item_uuid: copy.deepcopy(handle)}
    return body, rewritten, item_uuid


@pytest.mark.asyncio
@pytest.mark.parametrize("local_image", [False, True], ids=["metadata-only", "pixels"])
async def test_missing_remote_encoding_is_request_scoped(servers, local_image):
    """A terminal NACK must preserve local fallback and subsequent serving."""
    encoder, consumer = servers
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        original, rewritten, mm_hash = await _encode(session, encoder)
        request = original if local_image else rewritten

        # Keep the real encoder's shape/metadata; replace only the control peer.
        # The socket stays bound until completion, so no port or timing race is
        # needed to make the consumer receive NACK_MISSING.
        with zmq.asyncio.Context() as ctx, ctx.socket(zmq.ROUTER) as router:
            router.setsockopt(zmq.LINGER, 0)
            port = router.bind_to_random_port("tcp://127.0.0.1")
            request["ec_transfer_params"][mm_hash].update(
                peer_host="127.0.0.1", peer_port=port
            )

            async def reject_read():
                identity, delimiter, payload = await router.recv_multipart()
                req = msgspec.msgpack.decode(payload, type=XferReq)
                assert req.mm_hash == mm_hash
                ack = XferAck(
                    mm_hash=req.mm_hash,
                    status=XferStatus.NACK_MISSING,
                    session_id=req.session_id,
                )
                await router.send_multipart(
                    [identity, delimiter, msgspec.msgpack.encode(ack)]
                )

            rejection = asyncio.create_task(asyncio.wait_for(reject_read(), 30))
            try:
                status, result = await _post(session, consumer, request)
                # A frontend rejection must not count as successful injection.
                await rejection
            finally:
                rejection.cancel()
                await asyncio.gather(rejection, return_exceptions=True)

        if local_image:
            assert status == 200, result
            assert result["choices"][0]["message"]["content"], result
        else:
            assert status == 500, result
            assert result["error"]["type"] == "InternalServerError", result

        async with session.get(consumer.url_for("health")) as health:
            assert health.status == 200, "The failed read killed EngineCore"
        # A fresh metadata-only image must cross the real NIXL data path after
        # the fault, not just pass a frontend health check or hit a local cache.
        _, healthy, _ = await _encode(session, encoder)
        status, result = await _post(session, consumer, healthy)
        assert status == 200, result
        assert result["choices"][0]["message"]["content"], result
