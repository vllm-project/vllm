# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for multimodal features through the /inference/v1/generate endpoint.

Mirrors test_serving_tokens.py but exercises the multimodal piping
using Qwen/Qwen3-VL-2B-Instruct end-to-end via the server's /render ->
/generate -> /detokenize path. Intentionally avoids running the HF
processor in the pytest parent process to keep os.fork() in other
tests (e.g. entrypoints/rl/weight_transfer/test_offline.py) deadlock-free.
"""

import asyncio
import hashlib
import json
import threading
import uuid
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
import pytest_asyncio
import torch
from PIL import Image
from safetensors.torch import save as save_safetensors

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import (
    decode_mm_kwargs_item,
    encode_mm_kwargs_item,
)
from vllm.multimodal.inputs import MultiModalBatchedField, MultiModalKwargsItem
from vllm.multimodal.utils import encode_image_url

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
GEN_ENDPOINT = "/inference/v1/generate"
RENDER_ENDPOINT = "/v1/chat/completions/render"
DETOKENIZE_ENDPOINT = "/detokenize"
S3_BUCKET = "mm-tensors"
S3_PREFIX = "rollouts"
# GETs of keys under `concurrent/` wait until this many are in flight.
CONCURRENT_READS = threading.Barrier(2, timeout=30)
# Keys of every GET the store served.
GET_PATHS: list[str] = []


@pytest.fixture(scope="module")
def test_image():
    return Image.new("RGB", (224, 224), color=(255, 0, 0))


@pytest.fixture(scope="module")
def s3_store():
    """Objects served by a local stand-in for an S3 endpoint, by `bucket/key`."""
    objects: dict[str, bytes] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.removeprefix("/")
            GET_PATHS.append(path)
            if "/concurrent/" in path:
                CONCURRENT_READS.wait()
            body = objects.get(path)
            self.send_response(404 if body is None else 200)
            body = body or b"<Error><Code>NoSuchKey</Code></Error>"
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", objects
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


@pytest.fixture(scope="module")
def server(s3_store):
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--enable-scale-out",
        "--mm-tensor-refs-config",
        json.dumps(
            {
                "s3_allowed_uri_prefix": f"s3://{S3_BUCKET}/{S3_PREFIX}",
                # Room for about three 224x224 items, so a leaked
                # reservation blocks later requests.
                "max_request_bytes": 1024**2,
                "max_reserved_bytes": 2 * 1024**2,
            }
        ),
    ]
    env = {
        "S3_ENDPOINT_URL": s3_store[0],
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "AWS_DEFAULT_REGION": "us-east-1",
    }

    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=env) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server: RemoteOpenAIServer):
    transport = httpx.AsyncHTTPTransport(uds=server.uds) if server.uds else None
    headers = {"Authorization": f"Bearer {server.DUMMY_API_KEY}"}
    async with httpx.AsyncClient(
        transport=transport,
        base_url=server.url_root,
        timeout=600,
        headers=headers,
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_render_to_generate_roundtrip(client, test_image):
    """End-to-end: render a multimodal chat -> feed into generate -> decode.

    All preprocessing and detokenization happens in the server subprocess;
    the pytest parent never imports transformers or touches torch tensors.
    """
    data_url = encode_image_url(test_image, format="PNG")

    render_payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {
                        "type": "text",
                        "text": "What color is this image? Answer in one word.",
                    },
                ],
            }
        ],
    }

    render_resp = await client.post(RENDER_ENDPOINT, json=render_payload)
    render_resp.raise_for_status()
    render_data = render_resp.json()

    # Validate render output structure: keys exist and values are non-empty
    # and well-typed.
    assert "token_ids" in render_data
    assert isinstance(render_data["token_ids"], list)
    assert len(render_data["token_ids"]) > 0
    assert all(isinstance(t, int) for t in render_data["token_ids"])

    assert "features" in render_data
    features = render_data["features"]
    assert features is not None
    assert isinstance(features, dict)

    assert "mm_hashes" in features
    assert "image" in features["mm_hashes"]
    image_hashes = features["mm_hashes"]["image"]
    assert isinstance(image_hashes, list)
    assert len(image_hashes) > 0
    assert all(isinstance(h, str) and h for h in image_hashes)

    assert "mm_placeholders" in features
    assert "image" in features["mm_placeholders"]
    image_placeholders = features["mm_placeholders"]["image"]
    assert isinstance(image_placeholders, list)
    assert len(image_placeholders) > 0
    for p in image_placeholders:
        assert isinstance(p.get("offset"), int)
        assert isinstance(p.get("length"), int)
        assert p["length"] > 0

    assert "kwargs_data" in features
    assert "image" in features["kwargs_data"]
    assert len(features["kwargs_data"]["image"]) > 0

    # Build generate request from render output
    generate_payload = render_data
    generate_payload["sampling_params"] = {
        "max_tokens": 10,
        "temperature": 0.0,
    }

    gen_resp = await client.post(GEN_ENDPOINT, json=generate_payload)
    gen_resp.raise_for_status()
    gen_data = gen_resp.json()

    assert "choices" in gen_data
    assert isinstance(gen_data["choices"], list)
    assert len(gen_data["choices"]) >= 1
    choice = gen_data["choices"][0]
    assert "token_ids" in choice
    assert isinstance(choice["token_ids"], list)
    assert len(choice["token_ids"]) > 0
    assert all(isinstance(t, int) for t in choice["token_ids"])

    detok_resp = await client.post(
        DETOKENIZE_ENDPOINT,
        json={"model": MODEL_NAME, "tokens": choice["token_ids"]},
    )
    detok_resp.raise_for_status()
    detok_data = detok_resp.json()
    assert "prompt" in detok_data
    text = detok_data["prompt"]
    assert isinstance(text, str)
    assert len(text) > 0
    assert "red" in text.lower(), (
        f"Expected model to identify the red image, got: {text!r}"
    )


def _collapse_placeholders(
    token_ids: list[int], mm_placeholders: dict[str, list[dict[str, int]]]
) -> list[int]:
    """Undo placeholder expansion: each expanded range becomes one token.

    `/generate` with `content_parts` expects unexpanded placeholders and
    expands them itself, so the roundtrip must start from a collapsed prompt.
    """
    ranges = sorted(
        (r["offset"], r["length"]) for rs in mm_placeholders.values() for r in rs
    )
    collapsed: list[int] = []
    pos = 0
    for offset, length in ranges:
        collapsed.extend(token_ids[pos:offset])
        collapsed.append(token_ids[offset])
        pos = offset + length
    collapsed.extend(token_ids[pos:])
    return collapsed


async def _render_image_prompt(client, *data_urls: str) -> dict:
    render_resp = await client.post(
        RENDER_ENDPOINT,
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *(
                            {"type": "image_url", "image_url": {"url": u}}
                            for u in data_urls
                        ),
                        {"type": "text", "text": "What color is this?"},
                    ],
                }
            ],
        },
    )
    render_resp.raise_for_status()
    return render_resp.json()


@pytest.mark.asyncio
async def test_content_parts_generates_tokens(client, test_image):
    """Unexpanded prompt + content_parts must expand exactly like /render."""
    data_url = encode_image_url(test_image, format="PNG")
    render_data = await _render_image_prompt(client, data_url)
    expanded = render_data["token_ids"]
    mm_placeholders = render_data["features"]["mm_placeholders"]
    unexpanded = _collapse_placeholders(expanded, mm_placeholders)
    assert len(unexpanded) < len(expanded)

    gen_resp = await client.post(
        GEN_ENDPOINT,
        json={
            "token_ids": unexpanded,
            "content_parts": [{"type": "image_url", "url": data_url}],
            "sampling_params": {"max_tokens": 10, "temperature": 0.0},
            "return_token_ids": True,
        },
    )
    gen_resp.raise_for_status()
    gen_data = gen_resp.json()

    assert "choices" in gen_data
    choice = gen_data["choices"][0]
    assert "token_ids" in choice
    assert len(choice["token_ids"]) > 0
    assert gen_data["prompt_token_ids"] == expanded
    assert gen_data["mm_placeholders"] == mm_placeholders
    assert gen_data["mm_hashes"] == render_data["features"]["mm_hashes"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [False, True])
async def test_features_with_content_parts(client, test_image, cached):
    """Earlier items as features plus a new raw item must match /render.

    With `cached`, features carry only the `mm_hashes` and `mm_placeholders`
    returned by turn 1 (as in the docs), so the earlier item resolves from the
    cache that request filled.
    """
    old_url = encode_image_url(test_image, format="PNG")
    new_url = encode_image_url(Image.new("RGB", (224, 224)), format="PNG")
    render_data = await _render_image_prompt(client, old_url, new_url)
    expanded = render_data["token_ids"]
    features = render_data["features"]
    old, new = features["mm_placeholders"]["image"]

    if cached:
        turn1 = await client.post(
            GEN_ENDPOINT,
            json={
                "token_ids": _collapse_placeholders(
                    expanded[: new["offset"]], {"image": [old]}
                ),
                "content_parts": [{"type": "image_url", "url": old_url}],
                "sampling_params": {"max_tokens": 1},
                "return_token_ids": True,
            },
        )
        turn1.raise_for_status()
        old_features = {
            "mm_hashes": turn1.json()["mm_hashes"],
            "mm_placeholders": turn1.json()["mm_placeholders"],
        }
    else:
        old_features = {k: {"image": v["image"][:1]} for k, v in features.items() if v}

    gen_resp = await client.post(
        GEN_ENDPOINT,
        json={
            "token_ids": _collapse_placeholders(expanded, {"image": [new]}),
            "features": old_features,
            "content_parts": [{"type": "image_url", "url": new_url}],
            "sampling_params": {"max_tokens": 1},
            "return_token_ids": True,
        },
    )
    gen_resp.raise_for_status()
    assert gen_resp.json()["prompt_token_ids"] == expanded
    assert gen_resp.json()["mm_placeholders"] == features["mm_placeholders"]
    assert gen_resp.json()["mm_hashes"] == features["mm_hashes"]


@pytest.mark.asyncio
async def test_content_parts_before_features_is_rejected(client, test_image):
    """A content_parts placeholder before the last features range is a 400."""
    data_url = encode_image_url(test_image, format="PNG")
    render_data = await _render_image_prompt(client, data_url, data_url)
    features = render_data["features"]

    gen_resp = await client.post(
        GEN_ENDPOINT,
        json={
            "token_ids": render_data["token_ids"],
            "features": {
                k: {"image": v["image"][1:]} for k, v in features.items() if v
            },
            "content_parts": [{"type": "image_url", "url": data_url}],
            "sampling_params": {"max_tokens": 1},
        },
    )
    assert gen_resp.status_code == 400
    assert "Failed to apply prompt replacement" in gen_resp.text
    assert "after the last features range" in gen_resp.text


@pytest.mark.asyncio
async def test_content_parts_streaming(client, test_image):
    """content_parts should work with streaming."""
    import json as json_mod

    data_url = encode_image_url(test_image, format="PNG")
    render_data = await _render_image_prompt(client, data_url)
    expanded = render_data["token_ids"]
    mm_placeholders = render_data["features"]["mm_placeholders"]
    unexpanded = _collapse_placeholders(expanded, mm_placeholders)

    async with client.stream(
        "POST",
        GEN_ENDPOINT,
        json={
            "token_ids": unexpanded,
            "content_parts": [{"type": "image_url", "url": data_url}],
            "sampling_params": {"max_tokens": 8, "temperature": 0.0},
            "stream": True,
            "return_token_ids": True,
        },
    ) as resp:
        resp.raise_for_status()
        chunks = []
        async for line in resp.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json_mod.loads(line[6:]))

    assert len(chunks) > 0
    prompt_metadata_chunks = [
        chunk for chunk in chunks if chunk.get("prompt_token_ids") is not None
    ]
    assert len(prompt_metadata_chunks) == 1
    assert prompt_metadata_chunks[0]["prompt_token_ids"] == expanded
    assert prompt_metadata_chunks[0]["mm_placeholders"] == mm_placeholders
    first = prompt_metadata_chunks[0]
    assert first["mm_hashes"] == render_data["features"]["mm_hashes"]
    assert all("mm_hashes" not in c for c in chunks if c is not first)


def _upload_processed_image(
    s3_store, render_data: dict, key_dir: str = "", **extra_tensors
) -> dict:
    """Store the rendered item's tensors with their batch dimension.

    Returns its `kwargs_refs` entry.
    """
    _, objects = s3_store
    item = decode_mm_kwargs_item(render_data["features"]["kwargs_data"]["image"][0])
    tensors = {}
    for name, elem in item.items():
        data = elem.data
        if isinstance(elem.field, MultiModalBatchedField):
            assert isinstance(data, torch.Tensor)
            data = data.unsqueeze(0)
        tensors[name] = data
    # A `None` value drops that tensor.
    tensors = {k: v for k, v in {**tensors, **extra_tensors}.items() if v is not None}
    payload = save_safetensors(tensors)
    key = f"{S3_PREFIX}/{key_dir}{uuid.uuid4().hex}/image-0"
    objects[f"{S3_BUCKET}/{key}"] = payload
    return {
        "uri": f"s3://{S3_BUCKET}/{key}",
        "format": "safetensors-v1",
        "nbytes": len(payload),
        "digest": hashlib.sha256(payload).hexdigest(),
    }


@pytest.mark.asyncio
async def test_kwargs_refs_generate_like_inline_kwargs_data(
    client, s3_store, test_image
):
    """Tensors loaded from S3 must generate the same tokens as inline ones.

    The referenced item gets a new `mm_hash` so the encoder runs on the
    loaded tensors instead of reusing the inline request's output. The
    requests run one at a time, so greedy outputs are comparable.
    """
    render_data = await _render_image_prompt(
        client, encode_image_url(test_image, format="PNG")
    )
    features = render_data["features"]
    reference = _upload_processed_image(s3_store, render_data)

    outputs = []
    for request_features in (
        features,
        {
            "mm_hashes": {"image": [f"ref-{uuid.uuid4().hex}"]},
            "mm_placeholders": features["mm_placeholders"],
            "kwargs_refs": {"image": [reference]},
        },
    ):
        gen_resp = await client.post(
            GEN_ENDPOINT,
            json={
                "token_ids": render_data["token_ids"],
                "features": request_features,
                "sampling_params": {"max_tokens": 10, "temperature": 0.0},
            },
        )
        gen_resp.raise_for_status()
        outputs.append(gen_resp.json()["choices"][0]["token_ids"])

    assert outputs[0]
    assert outputs[1] == outputs[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("change", "status", "error"),
    [
        ("uri", 400, "configured prefix"),
        ("request_size", 400, "in one request declare"),
        ("missing", 422, "Could not load multimodal tensor object"),
        ("conflicting_metadata", 400, "doesn't match its kwargs_refs object"),
        ("extra_tensor", 422, "not compatible with the served model"),
        ("zero_dim", 422, "not compatible with the served model"),
        ("short_rows", 422, "not compatible with the served model"),
        ("extra_rows", 422, "not compatible with the served model"),
        ("negative_grid", 422, "not compatible with the served model"),
        ("grid_only", 422, "not compatible with the served model"),
    ],
)
async def test_invalid_kwargs_refs_are_rejected(
    client, s3_store, test_image, change, status, error
):
    """Bad references fail with a client error and leave the server healthy."""
    render_data = await _render_image_prompt(
        client, encode_image_url(test_image, format="PNG")
    )
    item = decode_mm_kwargs_item(render_data["features"]["kwargs_data"]["image"][0])
    pixel_values = item["pixel_values"].data
    grid = item["image_grid_thw"].data
    assert isinstance(grid, torch.Tensor)
    grid = grid.unsqueeze(0)
    extra = {
        "extra_tensor": {"attention_mask": torch.ones(1, 4)},
        "zero_dim": {
            "pixel_values": torch.tensor(1, dtype=torch.uint8),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
        # Fewer or more patches than image_grid_thw describes.
        "short_rows": {"pixel_values": pixel_values[:-4]},
        "extra_rows": {"pixel_values": torch.cat([pixel_values, pixel_values[:4]])},
        # A negative size would slice from the end and still yield enough rows.
        "negative_grid": {
            "pixel_values": torch.cat([pixel_values, pixel_values]),
            "image_grid_thw": grid * torch.tensor([1, -1, 1]),
        },
        # Only the grid, nothing for the vision encoder.
        "grid_only": {"pixel_values": None},
    }.get(change, {})
    reference = _upload_processed_image(s3_store, render_data, **extra)
    if change == "uri":
        reference["uri"] = reference["uri"].replace(S3_PREFIX, "other", 1)
    elif change == "request_size":
        reference["nbytes"] = 1024**2 + 1
    elif change == "missing":
        s3_store[1].clear()

    features = render_data["features"]
    request_features = {
        "mm_hashes": features["mm_hashes"],
        "mm_placeholders": features["mm_placeholders"],
        "kwargs_refs": {"image": [reference]},
    }
    if change == "conflicting_metadata":
        # A stale grid in mm_metadata would undo the row-count check.
        grid_elem = item["image_grid_thw"]
        stale = replace(grid_elem, data=grid_elem.data * torch.tensor([1, 2, 1]))
        request_features["mm_metadata"] = {
            "image": [
                encode_mm_kwargs_item(MultiModalKwargsItem({"image_grid_thw": stale}))
            ]
        }
    gen_resp = await client.post(
        GEN_ENDPOINT,
        json={
            "token_ids": render_data["token_ids"],
            "features": request_features,
            "sampling_params": {"max_tokens": 1},
        },
    )

    assert gen_resp.status_code == status
    assert error in gen_resp.text
    assert (await client.get("/health")).status_code == 200


@pytest.mark.asyncio
async def test_kwargs_refs_are_read_concurrently(client, s3_store, test_image):
    """Two requests must be able to read their references at the same time.

    Each GET blocks until the other one arrives, so serialized reads time out.
    """
    render_data = await _render_image_prompt(
        client, encode_image_url(test_image, format="PNG")
    )
    features = render_data["features"]

    async def generate():
        reference = _upload_processed_image(s3_store, render_data, "concurrent/")
        return await client.post(
            GEN_ENDPOINT,
            json={
                "token_ids": render_data["token_ids"],
                "features": {
                    "mm_hashes": {"image": [f"ref-{uuid.uuid4().hex}"]},
                    "mm_placeholders": features["mm_placeholders"],
                    "kwargs_refs": {"image": [reference]},
                },
                "sampling_params": {"max_tokens": 1},
            },
        )

    for gen_resp in await asyncio.gather(generate(), generate()):
        gen_resp.raise_for_status()


@pytest.mark.asyncio
async def test_kwargs_refs_release_their_memory(client, s3_store, test_image):
    """Each request returns its reservation, streamed or not.

    The server's budget fits about three of these items, so a request that
    kept its reservation would make a later one wait forever.
    """
    render_data = await _render_image_prompt(
        client, encode_image_url(test_image, format="PNG")
    )
    features = render_data["features"]
    async with asyncio.timeout(120):
        for stream in [False, True] * 4:
            body = {
                "token_ids": render_data["token_ids"],
                "features": {
                    "mm_hashes": {"image": [f"ref-{uuid.uuid4().hex}"]},
                    "mm_placeholders": features["mm_placeholders"],
                    "kwargs_refs": {
                        "image": [_upload_processed_image(s3_store, render_data)]
                    },
                },
                "sampling_params": {"max_tokens": 2},
                "stream": stream,
            }
            async with client.stream("POST", GEN_ENDPOINT, json=body) as resp:
                assert resp.status_code == 200
                text = "".join([chunk async for chunk in resp.aiter_text()])
            assert "token_ids" in text


@pytest.mark.asyncio
async def test_bad_reference_is_rejected_before_any_download(
    client, s3_store, test_image
):
    """A disallowed reference fails the request before the valid one is read."""
    data_url = encode_image_url(test_image, format="PNG")
    render_data = await _render_image_prompt(client, data_url, data_url)
    valid = _upload_processed_image(s3_store, render_data)
    outside = {**valid, "uri": valid["uri"].replace(S3_PREFIX, "other", 1)}
    features = render_data["features"]

    gen_resp = await client.post(
        GEN_ENDPOINT,
        json={
            "token_ids": render_data["token_ids"],
            "features": {
                "mm_hashes": features["mm_hashes"],
                "mm_placeholders": features["mm_placeholders"],
                "kwargs_refs": {"image": [valid, outside]},
            },
            "sampling_params": {"max_tokens": 1},
        },
    )

    assert gen_resp.status_code == 400
    assert "configured prefix" in gen_resp.text
    assert valid["uri"].removeprefix("s3://") not in GET_PATHS
