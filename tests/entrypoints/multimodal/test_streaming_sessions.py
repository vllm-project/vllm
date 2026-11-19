# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the public session API with real weights past its retention window."""

import io
import os

import httpx
import pytest
from PIL import Image
from transformers import AutoConfig

from tests.utils import RemoteOpenAIServer

MODEL = os.environ.get("VLLM_TEST_STREAMING_MODEL", "Qwen/Qwen3-VL-2B-Instruct")


@pytest.mark.parametrize(
    ("max_tokens", "labels"),
    [(1, ["A", "B"]), (16, ["  red  ", "  blue  "])],
)
def test_bounded_image_session_preserves_constrained_replies(capfd, max_tokens, labels):
    model_config = AutoConfig.from_pretrained(MODEL)
    position_limit = model_config.get_text_config().max_position_embeddings
    args = [
        "--enable-streaming",
        "--enforce-eager",
        "--no-async-scheduling",
        "--api-server-count",
        "1",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "1",
        "--limit-mm-per-prompt",
        '{"image": 4}',
        "--enable-prefix-caching",
        "--served-model-name",
        MODEL,
    ]
    frames = []
    for color in ("red", "blue"):
        with Image.new("RGB", (448, 448), color) as image:
            encoded = io.BytesIO()
            image.save(encoded, "PNG")
            frames.append(encoded.getvalue())
    with (
        RemoteOpenAIServer(
            MODEL, args, env_dict={"VLLM_LOGGING_LEVEL": "INFO"}
        ) as server,
        httpx.Client(base_url=server.url_root, timeout=150) as client,
    ):
        response = client.post(
            "/v1/streaming/sessions",
            json={
                "system_prompt": (
                    "Identify the dominant color of the most recent image. "
                    f"Reply exactly {labels[0]!r} for red and {labels[1]!r} for blue."
                ),
                "retention": {
                    "max_video_segments": 2,
                    "max_session_tokens": 1024,
                    # Replay while this short test is running, with room for
                    # the full 1024-token retained prompt and its reply.
                    "reprefill_threshold": 1100 / position_limit,
                },
                "sampling": {
                    "temperature": 0,
                    "repetition_penalty": 1,
                    "max_tokens": max_tokens,
                    "structured_outputs": {"choice": labels},
                },
            },
        )
        response.raise_for_status()
        session_url = "/v1/streaming/sessions/" + response.json()["session_id"]
        try:
            for index in range(64):
                response = client.post(
                    session_url + "/frame", content=frames[index % 2]
                )
                response.raise_for_status()
                frame = response.json()
                assert frame["frame_index"] == index
                assert 0 < frame["token_count"] <= max_tokens
                assert frame["text"] == labels[index % 2]
                assert frame["finish_reason"] == (
                    "length" if max_tokens == 1 else "stop"
                )
        finally:
            response = client.delete(session_url)
        response.raise_for_status()
        assert response.json()["frames"] == 64
        # The ordinary endpoint remains usable after a bounded session releases KV.
        with server.get_client() as openai:
            reply = openai.chat.completions.create(
                model=MODEL,
                temperature=0,
                max_tokens=4,
                messages=[{"role": "user", "content": "Say hello."}],
            )
            assert reply.choices[0].message.content
    captured = capfd.readouterr()
    assert "Consolidating streaming session " in captured.out + captured.err
