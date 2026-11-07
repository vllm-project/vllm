# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-GPU filesystem-tiering restart tests.

The first server computes one visual encoding and writes it to the filesystem
tier. The second server starts with an empty CPU tier, recovers the filesystem
index, and must promote the same encoding through the existing CPU-to-GPU load
path, including concurrent requests for the same image.

Run on one CUDA GPU with::

    .venv/bin/python -m pytest -q -s \
      tests/v1/ec_connector/integration/one_card/test_filesystem_tiering.py
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
import requests
from PIL import Image

from tests.utils import RemoteOpenAIServer
from vllm.multimodal.utils import encode_image_url
from vllm.platforms import current_platform

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-VL-8B-Instruct")
_ITEM_UUID = "ec-tiering-one-card-image"


def _ec_config(cache_root: Path) -> str:
    return json.dumps(
        {
            "ec_connector": "ECTieringConnector",
            "ec_role": "ec_both",
            "ec_connector_extra_config": {
                "ec_cpu_bytes": 512 << 20,
                "secondary_tiers": [
                    {
                        "type": "fs",
                        "root_dir": str(cache_root),
                        "capacity_bytes": 10 << 30,
                        "n_read_threads": 4,
                        "n_write_threads": 2,
                        "backpressure": {"enabled": False},
                    }
                ],
            },
        }
    )


def _server_args(cache_root: Path, max_num_seqs: int = 1) -> list[str]:
    return [
        "--host",
        "127.0.0.1",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        "0.8",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--ec-transfer-config",
        _ec_config(cache_root),
    ]


def _request_body() -> dict:
    with Image.open(Path(__file__).parents[1] / "hato.jpg") as source:
        image = source.convert("RGB")
    image.thumbnail((224, 224))
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_url(image)},
                        "uuid": _ITEM_UUID,
                    },
                    {
                        "type": "text",
                        "text": "Describe this image briefly.",
                    },
                ],
            }
        ],
        "max_tokens": 16,
        "temperature": 0,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _complete(server: RemoteOpenAIServer, body: dict) -> str:
    response = requests.post(
        server.url_for("v1/chat/completions"),
        json=body,
        timeout=600,
    )
    result = response.json()
    assert response.status_code == 200, result
    content = result["choices"][0]["message"]["content"]
    assert content, result
    return content


def _wait_for_entries(cache_root: Path, timeout: float = 60) -> list[Path]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        entries = list(cache_root.rglob("*.ec"))
        if entries and all(entry.stat().st_size > 0 for entry in entries):
            return entries
        time.sleep(0.1)
    raise AssertionError("no complete .ec entry was published before timeout")


@pytest.mark.parametrize("request_count", [1, 2], ids=["single", "concurrent"])
def test_filesystem_entry_survives_restart_and_is_promoted(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
    request_count: int,
) -> None:
    if not current_platform.is_cuda() or current_platform.device_count() < 1:
        pytest.skip("Requires one CUDA GPU")

    cache_root = tmp_path / "encoder-cache"
    common = _server_args(cache_root)
    env = {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_LOGGING_LEVEL": "DEBUG",
    }
    body = _request_body()

    with RemoteOpenAIServer(
        MODEL,
        common,
        env_dict=env,
        max_wait_seconds=1200,
    ) as first_server:
        first_output = _complete(first_server, body)
        entries = _wait_for_entries(cache_root)

    first_logs = capfd.readouterr()
    assert entries
    first_log_text = (first_logs.out + first_logs.err).lower()
    assert "encoder-cache filesystem store failed" not in first_log_text
    assert "failed to submit encoder-cache filesystem store" not in first_log_text

    with RemoteOpenAIServer(
        MODEL,
        _server_args(cache_root, max_num_seqs=request_count),
        env_dict=env,
        max_wait_seconds=1200,
    ) as second_server:
        if request_count == 1:
            second_outputs = [_complete(second_server, body)]
        else:
            barrier = Barrier(request_count)

            def complete_together() -> str:
                barrier.wait(timeout=30)
                return _complete(second_server, body)

            with ThreadPoolExecutor(max_workers=request_count) as executor:
                futures = [
                    executor.submit(complete_together) for _ in range(request_count)
                ]
                second_outputs = [future.result(timeout=660) for future in futures]

    second_logs = capfd.readouterr()
    combined_logs = second_logs.out + second_logs.err
    log_tail = combined_logs[-8000:]
    assert second_outputs == [first_output] * request_count
    assert "EC consumer: mm_hash=" in combined_logs, log_tail
    assert "unpinned after load" in combined_logs, log_tail
    assert "encoder-cache filesystem load failed" not in combined_logs.lower()
