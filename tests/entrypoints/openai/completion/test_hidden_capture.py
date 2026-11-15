# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import json
import os

import httpx
import numpy as np
import pybase64 as base64
import pytest

from tests.utils import RemoteOpenAIServer

MODEL_NAME = os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def capture_server():
    with RemoteOpenAIServer(
        MODEL_NAME,
        [
            "--max-model-len",
            "256",
            "--max-num-seqs",
            "2",
            "--gpu-memory-utilization",
            "0.35",
            "--enforce-eager",
            "--no-async-scheduling",
            "--enable-return-hidden-states",
            "--enable-scale-out",
        ],
        env_dict={"VLLM_USE_V2_MODEL_RUNNER": "1"},
    ) as server:
        yield server


@pytest.mark.parametrize(
    "endpoint", ["v1/completions", "v1/chat/completions", "inference/v1/generate"]
)
@pytest.mark.parametrize("stream", [False, True])
def test_hidden_inline_http_roundtrip(capture_server, endpoint, stream):
    common = dict(
        temperature=0.8,
        seed=42,
        top_k=128,
        top_p=1,
        max_tokens=4,
        min_tokens=4,
        ignore_eos=True,
        n=2,
    )
    if endpoint.endswith("generate"):
        body = dict(
            model=MODEL_NAME,
            token_ids=[100, 200],
            stream=stream,
            sampling_params={**common, "aux_outputs": ["hidden_states"]},
        )
    else:
        prompt = (
            {"messages": [{"role": "user", "content": "Say hello."}]}
            if "chat" in endpoint
            else {"prompt": [100, 200]}
        )
        body = dict(
            model=MODEL_NAME,
            stream=stream,
            aux_outputs=["hidden_states"],
            **prompt,
            **common,
        )
    with httpx.Client(timeout=90) as client:
        response = client.post(capture_server.url_for(endpoint), json=body)
        assert response.status_code == 200, response.text
        if stream:
            messages = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
        else:
            messages = [response.json()]
        final = {}
        for message in messages:
            for choice in message["choices"]:
                if choice.get("finish_reason") is None:
                    assert "hidden_states" not in choice
                else:
                    final[choice["index"]] = choice
        assert set(final) == {0, 1}
        for choice in final.values():
            rows = np.load(
                io.BytesIO(base64.b64decode(choice["hidden_states"])),
                allow_pickle=False,
            )
            assert rows.shape[0] == 4
            assert choice["hidden_states_dtype"] == "bfloat16"
            assert rows.dtype == np.uint16
            assert choice["hidden_states_start"] >= 1
            assert choice["weight_versions"]
        if not stream:
            if endpoint.endswith("generate"):
                body["sampling_params"].pop("aux_outputs")
            else:
                body.pop("aux_outputs")
            plain = client.post(capture_server.url_for(endpoint), json=body)
            assert plain.status_code == 200, plain.text
            for choice in plain.json()["choices"]:
                assert "hidden_states" not in choice
                assert "hidden_states_start" not in choice
                assert "hidden_states_dtype" not in choice


def test_hidden_http_admission_and_metadata(capture_server):
    with httpx.Client(timeout=90) as client:
        metadata = client.get(
            capture_server.url_for("inference/v1/hidden_states_metadata")
        )
        assert metadata.status_code == 200, metadata.text
        assert metadata.json()["codec"] == "raw"
        assert metadata.json()["vocab_size"] > 128
        assert "/data/" not in metadata.text
        for extra in (
            {"presence_penalty": 1},
            {"logit_bias": {"1": 1}},
            {"hidden_states_end": 300},
        ):
            response = client.post(
                capture_server.url_for("v1/completions"),
                json={
                    "model": MODEL_NAME,
                    "prompt": [100, 200],
                    "max_tokens": 2,
                    "aux_outputs": ["hidden_states"],
                    **extra,
                },
            )
            assert response.status_code == 400, response.text
