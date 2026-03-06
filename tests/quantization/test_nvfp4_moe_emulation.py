# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any

import pytest

from tests.utils import RemoteOpenAIServer


def can_initialize(
    model: str,
    hf_overrides: dict[str, Any] | None = None,
    extra_args: list[str] | None = None,
):
    # Server arguments
    extra_args = extra_args if extra_args is not None else []
    server_args = [
        "--max-model-len",
        "2048",
        "--max-num-batched-tokens",
        "256",
        "--load-format",
        "dummy",
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        json.dumps({"image": 0}),
        *extra_args,
    ]

    # Launch server and make a simple request
    with RemoteOpenAIServer(
        model,
        server_args,
        max_wait_seconds=1500,  # Due to FlashInfer compile
        override_hf_configs=hf_overrides,
    ) as server:
        client = server.get_client()
        # Make a simple request to verify the server works
        completion = client.completions.create(
            model=model,
            prompt=["1 2 3 4 5 6 7"],
            temperature=0,
            max_tokens=2,
        )
        print(completion)
        assert completion.choices[0].text is not None


HF_OVERRIDE_MM = {
    "text_config": {"num_layers": 4, "num_hidden_layers": 4},
}


def test_llama4_nvfp4_moe_emulation_modelopt(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "nvidia/Llama-4-Scout-17B-16E-Instruct-FP4",
        hf_overrides=HF_OVERRIDE_MM,
        extra_args=["--moe-backend=emulation", "--gpu-memory-utilization", "0.8"],
    )


def test_llama4_nvfp4_moe_emulation_compressed_tensors(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "RedHatAI/Llama-4-Scout-17B-16E-Instruct-NVFP4",
        hf_overrides=HF_OVERRIDE_MM,
        extra_args=["--moe-backend=emulation", "--gpu-memory-utilization", "0.8"],
    )
