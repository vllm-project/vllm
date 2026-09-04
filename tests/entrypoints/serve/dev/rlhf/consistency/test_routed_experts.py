# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for routed-experts capture (issue #45585, section R3)."""

import io
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import patch

import httpx
import numpy as np
import openai
import pybase64 as base64
import pytest
import torch

from tests.utils import VLLM_PATH
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

from ..conftest import SERVED_MODEL_NAME
from ..conftest import server as _server

# tiny-mixtral: 8 experts / top-2 / 2 layers; sliding_window=null keeps a
# FullAttention KV group, which the routed-experts slot buffer requires.
MOE_MODEL = os.environ.get("VLLM_TEST_MODEL", "TitanML/tiny-mixtral")


@dataclass(frozen=True)
class RoutingShape:
    """Expected routed_experts geometry."""

    num_layers: int
    num_experts_per_tok: int
    num_experts: int


ROUTING_SHAPE = RoutingShape(num_layers=2, num_experts_per_tok=2, num_experts=8)


@dataclass(frozen=True)
class ModelSpec:
    """One model-matrix entry: checkpoint and its reduction config."""

    model: str
    overrides: dict[str, int]


# Reduced geometry shared by both families; only the expert-count field
# name differs per family.
_REDUCED_BASE = {
    "num_hidden_layers": 4,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_experts_per_tok": 2,
}
QWEN3_5_OVERRIDES = {**_REDUCED_BASE, "num_experts": 8}
# V4-Flash needs a Blackwell GPU; skipped elsewhere below.
DEEPSEEK_V4_OVERRIDES = {**_REDUCED_BASE, "n_routed_experts": 8}

REDUCED_SHAPE = RoutingShape(num_layers=4, num_experts_per_tok=2, num_experts=8)

MODEL_SPECS = [
    pytest.param(
        ModelSpec("codecho/Qwen3.5-35B-A3B-text-only", QWEN3_5_OVERRIDES),
        id="qwen3.5-moe",
    ),
    pytest.param(
        ModelSpec("deepseek-ai/DeepSeek-V4-Flash", DEEPSEEK_V4_OVERRIDES),
        id="deepseek-v4-flash",
        marks=pytest.mark.skipif(
            not (
                current_platform.is_cuda()
                and (
                    current_platform.is_device_capability_family(100)
                    or current_platform.is_device_capability_family(120)
                )
            ),
            reason="DeepSeek-V4-Flash requires a Blackwell GPU "
            "(sm_10x/sm_12x, MHC/DeepGEMM)",
        ),
    ),
]


def assert_valid_routed_experts(encoded: str | None, shape: RoutingShape) -> None:
    """Decode the payload and assert its geometry."""
    assert encoded is not None
    routed_experts = np.load(io.BytesIO(base64.b64decode(encoded)))
    assert routed_experts.ndim == 3
    num_tokens, layers, topk = routed_experts.shape
    assert num_tokens > 0
    assert layers == shape.num_layers
    assert topk == shape.num_experts_per_tok
    assert (routed_experts >= 0).all()
    assert (routed_experts < shape.num_experts).all()


@contextmanager
def _launch(
    model: str,
    extra_args: list[str],
    use_v2: bool,
    env_vars: dict[str, str] | None = None,
):
    env = {"VLLM_USE_V2_MODEL_RUNNER": "1" if use_v2 else "0"}
    if env_vars:
        env.update(env_vars)
    with (
        patch.dict(os.environ, env),
        _server(model, extra_args=extra_args, port=get_open_port()) as url,
    ):
        yield url


@pytest.fixture(scope="module", params=[False, True], ids=["mrv1", "mrv2"])
def use_v2(request):
    return request.param


# Function scope: each server is torn down before the next one starts.
@pytest.fixture()
def scale_out_server(use_v2):
    if torch.cuda.device_count() < 4:
        pytest.skip("TP2+DP2 scenario needs 4 GPUs")
    extra_args = [
        "--enable-return-routed-experts",
        "--hf-overrides",
        '{"sliding_window": null}',
        "--tensor-parallel-size",
        "2",
        "--data-parallel-size",
        "2",
    ]
    # The token-in-token-out endpoints are opt-in (PR #54579).
    env_vars = {"VLLM_ENABLE_SCALE_OUT_ENDPOINTS": "1"}
    with _launch(MOE_MODEL, extra_args, use_v2, env_vars) as url:
        yield url


@pytest.fixture(params=MODEL_SPECS)
def models_server(request):
    spec = request.param
    extra_args = [
        "--enable-return-routed-experts",
        "--load-format",
        "dummy",
        "--hf-overrides",
        json.dumps(spec.overrides),
        # Chat requests need a template; pin the repo example so the test
        # does not depend on the remote tokenizer_config.json shipping one.
        "--chat-template",
        str(VLLM_PATH / "examples/template_chatml.jinja"),
    ]
    with _launch(spec.model, extra_args, True) as url:
        yield url, REDUCED_SHAPE


class TestRoutedExperts:
    # Token-in-token-out frontend (/inference/v1/generate) under TP2+DP2.
    def test_generate_routed_experts_parallel(self, scale_out_server):
        payload = {
            "model": SERVED_MODEL_NAME,
            "token_ids": [1, 2, 3],
            "sampling_params": {"max_tokens": 10, "temperature": 0.0},
            "stream": False,
        }
        response = httpx.post(
            f"{scale_out_server}/inference/v1/generate",
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]

        assert choice["token_ids"] is not None
        assert_valid_routed_experts(choice["routed_experts"], ROUTING_SHAPE)

    # OpenAI frontends (/v1/completions and /v1/chat/completions) across
    # the reduced model matrix (Qwen3.5-MoE / DeepSeek-V4-Flash, dummy
    # weights). Both frontends run against every model so the coverage
    # survives the Blackwell skip: on H100s Qwen still exercises both.
    def test_openai_frontends_routed_experts_models(self, models_server):
        url, shape = models_server
        client = openai.OpenAI(
            base_url=f"{url}/v1", api_key="EMPTY", max_retries=0
        )
        responses = {
            "completions": client.completions.create(
                model=SERVED_MODEL_NAME,
                prompt="Hello, world",
                max_tokens=10,
                temperature=0,
                extra_body={"return_token_ids": True},
            ),
            "chat": client.chat.completions.create(
                model=SERVED_MODEL_NAME,
                messages=[{"role": "user", "content": "Hello, world"}],
                max_tokens=10,
                temperature=0,
                extra_body={"return_token_ids": True},
            ),
        }
        for frontend, response in responses.items():
            choice = response.model_dump()["choices"][0]
            assert choice["token_ids"] is not None, frontend
            assert_valid_routed_experts(choice["routed_experts"], shape)
