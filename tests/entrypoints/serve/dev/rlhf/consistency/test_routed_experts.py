# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-experts capture e2e scenarios (issue #45585, section R3).

Two tests consolidate the routed-experts e2e coverage that used to live
in ``tests/entrypoints/openai/test_return_routed_experts.py`` (OpenAI
frontend) and
``tests/entrypoints/scale_out/token_in_token_out/test_return_routed_experts.py``
(token-in-token-out frontend, since deleted):

  1. ``test_generate_routed_experts_parallel``: the
     ``/inference/v1/generate`` frontend with TP=2 + DP=2 together on
     the tiny MoE model.
  2. ``test_completions_routed_experts_models``: ``/v1/completions``,
     parametrized over layer-reduced Qwen3.5-MoE and DeepSeek-V4-Flash
     checkpoints with dummy weights (pattern from PR #49555).

The parallel scenario runs on both model runners (MRV1 / MRV2); the
model matrix runs on MRV2 only. Every server runs eagerly (the shared
harness passes ``--enforce-eager``), and validates the decoded payload:
shape ``(num_tokens, num_layers, num_experts_per_tok)`` with valid
expert IDs.
"""

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

from vllm.utils.network_utils import get_open_port

from ..conftest import SERVED_MODEL_NAME
from ..conftest import server as _server

# MoE model for the DP/TP scenarios; VLLM_TEST_MODEL overrides it.
# tiny-mixtral: 8 local experts, top-2 routing, 2 hidden layers. The
# published config has sliding_window=4096, which produces
# SlidingWindowSpec kv-cache groups; the routed-experts slot buffer
# requires a FullAttentionSpec group, so we override sliding_window=null.
MOE_MODEL = os.environ.get("VLLM_TEST_MODEL", "TitanML/tiny-mixtral")


@dataclass(frozen=True)
class RoutingShape:
    """Expected routed_experts geometry for a (reduced) model."""

    num_layers: int
    num_experts_per_tok: int
    num_experts: int


# Expected geometry shared by every scenario: both the tiny MoE model and
# the layer-reduced model matrix use 2 layers / top-2 / 8 experts.
ROUTING_SHAPE = RoutingShape(num_layers=2, num_experts_per_tok=2, num_experts=8)

# Shared tiny geometry for the model matrix (dummy weights, so only the
# HF config is fetched). Field names are generation-specific; adjust per
# family once verified on a real machine.
REDUCED_OVERRIDES = {
    "num_hidden_layers": 2,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
}

MODEL_NAMES = [
    pytest.param("codecho/Qwen3.5-35B-A3B-text-only", id="qwen3.5-moe"),
    pytest.param("deepseek-ai/DeepSeek-V4-Flash", id="deepseek-v4-flash"),
]


def assert_valid_routed_experts(encoded: str | None, shape: RoutingShape) -> None:
    """Decode and validate the base64 ``.npy`` routed-experts payload."""
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
def _launch(model: str, extra_args: list[str], use_v2: bool):
    """Start one server for the given runner; yield its base URL."""
    env_vars = {"VLLM_USE_V2_MODEL_RUNNER": "1" if use_v2 else "0"}
    with (
        patch.dict(os.environ, env_vars),
        _server(model, extra_args=extra_args, port=get_open_port()) as url,
    ):
        yield url


@pytest.fixture(scope="module", params=[False, True], ids=["mrv1", "mrv2"])
def use_v2(request):
    return request.param


@pytest.fixture(scope="module")
def scale_out_server(use_v2):
    """TITO server with TP=2 + DP=2 (needs 4 GPUs)."""
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
    with _launch(MOE_MODEL, extra_args, use_v2) as url:
        yield url


@pytest.fixture(scope="module", params=MODEL_NAMES)
def models_server(request):
    """Completions server per layer-reduced model, MRV2 only (dummy weights)."""
    extra_args = [
        "--enable-return-routed-experts",
        "--load-format",
        "dummy",
        "--hf-overrides",
        json.dumps(REDUCED_OVERRIDES),
    ]
    with _launch(request.param, extra_args, True) as url:
        yield url


class TestRoutedExperts:
    """End-to-end routed-experts capture scenarios."""

    def test_generate_routed_experts_parallel(self, scale_out_server):
        """/inference/v1/generate returns routed_experts under TP/DP."""
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

    def test_completions_routed_experts_models(self, models_server):
        """/v1/completions returns routed_experts for the model matrix."""
        response = openai.OpenAI(
            base_url=f"{models_server}/v1", api_key="EMPTY", max_retries=0
        ).completions.create(
            model=SERVED_MODEL_NAME,
            prompt="Hello, world",
            max_tokens=10,
            temperature=0,
            extra_body={"return_token_ids": True},
        )
        choice = response.model_dump()["choices"][0]

        assert choice["token_ids"] is not None
        assert_valid_routed_experts(choice["routed_experts"], ROUTING_SHAPE)
