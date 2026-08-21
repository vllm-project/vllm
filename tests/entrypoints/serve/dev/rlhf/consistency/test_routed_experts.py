# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-experts capture + replay consistency (issue #45585, section R3).

Three tests cover the replay invariants end-to-end; each test exercises
two or three scenarios so the matrix below is covered with few cases:

  1. ``test_capture_shape_and_range``: capture shape/range and
     token_ids/completion alignment on both endpoints, with and without
     ``return_token_ids``.
  2. ``test_replay_identity``: deterministic reruns (prefix-cache slot
     recovery) and prefix identity when the prompt is extended.
  3. ``test_batch_isolation``: batch composition / concurrent load must
     not perturb routing; varied-length requests all stay valid.

Every test runs on every server configuration below — five startups
covering the V1/V2 runner, eager/CUDA-graph and modular/monolithic
kernel matrix, reusing the shared ``server`` harness from the rlhf
conftest:

    v1-eager              V1 runner, eager (Triton modular kernel)
    v1-graph              V1 runner, -O2 CUDA graphs (default)
    v1-monolithic-eager   V1 runner, flashinfer_trtllm monolithic kernel
    mrv2-eager            MRV2 runner, eager
    mrv2-graph            MRV2 runner, -O2 CUDA graphs

The monolithic configuration is skipped unless flashinfer's TRTLLM MoE
kernels are available on a Blackwell (SM100) GPU.
"""

import io
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import numpy as np
import openai
import pybase64 as base64
import pytest

from vllm.utils.network_utils import get_open_port

from ..conftest import server as _server

# tiny-mixtral config: 8 local experts, top-2 routing, 2 hidden layers.
# The published config has sliding_window=4096, which produces
# SlidingWindowSpec kv-cache groups; the routed-experts slot buffer
# requires a FullAttentionSpec group, so we override sliding_window=null.
MOE_MODEL_NAME = "TitanML/tiny-mixtral"
MOE_NUM_LOCAL_EXPERTS = 8
MOE_NUM_EXPERTS_PER_TOK = 2
MOE_NUM_HIDDEN_LAYERS = 2

# ~10 tokens per sentence; the long prompts force multi-step chunked
# prefill, exercising the slot-indexed routing buffer across steps.
_PROMPT_SEED = "The quick brown fox jumps over the lazy dog."

# (runner_type, enforce_eager, moe_backend)
SERVER_CONFIGS = [
    pytest.param(("v1", True, None), id="v1-eager"),
    pytest.param(("v1", False, None), id="v1-graph"),
    pytest.param(("v1", True, "flashinfer_trtllm"), id="v1-monolithic-eager"),
    pytest.param(("mrv2", True, None), id="mrv2-eager"),
    pytest.param(("mrv2", False, None), id="mrv2-graph"),
]


@dataclass
class Generation:
    """One completion with its decoded routing payload."""

    token_ids: list[int] | None
    routed_experts: np.ndarray
    prompt_tokens: int
    completion_tokens: int

    @property
    def num_forwarded_tokens(self) -> int:
        """Tokens that went through a forward pass.

        The last sampled token is never forwarded, so this is one less
        than the full sequence length.
        """
        return self.prompt_tokens + self.completion_tokens - 1


def decode_routed_experts(encoded: str | None) -> np.ndarray:
    """Decode the base64-encoded ``.npy`` routed-experts payload."""
    assert encoded is not None
    return np.load(io.BytesIO(base64.b64decode(encoded)))


def assert_valid_routed_experts(
    routed_experts: np.ndarray,
    *,
    num_layers: int = MOE_NUM_HIDDEN_LAYERS,
    num_experts_per_tok: int = MOE_NUM_EXPERTS_PER_TOK,
    num_experts: int = MOE_NUM_LOCAL_EXPERTS,
) -> None:
    """Assert shape ``(n, num_layers, top_k)`` and valid expert IDs."""
    assert routed_experts.ndim == 3
    _, layers, topk = routed_experts.shape
    assert layers == num_layers
    assert topk == num_experts_per_tok
    assert (routed_experts >= 0).all()
    assert (routed_experts < num_experts).all()


def _extra_body(*, return_token_ids: bool, ignore_eos: bool) -> dict[str, Any]:
    """Assemble the vLLM-specific request fields."""
    body: dict[str, Any] = {}
    if return_token_ids:
        body["return_token_ids"] = True
    if ignore_eos:
        body["ignore_eos"] = True
    return body


def _generate(
    url: str,
    prompt: str,
    *,
    max_tokens: int = 8,
    return_token_ids: bool = True,
    ignore_eos: bool = False,
) -> Generation:
    """Run one completion and decode its routed-experts payload."""
    response = openai.OpenAI(
        base_url=f"{url}/v1", api_key="EMPTY", max_retries=0
    ).completions.create(
        model=MOE_MODEL_NAME,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0,
        extra_body=_extra_body(
            return_token_ids=return_token_ids, ignore_eos=ignore_eos
        ),
    )
    return _parse(response.model_dump())


def _generate_chat(
    url: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 8,
    return_token_ids: bool = True,
    ignore_eos: bool = False,
) -> Generation:
    """Run one chat completion and decode its routed-experts payload."""
    response = openai.OpenAI(
        base_url=f"{url}/v1", api_key="EMPTY", max_retries=0
    ).chat.completions.create(
        model=MOE_MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
        extra_body=_extra_body(
            return_token_ids=return_token_ids, ignore_eos=ignore_eos
        ),
    )
    return _parse(response.model_dump())


def _parse(data: dict[str, Any]) -> Generation:
    """Decode one completions/chat response into a validated Generation."""
    choice = data["choices"][0]
    generation = Generation(
        token_ids=choice["token_ids"],
        routed_experts=decode_routed_experts(choice["routed_experts"]),
        prompt_tokens=data["usage"]["prompt_tokens"],
        completion_tokens=data["usage"]["completion_tokens"],
    )
    # Every response must satisfy the documented invariants: shape
    # (num_forwarded_tokens, num_layers, top_k), valid expert IDs, and
    # one row per forwarded token.
    assert_valid_routed_experts(generation.routed_experts)
    num_rows, _, _ = generation.routed_experts.shape
    assert num_rows == generation.num_forwarded_tokens
    if generation.token_ids is not None:
        # token_ids covers the generated tokens only (the prompt lives in
        # the separate prompt_token_ids field unless echo=True); its
        # length must match the completion count.
        assert len(generation.token_ids) == generation.completion_tokens
    return generation


def _assert_identical(a: Generation, b: Generation) -> None:
    """Assert two generations have identical tokens and routing."""
    assert a.token_ids == b.token_ids
    assert np.array_equal(a.routed_experts, b.routed_experts)


def _supports_monolithic_trtllm() -> bool:
    """Mirror TrtLlmBf16Experts._supports_current_device.

    The flashinfer_trtllm MoE kernels only run on Blackwell (SM100)
    GPUs; elsewhere the server fails to start instead of falling back,
    so the monolithic configuration must be skipped up front.
    """
    from vllm.platforms import current_platform
    from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe

    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and has_flashinfer_trtllm_fused_moe()
    )


@pytest.fixture(scope="module", params=SERVER_CONFIGS)
def server_url(request):
    """Module-scoped server URL, one startup per scenario configuration."""
    runner_type, enforce_eager, moe_backend = request.param
    if moe_backend is not None and not _supports_monolithic_trtllm():
        pytest.skip(
            "flashinfer_trtllm MoE kernels require flashinfer on Blackwell (SM100) GPUs"
        )
    extra_args = [
        "--enable-return-routed-experts",
        "--hf-overrides",
        '{"sliding_window": null}',
    ]
    if enforce_eager:
        # The harness no longer forces eager; opt in per scenario so the
        # graph configurations keep the default -O2 CUDA graphs.
        extra_args += ["--enforce-eager"]
    if moe_backend is not None:
        extra_args += ["--moe-backend", moe_backend]
    env_vars = {"VLLM_USE_V2_MODEL_RUNNER": "0" if runner_type == "v1" else "1"}
    with (
        patch.dict(os.environ, env_vars),
        _server(MOE_MODEL_NAME, extra_args=extra_args, port=get_open_port()) as url,
    ):
        yield url


def test_capture_shape_and_range(server_url):
    """Capture correctness on both endpoints, with and without token_ids."""
    generation = _generate(server_url, "Hello, world", max_tokens=10)
    assert generation.token_ids is not None

    chat = _generate_chat(
        server_url, [{"role": "user", "content": "Hello, world"}], max_tokens=10
    )
    assert chat.token_ids is not None

    no_ids = _generate(server_url, "Hello, world", max_tokens=4, return_token_ids=False)
    assert no_ids.token_ids is None


def test_replay_identity(server_url):
    """Reruns and extended prompts reproduce routing exactly."""
    prompt = f"{_PROMPT_SEED} Count to three."
    first = _generate(server_url, prompt, max_tokens=6, ignore_eos=True)
    # The second run prefix-cache-hits the first run's blocks, so the
    # comparison also covers slot-buffer recovery against live capture.
    second = _generate(server_url, prompt, max_tokens=6, ignore_eos=True)
    _assert_identical(second, first)

    prefix = (_PROMPT_SEED + " ") * 15
    pref = _generate(server_url, prefix, max_tokens=4, ignore_eos=True)
    extended = _generate(
        server_url, prefix + (_PROMPT_SEED + " ") * 4, max_tokens=4, ignore_eos=True
    )
    # Routing is causal: the extended request's rows for the shared
    # prompt prefix tokens must equal the prefix request's rows.
    assert np.array_equal(
        extended.routed_experts[: pref.prompt_tokens],
        pref.routed_experts[: pref.prompt_tokens],
    )


def test_batch_isolation(server_url):
    """Concurrent load must not perturb routing or response shape."""
    prompt_a = f"{_PROMPT_SEED} Tell me a short story."
    isolated = _generate(server_url, prompt_a, max_tokens=6, ignore_eos=True)

    jobs = [
        (prompt_a, 6),
        ((_PROMPT_SEED + " ") * 3, 4),
        ("A.", 4),
        ("List three animals.", 4),
        ("List three animals.", 4),
    ]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        generations = list(
            pool.map(
                lambda job: _generate(
                    server_url, job[0], max_tokens=job[1], ignore_eos=True
                ),
                jobs,
            )
        )

    # Prompt A's routing in the concurrent batch matches its isolated run.
    _assert_identical(generations[0], isolated)
    # Two identical prompts interleaved in one batch must agree exactly.
    _assert_identical(generations[3], generations[4])
