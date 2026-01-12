# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ATOM modeling backend (`model_impl="atom"`).

These are intentionally split into:
- lightweight "registry/selection" tests that do not require the external
  `atom` Python package to be installed, and
- optional runtime smoke tests that require ROCm + `atom`.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

import pytest  # type: ignore[import-not-found]

from vllm.platforms import current_platform

from ..conftest import VllmRunner
from ..utils import multi_gpu_test
from .utils import check_logprobs_close

# NOTE: This model is very large and is expected to be run only in
# dedicated ROCm+ATOM environments (hence most tests here are optional).
ATOM_TEST_MODEL_ID = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

# Follow the common test pattern in this repo:
# - use module-level `pytestmark` for simple platform gating
# - use runtime `pytest.skip(...)` for dynamic conditions / error-based skipping
_ATOM_PKG_AVAILABLE = importlib.util.find_spec("atom") is not None
pytestmark = pytest.mark.skipif(
    (not current_platform.is_rocm()) or (not _ATOM_PKG_AVAILABLE),
    reason=(
        "ATOM backend tests require ROCm and the Python package `atom` to be installed."
    ),
)


def check_implementation(
    runner_ref: type[VllmRunner],
    runner_test: type[VllmRunner],
    example_prompts: list[str],
    model: str,
    kwargs_ref: dict[str, Any] | None = None,
    kwargs_test: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Compare vLLM(atom) vs vLLM(reference) logprobs."""
    if kwargs_ref is None:
        kwargs_ref = {}
    if kwargs_test is None:
        kwargs_test = {}

    max_tokens = 32
    num_logprobs = 5
    args = (example_prompts, max_tokens, num_logprobs)

    # We need VLLM_ATTENTION_BACKEND=CUSTOM for the ATOM path.
    os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM"
    with runner_test(model, **kwargs_test, **kwargs) as model_test:
        model_config = model_test.llm.llm_engine.model_config
        assert model_config.model_impl == "atom"
        assert model_config.architecture == "ATOMMoEForCausalLM"

        outputs_test = model_test.generate_greedy_logprobs(*args)

    # Reference path must not use CUSTOM.
    os.environ.pop("VLLM_ATTENTION_BACKEND", None)

    with runner_ref(model, **kwargs_ref, **kwargs) as model_ref:
        outputs_ref = model_ref.generate_greedy_logprobs(*args)

    check_logprobs_close(
        outputs_0_lst=outputs_ref,
        outputs_1_lst=outputs_test,
        name_0="auto",
        name_1="atom",
    )


@multi_gpu_test(num_gpus=8)
@pytest.mark.optional
def test_models(
    vllm_runner: type[VllmRunner],
    example_prompts: list[str],
) -> None:
    """ATOM backend test (8-GPUs): atom should run and generate."""
    model_id = ATOM_TEST_MODEL_ID
    prompts = [str(p).strip() for p in example_prompts[:2]]

    kwargs_common = dict(
        tensor_parallel_size=8,
        enable_expert_parallel=True,
    )

    check_implementation(
        vllm_runner,
        vllm_runner,
        prompts,
        model_id,
        kwargs_ref={"model_impl": "auto", **kwargs_common},
        kwargs_test={"model_impl": "atom", **kwargs_common},
    )
