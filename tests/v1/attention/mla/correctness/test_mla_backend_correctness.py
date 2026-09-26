# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.v1.attention._mla_backends import (
    _prefill_backend_dimension_params,
    _run_backend_correctness,
)
from vllm.v1.attention.backends.mla.prefill import MLAPrefillBackendEnum


@pytest.mark.parametrize(
    "batch_spec_name",
    [
        "small_decode",
        "small_prefill",
        "mixed_small",
        "medium_decode",
        "medium_prefill",
        "mixed_medium",
        "large_decode",
        "large_prefill",
        "single_decode",
        "single_prefill",
        "spec_decode_small",
        "spec_decode_medium",
    ],
)
@pytest.mark.parametrize("model", ["deepseek-ai/DeepSeek-R1"])
@pytest.mark.parametrize("tensor_parallel_size", [1, 4, 8, 16])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8", "fp8_e4m3"])
@pytest.mark.parametrize(("q_scale", "k_scale"), [(1.0, 1.0), (2.0, 3.0)])
@pytest.mark.parametrize(
    ("prefill_backend", "qk_nope_head_dim", "v_head_dim"),
    _prefill_backend_dimension_params(),
)
def test_backend_correctness(
    default_vllm_config,
    dist_init,
    workspace_init,
    batch_spec_name: str,
    model: str,
    tensor_parallel_size: int,
    kv_cache_dtype: str,
    q_scale: float,
    k_scale: float,
    prefill_backend: MLAPrefillBackendEnum,
    qk_nope_head_dim: int,
    v_head_dim: int,
):
    _run_backend_correctness(
        default_vllm_config,
        dist_init,
        workspace_init,
        batch_spec_name,
        model,
        tensor_parallel_size,
        kv_cache_dtype,
        q_scale,
        k_scale,
        prefill_backend,
        qk_nope_head_dim,
        v_head_dim,
    )


@pytest.mark.parametrize(
    ("prefill_backend", "qk_nope_head_dim", "v_head_dim"),
    _prefill_backend_dimension_params(),
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_chunked_context_backend_correctness(
    default_vllm_config,
    dist_init,
    workspace_init,
    prefill_backend: MLAPrefillBackendEnum,
    qk_nope_head_dim: int,
    v_head_dim: int,
    kv_cache_dtype: str,
):
    """Split, packed, and context-free requests match the SDPA reference."""
    _run_backend_correctness(
        default_vllm_config,
        dist_init,
        workspace_init,
        batch_spec_name="chunked_context_prefill",
        model="deepseek-ai/DeepSeek-R1",
        tensor_parallel_size=16,
        kv_cache_dtype=kv_cache_dtype,
        q_scale=1.0,
        k_scale=1.0,
        prefill_backend=prefill_backend,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
        chunked_prefill_workspace_size=1024,
    )
