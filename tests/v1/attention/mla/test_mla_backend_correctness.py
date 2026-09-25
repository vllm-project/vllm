# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA backend correctness across every GPU MLA prefill backend.

The test bodies live in ``tests/v1/attention/_mla_backends.py``. Every GPU
prefill backend in ``MLAPrefillBackendEnum`` is parametrized here, so a newly
added backend is collected automatically; one that cannot run on this device
collects a skip-marked case instead of disappearing from the matrix.
"""

import pytest

from tests.v1.attention._mla_backends import (
    BACKEND_CORRECTNESS_BATCH_SPEC_NAMES,
    prefill_backend_dimension_params,
    run_backend_correctness,
)
from vllm.v1.attention.backends.mla.prefill import MLAPrefillBackendEnum

PREFILL_BACKEND_DIMENSIONS = [
    param
    for prefill_backend in MLAPrefillBackendEnum
    if prefill_backend not in (MLAPrefillBackendEnum.CPU, MLAPrefillBackendEnum.CUSTOM)
    for param in prefill_backend_dimension_params(prefill_backend)
]


@pytest.mark.parametrize("batch_spec_name", BACKEND_CORRECTNESS_BATCH_SPEC_NAMES)
@pytest.mark.parametrize("model", ["deepseek-ai/DeepSeek-R1"])
@pytest.mark.parametrize("tensor_parallel_size", [1, 4, 8, 16])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8", "fp8_e4m3"])
@pytest.mark.parametrize(("q_scale", "k_scale"), [(1.0, 1.0), (2.0, 3.0)])
@pytest.mark.parametrize(
    ("prefill_backend", "qk_nope_head_dim", "v_head_dim"),
    PREFILL_BACKEND_DIMENSIONS,
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
    run_backend_correctness(
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
    PREFILL_BACKEND_DIMENSIONS,
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
    run_backend_correctness(
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
