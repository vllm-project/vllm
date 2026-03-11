# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for FlashInfer MoeAlltoAll/One-sided NVLink (trtllm_moe_alltoall) kernel backend.

Validates the _supports_parallel_config incompatibility matrix to ensure
each Expert backend correctly accepts or rejects the flashinfer_nvlink_one_sided
parallel configuration.  No GPU required.

See also:
  - mk_objects.py for combinatorial registration of the new P/F and Experts
"""

import importlib

import pytest

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
)


def _make_parallel_config(all2all_backend: str) -> FusedMoEParallelConfig:
    """Create a FusedMoEParallelConfig with EP enabled for the given backend."""
    return FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=2,
        ep_size=2,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        all2all_backend=all2all_backend,
        enable_eplb=False,
    )


def _import_expert_cls(module_path: str, class_name: str, skip_reason: str | None):
    """Import an Expert class, skipping the test if unavailable."""
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except (ImportError, AttributeError):
        if skip_reason:
            pytest.skip(skip_reason)
        raise


# (module_path, class_name, supports_flashinfer_nvlink_one_sided, skip_reason)
_EXPERT_COMPAT_CASES = [
    # Backends that reject flashinfer_nvlink_one_sided (Standard format, no all2allv)
    (
        "vllm.model_executor.layers.fused_moe.fused_moe",
        "TritonExperts",
        False,
        None,
    ),
    (
        "vllm.model_executor.layers.fused_moe.deep_gemm_moe",
        "DeepGemmExperts",
        False,
        "requires deep_gemm",
    ),
    (
        "vllm.model_executor.layers.fused_moe.fused_marlin_moe",
        "MarlinExperts",
        False,
        None,
    ),
    (
        "vllm.model_executor.layers.fused_moe.cutlass_moe",
        "CutlassExpertsFp8",
        False,
        "requires cutlass_fp8",
    ),
    # Backends that accept flashinfer_nvlink_one_sided
    (
        "vllm.model_executor.layers.fused_moe.fused_batched_moe",
        "BatchedTritonExperts",
        True,
        None,
    ),
    (
        "vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe",
        "FlashInferExperts",
        True,
        "requires flashinfer_cutlass on Blackwell",
    ),
    (
        "vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe",
        "TrtLlmNvFp4ExpertsModular",
        True,
        "requires flashinfer trtllm",
    ),
]


@pytest.mark.parametrize(
    "module_path,class_name,expected_support,skip_reason",
    _EXPERT_COMPAT_CASES,
    ids=[c[1] for c in _EXPERT_COMPAT_CASES],
)
def test_supports_parallel_config_flashinfer_nvlink_one_sided(
    module_path: str,
    class_name: str,
    expected_support: bool,
    skip_reason: str | None,
):
    """Verify _supports_parallel_config for the flashinfer_nvlink_one_sided backend."""
    cls = _import_expert_cls(module_path, class_name, skip_reason)
    config = _make_parallel_config("flashinfer_nvlink_one_sided")
    result = cls._supports_parallel_config(config)
    assert result == expected_support, (
        f"{class_name}._supports_parallel_config('flashinfer_nvlink_one_sided') "
        f"returned {result}, expected {expected_support}"
    )


@pytest.mark.parametrize(
    "module_path,class_name,expected_support,skip_reason",
    _EXPERT_COMPAT_CASES,
    ids=[c[1] for c in _EXPERT_COMPAT_CASES],
)
def test_supports_parallel_config_parity_with_all2allv(
    module_path: str,
    class_name: str,
    expected_support: bool,
    skip_reason: str | None,
):
    """Verify flashinfer_nvlink_one_sided and flashinfer_all2allv share the same
    incompatibility matrix (both reject and accept the same Expert backends).
    """
    cls = _import_expert_cls(module_path, class_name, skip_reason)
    config = _make_parallel_config("flashinfer_all2allv")
    result = cls._supports_parallel_config(config)
    assert result == expected_support, (
        f"{class_name}._supports_parallel_config('flashinfer_all2allv') "
        f"returned {result}, expected {expected_support}. "
        f"flashinfer_nvlink_one_sided and flashinfer_all2allv should share the same "
        f"incompatibility matrix."
    )
