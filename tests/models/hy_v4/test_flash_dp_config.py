# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HY4-Flash DP/EP parallel config validation."""

from types import SimpleNamespace

import pytest

from vllm.models.hy_v4_flash.configs import HYV4FlashConfig


def _make_parallel(all2all_backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_eplb=False,
        enable_dbo=False,
        tensor_parallel_size=1,
        data_parallel_size=8,
        enable_expert_parallel=True,
        enable_elastic_ep=False,
        enable_fault_tolerance=False,
        use_ubatching=False,
        nnodes=1,
        expert_placement_strategy="linear",
        all2all_backend=all2all_backend,
    )


@pytest.mark.parametrize("backend", ["allgather_reducescatter", "passthrough"])
def test_flash_dp_allows_supported_all2all_backends(backend: str):
    config = HYV4FlashConfig(n_routed_experts=16)
    config.verify_parallel_config(_make_parallel(backend))  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", ["deepep_low_latency", "pplx"])
def test_flash_dp_rejects_unsupported_all2all_backends(backend: str):
    config = HYV4FlashConfig(n_routed_experts=16)
    with pytest.raises(
        ValueError, match="Flash DP requires allgather_reducescatter or passthrough"
    ):
        config.verify_parallel_config(_make_parallel(backend))  # type: ignore[arg-type]
