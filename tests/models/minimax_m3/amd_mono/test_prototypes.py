# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.minimax_m3.amd.mono.prototypes import (
    a4w4_resource,
    bf16_gemv_resource,
    compile_a4w4_prototype,
    fp32_router_resource,
    run_bf16_gemv_prototype,
    run_fp32_router_prototype,
)
from vllm.platforms import current_platform


def _require_gfx950():
    if not current_platform.is_rocm():
        pytest.skip("ROCm-only")
    from vllm.platforms.rocm import get_cdna_version

    if get_cdna_version() != 4:
        pytest.skip("requires gfx950")


@pytest.mark.parametrize("rows", [1, 16])
def test_bf16_gemv_prototype(rows: int):
    _require_gfx950()
    torch.manual_seed(1234 + rows)
    hidden = torch.randn(rows, 6144, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(2560, 6144, dtype=torch.bfloat16, device="cuda")
    actual = run_bf16_gemv_prototype(hidden, weight)
    expected = torch.mm(hidden, weight.t())
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.1)


@pytest.mark.parametrize("rows", [1, 16])
def test_fp32_router_prototype(rows: int):
    _require_gfx950()
    torch.manual_seed(4321 + rows)
    hidden = torch.randn(rows, 6144, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(128, 6144, dtype=torch.float32, device="cuda")
    actual = run_fp32_router_prototype(hidden, weight)
    expected = torch.mm(hidden.float(), weight.t())
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)


def test_a4w4_compile_and_resource(monkeypatch, tmp_path):
    _require_gfx950()
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    compile_a4w4_prototype()
    resource = a4w4_resource()
    assert resource.threads == 256
    assert resource.lds_bytes == 16640
    assert resource.grid_ctas == 256
    assert not resource.residency_proven


def test_unfolded_stage_resources_are_not_residency_proofs():
    assert not bf16_gemv_resource().residency_proven
    assert not fp32_router_resource(16).residency_proven
