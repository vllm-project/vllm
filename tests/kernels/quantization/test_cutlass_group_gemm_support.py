# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm import _custom_ops as ops


def test_cutlass_group_gemm_python_guard_allows_thor(monkeypatch):
    seen_capabilities: list[int] = []

    def fake_cutlass_group_gemm_supported(capability: int) -> bool:
        seen_capabilities.append(capability)
        return True

    monkeypatch.setattr(
        torch.ops,
        "_C",
        SimpleNamespace(cutlass_group_gemm_supported=fake_cutlass_group_gemm_supported),
    )

    # CUDA 12 reports Thor as SM101 and CUDA 13 reports it as SM110. Both
    # should reach the C++ query for the SM10x/SM11x CUTLASS MoE kernel.
    assert ops.cutlass_group_gemm_supported(101)
    assert ops.cutlass_group_gemm_supported(110)

    # SM120 uses separate kernels and must not be advertised by this path.
    assert not ops.cutlass_group_gemm_supported(120)
    assert seen_capabilities == [101, 110]
