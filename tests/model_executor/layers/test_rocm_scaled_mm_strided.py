# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The activation contract of the ROCm skinny GEMM dispatches.

wvSplitKQ stages the activation by walking stride(0) for N rows while its compute
loop stops at the valid width, so it needs a densely packed activation. The
caller builds one with `x_data.view(-1, x_data.shape[-1])`, which is a no-op on an
already-2D tensor and therefore forwards a strided view unchanged.

`is_contiguous()` cannot express this contract: it ignores the stride of a
size-1 dimension, so a single-row view keeps whatever row stride it was built
with and still reports contiguous. These tests pin the stride, not the flag.
"""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.model_executor.kernels.linear.scaled_mm import rocm as rocm_scaled_mm


def is_dense(t: torch.Tensor) -> bool:
    return t.stride() == (t.size(1), 1)


def _run(monkeypatch, A):
    """Dispatch A through the fp8 skinny path and return what the kernel saw."""
    B = torch.randn(64, A.size(1), dtype=torch.float16).to(torch.float8_e4m3fn)
    seen = {}

    def fake_wvsplitkq(b_t, a, out_dtype, As, Bs, cu, bias):
        seen["a"] = a
        return torch.zeros(a.size(0), b_t.size(0), dtype=out_dtype)

    monkeypatch.setattr(
        rocm_scaled_mm.ops, "wvSplitKQ", MagicMock(side_effect=fake_wvsplitkq)
    )
    monkeypatch.setattr(rocm_scaled_mm, "num_compute_units", lambda: 120)
    rocm_scaled_mm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
        A,
        B,
        torch.bfloat16,
        torch.ones(1, dtype=torch.float32),
        torch.ones(1, dtype=torch.float32),
        None,
    )
    assert "a" in seen, "the skinny path did not run"
    return seen["a"]


@pytest.mark.parametrize("rows", [1, 2, 4])
def test_row_gapped_activation_is_densified(monkeypatch, rows):
    """A window inside a wider row block: the layout that motivated this."""
    stride = 6288
    base = torch.empty((rows + 2) * stride, dtype=torch.float16).to(torch.float8_e4m3fn)
    A = torch.as_strided(base, (rows, 128), (stride, 1), 6144)
    assert A.stride() == (stride, 1)
    # rows == 1 is the case a contiguity flag cannot catch.
    assert A.is_contiguous() == (rows == 1)

    assert is_dense(_run(monkeypatch, A)), "kernel received a strided activation"


def test_transposed_activation_is_densified(monkeypatch):
    A = torch.randn(128, 4, dtype=torch.float16).to(torch.float8_e4m3fn).t()
    assert A.shape == (4, 128) and A.stride() == (1, 4)

    assert is_dense(_run(monkeypatch, A))


def test_dense_activation_is_forwarded_unchanged(monkeypatch):
    """No copy on the path that was already correct."""
    A = torch.randn(4, 128, dtype=torch.float16).to(torch.float8_e4m3fn)

    seen = _run(monkeypatch, A)
    assert is_dense(seen)
    assert seen.data_ptr() == A.data_ptr()


def test_kernel_rejects_strided_activation():
    """The kernel-side contract, exercised against the real op.

    The densification above is what production relies on; this pins the contract
    the kernel itself now states, so a future call site that forgets cannot
    silently over-read. Without the TORCH_CHECK this call walks stride(0) * N
    elements against K * N valid ones and faults once the over-read leaves the
    mapping -- an 8 MB over-read reproduces `Memory access fault by GPU node-3`
    on MI355X.
    """
    ops = pytest.importorskip("vllm._custom_ops")
    if not hasattr(ops, "wvSplitKQ"):
        pytest.skip("wvSplitKQ is not built in this configuration")

    K, rows, stride = 1024, 1, 1024 * 64
    base = torch.empty((rows + 1) * stride, dtype=torch.float16, device="cuda").to(
        torch.float8_e4m3fn
    )
    act = torch.as_strided(base, (rows, K), (stride, 1), stride - K)
    assert act.stride() == (stride, 1)
    weight = torch.randn(64, K, dtype=torch.float16, device="cuda").to(
        torch.float8_e4m3fn
    )
    scale = torch.ones(1, dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="densely packed"):
        ops.wvSplitKQ(weight, act, torch.bfloat16, scale, scale, 120, None)
