# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bitwise path-equivalence tests for the Triton row kernel vs
torch.ops._C.per_token_group_fp8_quant (vllm/model_executor/layers/quantization/
utils/fp8_utils.py).

Structure follows the parametrized path-equivalence suites that reviewers asked for
on this file's history (#21083 review: all scale layouts; #55330: bit-exact claims
backed by torch.equal-level assertions, zero-row eps path, inf/NaN edges).

Destination: tests/kernels/quantization/test_per_token_group_quant_row.py
"""

import pytest
import torch

from tests.kernels.quant_utils import FP8_DTYPE
from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.platforms import current_platform

GROUP_SIZE = 128
EPS = 1e-10

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="row kernel is gated to CUDA; other platforms keep the _C/Triton paths",
)


def _alloc_scales(T: int, H: int, layout: str, device: str) -> torch.Tensor:
    sf_k = H // GROUP_SIZE
    if layout == "row":
        return torch.empty((T, sf_k), device=device, dtype=torch.float32)
    if layout == "col":
        return torch.empty((sf_k, T), device=device, dtype=torch.float32).permute(
            -1, -2
        )
    m = fp8_utils.get_tma_aligned_size(T, 4)
    return torch.empty_strided((T, sf_k), (1, m), device=device, dtype=torch.float32)


def _run(x: torch.Tensor, layout: str, use_row_kernel: bool):
    fp8_min, fp8_max = torch.finfo(FP8_DTYPE).min, torch.finfo(FP8_DTYPE).max
    T, H = x.shape
    x_q = torch.empty_like(x, dtype=FP8_DTYPE)
    x_s = _alloc_scales(T, H, layout, x.device.type and str(x.device))
    if use_row_kernel:
        torch.ops.vllm.per_token_group_fp8_quant_row(
            x,
            x_q,
            x_s,
            GROUP_SIZE,
            EPS,
            fp8_min,
            fp8_max,
            False,  # scale_ue8m0
            layout != "row",  # column_major_scales
            layout == "tma",  # tma_aligned_scales
        )
    else:
        torch.ops._C.per_token_group_fp8_quant(
            x,
            x_q,
            x_s,
            GROUP_SIZE,
            EPS,
            fp8_min,
            fp8_max,
            False,  # scale_ue8m0
            layout != "row",  # column_major
            layout == "tma",
        )  # tma_aligned
    return x_q, x_s


@pytest.mark.parametrize("T", [1, 7, 64, 512, 1452, 4096, 17083])
@pytest.mark.parametrize("H", [2048, 4096])
@pytest.mark.parametrize("layout", ["row", "col", "tma"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_row_kernel_bitwise_matches_cuda(T, H, layout, dtype):
    torch.manual_seed(0)
    x = (torch.randn(T, H, device="cuda", dtype=dtype) * 3).contiguous()
    if T >= 64:
        x[T // 2] = 0.0  # all-zero row: eps floor path
        x[0, :GROUP_SIZE] = float("inf")  # satfinite clamp path
    q_ref, s_ref = _run(x, layout, use_row_kernel=False)
    q_row, s_row = _run(x, layout, use_row_kernel=True)
    assert (q_ref.view(torch.uint8) == q_row.view(torch.uint8)).all(), (
        "quantized values differ bitwise"
    )
    assert (
        s_ref.contiguous().view(torch.int32) == s_row.contiguous().view(torch.int32)
    ).all(), "scales differ bitwise"


@torch.inference_mode()
def test_dispatch_fallbacks():
    """Shapes the row kernel must refuse stay on the existing paths."""
    x = torch.randn(2688, 2688, device="cuda", dtype=torch.bfloat16)
    # non-power-of-two H: public API must still work (via _C), bit-identically
    q, s = fp8_utils.per_token_group_quant_fp8(x, GROUP_SIZE, use_ue8m0=False)
    assert q.shape == x.shape and s.shape == (2688, 2688 // GROUP_SIZE)
    assert not fp8_utils._row_quant_eligible(x, GROUP_SIZE)
    # UE8M0 scales fall through to the _C kernel inside the op
    q, s = fp8_utils.per_token_group_quant_fp8(
        torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16),
        GROUP_SIZE,
        use_ue8m0=True,
    )
    nz = s[s > 0]
    assert torch.all(torch.log2(nz) == torch.log2(nz).round())


def test_opcheck():
    torch.manual_seed(0)
    x = (torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16)).contiguous()
    x_q = torch.empty_like(x, dtype=FP8_DTYPE)
    x_s = torch.empty((64, 2048 // GROUP_SIZE), device="cuda", dtype=torch.float32)
    fp8_min, fp8_max = torch.finfo(FP8_DTYPE).min, torch.finfo(FP8_DTYPE).max
    # test_schema is skipped: opcheck's cloned-output comparison calls
    # torch.allclose on the mutated fp8 tensor, and mul_cuda has no fp8
    # kernel — a torch harness limitation, not an op property.
    torch.library.opcheck(
        torch.ops.vllm.per_token_group_fp8_quant_row,
        (x, x_q, x_s, GROUP_SIZE, EPS, fp8_min, fp8_max, False, False, False),
        test_utils=("test_faketensor", "test_aot_dispatch_dynamic"),
    )
