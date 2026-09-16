# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fwht128_quant_fp8 must emit the platform fp8 dtype.

On ROCm the indexer query produced by ``fwht128_quant_fp8`` flows into
``rocm_fp8_mqa_logits``. gfx942 AITER's flydsl kernel cannot compile with a
``float8_e4m3fn`` query, and the hardcoded 448 clamp is the e4m3fn max,
wrong for e4m3fnuz. These tests pin the output dtype and the clamp source;
the numeric Hadamard check runs wherever Triton is active.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skip_global_cleanup,
]


def test_fwht_allocates_platform_fp8_dtype(monkeypatch):
    """The wrapper must derive the output dtype from the platform, not e4m3fn.

    Intercepts torch.empty inside fwht128_quant_fp8 so no accelerator is
    needed; the zero-row early-return path still allocates both outputs.
    """
    from vllm.models.glm5next.nvidia.ops import kpool_compress as nvidia_ops

    seen_dtypes = []
    real_empty = torch.empty

    def spy_empty(*args, **kwargs):
        if kwargs.get("dtype") in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            seen_dtypes.append(kwargs["dtype"])
        return real_empty(*args, **kwargs)

    q = torch.zeros(0, 128, dtype=torch.bfloat16)
    monkeypatch.setattr(nvidia_ops.torch, "empty", spy_empty)
    try:
        q_fp8, _ = nvidia_ops.fwht128_quant_fp8(q)
    finally:
        monkeypatch.setattr(nvidia_ops.torch, "empty", real_empty)

    assert seen_dtypes, "expected an fp8 allocation"
    assert seen_dtypes[-1] == current_platform.fp8_dtype(), (
        f"output dtype must be the platform fp8 dtype "
        f"{current_platform.fp8_dtype()}, got {seen_dtypes[-1]}"
    )
    assert q_fp8.shape == (0, 128)


def test_fwht_kernel_clamp_uses_parameterized_max():
    """Source-contract check for the absmax clamp (accelerator-free CI).

    On machines without Triton this is the only guard that the kernel takes
    its clamp bound from the FP8_MAX constexpr; where the numeric test runs,
    it covers the behavioral side.
    """

    from vllm.models.glm5next.nvidia.ops import kpool_compress as nvidia_ops

    with open(nvidia_ops.__file__) as f:
        src = f.read()
    fwht_kernel_src = src.split("def _fwht_quant_kernel")[1].split("\ndef ")[0]
    assert "448" not in fwht_kernel_src, (
        "kernel must not hardcode the e4m3fn max; use the FP8_MAX constexpr"
    )
    assert "FP8_MAX" in fwht_kernel_src


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a CUDA/ROCm accelerator for the Triton kernel",
)
@pytest.mark.skipif(
    __import__("vllm.triton_utils.importing", fromlist=["HAS_TRITON"]).HAS_TRITON
    is False,
    reason="Triton disabled in this environment",
)
def test_fwht128_quant_fp8_matches_torch_reference():
    """Quantization-aware comparison against an unfused Hadamard reference.

    fp8 e4m3's worst-case relative step is 2^-4, so dequantized outputs must
    recover the reference transform within that bound; scales must match the
    ue8m0 power-of-two formula exactly.
    """
    from scipy.linalg import hadamard

    from vllm.models.glm5next.nvidia.ops.kpool_compress import (
        fwht128_quant_fp8,
    )

    torch.manual_seed(0)
    device = "cuda"
    fp8_dtype = current_platform.fp8_dtype()
    fp8_max = float(torch.finfo(fp8_dtype).max)

    for magnitude in (3.0, 6.0, 12.0):
        n_rows = 257  # non-multiple of BLOCK_R
        q = torch.randn(n_rows, 128, dtype=torch.bfloat16, device=device)
        q = q * magnitude

        q_fp8, q_scale = fwht128_quant_fp8(q)

        assert q_fp8.dtype == fp8_dtype
        assert not torch.isnan(q_fp8.to(torch.float32)).any()
        assert q_fp8.to(torch.float32).abs().max() <= fp8_max

        h = torch.from_numpy(hadamard(128)).to(device=device, dtype=torch.float32)
        x = (q.to(torch.float32) @ h) * (1.0 / 128.0**0.5)
        x = x.to(torch.bfloat16).to(torch.float32)

        absmax = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=1e-4)
        scale_ref = torch.exp2(torch.ceil(torch.log2(absmax / fp8_max)))
        assert torch.equal(q_scale, scale_ref)

        deq = q_fp8.to(torch.float32) * q_scale
        rel = ((deq - x).abs() / x.abs().clamp(min=1e-2)).max()
        assert rel <= 0.07, f"dequant rel err {rel} exceeds fp8 step bound"
