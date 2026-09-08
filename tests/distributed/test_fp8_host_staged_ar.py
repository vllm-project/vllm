# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.distributed.device_communicators.fp8_host_staged_all_reduce import (
    QUANT_BLOCK,
    Fp8HostStagedAllReduce,
    _quant_fp8_kernel,
)


def _quant_reference(x: torch.Tensor):
    """b12x e4m3/128 codec in torch: two RN roundings, clamped RTNE cast.

    Divisions must use IEEE-RN semantics to match tl.math.div_rn in the
    kernel. torch tensor/tensor division and scalar-left reciprocal are
    RN on CUDA; the ``tensor / python_float`` scalar path is not, so use
    an fp32 tensor divisor.
    """
    xb = x.view(-1, QUANT_BLOCK).float()
    amax = xb.abs().amax(dim=1)
    fp8_max = torch.full_like(amax, 448.0)
    scale = torch.where(amax > 0, amax / fp8_max, torch.ones_like(amax))
    inv = 1.0 / scale
    y = (xb * inv.unsqueeze(1)).clamp(-448.0, 448.0)
    payload = y.to(torch.float8_e4m3fn).view(-1)
    return payload, scale


def _uninitialized_comm(device: torch.device) -> Fp8HostStagedAllReduce:
    comm = Fp8HostStagedAllReduce.__new__(Fp8HostStagedAllReduce)
    comm.rank = 0
    comm.peer = 1
    comm.device = device
    comm._cap = 0
    comm._payload = None
    comm._scale = None
    comm.disabled = False
    return comm


@pytest.fixture(scope="module")
def dev():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch.device("cuda:0")


def _make_input(dev: torch.device, numel: int) -> torch.Tensor:
    torch.manual_seed(0)
    return (
        (torch.randn(numel, dtype=torch.float32, device=dev) * 3).to(torch.bfloat16)
    )


def test_quant_bitexact_vs_reference(dev):
    x = _make_input(dev, 4096 * 5120)
    comm = _uninitialized_comm(dev)
    payload, scale = comm.quantize(x)
    ref_payload, ref_scale = _quant_reference(x)
    assert payload.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    assert payload.numel() == x.numel()
    assert scale.numel() == x.numel() // QUANT_BLOCK
    assert torch.equal(
        payload.view(torch.uint8), ref_payload.view(torch.uint8)
    ), "payload must be bit-identical to the reference codec"
    assert torch.equal(scale, ref_scale), "scales must be bit-identical"


def test_quant_roundtrip_error_bound(dev):
    """Codec error bound: per element, err <= 0.0625 * block amax.

    E4M3 has a 3-bit mantissa (half step = 2^-4 relative in the normal
    range); the subnormal floor (2^-10 * scale) and the two RN roundings
    are covered by the 1.001 factor. Compared in FP32 (no BF16 cast) so
    the bound characterizes the codec alone.
    """
    x = _make_input(dev, 4096 * 5120)
    comm = _uninitialized_comm(dev)
    payload, scale = comm.quantize(x)
    dq = (payload.float() * scale.repeat_interleave(QUANT_BLOCK)).view(-1, QUANT_BLOCK)
    err = (dq - x.float().view(-1, QUANT_BLOCK)).abs()
    amax = x.float().abs().view(-1, QUANT_BLOCK).amax(dim=1, keepdim=True)
    bound = 0.0625 * amax * 1.001
    assert (err <= bound).all(), f"max err {err.max().item():.6f}"


def test_dequant_add_within_one_bf16_ulp(dev):
    """Kernel dequant-add vs the double-rounding torch reference.

    The kernel fuses q0*s0 + q1*s1 into an fma.rn.f32 (single rounding of
    the exact 4-bit-mantissa product); the reference rounds each product
    first. The two sums differ by at most 1 fp32 ulp, so the bf16 outputs
    must be within one bf16 ulp (with slack). Any codec-level deviation
    (wrong scale convention, dropped scale, ...) is ~2^-4 relative and
    far outside this bound.
    """
    x0 = _make_input(dev, 4096 * 5120)
    x1 = _make_input(dev, 4096 * 5120)
    n = x0.numel()
    payload_buf = torch.empty((2, n), dtype=torch.uint8, device=dev)
    scale_buf = torch.empty((2, n // QUANT_BLOCK), dtype=torch.float32, device=dev)
    for row, x in enumerate((x0, x1)):
        p = payload_buf[row, :n].view(torch.float8_e4m3fn)
        s = scale_buf[row, : n // QUANT_BLOCK]
        _quant_fp8_kernel[(n // QUANT_BLOCK,)](
            x, p, s, BLOCK=QUANT_BLOCK, num_warps=4
        )
    out = torch.empty_like(x0)
    comm = _uninitialized_comm(dev)
    comm.dequant_add(
        payload_buf[0, :n].view(torch.float8_e4m3fn),
        scale_buf[0, : n // QUANT_BLOCK],
        payload_buf[1, :n].view(torch.float8_e4m3fn),
        scale_buf[1, : n // QUANT_BLOCK],
        out,
    )
    rep = lambda s: s.repeat_interleave(QUANT_BLOCK).view(x0.shape)  # noqa: E731
    ref = (
        payload_buf[0, :n].view(torch.float8_e4m3fn).float() * rep(scale_buf[0])
        + payload_buf[1, :n].view(torch.float8_e4m3fn).float() * rep(scale_buf[1])
    ).to(torch.bfloat16)
    err = (out.float() - ref.float()).abs()
    bound = ref.float().abs() * 2**-6 + 2e-30
    assert (err <= bound).all(), (
        f"dequant+add deviates beyond one bf16 ulp; max err {err.max().item()}"
    )


@pytest.mark.parametrize(
    ("dtype", "numel", "contiguous", "expect"),
    [
        (torch.bfloat16, 1_000_064, True, True),  # >= MIN and %128 == 0
        (torch.float16, 1_000_064, True, False),  # dtype gate
        (torch.bfloat16, 999_936, True, False),  # < MIN_ELEMS
        (torch.bfloat16, 1_000_114, True, False),  # %128 != 0
        (torch.bfloat16, 1_000_064, False, False),  # non-contiguous
    ],
)
def test_should_use_gates(dev, dtype, numel, contiguous, expect):
    comm = _uninitialized_comm(dev)
    x = torch.randn(numel, dtype=dtype, device=dev)
    if not contiguous:
        x = x.view(-1, 2).t()
    assert comm.should_use(x) is expect


def test_should_use_rejects_graph_capture(dev, monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: True
    )
    comm = _uninitialized_comm(dev)
    x = torch.randn(1_000_064, dtype=torch.bfloat16, device=dev)
    assert comm.should_use(x) is False
