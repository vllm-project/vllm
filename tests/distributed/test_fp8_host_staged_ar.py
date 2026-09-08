# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators.fp8_host_staged_all_reduce import (
    QUANT_BLOCK,
    KERNEL_BLOCK,
    Fp8HostStagedAllReduce,
    _quant_fp8_kernel,
)
from vllm.distributed.parallel_state import get_tp_group

from ..utils import (
    init_test_distributed_environment,
    multi_process_parallel,
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
    comm._cpu_group = None
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


def test_dequant_add_bitexact(dev):
    """Kernel dequant-add is bit-identical to the bf16-first torch reference.

    The kernel rounds each dequantized side to BF16 (RN mul + RN cast, the
    cvt breaking any fma contraction) and then adds the two BF16 values in
    FP32 (exact) and casts to BF16. The reference applies the same three
    roundings, so the outputs must be bitwise equal. Bitwise commutativity
    across the operand roles is what makes the replicated TP outputs
    bit-identical.
    """
    x0 = _make_input(dev, 4096 * 5120)
    x1 = _make_input(dev, 4096 * 5120)
    n = x0.numel()
    payload_buf = torch.empty((2, n), dtype=torch.uint8, device=dev)
    scale_buf = torch.empty((2, n // QUANT_BLOCK), dtype=torch.float32, device=dev)
    for row, x in enumerate((x0, x1)):
        p = payload_buf[row, :n].view(torch.float8_e4m3fn)
        s = scale_buf[row, : n // QUANT_BLOCK]
        _quant_fp8_kernel[(n // KERNEL_BLOCK,)](
            x, p, s, BLOCK=KERNEL_BLOCK, GROUP=QUANT_BLOCK, num_warps=4
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
    side0 = (
        payload_buf[0, :n].view(torch.float8_e4m3fn).float() * rep(scale_buf[0])
    ).to(torch.bfloat16)
    side1 = (
        payload_buf[1, :n].view(torch.float8_e4m3fn).float() * rep(scale_buf[1])
    ).to(torch.bfloat16)
    ref = (side0.float() + side1.float()).to(torch.bfloat16)
    assert torch.equal(out, ref), (
        f"dequant+add not bit-identical to the bf16-first reference; "
        f"max err {(out.float() - ref.float()).abs().max().item()}"
    )


@pytest.mark.parametrize(
    ("dtype", "numel", "contiguous", "expect"),
    [
        (torch.bfloat16, 1_024_000, True, True),  # >= MIN and %KERNEL_BLOCK == 0
        (torch.float16, 1_024_000, True, False),  # dtype gate
        (torch.bfloat16, 999_936, True, False),  # < MIN_ELEMS
        (torch.bfloat16, 1_000_064, True, False),  # %128 == 0 but %KERNEL_BLOCK != 0
        (torch.bfloat16, 1_024_000, False, False),  # non-contiguous
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
    x = torch.randn(1_024_000, dtype=torch.bfloat16, device=dev)
    assert comm.should_use(x) is False


@ray.remote(num_gpus=1, max_calls=1)
def fp8_hs_ar_target(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    # Ray workers must see all GPUs (the project never uses
    # CUDA_VISIBLE_DEVICES).
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    # opt in before the TP group (and its CudaCommunicator) is constructed
    os.environ["VLLM_FP8_HOST_STAGED_AR"] = "1"
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    comm = get_tp_group().device_communicator
    assert comm.fp8_hs_ar is not None, "FP8 host-staged AR was not constructed"
    hs = comm.fp8_hs_ar
    assert hs.disabled is False

    # warmup (also grows the buffers to the size used below)
    x = torch.randn(4096, 5120, dtype=torch.bfloat16, device=device)
    hs.all_reduce(x)
    torch.accelerator.synchronize()

    # correctness vs FP32 reference + rank-identical outputs (D4 contract)
    x = torch.randn(4096, 5120, dtype=torch.bfloat16, device=device)
    ref = x.float()
    group = get_tp_group().device_group
    dist.all_reduce(ref, group=group)
    torch.accelerator.synchronize()
    out = hs.all_reduce(x)
    torch.accelerator.synchronize()
    outs = [torch.empty_like(out) for _ in range(tp_size)]
    dist.all_gather(outs, out, group=group)
    torch.accelerator.synchronize()
    assert torch.equal(outs[0], outs[1]), (
        "replicated outputs must be bit-identical across ranks"
    )
    gmax = x.abs().max().float()
    dist.all_reduce(gmax, group=group, op=dist.ReduceOp.MAX)
    torch.accelerator.synchronize()
    # per-side E4M3 quantization (0.0625 * amax) + bf16-first dequant
    # roundings (two side casts + one sum cast, each <= 2^-8 of its operand,
    # operands <= 1.0625 gmax; the sum cast sees |out| <= 2.125 gmax)
    # + NCCL bf16 reduction rounding of the reference (2^-7 of each input)
    bound = (
        0.0625 * 2 * gmax * 1.001
        + 4.25 * gmax * 2**-8
        + 2.0 * gmax * 2**-7
    )
    err = (out.float() - ref).abs().max().item()
    assert err <= bound, f"max err {err} exceeds bound {bound.item()}"

    # dispatch: admitted message => exactly 2 NCCL sends on this rank (one
    # per phase of the one-way exchange), no ncclAllReduce; sub-MIN message
    # => plain NCCL allreduce (0 sends), bit-exact result
    counts = {"send": 0}
    orig_send = comm.pynccl_comm.send

    def spy_send(t, dst, stream=None):
        counts["send"] += 1
        return orig_send(t, dst, stream)

    comm.pynccl_comm.send = spy_send
    try:
        big = torch.randn(4096, 5120, dtype=torch.bfloat16, device=device)
        tensor_model_parallel_all_reduce(big)
        torch.accelerator.synchronize()
        assert counts["send"] == 2, f"expected 2 sends, got {counts['send']}"
        counts["send"] = 0
        small = torch.randn(5120, dtype=torch.bfloat16, device=device)
        out_small = tensor_model_parallel_all_reduce(small)
        torch.accelerator.synchronize()
        assert counts["send"] == 0, "sub-MIN message must stay on NCCL"
        ref_small = small.clone()
        dist.all_reduce(ref_small, group=group)
        torch.accelerator.synchronize()
        torch.testing.assert_close(out_small, ref_small, rtol=0, atol=0)
    finally:
        comm.pynccl_comm.send = orig_send


@pytest.mark.parametrize("tp_size", [2])
def test_fp8_hs_ar_2gpu(monkeypatch: pytest.MonkeyPatch, tp_size):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(monkeypatch, tp_size, 1, fp8_hs_ar_target)
