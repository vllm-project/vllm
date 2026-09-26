# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-staged compressed allreduce for TP2 on PCIe fabrics without P2P.

Quantizes the local input with a wire codec and exchanges the two
[payload|scale] wire messages over NCCL send/recv in two one-way
phases (one message per phase) separated by a CPU barrier:
interleaved bidirectional P2P faults in the NCCL SHM transport on
P2P-dead platforms (Xid 31), while one-way exchanges are stable at all
sizes. The NCCL SHM transport stages the payloads through host memory
when P2P is unavailable. Both ranks materialize the result from the
same two wire payloads, and each dequantized side is rounded to BF16
before the add, so the replicated activations stay bit-identical.

Codecs: "fp8" (default) is the measured b12x codec — per-128 E4M3
payload + FP32 scale, n + n/32 wire bytes. "nvfp4" is OCP NVFP4 —
e2m1 payload (2 elements/byte) + per-16 E4M3 scale, n/2 + n/16 wire
bytes (54.5% of fp8); quality-sensitive, opt-in.
"""

import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.logger import init_logger

logger = init_logger(__name__)

FP8_MAX = tl.constexpr(448.0)
# Scale granularity (codec): one FP32 scale per QUANT_BLOCK elements, per the
# measured b12x e4m3/128 codec. This defines the wire format and admission.
QUANT_BLOCK = 128
# Elements processed per program (a multiple of QUANT_BLOCK). Batching several
# scale groups per program cuts the program count (and its launch overhead) by
# that factor; the codec (per-QUANT_BLOCK scale) is untouched, so outputs stay
# bit-identical. 1024 is the measured sweet spot and divides the model hidden
# size (5120), so every AR size (tokens x 5120) is a whole number of programs.
KERNEL_BLOCK = 1024
MIN_ELEMS = 1_000_000
# NVFP4 wire codec (OCP FP4): e2m1 payload, 2 elements per byte, one E4M3
# scale per NVFP4_SCALE_BLOCK (16) elements. Wire = n/2 + n/16 bytes,
# 54.5% of the E4M3 codec (n + n/32).
NVFP4_SCALE_BLOCK = 16
# Max magnitude of the e2m1 value grid {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
NVFP4_MAX = tl.constexpr(6.0)


@triton.jit
def _quant_fp8_kernel(
    x_ptr, payload_ptr, scale_ptr,
    BLOCK: tl.constexpr, GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    xg = tl.reshape(x, (BLOCK // GROUP, GROUP))
    amax = tl.max(tl.abs(xg), axis=1)
    # The codec specifies IEEE RN divisions (b12x uses div_rn_f32 twice);
    # Triton's default `/` lowers to div.full.f32, which is not
    # correctly rounded, so use div_rn explicitly.
    scale = tl.where(amax > 0.0, tl.math.div_rn(amax, FP8_MAX), 1.0)
    inv = tl.math.div_rn(1.0, scale)
    y = tl.clamp(xg * tl.reshape(inv, (BLOCK // GROUP, 1)), -FP8_MAX, FP8_MAX)
    tl.store(payload_ptr + offs, tl.reshape(y, (BLOCK,)).to(tl.float8e4nv))
    tl.store(
        scale_ptr + pid * (BLOCK // GROUP) + tl.arange(0, BLOCK // GROUP),
        scale,
    )


@triton.jit
def _dequant_add_kernel(
    p0_ptr, s0_ptr, p1_ptr, s1_ptr, out_ptr,
    BLOCK: tl.constexpr, GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    q0 = tl.reshape(tl.load(p0_ptr + offs).to(tl.float32), (BLOCK // GROUP, GROUP))
    q1 = tl.reshape(tl.load(p1_ptr + offs).to(tl.float32), (BLOCK // GROUP, GROUP))
    s0 = tl.load(s0_ptr + pid * (BLOCK // GROUP) + tl.arange(0, BLOCK // GROUP))
    s1 = tl.load(s1_ptr + pid * (BLOCK // GROUP) + tl.arange(0, BLOCK // GROUP))
    # Round each dequantized side to BF16 before the add. The cvt breaks
    # the compiler's fma contraction of q0*s0 + q1*s1 (fma(q0, s0, q1*s1)
    # is not bitwise commutative: the ranks swap the operand roles and
    # would differ by 1 FP32 ulp). Cost: one extra rounding per side.
    v0 = (q0 * tl.reshape(s0, (BLOCK // GROUP, 1))).to(tl.bfloat16).to(tl.float32)
    v1 = (q1 * tl.reshape(s1, (BLOCK // GROUP, 1))).to(tl.bfloat16).to(tl.float32)
    out = tl.reshape(v0 + v1, (BLOCK,))
    tl.store(out_ptr + offs, out.to(tl.bfloat16))


@triton.jit
def _e2m1_code(v: tl.tensor) -> tl.tensor:
    # Nearest e2m1 magnitude code 0..7 for v >= 0 on the OCP grid
    # {0, 0.5, 1, 1.5, 2, 3, 4, 6}, IEEE RTNE at the midpoints (ties go
    # to the even magnitude index). A pure function of the input, so
    # identical on every rank.
    return (
        (v > 0.25).to(tl.int32)
        + (v >= 0.75).to(tl.int32)
        + (v > 1.25).to(tl.int32)
        + (v >= 1.75).to(tl.int32)
        + (v > 2.5).to(tl.int32)
        + (v >= 3.5).to(tl.int32)
        + (v > 5.0).to(tl.int32)
    )


@triton.jit
def _quant_nvfp4_kernel(
    x_ptr, payload_ptr, scale_ptr,
    BLOCK: tl.constexpr, GROUP: tl.constexpr,
):
    # OCP NVFP4 wire: e2m1 payload, 2 elements per byte (even element in
    # the low nibble), one E4M3 scale per GROUP (16) elements. payload_ptr
    # is the uint8 region and scale_ptr the E4M3 region of one wire side.
    pid = tl.program_id(0)
    NG: tl.constexpr = BLOCK // GROUP
    G2: tl.constexpr = GROUP // 2
    g = tl.arange(0, NG)[:, None]
    b = tl.arange(0, G2)[None, :]
    x_lo = tl.load(x_ptr + pid * BLOCK + g * GROUP + b * 2).to(tl.float32)
    x_hi = tl.load(x_ptr + pid * BLOCK + g * GROUP + b * 2 + 1).to(tl.float32)
    # Same two-RN-division style as the E4M3 codec; e2m1 max magnitude is
    # 6, so the per-group scale is amax / 6 (the E4M3 byte is the wire scale).
    amax = tl.maximum(tl.max(tl.abs(x_lo), axis=1), tl.max(tl.abs(x_hi), axis=1))
    s = tl.where(amax > 0.0, tl.math.div_rn(amax, NVFP4_MAX), 1.0)
    s = s.to(tl.float8e4nv).to(tl.float32)
    tl.store(scale_ptr + pid * NG + tl.arange(0, NG), s.to(tl.float8e4nv))
    inv = tl.math.div_rn(1.0, s)[:, None]
    c_lo = _e2m1_code(tl.abs(x_lo) * inv)
    c_hi = _e2m1_code(tl.abs(x_hi) * inv)
    byte = (
        (c_lo | (((x_lo < 0).to(tl.int32) & (c_lo > 0)) << 3))
        | ((c_hi | (((x_hi < 0).to(tl.int32) & (c_hi > 0)) << 3)) << 4)
    )
    tl.store(
        payload_ptr + pid * (BLOCK // 2) + g * G2 + b,
        byte.to(tl.uint8),
    )


def _load_fp4_cuda():
    """Build (once, cached) or load the sm_120a hardware-FP4 extension.

    The NVFP4 payload unpacks with the F2FP hardware instruction
    (cvt.rn.f16x2.e2m1x2), which ptxas only accepts on the
    architecture-specific sm_120a target, so force it for the build.
    """
    from torch.utils import cpp_extension

    src = os.path.join(os.path.dirname(__file__), "fp4_ar_cuda.cu")
    old = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    try:
        return cpp_extension.load(
            name="fp4_ar_cuda",
            sources=[src],
            # torch 2.11 headers (c10::List) fail under nvcc's c++17 default.
            extra_cuda_cflags=["-O3", "-std=c++20"],
        )
    finally:
        if old is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old


class HostStagedAllReduce:
    """Quantized allreduce exchanged over NCCL send/recv.

    World size must be 2. The wire codec is "fp8" (default: per-128
    E4M3 payload + FP32 scale) or "nvfp4" (OCP NVFP4: e2m1 payload,
    2 elements/byte, per-16 E4M3 scale). Buffers grow lazily to the
    largest admitted message; NCCL owns the host staging (SHM
    transport), so no pinned buffers live here.
    """

    def __init__(
        self,
        pynccl_comm: PyNcclCommunicator,
        rank: int,
        device: torch.device,
        cpu_group: dist.ProcessGroup,
        codec: str = "fp8",
    ) -> None:
        assert codec in ("fp8", "nvfp4")
        assert pynccl_comm.world_size == 2
        self._comm = pynccl_comm
        self.rank = rank
        self.peer = (rank + 1) % 2
        self.device = device
        self._cpu_group = cpu_group
        self._codec = codec
        # NVFP4 dequant runs on the sm_120a F2FP extension; build it up
        # front so a broken toolchain fails the startup, not the first AR.
        self._fp4_cuda = _load_fp4_cuda() if codec == "nvfp4" else None
        self._cap = 0
        self._wire: torch.Tensor | None = None
        self.disabled = False
        # Quality-sensitive wire codec: keep the active choice visible in
        # the journal (the backend-selection line cannot distinguish it).
        logger.info_once(f"Host-staged AR active: wire codec={codec}")

    def _wire_bytes(self, n: int) -> int:
        # fp8: n payload bytes + 4B scale per QUANT_BLOCK.
        # nvfp4: n/2 payload bytes + 1B scale per NVFP4_SCALE_BLOCK.
        if self._codec == "fp8":
            return n + 4 * (n // QUANT_BLOCK)
        return n // 2 + n // NVFP4_SCALE_BLOCK

    def _ensure_capacity(self, n: int) -> None:
        if n <= self._cap:
            return
        # One uint8 region per side: [payload | scales], so each one-way
        # phase is one NCCL message (n % 1024 == 0 by admission keeps
        # both codecs' views aligned).
        self._wire = torch.empty(
            (2, self._wire_bytes(n)), dtype=torch.uint8, device=self.device
        )
        self._cap = n

    def _side_views(
        self, n: int, side: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(payload, scale) views into wire side `side` for n elements."""
        wire = self._wire[side, : self._wire_bytes(n)]
        if self._codec == "fp8":
            payload = wire[:n].view(torch.float8_e4m3fn)
            scale = wire[n:].view(torch.float32)
        else:
            payload = wire[: n // 2]
            scale = wire[
                n // 2 : n // 2 + n // NVFP4_SCALE_BLOCK
            ].view(torch.float8_e4m3fn)
        return payload, scale

    def should_use(self, input_: torch.Tensor) -> bool:
        return (
            input_.dtype == torch.bfloat16
            and input_.is_contiguous()
            and input_.numel() >= MIN_ELEMS
            and input_.numel() % KERNEL_BLOCK == 0
            and not torch.cuda.is_current_stream_capturing()
        )

    def _quantize(
        self,
        input_: torch.Tensor,
        payload: torch.Tensor,
        scale: torch.Tensor,
        n: int,
    ) -> None:
        if self._codec == "fp8":
            _quant_fp8_kernel[(n // KERNEL_BLOCK,)](
                input_, payload, scale,
                BLOCK=KERNEL_BLOCK, GROUP=QUANT_BLOCK, num_warps=4
            )
        else:
            _quant_nvfp4_kernel[(n // KERNEL_BLOCK,)](
                input_, payload, scale,
                BLOCK=KERNEL_BLOCK, GROUP=NVFP4_SCALE_BLOCK, num_warps=4
            )

    def quantize(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a contiguous BF16 tensor with the active wire codec."""
        n = input_.numel()
        self._ensure_capacity(n)
        payload, scale = self._side_views(n, 0)
        self._quantize(input_, payload, scale, n)
        return payload, scale

    def dequant_add(
        self,
        p0: torch.Tensor,
        s0: torch.Tensor,
        p1: torch.Tensor,
        s1: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """out = dequant(p0, s0) + dequant(p1, s1), FP32 add, BF16 out.

        Dispatches on the payload dtype: E4M3 (one byte per element) or
        packed NVFP4 (two elements per byte).
        """
        if p0.dtype == torch.uint8:
            self._fp4_cuda.dequant_add_nvfp4(p0, s0, p1, s1, out)
        else:
            n = p0.numel()
            _dequant_add_kernel[(n // KERNEL_BLOCK,)](
                p0, s0, p1, s1, out,
                BLOCK=KERNEL_BLOCK, GROUP=QUANT_BLOCK, num_warps=4
            )
        return out

    def all_reduce(
        self, input_: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not self.should_use(input_):
            raise ValueError(
                "input not admitted by host-staged allreduce: "
                f"shape={tuple(input_.shape)} dtype={input_.dtype} "
                f"contiguous={input_.is_contiguous()}"
            )
        if out is None:
            out = torch.empty_like(input_)
        n = input_.numel()
        self._ensure_capacity(n)
        # One wire message per direction: [payload | scales] in the active
        # codec's layout. The on-wire bytes are the codec's quantized
        # representation, so the replicated results stay bit-identical.
        w = self._wire_bytes(n)
        own = self._wire[0, :w]
        peer = self._wire[1, :w]
        p_own, s_own = self._side_views(n, 0)
        p_peer, s_peer = self._side_views(n, 1)
        self._quantize(input_, p_own, s_own, n)
        # Two one-way phases: rank 0 -> rank 1, CPU barrier, rank 1 -> rank
        # 0. Each op is stream-ordered, so a phase's data is fully received
        # on the destination before the reverse direction starts.
        if self.rank == 0:
            self._comm.send(own, 1)
            dist.barrier(group=self._cpu_group)
            self._comm.recv(peer, 1)
        else:
            self._comm.recv(peer, 0)
            dist.barrier(group=self._cpu_group)
            self._comm.send(own, 0)
        self.dequant_add(p_own, s_own, p_peer, s_peer, out)
        return out
