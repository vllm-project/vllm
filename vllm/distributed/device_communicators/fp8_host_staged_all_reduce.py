# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-staged FP8 (E4M3) allreduce for TP2 on PCIe fabrics without P2P.

Quantizes the local input (per-128-element E4M3 payload + FP32 scale,
following the measured b12x codec) and exchanges the two [payload|scale]
wire messages over NCCL send/recv in two one-way phases (one message per
phase) separated by a CPU barrier:
interleaved bidirectional P2P faults in the NCCL SHM transport on
P2P-dead platforms (Xid 31), while one-way exchanges are stable at all
sizes. The NCCL SHM transport stages the payloads through host memory
when P2P is unavailable. Both ranks materialize the result from the
same two wire payloads, and each dequantized side is rounded to BF16
before the add, so the replicated activations stay bit-identical.
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

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


class Fp8HostStagedAllReduce:
    """FP8-compressed allreduce exchanged over NCCL send/recv.

    World size must be 2. Buffers grow lazily to the largest admitted
    message; NCCL owns the host staging (SHM transport), so no pinned
    buffers live here.
    """

    def __init__(
        self,
        pynccl_comm: PyNcclCommunicator,
        rank: int,
        device: torch.device,
        cpu_group: dist.ProcessGroup,
    ) -> None:
        assert pynccl_comm.world_size == 2
        self._comm = pynccl_comm
        self.rank = rank
        self.peer = (rank + 1) % 2
        self.device = device
        self._cpu_group = cpu_group
        self._cap = 0
        self._wire: torch.Tensor | None = None
        self.disabled = False

    def _ensure_capacity(self, n: int) -> None:
        if n <= self._cap:
            return
        # One buffer per side: [fp8 payload | fp32 scales] as a single uint8
        # region, so each one-way phase is one NCCL message (n % 1024 == 0
        # by admission keeps the fp32 view 4-byte aligned).
        self._wire = torch.empty(
            (2, n + 4 * (n // QUANT_BLOCK)), dtype=torch.uint8, device=self.device
        )
        self._cap = n

    def should_use(self, input_: torch.Tensor) -> bool:
        return (
            input_.dtype == torch.bfloat16
            and input_.is_contiguous()
            and input_.numel() >= MIN_ELEMS
            and input_.numel() % KERNEL_BLOCK == 0
            and not torch.cuda.is_current_stream_capturing()
        )

    def quantize(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a contiguous BF16 tensor to (E4M3 payload, FP32 scales)."""
        n = input_.numel()
        self._ensure_capacity(n)
        wire = self._wire[0, : n + 4 * (n // QUANT_BLOCK)]
        payload = wire[:n].view(torch.float8_e4m3fn)
        scale = wire[n:].view(torch.float32)
        _quant_fp8_kernel[(n // KERNEL_BLOCK,)](
            input_, payload, scale,
            BLOCK=KERNEL_BLOCK, GROUP=QUANT_BLOCK, num_warps=4
        )
        return payload, scale

    def dequant_add(
        self,
        p0: torch.Tensor,
        s0: torch.Tensor,
        p1: torch.Tensor,
        s1: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """out = dequant(p0, s0) + dequant(p1, s1), FP32 add, BF16 out."""
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
                "input not admitted by FP8 host-staged allreduce: "
                f"shape={tuple(input_.shape)} dtype={input_.dtype} "
                f"contiguous={input_.is_contiguous()}"
            )
        if out is None:
            out = torch.empty_like(input_)
        n = input_.numel()
        self._ensure_capacity(n)
        # One wire message per direction: [fp8 payload | fp32 scales]. The
        # on-wire bytes are the concatenation the previous two-message
        # exchange carried, so results stay bit-identical.
        w = n + 4 * (n // QUANT_BLOCK)
        own = self._wire[0, :w]
        peer = self._wire[1, :w]
        p_own = own[:n].view(torch.float8_e4m3fn)
        s_own = own[n:].view(torch.float32)
        p_peer = peer[:n].view(torch.float8_e4m3fn)
        s_peer = peer[n:].view(torch.float32)
        _quant_fp8_kernel[(n // KERNEL_BLOCK,)](
            input_, p_own, s_own,
            BLOCK=KERNEL_BLOCK, GROUP=QUANT_BLOCK, num_warps=4
        )
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
        _dequant_add_kernel[(n // KERNEL_BLOCK,)](
            p_own, s_own, p_peer, s_peer, out,
            BLOCK=KERNEL_BLOCK, GROUP=QUANT_BLOCK, num_warps=4
        )
        return out
