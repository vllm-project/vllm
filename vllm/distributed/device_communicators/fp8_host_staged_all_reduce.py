# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-staged FP8 (E4M3) allreduce for TP2 on PCIe fabrics without P2P.

Quantizes the local input (per-128-element E4M3 payload + FP32 scale,
following the measured b12x codec), exchanges the two payloads over
NCCL send/recv (which the NCCL SHM transport stages through host
memory when P2P is unavailable), and dequantizes and reduces locally.
Both ranks materialize the result from the same two wire payloads so
the replicated activations stay bit-identical.
"""

import torch
import triton
import triton.language as tl

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

FP8_MAX = tl.constexpr(448.0)
QUANT_BLOCK = 128
MIN_ELEMS = 1_000_000


@triton.jit
def _quant_fp8_kernel(
    x_ptr, payload_ptr, scale_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    # The codec specifies IEEE RN divisions (b12x uses div_rn_f32 twice);
    # Triton's default `/` lowers to div.full.f32, which is not
    # correctly rounded, so use div_rn explicitly.
    scale = tl.where(amax > 0.0, tl.math.div_rn(amax, FP8_MAX), 1.0)
    inv = tl.math.div_rn(1.0, scale)
    y = tl.clamp(x * inv, -FP8_MAX, FP8_MAX)
    tl.store(payload_ptr + offs, y.to(tl.float8e4nv))
    tl.store(scale_ptr + pid, scale)


@triton.jit
def _dequant_add_kernel(
    p0_ptr, s0_ptr, p1_ptr, s1_ptr, out_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    q0 = tl.load(p0_ptr + offs).to(tl.float32)
    q1 = tl.load(p1_ptr + offs).to(tl.float32)
    s0 = tl.load(s0_ptr + pid)
    s1 = tl.load(s1_ptr + pid)
    out = (q0 * s0 + q1 * s1).to(tl.bfloat16)
    tl.store(out_ptr + offs, out)


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
    ) -> None:
        assert pynccl_comm.world_size == 2
        self._comm = pynccl_comm
        self.rank = rank
        self.peer = (rank + 1) % 2
        self.device = device
        self._cap = 0
        self._payload: torch.Tensor | None = None
        self._scale: torch.Tensor | None = None
        self.disabled = False

    def _ensure_capacity(self, n: int) -> None:
        if n <= self._cap:
            return
        self._payload = torch.empty((2, n), dtype=torch.uint8, device=self.device)
        self._scale = torch.empty(
            (2, n // QUANT_BLOCK), dtype=torch.float32, device=self.device
        )
        self._cap = n

    def should_use(self, input_: torch.Tensor) -> bool:
        return (
            input_.dtype == torch.bfloat16
            and input_.is_contiguous()
            and input_.numel() >= MIN_ELEMS
            and input_.numel() % QUANT_BLOCK == 0
            and not torch.cuda.is_current_stream_capturing()
        )

    def quantize(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a contiguous BF16 tensor to (E4M3 payload, FP32 scales)."""
        n = input_.numel()
        self._ensure_capacity(n)
        payload = self._payload[0, :n].view(torch.float8_e4m3fn)
        scale = self._scale[0, : n // QUANT_BLOCK]
        _quant_fp8_kernel[(n // QUANT_BLOCK,)](
            input_, payload, scale, BLOCK=QUANT_BLOCK, num_warps=4
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
        _dequant_add_kernel[(n // QUANT_BLOCK,)](
            p0, s0, p1, s1, out, BLOCK=QUANT_BLOCK, num_warps=4
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
        own = self._payload[0, :n].view(torch.float8_e4m3fn)
        peer = self._payload[1, :n].view(torch.float8_e4m3fn)
        s_own = self._scale[0, : n // QUANT_BLOCK]
        s_peer = self._scale[1, : n // QUANT_BLOCK]
        _quant_fp8_kernel[(n // QUANT_BLOCK,)](
            input_, own, s_own, BLOCK=QUANT_BLOCK, num_warps=4
        )
        self._comm.send(own, self.peer)
        self._comm.send(s_own, self.peer)
        self._comm.recv(peer, self.peer)
        self._comm.recv(s_peer, self.peer)
        _dequant_add_kernel[(n // QUANT_BLOCK,)](
            own, s_own, peer, s_peer, out, BLOCK=QUANT_BLOCK, num_warps=4
        )
        return out
