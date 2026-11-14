# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused TP all-reduce + mHC boundary on Hopper, and the MTP draft's
all-reduce + RMSNorm.

`HopperBoundary` is `AllReduceMhcBoundary` with the SM90 boundary kernel
(`_boundary_kernels`), for TP 2 or 4. Differences from the GB300 class:

- The Hopper MoE finalizes its top-k sum in its own kernel, so every sublayer
  hands a TP-partial BF16 tensor; there is no fused MoE finalize.
- Per token count, the input is either the Lamport mailbox ("lamport": main's
  publish kernel multicasts each rank's partial over NVLink) or the stock
  all-reduce's output ("reduced"), with `tokens` tokens per 8-CTA cluster.

The post-mix and the Sinkhorn residual mix run on a side stream, as on GB300.
"""

from __future__ import annotations

from collections.abc import Mapping

import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Int32, Int64
from cutlass.cute.runtime import make_fake_compact_tensor

from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.models.common.ops.moe_tail_fusion.primitives import (
    LAMPORT_GENERATIONS,
    current_cu_stream,
    make_fake_dynamic_compact_tensor,
    to_cute,
    to_cute_dynamic,
)

from ..all_reduce_mhc import HC, MIX, _PublishDeviceKernel
from ..all_reduce_norm import AllReduceNorm
from ._boundary_kernels import compile_boundary, compile_mixes

# (input, tokens per cluster) by token count. H200 holds 15 of the 8-CTA
# clusters at once (one CTA an SM), so each 15 tokens add one token a cluster.
DEFAULT_MODES: dict[int, tuple[str, int]] = {
    m: ("lamport", (m + 14) // 15) for m in range(1, 33)
}


def parse_mode(choice: str) -> tuple[str, int]:
    """`"lamport2"` -> `("lamport", 2)`: a choice of the runtime policy's
    `boundary` entry as a `modes` value."""
    for name in ("lamport", "reduced"):
        if choice.startswith(name):
            tokens = int(choice[len(name) :] or 1)
            if tokens >= 1:
                return name, tokens
    raise ValueError(f"Unknown boundary mode {choice!r}")


class HopperBoundary:
    """`tensor_model_parallel_all_reduce` + `hc_fused_post_pre` for the token
    counts in `modes`: up to two kernels on the current stream and one on a
    side stream.

    The returned post-mix and residual mix are written on the side stream; the
    next call joins it, and a stock consumer must call `join` first.
    Construction is collective over the TP group; one instance serves every
    boundary of the model, which run one after another.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        max_num_tokens: int,
        rms_eps: float,
        hc_eps: float,
        post_mult: float,
        sinkhorn_repeat: int,
        norm_eps: float,
        device: torch.device,
        modes: Mapping[int, tuple[str, int]] | None = None,
    ) -> None:
        group = get_tp_group().device_group
        self.hidden_size = hidden = hidden_size
        self.capacity = capacity = max_num_tokens
        tp = dist.get_world_size(group)
        rank = dist.get_rank(group)
        device = torch.device(device)
        modes = DEFAULT_MODES if modes is None else modes
        self.modes = {m: mode for m, mode in modes.items() if 1 <= m <= capacity}
        lamport = any(name == "lamport" for name, _ in self.modes.values())

        with torch.accelerator.device_index(device.index):
            self._kernels = {
                mode: compile_boundary(
                    tokens=mode[1],
                    reduced=mode[0] == "reduced",
                    hidden=hidden,
                    tp=tp,
                    capacity=capacity,
                    rms_eps=rms_eps,
                    pre_eps=hc_eps,
                    norm_eps=norm_eps,
                )
                for mode in set(self.modes.values())
            }
            self._mixes = compile_mixes(
                sinkhorn_eps=hc_eps,
                post_mult=post_mult,
                repeat=sinkhorn_repeat,
                capacity=capacity,
            )
            self._stage_state = torch.zeros(2, dtype=torch.int32, device=device)
            self._mailbox = None
            if lamport:
                self._publish = cute.compile(
                    _PublishDeviceKernel(
                        hidden=hidden, tp=tp, rank=rank, capacity_m=capacity
                    ),
                    make_fake_dynamic_compact_tensor(
                        BFloat16, alignment=16, divisibility=hidden
                    ),
                    make_fake_compact_tensor(Int32, (2,), assumed_align=4),
                    Int64(0),
                    Int32(capacity),
                    current_cu_stream(),
                )
                self._mailbox = symm_mem.empty(
                    (LAMPORT_GENERATIONS, tp, capacity, hidden),
                    dtype=torch.bfloat16,
                    device=device,
                )
                self._mailbox_handle = symm_mem.rendezvous(self._mailbox, group)
                multicast = int(self._mailbox_handle.multicast_ptr or 0)
                if not multicast or multicast % 16:
                    raise RuntimeError("NVLink multicast mapping is unavailable")
                self._multicast = multicast
                # Every slot starts as the Lamport sentinel, BF16 negative zero.
                self._mailbox.view(torch.int16).fill_(-32768)
            self._side = torch.cuda.Stream(device)
            self._pending: torch.cuda.Event | None = None
            # The raw mixes the side stream reads: allocated on the current
            # stream, so they must outlive the side kernel.
            self._in_flight: torch.Tensor | None = None
            torch.accelerator.synchronize()
        dist.barrier(group=group)

    def supports(self, tokens: int) -> bool:
        return tokens in self.modes

    def join(self) -> None:
        """Makes the current stream wait for the last mixes kernel."""
        if self._pending is not None:
            torch.cuda.current_stream().wait_event(self._pending)
            self._pending = self._in_flight = None

    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the residual streams, post-mix, residual mix and layer input
        of `hc_fused_post_pre` applied to the TP sum of `x`.

        Args:
            x: [M, hidden] BF16 TP-partial sublayer output.
            residual: [M, hc, hidden] BF16 residual streams.
            post: [M, hc, 1] FP32 post-mix.
            comb: [M, hc, hc] FP32 residual mix, indexed [source, target].
            fn: [24, hc * hidden] BF16 prenorm GEMV weight.
            scale: [3] FP32 mix scales.
            base: [24] FP32 mix biases.
            norm_weight: [hidden] BF16 RMSNorm weight.

        """
        hidden = self.hidden_size
        m = x.shape[0]
        mode = self.modes[m]
        # Join the last mixes kernel before the publish (or the all-reduce), as
        # main does. Under graph capture the join becomes a programmatic edge
        # into the next PDL kernel, which may then start before the mixes kernel
        # has flushed its writes. The publish reads nothing before its wait, and
        # its wait orders the boundary kernel's early `post` / `comb` reads.
        self.join()
        if mode[0] == "reduced":
            source = to_cute_dynamic(
                tensor_model_parallel_all_reduce(x).flatten(), 16, divisibility=hidden
            )
        else:
            assert self._mailbox is not None
            self._publish(
                to_cute_dynamic(x.flatten(), 16, divisibility=hidden),
                to_cute(self._stage_state, 4),
                Int64(self._multicast),
                Int32(m),
                current_cu_stream(),
            )
            source = to_cute(self._mailbox.flatten(), 16)
        device = residual.device
        layer_input = torch.empty_like(x)
        residual_output = torch.empty_like(residual)
        mixes = torch.empty((m, MIX), dtype=torch.float32, device=device)
        post_output = torch.empty((m, HC, 1), dtype=torch.float32, device=device)
        comb_output = torch.empty((m, HC, HC), dtype=torch.float32, device=device)
        self._kernels[mode](
            source,
            to_cute_dynamic(residual.flatten(), 16, divisibility=HC * hidden),
            to_cute_dynamic(post.flatten(), 4, divisibility=HC),
            to_cute_dynamic(comb.flatten(), 4, divisibility=HC * HC),
            to_cute(fn.flatten(), 16),
            to_cute(scale, 4),
            to_cute(base, 4),
            to_cute(norm_weight, 16),
            to_cute_dynamic(residual_output.flatten(), 16, divisibility=HC * hidden),
            to_cute_dynamic(mixes.flatten(), 4, divisibility=MIX),
            to_cute_dynamic(layer_input.flatten(), 16, divisibility=hidden),
            to_cute(self._stage_state, 4),
            Int32(m),
            current_cu_stream(),
        )
        self._side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._side):
            self._mixes(
                to_cute_dynamic(mixes.flatten(), 4, divisibility=MIX),
                to_cute(scale, 4),
                to_cute(base, 4),
                to_cute_dynamic(post_output.flatten(), 4, divisibility=HC),
                to_cute_dynamic(comb_output.flatten(), 4, divisibility=HC * HC),
                Int32(m),
                current_cu_stream(),
            )
            self._pending = torch.cuda.Event()
            self._pending.record()
        self._in_flight = mixes
        return residual_output, post_output, comb_output, layer_input


class HopperDraftAllReduceNorm(AllReduceNorm):
    """Main's `AllReduceNorm` (all-reduce + residual add + RMSNorm, and the
    vocab-parallel embedding all-reduce + enorm / hnorm), whose kernels run
    unchanged on SM90 at any TP size. The Hopper MoE hands it finalized
    TP-partial outputs, so its fused MoE finalize goes unused.

    Its norm kernel reads the residual before its PDL wait, which holds under
    the same two premises as the boundary kernel's early reads: the first
    kernel after the previous writer waits before it triggers, and every
    kernel in the chain has at least one thread that executes the wait."""
