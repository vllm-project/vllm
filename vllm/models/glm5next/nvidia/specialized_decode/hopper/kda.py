# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA layer of the Hopper (SM90) decode in three PDL-chained launches:

1. in_proj (q, k, v, b, f_a, g_a) as the warp split-K BF16 GEMM;
2. `KdaSplit` with the low-rank output gate folded in: conv update, F_B projection
   and decay, q/k L2 norm, delta-rule update, g_b projection and gated RMSNorm;
3. o_proj as the warp split-K GEMM, TP-partial for the boundary's all-reduce.

The conv and recurrent states are updated in place with stock's decode and MTP
verify semantics, so the stock layer resumes from them on any step this one skips.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.warp_splitk import (
    PreparedWarpSplitK,
    WarpSplitKTactic,
)

from ..cutedsl.host import fake, layout, stream, to_cute
from ..cutedsl.policy import batch_table
from ..kda_split import KdaSplit

HEAD_DIM = 128
HEADS = 16  # one TP4 rank of GLM-5.3-Flash
MAX_TOKENS = 32  # the warp split-K GEMM's row cap

# Warp split-K tactics (output tile, token tile, K tile, stages, B loader warps) by
# token tile. in_proj keeps 16-row output tiles: its 6416 rows do not split into
# 32-row tiles.
IN_PROJ_TACTICS = {
    8: WarpSplitKTactic(16, 8, 128, 6, 1),
    16: WarpSplitKTactic(16, 16, 256, 3, 2),
    32: WarpSplitKTactic(16, 32, 128, 3, 2),
}
O_PROJ_TACTICS = {
    8: WarpSplitKTactic(16, 8, 128, 16, 2),
    16: WarpSplitKTactic(32, 16, 256, 8, 2),
    32: WarpSplitKTactic(32, 32, 128, 12, 2),
}

# `KdaSplit` schedules ("early.lanes.occupancy.cluster") by request count; counts
# without an entry take the kernel defaults.
DECODE_SCHEDULE = batch_table("1-2:1.8.0.8,3-32:1.16.0.4")
VERIFY_SCHEDULE = batch_table("1:1.8.0.8,2:1.8.3.8,3-8:1.8.3.4")


def _conv_taps(weight):
    """The `[3 * D, 1, width]` conv parameter as the core's `[3, width, D]` operand,
    a strided view: the core loads the taps one by one."""
    return weight.detach().view(3, -1, weight.shape[-1]).transpose(1, 2)


def _token_tile(tokens: int) -> int:
    return 8 if tokens <= 8 else 16 if tokens <= 16 else 32


@cache
def _core(specs, heads, device, tokens, schedule):
    """The folded-gate `KdaSplit` for operand layouts `specs` (see `_operands`)."""
    args = tuple(fake(shape, stride, dtype) for shape, stride, dtype in specs)
    requests = specs[0][0][0] // tokens
    kernel = KdaSplit(heads, requests, 0, False, schedule, tokens, low_rank_gate=True)
    scalars = (cutlass.Float32(v) for v in (-5, HEAD_DIM**-0.5, 1e-6))
    with torch.accelerator.device_index(device):
        return cute.compile(
            kernel, *args[:13], *scalars, *args[13:], make_fake_stream()
        )


class HopperKdaLayer:
    """One KDA layer's decode for the planned token counts `batches` at `q` tokens
    per request (plain decode at q = 1, MTP verify above).

    Built from the stock `Glm5NextLinearAttention` after weight loading; reads its
    weights and caches in place. `prepare_caches` compiles the fused core for the
    bound cache layout before graph capture; `run` is only meant for CUDA graph
    capture, since it builds kernel views on the host every call."""

    def __init__(self, attention, batches, q):
        self.attention = a = attention
        self.heads = a.local_num_heads
        if self.heads != HEADS:
            raise ValueError(f"Hopper KDA decode expects {HEADS} heads per rank")
        self.dim = self.heads * HEAD_DIM
        weights = (a.in_proj_qkvbfg_a, a.f_b_proj, a.g_b_proj, a.o_proj)
        if any(m.weight.dtype != torch.bfloat16 for m in weights):
            raise ValueError("Hopper KDA decode expects BF16 KDA projections")
        if not 1 <= q <= 8:
            raise ValueError(f"Hopper KDA decode supports 1-8 tokens per request; {q}")
        # Keys: (requests, tokens per request).
        self.keys = sorted(
            {(b // q, q) for b in batches if b % q == 0 and b <= MAX_TOKENS}
        )
        if not self.keys:
            raise ValueError(f"No planned token count for q={q} up to {MAX_TOKENS}")
        counts = sorted({r * t for r, t in self.keys})
        self.in_proj = PreparedWarpSplitK(
            a.in_proj_qkvbfg_a.weight,
            {m: IN_PROJ_TACTICS[_token_tile(m)] for m in counts},
        )
        self.o_proj = PreparedWarpSplitK(
            a.o_proj.weight, {m: O_PROJ_TACTICS[_token_tile(m)] for m in counts}
        )
        self.pitch = a.in_proj_qkvbfg_a.weight.shape[0]
        self.hidden = a.o_proj.weight.shape[0]
        # Views of the parameters, never copies: weight writes land in place and
        # captured graphs keep reading them.
        self.weights = (
            _conv_taps(a.conv1d.weight),
            a.A_log.detach().view(-1),
            a.dt_bias,
            a.o_norm.weight.detach(),
            a.f_b_proj.weight,
            a.g_b_proj.weight,
        )
        self.scalars = tuple(
            map(cutlass.Float32, (a.kda_lower_bound, HEAD_DIM**-0.5, a.o_norm.eps))
        )
        self.device = a.dt_bias.device
        self.conv_dim_first = is_conv_state_dim_first()
        self.kernels: dict = {}
        # Resident intermediate and output buffers: the kernels release their PDL
        # dependents early, so capture-pool buffers could be reused while still
        # read. The executor consumes the output before the layer runs again.
        rows = max(counts)
        empty = lambda width: torch.empty(
            rows, width, dtype=torch.bfloat16, device=self.device
        )
        self.projected = empty(self.pitch)
        self.core_out = empty(self.dim)
        self.partial = empty(self.hidden)

    def supports(self, tokens: int, q: int) -> bool:
        return tokens % q == 0 and (tokens // q, q) in self.kernels

    def _schedule(self, requests, q):
        table = VERIFY_SCHEDULE if q > 1 else DECODE_SCHEDULE
        return table.get(requests, (0, 0, 0))

    def _operands(self, caches, projected, slots, out, accepted=None):
        conv, state = caches[:2]
        conv = conv if self.conv_dim_first else conv.transpose(-1, -2)
        w, a_log, dt, norm, f_b, g_b = self.weights
        d, h = self.dim, self.heads
        p = lambda start, width: projected[:, start : start + width]
        g_a = p(3 * d + h + HEAD_DIM, HEAD_DIM)
        # The kernel needs a bias tensor; dt_bias stands in (has_bias=False).
        ops = (p(0, 3 * d), w, dt, conv, p(3 * d + h, HEAD_DIM), p(3 * d, h), a_log)
        tail = (g_b,) if accepted is None else (accepted, g_b)
        return ops + (dt, slots, state, out, g_a, norm, f_b) + tail

    def prepare_caches(self, caches):
        """Compiles the fused core for every key against the bound caches."""
        meta = dict(device="meta")
        for requests, q in self.keys:
            tokens = requests * q
            projected = torch.empty(tokens, self.pitch, dtype=torch.bfloat16, **meta)
            out = torch.empty(tokens, self.dim, dtype=torch.bfloat16, **meta)
            if q > 1:
                slots = torch.empty(requests, q, dtype=torch.int32, **meta)
                accepted = torch.empty(requests, dtype=torch.int32, **meta)
            else:
                slots, accepted = torch.empty(tokens, dtype=torch.int32, **meta), None
            operands = self._operands(caches, projected, slots, out, accepted)
            self.kernels[(requests, q)] = _core(
                tuple(map(layout, operands)),
                self.heads,
                self.device.index,
                q,
                self._schedule(requests, q),
            )

    def run(self, x, metadata, q, reduce=False):
        """The layer's attention output (`[T, hidden]` BF16) for normed hidden
        states `x` (`[T, hidden]` BF16), TP-partial unless `reduce`."""
        tokens = x.shape[0]
        requests = tokens // q
        x = x.contiguous()
        projected = self.projected[:tokens]
        self.in_proj.run(x, projected)
        if q > 1:
            slots = metadata.spec_state_indices_tensor[:requests]
            if slots.stride() != (q, 1):
                slots = slots.contiguous()
            accepted = metadata.num_accepted_tokens[:requests]
        else:
            slots, accepted = metadata.non_spec_state_indices_tensor[:tokens], None
        out = self.core_out[:tokens]
        tensors = self._operands(
            self.attention.kv_cache, projected, slots, out, accepted
        )
        views = tuple(to_cute(t, 16) for t in tensors)
        kernel = self.kernels[(requests, q)]
        kernel(*views[:13], *self.scalars, *views[13:], stream(self.device))
        partial = self.partial[:tokens]
        self.o_proj.run(out, partial)
        return tensor_model_parallel_all_reduce(partial) if reduce else partial
