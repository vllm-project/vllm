# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA decode stage: the stock input projection and low-rank output-gate GEMMs,
then one fused kernel (conv update, L2 norm, F_B projection and decay, delta-rule
update, gated RMSNorm) in place of four, then the stock o_proj. At ``fold_tokens``
the fused kernel also runs g_b_proj (its gate warp, during the state update).

The fused kernel is a copy of Kimi K3's ``KdaSplit`` (``kda_split``). GLM-5.3-Flash
has the same KDA math (bounded sigmoid gate, sigmoid beta, conv width 4, head dim
128, BF16 conv and FP32 recurrent state) but packs its projection as q, k, v, b,
f_a, g_a and has a low-rank output gate (g_a -> g_b_proj) where K3 has a full-rank
one; the copy adds that gate.
"""

import cutlass
import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    layout,
    stream,
    to_cute,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.prepared import (
    decode_conv_weight,
)

from .dsa_attention import head_gemv
from .gemm import Projection
from .kda_split import executable

HEAD_DIM = 128


class KdaStage:
    def __init__(
        self,
        attention,
        batches,
        q,
        gemm=False,
        gate_tokens=frozenset(),
        fold_tokens=frozenset(),
    ):
        self.attention = a = attention
        # Skinny (in_proj, o_proj) GEMMs.
        self.projections: tuple[Projection, Projection] | None = None
        if gemm:
            self.projections = (
                Projection(a.in_proj_qkvbfg_a.weight),
                Projection(a.o_proj.weight),
            )
        self.heads = a.local_num_heads
        # The low-rank output gate's g_b_proj as a CuTe GEMV at gate_tokens
        # (policy kda_gate): its transposed weight, [1, rank, heads x 128].
        self.g_b = None
        self.gate_tokens = gate_tokens
        if gate_tokens and a.g_b_proj.weight.dtype == torch.bfloat16:
            self.g_b = a.g_b_proj.weight.detach().t().contiguous().unsqueeze(0)
        self.dim = self.heads * HEAD_DIM
        self.fold_tokens: frozenset[int] = frozenset()
        if a.g_b_proj.weight.dtype == torch.bfloat16:
            self.fold_tokens = frozenset(fold_tokens)
        conv = torch.cat(
            [m.weight.detach() for m in (a.q_conv1d, a.k_conv1d, a.v_conv1d)]
        )
        norm = a.o_norm.weight.detach().float()
        dt = a.dt_bias
        self.weights = (
            decode_conv_weight(conv),
            a.A_log.detach().view(-1),
            dt,
            norm,
            a.f_b_proj.weight,
        )
        self.scalars = tuple(
            map(cutlass.Float32, (a.kda_lower_bound, HEAD_DIM**-0.5, a.o_norm.eps))
        )
        self.device = dt.device
        self.conv_dim_first = is_conv_state_dim_first()
        # Policy keys: a batch, or (requests, tokens) for verify steps.
        self.keys: list = list(batches)
        self.keys += [(b // q, q) for b in batches if q > 1 and b % q == 0]
        self.kernels: dict = {}

    def _operands(self, caches, projected, gate, slots, out, accepted=None, g_b=None):
        conv, state = caches[:2]
        conv = conv if self.conv_dim_first else conv.transpose(-1, -2)
        w, a_log, dt, norm, fb = self.weights
        d, h = self.dim, self.heads
        p = lambda start, width: projected[:, start : start + width]
        # The kernel needs a bias tensor; dt_bias stands in (has_bias=False).
        ops = (p(0, 3 * d), w, dt, conv, p(3 * d + h, HEAD_DIM), p(3 * d, h), a_log)
        tail = (fb,) if accepted is None else (fb, accepted)
        tail += () if g_b is None else (g_b,)
        return ops + (dt, slots, state, out, gate, norm) + tail

    def prepare_caches(self, caches):
        pitch = self.attention.in_proj_qkvbfg_a.weight.shape[0]
        meta = dict(device="meta")
        for key in self.keys:
            requests, tokens = key if isinstance(key, tuple) else (key, 1)
            batch = requests * tokens
            projected = torch.empty((batch, pitch), dtype=torch.bfloat16, **meta)
            out = torch.empty(batch, self.dim, dtype=torch.bfloat16, **meta)
            gate = torch.empty_like(out)
            accepted = torch.empty(requests, dtype=torch.int32, **meta)
            shape = (requests, tokens) if tokens > 1 else (batch,)
            slots = torch.empty(shape, dtype=torch.int32, **meta)
            accepted = accepted if tokens > 1 else None
            fold = batch in self.fold_tokens
            g_b = self.attention.g_b_proj.weight if fold else None
            if fold:
                gate = self._g_a(projected)
            operands = self._operands(
                caches, projected, gate, slots, out, accepted, g_b
            )
            self.kernels[key] = executable(
                tuple(map(layout, operands)),
                self.heads,
                self.device.index,
                tokens,
                fold,
            )

    def _g_a(self, projected):
        start = 3 * self.dim + self.heads + HEAD_DIM
        return projected[:, start : start + HEAD_DIM]

    def run(self, x, metadata, q, reduce=True):
        """The attention output, TP-reduced unless ``reduce`` is False."""
        a = self.attention
        skinny = self.projections
        if skinny is not None and not skinny[0].supports(x.shape[0]):
            skinny = None
        projected = skinny[0](x) if skinny is not None else a.in_proj_qkvbfg_a(x)[0]
        d = self.dim
        g_a, g_b = self._g_a(projected), None
        if x.shape[0] in self.fold_tokens:
            gate, g_b = g_a, a.g_b_proj.weight
        elif self.g_b is not None and x.shape[0] in self.gate_tokens:
            gate = torch.empty((x.shape[0], d), dtype=x.dtype, device=x.device)
            head_gemv(g_a.unsqueeze(1), self.g_b, gate.unsqueeze(1))
        else:
            gate = a.g_b_proj(g_a)[0]
        out = torch.empty((x.shape[0], d), dtype=x.dtype, device=x.device)
        batch = x.shape[0]
        if q > 1:
            rows = batch // q
            slots = metadata.spec_state_indices_tensor[:rows]
            if slots.stride() != (q, 1):
                slots = slots.contiguous()
            key, accepted = (rows, q), metadata.num_accepted_tokens[:rows]
        else:
            slots = metadata.non_spec_state_indices_tensor[:batch]
            key, accepted = batch, None
        tensors = self._operands(a.kv_cache, projected, gate, slots, out, accepted, g_b)
        views = tuple(to_cute(t, 16) for t in tensors)
        self.kernels[key](*views[:13], *self.scalars, *views[13:], stream(self.device))
        if skinny is not None:
            partial = skinny[1](out)
        elif reduce:
            return a.o_proj(out)[0]
        else:
            partial = a.o_proj.quant_method.apply(a.o_proj, out)
        return tensor_model_parallel_all_reduce(partial) if reduce else partial
