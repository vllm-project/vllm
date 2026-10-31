# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's fused KDA core (``KdaStage`` on Kimi K3's KdaSplit) at
32 and 64 tokens, plain decode and q = 4 verify, against an FP64 reference: its
output and recurrent-state error must not exceed the stock kernels' (conv update,
F_B GEMM, fused recurrence, gated RMSNorm), and its conv rows must match. With
g_b_proj folded in, the reference takes the FP64 gate and stock its BF16 GEMM."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

# One TP4 shard of GLM-5.3-Flash's KDA.
HEADS, HEAD_DIM = 16, 128
DIM = HEADS * HEAD_DIM
PITCH = 3 * DIM + HEADS + 2 * HEAD_DIM
LOWER_BOUND, EPS = -5.0, 1e-5

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="Requires SM100"
)


class _Case:
    """Weights, caches and inputs for ``requests`` x ``q`` tokens."""

    def __init__(self, requests, q):
        g = torch.Generator(device="cuda").manual_seed(requests * 8 + q)
        rand = lambda *shape, scale=1.0: (
            torch.randn(*shape, generator=g, device="cuda") * scale
        )
        self.requests, self.q, self.tokens = requests, q, requests * q
        self.conv_weight = [rand(DIM, 1, 4, scale=0.5) for _ in range(3)]
        self.a_log = (torch.rand(HEADS, generator=g, device="cuda") * 15 + 1).log()
        self.dt_bias = rand(DIM, scale=0.5)
        self.norm = (1 + rand(HEAD_DIM, scale=0.1)).bfloat16()
        self.f_b = rand(DIM, HEAD_DIM, scale=HEAD_DIM**-0.5).bfloat16()
        slots = self.tokens + 1  # slot 0 stays unused
        state = rand(slots, HEADS, HEAD_DIM, HEAD_DIM, scale=0.05)
        conv = rand(slots, 2 + q, 3 * DIM).bfloat16()  # SD layout, width 3 + q - 1
        self.caches = (conv, state)
        order = torch.randperm(slots - 1, generator=g, device="cuda").int() + 1
        self.slots = order[: self.tokens].view(requests, q).contiguous()
        self.accepted = torch.randint(
            1, q + 1, (requests,), generator=g, device="cuda", dtype=torch.int32
        )
        self.projected = rand(self.tokens, PITCH).bfloat16()
        self.g_b = rand(DIM, HEAD_DIM, scale=HEAD_DIM**-0.5).bfloat16()
        self.gate = F.linear(self.g_a(), self.g_b)

    def g_a(self):
        start = 3 * DIM + HEADS + HEAD_DIM
        return self.projected[:, start : start + HEAD_DIM]

    def clone_caches(self):
        return tuple(c.clone() for c in self.caches)

    def specialized(self, caches, fold):
        from vllm.models.glm5next.nvidia.specialized_decode.kda import KdaStage

        conv = lambda w: SimpleNamespace(weight=w)
        attention = SimpleNamespace(
            local_num_heads=HEADS,
            q_conv1d=conv(self.conv_weight[0]),
            k_conv1d=conv(self.conv_weight[1]),
            v_conv1d=conv(self.conv_weight[2]),
            o_norm=SimpleNamespace(weight=self.norm, eps=EPS),
            dt_bias=self.dt_bias,
            A_log=self.a_log.view(1, 1, HEADS, 1),
            kda_lower_bound=LOWER_BOUND,
            f_b_proj=SimpleNamespace(weight=self.f_b),
            kv_cache=caches,
            # Projection stand-ins: the stage's core output passes through o_proj.
            in_proj_qkvbfg_a=_Module(torch.empty(PITCH, 1), lambda x: self.projected),
            g_b_proj=_Module(self.g_b, lambda g_a: self.gate),
            o_proj=_Module(None, lambda out: out),
        )
        fold_tokens = {self.tokens} if fold else ()
        stage = KdaStage(attention, [self.tokens], self.q, fold_tokens=fold_tokens)
        stage.prepare_caches(caches)
        x = torch.empty(self.tokens, 1, device="cuda", dtype=torch.bfloat16)
        metadata = SimpleNamespace(
            non_spec_state_indices_tensor=self.slots.view(-1),
            spec_state_indices_tensor=self.slots,
            num_accepted_tokens=self.accepted,
        )
        return stage.run(x, metadata, self.q)

    def stock(self, caches):
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_update,
        )
        from vllm.models.glm5next.nvidia.ops.third_party.kda import (
            fused_recurrent_kda,
        )
        from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated

        conv, state = caches
        tokens, q = self.tokens, self.q
        weight = torch.cat(self.conv_weight).view(3 * DIM, 4)
        qkv = self.projected[:, : 3 * DIM].clone()  # the conv update runs in place
        starts = torch.arange(0, tokens + 1, q, device="cuda", dtype=torch.int32)
        if q == 1:
            slots, accepted = self.slots.view(-1), None
            qkv = causal_conv1d_update(
                qkv,
                conv.transpose(-1, -2),
                weight,
                None,
                activation="silu",
                conv_state_indices=slots,
            )
        else:
            slots, accepted = self.slots, self.accepted
            qkv = causal_conv1d_update(
                qkv,
                conv.transpose(-1, -2),
                weight,
                None,
                activation="silu",
                conv_state_indices=slots[:, 0],
                num_accepted_tokens=accepted,
                query_start_loc=starts,
                max_query_len=q,
            )
        shape = (1, tokens, HEADS, HEAD_DIM)
        query, key, value = (t.reshape(shape) for t in qkv.split(DIM, dim=-1))
        f_a = self.projected[:, 3 * DIM + HEADS : 3 * DIM + HEADS + HEAD_DIM]
        out = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        fused_recurrent_kda(
            q=query,
            k=key,
            v=value,
            g=F.linear(f_a, self.f_b).reshape(shape),
            beta=self.projected[:, 3 * DIM : 3 * DIM + HEADS].unsqueeze(0),
            initial_state=state,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=starts,
            ssm_state_indices=slots,
            num_accepted_tokens=accepted,
            out=out,
            sigmoid_beta=True,
            a_log=self.a_log,
            g_bias=self.dt_bias,
            compute_gate=True,
            lower_bound=LOWER_BOUND,
        )
        heads = (tokens, HEADS, HEAD_DIM)
        return rms_norm_gated(
            out.view(heads),
            self.gate.view(heads),
            self.norm,
            None,
            activation="sigmoid",
            eps=EPS,
        ).view(tokens, DIM)

    def reference(self, exact_gate=False):
        """FP64 outputs, the states written per slot and the conv rows per slot;
        ``exact_gate``: from the FP64 g_b_proj instead of the BF16 one."""
        gates = self.gate.double()
        if exact_gate:
            gates = self.g_a().double() @ self.g_b.double().T
        conv0, state0 = (c.double() for c in self.caches)
        x = self.projected.double()
        weight = torch.cat(self.conv_weight).double().view(3 * DIM, 4)
        f_b, decay_rate = self.f_b.double(), self.a_log.double().exp()[:, None]
        out = torch.empty(self.tokens, DIM, dtype=torch.float64, device="cuda")
        states, convs = {}, {}
        unit = lambda t: t / ((t * t).sum(-1, keepdim=True) + 1e-6).sqrt()
        for r in range(self.requests):
            offset = int(self.accepted[r]) - 1 if self.q > 1 else 0
            slots = self.slots[r].tolist()
            state = state0[slots[offset]].clone()
            window = conv0[slots[0], offset : offset + 3]
            for t in range(self.q):
                n = r * self.q + t
                window = torch.cat([window, x[n, None, : 3 * DIM]])[-4:]
                y = (window.T * weight).sum(-1)
                y = y * torch.sigmoid(y)
                query, key, value = (
                    y[i * DIM : (i + 1) * DIM].view(HEADS, -1) for i in range(3)
                )
                query, key = unit(query) * HEAD_DIM**-0.5, unit(key)
                beta = torch.sigmoid(x[n, 3 * DIM : 3 * DIM + HEADS])[:, None]
                f_a = x[n, 3 * DIM + HEADS : 3 * DIM + HEADS + HEAD_DIM]
                raw = (f_a @ f_b.T).view(HEADS, -1)
                gate = decay_rate * (raw + self.dt_bias.double().view(HEADS, -1))
                state = state * (LOWER_BOUND * torch.sigmoid(gate)).exp()[:, None]
                delta = value - torch.einsum("hvk,hk->hv", state, key)
                state = state + (delta * beta)[..., None] * key[:, None]
                o = torch.einsum("hvk,hk->hv", state, query)
                o = o / ((o * o).mean(-1, keepdim=True) + EPS).sqrt()
                gate_out = torch.sigmoid(gates[n].view(HEADS, -1))
                out[n] = (o * self.norm.double() * gate_out).view(-1)
                states[slots[t]] = state
            rows = x[r * self.q : (r + 1) * self.q, : 3 * DIM]
            convs[slots[0]] = torch.cat(
                [conv0[slots[0], offset + 1 : offset + 3], rows]
            )
        return out, states, convs


class _Module:
    def __init__(self, weight, fn):
        self.weight, self.fn = weight, fn

    def __call__(self, x):
        return (self.fn(x),)


def _errors(out, caches, reference):
    ref, states, _ = reference
    error = (out.double() - ref).abs().flatten()
    state = max(
        float((caches[1][s].double() - expected).abs().max())
        for s, expected in states.items()
    )
    quantiles = (error.median(), error.quantile(0.99), error.max())
    return [float(v) for v in quantiles] + [state]


@pytest.mark.parametrize("fold", [False, True], ids=["gate", "fold_g_b"])
@pytest.mark.parametrize(("requests", "q"), [(64, 1), (16, 4)])
def test_kda_core_no_worse_than_stock(requests, q, fold):
    case = _Case(requests, q)
    reference = case.reference(exact_gate=fold)
    ours_caches, stock_caches = case.clone_caches(), case.clone_caches()
    ours = _errors(case.specialized(ours_caches, fold), ours_caches, reference)
    stock = _errors(case.stock(stock_caches), stock_caches, reference)
    # Median, p99 and max output error and the max state error.
    assert all(a <= b for a, b in zip(ours, stock)), (ours, stock)
    for slot, rows in reference[2].items():
        torch.testing.assert_close(
            ours_caches[0][slot, : rows.shape[0]].double(), rows, atol=0, rtol=0
        )
