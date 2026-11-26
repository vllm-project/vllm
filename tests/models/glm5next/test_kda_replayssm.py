# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KDA ReplaySSM verify step over consecutive steps with partial acceptance
against vLLM's per-token-state KDA route (conv update, fused recurrence, gated
RMSNorm). Each arm carries its own state: ReplaySSM in the checkpoint and the
rings of a separate block, the stock route in one state block per token."""

import pytest
import torch
import torch.nn.functional as F

from vllm.models.glm5next.nvidia import kda_replayssm
from vllm.platforms import current_platform

D, LOWER_BOUND, EPS, STEPS = 128, -5.0, 1e-5, 8
L, R = kda_replayssm.WINDOW, kda_replayssm.RING_LEN

pytestmark = pytest.mark.skipif(
    current_platform.get_device_capability() not in ((9, 0), (10, 0), (10, 3)),
    reason="Requires SM90, SM100 or SM103",
)


class _Case:
    """Weights, the ReplaySSM pages (state block `slot[n]`, ring block
    `ring_slot[n]`) and the stock pool (`T` state blocks per request)."""

    def __init__(self, n, t, h, g):
        rand = lambda *shape, scale=1.0: (
            torch.randn(*shape, generator=g, device="cuda") * scale
        )
        self.n, self.t, self.h, self.g = n, t, h, g
        self.conv_weight = rand(3, 4, h, D, scale=0.5)
        self.a_log = (torch.rand(h, generator=g, device="cuda") * 15 + 1).log()
        self.dt_bias = rand(h, D, scale=0.5)
        self.norm = (1 + rand(D, scale=0.1)).bfloat16().float()
        self.f_b = rand(h, D, D, scale=D**-0.5).bfloat16()
        self.g_b = rand(h, D, D, scale=D**-0.5).bfloat16()
        conv0 = rand(n, t + 2, 3, h, D).bfloat16()
        state0 = rand(n, h, D, D, scale=0.05)

        conv_bytes, state_bytes = conv0[0].nbytes, state0[0].nbytes
        ring_bytes = h * R * D * (4 + 2 + 4)  # fp32 u, bf16 k, fp32 g
        page = max(conv_bytes + state_bytes, ring_bytes)
        pages = torch.empty(2 * n + 1, page, dtype=torch.int8, device="cuda")
        pages.view(torch.bfloat16).normal_(generator=g)
        self.conv = pages[:, :conv_bytes].view(torch.bfloat16).view(-1, *conv0[0].shape)
        self.state = (
            pages[:, conv_bytes : conv_bytes + state_bytes]
            .view(torch.float32)
            .view(-1, h, D, D)
        )
        self.rings = kda_replayssm.ring_views(pages, h, D)
        blocks = torch.randperm(2 * n, generator=g, device="cuda").int() + 1
        self.slot, self.ring_slot = blocks[:n], blocks[n:]
        self.conv[self.slot.long()] = conv0
        self.state[self.slot.long()] = state0

        self.idx = torch.arange(1, n * t + 1, device="cuda", dtype=torch.int32)
        self.idx = self.idx.view(n, t)
        self.stock_conv = conv0.new_zeros(n * t + 1, t + 2, 3 * h * D)
        self.stock_state = state0.new_zeros(n * t + 1, h, D, D)
        self.stock_conv[self.idx[:, 0].long()] = conv0.flatten(2)
        self.stock_state[self.idx[:, 0].long()] = state0

        # Cursors after prefill: empty rings at a random start.
        self.c = torch.zeros(n, dtype=torch.int32, device="cuda")
        self.r0 = torch.randint(0, R, (n,), generator=g, device="cuda").int()
        self.acc_prev = torch.ones(n, dtype=torch.int32, device="cuda")

    def draw(self):
        """One step's lengths (request 0 full, the rest 1..T), acceptances
        (request 0 all, so it flushes every few steps) and projection rows."""
        n, t, g = self.n, self.t, self.g
        q = torch.randint(1, t + 1, (n,), generator=g, device="cuda").int()
        q[0] = t
        acc = (torch.rand(n, generator=g, device="cuda") * q).int() + 1
        acc[0] = t
        qsl = F.pad(q.cumsum(0), (1, 0)).int()
        pitch = 3 * self.h * D + self.h + 2 * D
        proj = torch.randn(int(qsl[-1]), pitch, generator=g, device="cuda")
        return q, acc, qsl, proj.bfloat16()

    def split(self, proj):
        hd = self.h * D
        return (
            proj[:, : 3 * hd],
            proj[:, 3 * hd : 3 * hd + self.h],
            proj[:, 3 * hd + self.h : 3 * hd + self.h + D],
            proj[:, 3 * hd + self.h + D :],
        )

    def replayssm(self, qsl, proj):
        qkv, beta, f_a, g_a = self.split(proj)
        u_ring, k_ring, g_ring = self.rings
        shape, nan = (self.n * self.t, self.h, D), float("nan")
        out = torch.full(shape, nan, dtype=torch.bfloat16, device="cuda")
        kda_replayssm.kda_replayssm_step(
            qkv.unflatten(-1, (3, self.h, D)), beta, f_a, g_a, self.conv_weight,
            self.a_log, self.dt_bias, self.f_b, self.g_b, self.norm, qsl,
            self.slot, self.ring_slot, self.acc_prev, self.r0, self.c,
            self.conv, self.state, u_ring, k_ring, g_ring, out, LOWER_BOUND,
            D**-0.5, EPS,
        )  # fmt: skip
        return out

    def stock(self, qsl, proj):
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_update,
        )
        from vllm.models.glm5next.nvidia.ops.third_party.kda import (
            fused_recurrent_kda,
        )
        from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated

        qkv, beta, f_a, g_a = self.split(proj)
        h, m = self.h, proj.shape[0]
        weight = self.conv_weight.permute(0, 2, 3, 1).reshape(3 * h * D, 4)
        qkv = causal_conv1d_update(
            qkv.clone(),
            self.stock_conv.transpose(-1, -2),
            weight.contiguous(),
            None,
            activation="silu",
            conv_state_indices=self.idx[:, 0],
            num_accepted_tokens=self.acc_prev,
            query_start_loc=qsl,
            max_query_len=self.t,
        )
        query, key, value = (x.reshape(1, m, h, D) for x in qkv.split(h * D, -1))
        core = torch.empty(1, m, h, D, device="cuda", dtype=torch.bfloat16)
        fused_recurrent_kda(
            q=query,
            k=key,
            v=value,
            g=F.linear(f_a, self.f_b.view(h * D, D)).view(1, m, h, D),
            beta=beta.unsqueeze(0),
            initial_state=self.stock_state,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=qsl,
            ssm_state_indices=self.idx,
            num_accepted_tokens=self.acc_prev,
            out=core,
            sigmoid_beta=True,
            a_log=self.a_log.view(1, 1, h, 1),
            g_bias=self.dt_bias.view(-1),
            compute_gate=True,
            lower_bound=LOWER_BOUND,
        )
        gate = F.linear(g_a, self.g_b.view(h * D, D)).view(m, h, D)
        norm = self.norm.bfloat16()
        return rms_norm_gated(
            core[0], gate, norm, None, activation="sigmoid", eps=EPS
        ).view(m, h, D)

    def advance(self, q, acc):
        """The ReplaySSM tracker: a step with c + q > L flushes the committed
        records into the checkpoint and its records start the ring."""
        flush = self.c + q > L
        self.r0 = torch.where(flush, (self.r0 + self.c) % R, self.r0)
        self.c = torch.where(flush, acc, self.c + acc)
        self.acc_prev = acc

    def effective_state(self):
        """The checkpoint with the c committed ring records replayed onto it."""
        s = self.state[self.slot.long()].clone()
        u, k, g = (ring[self.ring_slot.long()].float() for ring in self.rings)
        n = torch.arange(self.n, device="cuda")
        for j in range(int(self.c.max())):
            row = (self.r0 + j) % R
            u_j, k_j, g_j = (x[n, :, row] for x in (u, k, g))
            step = s * g_j.exp()[:, :, None] + u_j[..., None] * k_j[:, :, None]
            s = torch.where((self.c > j).view(-1, 1, 1, 1), step, s)
        return s


@pytest.mark.parametrize("n", [1, 7, 256, 1024])
@pytest.mark.parametrize("heads", [8, 16], ids=["tp8", "tp4"])
@pytest.mark.parametrize("t", [6, 9])
def test_kda_replayssm_matches_per_token_state(t, heads, n):
    g = torch.Generator(device="cuda").manual_seed(1000 * t + heads + n)
    case = _Case(n, t, heads, g)
    for _ in range(STEPS):
        q, acc, qsl, proj = case.draw()
        m = proj.shape[0]
        expected = case.stock(qsl, proj)
        out = case.replayssm(qsl, proj)
        torch.testing.assert_close(out[:m], expected, atol=3e-2, rtol=1.6e-2)
        assert not out[m:].any()
        torch.testing.assert_close(
            case.conv[case.slot.long()].flatten(2),
            case.stock_conv[case.idx[:, 0].long()],
            atol=0,
            rtol=0,
        )
        case.advance(q, acc)
    requests = torch.arange(n, device="cuda")
    torch.testing.assert_close(
        case.effective_state(),
        case.stock_state[case.idx[requests, case.acc_prev - 1].long()],
        atol=3e-3,
        rtol=1e-2,
    )
