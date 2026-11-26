# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KDA ReplaySSM prefix-cache snapshot (`replayssm_materializer`) against an
fp64 replay of the ring records and against the checkpoint the verify kernel
itself stores when it flushes."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.glm5next.nvidia import kda_replayssm
from vllm.platforms import current_platform

D, L, R = 128, kda_replayssm.WINDOW, kda_replayssm.RING_LEN


def _layer(n, heads, g, t=9):
    """A KDA layer's pages as bound: conv, checkpoint and the three rings, which
    share each page. Slots n + 1 .. 2n hold rings, the others states."""
    num_slots = 3 * n + 1
    conv_bytes = (t + 2) * 3 * heads * D * 2
    pages = torch.empty(
        num_slots, conv_bytes + heads * D * D * 4, dtype=torch.int8, device="cuda"
    )
    conv = pages[:, :conv_bytes].view(torch.bfloat16).view(-1, t + 2, 3, heads, D)
    state = pages[:, conv_bytes:].view(torch.float32).view(-1, heads, D, D)
    u, k, g_ring = kda_replayssm.ring_views(pages, heads, D)
    conv.copy_(torch.randn(conv.shape, generator=g, device="cuda"))
    state.normal_(generator=g).mul_(0.1)
    rings = slice(n + 1, 2 * n + 1)
    u[rings].normal_(generator=g)
    k[rings] = torch.randn(k[rings].shape, generator=g, device="cuda")
    g_ring[rings].uniform_(-5.0, 0.0, generator=g)
    return SimpleNamespace(kv_cache=(conv, state, u, k, g_ring))


def _replay(state, u, k, g, start, count):
    """fp64 reference: S <- S exp(g) + u k^T per record, `[H, V, K]`."""
    s = state.double()
    for j in range(count):
        row = (start + j) % R
        s = s * g[:, row].double().exp()[:, None] + (
            u[:, row].double()[:, :, None] * k[:, row].double()[:, None]
        )
    return s


@pytest.mark.parametrize("in_place", [False, True], ids=["copy", "in_place"])
def test_materialize_matches_fp64_replay(in_place):
    g = torch.Generator(device="cuda").manual_seed(int(in_place))
    num_layers, heads, max_num_reqs = 2, 16, 32
    flush = list(range(L + 9 + 1)) + [-1]  # every count a step can publish
    n = len(flush)
    layers = [_layer(n, heads, g) for _ in range(num_layers)]
    originals = [layer.kv_cache[1].clone() for layer in layers]

    cuda = dict(dtype=torch.int32, device="cuda")
    src = torch.randperm(n, generator=g, device="cuda").int() + 1
    ring = src + n
    dst = src if in_place else src + 2 * n
    start = torch.randint(0, R, (n,), generator=g, device="cuda").int()
    flush_count = torch.full((max_num_reqs,), -1, **cuda)
    flush_count[:n] = torch.tensor(flush, **cuda)
    ring_start = torch.zeros(max_num_reqs, **cuda)
    ring_start[:n] = start
    slots = []
    for column in (src, ring, dst):
        table = torch.zeros(num_layers, max_num_reqs, **cuda)
        table[:, :n] = column
        slots.append(table)
    active = torch.full((max_num_reqs,), -1, **cuda)
    active[: n - 1] = torch.arange(n - 1, **cuda)

    kda_replayssm.replayssm_materializer(layers)(
        *slots, ring_start, flush_count, active, n
    )

    for layer, original in zip(layers, originals):
        state, u, k, g_ring = layer.kv_cache[1:]
        expected = original.clone()
        for i in range(n - 1):
            expected[dst[i]] = _replay(
                original[src[i]], u[ring[i]], k[ring[i]], g_ring[ring[i]],
                int(start[i]), flush[i],
            ).float()  # fmt: skip
        torch.testing.assert_close(state, expected, atol=1e-5, rtol=1e-5)
        # The inactive request's slots are untouched.
        assert torch.equal(state[dst[-1]], original[dst[-1]])


@pytest.mark.skipif(
    current_platform.get_device_capability() not in ((9, 0), (10, 0), (10, 3)),
    reason="Requires SM90, SM100 or SM103",
)
@pytest.mark.parametrize("t", [6, 9])
def test_materialize_matches_kernel_flush(t):
    """Publishing the committed records gives the checkpoint the verify kernel
    stores when the next step's tokens overflow the window."""
    g = torch.Generator(device="cuda").manual_seed(t)
    n, heads = 4, 16
    layer = _layer(n, heads, g, t)
    conv, state, u, k, g_ring = layer.kv_cache
    rand = lambda *shape, scale=1.0: (
        torch.randn(*shape, generator=g, device="cuda") * scale
    )
    weights = (
        rand(3, 4, heads, D, scale=0.5),
        (torch.rand(heads, generator=g, device="cuda") * 15 + 1).log(),
        rand(heads, D, scale=0.5),
        rand(heads, D, D, scale=D**-0.5).bfloat16(),
        rand(heads, D, D, scale=D**-0.5).bfloat16(),
        (1 + rand(D, scale=0.1)).bfloat16().float(),
    )
    cuda = dict(dtype=torch.int32, device="cuda")
    slot = torch.arange(1, n + 1, **cuda)
    ring_slot, dst_slot = slot + n, slot + 2 * n
    start = torch.tensor([0, 7, 29, 31], **cuda)
    committed = torch.zeros(n, **cuda)
    accepted = torch.full((n,), t, **cuda)
    qsl = torch.arange(0, n * t + 1, t, **cuda)

    def step():
        proj = rand(n * t, 3 * heads * D + heads + 2 * D).bfloat16()
        qkv, beta, f_a, g_a = proj.split((3 * heads * D, heads, D, D), -1)
        out = torch.empty(n * t, heads, D, dtype=torch.bfloat16, device="cuda")
        conv_weight, a_log, dt_bias, f_b, g_b, norm = weights
        kda_replayssm.kda_replayssm_step(
            qkv.unflatten(-1, (3, heads, D)), beta, f_a, g_a, conv_weight, a_log,
            dt_bias, f_b, g_b, norm, qsl, slot, ring_slot, accepted, start,
            committed, conv, state, u, k, g_ring, out, -5.0, D**-0.5, 1e-5,
        )  # fmt: skip

    while int(committed[0]) + t <= L:
        step()
        committed += t
    kda_replayssm.replayssm_materializer([layer])(
        slot.view(1, n), ring_slot.view(1, n), dst_slot.view(1, n), start,
        committed, torch.arange(n, **cuda), n,
    )  # fmt: skip
    step()  # c + t > L: stores the c committed records into the checkpoint
    # The kernel applies the records with an f16 product (fp32 accumulate).
    torch.testing.assert_close(
        state[slot.long()], state[dst_slot.long()], atol=2e-3, rtol=1e-2
    )
