# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.xpress_head import XPressRefinerHead

V, H, B, R = 61, 32, 8, 16


def _head(seed: int = 0) -> XPressRefinerHead:
    torch.manual_seed(seed)
    head = XPressRefinerHead(
        vocab_size=V, hidden_size=H, block_size=B, rank=R, mlp_hidden=2 * R
    ).double()
    with torch.no_grad():
        for w in (
            head.w1.weight,
            head.w2.weight,
            head.down_h.weight,
            head.down_g.weight,
            head.in_proj.weight,
            head.mlp_gate.weight,
            head.mlp_up.weight,
            head.mlp_down.weight,
        ):
            w.normal_(0.0, 0.3)
    return head


def _inputs(n: int = 3, seed: int = 7):
    torch.manual_seed(seed)
    h = torch.randn(n, B, H, dtype=torch.float64)
    prev = torch.randint(0, V, (n, B))
    return h, prev


# Serving stores L*tril + I and drops the residual; training stores raw L and adds it.
# If these ever diverge, a checkpoint silently means something different at
# serving time.
def test_fold_matches_the_unfolded_sublayer():
    head = _head()
    torch.manual_seed(3)
    raw_l = torch.randn(R, B, B, dtype=torch.float64) * 0.3
    h, prev = _inputs()
    hcache = head.hidden_cache(h)

    tril = torch.tril(torch.ones(B, B, dtype=torch.float64))
    lat = head.w1(prev)
    x = head.in_proj(torch.cat([hcache, lat], dim=-1))
    mixed = torch.bmm((raw_l * tril), x.permute(2, 1, 0)).permute(2, 1, 0)
    x = x + mixed
    x = x + head.mlp_down(torch.nn.functional.silu(head.mlp_gate(x)) * head.mlp_up(x))
    expected = head.w2(x)

    head.fold_from_raw_(raw_l)
    torch.testing.assert_close(head.refine_bias(prev, hcache), expected)


# Jacobi iteration is only valid because of this: a settled prefix cannot be disturbed
# by later slots that are still changing.
def test_block_mixing_is_causal():
    head = _head()
    head.fold_from_raw_(torch.randn(R, B, B, dtype=torch.float64) * 0.3)
    h, prev = _inputs()
    hcache = head.hidden_cache(h)

    cut = 5
    perturbed = prev.clone()
    perturbed[:, cut] = (perturbed[:, cut] + 7) % V

    base = head.refine_bias(prev, hcache)
    other = head.refine_bias(perturbed, hcache)

    torch.testing.assert_close(other[:, :cut], base[:, :cut])
    assert not torch.allclose(other[:, cut:], base[:, cut:]), (
        "perturbing prev[cut] changed nothing at or after cut -- the mixer is "
        "not wired to the block at all"
    )


# Greedy refine has no sampling, so repeated calls must agree exactly.
def test_jacobi_is_deterministic():
    head = _head()
    head.fold_from_raw_(torch.randn(R, B, B, dtype=torch.float64) * 0.3)
    torch.manual_seed(13)
    base = torch.randn(2, B, V, dtype=torch.float64)
    h = torch.randn(2, B, H, dtype=torch.float64)
    anchor = torch.randint(0, V, (2,))
    tok_am1 = torch.randint(0, V, (2,))

    first = head.jacobi_refine_greedy(base, h, anchor, tok_am1, 4)
    second = head.jacobi_refine_greedy(base, h, anchor, tok_am1, 4)
    assert first.shape == (2, B - 1)
    assert torch.equal(first, second)
