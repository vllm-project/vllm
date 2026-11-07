# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import pytest
import torch

from vllm.model_executor.models.qwen3_xpress import XPressRefinerHead
from vllm.utils.mem_utils import get_max_shared_memory_bytes
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.gpu.spec_decode.xpress import kernels

V, H, B, R = 61, 32, 8, 16


# Serving stores L*tril + I and drops the residual; training stores raw L and adds it.
# If these ever diverge, a checkpoint silently means something different at
# serving time.
def test_fold_matches_the_unfolded_sublayer():
    head = _head()
    torch.manual_seed(3)
    raw_l = torch.randn(R, B, B, dtype=torch.float64) * 0.3
    h, prev = _inputs()
    hcache = head.hidden_cache(h)

    expected = _unfolded_refine_bias(head, raw_l, prev, hcache)

    head.fold_from_raw_(raw_l)
    actual = head.refine_bias(prev, hcache)

    torch.testing.assert_close(actual, expected)


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


# A weight update reruns the post-load hook: the fused copies take the new weights
# and keep their storage, which captured CUDA graphs read.
def test_fused_buffers_refilled_in_place():
    head = _head()
    head.fold_from_raw_(torch.randn(R, B, B, dtype=torch.float64) * 0.3)
    head.process_weights_after_loading()
    buf = head._fused_buf
    assert buf is not None
    ptrs = {k: v.data_ptr() for k, v in buf.items()}

    with torch.no_grad():
        head.mlp_down.weight.mul_(-2)
    head.fold_from_raw_(torch.randn(R, B, B, dtype=torch.float64) * 0.3)
    head.process_weights_after_loading()

    assert head._fused_buf is buf
    assert {k: v.data_ptr() for k, v in buf.items()} == ptrs
    torch.testing.assert_close(buf["wd_t"], head.mlp_down.weight.t())
    torch.testing.assert_close(buf["mix_kjc"], head.mix_L.permute(1, 2, 0))


# Greedy refine has no sampling, so repeated calls must agree exactly, also when
# one returns the refined logits the online acceptance estimator reads. Those
# must be the logits the draft was picked from. The fused kernels are the only
# implementation, so this runs where they do: on a GPU, at the serving dtype.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("topc", [0, 16])
def test_jacobi_is_deterministic(topc: int):
    dtype = torch.bfloat16
    head = _head(device="cuda", dtype=dtype)
    head.topc = topc
    head.fold_from_raw_(torch.randn(R, B, B, dtype=dtype, device="cuda") * 0.3)
    head.process_weights_after_loading()
    torch.manual_seed(13)
    base = torch.randn(2, B, V, dtype=dtype, device="cuda")
    h = torch.randn(2, B, H, dtype=dtype, device="cuda")
    anchor = torch.randint(0, V, (2,), device="cuda")
    tok_am1 = torch.randint(0, V, (2,), device="cuda")

    first, no_logits = head.jacobi_refine_greedy(base, h, anchor, tok_am1, 4)
    second, logits = head.jacobi_refine_greedy(
        base, h, anchor, tok_am1, 4, return_logits=True
    )
    assert first.shape == (2, B - 1)
    assert torch.equal(first, second)
    assert no_logits is None
    assert logits is not None and logits.shape == (2 * (B - 1), topc or V)
    if not topc:
        picked = logits.gather(1, first.reshape(-1, 1)).squeeze(1)
        torch.testing.assert_close(picked, logits.max(dim=-1).values)


# Every launch config of the latent pass must compute the same latent, at the
# Qwen3-8B-XPress-b16 shapes the tile sizes were chosen for. Each pipeline stage of
# the in-kernel MLP holds its three [R, HT] weight tiles in shared memory (plus one
# more as headroom); per stage count, the largest HT that fits this device runs.
_RANK, _MLP_HIDDEN, _BLOCK = 256, 512, 16


def _fused_mlp_tiles() -> list[tuple[int, int]]:
    if not torch.cuda.is_available():
        return []
    smem = get_max_shared_memory_bytes()
    tiles = []
    for stages in (1, 2, 3):
        fit = [ht for ht in (64, 32, 16) if (3 * stages + 1) * _RANK * ht * 2 <= smem]
        tiles += [(fit[0], stages)] if fit else []
    return tiles


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("kc", [None, _RANK // 2, _RANK // 4])
@pytest.mark.parametrize(
    "do_mlp,ht,num_stages",
    [(None, 64, None), (False, 64, 1)]
    + [(True, ht, s) for ht, s in _fused_mlp_tiles()],
)
def test_latent_pass_matches_reference(
    do_mlp: bool | None, kc: int | None, ht: int, num_stages: int | None
):
    set_random_seed(0)
    n, v = 5, 1000
    head = XPressRefinerHead(
        vocab_size=v,
        hidden_size=64,
        block_size=_BLOCK,
        rank=_RANK,
        mlp_hidden=_MLP_HIDDEN,
    ).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        # At the default init the MLP adds only ~5% to its input, below what the
        # tolerance can tell apart; doubling its weights makes it ~45%.
        for w in (head.mlp_gate.weight, head.mlp_up.weight, head.mlp_down.weight):
            w.mul_(2)
    head.fold_from_raw_(torch.randn(_RANK, _BLOCK, _BLOCK, device="cuda") * 0.3)
    head.process_weights_after_loading()
    ref_head = copy.deepcopy(head).float()
    buf = head._fused_buf
    blk = torch.randint(0, v, (n, _BLOCK), device="cuda")
    tok_am1 = torch.randint(0, v, (n,), device="cuda")
    hcache = ref_head.hidden_cache(torch.randn(n, _BLOCK, 64, device="cuda"))
    prev = torch.cat([tok_am1[:, None], blk[:, :-1]], dim=1)
    expected = ref_head._refine_latent(prev, hcache)[:, 1:]

    xh = (hcache @ ref_head.in_proj.weight[:, : 2 * _RANK].t()).bfloat16()
    lat = torch.empty(n, _BLOCK - 1, _RANK, dtype=torch.bfloat16, device="cuda")
    kernels.xpress_latent_pass(
        blk,
        tok_am1,
        xh,
        lat,
        head.w1.weight,
        buf["wlat_t"],
        buf["mix_kjc"],
        buf["wg_t"],
        buf["wu_t"],
        buf["wd_t"],
        torch.cat([buf["wg_t"], buf["wu_t"]], dim=1),
        do_mlp=do_mlp,
        kc=kc,
        ht=ht,
        num_stages=num_stages,
    )

    # The latent is stored in bf16 and stays below 4, where one bf16 ulp is 2**-6.
    torch.testing.assert_close(lat.float(), expected, atol=2**-6, rtol=1.6e-2)


def _head(
    seed: int = 0, device: str = "cpu", dtype: torch.dtype = torch.float64
) -> XPressRefinerHead:
    torch.manual_seed(seed)
    head = XPressRefinerHead(
        vocab_size=V, hidden_size=H, block_size=B, rank=R, mlp_hidden=2 * R
    ).to(device=device, dtype=dtype)
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


def _unfolded_refine_bias(head, raw_l, prev, hcache):
    """refine_bias written out in the training-time form.

    The mixer is applied as a residual around the raw, strictly causal L, which
    is what a training checkpoint stores. Everything else reuses the head's own
    submodules: the only thing under test is the fold.
    """
    tril = torch.tril(torch.ones(B, B, dtype=torch.float64))
    lat = head.w1(prev)
    x = head.in_proj(torch.cat([hcache, lat], dim=-1))
    mixed = torch.bmm((raw_l * tril), x.permute(2, 1, 0)).permute(2, 1, 0)
    x = x + mixed
    x = x + head.mlp_down(torch.nn.functional.silu(head.mlp_gate(x)) * head.mlp_up(x))
    return head.w2(x)
