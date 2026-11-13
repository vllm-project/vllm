# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's MonoKernels on MI355X against the ops vLLM runs for the same
regular-decode or spec-verify step, with random checkpoint-format weights:

- K1 (one GPU): AttnRes -> in_proj -> f_b -> conv -> KDA recurrence -> gated
  norm, the conv / SSM state updates (states in vLLM's padded, strided pages)
  and a block-write layer's block store; against the Triton chain and against
  the KDA decode KimiK3DeltaAttention dispatches (AITER's fused decode). A
  graph replay's padded requests write nothing and leave the rest bit-identical.
- K2 (eight GPUs): o_proj + all-reduce + the MLP AttnRes + the latent MoE. The
  MoE is compared on the kernel's own MLP input, where routing cannot differ.
  The route on K2's own logits is KimiMoE's router's (ids and weight bits, near
  and exact ties), the shared SiTU the _C kernel's at every magnitude, which
  must also be what vLLM's compiled SituAndMul gives. Then a CUDA graph of K2
  replayed 300 times, bit-identical, and a stalled rank (the fault hook): every
  rank's launch finishes and the output rank's flags fail the step.
- M1a (AttnRes -> qkv_a / g_proj -> q / kv RMSNorm + the fp8 KV write -> q_b
  -> fp4 W_UK -> the DCP query gather) and M1b (the a2a LSE combine -> fp4
  W_UV -> the gate): one GPU, then TP8 + DCP8 with a graph of two MLA layers
  (M1a -> M1b -> K2) replayed. The KV write stores only its slots' rows; its
  fp8 codes and the q RMSNorm are AITER's on M1a's own intermediates; pages
  past 2 and 4 GiB (M1c's 32- and 64-bit addressing too) read and write as low
  ones.
"""

import pytest
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange

from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)
from vllm.platforms import current_platform

bf = torch.bfloat16
COS_MIN = 0.9999


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


needs_gfx950 = pytest.mark.skipif(
    not _on_gfx950(), reason="the K3 mono kernels need gfx950 (MI355X)"
)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return F.cosine_similarity(a, b, dim=0).item()


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def _randn(g, *shape, scale=1.0, dtype=bf):
    return (torch.randn(*shape, generator=g, device=g.device) * scale).to(dtype)


def _pair_words(buf: torch.Tensor, off: int, n: int) -> torch.Tensor:
    """The value words (int32) of the `n` (value, tag) mailbox pairs from byte
    `off` of the uint8 `buf`."""
    return buf[off : off + n * 8].view(torch.int32).view(n, 2)[:, 0].contiguous()


def _strided_clone(t: torch.Tensor) -> torch.Tensor:
    """`t` copied at its own strides: `clone` drops a padded page's gaps."""
    out = torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, device=t.device)
    return out.copy_(t)


def _aiter_env(monkeypatch) -> None:
    """The AITER switches the MI355X recipe serves with (`runtime.unsupported`
    requires them)."""
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    rocm_aiter_ops.refresh_env_variables()


@pytest.fixture
def aiter_env(monkeypatch):
    from vllm._aiter_ops import rocm_aiter_ops

    _aiter_env(monkeypatch)
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


# ---------------------------------------------------------------- K1


def _kda_inputs(nb: int, nacc: int, dim_first: bool, S: int, L: int):
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        HD,
        HIDDEN,
        NH,
        NPROJ,
        PROJ,
        QKV,
    )

    g = torch.Generator(device="cuda").manual_seed(0)
    R = S // L
    slots = S + 4
    sl = 3 + (L - 1)
    w = {
        "prefix": _randn(g, S, HIDDEN, scale=2.0),
        "delta": _randn(g, S, HIDDEN, scale=0.5),
        "blocks": _randn(g, S, 8, HIDDEN, scale=2.0),
        "ares_nw": (1 + _randn(g, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "ares_qk": _randn(g, HIDDEN, scale=0.05),
        "in_nw": (1 + _randn(g, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "w_in": _randn(g, NPROJ, HIDDEN, scale=0.02),
        "w_fb": _randn(g, PROJ, HD, scale=0.08),
        "conv_w": _randn(g, QKV, 4, scale=0.4, dtype=torch.float32),
        "a_log": _randn(g, NH, scale=0.5, dtype=torch.float32),
        "dt_bias": _randn(g, PROJ, scale=0.5, dtype=torch.float32),
        "on_w": (1 + _randn(g, HD, scale=0.1, dtype=torch.float32)).to(bf),
    }
    w["w_in"][NPROJ - 4 :].zero_()
    # vLLM packs a layer's conv and SSM state in one padded page: strided slots
    pad = _randn(g, slots, NH * HD * HD + 24576, scale=0.05, dtype=torch.float32)
    w["rstate"] = pad[:, : NH * HD * HD].view(slots, NH, HD, HD)
    conv = _randn(g, slots, QKV * sl + 4096)[:, : QKV * sl]
    if dim_first:
        w["conv_state"] = conv.view(slots, QKV, sl)
    else:
        w["conv_state"] = conv.view(slots, sl, QKV).transpose(-1, -2)
    w["st_idx"] = torch.arange(3, 3 + S, dtype=torch.int32, device="cuda").view(R, L)
    # the requests' accepted counts differ, from nacc on
    w["num_acc"] = torch.tensor(
        [1 + (nacc - 1 + r) % L for r in range(R)], dtype=torch.int32, device="cuda"
    )
    return w


def _kda_attn_res(w, nb: int, write_idx: int):
    """AttnRes as vLLM runs it, on copies of `w`'s prefix and blocks: (prefix,
    blocks, x)."""
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

    prefix, blocks = w["prefix"].clone(), w["blocks"].clone()
    x = attn_res(
        prefix,
        w["delta"],
        blocks,
        w["ares_nw"],
        w["ares_qk"],
        w["in_nw"],
        nb,
        write_idx,
        1e-5,
        1e-5,
    )
    return prefix, blocks, x


def _kda_reference(w, nb: int, S: int, L: int, write_idx: int):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.models.kimi_k3.amd.ops.third_party.kda import fused_recurrent_kda
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        HD,
        NH,
        PROJ,
        QKV,
    )
    from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

    prefix, blocks, x = _kda_attn_res(w, nb, write_idx)
    proj = F.linear(x, w["w_in"])
    mixed_qkv, g2, f_a, beta = proj.split([QKV, PROJ, HD, NH + 4], dim=-1)[:4]
    g1 = F.linear(f_a, w["w_fb"])
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    qsl = torch.arange(0, S + 1, L, dtype=torch.int32, device="cuda")
    conv = causal_conv1d_update(
        mixed_qkv.contiguous(),
        conv_state,
        w["conv_w"],
        None,
        activation="silu",
        conv_state_indices=w["st_idx"][:, 0].contiguous(),
        num_accepted_tokens=w["num_acc"],
        query_start_loc=qsl,
        max_query_len=L,
        validate_data=False,
        out=torch.empty_like(mixed_qkv),
    )
    q, k, v = (
        rearrange(t, "n (h d) -> 1 n h d", d=HD) for t in conv.split(PROJ, dim=-1)
    )
    core = torch.empty(1, S, NH, HD, dtype=bf, device="cuda")
    fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        raw_g=rearrange(g1, "n (h d) -> 1 n h d", d=HD),
        raw_beta=beta[:, :NH].unsqueeze(0),
        A_log=w["a_log"],
        dt_bias=w["dt_bias"],
        lower_bound=-5.0,
        initial_state=rstate,
        cu_seqlens=qsl,
        ssm_state_indices=w["st_idx"],
        num_accepted_tokens=w["num_acc"],
        uniform_sequence_length=L,
        out=core,
    )
    with set_current_vllm_config(VllmConfig()):
        norm = FusedRMSNormGated(HD, activation="sigmoid").cuda()
    norm.weight.data = w["on_w"].clone()
    out = norm.forward_cuda(core, rearrange(g2, "n (h d) -> 1 n h d", d=HD))
    return {
        "prefix": prefix,
        "blocks": blocks,
        "conv_state": conv_state,
        "rstate": rstate,
        "core": out.reshape(S, PROJ),
    }


def _kda_key(w, nb: int, S: int, L: int, write_idx: int):
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        KdaPreBuild,
    )

    return KdaPreBuild(
        tokens=S,
        qlen=L,
        nblocks=nb,
        delta=True,
        state_len=w["conv_state"].size(-1),
        write_idx=write_idx,
    )


def _kda_launch(key, w, st_idx, num_acc, core):
    """One K1 launch on copies of `w`'s prefix, blocks and (strided) states;
    returns them after it."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        kda_pre,
        scratch_bytes,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import HEALTH_WORDS

    prefix, blocks = w["prefix"].clone(), w["blocks"].clone()
    conv_state = _strided_clone(w["conv_state"])
    rstate = _strided_clone(w["rstate"])
    health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device="cuda")
    flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    kda_pre(
        key,
        prefix=prefix,
        delta=w["delta"],
        blocks=blocks,
        ares_nw=w["ares_nw"],
        ares_qk=w["ares_qk"],
        in_nw=w["in_nw"],
        w_in=w["w_in"],
        w_fb=w["w_fb"],
        conv_w=w["conv_w"],
        conv_state=conv_state,
        a_log=w["a_log"],
        dt_bias=w["dt_bias"],
        on_w=w["on_w"],
        rstate=rstate,
        st_idx=st_idx,
        num_acc=num_acc,
        core_out=core,
        scratch=torch.zeros(scratch_bytes(key), dtype=torch.uint8, device="cuda"),
        layer=0,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        health=health,
        alarm=flags.data_ptr(),
    )
    torch.accelerator.synchronize()
    assert not health.any() and not flags.any()
    return prefix, blocks, conv_state, rstate


# (rows, tokens a request): served regular-decode widths, then speculative
# verification and larger kernel-capacity shapes.
K1_SHAPES = [
    (1, 1),
    (4, 1),
    (8, 1),
    (16, 1),
    (8, 8),
    (16, 8),
    (16, 4),
    (16, 16),
    (4, 4),
    (32, 8),
    (64, 4),
    (64, 16),
]


@needs_gfx950
@pytest.mark.parametrize("S, L", K1_SHAPES)
@pytest.mark.parametrize("dim_first", [True, False])
@pytest.mark.parametrize("nacc", [1, 3])
@pytest.mark.parametrize("write", [False, True])
def test_k1_matches_vllm_ops(
    S: int, L: int, dim_first: bool, nacc: int, write: bool
) -> None:
    """K1 against attn_res -> in_proj -> f_b -> causal_conv1d_update ->
    fused_recurrent_kda -> FusedRMSNormGated on one regular-decode or spec-verify
    step (S / L requests of L tokens), including the state each step leaves
    behind."""
    if L == 1 and nacc != 1:
        pytest.skip("at L = 1 every request accepts 1 token: the nacc = 1 case")
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import PROJ

    nb = 4
    # a block-write layer appends block nb (block_write_idx == prev_valid_blocks)
    write_idx = nb if write else -1
    w = _kda_inputs(nb, nacc, dim_first, S, L)
    ref = _kda_reference(w, nb, S, L, write_idx)

    core = torch.zeros(S, PROJ, dtype=bf, device="cuda")
    prefix, blocks, conv_state, rstate = _kda_launch(
        _kda_key(w, nb, S, L, write_idx), w, w["st_idx"], w["num_acc"], core
    )

    torch.testing.assert_close(prefix, ref["prefix"], atol=0, rtol=0)
    torch.testing.assert_close(blocks, ref["blocks"], atol=0, rtol=0)
    # the conv output feeds the recurrence in fp32 here, bf16 in the reference
    errs = {
        "conv_state": _rel_l2(conv_state, ref["conv_state"]),
        "rstate": _rel_l2(rstate, ref["rstate"]),
        "core": _rel_l2(core, ref["core"]),
    }
    assert errs["conv_state"] < 1e-3 and errs["rstate"] < 1e-3, errs
    assert _cos(core, ref["core"]) > COS_MIN, errs


# (rows, tokens a request, padded requests): trailing pads as a graph replay
# has them, a middle one, and a step of pads only
K1_PAD_CASES = [
    (4, 1, [3]),
    (16, 1, [5, 13, 14, 15]),
    (16, 4, [1, 3]),
    (16, 8, [1]),
    (8, 8, [0]),
]


@needs_gfx950
@pytest.mark.parametrize("S, L, pads", K1_PAD_CASES)
@pytest.mark.parametrize("dim_first", [True, False])
@pytest.mark.parametrize("write", [False, True])
def test_k1_padded_rows(
    S: int, L: int, pads: list[int], dim_first: bool, write: bool
) -> None:
    """A CUDA graph replay's padded requests (state slot NULL_BLOCK_ID, one
    accepted token, as the GDN metadata builder pads them): their core rows are
    zero, no conv or SSM state is written (the null slot's included), and the
    other rows, the prefix and the blocks are bit-identical to the step
    without pads."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import PROJ
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    nb, R = 4, S // L
    w = _kda_inputs(nb, 3, dim_first, S, L)
    key = _kda_key(w, nb, S, L, nb if write else -1)
    st_idx, num_acc = w["st_idx"].clone(), w["num_acc"].clone()
    st_idx[pads] = NULL_BLOCK_ID
    num_acc[pads] = 1

    def run(idx, acc):
        # ones: a padded row's zeros must be the kernel's stores
        core = torch.ones(S, PROJ, dtype=bf, device="cuda")
        return (*_kda_launch(key, w, idx, acc, core), core)

    prefix0, blocks0, conv0, rstate0, core0 = run(w["st_idx"], w["num_acc"])
    prefix, blocks, conv, rstate, core = run(st_idx, num_acc)

    rows, want = core.view(R, L, PROJ), core0.view(R, L, PROJ)
    keep = [r for r in range(R) if r not in pads]
    assert not rows[pads].any()
    torch.testing.assert_close(rows[keep], want[keep], atol=0, rtol=0)
    torch.testing.assert_close(prefix, prefix0, atol=0, rtol=0)
    torch.testing.assert_close(blocks, blocks0, atol=0, rtol=0)
    # the padded requests' own slots and the null slot keep their inputs
    own = [NULL_BLOCK_ID, *w["st_idx"][pads].flatten().tolist()]
    conv0[own], rstate0[own] = w["conv_state"][own], w["rstate"][own]
    torch.testing.assert_close(conv, conv0, atol=0, rtol=0)
    torch.testing.assert_close(rstate, rstate0, atol=0, rtol=0)


def _production_kda(monkeypatch, w, x, S: int, L: int, dim_first: bool):
    """The KDA decode KimiK3DeltaAttention._forward dispatches for this step: a
    real layer (BF16 conv / FP32 SSM state, as the recipe serves) with `w`'s
    weights and copies of its states, on the projections of the AttnRes output
    `x`. Returns the layer, its core output and the states after it."""
    from tests.v1.attention.utils import create_vllm_config
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.models.kimi_k3.amd import kda
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        HD,
        HIDDEN,
        NH,
        PROJ,
        QKV,
    )
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    monkeypatch.setattr(kda, "is_conv_state_dim_first", lambda: dim_first)
    vllm_config = create_vllm_config(model_name="Qwen/Qwen3.5-0.8B", dtype=bf)
    vllm_config.cache_config.mamba_ssm_cache_dtype = "float32"
    config = KimiLinearConfig(
        hidden_size=HIDDEN,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "head_dim": HD,
            "num_heads": NH,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(bf),
        torch.device("cuda"),
    ):
        layer = kda.KimiK3DeltaAttention(
            config, vllm_config, prefix="model.layers.0.self_attn"
        )
    layer.conv1d.weight.copy_(w["conv_w"].unsqueeze(1))
    layer.A_log.copy_(w["a_log"])
    layer.dt_bias.copy_(w["dt_bias"])
    layer.o_norm.weight.copy_(w["on_w"])
    layer.process_weights_after_loading()
    conv_state, rstate = _strided_clone(w["conv_state"]), _strided_clone(w["rstate"])
    # the cache as allocated: DS (dim, state_len) or SD (state_len, dim) a slot
    layer.kv_cache = (conv_state if dim_first else conv_state.transpose(1, 2), rstate)

    R, dev = S // L, x.device
    qsl = torch.arange(0, S + 1, L, dtype=torch.int32, device=dev)
    steps = dict(num_prefills=0, num_prefill_tokens=0, num_actual_tokens=S)
    if L == 1:
        m = GDNAttentionMetadata(
            **steps, num_decodes=S, num_decode_tokens=S, num_spec_decodes=0,
            num_spec_decode_tokens=0, non_spec_query_start_loc=qsl,
            non_spec_state_indices_tensor=w["st_idx"][:, 0].contiguous(),
            has_initial_state=torch.ones(S, dtype=torch.bool, device=dev),
        )  # fmt: skip
    else:
        m = GDNAttentionMetadata(
            **steps, num_decodes=0, num_decode_tokens=0, num_spec_decodes=R,
            num_spec_decode_tokens=S, spec_query_start_loc=qsl,
            spec_state_indices_tensor=w["st_idx"],
            spec_sequence_masks=torch.ones(R, dtype=torch.bool, device=dev),
            num_accepted_tokens=w["num_acc"], uniform_spec_sequence_length=L,
        )  # fmt: skip
    proj = F.linear(x, w["w_in"])
    mixed_qkv, g2, f_a, beta = proj.split([QKV, PROJ, HD, NH, 4], dim=-1)[:4]
    core = torch.zeros(1, S, NH, HD, dtype=bf, device=dev)
    with set_forward_context({layer.prefix: m}, vllm_config):
        layer._forward(
            mixed_qkv=mixed_qkv,
            g1=rearrange(F.linear(f_a, w["w_fb"]), "n (h d) -> 1 n h d", d=HD),
            g2=rearrange(g2, "n (h d) -> n h d", d=HD),
            beta=beta.unsqueeze(0),
            core_attn_out=core,
        )
    torch.accelerator.synchronize()
    return layer, core.view(S, PROJ), conv_state, rstate


@needs_gfx950
@pytest.mark.parametrize("S, L", [(1, 1), (8, 1), (16, 1), (4, 4), (8, 8), (16, 8)])
@pytest.mark.parametrize("dim_first", [True, False])
@pytest.mark.parametrize("nacc", [1, 3])
@torch.inference_mode()
def test_k1_matches_production_kda(
    dist_init, aiter_env, monkeypatch, S: int, L: int, dim_first: bool, nacc: int
) -> None:
    """K1 against the KDA decode vLLM serves MI355X with: KimiK3DeltaAttention's
    own dispatch (AITER's fused Gluon decode, regular or spec-verify) on the same
    step's projections: the conv / SSM state slots each writes, their contents
    and the outputs."""
    if L == 1 and nacc != 1:
        pytest.skip("at L = 1 every request accepts 1 token: the nacc = 1 case")
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import PROJ

    nb, write_idx = 4, -1
    w = _kda_inputs(nb, nacc, dim_first, S, L)
    _, _, x = _kda_attn_res(w, nb, write_idx)
    layer, want, want_conv, want_rstate = _production_kda(
        monkeypatch, w, x, S, L, dim_first
    )
    assert layer.use_aiter_decode, "MI355X serves KDA decode on AITER's Gluon kernel"

    core = torch.zeros(S, PROJ, dtype=bf, device="cuda")
    _, _, conv_state, rstate = _kda_launch(
        _kda_key(w, nb, S, L, write_idx), w, w["st_idx"], w["num_acc"], core
    )

    def written(states, before):
        return (states != before).flatten(1).any(1)

    conv_w = written(want_conv, w["conv_state"])
    rstate_w = written(want_rstate, w["rstate"])
    assert torch.equal(written(conv_state, w["conv_state"]), conv_w)
    assert torch.equal(written(rstate, w["rstate"]), rstate_w)
    errs = {
        "conv_state": _rel_l2(conv_state[conv_w], want_conv[conv_w]),
        "rstate": _rel_l2(rstate[rstate_w], want_rstate[rstate_w]),
        "core": _rel_l2(core, want),
    }
    # AITER's SSM states are themselves ~2e-3 off the Triton chain's
    assert errs["conv_state"] < 1e-3 and errs["rstate"] < 4e-3, errs
    assert _cos(core, want) > COS_MIN, errs


# ---------------------------------------------------------------- M1a (MLA)


def _rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(bf)


MLA_BS = 16  # tokens a KV block


def _page_view(pages: torch.Tensor) -> torch.Tensor:
    """The [blocks, 16, 576] cache over hybrid-aligned pages (a block's rows
    dense, the blocks strided by the page)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import ENTRY

    return pages[:, : MLA_BS * ENTRY].view(pages.size(0), MLA_BS, ENTRY)


def _fp8_codes_adjacent(ours: torch.Tensor, theirs: torch.Tensor) -> bool:
    """Whether two E4M3 byte tensors differ by at most one code step (a rounding
    boundary: the same sign, adjacent magnitudes)."""
    ours, theirs = ours.int(), theirs.int()
    step = ((ours & 0x7F) - (theirs & 0x7F)).abs()
    same_sign = (ours & 0x80) == (theirs & 0x80)
    zero = ((ours | theirs) & 0x7F) <= 1  # +-0 and the smallest subnormal
    return bool(((same_sign & (step <= 1)) | zero).all())


def _check_rows_only(pages, before, slots) -> None:
    """Every byte of `pages` but the stored tokens' rows (slot >= 0) is
    `before`'s (the other rows, the page pads)."""
    s = slots[slots >= 0]
    blk, off = s // MLA_BS, s % MLA_BS
    rest = pages.clone()
    _page_view(rest)[blk, off] = _page_view(before)[blk, off]
    assert torch.equal(rest, before), "M1a wrote bytes outside its slots' rows"


def _check_cache(pages, ref_pages, slots, off_max: float = 0.05) -> None:
    """M1a's KV pages against the reference's: the stored tokens' rows (slot
    >= 0) an E4M3 step off at most, few of their bytes off; every other byte of
    the pages untouched."""
    s = slots[slots >= 0]
    blk, off = s // MLA_BS, s % MLA_BS
    got, want = _page_view(pages)[blk, off], _page_view(ref_pages)[blk, off]
    assert _fp8_codes_adjacent(got, want)
    frac = (got != want).float().mean().item() if s.numel() else 0.0
    assert frac <= off_max, frac
    _check_rows_only(pages, ref_pages, slots)


def _mla_inputs(S: int, nb: int, rank: int = 0, ranks: int = 1):
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import (
        ENTRY,
        GROWS,
        KVL,
        NH,
        NOPE,
        QB,
        QKV_A,
        QL,
    )

    g = torch.Generator(device="cuda").manual_seed(0)  # replicated
    gr = torch.Generator(device="cuda").manual_seed(300 + rank)  # TP shards
    w = {
        "prefix": _randn(g, S, HIDDEN, scale=2.0),
        "delta": _randn(g, S, HIDDEN, scale=0.5),
        "blocks": _randn(g, S, 8, HIDDEN, scale=2.0),
        "ares_nw": (1 + _randn(g, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "ares_qk": _randn(g, HIDDEN, scale=0.05),
        "in_nw": (1 + _randn(g, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "w_qkv": _randn(g, QKV_A, HIDDEN, scale=0.02),
        "w_g": _randn(gr, GROWS, HIDDEN, scale=0.02),
        "q_nw": (1 + _randn(g, QL, scale=0.1, dtype=torch.float32)).to(bf),
        "kv_nw": (1 + _randn(g, KVL, scale=0.1, dtype=torch.float32)).to(bf),
        "w_qb": _randn(gr, QB, QL, scale=0.03),
    }
    # W_UK as vLLM's MLA keeps it: rows (n, h) quantized by aiter, the scales
    # column-major: W_K (12, 512, 64) strides (64, 768, 1), its scales (1, 12, 6144)
    uk = _randn(gr, KVL * NH, NOPE, scale=0.1)
    wq, ws = dynamic_mxfp4_quant(uk)
    w["w_k"] = wq.view(torch.uint8).unflatten(0, (KVL, NH)).transpose(0, 1)
    w["w_k_scale"] = ws.view(torch.uint8).unflatten(0, (KVL, NH)).transpose(0, 1)
    # hybrid page alignment: entries dense, blocks strided; a token another
    # DCP rank's (slot -1): every other one, or t % ranks != rank
    bs, nblk = MLA_BS, 2 * S // MLA_BS + 2
    # finite E4M3 bytes (0x7f is NaN)
    pages = torch.randint(
        0,
        0x70,
        (nblk, bs * ENTRY + 512),
        dtype=torch.uint8,
        device="cuda",
        generator=gr,
    )
    w["pages"] = pages
    w["kv_cache"] = _page_view(pages)
    perm = torch.randperm(nblk * bs, device="cuda", generator=gr)[:S]
    if ranks == 1:
        perm[1::2] = -1
    else:
        perm[torch.arange(S, device="cuda") % ranks != rank] = -1
    w["slots"] = perm.to(torch.int64)
    return w


def _mla_reference(w, nb: int):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import (
        KVL,
        NH,
        NOPE,
        QL,
        ROPE,
    )
    from vllm.platforms import current_platform

    fp8 = current_platform.fp8_dtype()
    prefix, blocks = w["prefix"].clone(), w["blocks"].clone()
    x = attn_res(
        prefix,
        w["delta"],
        blocks,
        w["ares_nw"],
        w["ares_qk"],
        w["in_nw"],
        nb,
        -1,
        1e-5,
        1e-5,
    )
    qkv = F.linear(x, w["w_qkv"])
    q_c, kv_c, k_pe = qkv.split([QL, KVL, ROPE], -1)
    q_c, kv_c = _rms(q_c, w["q_nw"], 1e-6), _rms(kv_c, w["kv_nw"], 1e-6)
    q = F.linear(q_c, w["w_qb"]).view(-1, NH, NOPE + ROPE)
    q_nope, q_pe = q.split([NOPE, ROPE], -1)
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    ql = rocm_aiter_ops.batched_gemm_a16wfp4(
        q_nope.transpose(0, 1),
        w["w_k"],
        w["w_k_scale"],
        transpose_bm=True,
        prequant=True,
        y_scale=one,
    )
    qb = torch.cat([ql, q_pe], -1)
    q8 = qb.float().clamp(-448, 448).to(fp8)
    pages = w["pages"].clone()
    entry = torch.cat([kv_c, k_pe], -1).float().clamp(-448, 448).to(fp8)
    s = w["slots"][w["slots"] >= 0]
    _page_view(pages)[s // MLA_BS, s % MLA_BS] = entry[w["slots"] >= 0].view(
        torch.uint8
    )
    gate = F.linear(x, w["w_g"]).sigmoid()
    return {
        "prefix": prefix, "blocks": blocks, "q": q8, "qb": qb, "pages": pages,
        "cache": _page_view(pages), "gate": gate, "qkv": qkv,
    }  # fmt: skip


def _fp8_f(t):
    from vllm.platforms import current_platform

    return t.view(current_platform.fp8_dtype()).float()


def _m1a_launch(w, nb: int, pages, slots, qbf16: bool = False) -> dict:
    """M1a at DCP1 on `w`, its KV rows into `pages` (`_page_view`) at `slots`:
    the prefix, blocks, query and gate, and from its scratch the qkv_a
    projection (`qa`, bf16 [S, 2112]) and the normed q_c (`qc`)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import (
        ENTRY,
        GROWS,
        NH,
        QKV_A,
        QL,
        MlaPreBuild,
        mla_pre,
        scratch_bytes,
        scratch_layout,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import HEALTH_WORDS

    S = w["prefix"].size(0)
    key = MlaPreBuild(tokens=S, nblocks=nb, delta=True, ranks=1, qbf16=qbf16)
    prefix, blocks = w["prefix"].clone(), w["blocks"].clone()
    q = torch.zeros(S, NH, ENTRY, dtype=bf if qbf16 else torch.uint8, device="cuda")
    gate = torch.zeros(S, GROWS, dtype=bf, device="cuda")
    scratch = torch.zeros(scratch_bytes(key), dtype=torch.uint8, device="cuda")
    health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device="cuda")
    flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    mla_pre(
        key,
        prefix=prefix,
        delta=w["delta"],
        blocks=blocks,
        ares_nw=w["ares_nw"],
        ares_qk=w["ares_qk"],
        in_nw=w["in_nw"],
        w_qkv=w["w_qkv"],
        w_g=w["w_g"],
        q_nw=w["q_nw"],
        kv_nw=w["kv_nw"],
        w_qb=w["w_qb"],
        w_k=w["w_k"],
        w_k_scale=w["w_k_scale"],
        kv_cache=_page_view(pages),
        slots=slots,
        gate_out=gate,
        q_out=q,
        scratch=scratch,
        peers=None,
        rank=0,
        layer=3,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        health=health,
        alarm=flags.data_ptr(),
    )
    torch.accelerator.synchronize()
    assert not health.any() and not flags.any()
    lay = scratch_layout(key)
    qa = _pair_words(scratch, lay["qa"][0], S * QKV_A // 2).view(bf)
    o, n = lay["qc"]
    return {
        "key": key, "prefix": prefix, "blocks": blocks, "q": q, "gate": gate,
        "qa": qa.view(S, QKV_A), "qc": scratch[o : o + n].view(bf).view(S, QL),
    }  # fmt: skip


@needs_gfx950
@pytest.mark.parametrize("qbf16", [False, True])
@pytest.mark.parametrize("S", [1, 2, 3, 4, 8, 16, 32, 64])
def test_m1a_single_gpu_matches_vllm_ops(S: int, qbf16: bool) -> None:
    """M1a at DCP1 (ranks = 1: no gather, no peer memory) against attn_res ->
    qkv_a -> q / kv RMSNorm -> q_b -> aiter's fp4 W_UK -> the fp8 query (bf16
    with `qbf16`, M1c's), the fp8 cache rows and the gate; no other byte of
    the cache pages written."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import KVL

    nb = 4
    w = _mla_inputs(S, nb)
    ref = _mla_reference(w, nb)
    pages = w["pages"].clone()
    m = _m1a_launch(w, nb, pages, w["slots"], qbf16)
    q, gate = m["q"], m["gate"]

    torch.testing.assert_close(m["prefix"], ref["prefix"], atol=0, rtol=0)
    torch.testing.assert_close(m["blocks"], ref["blocks"], atol=0, rtol=0)
    # GEMV sums differ from hipBLASLt's / aiter's in order only: an fp8 or bf16
    # value now and then lands one step off
    if qbf16:
        # bf16 values a bf16 step off now and then, the same elementwise
        # rounding as the fp8 path's before its quant
        rq = ref["qb"].float()
        assert _rel_l2(q[..., :KVL].float(), rq[..., :KVL]) < 1e-2
        assert _rel_l2(q[..., KVL:].float(), rq[..., KVL:]) < 1e-2
        assert _cos(q.float(), rq) > 0.9999
        q = ref["q"].view(torch.uint8)  # the fp8 checks below: vacuous
    errs = {
        "gate": _rel_l2(gate, ref["gate"]),
        "q_nope": _rel_l2(_fp8_f(q[..., :KVL]), ref["q"][..., :KVL].float()),
        "q_pe": _rel_l2(_fp8_f(q[..., KVL:]), ref["q"][..., KVL:].float()),
    }
    assert errs["gate"] < 1e-2, errs
    assert errs["q_nope"] < 3e-2 and errs["q_pe"] < 3e-2, errs
    assert _cos(_fp8_f(q), ref["q"].float()) > 0.999, errs
    # the stored tokens' rows; unstored tokens' (slot -1) and the pads untouched
    _check_cache(pages, ref["pages"], w["slots"])


def _aiter_mla_writer(qb, kv_c, k_pe):
    """DCP1's production write (`KimiK3MultiHeadLatentAttentionWrapper.
    _fused_decode`): AITER's `fused_qk_rope_concat_and_cache_mla`, identity
    RoPE, unit scales, every token stored. Returns the fp8 query [S, 12, 576]
    and the KV rows [S, 576], as bytes."""
    from aiter.ops.cache import fused_qk_rope_concat_and_cache_mla

    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import (
        ENTRY,
        KVL,
    )
    from vllm.platforms import current_platform

    fp8, dev = current_platform.fp8_dtype(), qb.device
    S, half = qb.size(0), (ENTRY - KVL) // 2
    rows = torch.zeros(S, 1, ENTRY, dtype=fp8, device=dev)
    q8 = torch.zeros(qb.shape, dtype=fp8, device=dev)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    fused_qk_rope_concat_and_cache_mla(
        qb[..., :KVL].contiguous(),
        qb[..., KVL:].contiguous(),
        kv_c.contiguous(),
        k_pe.contiguous(),
        rows,
        q8,
        torch.arange(S, device=dev),
        one,
        one,
        torch.zeros(S, dtype=torch.int64, device=dev),
        torch.ones(1, half, dtype=bf, device=dev),
        torch.zeros(1, half, dtype=bf, device=dev),
        is_neox=True,
        is_nope_first=True,
    )
    return q8.view(torch.uint8), rows.view(torch.uint8).view(S, ENTRY)


def _stock_mla_writer(kv_c, k_pe):
    """DCP8's production write (the MLA backend's `do_kv_cache_update`):
    `concat_and_cache_mla`, fp8, unit scale, every token stored; the KV rows
    [S, 576] as bytes."""
    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import ENTRY

    S, dev = kv_c.size(0), kv_c.device
    rows = torch.zeros(S, 1, ENTRY, dtype=torch.uint8, device=dev)
    ops.concat_and_cache_mla(
        kv_c.contiguous(),
        k_pe.contiguous(),
        rows,
        torch.arange(S, device=dev),
        "fp8",
        torch.ones(1, dtype=torch.float32, device=dev),
    )
    return rows.view(S, ENTRY)


def _stored_rows(pages, slots):
    """The KV rows [stored tokens, 576] of the tokens with a slot (>= 0)."""
    s = slots[slots >= 0]
    return _page_view(pages)[s // MLA_BS, s % MLA_BS]


@needs_gfx950
@pytest.mark.parametrize("S", [1, 4, 16, 64])
def test_m1a_fp8_rounding_matches_production(S: int) -> None:
    """M1a's E4M3 rounding is the production writers', bit for bit, on M1a's
    own bf16 values (exact ties included: a random bf16 value is one 1 time in
    32): its fp8 query AITER fused_qk_rope_concat_and_cache_mla's (DCP1) of
    its bf16 query (`qbf16`), its KV rows' k_pe bytes that and
    concat_and_cache_mla's (DCP8) of its projection's k_pe."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import (
        KVL,
        QL,
    )

    nb = 4
    w = _mla_inputs(S, nb)
    pages = w["pages"].clone()
    m8 = _m1a_launch(w, nb, pages, w["slots"])
    mb = _m1a_launch(w, nb, w["pages"].clone(), w["slots"], qbf16=True)
    assert torch.equal(mb["qa"], m8["qa"])
    kv_c, k_pe = m8["qa"][:, QL : QL + KVL], m8["qa"][:, QL + KVL :]
    q8, rows = _aiter_mla_writer(mb["q"], kv_c, k_pe)
    stored = w["slots"] >= 0
    pe = _stored_rows(pages, w["slots"])[:, KVL:]
    off = {
        "q": (m8["q"] != q8).sum().item(),
        "k_pe_aiter": (pe != rows[stored][:, KVL:]).sum().item(),
        "k_pe_stock": (pe != _stock_mla_writer(kv_c, k_pe)[stored][:, KVL:])
        .sum()
        .item(),
    }
    assert not any(off.values()), off


@needs_gfx950
@pytest.mark.parametrize("S", [1, 4, 16, 64])
def test_m1a_rms_norm_matches_production(S: int) -> None:
    """M1a's q / kv RMSNorm of its own qkv_a projection is the production
    layer's, bit for bit: fused_mla_dual_rms_norm (AITER's fused_qk_rmsnorm,
    eager and the compile pass's) for the normed q_c rows, and the KV rows'
    kv_c bytes the production writers' of its normed kv_c."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import (
        KVL,
        QL,
    )

    nb = 4
    w = _mla_inputs(S, nb)
    ref = _mla_reference(w, nb)
    pages = w["pages"].clone()
    m = _m1a_launch(w, nb, pages, w["slots"])
    qa, key = m["qa"], m["key"]
    assert _cos(qa, ref["qkv"]) > COS_MIN  # the scratch read back
    q_c, kv_c = rocm_aiter_ops.get_fused_mla_dual_rms_norm_op()(
        qa[:, :QL], w["q_nw"], qa[:, QL : QL + KVL], w["kv_nw"], key.q_eps, key.kv_eps
    )
    stored = w["slots"] >= 0
    rows = _stored_rows(pages, w["slots"])[:, :KVL]
    k_pe = qa[:, QL + KVL :]
    off = {
        "q_c": (m["qc"].view(torch.int16) != q_c.view(torch.int16)).sum().item(),
        "kv_c_stock": (rows != _stock_mla_writer(kv_c, k_pe)[stored][:, :KVL])
        .sum()
        .item(),
    }
    assert not any(off.values()), off


# KV blocks at the hybrid-aligned page (16 rows of 576 B + 512 B): the first
# whose rows lie 2^31, then 2^32 bytes past the cache's base (a 32-bit block x
# page byte offset wraps from there)
MLA_PAGE = MLA_BS * 576 + 512
HIGH_BLOCK = -(-(1 << 31) // MLA_PAGE)
WIDE_BLOCK = -(-(1 << 32) // MLA_PAGE)


@needs_gfx950
@pytest.mark.parametrize("S", [16, 64])
def test_m1a_high_pages(S: int) -> None:
    """M1a's KV rows into blocks past 2^31 and 2^32 bytes of a > 4 GiB cache (a
    request whose blocks sit high in a large KV pool; the tests above use low
    blocks only): the rows it writes at low blocks, its other outputs
    unchanged, and no other byte of the pages written."""
    nb = 4
    w = _mla_inputs(S, nb)
    low = w["pages"].clone()
    want = _m1a_launch(w, nb, low, w["slots"])
    blocks = [2, HIGH_BLOCK - 1, HIGH_BLOCK, HIGH_BLOCK + 5]
    blocks += [WIDE_BLOCK - 1, WIDE_BLOCK, WIDE_BLOCK + 6]
    stored = (w["slots"] >= 0).nonzero().flatten()
    i = torch.arange(stored.numel(), device="cuda")
    blk = torch.tensor(blocks, device="cuda")[i % len(blocks)]
    row = i // len(blocks)
    slots = w["slots"].clone()
    slots[stored] = blk * MLA_BS + row
    fill = 0x35
    pages = torch.full(
        (WIDE_BLOCK + 8, MLA_PAGE), fill, dtype=torch.uint8, device="cuda"
    )
    got = _m1a_launch(w, nb, pages, slots)
    for k in ("prefix", "blocks", "q", "gate"):
        assert torch.equal(got[k], want[k]), k
    view = _page_view(pages)
    assert torch.equal(view[blk, row], _stored_rows(low, w["slots"]))
    view[blk, row] = fill
    assert bool((pages == fill).all()), "M1a wrote bytes outside its slots' rows"


def _a2a_combine(o, lse, seq_lens, qsl, q: int):
    """`dcp_a2a_lse_reduce`'s pack + unpack-combine for one rank's heads,
    every rank's partials stacked: o [R, S, 12, 512], lse [R, S, 12], each
    sender's local KV lengths seq_lens [R, requests]."""
    S = o.size(1)
    t = torch.arange(S, device=o.device)
    empty = (seq_lens[:, t // q] == 0) | (t >= qsl[S // q])
    lse = lse.masked_fill(empty.unsqueeze(-1), float("-inf"))
    lse = torch.where(lse.isnan() | (lse == float("inf")), float("-inf"), lse)
    m = lse.max(0).values
    m = torch.where(m == float("-inf"), 0.0, m)
    glse = (lse - m).exp().sum(0).log() + m
    w = (lse - glse).exp().nan_to_num(0.0)
    p = torch.where(w.unsqueeze(-1) == 0, 0.0, o.float())
    return (p * w.unsqueeze(-1)).sum(0).to(bf)


def _mla_back_inputs(S: int, q: int, R: int, rank: int = 0):
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import (
        GROWS,
        KVL,
        NH,
    )

    g = torch.Generator(device="cuda").manual_seed(1 + rank)
    nreq = S // q
    w = {
        # AITER's padded 128-head outputs, viewed to the gathered heads
        "o": _randn(g, S, 128, KVL, scale=0.5)[:, : R * NH],
        "lse": (torch.randn(S, 128, generator=g, device="cuda") * 3)[:, : R * NH],
        "gate": torch.rand(S, GROWS, generator=g, device="cuda").to(bf),
        # the first request with no local KV (when there are others)
        "seq_lens": torch.randint(
            1, 900, (nreq,), dtype=torch.int32, device="cuda", generator=g
        ),
        "qsl": torch.arange(0, S + 1, q, dtype=torch.int32, device="cuda"),
    }
    # a request this rank holds none of, and with ranks, one no rank holds
    if nreq > 1:
        w["seq_lens"][rank % nreq] = 0
    if R > 1 and nreq > 2:
        w["seq_lens"][-1] = 0
    uv = _randn(g, 128 * NH, KVL, scale=0.05)
    wq, ws = dynamic_mxfp4_quant(uv)
    # (12, 128, 256) dense rows a head; the scales column-major as vLLM keeps them
    w["w_v"] = wq.view(torch.uint8).view(NH, 128, KVL // 2)
    w["w_v_scale"] = ws.view(torch.uint8).unflatten(0, (NH, 128))
    return w


def _uv_gate(lat, w):
    """The gate times vLLM's `_v_up_proj` (aiter's fp4 W_UV)."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import NH

    S = lat.size(0)
    out = torch.empty(S, NH, 128, dtype=bf, device=lat.device)
    rocm_aiter_ops.batched_gemm_a16wfp4(
        lat.transpose(0, 1),
        w["w_v"],
        w["w_v_scale"],
        out,
        transpose_bm=True,
        prequant=True,
        y_scale=None,
    )
    return out.view(S, -1) * w["gate"]


def _mla_back_reference(w):
    """DCP1: AITER's output is the latent (vLLM's `_v_up_proj` input)."""
    return _uv_gate(w["o"], w)


@needs_gfx950
@pytest.mark.parametrize(
    "S, q",
    [(1, 1), (4, 1), (16, 1), (4, 4), (8, 4), (16, 4), (32, 4), (64, 4), (64, 8)],
)
def test_m1b_single_gpu_matches_vllm_ops(S: int, q: int) -> None:
    """M1b at DCP1 (ranks = 1: no LSE, no peer memory): AITER's strided
    output -> aiter's fp4 W_UV -> the gate."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import GROWS
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla_back import (
        MlaPostBuild,
        mla_post,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import HEALTH_WORDS

    w = _mla_back_inputs(S, q, 1)
    ref = _mla_back_reference(w)
    health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device="cuda")
    flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    core = torch.zeros(S, GROWS, dtype=bf, device="cuda")
    mla_post(
        MlaPostBuild(tokens=S, qlen=q, ranks=1),
        o=w["o"],
        lse=None,
        seq_lens=w["seq_lens"],
        qsl=w["qsl"],
        w_v=w["w_v"],
        w_v_scale=w["w_v_scale"],
        gate=w["gate"],
        core=core,
        peers=None,
        rank=0,
        layer=5,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        health=health,
        alarm=flags.data_ptr(),
    )
    torch.accelerator.synchronize()
    assert not health.any() and not flags.any()
    err = _rel_l2(core, ref)
    assert err < 1e-2 and _cos(core, ref) > COS_MIN, err


def _mla_attn_reference(qb, kv8, idx, lens, q, scale):
    """fp32 causal MLA decode a request: [S, 12, 512] (a request with no KV 0)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import KVL

    out, p0 = [], 0
    for r, n in enumerate(lens):
        qq = qb[r * q : (r + 1) * q].float()
        if n == 0:
            out.append(qq.new_zeros(q, qq.size(1), KVL))
            continue
        kv = kv8[idx[p0 : p0 + n].long()].float()
        p0 += n
        sc = torch.einsum("thd,kd->thk", qq, kv) * scale
        lim = n - q + torch.arange(q, device=qq.device)
        pos = torch.arange(n, device=qq.device)
        sc = sc.masked_fill(pos[None, None] > lim[:, None, None], -torch.inf)
        out.append(torch.einsum("thk,kd->thd", sc.softmax(-1), kv[:, :KVL]))
    return torch.cat(out)


@needs_gfx950
@pytest.mark.parametrize(
    "q, lens",
    [(8, [8704]), (8, [3000, 0]), (4, [5000, 17, 9000, 64]), (8, [9]), (8, [40000])],
)
def test_m1c_m1b_splits_single_gpu(q: int, lens: list) -> None:
    """M1c (`mla_attn`: bf16 query, fp8 KV, the splits' partials) against
    fp32 attention, and M1b's `splits` combine -> W_UV -> the gate against
    the partials merged in torch."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import mla_attn as m
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import (
        ENTRY,
        GROWS,
        NH,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla_back import (
        MlaPostBuild,
        mla_post,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import HEALTH_WORDS

    S = q * len(lens)
    g = torch.Generator(device="cuda").manual_seed(7)
    ntok = 200_000
    kv8 = (torch.randn(ntok, ENTRY, generator=g, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    idx = torch.randperm(ntok, generator=g, device="cuda")[: sum(lens)].int()
    indptr = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()],
                          dtype=torch.int32, device="cuda")  # fmt: skip
    qb = _randn(g, S, NH, ENTRY, scale=0.5)
    scale, kv_scale = 1.0 / 192**0.5, 0.75
    key = m.MlaAttnBuild(tokens=S, qlen=q)
    so, sl = m.partial_shapes(key)
    po = torch.empty(so, dtype=bf, device="cuda")
    pl = torch.empty(sl, dtype=torch.float32, device="cuda")
    m.mla_attn(key, q=qb, kv=kv8, pages=idx, indptr=indptr, part_o=po, part_lse=pl,
               scale=scale, kv_scale=kv_scale)  # fmt: skip
    att = _mla_attn_reference(qb, kv8, idx, lens, q, scale * kv_scale) * kv_scale
    # the partials merged in torch: a request's splits only (the rest unwritten;
    # an empty request's lse is -inf: 0 weight)
    per = so[0] // len(lens)
    lat = []
    for r in range(len(lens)):
        lo, ll = (
            po[r * per : (r + 1) * per, r * q : (r + 1) * q],
            pl[r * per : (r + 1) * per, r * q : (r + 1) * q],
        )
        e = (ll - ll.amax(0).nan_to_num(neginf=0.0)).exp()
        lat.append(
            (lo.float() * e[..., None]).sum(0) / e.sum(0).clamp_min(1e-30)[..., None]
        )
    lat = torch.cat(lat)
    assert _rel_l2(lat, att) < 5e-3
    # W_UV quantizes the latent to mxfp4: compare at the same latent
    w = _mla_back_inputs(S, q, 1)
    ref = _uv_gate(lat.to(bf), w)
    health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device="cuda")
    flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    core = torch.zeros(S, GROWS, dtype=bf, device="cuda")
    mla_post(
        MlaPostBuild(tokens=S, qlen=q, ranks=1, splits=m.splits(q)),
        o=po[0], lse=pl[0], seq_lens=indptr.diff().int(), qsl=w["qsl"],
        w_v=w["w_v"], w_v_scale=w["w_v_scale"], gate=w["gate"], core=core,
        peers=None, rank=0, layer=5,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        health=health, alarm=flags.data_ptr(),
    )  # fmt: skip
    torch.accelerator.synchronize()
    assert not health.any() and not flags.any()
    assert torch.isfinite(core.float()).all()
    err = _rel_l2(core, ref)
    assert err < 1e-2 and _cos(core, ref) > COS_MIN, err


# M1c's pool rows of 576 B: the first past 2^31 bytes (a signed 32-bit byte
# offset turns negative) and the last of a 4 GiB pool (`mla_attn.wide_kv`: the
# largest pool M1c reads at 32-bit offsets)
M1C_HIGH_ROW = -(-(1 << 31) // 576)
M1C_ROWS_4G = (1 << 32) // 576


def _m1c_high_pool(kv, idx, lo: int, rows: int, gen):
    """A pool of `rows` rows holding `kv[idx]` at distinct random rows from
    `lo` on (the rest unwritten); returns (pool, their rows)."""
    high = lo + torch.randperm(rows - lo, generator=gen, device=kv.device)
    high = high[: idx.numel()]
    pool = torch.empty(rows, kv.size(1), dtype=kv.dtype, device=kv.device)
    pool[high] = kv[idx.long()]
    return pool, high.int()


@needs_gfx950
@pytest.mark.parametrize("wide", [False, True], ids=["4GiB-pool", "wide-pool"])
def test_m1c_high_pages(wide: bool) -> None:
    """M1c on KV rows past 2^31 bytes of a pool of at most 4 GiB (32-bit
    offsets off its base), and past 2^32 bytes of a larger one (`wide`: 64-bit
    addresses): its partials are the same rows' at low indices, bit for bit
    (the tests above use pools of 200K rows at most)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import mla_attn as m
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import (
        ENTRY,
        NH,
    )

    q, lens = 8, [3000, 700]
    S = q * len(lens)
    g = torch.Generator(device="cuda").manual_seed(9)
    kv8 = (torch.randn(sum(lens), ENTRY, generator=g, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    idx = torch.randperm(sum(lens), generator=g, device="cuda").int()
    indptr = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()],
                          dtype=torch.int32, device="cuda")  # fmt: skip
    qb = _randn(g, S, NH, ENTRY, scale=0.5)
    if wide:
        pool, high = _m1c_high_pool(kv8, idx, M1C_ROWS_4G - 1000, M1C_ROWS_4G + 4096, g)
    else:
        pool, high = _m1c_high_pool(kv8, idx, M1C_HIGH_ROW - 1000, M1C_ROWS_4G, g)
    assert m.wide_kv(pool) == wide
    key = m.MlaAttnBuild(tokens=S, qlen=q)
    so, sl = m.partial_shapes(key)
    outs = []
    for kv, pages in ((kv8, idx), (pool, high)):
        po = torch.zeros(so, dtype=bf, device="cuda")
        pl = torch.zeros(sl, dtype=torch.float32, device="cuda")
        m.mla_attn(key, q=qb, kv=kv, pages=pages, indptr=indptr, part_o=po,
                   part_lse=pl, scale=1.0 / 192**0.5, kv_scale=0.75)  # fmt: skip
        torch.accelerator.synchronize()
        outs.append((po, pl))
    for x, y in zip(*outs):
        assert torch.equal(x, y)


# ---------------------------------------------------------------- K2 / MoE


def _situ(g, u):
    g, u = g.float(), u.float()
    a = (2.0 * torch.tanh(g * 0.25)) * (1.0 + torch.tanh(g * 0.5))
    return (a * (25.0 * torch.tanh(u * 0.04))).to(bf)


def _all_reduce_fp32(t: torch.Tensor) -> torch.Tensor:
    """The ranks' bf16 partials summed in fp32, as vLLM's custom all-reduce."""
    t = t.float()
    dist.all_reduce(t)
    return t.to(bf)


def _moe_reference(x, mw, rank: int):
    """KimiMoE's decode path on `x`: sigmoid router + bias top-16, the latent
    down, aiter's MXFP4 experts, the latent all-reduce + RMSNorm, this rank's
    up_proj rows into the shared experts' output, the final all-reduce."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import (
        SI,
        TOPK,
        UP_N,
    )

    sig = torch.sigmoid(torch.mm(x.float(), mw["w_gate"].float().t()))
    ids = torch.topk(sig + mw["bias"], TOPK, dim=-1, sorted=True).indices
    wts = sig.gather(-1, ids)
    wts = wts / wts.sum(-1, keepdim=True)
    gu = F.linear(x, mw["w_sgu"])
    shared = F.linear(_situ(gu[:, :SI], gu[:, SI:]), mw["w_sd"])
    y = fused_moe(
        F.linear(x, mw["w_ld"]),
        mw["w13"],
        mw["w2"],
        wts.float(),
        ids.to(torch.int32),
        None,
        ActivationType.Situv2,
        QuantType.per_1x32,
        False,
        mw["w13s"],
        mw["w2s"],
        None,
        None,
        dtype=bf,
        gate_mode="separated",
        beta=4.0,
        linear_beta=25.0,
    )
    lat = _all_reduce_fp32(y).float()
    ln = lat * torch.rsqrt(lat.pow(2).mean(-1, keepdim=True) + 1e-5) * mw["ln_w"]
    lo = rank * UP_N
    up = ln.to(bf).float() @ mw["w_up_shard"].float().t()
    shared[:, lo : lo + UP_N] = (shared[:, lo : lo + UP_N].float() + up).to(bf)
    return _all_reduce_fp32(shared)


def _moe_weights(rank: int, device):
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import (
        HIDDEN,
        LAT,
        RI,
        SI,
        UP_N,
        E,
    )

    gc = torch.Generator(device=device).manual_seed(7)  # replicated weights
    gr = torch.Generator(device=device).manual_seed(100 + rank)  # TP shards

    def u8(g, *shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=g, device=device).to(torch.uint8)

    fp4, e8 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    w_up = _randn(gc, HIDDEN, LAT, scale=0.02)
    return {
        "w_gate": _randn(gc, E, HIDDEN, scale=0.02),
        "bias": _randn(gc, E, scale=0.02, dtype=torch.float32),
        "w_ld": _randn(gc, LAT, HIDDEN, scale=0.02),
        "ln_w": (1 + _randn(gc, LAT, scale=0.1, dtype=torch.float32)).to(bf),
        "w_up_shard": w_up[rank * UP_N : (rank + 1) * UP_N].contiguous(),
        "w_sgu": _randn(gr, 2 * SI, HIDDEN, scale=0.02),
        "w_sd": _randn(gr, HIDDEN, SI, scale=0.03),
        "w13": rocm_aiter_ops.shuffle_weight_a16w4(
            u8(gr, E, 2 * RI, LAT // 2).view(fp4), 16, False
        ),
        "w2": rocm_aiter_ops.shuffle_weight_a16w4(
            u8(gr, E, LAT, RI // 2).view(fp4), 16, False
        ),
        "w13s": rocm_aiter_ops.shuffle_scale_a16w4(
            u8(gr, E, 2 * RI, LAT // 32, lo=118, hi=123).view(e8).view(-1, LAT // 32),
            E,
            False,
        ),
        "w2s": e8m0_shuffle(
            u8(gr, E, LAT, RI // 32, lo=118, hi=123).view(e8).view(-1, RI // 32)
        ),
    }


def _moe_args(mw):
    return {
        "w_gate": mw["w_gate"],
        "bias": mw["bias"],
        "w_ld": mw["w_ld"],
        "w_sgu": mw["w_sgu"],
        "w_sd": mw["w_sd"],
        "w13": mw["w13"],
        "w13s": mw["w13s"],
        "w2": mw["w2"],
        "w2s": mw["w2s"],
        "ln_w": mw["ln_w"],
        "w_up": mw["w_up_shard"],
    }


def _k2_inputs(rank: int, device, S: int = 8, seed: int = 0):
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        HIDDEN,
        PROJ,
    )

    gc = torch.Generator(device=device).manual_seed(11 + 1000 * seed)
    gr = torch.Generator(device=device).manual_seed(200 + rank + 1000 * seed)
    return {
        "core": _randn(gr, S, PROJ),
        "w_o": _randn(gr, HIDDEN, PROJ, scale=0.02),
        "prefix": _randn(gc, S, HIDDEN, scale=2.0),
        "blocks": _randn(gc, S, 8, HIDDEN, scale=2.0),
        "ares_nw": (1 + _randn(gc, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "ares_qk": _randn(gc, HIDDEN, scale=0.05),
        "in_nw": (1 + _randn(gc, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
    }


def _k2_launch(
    runner,
    w,
    mw,
    prefix,
    out,
    reset: bool,
    nb: int = 4,
    layer: int = 3,
    m1b=None,
    splits: int = 0,
    m1c_in=None,
    opt: int | None = None,
):
    """`m1b`: (`mla_post`'s inputs, qlen, ranks) to run M1b in K2's launch;
    `splits` its M1c splits, `m1c_in` (`mla_attn.host_args`'s) to run M1c
    in the launch too. `opt`: `K2Build.opt` (None: the default)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2

    m1b_in, fused = None, {}
    if m1b is not None:
        m1b_in, q, ranks = m1b
        fused = dict(
            m1b_qlen=q,
            m1b_ranks=ranks,
            m1b_off=runner.mla_post_off,
            m1b_splits=splits,
            m1c=m1c_in is not None,
        )
    if opt is not None:
        fused["opt"] = opt
    key = k2.K2Build(
        tokens=w["core"].size(0),
        nblocks=nb + int(reset),
        reset=reset,
        **fused,
        **runner.fault,
    )
    k2.k2_launch(
        key,
        core=w["core"],
        w_o=w["w_o"],
        prefix=prefix,
        blocks=w["blocks"],
        ares_nw=w["ares_nw"],
        ares_qk=w["ares_qk"],
        in_nw=w["in_nw"],
        out=out,
        layer=128 + layer,
        m1b_in=m1b_in,
        m1c_in=m1c_in,
        **_moe_args(mw),
        **runner.launch_args(),
        **runner.peer_args(),
    )
    return key


def _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    return device


@ray.remote(num_gpus=1, max_calls=1)
def _k2_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res
    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

    nb = 4
    runner = MonoRunner(device)
    mw = _moe_weights(rank, device)
    for S, reset in K2_CASES:
        w = _k2_inputs(rank, device, S)
        runner.step_begin()
        prefix = w["prefix"].clone()
        out = torch.empty(S, HIDDEN, dtype=bf, device=device)
        key = _k2_launch(runner, w, mw, prefix, out, reset)
        torch.accelerator.synchronize()

        attn = _all_reduce_fp32(F.linear(w["core"], w["w_o"]))
        # a block-write layer's prefix restarts at the attention output
        ref_prefix = attn if reset else w["prefix"].clone()
        delta = None if reset else attn
        x_ref = attn_res(
            ref_prefix,
            delta,
            w["blocks"],
            w["ares_nw"],
            w["ares_qk"],
            w["in_nw"],
            nb + int(reset),
            -1,
            1e-5,
            1e-5,
        )
        # the kernel sums the ranks in rank order, NCCL in its own order
        assert _rel_l2(prefix, ref_prefix) < 1e-2
        lay = k2.scratch_layout(key)["xrow"][0]
        x = runner.scratch[lay : lay + S * HIDDEN * 2].view(bf).view(S, HIDDEN)
        cos_x = _cos(x, x_ref)
        assert cos_x > COS_MIN, (S, reset, cos_x)
        cos_out = _cos(out, _moe_reference(x, mw, rank))
        assert cos_out > COS_MIN, (S, reset, cos_out)
        outs = [torch.empty_like(out) for _ in range(tp_size)]
        dist.all_gather(outs, out)
        assert all(torch.equal(outs[0], o) for o in outs), "ranks disagree"

    # A graph of one step (the epoch bump captured with it), replayed: a
    # block-write launch writes its prefix, so every replay must match the first.
    S = K2_CASES[-1][0]
    w = _k2_inputs(rank, device, S)
    prefix = torch.empty_like(w["prefix"])
    out = torch.empty(S, HIDDEN, dtype=bf, device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.step_begin()
        _k2_launch(runner, w, mw, prefix, out, reset=True)
    graph.replay()
    torch.accelerator.synchronize()
    want_prefix, want_out = prefix.clone(), out.clone()
    for _ in range(300):
        graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(prefix, want_prefix) and torch.equal(out, want_out)
    assert not runner.health.any() and not runner.flags.any()


# (rows, block-write layer): every width up to the runner's MAX_TOKENS
K2_CASES = [
    (1, False),
    (1, True),
    (8, False),
    (8, True),
    (4, False),
    (4, True),
    (16, False),
    (16, True),
    (32, False),
    (64, False),
    (64, True),
]


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_k2_tp8_matches_vllm_ops_and_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _k2_worker)


@ray.remote(num_gpus=1, max_calls=1)
def _k2_opt_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    mw = _moe_weights(rank, device)

    def run(w, S, reset, layer, opt):
        runner.step_begin()
        prefix = w["prefix"].clone()
        out = torch.empty(S, HIDDEN, dtype=bf, device=device)
        _k2_launch(runner, w, mw, prefix, out, reset, layer=layer, opt=opt)
        torch.accelerator.synchronize()
        return out, prefix

    for S, reset in K2_CASES:
        # two input sets with different words: a sentinel region the first
        # K2_OPT launch left un-re-armed would hand the second launch the
        # first one's words, which no longer match opt=0's bits
        ws = [_k2_inputs(rank, device, S, seed=i) for i in range(2)]
        # both layer parities: the sentinel regions alternate by it
        for layer in (3, 4):
            want = [run(w, S, reset, layer, 0) for w in ws]
            assert not torch.equal(want[0][0], want[1][0]), (S, reset, layer)
            # back to back on the same regions: the second polls re-armed ones
            for i, w in enumerate(ws):
                out, prefix = run(w, S, reset, layer, k2.K2_OPT)
                assert torch.equal(out, want[i][0]), (S, reset, layer, i)
                assert torch.equal(prefix, want[i][1]), (S, reset, layer, i)
    assert not runner.health.any() and not runner.flags.any()


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_k2_tp8_opt_bits_are_bit_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """K2's critical-path options (`layer.K2_OPT`, the default) give the
    plain K2's (`opt=0`) bits at every width, block-write layer or not, on
    both layer parities, and again after their regions were re-armed."""
    multi_process_parallel(monkeypatch, 8, 1, _k2_opt_worker)


def _k2_regions(runner, key) -> dict:
    """K2's last launch (`opt` 0): every rank's router logits [S, 896] f32 (this
    rank's RLOG peer region), the route's expert ids and f32 weights [S, 16],
    and this rank's shared gate | up [S, 1536] and SiTU h [S, 768] (bf16)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2
    from vllm.models.kimi_k3.amd.specialized_decode.mono.abi import PEER_HEAD
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import (
        SI,
        TOPK,
        TP,
        E,
    )

    S, lay = key.tokens, k2.scratch_layout(key)
    par = int(runner.epoch.item()) % 2
    off = PEER_HEAD + k2.peer_layout(S)["rlog"][0] + par * (k2.peer_bytes(S) // 2)
    logits = _pair_words(runner.peers.bytes, off, TP * S * (E // TP))
    logits = logits.view(torch.float32).view(TP, S, E // TP).transpose(0, 1)
    route = _pair_words(runner.scratch, lay["route"][0], S * 2 * TOPK).view(S, -1)
    return {
        "logits": logits.reshape(S, E),
        "ids": route[:, :TOPK],
        "weights": route[:, TOPK:].view(torch.float32),
        "sgu": _pair_words(runner.scratch, lay["sgu"][0], S * SI).view(bf).view(S, -1),
        "h": _pair_words(runner.scratch, lay["h"][0], S * SI // 2).view(bf).view(S, -1),
    }


def _k3_router(bias):
    """Kimi-K3's router as KimiMoE has vLLM build it: create_fused_moe_router
    on KimiLinearConfig's routing (sigmoid, renormalized, one expert group)
    and the correction bias."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.router.router_factory import (
        create_fused_moe_router,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import TOPK, E
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    c = KimiLinearConfig()
    with set_current_vllm_config(VllmConfig()):
        return create_fused_moe_router(
            top_k=TOPK,
            global_num_experts=E,
            renormalize=c.moe_renormalize,
            use_grouped_topk=c.use_grouped_topk,
            num_expert_group=c.num_expert_group,
            topk_group=c.topk_group,
            scoring_func=c.moe_router_activation_func,
            e_score_correction_bias=bias,
        )


def _by_expert(ids, weights):
    order = ids.long().argsort(-1)
    return ids.long().gather(-1, order), weights.gather(-1, order)


def _k2_route(runner, w, m, S: int) -> dict:
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN

    runner.step_begin()
    out = torch.empty(S, HIDDEN, dtype=bf, device=runner.scratch.device)
    key = _k2_launch(runner, w, m, w["prefix"].clone(), out, False, opt=0)
    torch.accelerator.synchronize()
    assert not runner.health.any() and not runner.flags.any()
    return _k2_regions(runner, key)


def _k2_route_run(monkeypatch, tp_size, pp_size, rank, port, check: str) -> None:
    """K2's routes against the production router's. `check`: "ids" (random
    scores and near ties: fails listing, per (rows, case), some tokens' experts
    only K2 / only the router picked), "weights" (the same cases: the tokens
    with the same experts but weight bits off) or "ties" (an exact tie's ids)."""
    ties = check == "ties"
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, port)
    _aiter_env(monkeypatch)

    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import E
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    mw = _moe_weights(rank, device)
    # 15 experts every token picks, every other expert out but p and q: the
    # 16th pick is between them
    bias = torch.full((E,), -1.0, device=device)
    bias[torch.arange(15, device=device) * 59 + 3] = 2.0
    p, q = 130, 700
    off_ids, off_w = {}, {}
    for S in (1, 8, 16):
        w = _k2_inputs(rank, device, S)
        if ties:
            # q's gate row p's: every token's p and q scores tie
            tie = dict(mw, w_gate=mw["w_gate"].clone(), bias=bias.clone())
            tie["w_gate"][q] = tie["w_gate"][p]
            tie["bias"][[p, q]] = 0.5
            cases = {"exact_tie": tie}
        else:
            cases = {"random": mw}
            s = _k2_route(runner, w, mw, S)["logits"].sigmoid()
            # token 0's biased p and q scores 1 or 2 f32 steps apart around 1.0
            # (the step is 2^-24 below it, 2^-23 above)
            for k in (-2, -1, 2, 4):
                b = bias.clone()
                b[p] = 1.0 - s[0, p]
                b[q] = 1.0 - s[0, q] + k * 2.0**-24
                cases[f"near_tie{k:+d}"] = dict(mw, bias=b)
        hs = torch.empty(S, HIDDEN, dtype=bf, device=device)
        for name, m in cases.items():
            r = _k2_route(runner, w, m, S)
            if ties:
                lg = r["logits"]
                assert torch.equal(lg[:, p], lg[:, q]), "one gate row, two logits"
            want_w, want_ids = _k3_router(m["bias"]).select_experts(hs, r["logits"])
            ids, wts = _by_expert(r["ids"], r["weights"])
            want_ids, want_w = _by_expert(want_ids, want_w)
            bad = (ids != want_ids).any(-1)
            if bad.any():
                off_ids[(S, name)] = {
                    t: (
                        sorted(set(ids[t].tolist()) - set(want_ids[t].tolist())),
                        sorted(set(want_ids[t].tolist()) - set(ids[t].tolist())),
                    )
                    for t in bad.nonzero().flatten()[:3].tolist()
                }
            ulps = (wts.view(torch.int32) - want_w.view(torch.int32)).abs()[~bad]
            if ulps.any():
                n = int((ulps > 0).any(-1).sum())
                off_w[(S, name)] = (n, int(ulps.max()))
    if check == "weights":
        assert not off_w, (
            f"(rows, case): (tokens with weight bits off, max f32 steps): {off_w}"
        )
    else:
        assert not off_ids, (
            f"(rows, case): {{token: (K2-only experts, router-only)}}: {off_ids}"
        )


@ray.remote(num_gpus=1, max_calls=1)
def _k2_route_worker(monkeypatch, tp_size, pp_size, rank, port) -> None:
    _k2_route_run(monkeypatch, tp_size, pp_size, rank, port, "ids")


@ray.remote(num_gpus=1, max_calls=1)
def _k2_weights_worker(monkeypatch, tp_size, pp_size, rank, port) -> None:
    _k2_route_run(monkeypatch, tp_size, pp_size, rank, port, "weights")


@ray.remote(num_gpus=1, max_calls=1)
def _k2_tie_worker(monkeypatch, tp_size, pp_size, rank, port) -> None:
    _k2_route_run(monkeypatch, tp_size, pp_size, rank, port, "ties")


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_k2_routing_matches_production_router_tp8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K2's route on its own router logits picks KimiMoE's router's experts
    (AITER biased_grouped_topk on an AITER MoE deployment): random scores, and a
    16th pick between two biased scores 1 or 2 f32 steps apart."""
    multi_process_parallel(monkeypatch, 8, 1, _k2_route_worker)


@needs_gfx950
@multi_gpu_test(num_gpus=8)
@pytest.mark.xfail(
    strict=True,
    reason="K2's routing weights (sigmoid as v_rcp(1 + v_exp2(-x log2 e)), "
    "renormalized by an IEEE 1 / sum) are up to 3 f32 steps off AITER "
    "biased_grouped_topk's",
)
def test_k2_routing_weights_match_production_router_tp8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On the same cases, the picked experts' weights are the router's, bit for
    bit."""
    multi_process_parallel(monkeypatch, 8, 1, _k2_weights_worker)


@needs_gfx950
@multi_gpu_test(num_gpus=8)
@pytest.mark.xfail(
    strict=True,
    reason="on an exact tie for the 16th pick K2 takes the lower expert id; AITER "
    "biased_grouped_topk took the higher one",
)
def test_k2_routing_ties_match_production_router_tp8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The router's experts on an exact tie for the 16th pick (two experts of one
    gate row)."""
    multi_process_parallel(monkeypatch, 8, 1, _k2_tie_worker)


@ray.remote(num_gpus=1, max_calls=1)
def _k2_situ_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SituAndMul
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import (
        BETA,
        LBETA,
        SI,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

    # forward_cuda is the _C situ_and_mul kernel (torch.ops itself does not
    # pickle into the Ray worker)
    with set_current_vllm_config(VllmConfig()):
        situ = SituAndMul(BETA, LBETA)
    runner = MonoRunner(device)
    mw = _moe_weights(rank, device)
    # gate | up row j: one x column times 2^e, e from -40 to 15 (exact): the
    # shared gate and up span every magnitude down to tiny
    j = torch.arange(2 * SI, device=device)
    sgu = torch.zeros(2 * SI, HIDDEN, device=device)
    sgu[j, (j * 37 + 11 * rank) % HIDDEN] = torch.exp2((j * 7 % 56 - 40).float())
    mw = dict(mw, w_sgu=sgu.to(bf))
    off = {}
    for S in (1, 16, 64):
        w = _k2_inputs(rank, device, S)
        runner.step_begin()
        out = torch.empty(S, HIDDEN, dtype=bf, device=device)
        key = _k2_launch(runner, w, mw, w["prefix"].clone(), out, False, opt=0)
        torch.accelerator.synchronize()
        assert not runner.health.any() and not runner.flags.any()
        r = _k2_regions(runner, key)
        want = situ.forward_cuda(r["sgu"])
        bad = r["h"].view(torch.int16) != want.view(torch.int16)
        if bad.any():
            g, u = r["sgu"][:, :SI][bad], r["sgu"][:, SI:][bad]
            off[S] = (int(bad.sum()), list(zip(g[:4].tolist(), u[:4].tolist())))
    assert not off, f"rows: (SiTU outputs off _C situ_and_mul, (g, u) of some): {off}"


@needs_gfx950
@multi_gpu_test(num_gpus=8)
@pytest.mark.xfail(
    strict=True,
    reason="K2's tanh_libm, (1 - e) / (1 + e) with e = exp2(-2 |x| log2 e), "
    "cancels for small |x| (tanhf does not): the SiTU of a small gate or up "
    "loses bf16 bits, and is 0 for the tiniest",
)
def test_k2_shared_situ_matches_c_kernel_tp8(monkeypatch: pytest.MonkeyPatch) -> None:
    """K2's shared-expert SiTU (`situ_vllm`) of its own gate | up rows is the _C
    situ_and_mul kernel's, bit for bit, at every magnitude of gate and up, tiny
    ones included (the random weights above rarely make them)."""
    multi_process_parallel(monkeypatch, 8, 1, _k2_situ_worker)


def _situ_grid(count: int, gen) -> torch.Tensor:
    """[count + 9, 2 * 65536] gate | up rows: every bf16 gate against sampled
    finite ups and special ones."""
    pats = torch.arange(-32768, 32768, dtype=torch.int32, device="cuda")
    pats = pats.to(torch.int16).view(bf)
    finite = pats[torch.isfinite(pats.float())]
    picks = torch.randint(0, finite.numel(), (count,), generator=gen, device="cuda")
    special = [0.0, -0.0, 1.0, -1.0, 25.0, -100.0, 1e-30, float("inf"), float("-inf")]
    ups = torch.cat([finite[picks], torch.tensor(special, dtype=bf, device="cuda")])
    gates = pats.unsqueeze(0).expand(ups.numel(), -1)
    return torch.cat([gates, ups.view(-1, 1).expand(-1, 65536)], dim=1).contiguous()


@needs_gfx950
@pytest.mark.xfail(
    strict=True,
    reason="_C situ_and_mul (K2 too) takes sigmoid(g) as (1 + tanh(g / 2)) / 2, "
    "which loses bf16 bits for g below about -8 and is 0 below about -18",
)
@torch.inference_mode()
def test_production_shared_situ_is_c_kernel() -> None:
    """K2's shared SiTU is to be the _C situ_and_mul kernel's (the TP8 test
    above), and so is vLLM's. MI355X serves Kimi-K3 compiled (the recipe's compilation
    mode 3: ROCm opts it out of breakable CUDA graphs), where custom ops are off,
    so KimiMLP's SituAndMul dispatches to its compiled forward_native: on every
    bf16 gate against sampled ups, its bits must be the _C kernel's."""
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.model_executor.layers.activation import SituAndMul
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import (
        BETA,
        LBETA,
    )

    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE, custom_ops=["+fused_rms_norm_gated"]
        )
    )
    with set_current_vllm_config(config):
        op = SituAndMul(BETA, LBETA)
        x = _situ_grid(16, torch.Generator(device="cuda").manual_seed(0))
        got = op(x)
    want = torch.empty_like(got)
    torch.ops._C.situ_and_mul(want, x, BETA, LBETA)
    nan = torch.isnan(want.float()) & torch.isnan(got.float())
    bad = (got.view(torch.int16) != want.view(torch.int16)) & ~nan
    assert not bad.any(), f"{int(bad.sum())} of {bad.numel()} SiTU outputs differ"


def _gather(t: torch.Tensor) -> torch.Tensor:
    """Every rank's `t` stacked [ranks, ...]."""
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t.contiguous())
    return torch.stack(out)


@ray.remote(num_gpus=1, max_calls=1)
def _m1_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import (
        ENTRY,
        GROWS,
        NH,
        MlaPreBuild,
        mla_pre,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla_back import (
        MlaPostBuild,
        mla_post,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

    R, nb = tp_size, 4
    # DCP8 over the TP group; the test's distributed init sets up no DCP group
    runner = MonoRunner(device, dcp=R)

    def pre(w, prefix, blocks, cache, q8, gate, layer=3):
        S = prefix.size(0)
        key = MlaPreBuild(
            tokens=S,
            nblocks=nb,
            delta=True,
            ranks=R,
            peer_off=runner.mla_peer_off,
            qheads=q8.size(1) if q8.size(1) != R * NH else 0,
        )
        mla_pre(
            key,
            prefix=prefix,
            delta=w["delta"],
            blocks=blocks,
            ares_nw=w["ares_nw"],
            ares_qk=w["ares_qk"],
            in_nw=w["in_nw"],
            w_qkv=w["w_qkv"],
            w_g=w["w_g"],
            q_nw=w["q_nw"],
            kv_nw=w["kv_nw"],
            w_qb=w["w_qb"],
            w_k=w["w_k"],
            w_k_scale=w["w_k_scale"],
            kv_cache=cache,
            slots=w["slots"],
            gate_out=gate,
            q_out=q8,
            layer=layer,
            **runner.launch_args(),
            **runner.peer_args(),
        )

    def post(b, q, gate, core, layer=3, ranks=R):
        key = MlaPostBuild(
            tokens=core.size(0), qlen=q, ranks=ranks, peer_off=runner.mla_post_off
        )
        mla_post(
            key,
            o=b["o"],
            lse=b["lse"],
            seq_lens=b["seq_lens"],
            qsl=b["qsl"],
            w_v=b["w_v"],
            w_v_scale=b["w_v_scale"],
            gate=gate,
            core=core,
            layer=layer,
            **runner.launch_args(),
            **runner.peer_args(),
        )

    from vllm.v1.attention.backends.mla import rocm_aiter_mla as am
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper

    # the head count cprr runs the gathered query at (128 for 96)
    qheads = am._asm_dcp_verify_heads(R * NH)
    assert qheads >= R * NH

    # M1a: this rank's query rows land at heads 12 * rank.. on every rank
    for S in (1, 3, 4, 8, 16, 32, 64):
        w = _mla_inputs(S, nb, rank, R)
        ref = _mla_reference(w, nb)
        runner.step_begin()
        prefix, blocks = w["prefix"].clone(), w["blocks"].clone()
        pages = w["pages"].clone()
        q8 = torch.zeros(S, R * NH, ENTRY, dtype=torch.uint8, device=device)
        gate = torch.zeros(S, GROWS, dtype=bf, device=device)
        pre(w, prefix, blocks, _page_view(pages), q8, gate)
        torch.accelerator.synchronize()
        assert not runner.health.any() and not runner.flags.any(), S
        torch.testing.assert_close(prefix, ref["prefix"], atol=0, rtol=0)
        want = _gather(ref["q"].view(torch.uint8)).transpose(0, 1).reshape_as(q8)
        errs = {
            "q": _rel_l2(_fp8_f(q8), _fp8_f(want)),
            "gate": _rel_l2(gate, ref["gate"]),
        }
        assert errs["q"] < 3e-2 and errs["gate"] < 1e-2, (S, errs)
        assert _cos(_fp8_f(q8), _fp8_f(want)) > 0.999, (S, errs)
        # this rank's tokens' rows; the other ranks' tokens' (slot -1) untouched
        _check_cache(pages, ref["pages"], w["slots"])
        # AITER cprr's padded query (96 -> 128 heads) equals the pad
        # forward_mqa builds itself (from the contiguous q8: a view of qp
        # would hand qp back). test_m1a_pad_through_aiter_cprr runs it.
        if qheads == R * NH:
            continue
        runner.step_begin()
        prefix, blocks = w["prefix"].clone(), w["blocks"].clone()
        qp = torch.zeros(S, qheads, ENTRY, dtype=torch.uint8, device=device)
        pre(w, prefix, blocks, _page_view(w["pages"].clone()), qp, gate)
        torch.accelerator.synchronize()
        assert not runner.health.any() and not runner.flags.any(), S
        pad = AiterMLAHelper.get_mla_padded_q(R * NH, q8, target_heads=qheads)
        assert pad.data_ptr() != qp.data_ptr()
        assert torch.equal(qp, pad), S

    # M1b: every sender's partials for this rank's heads, each with its own
    # empty-shard mask, combined in rank order
    for S, q in ((1, 1), (16, 1), (4, 4), (16, 4), (64, 4), (64, 8)):
        b = _mla_back_inputs(S, q, R, rank)
        runner.step_begin()
        core = torch.zeros(S, GROWS, dtype=bf, device=device)
        post(b, q, b["gate"], core)
        torch.accelerator.synchronize()
        assert not runner.health.any() and not runner.flags.any(), (S, q)
        heads = slice(NH * rank, NH * (rank + 1))
        lat = _a2a_combine(
            _gather(b["o"])[:, :, heads],
            _gather(b["lse"])[:, :, heads],
            _gather(b["seq_lens"]),
            b["qsl"],
            q,
        )
        ref = _uv_gate(lat, b)
        if S // q > 2:  # the last request no rank holds
            assert not core[-q:].any(), (S, q)
        err = _rel_l2(core, ref)
        assert err < 1e-2 and _cos(core, ref) > COS_MIN, (S, q, err)

    def m1b_in(b, gate):
        keys = ("o", "lse", "seq_lens", "qsl", "w_v", "w_v_scale")
        return dict({n: b[n] for n in keys}, gate=gate)

    # K2 with M1b in its launch (``K2Build.m1b_qlen``): core, prefix and the
    # MoE output bit-identical to M1b's launch then K2's, at DCP8 and DCP1
    mw = _moe_weights(rank, device)
    for S, q in ((1, 1), (8, 1), (16, 1), (8, 8), (16, 8), (16, 4), (64, 4)):
        k = _k2_inputs(rank, device, S)
        for ranks in (R, 1):
            b = _mla_back_inputs(S, q, ranks, rank)
            if ranks == 1:
                b["lse"] = None
            got = []
            for fused in (False, True):
                runner.step_begin()
                core = torch.zeros(S, GROWS, dtype=bf, device=device)
                prefix = k["prefix"].clone()
                out = torch.empty(S, HIDDEN, dtype=bf, device=device)
                if not fused:
                    post(b, q, b["gate"], core, ranks=ranks)
                f = (m1b_in(b, b["gate"]), q, ranks) if fused else None
                _k2_launch(runner, dict(k, core=core), mw, prefix, out, False, m1b=f)
                torch.accelerator.synchronize()
                assert not runner.health.any() and not runner.flags.any(), (S, q)
                got.append((core, prefix, out))
            for x, y in zip(*got):
                assert torch.equal(x, y), (S, q, ranks)

    # DCP1, M1c's partials (``mla_attn``) merged by K2's M1b (``splits``), then
    # M1c in K2's launch too (``K2Build.m1c``): bit-identical, in a graph of two
    # layers replayed
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import mla_attn

    gk = torch.Generator(device=device).manual_seed(11)
    kv8 = (torch.randn(100_000, ENTRY, generator=gk, device=device) * 0.5).to(
        torch.float8_e4m3fn
    )

    def m1c_case(kv8, idx, qb, q, lens):
        S = qb.size(0)
        k = _k2_inputs(rank, device, S)
        b = _mla_back_inputs(S, q, 1, rank)
        indptr = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()],
                              dtype=torch.int32, device=device)  # fmt: skip
        b["seq_lens"] = indptr.diff().int()
        c_in = dict(q=qb, kv=kv8, pages=idx.int(), indptr=indptr,
                    scale=1 / 192**0.5, kv_scale=0.75)  # fmt: skip
        key = mla_attn.MlaAttnBuild(tokens=S, qlen=q)
        so, sl = mla_attn.partial_shapes(key)
        n = mla_attn.splits(q)
        po = torch.empty(so, dtype=bf, device=device)
        pl = torch.empty(sl, dtype=torch.float32, device=device)
        b.update(o=po[0], lse=pl[0])
        core = torch.zeros(S, GROWS, dtype=bf, device=device)
        prefix = torch.empty_like(k["prefix"])
        out = torch.empty(S, HIDDEN, dtype=bf, device=device)
        k2w = dict(k, core=core)
        got = []
        for fused in (False, True):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                runner.step_begin()
                prefix.copy_(k["prefix"])
                for layer in (3, 5):
                    if not fused:
                        mla_attn.mla_attn(key, part_o=po, part_lse=pl, **c_in)
                    _k2_launch(
                        runner, k2w, mw, prefix, out, False, layer=layer,
                        m1b=(m1b_in(b, b["gate"]), q, 1), splits=n,
                        m1c_in=c_in if fused else None,
                    )  # fmt: skip
            graph.replay()
            torch.accelerator.synchronize()
            want = [t.clone() for t in (core, prefix, out)]
            for _ in range(50):
                graph.replay()
            torch.accelerator.synchronize()
            assert not runner.health.any() and not runner.flags.any(), (S, q, fused)
            for x, y in zip((core, prefix, out), want):
                assert torch.equal(x, y), (S, q, fused)
            got.append(want)
        for x, y in zip(*got):
            assert torch.equal(x, y), (S, q)
        return got[0]

    for S, q, lens in (
        (8, 8, [8704]),
        (16, 8, [3000, 0]),
        (16, 4, [900, 17, 5000, 64]),
    ):
        idx = torch.randperm(kv8.size(0), generator=gk, device=device)[: sum(lens)]
        m1c_case(kv8, idx, _randn(gk, S, NH, ENTRY, scale=0.5), q, lens)

    # the same rows past 2^32 bytes of a wide pool (``K2Build.m1c_wide``)
    q, lens = 8, [3000, 600]
    idx = torch.randperm(kv8.size(0), generator=gk, device=device)[: sum(lens)]
    qb = _randn(gk, q * len(lens), NH, ENTRY, scale=0.5)
    want = m1c_case(kv8, idx, qb, q, lens)
    pool, high = _m1c_high_pool(kv8, idx, M1C_ROWS_4G - 1000, M1C_ROWS_4G + 4096, gk)
    assert mla_attn.wide_kv(pool)
    for x, y in zip(m1c_case(pool, high, qb, q, lens), want):
        assert torch.equal(x, y)
    del pool

    # A graph of one step, two MLA layers of M1a -> M1b -> K2 (each layer its
    # own tags, the peer regions reused), replayed bit-identical; then the
    # same with M1b in K2's launch, equal to it
    S, q = 64, 4
    w = _mla_inputs(S, nb, rank, R)
    b = _mla_back_inputs(S, q, R, rank)
    k = _k2_inputs(rank, device, S)
    prefix, blocks = torch.empty_like(w["prefix"]), torch.empty_like(w["blocks"])
    pages = w["pages"].clone()
    cache = _page_view(pages)
    q8 = torch.zeros(S, R * NH, ENTRY, dtype=torch.uint8, device=device)
    gate = torch.zeros(S, GROWS, dtype=bf, device=device)
    core = torch.zeros(S, GROWS, dtype=bf, device=device)
    out = torch.empty(S, HIDDEN, dtype=bf, device=device)
    wants = []
    for fused in (False, True):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runner.step_begin()
            prefix.copy_(w["prefix"])
            blocks.copy_(w["blocks"])
            for layer in (3, 5):
                pre(w, prefix, blocks, cache, q8, gate, layer)
                if not fused:
                    post(b, q, gate, core, layer)
                f = (m1b_in(b, gate), q, R) if fused else None
                k2w = dict(k, core=core, blocks=blocks)
                _k2_launch(runner, k2w, mw, prefix, out, False, layer=layer, m1b=f)
        graph.replay()
        torch.accelerator.synchronize()
        want = [t.clone() for t in (prefix, q8, core, out, pages)]
        for _ in range(300):
            graph.replay()
        torch.accelerator.synchronize()
        for x, y in zip((prefix, q8, core, out, pages), want):
            assert torch.equal(x, y), fused
        assert not runner.health.any() and not runner.flags.any()
        # the two layers write the same rows, and only those
        _check_rows_only(pages, w["pages"], w["slots"])
        wants.append(want)
    for x, y in zip(*wants):
        assert torch.equal(x, y)


@needs_gfx950
@pytest.mark.parametrize("qlen", [4, 8])
@torch.inference_mode()
def test_m1a_pad_through_aiter_cprr(monkeypatch: pytest.MonkeyPatch, qlen: int) -> None:
    """AITER's cprr asm decode, at DCP8 (96 -> 128 heads), on the padded query
    M1a writes (mla.attend hands it over) equals the decode on the pad
    forward_mqa builds itself, and both match attention in torch. With
    _m1_worker's M1a == get_mla_padded_q, an AITER change the hand-off misses
    fails here. One GPU, a DCP rank simulated (the cprr numerics test's
    harness: real metadata builder, real kernel)."""
    from types import SimpleNamespace

    import vllm.distributed.parallel_state as ps
    import vllm.model_executor.layers.attention.mla_attention as mla_attention
    from tests.v1.attention import test_rocm_aiter_mla_dcp_cprr_numerics as cp
    from vllm.config import set_current_vllm_config
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import NH
    from vllm.models.kimi_k3.amd.specialized_decode.runtime import mla
    from vllm.v1.attention.backends.mla import rocm_aiter_mla as am
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    dcp, rank, nr, ctx = 8, 3, 3, 700
    n = dcp * NH
    H = am._asm_dcp_verify_heads(n)
    if n == H:
        pytest.skip(f"cprr runs {n} heads natively: M1a writes no pad")
    # _install_fake_groups patches these; monkeypatch restores them
    for mod, name in (
        (ps, "get_dcp_group"),
        (ps, "get_tp_group"),
        (am, "get_dcp_group"),
        (mla_attention, "get_dcp_group"),
    ):
        monkeypatch.setattr(mod, name, getattr(mod, name))
    monkeypatch.setattr(cp, "QLEN", qlen)

    device = torch.device("cuda", 0)
    torch.accelerator.set_device_index(device.index)
    g = torch.Generator(device=device).manual_seed(0)
    cp._ensure_workspace_manager(device)
    cp._install_fake_groups(dcp, rank)
    config = cp._make_vllm_config(NH)
    with set_current_vllm_config(config):
        name = "model.layers.0.self_attn.attn"
        config.compilation_config.static_forward_context[name] = cp._make_ctx_layer(
            config, monkeypatch
        )
        spec = MLAAttentionSpec(
            block_size=1,
            num_kv_heads=1,
            head_size=cp.HEAD_SIZE,
            dtype=config.model_config.dtype,
        )
        L = ctx + qlen
        fp8 = current_platform.fp8_dtype()
        kv = _randn(g, nr * L, 1, cp.HEAD_SIZE, dtype=torch.float32).to(fp8)
        q8 = _randn(g, nr * qlen, n, cp.HEAD_SIZE, scale=0.5, dtype=torch.float32)
        q8 = q8.to(fp8)
        pos = torch.arange(rank, L, dcp, device=device)  # rank's residue class
        # forward_mqa pads q8 itself
        md, o, lse, _ = cp._run_shard(
            config, spec, name, device, rank, rank, q8, kv, nr, qlen, ctx, dcp, pos
        )
        assert md.decode.dcp_route is am._DCPDecodeRoute.CPRR
        assert md.decode.asm_decode_num_heads == H
        # M1a's buffer (_m1_worker: M1a == get_mla_padded_q of its q8)
        qp = torch.empty(nr * qlen, H, cp.HEAD_SIZE, dtype=fp8, device=device)
        qp.copy_(am.AiterMLAHelper.get_mla_padded_q(n, q8, target_heads=H))
        layer = cp._Layer(device)
        layer.impl = cp._build_impl(NH, dcp, rank)  # type: ignore[attr-defined]
        o_m1a, lse_m1a = mla.attend(SimpleNamespace(mla_attn=layer), qp, kv, md)
        torch.accelerator.synchronize()

    assert torch.equal(o_m1a, o) and torch.equal(lse_m1a, lse)
    want, want_lse = cp._torch_reference(
        q8.float().view(nr, qlen, n, -1),
        kv.float().view(nr, L, -1),
        ctx,
        qlen,
        cp.SCALE,
        pos,
    )
    o = o.float().reshape(want.shape)
    err = (o - want).abs().max().item() / want.abs().max().item()
    assert err < cp.SHARD_RTOL, err
    assert _rel_l2(o, want) < 3e-2
    torch.testing.assert_close(
        lse.float().reshape(want_lse.shape), want_lse, atol=5e-2, rtol=0
    )


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_m1_tp8_dcp8_matches_vllm_ops_and_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _m1_worker)


STALLED = 3


@ray.remote(num_gpus=1, max_calls=1)
def _fault_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.debug import (
        given_up_waits,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.sync import WAIT_TICKS
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    # rank STALLED's CTA 0 stalls past every wait's bound in the epoch 2 launch
    runner.fault = {
        "fault_rank": STALLED,
        "fault_epoch": 2,
        "fault_layer": 128 + 3,  # K2's slot of layer 3 (_k2_launch's default)
        "fault_ticks": WAIT_TICKS * 3 // 2,
    }
    mw = _moe_weights(rank, device)
    w = _k2_inputs(rank, device)
    out = torch.empty(8, HIDDEN, dtype=bf, device=device)
    runner.step_begin()  # epoch 1: no stall
    _k2_launch(runner, w, mw, w["prefix"].clone(), out, reset=True)
    torch.accelerator.synchronize()
    assert not runner.flags.any()

    runner.step_begin()  # epoch 2: rank STALLED stalls; every launch ends
    _k2_launch(runner, w, mw, w["prefix"].clone(), out, reset=True)
    torch.accelerator.synchronize()
    waits = given_up_waits(runner.health.cpu(), k2.REGIONS)
    gave_up = torch.tensor([int(bool(waits))], device=device)
    dist.all_reduce(gave_up)
    assert gave_up.item() > 0, "no rank's wait gave up"
    if rank == 0:
        assert runner.flags.any(), "the output rank's flags missed the failure"


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_stalled_rank_fails_the_step_without_a_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _fault_worker)


@needs_gfx950
def test_warmup_compiles_without_launching() -> None:
    """The JIT warm-up compiles K1, K2, M1a and M1b before capture without running them:
    a launch on its null arguments would fault."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        KdaPreBuild,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import (
        MlaPreBuild,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla_back import (
        MlaPostBuild,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.layer import K2Build
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.layers import compile_build

    for s, q in ((8, 8), (16, 4), (64, 16)):
        compile_build(
            KdaPreBuild(tokens=s, qlen=q, nblocks=4, delta=True, state_len=3 + q - 1)
        )
        compile_build(K2Build(tokens=s, nblocks=4))
        compile_build(MlaPreBuild(tokens=s, nblocks=4, delta=True))
        compile_build(MlaPostBuild(tokens=s, qlen=q))
    torch.accelerator.synchronize()
