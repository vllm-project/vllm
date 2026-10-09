# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics of the Qwen3.8 mono decode kernels on MI355X against the ops vLLM
runs for the same decode step, random weights in the checkpoint's formats at
TP8 shapes.

- K1 (one GPU): input add + input_layernorm -> in_proj_qkvz / in_proj_ba ->
  AITER's conv update and fused gated delta rule -> RMSNormGated, the conv / SSM
  state each step leaves behind, pad rows.
- K2 (eight GPUs): o_proj + all-reduce + residual add + post_attention_layernorm
  + the sparse MoE block (softmax top-10 router, MXFP4 experts through aiter's
  ``fused_moe``, the sigmoid-gated shared expert) + the final all-reduce. The
  MoE is compared on the kernel's own norm output, where routing cannot differ.
"""

import pytest
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F

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


pytestmark = pytest.mark.skipif(
    not _on_gfx950(), reason="the Qwen3.8 mono kernels need gfx950 (MI355X)"
)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return F.cosine_similarity(a, b, dim=0).item()


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def _randn(g, *shape, scale=1.0, dtype=bf):
    return (torch.randn(*shape, generator=g, device=g.device) * scale).to(dtype)


def _all_reduce_fp32(t: torch.Tensor) -> torch.Tensor:
    """The ranks' bf16 partials summed in fp32, as vLLM's custom all-reduce."""
    t = t.float()
    dist.all_reduce(t)
    return t.to(bf)


# ---------------------------------------------------------------- K1


def _pairs(scratch, region, n):
    """A scratch mailbox region's first ``n`` values as f32."""
    off = region[0]
    return (
        scratch[off : off + n * 8]
        .view(torch.int32)
        .view(n, 2)[:, 0]
        .view(torch.float32)
    )


def _gdn_inputs(s: int, n_real: int, dim_first: bool, first: bool, slots: int = 16):
    from vllm.models.qwen3_5.amd.mono.layout import BA, CONV, HD, HIDDEN, NV, QKVZ

    g = torch.Generator(device="cuda").manual_seed(0)
    w = {
        "hidden": _randn(g, s, HIDDEN, scale=2.0),
        "residual": None if first else _randn(g, s, HIDDEN, scale=2.0),
        "ln_w": _randn(g, HIDDEN, scale=0.1),
        "w_qkvz": _randn(g, QKVZ, HIDDEN, scale=0.02),
        "w_ba": _randn(g, BA, HIDDEN, scale=0.05),
        "conv_w": _randn(g, CONV, 1, 4, scale=0.4),
        "a_log": _randn(g, NV, scale=0.5, dtype=torch.float32),
        "dt_bias": _randn(g, NV, scale=0.5),
        "norm_w": (1 + _randn(g, HD, scale=0.1, dtype=torch.float32)).to(bf),
    }
    # vLLM packs a layer's conv and SSM state in one padded page: strided slots
    pad = _randn(g, slots, NV * HD * HD + 8192, scale=0.05, dtype=torch.float32)
    w["rstate"] = pad[:, : NV * HD * HD].view(slots, NV, HD, HD)
    conv = _randn(g, slots, CONV * 3 + 1024)[:, : CONV * 3]
    if dim_first:
        w["conv_state"] = conv.view(slots, CONV, 3)
    else:
        w["conv_state"] = conv.view(slots, 3, CONV).transpose(-1, -2)
    # live rows on slots 3 ..; pad rows on the null block 0
    idx = torch.zeros(s, dtype=torch.int32, device="cuda")
    idx[:n_real] = torch.arange(3, 3 + n_real, dtype=torch.int32, device="cuda")
    w["st_idx"] = idx
    return w


def _gdn_reference(w, n: int):
    """Qwen3_5DecoderLayer's input norm and the GDN decode path on ROCm
    (``forward_hip`` -> ``_forward_core_decode_aiter`` -> ``_output_projection``
    without out_proj) on the first ``n`` rows; the rest are the cudagraph pad."""
    from aiter.ops.triton.causal_conv1d_update_single_token import (
        fused_reshape_causal_conv1d_update_single_token,
    )
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule,
    )

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNormGated
    from vllm.models.qwen3_5.amd.mono.layout import (
        CONV,
        CORE,
        HD,
        HIDDEN,
        KEY,
        NK,
        NV,
        VAL,
    )

    s = w["hidden"].size(0)
    with set_current_vllm_config(VllmConfig()):
        ln = GemmaRMSNorm(HIDDEN, eps=1e-6).cuda()
        norm = RMSNormGated(
            HD, eps=1e-6, norm_before_gate=True, activation="silu", device="cuda"
        )
    ln.weight.data = w["ln_w"].clone()
    norm.weight.data = w["norm_w"].clone()
    if w["residual"] is None:
        x, res = ln.forward_native(w["hidden"]), w["hidden"].clone()
    else:
        x, res = ln.forward_native(w["hidden"], w["residual"].clone())
    qkvz = F.linear(x, w["w_qkvz"])
    ba = F.linear(x, w["w_ba"])
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    z = torch.empty(s, NV, HD, dtype=bf, device="cuda")
    core = torch.empty(s, NV, HD, dtype=bf, device="cuda")
    mixed, b, a = fused_reshape_causal_conv1d_update_single_token(
        qkvz,
        n,
        NK,
        NV,
        HD,
        HD,
        ba,
        z,
        core,
        conv_state,
        w["conv_w"].view(CONV, 4),
        None,
        "silu",
        conv_state_indices=w["st_idx"][:n],
        validate_data=True,
        qkvz_layout="flat",
    )
    fused_rearrange_sigmoid_gated_delta_rule(
        A_log=w["a_log"],
        a=a,
        b=b,
        dt_bias=w["dt_bias"],
        qkv=mixed,
        key_dim=KEY,
        value_dim=VAL,
        head_k_dim=HD,
        head_v_dim=HD,
        initial_state=rstate,
        inplace_final_state=True,
        cu_seqlens=torch.arange(n + 1, dtype=torch.int32, device="cuda"),
        ssm_state_indices=w["st_idx"][:n],
        use_qk_l2norm_in_kernel=True,
        core_attn_out=core.reshape(-1),
    )
    return {
        "x": x,
        "residual": res,
        "proj": torch.cat([qkvz, ba], dim=-1),
        "conv_state": conv_state,
        "rstate": rstate,
        "core": norm.forward_native(core, z).reshape(s, CORE),
    }


@pytest.mark.parametrize("dim_first", [True, False])
@pytest.mark.parametrize(
    "s,n_real,first", [(1, 1, False), (5, 4, False), (8, 8, False), (8, 7, True)]
)
def test_k1_matches_vllm_ops(s: int, n_real: int, first: bool, dim_first: bool) -> None:
    """K1 against GemmaRMSNorm -> in_proj -> AITER's conv update + fused gated
    delta rule -> RMSNormGated on one decode step, the states it leaves behind,
    and the pad rows (null block: no state written, zero core)."""
    from vllm.models.qwen3_5.amd.mono import gdn
    from vllm.models.qwen3_5.amd.mono.layout import CORE, HIDDEN

    w = _gdn_inputs(s, n_real, dim_first, first)
    ref = _gdn_reference(w, n_real)
    key = gdn.K1Build(tokens=s, first=first)
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    res_out = torch.empty(s, HIDDEN, dtype=bf, device="cuda")
    core = torch.full((s, CORE), float("nan"), dtype=bf, device="cuda")
    scratch = torch.zeros(gdn.scratch_bytes(key), dtype=torch.uint8, device="cuda")
    gdn.gdn_pre(
        key,
        hidden=w["hidden"],
        residual=w["residual"],
        res_out=res_out,
        ln_w=w["ln_w"],
        w_qkvz=w["w_qkvz"],
        w_ba=w["w_ba"],
        conv_w=w["conv_w"],
        conv_state=conv_state,
        a_log=w["a_log"],
        dt_bias=w["dt_bias"],
        norm_w=w["norm_w"],
        rstate=rstate,
        st_idx=w["st_idx"],
        core=core,
        scratch=scratch,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        layer=0,
    )
    torch.accelerator.synchronize()

    lay = gdn.scratch_layout(key)
    torch.testing.assert_close(res_out, ref["residual"], atol=0, rtol=0)
    x = scratch[lay["xrow"][0] : lay["xrow"][0] + s * HIDDEN * 2].view(bf)
    assert _cos(x, ref["x"]) > COS_MIN
    proj = _pairs(scratch, lay["proj"], s * gdn.NPROJ).view(s, gdn.NPROJ)
    assert _cos(proj, ref["proj"]) > COS_MIN
    # the GEMV's summation order flips a few bf16 roundings of the conv input
    errs = {
        "conv_state": _rel_l2(conv_state, ref["conv_state"]),
        "rstate": _rel_l2(rstate, ref["rstate"]),
        "core": _rel_l2(core[:n_real], ref["core"][:n_real]),
    }
    assert errs["conv_state"] < 5e-3 and errs["rstate"] < 2e-3, errs
    assert _cos(core[:n_real], ref["core"][:n_real]) > COS_MIN, errs
    assert torch.equal(core[n_real:], torch.zeros_like(core[n_real:]))
    # the null block is left alone
    torch.testing.assert_close(rstate[0], w["rstate"][0], atol=0, rtol=0)
    torch.testing.assert_close(conv_state[0], w["conv_state"][0], atol=0, rtol=0)


# ---------------------------------------------------------------- K2


def _moe_weights(rank: int, device):
    """One layer's MoE block at TP8 as vLLM holds it after loading: bf16 router,
    shared expert and its gate; the routed experts' MXFP4 through the
    ``AITER_MXFP4_MXFP4`` load path (``shuffle_weights``, ``e8m0_shuffle``)."""
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.qwen3_5.amd.mono.layout import HIDDEN, RI, SI, E

    gc = torch.Generator(device=device).manual_seed(7)  # replicated weights
    gr = torch.Generator(device=device).manual_seed(100 + rank)  # TP shards

    def u8(g, *shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=g, device=device).to(torch.uint8)

    fp4, e8 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    w13, w2 = rocm_aiter_ops.shuffle_weights(
        u8(gr, E, 2 * RI, HIDDEN // 2).view(fp4), u8(gr, E, HIDDEN, RI // 2).view(fp4)
    )
    w13s = u8(gr, E * 2 * RI, HIDDEN // 32, lo=118, hi=123).view(e8)
    w2s = u8(gr, E * HIDDEN, RI // 32, lo=118, hi=123).view(e8)
    return {
        "w_gate": _randn(gc, E, HIDDEN, scale=0.02),
        "w_sg": _randn(gc, 1, HIDDEN, scale=0.02),
        "w_sgu": _randn(gr, 2 * SI, HIDDEN, scale=0.02),
        "w_sd": _randn(gr, HIDDEN, SI, scale=0.03),
        "w13": w13,
        "w2": w2,
        "w13s": e8m0_shuffle(w13s).view(E, 2 * RI, -1),
        "w2s": e8m0_shuffle(w2s).view(E, HIDDEN, -1),
    }


def _moe_reference(x, mw):
    """Qwen3NextSparseMoeBlock's decode path on ``x``: the bf16 router's softmax
    top-10 renormalized, aiter's MXFP4 experts on bf16 activations, the shared
    expert (SiluAndMul) times sigmoid(shared_expert_gate), the all-reduce."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    from vllm.models.qwen3_5.amd.mono.layout import SI, TOPK

    logits = F.linear(x, mw["w_gate"])
    probs = torch.softmax(logits.float(), dim=-1)
    wts, ids = torch.topk(probs, TOPK, dim=-1, sorted=True)
    wts = wts / wts.sum(-1, keepdim=True)
    gu = F.linear(x, mw["w_sgu"])
    h = (F.silu(gu[:, :SI].float()) * gu[:, SI:].float()).to(bf)
    shared = torch.sigmoid(F.linear(x, mw["w_sg"])) * F.linear(h, mw["w_sd"])
    routed = fused_moe(
        x,
        mw["w13"],
        mw["w2"],
        wts.float(),
        ids.to(torch.int32),
        None,
        ActivationType.Silu,
        QuantType.per_1x32,
        False,
        mw["w13s"],
        mw["w2s"],
        None,
        None,
        dtype=bf,
    )
    return _all_reduce_fp32(routed + shared)


@ray.remote(num_gpus=1, max_calls=1)
def _k2_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
    from vllm.models.qwen3_5.amd.mono import layer as k2
    from vllm.models.qwen3_5.amd.mono.layout import CORE, HIDDEN, MAX_TOKENS

    tp = get_tp_group()
    peers = PeerBuffer(k2.peer_bytes(MAX_TOKENS), tp.cpu_group, rank, tp_size, device)
    peers.bytes.zero_()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    mw = _moe_weights(rank, device)
    gc = torch.Generator(device=device).manual_seed(11)
    gr = torch.Generator(device=device).manual_seed(200 + rank)
    w_o = _randn(gr, HIDDEN, CORE, scale=0.02)
    ln_w = _randn(gc, HIDDEN, scale=0.1)

    # both epoch parities, a layer slot each; every decode width's build
    for step, s in enumerate((1, 3, 8, 8)):
        epoch.add_(1)
        key = k2.K2Build(tokens=s)
        core = _randn(gr, s, CORE)
        residual = _randn(gc, s, HIDDEN, scale=2.0)
        scratch = torch.zeros(k2.scratch_bytes(key), dtype=torch.uint8, device=device)
        out = torch.empty(s, HIDDEN, dtype=bf, device=device)
        res_out = torch.empty_like(out)
        k2.layer_post(
            key,
            core=core,
            residual=residual,
            w_o=w_o,
            ln_w=ln_w,
            out=out,
            res_out=res_out,
            scratch=scratch,
            peers=peers.addresses,
            rank=rank,
            epoch=epoch,
            layer=step,
            **mw,
        )
        torch.accelerator.synchronize()

        res_ref = _all_reduce_fp32(F.linear(core, w_o)) + residual
        # the kernel sums the ranks in rank order, NCCL in its own order
        assert _rel_l2(res_out, res_ref) < 1e-2, s
        v = res_ref.float()
        x_ref = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
        x_ref = (x_ref * (1.0 + ln_w.float())).to(bf)
        lay = k2.scratch_layout(key)["xrow"][0]
        x = scratch[lay : lay + s * HIDDEN * 2].view(bf).view(s, HIDDEN)
        assert _cos(x, x_ref) > COS_MIN, s
        assert _cos(out, _moe_reference(x.clone(), mw)) > COS_MIN, s
        outs = [torch.empty_like(out) for _ in range(tp_size)]
        dist.all_gather(outs, out)
        assert all(torch.equal(outs[0], o) for o in outs), "ranks disagree"
    peers.close()


@multi_gpu_test(num_gpus=8)
def test_k2_tp8_matches_vllm_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _k2_worker)
