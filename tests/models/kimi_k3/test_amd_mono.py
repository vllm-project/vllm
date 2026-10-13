# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics of the Kimi-K3 mono decode kernels (``VLLM_ROCM_MONO_DECODE``) on MI355X
against the ops vLLM runs for the same spec-verify step, random weights in the
checkpoint's formats at TP8 shapes.

- K1 (one GPU): AttnRes -> in_proj -> f_b -> conv -> KDA recurrence -> gated
  norm, the conv / SSM state updates and a block-write layer's block store.
- K2 and the standalone MoE launch (eight GPUs): o_proj + all-reduce + the MLP
  AttnRes + the latent MoE, including independently computed reference routing.
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


pytestmark = pytest.mark.skipif(
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


# ---------------------------------------------------------------- K1


def _kda_inputs(
    nb: int, nacc: int, dim_first: bool, S: int, slots: int = 16, qlen: int = 8
):
    from vllm.models.kimi_k3.amd.mono.attention.kda import (
        HD,
        HIDDEN,
        NH,
        NPROJ,
        PROJ,
        QKV,
    )

    g = torch.Generator(device="cuda").manual_seed(0)
    sl = 3 + (qlen - 1)
    slots = max(slots, S + 8)
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
    w["st_idx"] = torch.arange(3, 3 + S, dtype=torch.int32, device="cuda").view(
        S // qlen, qlen
    )
    w["num_acc"] = torch.full((S // qlen,), nacc, dtype=torch.int32, device="cuda")
    if qlen < S:
        w["num_acc"] = (
            1 + torch.arange(S // qlen, device="cuda", dtype=torch.int32) % qlen
        )
    return w


def _kda_reference(w, nb: int, S: int, write_idx: int, norm=None, inplace=False):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.models.kimi_k3.amd.mono.attention.kda import HD, NH, PROJ, QKV
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res
    from vllm.models.kimi_k3.amd.ops.third_party.kda import fused_recurrent_kda
    from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

    prefix = w["prefix"] if inplace else w["prefix"].clone()
    blocks = w["blocks"] if inplace else w["blocks"].clone()
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
    proj = F.linear(x, w["w_in"])
    mixed_qkv, g2, f_a, beta = proj.split([QKV, PROJ, HD, NH + 4], dim=-1)[:4]
    g1 = F.linear(f_a, w["w_fb"])
    conv_state = w["conv_state"] if inplace else w["conv_state"].clone()
    rstate = w["rstate"] if inplace else w["rstate"].clone()
    qlen = w["st_idx"].size(1)
    qsl = torch.arange(0, S + 1, qlen, dtype=torch.int32, device="cuda")
    conv = causal_conv1d_update(
        mixed_qkv.contiguous(),
        conv_state,
        w["conv_w"],
        None,
        activation="silu",
        conv_state_indices=w["st_idx"][:, 0].contiguous(),
        num_accepted_tokens=w["num_acc"],
        query_start_loc=qsl,
        max_query_len=qlen,
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
        uniform_sequence_length=qlen,
        out=core,
    )
    if norm is None:
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


@pytest.mark.parametrize(
    "dim_first,nacc,write,S,qlen,state_rows,pipeline",
    [
        (dim_first, nacc, write, 8, 8, 0, 0)
        for dim_first in (True, False)
        for nacc in (1, 3)
        for write in (False, True)
    ]
    + [
        (True, 1, False, 16, 8, 0, 0),
        (False, 1, True, 32, 8, 32, 0),
        (True, 1, False, 64, 8, 64, 0),
        (False, 1, True, 24, 6, 0, 0),
        (False, 1, True, 18, 6, 0, 0),
        (False, 1, True, 50, 5, 0, 0),
        (False, 1, True, 15, 3, 0, 0),
        (False, 1, True, 6, 6, 0, 7),
        (True, 3, False, 12, 6, 0, 7),
        (False, 1, True, 18, 6, 0, 7),
        (True, 1, True, 24, 6, 0, 7),
        (False, 1, False, 16, 8, 0, 7),
        (True, 1, True, 24, 8, 0, 7),
        (False, 1, False, 64, 8, 64, 7),
        (True, 3, False, 12, 6, 0, 8),
        (False, 1, True, 18, 6, 0, 8),
        (True, 1, True, 24, 6, 0, 8),
        (False, 1, False, 16, 8, 0, 8),
        (True, 1, True, 24, 8, 32, 8),
        (False, 1, False, 64, 8, 64, 8),
        (True, 3, False, 12, 6, 0, 512),
        (False, 1, True, 18, 6, 0, 512),
        (False, 3, True, 18, 6, 0, 520),
        (False, 1, True, 24, 6, 0, 1024),
        (True, 3, False, 16, 8, 0, 1024),
        (False, 1, True, 24, 8, 0, 1024),
        (True, 1, False, 64, 8, 64, 1024),
        (False, 1, True, 24, 8, 0, 1536),
        (False, 3, True, 6, 6, 0, 8192),
        (True, 3, False, 12, 6, 0, 8192),
        (False, 1, True, 18, 6, 0, 8192),
        (True, 1, True, 24, 6, 0, 8192),
        (False, 1, False, 16, 8, 0, 8192),
        (True, 3, True, 24, 8, 0, 8192),
        (False, 1, False, 64, 8, 64, 8192),
        (False, 3, True, 6, 6, 0, 16384),
        (True, 3, False, 12, 6, 0, 16384),
        (False, 1, True, 18, 6, 0, 16384),
        (True, 1, True, 24, 6, 0, 16384),
        (False, 1, False, 16, 8, 0, 16384),
        (True, 3, True, 24, 8, 0, 16384),
        (False, 1, False, 64, 8, 64, 16384),
        pytest.param(True, 3, False, 12, 6, 0, 35667968, id="v3-packed-bank-m12"),
        pytest.param(False, 1, True, 18, 6, 0, 33570816, id="v3-packed-m18"),
        pytest.param(True, 3, True, 24, 6, 0, 2113536, id="v3-bank-m24"),
        pytest.param(False, 1, False, 16, 8, 0, 33562624, id="v3-packed64-m16"),
        pytest.param(True, 1, True, 24, 8, 32, 35667968, id="v3-packed-bank-v32"),
        pytest.param(False, 3, False, 64, 8, 64, 33570816, id="v3-packed-m64"),
        pytest.param(True, 3, False, 12, 6, 16, 102776836, id="v5-token-output-m12"),
        pytest.param(False, 1, True, 16, 8, 16, 100671492, id="v5-token-output-m16"),
        pytest.param(True, 3, False, 12, 6, 16, 236994564, id="v5-mailbox-output-m12"),
        pytest.param(False, 1, True, 16, 8, 16, 234889220, id="v5-mailbox-output-m16"),
    ],
)
def test_k1_matches_vllm_ops(
    dim_first: bool,
    nacc: int,
    write: bool,
    S: int,
    qlen: int,
    state_rows: int,
    pipeline: int,
) -> None:
    """K1 against attn_res -> in_proj -> f_b -> causal_conv1d_update ->
    fused_recurrent_kda -> FusedRMSNormGated on one spec-verify step (one
    or more requests), including the state each step leaves behind."""
    from vllm.models.kimi_k3.amd.mono.attention.kda import (
        PROJ,
        KdaPreBuild,
        kda_pre,
        scratch_bytes,
    )

    nb = 4
    # a block-write layer appends block nb (block_write_idx == prev_valid_blocks)
    write_idx = nb if write else -1
    w = _kda_inputs(nb, nacc, dim_first, S, qlen=qlen)
    ref = _kda_reference(w, nb, S, write_idx)

    key = KdaPreBuild(
        tokens=S,
        qlen=qlen,
        nblocks=nb,
        delta=True,
        state_len=w["conv_state"].size(-1),
        write_idx=write_idx,
        state_rows=state_rows,
        pipeline=pipeline,
        head_workers=24 if pipeline & (1 << 26) else 16,
    )
    weight = w["w_in"]
    if pipeline & 33554432:
        from vllm.models.kimi_k3.amd.mono.attention.parallel import (
            pack_projection_weights,
        )

        weight = pack_projection_weights(weight)
    prefix, blocks = w["prefix"].clone(), w["blocks"].clone()
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    guarded = torch.full((S * PROJ + 512,), 123, dtype=bf, device="cuda")
    core = guarded[256:-256].view(S, PROJ)
    kda_pre(
        key,
        prefix=prefix,
        delta=w["delta"],
        blocks=blocks,
        ares_nw=w["ares_nw"],
        ares_qk=w["ares_qk"],
        in_nw=w["in_nw"],
        w_in=weight,
        w_fb=w["w_fb"],
        conv_w=w["conv_w"],
        conv_state=conv_state,
        a_log=w["a_log"],
        dt_bias=w["dt_bias"],
        on_w=w["on_w"],
        rstate=rstate,
        st_idx=w["st_idx"],
        num_acc=w["num_acc"],
        core_out=core,
        scratch=torch.zeros(scratch_bytes(key), dtype=torch.uint8, device="cuda"),
        layer=0,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
    )
    torch.accelerator.synchronize()

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
    assert torch.all(
        F.cosine_similarity(core.float(), ref["core"].float(), dim=1) > COS_MIN
    )
    assert torch.all(guarded[:256] == 123)
    assert torch.all(guarded[-256:] == 123)
    touched = w["st_idx"].flatten()
    untouched = torch.ones(rstate.size(0), dtype=torch.bool, device="cuda")
    untouched[touched] = False
    torch.testing.assert_close(
        rstate[untouched], w["rstate"][untouched], atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "S,qlen", [(6, 6), (12, 6), (18, 6), (24, 6), (16, 8), (24, 8), (64, 8)]
)
@pytest.mark.parametrize("serialized_offload", [False, True])
def test_k1_serving_dispatch_reuses_packed_weights(
    monkeypatch, S, qlen, serialized_offload
):
    """Serving updates state outside the model-initialization config context."""
    from types import SimpleNamespace

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CUDAGraphMode
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.models.kimi_k3.amd import mono_decode
    from vllm.models.kimi_k3.amd.mono import runner
    from vllm.models.kimi_k3.amd.mono.attention.kda import HD, NH, PROJ
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    w = _kda_inputs(4, 1, False, S, qlen=qlen)
    conv = w["conv_state"]
    attn = SimpleNamespace(
        prefix="test-k1",
        local_num_heads=NH,
        head_dim=HD,
        conv_size=4,
        use_local_post=False,
        in_proj_qkvgfab=SimpleNamespace(weight=w["w_in"]),
        f_b_proj=SimpleNamespace(weight=w["w_fb"]),
        o_norm=SimpleNamespace(weight=w["on_w"], eps=1e-5),
        conv1d=SimpleNamespace(weight=w["conv_w"]),
        A_log=w["a_log"],
        dt_bias=w["dt_bias"],
        gate_lower_bound=-5.0,
        kv_cache=(
            conv if is_conv_state_dim_first() else conv.transpose(-1, -2),
            w["rstate"],
        ),
    )
    layer = SimpleNamespace(
        self_attn=attn,
        layer_idx=0,
        prev_valid_blocks=4,
        is_block_write_layer=True,
        block_write_idx=4,
        input_layernorm=SimpleNamespace(weight=w["in_nw"], variance_epsilon=1e-5),
        self_attention_res_norm=SimpleNamespace(
            weight=w["ares_nw"], variance_epsilon=1e-5
        ),
        self_attention_res_proj=SimpleNamespace(weight=w["ares_qk"]),
    )
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=S // qlen,
        num_spec_decode_tokens=S,
        num_actual_tokens=S,
        spec_state_indices_tensor=w["st_idx"],
        spec_sequence_masks=torch.ones(S // qlen, dtype=torch.bool, device="cuda"),
        num_accepted_tokens=w["num_acc"],
        uniform_spec_sequence_length=qlen,
    )
    monkeypatch.setattr(
        mono_decode, "is_breakable_cudagraph_enabled", lambda: not serialized_offload
    )
    for name in ("scratch", "epoch", "queue"):
        monkeypatch.setattr(runner._State, name, None)
    monkeypatch.setattr(runner._State, "zero_bytes", 0)
    config = VllmConfig()
    if serialized_offload:
        monkeypatch.setenv("VLLM_ROCM_MONO_DECODE", "1")
        config.kv_transfer_config = KVTransferConfig(
            kv_connector="SimpleCPUOffloadConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"serialize_copies_with_compute": True},
        )
        config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    with set_current_vllm_config(config):
        mono = mono_decode.MonoKda(layer)
        assert mono.ok
        if serialized_offload:
            assert mono_decode.kda_enabled()
            assert not mono_decode.enabled()

    def call():
        runner.step_begin()
        mode = CUDAGraphMode.FULL if serialized_offload else CUDAGraphMode.NONE
        with set_forward_context(
            {attn.prefix: metadata}, config, cudagraph_runtime_mode=mode
        ):
            return mono.core(w["prefix"], w["blocks"], w["delta"], None)

    if serialized_offload:
        call()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = call()
    for step in range(2):
        w["num_acc"].fill_(1 + step)
        w["st_idx"].copy_(w["st_idx"].roll(1, 0))
        expected = _kda_reference(w, 4, S, 4)
        if serialized_offload:
            graph.replay()
        else:
            out = call()
        torch.accelerator.synchronize()
        assert out.shape == (S, PROJ)
        for field in ("prefix", "blocks"):
            torch.testing.assert_close(w[field], expected[field], rtol=0, atol=0)
        for field in ("conv_state", "rstate"):
            assert _rel_l2(w[field], expected[field]) < 1e-3
        assert _cos(out, expected["core"]) > COS_MIN


@pytest.mark.parametrize(
    "S,qlen,dim_first,route",
    [
        (rows, qlen, dim_first, route)
        for route in (
            "local_post",
            "parallel",
            "shared_head",
            "resident",
            "cached",
            "cached_wide",
            "v3_bank",
            "v3_packed",
            "v3_packed_bank",
        )
        for rows, qlen, dim_first in (
            (48, 6, False),
            (80 if route == "local_post" else 64, 8, True),
        )
    ],
)
def test_kda_graph_replays_changed_state_slots(S, qlen, dim_first, route):
    """Captured KDA reads fresh accepted offsets and ignores padded requests."""
    from vllm.models.kimi_k3.amd.mono.attention import kda, local_post
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

    nb = 4
    w = _kda_inputs(nb, 1, dim_first, S, qlen=qlen)
    original = {
        name: w[name].clone()
        for name in ("prefix", "blocks", "conv_state", "rstate", "st_idx")
    }
    out = torch.empty(S, kda.PROJ, dtype=bf, device="cuda")
    key = local_post.KdaPostBuild(S, qlen, w["conv_state"].size(-1))
    if route != "local_post":
        full_key = kda.KdaPreBuild(
            S,
            qlen,
            nb,
            True,
            key.state_len,
            write_idx=nb,
            pipeline={
                "shared_head": 8,
                "parallel": 7,
                "resident": 1024,
                "cached": 8192,
                "cached_wide": 16384,
                "v3_bank": 2113536,
                "v3_packed": 33570816,
                "v3_packed_bank": 35667968,
            }[route],
            state_rows=32 if route in ("shared_head", "v3_packed_bank") else 0,
        )
        weight = w["w_in"]
        if full_key.pipeline & 33554432:
            from vllm.models.kimi_k3.amd.mono.attention.parallel import (
                pack_projection_weights,
            )

            weight = pack_projection_weights(weight)
        scratch = torch.zeros(
            kda.scratch_bytes(full_key), dtype=torch.uint8, device="cuda"
        )
        epoch = torch.ones(1, dtype=torch.int32, device="cuda")

    def call():
        if route != "local_post":
            epoch.add_(1)
            kda.kda_pre(
                full_key,
                **{
                    name: weight if name == "w_in" else w[name]
                    for name in (
                        "prefix",
                        "delta",
                        "blocks",
                        "ares_nw",
                        "ares_qk",
                        "in_nw",
                        "w_in",
                        "w_fb",
                        "conv_w",
                        "conv_state",
                        "a_log",
                        "dt_bias",
                        "on_w",
                        "rstate",
                        "st_idx",
                        "num_acc",
                    )
                },
                core_out=out,
                scratch=scratch,
                layer=0,
                epoch=epoch,
            )
        else:
            x = attn_res(
                w["prefix"],
                w["delta"],
                w["blocks"],
                w["ares_nw"],
                w["ares_qk"],
                w["in_nw"],
                nb,
                nb,
                1e-5,
                1e-5,
            )
            projected = F.linear(x, w["w_in"])
            local_post.run(key, projected, w, out)

    call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    requests = S // qlen
    for step, active in enumerate((requests, requests - 3, requests - 1, requests)):
        for name in ("prefix", "blocks", "conv_state", "rstate"):
            w[name].copy_(original[name])
        w["st_idx"].copy_(original["st_idx"].roll(step, 0))
        w["st_idx"][active:].zero_()
        w["num_acc"].copy_(1 + (torch.arange(requests, device="cuda") + step) % qlen)
        w["num_acc"][active:].fill_(1)
        live = active * qlen
        inputs = dict(w)
        for name in ("prefix", "delta", "blocks"):
            inputs[name] = w[name][:live]
        inputs["st_idx"] = w["st_idx"][:active]
        inputs["num_acc"] = w["num_acc"][:active]
        expected = _kda_reference(inputs, nb, live, nb)
        graph.replay()
        torch.accelerator.synchronize()
        for name in ("prefix", "blocks"):
            torch.testing.assert_close(w[name][:live], expected[name], atol=0, rtol=0)
        for name in ("conv_state", "rstate"):
            assert _rel_l2(w[name], expected[name]) < 1e-3
        assert _cos(out[:live], expected["core"]) > COS_MIN
        assert torch.all(out[live:] == 0)


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
    """KimiMoE's decode path on ``x``: sigmoid router + bias top-16, the latent
    down, aiter's MXFP4 experts, the latent all-reduce + RMSNorm, this rank's
    up_proj rows into the shared experts' output, the final all-reduce."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    from vllm.models.kimi_k3.amd.mono.stages.moe import SI, TOPK, UP_N

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
    from vllm.models.kimi_k3.amd.mono.stages.moe import HIDDEN, LAT, RI, SI, UP_N, E

    gc = torch.Generator(device=device).manual_seed(7)  # replicated weights
    gr = torch.Generator(device=device).manual_seed(100 + rank)  # TP shards

    def u8(g, *shape, lo=0, hi=256):
        return torch.randint(
            lo, hi, shape, generator=g, device=device, dtype=torch.uint8
        )

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


def _check_k1_k2_serving_replay(rows, length, config, reset):
    """A captured serving call preserves state and the block-write prefix buffer."""
    from types import SimpleNamespace as NS

    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.config import CUDAGraphMode, set_current_vllm_config
    from vllm.distributed import get_tp_group
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.models.kimi_k3.amd import mono_decode
    from vllm.models.kimi_k3.amd.mono import layer as k2
    from vllm.models.kimi_k3.amd.mono import runner
    from vllm.models.kimi_k3.amd.mono.attention import kda as k1
    from vllm.models.kimi_k3.amd.mono.attention.parallel import pack_projection_weights
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    rank = get_tp_group().rank_in_group
    device = torch.device("cuda", torch.accelerator.current_device_index())
    w = _kda_inputs(4, 1, False, rows, qlen=length)
    mw = _moe_weights(rank, device)

    def norm(weight):
        return NS(weight=weight, variance_epsilon=1e-5, eps=1e-5)

    conv = w["conv_state"]
    gr = torch.Generator(device=device).manual_seed(200 + rank)
    attn = NS(
        prefix="test-overlap",
        local_num_heads=k1.NH,
        head_dim=k1.HD,
        conv_size=4,
        use_local_post=False,
        in_proj_qkvgfab=norm(w["w_in"]),
        f_b_proj=norm(w["w_fb"]),
        o_proj=norm(_randn(gr, k1.HIDDEN, k1.PROJ, scale=0.02)),
        o_norm=norm(w["on_w"]),
        conv1d=norm(w["conv_w"]),
        A_log=w["a_log"],
        dt_bias=w["dt_bias"],
        gate_lower_bound=-5.0,
        kv_cache=(
            conv if is_conv_state_dim_first() else conv.transpose(-1, -2),
            w["rstate"],
        ),
    )
    mlp = NS(
        _mono_moe=NS(w_up=lambda: mw["w_up_shard"]),
        gate=NS(weight=mw["w_gate"], e_score_correction_bias=mw["bias"]),
        routed_expert_down_proj=norm(mw["w_ld"]),
        routed_expert_norm=norm(mw["ln_w"]),
        shared_experts=NS(gate_up_proj=norm(mw["w_sgu"]), down_proj=norm(mw["w_sd"])),
        experts=NS(
            w13_weight=mw["w13"],
            w13_weight_scale=mw["w13s"],
            w2_weight=mw["w2"],
            w2_weight_scale=mw["w2s"],
        ),
    )
    layer = NS(
        self_attn=attn,
        mlp=mlp,
        layer_idx=0,
        prev_valid_blocks=4,
        is_block_write_layer=reset,
        block_write_idx=4,
        input_layernorm=norm(w["in_nw"]),
        self_attention_res_norm=norm(w["ares_nw"]),
        self_attention_res_proj=norm(w["ares_qk"]),
        mlp_res_norm=norm(w["ares_nw"]),
        mlp_res_proj=norm(w["ares_qk"]),
        post_attention_layernorm=norm(w["in_nw"]),
    )
    with set_current_vllm_config(config):
        mono = mono_decode.MonoKda(layer)
        assert mono.ok
        layer._mono_k2 = mono_decode.MonoK2(layer)
        runner.peers(device)
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=rows // length,
        num_spec_decode_tokens=rows,
        num_actual_tokens=rows,
        spec_state_indices_tensor=w["st_idx"],
        spec_sequence_masks=torch.ones(rows // length, dtype=torch.bool, device=device),
        num_accepted_tokens=w["num_acc"],
        uniform_spec_sequence_length=length,
    )
    core = torch.empty(rows, k1.PROJ, dtype=bf, device=device)
    out = torch.empty_like(w["prefix"])
    prefix = torch.empty_like(out) if reset else w["prefix"]
    tensors = dict(
        out=out,
        core=core,
        next_prefix=prefix,
        **{name: w[name] for name in ("prefix", "blocks", "conv_state", "rstate")},
    )
    initial = {
        name: w[name].clone() for name in ("prefix", "blocks", "conv_state", "rstate")
    }
    packed = pack_projection_weights(w["w_in"])
    first_key = k1.KdaPreBuild(
        tokens=rows,
        qlen=length,
        nblocks=4,
        delta=True,
        write_idx=4 if reset else -1,
        state_len=conv.size(-1),
        state_rows=16,
        pipeline=35667968 if length == 6 else 33562624,
    )
    first = {name: value for name, value in w.items() if name != "w_in"}
    first.update(w_in=packed, core_out=core)
    second = layer._mono_k2.arguments(core, prefix, w["blocks"], out)

    def prepare():
        for name, value in initial.items():
            w[name].copy_(value)
        runner.step_begin()

    def control():
        prepare()
        k1.kda_pre(
            first_key, **first, scratch=runner.scratch(), epoch=runner.epoch(), layer=1
        )
        k2.k2_launch(
            k2.K2Build(tokens=rows, nblocks=4 + int(reset), reset=reset),
            **second,
            **runner.launch_args(),
            layer=2,
        )
        torch.accelerator.synchronize()
        return {name: value.clone() for name, value in tensors.items()}

    def call():
        prepare()
        mono_decode._mono_kda_fused(
            mono, w["prefix"], w["blocks"], w["delta"], None, core, prefix, out
        )
        return out + 1

    def check(expected, downstream):
        torch.accelerator.synchronize()
        for name, value in tensors.items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        torch.testing.assert_close(downstream, expected["out"] + 1, rtol=0, atol=0)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with (
        torch.cuda.stream(stream),
        set_forward_context(
            {attn.prefix: metadata},
            config,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        ),
    ):
        expected = control()
        check(expected, call())
        dist.barrier()
        capture = BreakableCUDAGraphCapture()
        with capture:
            downstream = call()
        assert capture.num_eager_breaks == 1
        cache_size = len(mono._overlap_arguments)
        for step in range(2):
            initial["prefix"].add_(0.03125)
            w["st_idx"].copy_(w["st_idx"].roll(1, dims=1))
            w["num_acc"].fill_(step + 1)
            expected = control()
            capture.replay()
            check(expected, downstream)
        assert len(mono._overlap_arguments) == cache_size
    torch.cuda.current_stream().wait_stream(stream)


@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize("reset", [False, True])
def test_k1_k2_serving_replays_state_and_prefix(monkeypatch, reset):
    @ray.remote(num_gpus=1, max_calls=1)
    def worker(monkeypatch, tp_size, pp_size, rank, port):
        monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
        from vllm.config import ParallelConfig, VllmConfig

        init_test_distributed_environment(tp_size, pp_size, rank, port)
        config = VllmConfig(
            parallel_config=ParallelConfig(tensor_parallel_size=tp_size)
        )
        _check_k1_k2_serving_replay(16, 4, config, reset)

    multi_process_parallel(monkeypatch, 8, 1, worker)


def _dcp_mla_core(S, generator, backend):
    """Exercise weighted context merging and the BF16 V-up/gate handoff to K2."""
    from vllm.distributed import get_dcp_group
    from vllm.v1.attention.ops.dcp import cp_lse_ag_out_rs, dcp_a2a_lse_reduce

    group = get_dcp_group()
    size, rank = group.world_size, group.rank_in_group
    heads, latent, width = 12, 512, 128
    partial = _randn(generator, S, heads * size, latent)
    lse = _randn(generator, S, heads * size, scale=2.0, dtype=torch.float32)
    lse.add_(rank * 1.5)
    lengths = torch.ones(S, dtype=torch.int32, device=generator.device)
    if rank == 0:
        lengths[0] = 0
        lse[0].fill_(float("nan"))
    lengths[1] = 0
    lse[1].fill_(float("nan"))
    starts = torch.arange(S + 1, dtype=torch.int32, device=generator.device)
    w_v = _randn(generator, heads, latent, width, scale=0.05)
    gate = _randn(generator, S, heads * width).sigmoid()
    combine = cp_lse_ag_out_rs if backend == "ag_rs" else dcp_a2a_lse_reduce

    def forward():
        # AG+RS normalizes its input in place, so reset it on every replay.
        merged = combine(
            partial.clone(),
            lse.clone(),
            group,
            seq_lens=lengths,
            query_start_loc=starts,
        )
        projected = torch.bmm(merged.transpose(0, 1), w_v).transpose(0, 1)
        return merged, projected.reshape(S, heads * width) * gate

    def check(merged, core):
        outputs = [torch.empty_like(partial) for _ in range(size)]
        lses = [torch.empty_like(lse) for _ in range(size)]
        masked_lse = lse.masked_fill(lengths[:, None] == 0, -torch.inf)
        dist.all_gather(outputs, partial, group=group.device_group)
        dist.all_gather(lses, masked_lse, group=group.device_group)
        logs = torch.stack(lses)
        maximum = logs.amax(dim=0)
        maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
        weights = (logs - maximum).exp()
        weights /= weights.sum(dim=0).clamp_min(1e-30)
        reference = (torch.stack(outputs).float() * weights[..., None]).sum(dim=0)
        reference = reference[:, rank * heads : (rank + 1) * heads]
        assert torch.isfinite(merged).all()
        assert torch.count_nonzero(merged[1]) == 0
        assert _rel_l2(merged, reference) < 0.01
        projected = torch.bmm(reference.to(bf).transpose(0, 1), w_v).transpose(0, 1)
        expected = projected.reshape(S, heads * width) * gate
        assert _cos(core, expected) > COS_MIN

    for _ in range(2):
        merged, core = forward()
    check(merged, core)
    torch.accelerator.synchronize()
    dist.barrier()
    graph = None
    if backend == "ag_rs":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            merged, core = forward()
    # ProcessGroupNCCL A2A work.wait() queries a captured HIP event on this
    # backend. Keep its eager handoff covered without claiming graph support.
    for change in (0.0, 0.25, -0.5):
        partial.add_(change * (rank + 1))
        lse.add_(change * (rank + 1))
        if graph is None:
            merged, core = forward()
        else:
            graph.replay()
        torch.accelerator.synchronize()
        check(merged, core)
    return core.clone()


def _k2_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
    S: int = 8,
    shared_down_workers: int = 0,
    latent_quant_workers: int = 0,
    dcp_size: int = 1,
    dcp_backend: str = "ag_rs",
    producer_latent_quant: bool = False,
    down_workers: int = 0,
    bounded_prefetch: bool = False,
    stream_oproj: bool = False,
    balanced_oproj: bool = False,
    sparse_routes: bool = False,
    nblocks: int = 4,
    front_panels: bool = False,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    if dcp_size == 1:
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    else:
        from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
        from vllm.distributed import (
            ensure_model_parallel_initialized,
            init_distributed_environment,
        )

        config = VllmConfig(
            parallel_config=ParallelConfig(
                tensor_parallel_size=tp_size,
                decode_context_parallel_size=dcp_size,
            )
        )
        with set_current_vllm_config(config):
            init_distributed_environment(
                world_size=tp_size,
                rank=rank,
                local_rank=rank,
                distributed_init_method=f"tcp://localhost:{distributed_init_port}",
                backend="nccl",
            )
            ensure_model_parallel_initialized(
                tp_size, pp_size, decode_context_model_parallel_size=dcp_size
            )

    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.amd.mono import layer as k2
    from vllm.models.kimi_k3.amd.mono.attention.kda import HIDDEN, PROJ
    from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
    from vllm.models.kimi_k3.amd.mono.stages import moe
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

    nb = nblocks
    tp = get_tp_group()
    peers = PeerBuffer(
        max(k2.peer_bytes(S), moe.peer_bytes(S)), tp.cpu_group, rank, tp_size, device
    )
    peers.bytes.zero_()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    mw = _moe_weights(rank, device)
    gc = torch.Generator(device=device).manual_seed(11)
    gr = torch.Generator(device=device).manual_seed(200 + rank)
    core = _randn(gr, S, PROJ)
    if dcp_size > 1:
        core = _dcp_mla_core(S, gr, dcp_backend)
    w_o = _randn(gr, HIDDEN, PROJ, scale=0.02)
    prefix0 = _randn(gc, S, HIDDEN, scale=2.0)
    blocks = _randn(gc, S, 8, HIDDEN, scale=2.0)
    ares_nw = (1 + _randn(gc, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf)
    ares_qk = _randn(gc, HIDDEN, scale=0.05)
    in_nw = (1 + _randn(gc, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf)

    for reset in (False, True):
        epoch.add_(1)
        key = k2.K2Build(
            tokens=S,
            nblocks=nb + int(reset),
            reset=reset,
            shared_down_workers=shared_down_workers,
            parallel_shared_activation=shared_down_workers > 0,
            late_shared_wait=shared_down_workers > 0,
            latent_quant_workers=latent_quant_workers,
            producer_latent_quant=producer_latent_quant,
            down_workers=down_workers,
            full_shared_pipeline=down_workers > 0 and shared_down_workers > 0,
            down_batch=2 if bounded_prefetch else 5,
            inter_vec=2 if bounded_prefetch else 1,
            stream_oproj=stream_oproj,
            trim_oproj=stream_oproj,
            balanced_oproj=balanced_oproj,
            wide=sparse_routes,
            compact_lds=sparse_routes,
            sparse_routes=sparse_routes,
            cache_wide_inter=sparse_routes and S <= 32,
            stream_attn=sparse_routes and (S > 32 or nb + int(reset) >= 8),
            stream_x=front_panels,
            front_panels=front_panels,
        )
        scratch = torch.zeros(k2.scratch_bytes(key), dtype=torch.uint8, device=device)
        prefix = prefix0.clone()
        out = torch.empty(S, HIDDEN, dtype=bf, device=device)
        k2.k2_launch(
            key,
            core=core,
            w_o=w_o,
            prefix=prefix,
            blocks=blocks,
            ares_nw=ares_nw,
            ares_qk=ares_qk,
            in_nw=in_nw,
            out=out,
            scratch=scratch,
            peers=peers.addresses,
            rank=rank,
            epoch=epoch,
            layer=3,
            **_moe_args(mw),
        )
        torch.accelerator.synchronize()

        attn = _all_reduce_fp32(F.linear(core, w_o))
        # a block-write layer's prefix restarts at the attention output
        ref_prefix = attn if reset else prefix0.clone()
        delta = None if reset else attn
        x_ref = attn_res(
            ref_prefix,
            delta,
            blocks,
            ares_nw,
            ares_qk,
            in_nw,
            nb + int(reset),
            -1,
            1e-5,
            1e-5,
        )
        # the kernel sums the ranks in rank order, NCCL in its own order
        assert _rel_l2(prefix, ref_prefix) < 1e-2
        offset = k2.scratch_layout(key)["xrow"][0]
        x = scratch[offset : offset + S * HIDDEN * 2].view(bf).view(S, HIDDEN)
        assert _cos(x, x_ref) > COS_MIN
        assert _cos(out, _moe_reference(x, mw, rank)) > COS_MIN
        assert _cos(out, _moe_reference(x_ref, mw, rank)) > COS_MIN
        outs = [torch.empty_like(out) for _ in range(tp_size)]
        dist.all_gather(outs, out)
        assert all(torch.equal(outs[0], o) for o in outs), "ranks disagree"

        if front_panels:

            def launch(key=key, prefix=prefix, out=out, scratch=scratch):
                prefix.copy_(prefix0)
                epoch.add_(1)
                k2.k2_launch(
                    key,
                    core=core,
                    w_o=w_o,
                    prefix=prefix,
                    blocks=blocks,
                    ares_nw=ares_nw,
                    ares_qk=ares_qk,
                    in_nw=in_nw,
                    out=out,
                    scratch=scratch,
                    peers=peers.addresses,
                    rank=rank,
                    epoch=epoch,
                    layer=3,
                    **_moe_args(mw),
                )

            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(graph):
                launch()
            for change in (0.125, -0.25):
                core.add_(change * (rank + 1))
                prefix0.add_(change)
                launch()
                expected_out, expected_prefix = out.clone(), prefix.clone()
                graph.replay()
                torch.accelerator.synchronize()
                torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
                torch.testing.assert_close(prefix, expected_prefix, rtol=0, atol=0)
            del graph

    if S > moe.MOE_TILE_TOKENS:
        peers.close()
        return

    # the standalone MoE launch (a layer K2 does not serve)
    epoch.add_(1)
    x = _randn(gc, S, HIDDEN)
    out = torch.empty_like(x)
    queue = torch.zeros(moe.QUEUE_BYTES // 4, dtype=torch.int32, device=device)
    moe.moe(
        moe.MoeBuild(tokens=S),
        x=x,
        out=out,
        scratch=torch.zeros(
            moe.scratch_bytes(moe.MoeBuild(tokens=S)), dtype=torch.uint8, device=device
        ),
        queue=queue,
        peers=peers.addresses,
        rank=rank,
        epoch=epoch,
        layer=128 + 3,
        **_moe_args(mw),
    )
    torch.accelerator.synchronize()
    assert _cos(out, _moe_reference(x, mw, rank)) > COS_MIN
    peers.close()


@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize(
    "S,shared_down_workers,latent_quant_workers,producer_latent_quant,"
    "down_workers,bounded_prefetch,stream_oproj,balanced_oproj",
    [
        (8, 0, 0, False, 0, False, False, False),
        (16, 0, 0, False, 0, False, False, False),
        (12, 32, 0, False, 0, False, False, False),
        (16, 16, 0, False, 0, False, False, False),
        (6, 0, 2, False, 0, False, False, False),
        (16, 0, 4, False, 0, False, False, False),
        (6, 0, 0, True, 0, False, False, False),
        (16, 0, 0, True, 0, False, False, False),
        (6, 16, 0, True, 32, False, False, False),
        (16, 16, 0, True, 64, False, False, False),
        pytest.param(6, 0, 0, True, 0, True, False, False, id="bounded-prefetch-m6"),
        pytest.param(8, 0, 0, True, 0, True, False, False, id="bounded-prefetch-m8"),
        pytest.param(12, 0, 0, True, 0, True, False, False, id="bounded-prefetch-m12"),
        pytest.param(16, 0, 0, True, 0, True, False, False, id="bounded-prefetch-m16"),
        pytest.param(6, 0, 0, True, 0, True, True, False, id="streamed-projection-m6"),
        pytest.param(8, 0, 0, True, 0, True, True, False, id="streamed-projection-m8"),
        pytest.param(
            12, 0, 0, True, 0, True, True, False, id="streamed-projection-m12"
        ),
        pytest.param(
            16, 0, 0, True, 0, True, True, False, id="streamed-projection-m16"
        ),
        pytest.param(6, 0, 0, True, 0, True, True, True, id="balanced-projection-m6"),
        pytest.param(8, 0, 0, True, 0, True, True, True, id="balanced-projection-m8"),
        pytest.param(12, 0, 0, True, 0, True, True, True, id="balanced-projection-m12"),
        pytest.param(16, 0, 0, True, 0, True, True, True, id="balanced-projection-m16"),
    ],
)
def test_k2_and_moe_tp8_match_vllm_ops(
    monkeypatch: pytest.MonkeyPatch,
    S: int,
    shared_down_workers: int,
    latent_quant_workers: int,
    producer_latent_quant: bool,
    down_workers: int,
    bounded_prefetch: bool,
    stream_oproj: bool,
    balanced_oproj: bool,
) -> None:
    @ray.remote(num_gpus=1, max_calls=1)
    def worker(monkeypatch, tp_size, pp_size, rank, port):
        _k2_worker(
            monkeypatch,
            tp_size,
            pp_size,
            rank,
            port,
            S,
            shared_down_workers,
            latent_quant_workers,
            producer_latent_quant=producer_latent_quant,
            down_workers=down_workers,
            bounded_prefetch=bounded_prefetch,
            stream_oproj=stream_oproj,
            balanced_oproj=balanced_oproj,
        )

    multi_process_parallel(monkeypatch, 8, 1, worker)


@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize(
    "S,nblocks,front_panels",
    [
        (24, 4, False),
        (32, 7, False),
        (48, 7, False),
        pytest.param(18, 7, True, id="front-m18"),
        pytest.param(32, 7, True, id="front-m32"),
        pytest.param(42, 7, True, id="front-m42"),
        pytest.param(48, 7, True, id="front-m48"),
    ],
)
def test_k2_sparse_routes_preserve_prefix_and_output(
    monkeypatch, S, nblocks, front_panels
):
    """Packed expert picks preserve block-reset semantics through eight blocks."""

    @ray.remote(num_gpus=1, max_calls=1)
    def worker(monkeypatch, tp_size, pp_size, rank, port):
        _k2_worker(
            monkeypatch,
            tp_size,
            pp_size,
            rank,
            port,
            S,
            producer_latent_quant=True,
            bounded_prefetch=True,
            balanced_oproj=not front_panels,
            sparse_routes=True,
            nblocks=nblocks,
            front_panels=front_panels,
        )

    multi_process_parallel(monkeypatch, 8, 1, worker)


@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize(
    "S,dcp_size,backend,producer_quant,down_workers,bounded_prefetch,"
    "stream_oproj,balanced_oproj",
    [
        (8, 8, "ag_rs", False, 0, False, False, False),
        (16, 2, "ag_rs", False, 0, False, False, False),
        (16, 2, "a2a", False, 0, False, False, False),
        (8, 8, "ag_rs", True, 0, False, False, False),
        (8, 8, "ag_rs", True, 32, False, False, False),
        pytest.param(
            8, 8, "ag_rs", True, 0, True, False, False, id="bounded-prefetch-dcp8"
        ),
        pytest.param(
            16, 2, "a2a", True, 0, True, False, False, id="bounded-prefetch-dcp2-a2a"
        ),
        pytest.param(
            8, 8, "ag_rs", True, 0, True, True, False, id="streamed-projection-dcp8"
        ),
        pytest.param(
            16, 2, "a2a", True, 0, True, True, False, id="streamed-projection-dcp2-a2a"
        ),
        pytest.param(
            8, 8, "ag_rs", True, 0, True, True, True, id="balanced-projection-dcp8"
        ),
        pytest.param(
            16, 2, "a2a", True, 0, True, True, True, id="balanced-projection-dcp2-a2a"
        ),
    ],
)
def test_k2_consumes_completed_dcp_mla(
    monkeypatch: pytest.MonkeyPatch,
    S: int,
    dcp_size: int,
    backend: str,
    producer_quant: bool,
    down_workers: int,
    bounded_prefetch: bool,
    stream_oproj: bool,
    balanced_oproj: bool,
) -> None:
    """DCP head scattering preserves K2's independent TP projection/MoE reductions."""

    @ray.remote(num_gpus=1, max_calls=1)
    def worker(monkeypatch, tp_size, pp_size, rank, port):
        _k2_worker(
            monkeypatch,
            tp_size,
            pp_size,
            rank,
            port,
            S,
            latent_quant_workers=0 if producer_quant else 4,
            dcp_size=dcp_size,
            dcp_backend=backend,
            producer_latent_quant=producer_quant,
            down_workers=down_workers,
            shared_down_workers=16 if down_workers else 0,
            bounded_prefetch=bounded_prefetch,
            stream_oproj=stream_oproj,
            balanced_oproj=balanced_oproj,
        )

    multi_process_parallel(monkeypatch, 8, 1, worker)
