# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""amd_kernels K1 (VLLM_KIMI_AKR_K1) vs the stock specialized-decode `kda_pre` (single
GPU, real K3 TP8 shapes) through the `akr_k1k2.k1_launch` wiring: core output, prefix,
conv / recurrent state."""

import types

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("amd_kernels")

H, NPROJ, PROJ, QKV = 7168, 6288, 1536, 4608


@pytest.mark.parametrize("rows,qlen", [(12, 6), (8, 4), (16, 4), (6, 6)])
@pytest.mark.parametrize("delta", [True, False])
def test_k1_matches_stock(rows, qlen, delta):
    from vllm.models.kimi_k3.amd import akr_k1k2
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import kda
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import HEALTH_WORDS

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(3)
    r = lambda *s, sc=1.0, dt=torch.bfloat16: (
        torch.randn(*s, device=dev, generator=g) * sc
    ).to(dt)  # noqa: E731
    R = rows // qlen
    slots = 1 + R * (qlen + 1)
    nblocks, layer_idx = 4, 13
    attn = types.SimpleNamespace(
        in_proj_qkvgfab=types.SimpleNamespace(weight=r(NPROJ, H, sc=0.02)),
        f_b_proj=types.SimpleNamespace(weight=r(PROJ, 128, sc=0.1)),
        conv1d=types.SimpleNamespace(weight=r(QKV, 1, 4, sc=0.5, dt=torch.float32)),
        A_log=r(12, sc=0.5, dt=torch.float32),
        dt_bias=r(PROJ, sc=0.5, dt=torch.float32),
        o_norm=types.SimpleNamespace(weight=(1 + 0.1 * r(128)), eps=1e-6),
        gate_lower_bound=-5.0,
    )
    layer = types.SimpleNamespace(
        layer_idx=layer_idx,
        prev_valid_blocks=nblocks,
        is_block_write_layer=False,
        block_write_idx=-1,
        self_attention_res_norm=types.SimpleNamespace(
            weight=1 + 0.1 * r(H), variance_epsilon=1e-5
        ),
        self_attention_res_proj=types.SimpleNamespace(weight=r(1, H, sc=0.05)),
        input_layernorm=types.SimpleNamespace(
            weight=1 + 0.1 * r(H), variance_epsilon=1e-5
        ),
    )
    prefix0, delta_t = r(rows, H, sc=0.5), (r(rows, H, sc=0.3) if delta else None)
    blocks0 = r(rows, 8, H, sc=0.5)
    conv0 = r(slots, qlen + 3, QKV, sc=0.5)
    rs0 = r(slots, 12, 128, 128, sc=0.1, dt=torch.float32)
    idx = (1 + torch.arange(R * qlen, device=dev, dtype=torch.int32)).view(R, qlen)
    acc = torch.randint(1, qlen + 1, (R,), device=dev, dtype=torch.int32, generator=g)
    m = types.SimpleNamespace(spec_state_indices_tensor=idx, num_accepted_tokens=acc)

    # stock
    pa, ca, ra = prefix0.clone(), conv0.clone(), rs0.clone()
    key = kda.KdaPreBuild(
        tokens=rows,
        qlen=qlen,
        nblocks=nblocks,
        delta=delta,
        state_len=qlen + 3,
        eps=1e-5,
        out_eps=1e-5,
        onorm_eps=1e-6,
        lower_bound=-5.0,
        write_idx=-1,
    )
    core_a = torch.empty(rows, PROJ, dtype=torch.bfloat16, device=dev)
    scratch = torch.zeros(
        kda.scratch_bytes(
            kda.KdaPreBuild(
                tokens=rows, qlen=qlen, nblocks=8, delta=True, state_len=qlen + 3
            )
        ),
        dtype=torch.uint8,
        device=dev,
    )
    epoch = torch.ones(1, dtype=torch.int32, device=dev)
    health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device=dev)
    sink = torch.zeros(64, dtype=torch.int32, device=dev)
    kda.kda_pre(
        key,
        prefix=pa,
        delta=delta_t,
        blocks=blocks0.clone(),
        ares_nw=layer.self_attention_res_norm.weight,
        ares_qk=layer.self_attention_res_proj.weight.view(H),
        in_nw=layer.input_layernorm.weight,
        w_in=attn.in_proj_qkvgfab.weight,
        w_fb=attn.f_b_proj.weight,
        conv_w=attn.conv1d.weight.view(QKV, 4),
        conv_state=ca.transpose(-1, -2),
        a_log=attn.A_log,
        dt_bias=attn.dt_bias,
        on_w=attn.o_norm.weight,
        rstate=ra,
        st_idx=idx,
        num_acc=acc,
        core_out=core_a,
        scratch=scratch,
        layer=layer_idx,
        epoch=epoch,
        health=health,
        alarm=sink.data_ptr(),
    )
    torch.accelerator.synchronize()
    # amd_kernels K1
    pb, cb, rb = prefix0.clone(), conv0.clone(), rs0.clone()
    runner = types.SimpleNamespace(akr_k1=akr_k1k2.k1_workspace(dev))
    runner.akr_k1.begin_step()
    whole = types.SimpleNamespace(layer=layer, runner=runner)
    core_b = akr_k1k2.k1_launch(
        whole, None, pb, blocks0.clone(), delta_t, attn, m, cb.transpose(-1, -2), rb
    )
    assert core_b is not None
    torch.accelerator.synchronize()
    cos = torch.nn.functional.cosine_similarity(
        core_a.float().flatten(), core_b.float().flatten(), dim=0
    ).item()
    assert cos > 0.9999, cos
    assert torch.equal(pa, pb)
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()  # noqa: E731
    print(f"core cos {cos:.6f} rstate rel {rel(rb, ra):.2e} conv rel {rel(cb, ca):.2e}")
    assert rel(rb, ra) < 2e-3 and rel(cb, ca) < 4e-3
