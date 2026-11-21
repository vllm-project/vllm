# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-level check of the amd_kernels kda route (single GPU): akr_kda.run vs the stock
chain (kimi_k3_kda_fb_proj -> AITER kda_fused_recurrent_spec_decode) on real K3 TP8
shapes, random weights."""

import types

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("amd_kernels")

NH, D, LP = 12, 128, 1536
T_LIST = [(2, 4), (4, 4), (10, 4), (13, 5), (20, 4), (6, 6)]


def _layer(dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s, sc=1.0: torch.randn(*s, generator=g, device=dev) * sc  # noqa: E731
    layer = types.SimpleNamespace(
        local_num_heads=NH,
        head_dim=D,
        conv_size=4,
        use_full_rank_gate=True,
        gate_lower_bound=-5.0,
        decode_conv1d_weight=r(3, 4, LP, sc=0.5).contiguous(),
        A_log=r(NH, sc=0.5).float(),
        dt_bias=r(LP, sc=0.5).float(),
        f_b_proj=types.SimpleNamespace(weight=r(LP, D, sc=0.1).bfloat16()),
        o_norm=types.SimpleNamespace(weight=(1 + 0.1 * r(D)).bfloat16(), eps=1e-6),
    )
    return layer, g


@pytest.mark.parametrize("n_seq,t", T_LIST)
def test_matches_stock(n_seq, t):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.kimi_k3.amd import akr_kda
    from vllm.models.kimi_k3.amd.ops import (
        kda_fb_gemm,  # noqa: F401  (registers the op)
    )

    dev = torch.device("cuda")
    layer, g = _layer(dev)
    rows = n_seq * t
    ncols = 6288 + 16 - (6288 % 16)
    proj = (torch.randn(rows + 3, ncols, generator=g, device=dev) * 0.5).bfloat16()
    slots = 1 + n_seq * (t + 1)
    state = torch.randn(slots, NH, D, D, generator=g, device=dev) * 0.1
    conv_raw = (
        torch.randn(slots, 3 + t, 3 * LP, generator=g, device=dev) * 0.5
    ).bfloat16()
    idx = (1 + torch.arange(n_seq * t, device=dev, dtype=torch.int32)).view(n_seq, t)
    acc = torch.randint(1, t + 1, (n_seq,), device=dev, dtype=torch.int32, generator=g)
    cu = torch.arange(0, (n_seq + 1) * t, t, device=dev, dtype=torch.int32)
    m = types.SimpleNamespace(
        spec_state_indices_tensor=idx,
        spec_query_start_loc=cu,
        num_spec_decodes=n_seq,
        num_actual_tokens=rows,
        uniform_spec_sequence_length=t,
    )
    # stock
    st_a, cs_a = state.clone(), conv_raw.clone()
    f_a = proj[:rows, 4 * LP : 4 * LP + D]
    g1 = torch.ops.vllm.kimi_k3_kda_fb_proj(f_a, layer.f_b_proj.weight).view(
        1, rows, NH, D
    )
    beta = proj[:rows, 4 * LP + D : 4 * LP + D + NH].unsqueeze(0)
    q, k, v = (
        x.reshape(1, rows, NH, D) for x in proj[:rows, : 3 * LP].split(LP, dim=-1)
    )
    out_a = torch.empty(1, rows, NH, D, dtype=torch.bfloat16, device=dev)
    rocm_aiter_ops.kda_fused_recurrent_spec_decode(
        q=q,
        k=k,
        v=v,
        g=g1,
        beta=beta,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        lower_bound=-5.0,
        initial_state=st_a,
        cu_seqlens=cu,
        ssm_state_indices=idx,
        num_accepted_tokens=acc,
        conv_state=cs_a.transpose(-1, -2),
        conv_weight=layer.decode_conv1d_weight,
        out_gate=proj[:rows, 3 * LP : 4 * LP].reshape(1, rows, NH, D),
        norm_weight=layer.o_norm.weight,
        norm_eps=1e-6,
        out=out_a,
    )
    # akr
    st_b, cs_b = state.clone(), conv_raw.clone()
    out_b = torch.empty(1, rows, NH, D, dtype=torch.bfloat16, device=dev)
    assert akr_kda.run(layer, proj, cs_b.transpose(-1, -2), st_b, m, acc, out_b)
    torch.accelerator.synchronize()
    a, b = out_a.float().flatten(), out_b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    assert cos > 0.9999, cos
    assert (st_a - st_b).abs().max() <= 2e-3 * max(1.0, st_a.abs().max().item())
    assert torch.equal(cs_a, cs_b)


@pytest.mark.parametrize("t", [1, 4, 8])
def test_warm_up_runs(t):
    from vllm.models.kimi_k3.amd import akr_kda

    layer, _ = _layer(torch.device("cuda"))
    akr_kda.warm_up(layer, t)  # compiles both variants, launches on a padding sequence
