# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Layer-level check of the amd_kernels KDA prefill hook:
`KimiK3DeltaAttention._forward` with VLLM_KIMI_AKR_KDA_PREFILL=0 (stock conv +
chunk_kda_prefill + o_norm) vs =1, on realistic chunk shapes (ragged batches, cached
context -> initial states, decode-first mixed steps, checkpoint export). Single GPU.
usage: python check_kda_prefill.py"""

import os
from types import SimpleNamespace

import torch

import vllm.models.kimi_k3.amd.kda as kda
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.kimi_k3.amd import akr_kda_prefill as ak
from vllm.models.kimi_k3.amd import kda_metadata as km
from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

dev = "cuda"
H, D, LP = 12, 128, 1536
PAD = 6288 - 6284  # row padding of the in_proj output (stride % 8 == 0)


def make_layer(seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    conv_w = (0.3 * torch.randn(3 * LP, 1, 4, device=dev, generator=g)).float()
    o_norm = FusedRMSNormGated(D, activation="sigmoid").to(dev)
    with torch.no_grad():
        o_norm.weight.copy_(torch.randn(D, device=dev, generator=g) * 0.3 + 1)
    layer = SimpleNamespace(
        prefix="test.kda",
        use_aiter_decode=False,
        aiter_supports_strided_state=False,
        decode_conv1d_weight=None,
        decode_norm_weight=None,
        conv1d=SimpleNamespace(weight=conv_w, bias=None),
        local_projection_size=LP,
        head_dim=D,
        local_num_heads=H,
        conv_size=4,
        A_log=torch.log(torch.arange(1, H + 1, dtype=torch.float32, device=dev)),
        dt_bias=torch.empty(LP, device=dev).uniform_(-7.0, -4.0, generator=g),
        gate_lower_bound=-5.0,
        o_norm=o_norm,
        kda_prefill_backend="fused",
        _checkpoint_exporter=SimpleNamespace(export=lambda *a, **k: None),
    )
    layer._fused_qkv_conv_fn = None
    try:
        from aiter.ops.triton.gated_delta_net import (
            causal_conv1d_split_qkv_triton_tile_fn,
        )

        layer._fused_qkv_conv_fn = causal_conv1d_split_qkv_triton_tile_fn
    except Exception as e:  # noqa
        print("no fused conv fn", e)
    return layer


def make_batch(nd, plens, ctxs, seed=1, slots=200):
    """Nd decode rows (1 token, cached state) then prefill sequences (plens new tokens,
    ctxs cached)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    lens = [1] * nd + list(plens)
    T = sum(lens)
    row = 6288
    proj = torch.zeros(T, row, device=dev, dtype=torch.bfloat16)
    proj[:, :4608] = (0.7 * torch.randn(T, 4608, device=dev, generator=g)).bfloat16()
    # g2 (gate)
    proj[:, 4608:6144] = torch.randn(T, 1536, device=dev, generator=g).bfloat16()
    proj[:, 6144:6272] = torch.randn(T, 128, device=dev, generator=g).bfloat16()  # f_a
    proj[:, 6272:6284] = torch.randn(T, 12, device=dev, generator=g).bfloat16()  # beta
    mixed = proj[:, :4608]
    g2 = proj[:, 4608:6144].view(T, H, D)
    beta = proj[:, 6272:6284].unsqueeze(0)
    g1 = torch.randn(1, T, H, D, device=dev, generator=g).bfloat16()
    n = len(lens)
    cu = torch.tensor(
        [0] + torch.tensor(lens).cumsum(0).tolist(), device=dev, dtype=torch.int32
    )
    cu_cpu = cu.cpu()
    ctx_all = [1] * nd + list(ctxs)
    has = torch.tensor([c > 0 for c in ctx_all], device=dev)
    # served layout: the column view block_table[:, 0] (1 + speculative blocks per row)
    bt = torch.arange(1, 4 * n + 1, device=dev, dtype=torch.int32).view(n, 4)
    sidx = bt[:, 0]
    rec = (0.1 * torch.randn(slots, H, D, D, device=dev, generator=g)).float()
    # DS layout: [slots, dim, 3]; SD layout: [slots, 3, dim] (transposed to dim-first by
    # the layer)
    if is_conv_state_dim_first():
        conv = (0.7 * torch.randn(slots, 4608, 3, device=dev, generator=g)).bfloat16()
    else:
        conv = (0.7 * torch.randn(slots, 3, 4608, device=dev, generator=g)).bfloat16()
    np_ = len(plens)
    pcu, pcu_cpu = cu[nd:] - nd, cu_cpu[nd:] - nd
    ci, co = km.prepare_chunk_metadata_device(pcu, pcu_cpu, 64)
    m = GDNAttentionMetadata(
        num_prefills=np_,
        num_prefill_tokens=sum(plens),
        num_decodes=nd,
        num_decode_tokens=nd,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=T,
        has_initial_state=has,
        non_spec_query_start_loc=cu,
        non_spec_state_indices_tensor=sidx,
        chunk_indices=ci,
        chunk_offsets=co,
        prefill_query_start_loc=pcu,
        prefill_state_indices=sidx[nd:],
        prefill_has_initial_state=has[nd:],
        nums_dict={
            km.PREFILL_CONV_BLOCK_M: km.build_conv1d_chunk_schedule(
                cu_cpu, km.PREFILL_CONV_BLOCK_M, torch.device(dev)
            )
        },
    )
    counts = ak.chunk_counts_of(pcu_cpu[1:] - pcu_cpu[:-1])
    m.akr_chunk_counts = counts  # type: ignore[attr-defined]
    return dict(
        mixed=mixed, g1=g1, g2=g2, beta=beta, m=m, rec=rec, conv=conv, T=T, nd=nd
    )


def run(layer, b, akr, patch):
    os.environ["VLLM_KIMI_AKR_KDA_PREFILL"] = "1" if akr else "0"
    rec, conv = b["rec"].clone(), b["conv"].clone()
    layer.kv_cache = (conv, rec)
    out = torch.full((1, b["T"], H, D), float("nan"), device=dev, dtype=torch.bfloat16)
    patch(lambda: SimpleNamespace(attn_metadata={layer.prefix: b["m"]}))
    for attr in (ak._STEP, ak._SIDX):
        if hasattr(b["m"], attr):
            delattr(b["m"], attr)
    called = []
    orig = ak.try_prefill

    def spy(*a, **k):
        r = orig(*a, **k)
        called.append(r)
        return r

    ak.try_prefill = spy
    try:
        KimiK3DeltaAttention._forward(
            layer, b["mixed"], b["g1"], b["g2"], b["beta"], out
        )
    finally:
        ak.try_prefill = orig
    torch.accelerator.synchronize()
    return out, rec, conv, called


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-9)).item()


def timeit(fn, n=5):
    fn()
    torch.accelerator.synchronize()
    t = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.accelerator.synchronize()
        t.append(a.elapsed_time(b) * 1e3)
    return min(t)


def main():
    kda.get_forward_context = None

    def patch(f):
        kda.get_forward_context = f

    layer = make_layer()
    cases = [
        ("1x2048 fresh", 0, [2048], [0]),
        ("1x4096 ctx100K", 0, [4096], [100_000]),
        ("1x16384 ctx500K", 0, [16384], [500_000]),
        (
            "ragged 3 (3000,1500,700)+60",
            0,
            [3000, 1500, 700, 60],
            [0, 120_000, 40, 300_000],
        ),
        ("ragged 10240 = 6000+4240", 0, [6000, 4240], [200_000, 0]),
        ("mixed 12 decodes + 2x2000", 12, [2000, 2000], [50_000, 0]),
        ("mixed 40 decodes + 5000", 40, [5000], [10_000]),
        ("tiny 1+1+1", 0, [1, 1, 3], [10, 0, 5]),
    ]
    for name, nd, plens, ctxs in cases:
        b = make_batch(nd, plens, ctxs)
        o0, r0, c0, called0 = run(layer, b, False, patch)
        o1, r1, c1, called1 = run(layer, b, True, patch)
        nrow = b["T"]
        ok_used = called1 == [True]
        slots = b["m"].non_spec_state_indices_tensor.long()
        e_out = rel(o1[0], o0[0])
        e_pre = rel(o1[0, nd:], o0[0, nd:])
        e_dec = rel(o1[0, :nd], o0[0, :nd]) if nd else 0.0
        e_st = max(rel(r1[s], r0[s]) for s in slots.tolist())
        others = torch.ones(r0.shape[0], dtype=torch.bool, device=dev)
        others[slots] = False
        others_ok = torch.equal(r1[others], b["rec"][others])
        e_cv = (c1.float() - c0.float()).abs().max().item()
        untouched = bool((r1[0] == b["rec"][0]).all())
        print(
            f"{name:34s} T={nrow:6d} used={ok_used} out_relL2={e_out:.2e} (prefill "
            f"{e_pre:.2e}, decode {e_dec:.2e}) "
            f"state_relL2(max per seq)={e_st:.2e} conv_state_maxabs={e_cv:.1e} "
            f"nan={bool(torch.isnan(o1).any())} slot0_untouched={untouched} "
            f"other_slots_untouched={others_ok}",
            flush=True,
        )
        if b["T"] >= 2048:
            t0 = timeit(lambda b=b: run(layer, b, False, patch))
            t1 = timeit(lambda b=b: run(layer, b, True, patch))
            print(
                f"    eager call (incl. python, state clones) stock {t0:.0f} us  akr "
                f"{t1:.0f} us",
                flush=True,
            )
        b = None  # free the batch before the next case


if __name__ == "__main__":
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        main()
