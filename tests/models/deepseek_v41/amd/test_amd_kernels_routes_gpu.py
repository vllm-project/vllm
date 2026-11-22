# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 amd_kernels routes against the stock vLLM path they replace,
on one gfx950 GPU: the router, the shared expert and the mHC seam, at row
counts inside and outside each route's range, eager and under graph replay.

These check the glue (argument order, layouts, fallbacks) through the
`amd_kernels` op entries; the kernels' own numerics are amd-kernels' tests."""

import types

import pytest
import torch

from vllm.platforms import current_platform

H = 5120
HC = 4


def _entries_available() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return False
    try:
        import amd_kernels.moe as moe
        import amd_kernels.norm as norm
    except ImportError:
        return False
    return hasattr(moe, "router_topk_shapes") and hasattr(norm, "mhc_pack_gate_weight")


pytestmark = [
    pytest.mark.skipif(
        not _entries_available(), reason="needs gfx950 and the amd_kernels op entries"
    ),
    # the module fixture owns the distributed state for all its tests
    pytest.mark.skip_global_cleanup,
]


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


@pytest.fixture(scope="module")
def vllm_config():
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.models.deepseek_v41.amd import akr_config
    from vllm.utils.network_utils import get_open_port

    model_config = ModelConfig(
        "deepseek-ai/DeepSeek-V4.1-Flash",
        language_model_only=True,
        max_model_len=4096,
    )
    cfg = VllmConfig(model_config=model_config)
    with set_current_vllm_config(cfg):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        )
        initialize_model_parallel(1, 1)
        akr_config.reset()
        assert akr_config.routes() == frozenset(akr_config.ROUTES)
        yield cfg
        akr_config.reset()
        cleanup_dist_env_and_memory()


def _replayed(fn, x: torch.Tensor):
    """fn(x) captured once in a CUDA graph, then replayed on new data."""
    fn(x)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn(x)
    torch.accelerator.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        out = fn(x)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.accelerator.synchronize()
    return out


def _load_random(module: torch.nn.Module, seed: int) -> None:
    """Random MXFP8 codes and E8M0 scales, then the stock post-processing."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for p in module.parameters():
        if p.dtype == torch.float8_e4m3fn:
            values = torch.randn(p.shape, device="cuda", generator=gen) * 0.5
            p.data.copy_(values.to(p.dtype))
        elif p.element_size() == 1:
            # E8M0 scales of 32 x 32 blocks (2^-9 .. 2^-4), expanded per row
            # as the checkpoint loader does
            shape = (-(-p.shape[0] // 32), *p.shape[1:])
            blocks = torch.randint(118, 124, shape, device="cuda", generator=gen)
            rows = blocks.repeat_interleave(32, dim=0)[: p.shape[0]]
            p.data.copy_(rows.to(torch.uint8).view(p.dtype))
        else:
            p.data.copy_(torch.randn(p.shape, device="cuda", generator=gen) * 0.02)
    for sub in module.modules():
        method = getattr(sub, "quant_method", None)
        if method is not None:
            method.process_weights_after_loading(sub)


def _mxfp8_linear(vllm_config, n, k, prefix, seed, bmm_batch_size=None):
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.utils.torch_utils import set_default_torch_dtype

    with set_default_torch_dtype(torch.bfloat16):
        linear = ReplicatedLinear(
            k, n, bias=False, quant_config=vllm_config.quant_config, prefix=prefix
        ).cuda()
    if bmm_batch_size is not None:
        linear.bmm_batch_size = bmm_batch_size
    _load_random(linear, seed)
    return linear


# ---------------------------------------------------------------- dense GEMMs


@pytest.mark.parametrize("m", [16, 128, 384, 1024])
def test_dense_mxfp8_matches_stock(vllm_config, m):
    """wq_a | wkv on the entry where its blocker serves `m`, else None."""
    from vllm.models.deepseek_v41.amd import akr_kernels

    linear = _mxfp8_linear(vllm_config, 1792, 5120, "model.layers.0.attn.wq_a", 1)
    weight = akr_kernels.mxfp8_dense_weight(linear)
    assert weight is not None
    x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
    out = akr_kernels.mxfp8_dense(x, weight)
    if out is not None:
        ref, _ = linear(x)
        assert cos(out, ref) > 0.9999


def test_dense_mxfp8_codes_match_stock(vllm_config):
    """The indexer wq_b on `norm_dual`'s row-major MXFP8 codes."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.models.deepseek_v41.amd import akr_config, akr_kernels

    linear = _mxfp8_linear(
        vllm_config, 4096, 1280, "model.layers.2.attn.indexer.wq_b", 2
    )
    weight = akr_kernels.mxfp8_dense_weight(linear)
    assert weight is not None and akr_config.takes_mxfp8_codes(linear)
    x = torch.randn(32, 1280, device="cuda", dtype=torch.bfloat16)
    codes, scale = mxfp8_e4m3_quantize(x)
    qx = akr_config.quant_activation(codes, scale.view(torch.uint8), x.dtype)
    out = akr_kernels.mxfp8_dense(qx, weight)
    assert out is not None
    ref, _ = linear(qx)
    assert cos(out, ref) > 0.9999


def _codes_of(x: torch.Tensor):
    """`norm_dual`'s output form: MXFP8 rows whose scales are the first rows of a
    buffer padded to 32 rows."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.models.deepseek_v41.amd import akr_config

    codes, scale = mxfp8_e4m3_quantize(x)
    m = x.shape[0]
    padded = torch.empty(
        (m + 31) // 32 * 32, scale.shape[1], dtype=torch.uint8, device=x.device
    )
    padded[:m] = scale.view(torch.uint8)
    return akr_config.quant_activation(codes, padded[:m], x.dtype)


@pytest.mark.parametrize("m", [768, 769, 800, 848, 960, 1024])
def test_wq_b_codes_match_stock(vllm_config, m):
    """wq_b on `norm_dual`'s MXFP8 rows: the entry above 768 rows (the blocker
    declines below), the stock linear's result."""
    from vllm.models.deepseek_v41.amd import akr_config, akr_kernels

    linear = _mxfp8_linear(vllm_config, 32768, 1280, "model.layers.0.attn.wq_b", 5)
    weight = akr_kernels.mxfp8_dense_weight(linear)
    assert weight is not None and akr_config.takes_mxfp8_codes(linear)
    qx = _codes_of(torch.randn(m, 1280, device="cuda", dtype=torch.bfloat16))
    out = akr_kernels.mxfp8_dense(qx, weight)
    assert (out is not None) == akr_config.serves("wq_b", m)
    if out is not None:
        ref, _ = linear(qx)
        assert cos(out, ref) > 0.99999


@pytest.mark.parametrize("m", [769, 848, 1024])
def test_wo_b_on_bf16_matches_stock(vllm_config, m):
    """wo_b on wo_a's bf16 output through the entry vs the stock wo_b on that
    output's MXFP8 quantization."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.models.deepseek_v41.amd import akr_config, akr_kernels

    wo_b = _mxfp8_linear(vllm_config, 5120, 8192, "model.layers.0.attn.wo_b", 6)
    dense = akr_kernels.mxfp8_dense_weight(wo_b)
    if dense is None:
        pytest.skip("the stock wo_b kept no preshuffled MXFP8 weight")
    assert akr_config.serves("wo_b", m)
    z = torch.randn(m, 8192, device="cuda", dtype=torch.bfloat16)
    out = akr_kernels.mxfp8_dense(z, dense)
    assert out is not None
    codes, scale = mxfp8_e4m3_quantize(z)
    ref, _ = wo_b(akr_config.quant_activation(codes, scale.view(torch.uint8), z.dtype))
    assert cos(out, ref) > 0.9999


@pytest.mark.parametrize("m", [768, 769, 848, 1024])
def test_wo_a_wo_b_chain_matches_stock(vllm_config, m):
    """The full `wo_b` route through `akr_mla.wo_a_wo_b` (wo_a's bf16 output,
    then the dense entry) against the stock chain (stock grouped wo_a, stock
    wo_b) and against the route's neighbour below 769 rows (wo_a's MXFP8
    epilogue + the stock wo_b on its codes); None, having launched nothing,
    outside 769-1024 rows."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.models.deepseek_v41.amd import akr_config, akr_kernels, akr_mla
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_mxfp8_wo_a_bmm

    wo_a = _mxfp8_linear(
        vllm_config, 8 * 1024, 4096, "model.layers.0.attn.wo_a", 7, bmm_batch_size=8
    )
    wo_b = _mxfp8_linear(vllm_config, 5120, 8192, "model.layers.0.attn.wo_b", 8)
    grouped = akr_kernels.mxfp8_grouped_weight(wo_a, 8, 1024)
    dense = akr_kernels.mxfp8_dense_weight(wo_b)
    if grouped is None or dense is None:
        pytest.skip("the stock wo_a / wo_b kept no preshuffled MXFP8 weight")
    layer = types.SimpleNamespace(
        akr=types.SimpleNamespace(wo_a=grouped, wo_b=dense, want_codes=False),
        gemm_rs=None,
    )
    x = torch.randn(m, 8 * 4096, device="cuda", dtype=torch.bfloat16)
    codes, scale = mxfp8_e4m3_quantize(x)
    qx = akr_config.quant_activation(codes, scale.view(torch.uint8), x.dtype)
    out = akr_mla.wo_a_wo_b(layer, qx)
    assert (out is not None) == akr_config.serves("wo_b", m)
    if out is None:
        return
    assert out.shape == (m, 5120) and out.dtype == torch.bfloat16
    z = rocm_mxfp8_wo_a_bmm(codes, scale.view(torch.uint8), wo_a, 8, 1024)
    ref, _ = wo_b(z.flatten(1).contiguous())
    assert cos(out, ref) > 0.999
    z_codes = akr_mla.wo_a(layer, qx)
    assert z_codes is not None
    epilogue, _ = wo_b(z_codes)
    assert cos(out, epilogue) > 0.999
    # captured once, replayed on new codes: same result as an eager call
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        akr_mla.wo_a_wo_b(layer, qx)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        replayed = akr_mla.wo_a_wo_b(layer, qx)
    new_codes, new_scale = mxfp8_e4m3_quantize(torch.randn_like(x))
    codes.copy_(new_codes)
    scale.copy_(new_scale)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(replayed, akr_mla.wo_a_wo_b(layer, qx))


@pytest.mark.parametrize("m", [16, 192, 1024])
def test_grouped_wo_a_matches_stock(vllm_config, m):
    """wo_a with the MXFP8 epilogue vs the stock grouped GEMM (bf16 out)."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
        mxfp8_e4m3_quantize,
    )
    from vllm.models.deepseek_v41.amd import akr_config, akr_kernels
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_mxfp8_wo_a_bmm

    wo_a = _mxfp8_linear(
        vllm_config, 8 * 1024, 4096, "model.layers.0.attn.wo_a", 4, bmm_batch_size=8
    )
    weight = akr_kernels.mxfp8_grouped_weight(wo_a, 8, 1024)
    if weight is None:
        pytest.skip("the stock wo_a kernel kept no AITER-preshuffled weight")
    assert akr_kernels.mxfp8_grouped_ok(m, weight)
    x = torch.randn(m, 8 * 4096, device="cuda", dtype=torch.bfloat16)
    codes, scale = mxfp8_e4m3_quantize(x)
    qx = akr_config.quant_activation(codes, scale.view(torch.uint8), x.dtype)
    out_codes, out_scale = akr_kernels.mxfp8_grouped_gemm(qx, weight)
    ref = rocm_mxfp8_wo_a_bmm(codes, scale.view(torch.uint8), wo_a, 8, 1024)
    out = dequant_mxfp8_to_bf16(out_codes, out_scale)
    assert cos(out, ref.flatten(1)) > 0.999


def test_lm_head_matches_matmul():
    from vllm.models.deepseek_v41.amd import akr_kernels

    weight = torch.randn(129280, H, device="cuda", dtype=torch.bfloat16) * 0.02
    assert akr_kernels.bf16_linear_ok(weight)
    x = torch.randn(128, H, device="cuda", dtype=torch.bfloat16)
    assert cos(akr_kernels.bf16_linear(x, weight), x @ weight.T) > 0.9999


# ---------------------------------------------------------------- router


@pytest.fixture(scope="module")
def moe(vllm_config):
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.models.deepseek_v41.amd import akr_kernels, akr_moe

    gate = GateLinear(
        input_size=H,
        output_size=384,
        bias=False,
        out_dtype=torch.float32,
        params_dtype=torch.bfloat16,
    ).cuda()
    gen = torch.Generator(device="cuda").manual_seed(5)
    gate.weight.data.copy_(torch.randn(384, H, device="cuda", generator=gen) * 0.05)
    gate.e_score_correction_bias = torch.randn(384, device="cuda", generator=gen) * 0.1
    gate.tid2eid = None
    gate.bias_vl = None
    module = types.SimpleNamespace(
        gate=gate,
        n_activated_experts=6,
        scoring_func="sqrtsoftplus",
        renormalize=True,
        routed_scaling_factor=1.5,
        image_sentinel_lo=0,
    )
    router = akr_moe.Router("router", akr_moe._router_weight(module), module)
    assert router.gate is not None
    akr_kernels.router_warmup(router.gate, 6)
    module.akr = router
    return module


def _stock_router(moe, x):
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        fused_topk_bias,
    )

    logits, _ = moe.gate(x)
    return fused_topk_bias(
        hidden_states=x,
        gating_output=logits,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=moe.gate.e_score_correction_bias,
        topk=6,
        renormalize=True,
        input_tokens=None,
        hash_indices_table=None,
        routed_scaling_factor=1.5,
        bias_vl=None,
        image_sentinel_lo=0,
    )


@pytest.mark.parametrize("m", [1, 16, 192, 768, 769, 848, 1024, 1025])
def test_router_matches_stock(moe, m):
    x = torch.randn(m, H, device="cuda", dtype=torch.bfloat16)
    out = moe.akr.route(moe, x)
    if m > 1024:
        assert out is None
        return
    weights, ids = out
    ref_weights, ref_ids = _stock_router(moe, x)
    # the 6th and 7th scores may tie within the GEMMs' fp32 rounding
    same = (ids.sort(dim=1).values == ref_ids.sort(dim=1).values).all(dim=1)
    assert same.float().mean() >= 0.99
    exact = (ids == ref_ids).all(dim=1)
    torch.testing.assert_close(weights[exact], ref_weights[exact], rtol=1e-4, atol=1e-5)


def test_router_graph_replay(moe):
    x = torch.randn(192, H, device="cuda", dtype=torch.bfloat16)
    weights, ids = _replayed(lambda t: moe.akr.route(moe, t), x)
    ref_weights, ref_ids = moe.akr.route(moe, x)
    assert torch.equal(ids, ref_ids) and torch.equal(weights, ref_weights)


# ---------------------------------------------------------------- shared expert


@pytest.fixture(scope="module")
def mlp(vllm_config):
    from vllm.models.deepseek_v4.amd.model import DeepseekV4MLP
    from vllm.models.deepseek_v41.amd import akr_moe
    from vllm.utils.torch_utils import set_default_torch_dtype

    hf = vllm_config.model_config.hf_config
    with set_default_torch_dtype(torch.bfloat16):
        module = DeepseekV4MLP(
            hidden_size=hf.hidden_size,
            intermediate_size=hf.moe_intermediate_size * hf.n_shared_experts,
            hidden_act=hf.hidden_act,
            swiglu_limit=hf.swiglu_limit,
            quant_config=vllm_config.quant_config,
            reduce_results=False,
            prefix="model.layers.5.ffn.shared_experts",
        ).cuda()
    _load_random(module, seed=3)
    gate_up = akr_moe._shared_gate_up(module)
    if gate_up is None:
        pytest.skip("the stock linears did not keep a preshuffled MXFP8 weight")
    module.akr = akr_moe.SharedExpert(module, gate_up)
    module.akr.warmup(module)
    return module


def _stock_mlp(mlp, x):
    route, mlp.akr = mlp.akr, None
    try:
        return mlp(x)
    finally:
        mlp.akr = route


@pytest.mark.parametrize("m", [16, 48, 192, 384, 769])
def test_shared_expert_matches_stock(mlp, m):
    x = torch.randn(m, H, device="cuda", dtype=torch.bfloat16) * 0.5
    ref = _stock_mlp(mlp, x)
    out = mlp.akr.forward(mlp, x, None)
    if m > 768:
        assert out is None
        return
    assert out is not None and out.shape == ref.shape and out.dtype == ref.dtype
    assert cos(out, ref) > 0.9999


def test_shared_expert_graph_replay(mlp):
    x = torch.randn(16, H, device="cuda", dtype=torch.bfloat16)
    out = _replayed(lambda t: mlp(t), x)
    assert cos(out, _stock_mlp(mlp, x)) > 0.9999


# ---------------------------------------------------------------- mHC seam


@pytest.fixture(scope="module")
def seam(vllm_config):
    from vllm.model_executor.layers.mhc import MHCPreDelayedOp
    from vllm.models.deepseek_v41.amd import akr_mhc

    gen = torch.Generator(device="cuda").manual_seed(0)
    fn = torch.randn(24, HC * H, device="cuda", generator=gen) * 0.02
    layer = types.SimpleNamespace(
        hc_attn_fn=fn,
        hc_ffn_fn=fn.flip(0).contiguous(),
        attn_norm=types.SimpleNamespace(
            weight=(1 + 0.1 * torch.randn(H, device="cuda", generator=gen)).bfloat16()
        ),
    )
    mhc = akr_mhc.AkrMhc(layer, vllm_config)
    mhc.post_load()
    assert mhc.ready
    return types.SimpleNamespace(mhc=mhc, layer=layer, stock=MHCPreDelayedOp())


def _seam_call(seam, m, *, ours, aux=False, quant=False):
    gen = torch.Generator(device="cuda").manual_seed(m)
    residual = (torch.randn(m, HC, H, device="cuda", generator=gen) * 2).bfloat16()
    sublayer_out = torch.randn(m, H, device="cuda", generator=gen).bfloat16()
    post = torch.sigmoid(torch.randn(m, HC, 1, device="cuda", generator=gen)) * 2
    comb = torch.softmax(torch.randn(m, HC, HC, device="cuda", generator=gen), -1)
    pre = torch.sigmoid(torch.randn(m, HC, device="cuda", generator=gen)) + 1e-6
    args = (
        residual,
        seam.layer.hc_ffn_fn,
        torch.tensor([0.5, 0.7, 0.9], device="cuda"),
        torch.linspace(-0.5, 0.5, 24, device="cuda"),
        1e-20,
        1e-6,
        1e-6,
        2.0,
        20,
    )
    kwargs = dict(
        pre_mix=pre,
        norm_weight=seam.layer.attn_norm.weight,
        norm_eps=1e-6,
        sublayer_out=sublayer_out,
        post_layer_mix=post,
        comb_res_mix=comb,
    )
    if not ours:
        return seam.stock(*args, **kwargs)
    seam.mhc.want_aux, seam.mhc.want_quant = aux, quant
    return seam.mhc.pre(seam.stock, *args, **kwargs)


@pytest.mark.parametrize("m", [16, 192, 384, 385])
def test_mhc_seam_matches_stock(seam, m):
    ref = _seam_call(seam, m, ours=False)
    out = _seam_call(seam, m, ours=True)
    for got, want in zip(out, ref):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert cos(got, want) > 0.9999


@pytest.mark.parametrize("m", [16, 768])
def test_mhc_seam_extra_outputs(seam, m):
    """The stream mean (DSpark aux hidden state) at any row count, and the
    MXFP8 copy of the layer input for the MoE."""
    ref = _seam_call(seam, m, ours=False)
    out = _seam_call(seam, m, ours=True, aux=True, quant=m <= 384)
    assert cos(seam.mhc.last_aux, ref[0].float().mean(dim=1)) > 0.9999
    assert cos(out[3], ref[3]) > 0.9999
    if m <= 384:
        codes = seam.mhc.quant_output(torch.bfloat16)
        assert codes is not None and codes.data.shape == (m, H)
