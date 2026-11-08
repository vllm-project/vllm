# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode at DEP (TP1 attention, 96 heads per GPU): the split
MLA query (QueryNorm, the Q_B projection, MlaAbsorb; mla.query_wide) against the
stock numerics (vllm_c RMSNorm, BF16 Q_B rows, BF16 W_UK absorption, per-tensor
FP8 query) and the stock cache writers, and against MlaQuery itself at 96 heads;
the TP1 policies (prims_ts at 96 heads, the split query's overlap modes)."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import _prims_ts_row_buckets

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels need SM100/SM103",
)

HEADS = 96  # TP1


def pitch(heads=HEADS):
    """The packed [q_a | kv_a | gate | wk | wv] row of ``heads`` local heads."""
    return 2112 + heads * 128 + 128 + 32


PITCH = pitch()
FP8_ROW, NVFP4_ROW, UNIT, INDEX_ROW, EPS = 576, 352, 64, 132, 1e-6


def rms_norm(x, weight):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return (x * weight.float()).to(torch.bfloat16)


def cache(blocks, block, row):
    return torch.zeros(blocks, block, row, dtype=torch.uint8, device="cuda")


def split_query(batch, weights, mla, index, nvfp4, heads=HEADS):
    """The split query of ``batch`` rows at ``heads`` local heads: (run(projected,
    slots, q_scale_inv, k_scale) -> (query, q_c, Q_B rows))."""
    import cutlass

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.prepared import (
        compile_core,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.query_stage import (  # noqa: E501
        WideQuery,
        q_b_policies,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (  # noqa: E501
        PreparedProjection,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.query_wide import (  # noqa: E501
        fi_heads,
    )

    specs = tuple(layout(w) for w in (*weights, mla, index))
    device = torch.accelerator.current_device_index()
    kernel = compile_core(
        batch, heads, pitch(heads), specs, device, 1, fp8=True, nvfp4=nvfp4
    )[0]
    assert isinstance(kernel, WideQuery) and kernel.norm is not None
    q_norm, kv_norm, q_b, w_uk, _ = weights
    projection = PreparedProjection(q_b, q_b_policies([batch], heads)).prepare()
    bf = dict(device="cuda", dtype=torch.bfloat16)
    # FlashInfer's head-padded query rows (fi_heads: 128 at 96 heads): the pad
    # heads stay zero.
    padded = torch.zeros(batch, fi_heads(heads), 576, device="cuda", dtype=torch.uint8)
    query = padded[:, :heads]
    q_c = torch.zeros(batch, 1536, **bf)
    rows = torch.zeros(batch, heads * 192, **bf)

    def run(projected, slots, q_scale_inv, k_scale):
        s = stream(torch.device("cuda"))
        eps = cutlass.Float32(EPS)
        kernel.norm(to_cute(projected), to_cute(q_norm), to_cute(q_c), eps, s)
        projection.run(q_c, rows)
        args = (to_cute(projected), to_cute(q_norm), to_cute(kv_norm), to_cute(rows))
        uk = w_uk
        if kernel.bmm:  # the stock W_UK bmm, then MlaQuant (query_wide.absorb_bmm)
            nope = rows.view(batch, heads, 192)[..., :128].transpose(0, 1)
            uk = torch.bmm(nope, w_uk)
        args += (to_cute(uk), to_cute(mla), to_cute(slots, 8))
        args += (to_cute(q_scale_inv, 4), to_cute(k_scale, 4), eps)
        kernel.absorb(*args, to_cute(query), s)
        assert not padded[:, heads:].any(), "the pad heads stay zero"
        return query, q_c, rows

    return run


def inputs(batch, row, block=64, heads=HEADS):
    torch.manual_seed(batch)
    bf, dev = torch.bfloat16, "cuda"
    projected = torch.randn(batch, pitch(heads), device=dev, dtype=bf)
    q_norm = torch.rand(1536, device=dev, dtype=bf) + 0.5
    kv_norm = torch.rand(512, device=dev, dtype=bf) + 0.5
    q_b = torch.randn(heads * 192, 1536, device=dev, dtype=bf) / 40
    w_uk = torch.randn(heads, 128, 512, device=dev, dtype=bf) / 12
    w_uv = torch.randn(heads, 128, 512, device=dev, dtype=bf).transpose(1, 2)
    blocks = max(4, -(-(batch + 1) // block))
    mla = cache(blocks, block, row)
    index = cache(8, UNIT, INDEX_ROW)
    slots = torch.randperm(blocks * block - 1, device=dev)[:batch]
    slots[0] = blocks * block - 1  # the last row
    if batch > 2:
        slots[1] = -1  # a padded row writes nothing
    return projected, (q_norm, kv_norm, q_b, w_uk, w_uv), mla, index, slots


# (local heads, rows): TP1 at every batch; TP8 (12 heads) from SPLIT_FROM rows,
# including batches whose token tiles split over CTAs unevenly (MlaAbsorb splits).
SHAPES = [(96, 1), (96, 5), (96, 20), (96, 80), (96, 320)]
SHAPES += [(12, 25), (12, 40), (12, 80), (12, 160), (12, 320)]


@pytest.mark.parametrize("heads,batch", SHAPES)
def test_split_query_fp8(heads, batch):
    """Per-tensor FP8 cache (FlashInfer sparse): the query rows, the q_c rows and the
    KV row as the stock path computes and writes them (concat_and_cache_mla "fp8"),
    with non-unit scales."""
    from vllm import _custom_ops as ops

    projected, weights, mla, index, slots = inputs(batch, FP8_ROW, heads=heads)
    q_norm, kv_norm, q_b, w_uk, _ = weights
    q_scale = torch.full((1,), 0.5, device="cuda")
    k_scale = torch.full((1,), 2.0, device="cuda")
    run = split_query(batch, weights, mla, index, nvfp4=False, heads=heads)
    query, q_c, rows = run(projected, slots, 1 / q_scale, k_scale)
    torch.accelerator.synchronize()

    ref_q_c = rms_norm(projected[:, :1536], q_norm)
    torch.testing.assert_close(q_c, ref_q_c, atol=1e-2, rtol=1e-2)
    # The Q_B rows: BF16 of an FP32 sum (order aside, the stock GEMM's).
    ref_rows = (q_c.float() @ q_b.float().T).to(torch.bfloat16)
    torch.testing.assert_close(rows, ref_rows, atol=1e-2, rtol=1.6e-2)
    # The query from the kernel's own Q_B rows: the absorption's FP32 sum of BF16
    # products, rounded to BF16, then scaled to FP8 (as the stock path).
    q = rows.view(batch, heads, 192)
    latent = torch.bmm(q[..., :128].transpose(0, 1).float(), w_uk.float())
    latent = latent.transpose(0, 1).to(torch.bfloat16).float()
    ref = torch.cat([latent, q[..., 128:].float()], -1) / q_scale
    expected = ref.to(torch.float8_e4m3fn).view(torch.uint8)
    # FP8 bytes: equal up to the FP32 summation order of the absorption.
    assert (query != expected).float().mean() < 2e-3
    torch.testing.assert_close(
        query.view(torch.float8_e4m3fn).float(), ref, atol=3e-2, rtol=7e-2
    )

    stock = torch.zeros_like(mla)
    kv = rms_norm(projected[:, 1536:2048], kv_norm)
    ops.concat_and_cache_mla(kv, projected[:, 2048:2112], stock, slots, "fp8", k_scale)
    live = slots[slots >= 0]
    ours, theirs = (c.view(-1, FP8_ROW)[live] for c in (mla, stock))
    assert torch.equal(ours[:, 512:], theirs[:, 512:])  # k_pe
    # Latent bytes differ only where the RMSNorm rounds differently.
    assert (ours[:, :512] != theirs[:, :512]).float().mean() < 0.01
    # Nothing written for the padded row or past the live slots.
    written = (mla.view(-1, FP8_ROW) != 0).any(-1).nonzero().flatten()
    assert set(written.tolist()) <= set(live.tolist())


@pytest.mark.parametrize("heads,batch", [(96, 5), (96, 80), (12, 40), (12, 320)])
def test_split_query_matches_mla_query(heads, batch):
    """The split query against MlaQuery (the one-kernel query it replaces at TP1,
    and at TP8 from SPLIT_FROM rows): the same KV row bytes, the same query bytes up
    to the Q_B summation order."""
    import cutlass
    import cutlass.cute as cute

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.query import (
        MlaQuery,
    )

    projected, weights, mla, index, slots = inputs(batch, FP8_ROW, heads=heads)
    q_norm, kv_norm, q_b, w_uk, _ = weights
    unit = torch.ones(1, device="cuda")
    run = split_query(batch, weights, mla, index, nvfp4=False, heads=heads)
    query, q_c, _ = run(projected, slots, unit, unit)

    reference = MlaQuery(heads, batch, dsa=True, fp8=True)
    ref_mla = torch.zeros_like(mla)
    ref_query = torch.zeros_like(query)
    ref_q_c = torch.zeros_like(q_c)
    args = (to_cute(projected), *(to_cute(w) for w in (q_norm, kv_norm, q_b, w_uk)))
    args += (to_cute(ref_mla), to_cute(slots, 8), to_cute(unit, 4), to_cute(unit, 4))
    args += (cutlass.Float32(EPS), to_cute(ref_query), to_cute(ref_q_c))
    compiled = cute.compile(reference, *args, stream(torch.device("cuda")))
    compiled(*args, stream(torch.device("cuda")))
    torch.accelerator.synchronize()
    assert torch.equal(q_c, ref_q_c)
    assert torch.equal(mla, ref_mla)
    assert (query != ref_query).float().mean() < 5e-3


@pytest.mark.parametrize("heads,batch", [(96, 5), (96, 80), (96, 160), (12, 80)])
def test_split_query_nvfp4(heads, batch):
    """nvfp4_ds_mla cache (prims_ts): the KV row bytes as MlaQuery's nvfp4 insert
    (byte-exact with vLLM's writer, test_specialized_decode_dsa) and the FP8 query."""
    projected, weights, mla, index, slots = inputs(batch, NVFP4_ROW, heads=heads)
    unit = torch.ones(1, device="cuda")
    run = split_query(batch, weights, mla, index, nvfp4=True, heads=heads)
    query, _, rows = run(projected, slots, unit, unit)
    torch.accelerator.synchronize()
    q = rows.view(batch, heads, 192)
    w_uk = weights[3]
    latent = torch.bmm(q[..., :128].transpose(0, 1).float(), w_uk.float())
    latent = latent.transpose(0, 1).to(torch.bfloat16).float()
    ref = torch.cat([latent, q[..., 128:].float()], -1)
    torch.testing.assert_close(
        query.view(torch.float8_e4m3fn).float(), ref, atol=3e-2, rtol=7e-2
    )
    live = slots[slots >= 0]
    assert (mla.view(-1, NVFP4_ROW)[live] != 0).any(-1).all()


def test_tp1_policies():
    """96 heads: padded to 128 for FlashInfer (fi_heads) and there the stock
    decode's kernel (trtllm-gen on fp8, prims_ts on nvfp4); unpadded 96 prims_ts
    only (trtllm-gen takes at most 64 or exactly 128 heads); the split query on its
    own overlap modes; 12 heads keep the TP8 tables."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.attn_fi import (
        KERNELS,
        kernel_for,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.overlap import (
        WIDE_MODES,
        overlap_for,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.query_wide import (
        SPLIT_FROM,
        absorb_splits,
        fi_heads,
        split_query,
        wide_query,
    )

    assert wide_query(96) and not wide_query(12) and not wide_query(6)
    # TP8 / DCP8 (12 heads): the split query from SPLIT_FROM rows (under DCP
    # MlaAbsorb publishes into the q mailbox).
    assert not split_query(12, SPLIT_FROM - 1) and split_query(12, SPLIT_FROM)
    assert split_query(12, 320, dcp=8) and split_query(96, 1)
    assert absorb_splits(320, 96) == 1 and absorb_splits(320, 12) > 1
    assert fi_heads(96) == 128 and fi_heads(12) == 12 and fi_heads(128) == 128
    for rows in (1, 5, 20, 80, 160, 320):
        assert kernel_for(rows, False, KERNELS, 96) == "prims_ts"
        assert overlap_for(rows, wide=True) in WIDE_MODES
        assert kernel_for(rows, False, KERNELS, fi_heads(96)) == "trtllm"
        assert kernel_for(rows, False, ("prims_ts",), fi_heads(96)) == "prims_ts"
    assert kernel_for(5, False, KERNELS, 12) == "trtllm"
    assert kernel_for(30, True, KERNELS, 128) == kernel_for(30, True, KERNELS)
    with pytest.raises(ValueError):
        overlap_for(5, "late_front", wide=True)


def test_dep_linear_policies():
    """The DEP projection / o_proj tables: split-K per batch bin, cuBLAS past the
    split-K crossover; the per-cache-dtype row caps."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.dep import (
        DEP_MAX_ROWS,
        DEP_OPROJ,
        DEP_PROJECTION,
        DEP_ROWS,
        dep_policies,
    )

    rows = [1, 5, 20, 80, 160, 240, 320]
    projection = dep_policies(DEP_PROJECTION, rows)
    assert set(projection) == {1, 5, 20, 80, 160}  # 240 / 320: cuBLAS
    # One batch tile per weight stream up to 160 rows.
    assert all(t[1] >= min(b, 160) for b, t in projection.items() if b > 64)
    assert set(dep_policies(DEP_OPROJ, rows)) == {1, 5, 20, 80}
    assert DEP_ROWS["fp8"] == DEP_ROWS["nvfp4_ds_mla"] == DEP_MAX_ROWS


def dep_layer_step(core, batch, q, contexts):
    """One decode step of a real 96-head (TP1) K3.1 DSA layer through the stock
    KimiK3DSAAttention forward and through runtime.dep.DepAttention.run_wide (the
    DEP step's attention: split-K / cuBLAS projection, PreparedDsa at 96 heads,
    o_proj), each over its own copy of the same caches (written by the stock
    writers), on the same metadata. Returns both outputs, both rows entering
    o_proj, the top-k and what the FP32 reference needs."""
    from types import SimpleNamespace

    from tests.models.kimi_k3.nvidia.specialized_decode.test_specialized_decode_dsa import (  # noqa: E501
        GAP,
        LAYER_CORES,
        LAYER_HIDDEN,
        UNIT,
        assert_index_nan_free,
        common_metadata,
        dsa_layer,
        index_row,
        layer_config,
        owned_slots,
        paged_rows,
    )
    from vllm import _custom_ops as ops
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import create_forward_context, override_forward_context
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer import (
        indexer_k_norm_quant_insert,
    )
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import (
        mxfp4_k_norm_quant_insert,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.attn_fi import (
        fi_kernels,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.dep import (
        DepAttention,
    )
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import get_block_table_width

    cache_dtype, _, kernel = LAYER_CORES[core]
    config = layer_config(core, q)
    length = config.model_config.max_model_len
    row = {"fp8": FP8_ROW, "nvfp4_ds_mla": NVFP4_ROW}[cache_dtype]
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    bf = torch.bfloat16
    with set_current_vllm_config(config):
        topk_buffer = torch.full((length, 2048), -1, dtype=torch.int32, device=dev)
        attn = dsa_layer(config, topk_buffer, HEADS)
        layer, indexer = attn.mla_attn.mla_attn, attn.dsa_indexer
        spec = layer.get_kv_cache_spec(config)
        builder = layer.attn_backend.get_builder_cls()(
            spec, [layer.layer_name], config, dev
        )
        index_spec = indexer.k_cache.get_kv_cache_spec(config)
        index_builder = DeepseekV32IndexerMetadataBuilder(
            index_spec,
            [indexer.k_cache.prefix],
            config,
            dev,
            block_table_width=get_block_table_width(
                index_spec.max_num_blocks_per_req(config, length), UNIT
            ),
        )
        reqs = batch // q
        gen = torch.Generator(device=dev).manual_seed(batch + q)
        contexts = [contexts[r % len(contexts)] + 3 * r for r in range(reqs)]
        mla, mla_table = paged_rows([c + q for c in contexts], UNIT, row, UNIT, gen)
        index, index_table = paged_rows(
            [c + q for c in contexts], UNIT, index_row(indexer), GAP, gen
        )
        old = [range(c) for c in contexts]
        old_mla = owned_slots(mla_table, old)
        old_index = owned_slots(index_table, old)
        n = old_mla.numel()
        unit = torch.ones(1, dtype=torch.float32, device=dev)
        kv_c = torch.randn(n, 512, device=dev, dtype=bf, generator=gen)
        k_pe = torch.randn(n, 64, device=dev, dtype=bf, generator=gen)
        ops.concat_and_cache_mla(kv_c, k_pe, mla, old_mla, cache_dtype, unit)
        keys = torch.randn(n, 128, device=dev, dtype=bf, generator=gen)
        scratch = torch.empty(n, 2048, dtype=torch.int32, device=dev)
        norm = torch.ones(128, dtype=bf, device=dev)
        args = (norm, torch.zeros_like(norm), EPS, old_index, index, scratch, n)
        insert = indexer_k_norm_quant_insert
        if indexer.use_fp4_cache:
            insert = mxfp4_k_norm_quant_insert
        insert(keys, *args)
        assert_index_nan_free(index)
        new = [range(c, c + q) for c in contexts]
        new_mla = owned_slots(mla_table, new)
        new_index = owned_slots(index_table, new)
        hidden = torch.randn(batch, LAYER_HIDDEN, device=dev, dtype=bf, generator=gen)
        positions = torch.tensor([p for pos in new for p in pos], device=dev)
        names = layer.layer_name, indexer.k_cache.prefix

        def step(mla_cache, index_cache):
            layer.kv_cache, indexer.k_cache.kv_cache = mla_cache, index_cache
            md = builder.build(0, common_metadata(contexts, q, mla_table, new_mla))
            index_md = index_builder.build(
                0, common_metadata(contexts, q, index_table, new_index)
            )
            context = create_forward_context(
                dict(zip(names, (md, index_md))),
                config,
                slot_mapping=dict(zip(names, (new_mla, index_md.slot_mapping))),
            )
            return md, override_forward_context(context)

        entering = []
        hook = attn.o_proj.register_forward_pre_hook(
            lambda _, args: entering.append(args[0].clone())
        )
        stock_mla, stock_index = mla.clone(), index.clone()
        _, context = step(stock_mla, stock_index)
        with context:
            stock = attn(positions, hidden)
        hook.remove()
        stock_topk = topk_buffer[:batch].clone()

        policy = dict(kernels=(kernel,)) if kernel not in (None, "auto") else {}
        runtime = SimpleNamespace(
            device=dev,
            buffers={},
            wide=(batch,),
            queries=(q,),
            fi_kernels=policy.get("kernels", fi_kernels(cache_dtype)),
            prims_ts_rows=_prims_ts_row_buckets(config),
        )
        dep = DepAttention(SimpleNamespace(self_attn=attn), runtime)
        ours_mla, ours_index = mla.clone(), index.clone()
        md, context = step(ours_mla, ours_index)
        dep.prepare_layout([ours_mla])
        if dep.core.fused is not None:
            dep.core.fused.write_ids(topk_buffer)
        topk_buffer.fill_(-1)
        with context:
            out = dep.run_wide(hidden, [ours_mla], md, {names[0]: new_mla}, q)
        torch.accelerator.synchronize()
    return SimpleNamespace(
        attn=attn,
        out=out,
        stock=stock,
        local=dep.local[:batch].clone(),
        entering=entering[0],
        topk=topk_buffer[:batch].clone(),
        stock_topk=stock_topk,
        projected=dep.projected[:batch].clone(),
        caches=(ours_mla, stock_mla),
        mla_table=mla_table,
        kv=cache_dtype,
    )


def relative(x, ref):
    return float((x.float() - ref.float()).norm() / ref.float().norm())


@pytest.mark.parametrize("core", ["fi-auto", "fi-prims_ts"])
@pytest.mark.parametrize("batch,q", [(5, 5), (20, 5), (80, 5), (8, 1)])
def test_dep_attention_matches_stock_short_context(
    workspace_init, dist_init, core, batch, q
):
    """DepAttention.run_wide against the stock 96-head DSA layer at contexts below
    the top-k (2048): both attend over every token (no top-k tie can separate
    them), and both pre-o_proj rows against FP32 attention over the dequantized
    cache: ours at least as accurate as stock.

    The two paths differ by ~2.8% (rel. RMS) at equal accuracy (~3.4% each vs
    FP32), from two sources, both rounding-order effects (bisected stage by stage):
    (1) the stock q_a / kv_a RMSNorm (native) rounds x * rstd to BF16 before the
    weight multiply, ours rounds x * rstd * w once (the vllm_c kernel's math): 27%
    of the q_c elements differ by one BF16 ulp, which flips ~10% of the FP8 query
    codes (ours matches a single-rounding reference to <0.05% of bytes); (2) the FP8
    sparse MLA kernels quantize P to E4M3 per KV tile, so the same top-k set in another
    order changes the output by ~2.4% (the stock kernel against itself with its
    top-k rows permuted); the fused indexer emits key order, stock's top-k its own
    order. With identical inputs the core is the stock kernel."""
    from tests.models.kimi_k3.nvidia.specialized_decode.test_specialized_decode_dsa import (  # noqa: E501
        layer_reference,
    )

    step = dep_layer_step(core, batch, q, (600, 1100, 1500, 1980, 37))
    # Every token selected on both paths (contexts + q <= 2048).
    for t in range(batch):
        a = set(step.topk[t].tolist()) - {-1}
        b = set(step.stock_topk[t].tolist()) - {-1}
        assert a == b, t
    # Each pre-o_proj row set against its own FP32 reference (same top-k).
    args = step.attn, step.projected
    ours_ref = layer_reference(
        *args, step.caches[0], step.mla_table, step.topk, q, step.kv
    )
    stock_ref = layer_reference(
        *args, step.caches[1], step.mla_table, step.stock_topk, q, step.kv
    )
    ours_err, stock_err = (
        relative(step.local, ours_ref),
        relative(step.entering, stock_ref),
    )
    between = relative(step.local, step.entering)
    print(
        f"pre-o_proj vs FP32: ours {ours_err:.4f} stock {stock_err:.4f}; "
        f"ours vs stock {between:.4f}; layer output vs stock "
        f"{relative(step.out, step.stock):.4f}"
    )
    assert ours_err <= 1.1 * stock_err + 1e-3, (ours_err, stock_err)
    assert between < 0.04, between
    assert relative(step.out, step.stock) < 0.04


@pytest.mark.parametrize("heads,batch", [(96, 80), (96, 320), (12, 80), (12, 320)])
def test_value_bmm_gate_matches_mla_value(heads, batch):
    """The wide value projection (the stock W_UV bmm over the heads, then MlaGate)
    against MlaValue: the same gated rows up to the FP32 summation order of W_UV
    (one BF16 rounding each), an empty request's rows zero."""
    import cutlass.cute as cute

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.value import (
        MlaGate,
        MlaValue,
        value_bmm,
    )

    assert value_bmm(heads, batch) == (heads >= 64)  # TP8 keeps MlaValue
    q = 5
    torch.manual_seed(batch + heads)
    bf = dict(device="cuda", dtype=torch.bfloat16)
    padded = 128 if 64 < heads < 128 else heads  # FlashInfer's latent rows
    latent = torch.randn(batch, padded, 512, **bf)[:, :heads]
    w_uv = (torch.randn(heads, 128, 512, **bf) / 20).transpose(1, 2)  # latent-major
    pitch = heads * 128 + 64
    gate = torch.randn(batch, pitch, **bf)[:, : heads * 128]
    lengths = torch.randint(1, 1000, (batch // q,), device="cuda", dtype=torch.int32)
    lengths[1] = 0  # an empty request: its rows stay zero
    s = stream(torch.device("cuda"))

    ref = torch.zeros(batch, heads * 128, **bf)
    args = (to_cute(latent), to_cute(w_uv), to_cute(gate), to_cute(lengths, 4))
    args += (to_cute(ref),)
    cute.compile(MlaValue(heads, batch, q), *args, s)(*args, s)

    values = torch.bmm(latent.transpose(0, 1), w_uv)  # [H, B, 128]
    ours = torch.zeros_like(ref)
    args = (to_cute(values), to_cute(gate), to_cute(lengths, 4), to_cute(ours))
    cute.compile(MlaGate(heads, batch, q), *args, s)(*args, s)
    torch.accelerator.synchronize()
    assert not ours[q : 2 * q].any() and not ref[q : 2 * q].any()
    # Exact formula: BF16(values) * sigmoid(gate) (FP32), rounded to BF16.
    expected = values.transpose(0, 1).reshape(batch, -1).float()
    expected = (expected * torch.sigmoid(gate.float())).to(torch.bfloat16)
    expected[q : 2 * q] = 0
    torch.testing.assert_close(ours, expected, atol=1e-6, rtol=8e-3)
    # Against MlaValue: the W_UV sums differ only in FP32 order.
    assert (ours != ref).float().mean() < 0.02
    torch.testing.assert_close(ours.float(), ref.float(), atol=2e-2, rtol=2e-2)
