# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode (decode_backend=latency_specialized): the DSA
layer kernels against the stock cache writers, quantizers and a torch sparse
attention reference."""

import gc
import math
import os
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels need SM100/SM103",
)

UNIT, MLA_ROW, INDEX_ROW = 64, 656, 132  # indexer page tokens; cache row bytes
FP4_INDEX_ROW = 68  # MXFP4 indexer cache row (indexer_kv_dtype="mxfp4")
FP8_ROW = 576  # per-tensor FP8 MLA row (FlashInfer sparse)
NVFP4_ROW = 352  # nvfp4_ds_mla MLA row (FlashInfer sparse prims_ts)
GAP = 36  # rows of the other layers' pages between two blocks of a packed cache
FAR = 5 << 30  # bytes a far packed cache spans: its later blocks lie past 2 GiB
HEADS = 12  # TP8
GATE = HEADS * 128
PITCH = 2112 + GATE + 128 + 32  # packed [q_a | kv_a | gate | wk | wv] row
EPS = 1e-6


def packed_cache(blocks, block, row, gap=GAP):
    """One layer's view of a packed (BLHNC) cache: ``blocks`` blocks of ``block``
    rows, ``gap`` rows of other layers' pages between two consecutive blocks
    (``FAR``: as many as spread the blocks over ``FAR`` bytes)."""
    stride = FAR // blocks // 256 * 256 if gap == FAR else (block + gap) * row
    raw = torch.zeros(blocks * stride, dtype=torch.uint8, device="cuda")
    return torch.as_strided(raw, (blocks, block, row), (stride, row, 1))


def index_blocks(index):
    """Each indexer page's bytes: its keys (FP8, or MXFP4: 64 E2M1 bytes), then
    their scales (four packed UE8M0 bytes per key on SM100's MX path, else FP32)."""
    row = index.shape[2]
    return torch.as_strided(
        index, (index.shape[0], index.shape[1] * row), (index.stride(0), 1)
    )


def key_bytes(index):
    """Key bytes per token of an indexer cache (128 FP8, 64 MXFP4)."""
    return index.shape[2] - 4


def index_row(indexer):
    """The indexer's cache row bytes (its format: kimi_k3_indexer_uses_fp4)."""
    return FP4_INDEX_ROW if indexer.use_fp4_cache else INDEX_ROW


def assert_index_nan_free(index):
    """No NaN codes in an indexer cache: its E4M3 keys (0x7F / 0xFF; E2M1 has
    none) and their UE8M0 scale bytes (0xFF). NaN keys give NaN logits, on which
    DeepSelect returns out-of-range ids and the DCP merge's pack kernel faults
    (#63)."""
    pages, kb = index_blocks(index), key_bytes(index)
    keys, scales = pages[:, : UNIT * kb], pages[:, UNIT * kb :]
    if kb == 128:
        assert not ((keys & 0x7F) == 0x7F).any()
    assert not (scales == 0xFF).any()


def rms_norm(x, weight):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return (x * weight.float()).to(torch.bfloat16)


def kernels(batch, mla, index, *weights, fp8=False, nvfp4=False):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        layout,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.prepared import (
        compile_core,
    )

    specs = tuple(layout(w) for w in (*weights, mla, index))
    device = torch.accelerator.current_device_index()
    return compile_core(batch, HEADS, PITCH, specs, device, 1, fp8=fp8, nvfp4=nvfp4)


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def nvfp4_rows(rows, e4m3=False):
    """Dequantized [..., 576] values of nvfp4_ds_mla rows [..., 352]: E2M1 latent
    (low nibble first) times its block's E4M3 scale, then the E4M3 k_pe. ``e4m3``:
    the latent rounded to E4M3, as prims_ts converts it for its FP8 MMAs."""
    table = torch.tensor(E2M1, device=rows.device)
    table = torch.cat([table, -table])
    codes = torch.stack([rows[..., :256] & 15, rows[..., :256] >> 4], -1)
    values = table[codes.long()].flatten(-2)
    block = torch.arange(32, device=rows.device)
    at = 320 + 8 * (block % 4) + block // 4
    scales = rows[..., at].view(torch.float8_e4m3fn).float()
    latent = (values.unflatten(-1, (32, 16)) * scales.unsqueeze(-1)).flatten(-2)
    if e4m3:
        latent = latent.to(torch.float8_e4m3fn).float()
    return torch.cat([latent, rows[..., 256:320].view(torch.float8_e4m3fn).float()], -1)


def random_nvfp4(*shape, gen=None):
    """nvfp4_ds_mla rows [*shape, 352]: random E2M1 codes, k_pe of magnitude below 2
    and block scales in [2^-6, 2^-1]."""
    rows = torch.randint(
        0, 256, (*shape, NVFP4_ROW), dtype=torch.uint8, device="cuda", generator=gen
    )
    rows[..., 256:320] &= 0xBF
    rows[..., 320:] = 0x08 + rows[..., 320:] % 0x28  # positive normal E4M3
    return rows


@pytest.mark.parametrize(
    "block,gap", [(64, GAP), (1024, GAP), (64, FAR)], ids=["64", "1024", "far"]
)
@pytest.mark.parametrize("batch", [1, 2, 5, 16, 64, 80, 320])
def test_query_and_cache_insert(batch, block, gap):
    """BF16 query rows (W_UK absorbed), normalized q_c and the fp8_ds_mla KV row
    as the stock path computes and writes them, into the packed MLA blocks."""
    import cutlass

    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )

    torch.manual_seed(batch)
    bf, dev = torch.bfloat16, "cuda"
    projected = torch.randn(batch, PITCH, device=dev, dtype=bf)
    q_norm = torch.rand(1536, device=dev, dtype=bf) + 0.5
    kv_norm = torch.rand(512, device=dev, dtype=bf) + 0.5
    q_b = torch.randn(HEADS * 192, 1536, device=dev, dtype=bf) / 40
    w_uk = torch.randn(HEADS, 128, 512, device=dev, dtype=bf) / 12
    w_uv = torch.randn(HEADS, 128, 512, device=dev, dtype=bf).transpose(1, 2)
    blocks = max(4 if gap == FAR else 2, -(-(batch + 1) // block))
    mla = packed_cache(blocks, block, MLA_ROW, gap)
    index = packed_cache(8, UNIT, INDEX_ROW)
    slots = torch.randperm(blocks * block - 1, device=dev)[:batch]
    slots[0] = blocks * block - 1  # the last row
    if batch > 2:
        slots[1] = -1  # a padded row writes nothing
    weights = q_norm, kv_norm, q_b, w_uk, w_uv
    query_kernel, _, _ = kernels(batch, mla, index, *weights)
    query = torch.zeros(batch, 64, 576, device=dev, dtype=bf)
    q_c = torch.zeros(batch, 1536, device=dev, dtype=bf)
    unit = torch.ones(1, device=dev, dtype=torch.float32)
    args = (to_cute(projected), *(to_cute(w) for w in weights[:4]), to_cute(mla))
    args += (to_cute(slots, 8), to_cute(unit, 4), to_cute(unit, 4))
    query_kernel(*args, cutlass.Float32(EPS), to_cute(query), to_cute(q_c), stream(dev))

    ref_q_c = rms_norm(projected[:, :1536], q_norm)
    torch.testing.assert_close(q_c, ref_q_c, atol=1e-2, rtol=1e-2)
    q = (ref_q_c @ q_b.T).view(batch, HEADS, 192)
    latent = torch.bmm(q[..., :128].transpose(0, 1), w_uk).transpose(0, 1)
    torch.testing.assert_close(
        query[:, :HEADS], torch.cat([latent, q[..., 128:]], -1), atol=2e-2, rtol=2e-2
    )
    assert (query[:, HEADS:] == 0).all()

    stock = packed_cache(blocks, block, MLA_ROW, gap)
    kv = rms_norm(projected[:, 1536:2048], kv_norm)
    ops.concat_and_cache_mla(
        kv, projected[:, 2048:2112], stock, slots, "fp8_ds_mla", unit
    )
    live = slots[slots >= 0]
    ours, ref = (c.reshape(-1, MLA_ROW)[live] for c in (mla, stock))
    assert torch.equal(ours[:, 512:], ref[:, 512:])  # tile scales and k_pe
    # Latent bytes differ only where the RMSNorm rounds differently.
    assert (ours[:, :512] != ref[:, :512]).float().mean() < 0.01


def call_query(kernel, args, eps, query, q_c, weights, projected):
    """compile_query's kernel over MlaQuery's arguments (``args`` without eps and
    the outputs): MlaQuery, or the split query (WideQuery, wide batches)."""
    import cutlass

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.query_stage import (  # noqa: E501
        WideQuery,
    )

    if isinstance(kernel, WideQuery):
        return run_split_query(kernel, args, weights, projected, query, q_c, eps)
    out = (cutlass.Float32(eps), to_cute(query), to_cute(q_c))
    kernel(*args, *out, stream(projected.device))


def run_split_query(kernel, args, weights, projected, query, q_c, eps=EPS):
    """The split query (mla.query_wide) of compile_query's WideQuery over MlaQuery's
    arguments: QueryNorm, the Q_B projection, then MlaAbsorb (or the W_UK bmm and
    MlaQuant)."""
    import cutlass

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.query_stage import (  # noqa: E501
        q_b_policies,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (  # noqa: E501
        PreparedProjection,
    )

    batch, s, eps = projected.shape[0], stream(projected.device), cutlass.Float32(eps)
    q_norm, _, q_b, w_uk = weights[:4]
    kernel.norm(to_cute(projected), to_cute(q_norm), to_cute(q_c), eps, s)
    heads = w_uk.shape[0]
    rows = torch.empty(batch, heads * 192, device=q_c.device, dtype=q_c.dtype)
    PreparedProjection(q_b, q_b_policies([batch], heads)).prepare().run(q_c, rows)
    uk = w_uk
    if kernel.bmm:  # the stock W_UK bmm, then MlaQuant
        uk = torch.bmm(rows.view(batch, heads, 192)[..., :128].transpose(0, 1), w_uk)
    args = (*args[:3], to_cute(rows), to_cute(uk), *args[5:])
    kernel.absorb(*args, eps, to_cute(query), s)


@pytest.mark.parametrize("gap", [GAP, FAR])
@pytest.mark.parametrize("batch", [1, 5, 64, 80, 320])
def test_fp8_query_and_cache_insert(batch, gap):
    """FlashInfer core: the per-tensor FP8 query rows [B, H, 576] and the FP8 KV row
    in the packed MLA blocks, as the stock FlashInfer path quantizes and writes them
    (concat_and_cache_mla "fp8"), with non-unit scales."""
    import cutlass

    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.query_stage import (  # noqa: E501
        WideQuery,
    )

    torch.manual_seed(batch)
    bf, dev, fp8 = torch.bfloat16, "cuda", torch.float8_e4m3fn
    projected = torch.randn(batch, PITCH, device=dev, dtype=bf)
    q_norm = torch.rand(1536, device=dev, dtype=bf) + 0.5
    kv_norm = torch.rand(512, device=dev, dtype=bf) + 0.5
    q_b = torch.randn(HEADS * 192, 1536, device=dev, dtype=bf) / 40
    w_uk = torch.randn(HEADS, 128, 512, device=dev, dtype=bf) / 12
    w_uv = torch.randn(HEADS, 128, 512, device=dev, dtype=bf).transpose(1, 2)
    block = 64
    blocks = max(4, -(-(batch + 1) // block))
    mla = packed_cache(blocks, block, FP8_ROW, gap)
    index = packed_cache(8, UNIT, INDEX_ROW)
    slots = torch.randperm(blocks * block - 1, device=dev)[:batch]
    slots[0] = blocks * block - 1  # the last row
    if batch > 2:
        slots[1] = -1  # a padded row writes nothing
    weights = q_norm, kv_norm, q_b, w_uk, w_uv
    query_kernel, _, _ = kernels(batch, mla, index, *weights, fp8=True)
    query = torch.zeros(batch, HEADS, 576, device=dev, dtype=torch.uint8)
    q_c = torch.zeros(batch, 1536, device=dev, dtype=bf)
    q_scale = torch.full((1,), 0.5, device=dev)
    k_scale = torch.full((1,), 2.0, device=dev)
    args = (to_cute(projected), *(to_cute(w) for w in weights[:4]), to_cute(mla))
    # The query multiplies by 1 / q_scale; the KV row is divided by k_scale.
    args += (to_cute(slots, 8), to_cute(1 / q_scale, 4), to_cute(k_scale, 4))
    if isinstance(query_kernel, WideQuery):  # the split query (wide batches)
        run_split_query(query_kernel, args, weights, projected, query, q_c)
    else:
        args += (cutlass.Float32(EPS), to_cute(query), to_cute(q_c), stream(dev))
        query_kernel(*args)

    ref_q_c = rms_norm(projected[:, :1536], q_norm)
    torch.testing.assert_close(q_c, ref_q_c, atol=1e-2, rtol=1e-2)
    q = (ref_q_c @ q_b.T).view(batch, HEADS, 192)
    latent = torch.bmm(q[..., :128].transpose(0, 1), w_uk).transpose(0, 1)
    ref_query = torch.cat([latent, q[..., 128:]], -1).float() / q_scale
    torch.testing.assert_close(query.view(fp8).float(), ref_query, atol=3e-2, rtol=7e-2)

    stock = packed_cache(blocks, block, FP8_ROW, gap)
    kv = rms_norm(projected[:, 1536:2048], kv_norm)
    ops.concat_and_cache_mla(kv, projected[:, 2048:2112], stock, slots, "fp8", k_scale)
    live = slots[slots >= 0]
    ours, ref = (c[live // block, live % block] for c in (mla, stock))
    assert torch.equal(ours[:, 512:], ref[:, 512:])  # k_pe
    # Latent bytes differ only where the RMSNorm rounds differently.
    assert (ours[:, :512] != ref[:, :512]).float().mean() < 0.01


@pytest.mark.parametrize(
    "block,gap", [(64, GAP), (1024, GAP), (64, FAR)], ids=["64", "1024", "far"]
)
@pytest.mark.parametrize("batch", [1, 5, 16, 60, 64])
def test_nvfp4_cache_insert(batch, block, gap):
    """FlashInfer prims_ts on nvfp4_ds_mla: the FP8 query rows as on a per-tensor
    FP8 cache, and the KV row byte-exact with the stock writer (concat_and_cache_mla
    "nvfp4_ds_mla"): E2M1 latent, unscaled E4M3 k_pe, permuted E4M3 block scales.
    The latent rows have an RMS of exactly 1 and the norm no epsilon, so both
    RMSNorms are exact; the norm weights span the block scales' range (the 2^-9
    floor, E4M3 subnormals, saturation), with zero blocks and padded rows."""
    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        to_cute,
    )

    torch.manual_seed(batch + block)
    bf, dev = torch.bfloat16, "cuda"
    projected = torch.randn(batch, PITCH, device=dev, dtype=bf)
    # Each latent row: 64 +-2, 256 +-1 and 192 zeros (sum of squares 512).
    magnitudes = torch.tensor([2.0] * 64 + [1.0] * 256 + [0.0] * 192, device=dev)
    for t in range(batch):
        row = magnitudes[torch.randperm(512, device=dev)]
        if t == 0:
            row = magnitudes.flip(0)  # zero blocks first
        sign = torch.randint(0, 2, (512,), device=dev) * 2 - 1
        projected[t, 1536:2048] = (row * sign).to(bf)
    rope = torch.randn(batch, 64, device=dev) * 2.0 ** torch.randint(
        -12, 10, (batch, 64), device=dev
    )
    projected[:, 2048:2112] = rope.to(bf)  # past 448: E4M3 saturates
    q_norm = torch.rand(1536, device=dev, dtype=bf) + 0.5
    exponents = torch.randint(-16, 12, (32,), device=dev)
    exponents[-1] = 12  # row 0's last block holds its +-2s: its scale saturates
    exponents = exponents.repeat_interleave(16)
    kv_norm = ((torch.rand(512, device=dev) + 0.5) * 2.0**exponents).to(bf)
    q_b = torch.randn(HEADS * 192, 1536, device=dev, dtype=bf) / 40
    w_uk = torch.randn(HEADS, 128, 512, device=dev, dtype=bf) / 12
    w_uv = torch.randn(HEADS, 128, 512, device=dev, dtype=bf).transpose(1, 2)
    blocks = 4 if gap == FAR else 2
    mla = packed_cache(blocks, block, NVFP4_ROW, gap)
    index = packed_cache(8, UNIT, INDEX_ROW)
    slots = torch.randperm(blocks * block - 1, device=dev)[:batch]
    slots[0] = blocks * block - 1  # the last row
    if batch > 2:
        slots[1] = -1  # a padded row writes nothing
    weights = q_norm, kv_norm, q_b, w_uk, w_uv
    query_kernel, _, _ = kernels(batch, mla, index, *weights, fp8=True, nvfp4=True)
    query = torch.zeros(batch, HEADS, 576, device=dev, dtype=torch.uint8)
    q_c = torch.zeros(batch, 1536, device=dev, dtype=bf)
    q_scale = torch.full((1,), 0.5, device=dev)
    unit = torch.ones(1, device=dev)
    args = (to_cute(projected), *(to_cute(w) for w in weights[:4]), to_cute(mla))
    args += (to_cute(slots, 8), to_cute(1 / q_scale, 4), to_cute(unit, 4))
    call_query(query_kernel, args, 0.0, query, q_c, weights, projected)

    x = projected[:, :1536].float()
    ref_q_c = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)) * q_norm).to(bf)
    torch.testing.assert_close(q_c, ref_q_c, atol=1e-2, rtol=1e-2)
    q = (ref_q_c @ q_b.T).view(batch, HEADS, 192)
    latent = torch.bmm(q[..., :128].transpose(0, 1), w_uk).transpose(0, 1)
    ref_query = torch.cat([latent, q[..., 128:]], -1).float() / q_scale
    fp8 = torch.float8_e4m3fn
    torch.testing.assert_close(query.view(fp8).float(), ref_query, atol=3e-2, rtol=7e-2)

    stock = packed_cache(blocks, block, NVFP4_ROW, gap)
    kv = (projected[:, 1536:2048] * kv_norm).to(bf)  # exact: the norm is the identity
    ops.concat_and_cache_mla(
        kv, projected[:, 2048:2112], stock, slots, "nvfp4_ds_mla", unit
    )
    scales = stock.reshape(-1, NVFP4_ROW)[slots[slots >= 0]][:, 320:]
    assert (scales == 0x01).any() and (scales == 0x7E).any()  # the floor, saturation
    assert torch.equal(mla, stock)  # every byte: padded rows wrote nothing


@pytest.mark.parametrize("gap", [GAP, FAR])
@pytest.mark.parametrize("batch", [1, 5, 64])
def test_nvfp4_cache_insert_random(batch, gap):
    """nvfp4_ds_mla KV rows of random latent rows (an inexact RMSNorm): k_pe and
    the scales as the stock writer's, the E2M1 codes but where the norms round
    differently, and the dequantized rows close to the BF16 ones."""
    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        to_cute,
    )

    torch.manual_seed(batch)
    bf, dev = torch.bfloat16, "cuda"
    projected = torch.randn(batch, PITCH, device=dev, dtype=bf)
    q_norm = torch.rand(1536, device=dev, dtype=bf) + 0.5
    kv_norm = torch.rand(512, device=dev, dtype=bf) + 0.5
    q_b = torch.randn(HEADS * 192, 1536, device=dev, dtype=bf) / 40
    w_uk = torch.randn(HEADS, 128, 512, device=dev, dtype=bf) / 12
    w_uv = torch.randn(HEADS, 128, 512, device=dev, dtype=bf).transpose(1, 2)
    blocks, block = 4, 64
    mla = packed_cache(blocks, block, NVFP4_ROW, gap)
    index = packed_cache(8, UNIT, INDEX_ROW)
    slots = torch.randperm(blocks * block - 1, device=dev)[:batch]
    weights = q_norm, kv_norm, q_b, w_uk, w_uv
    query_kernel, _, _ = kernels(batch, mla, index, *weights, fp8=True, nvfp4=True)
    query = torch.zeros(batch, HEADS, 576, device=dev, dtype=torch.uint8)
    q_c = torch.zeros(batch, 1536, device=dev, dtype=bf)
    unit = torch.ones(1, device=dev)
    args = (to_cute(projected), *(to_cute(w) for w in weights[:4]), to_cute(mla))
    args += (to_cute(slots, 8), to_cute(unit, 4), to_cute(unit, 4))
    call_query(query_kernel, args, EPS, query, q_c, weights, projected)

    stock = packed_cache(blocks, block, NVFP4_ROW, gap)
    kv = rms_norm(projected[:, 1536:2048], kv_norm)
    ops.concat_and_cache_mla(
        kv, projected[:, 2048:2112], stock, slots, "nvfp4_ds_mla", unit
    )
    ours, ref = (c[slots // block, slots % block] for c in (mla, stock))
    assert torch.equal(ours[:, 256:320], ref[:, 256:320])  # k_pe
    assert (ours[:, 320:] != ref[:, 320:]).float().mean() < 0.01
    assert (ours[:, :256] != ref[:, :256]).float().mean() < 0.01
    dequant = nvfp4_rows(ours)
    error = (dequant[:, :512] - kv.float()).norm() / kv.float().norm()
    assert error < 0.12, error


@pytest.mark.parametrize("gap", [0, GAP, FAR])
@pytest.mark.parametrize("batch", [1, 3, 64, 320])
def test_indexer_inputs(batch, gap):
    """The FP8 indexer query, its folded head weights and the indexer key insert
    match the stock decode's writers (indexer_q_quant_fold_weights and
    indexer_k_norm_quant_insert: on SM100 the key scales are the MQA logits
    kernels' packed UE8M0 MX scales), into 64-entry indexer pages, contiguous or
    packed between other layers' pages."""
    import cutlass

    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer import (
        indexer_k_norm_quant_insert,
        indexer_q_quant_fold_weights,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )

    torch.manual_seed(batch)
    bf, dev = torch.bfloat16, "cuda"
    projected = torch.randn(batch, PITCH, device=dev, dtype=bf)
    iq = torch.randn(batch, 32 * 128, device=dev, dtype=bf)
    iq[0, :128] = 0  # an all-zero head
    norm_w = torch.rand(128, device=dev) + 0.5
    norm_b = torch.randn(128, device=dev) / 10
    mla, index = packed_cache(8, UNIT, MLA_ROW), packed_cache(8, UNIT, INDEX_ROW, gap)
    slots = torch.randperm(8 * UNIT - 1, device=dev)[:batch]
    slots[0] = 8 * UNIT - 1  # the last entry
    if batch > 2:
        slots[1] = -1  # a padded row or, under DCP, a token another rank owns
    weights = (
        torch.ones(1536, device=dev, dtype=bf),
        torch.ones(512, device=dev, dtype=bf),
        torch.zeros(HEADS * 192, 1536, device=dev, dtype=bf),
        torch.zeros(HEADS, 128, 512, device=dev, dtype=bf),
        torch.zeros(HEADS, 128, 512, device=dev, dtype=bf).transpose(1, 2),
    )
    _, index_kernel, _ = kernels(batch, mla, index, *weights)
    q_fp8 = torch.zeros(batch, 32, 128, device=dev, dtype=torch.uint8)
    head_w = torch.zeros(batch, 32, device=dev, dtype=bf)
    args = (to_cute(projected), to_cute(iq), to_cute(norm_w, 4), to_cute(norm_b, 4))
    args += (to_cute(index), to_cute(slots, 8), to_cute(q_fp8), to_cute(head_w, 4))
    index_kernel(*args, cutlass.Float32(EPS), stream(dev))

    key = 2112 + GATE
    ref_q, ref_w = indexer_q_quant_fold_weights(
        iq.view(batch, 32, 128),
        projected[:, key + 128 : key + 160],
        128**-0.5 * 32**-0.5,
        32,
    )
    assert torch.equal(q_fp8, ref_q.view(torch.uint8))
    # The stock decode casts the FP32 weights to BF16 for SM100's MQA logits.
    torch.testing.assert_close(head_w, ref_w.to(bf), atol=0, rtol=0)

    stock = packed_cache(8, UNIT, INDEX_ROW, gap)
    topk = torch.empty(batch, 2048, device=dev, dtype=torch.int32)
    k = projected[:, key : key + 128]
    indexer_k_norm_quant_insert(k, norm_w, norm_b, EPS, slots, stock, topk, batch)
    live = slots[slots >= 0]
    page, entry = live // UNIT, live % UNIT
    ours, ref = index_blocks(index)[page], index_blocks(stock)[page]
    at = UNIT * 128 + 4 * entry.view(-1, 1) + torch.arange(4, device=dev)
    assert torch.equal(ours.gather(1, at), ref.gather(1, at))  # the key scales
    assert (index_blocks(index) != index_blocks(stock)).float().mean() < 1e-3


@pytest.mark.parametrize("fp8", [False, True], ids=["ds", "fp8"])
@pytest.mark.parametrize("batch", [1, 2, 3, 4, 5, 8, 16, 40, 60, 64])
def test_query_overlap_kernels(batch, fp8):
    """Query overlap: QueryNorm's q_c rows are MlaQuery's, bit for bit, on both
    MlaQuery forms (one CTA per cluster part below 4 rows, tiled from 4), and the
    forked MlaQuery writes the same query rows and KV row as the serial one and no
    q_c rows."""
    import cutlass

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        fake,
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.query_stage import (  # noqa: E501
        compile_norm,
        compile_query,
        runs_split,
    )

    if runs_split(HEADS, batch, 0, fp8):
        pytest.skip("the split query serves this batch (test_specialized_decode_dep)")
    torch.manual_seed(batch)
    bf, dev = torch.bfloat16, "cuda"
    projected = torch.randn(batch, PITCH, device=dev, dtype=bf) * 3
    q_norm = torch.rand(1536, device=dev, dtype=bf) + 0.5
    kv_norm = torch.rand(512, device=dev, dtype=bf) + 0.5
    q_b = torch.randn(HEADS * 192, 1536, device=dev, dtype=bf) / 40
    w_uk = torch.randn(HEADS, 128, 512, device=dev, dtype=bf) / 12
    row = FP8_ROW if fp8 else MLA_ROW
    blocks = 2
    slots = torch.randperm(blocks * UNIT, device=dev)[:batch]
    if batch > 2:
        slots[1] = -1
    weights = q_norm, kv_norm, q_b, w_uk
    unit = torch.full((1,), 0.5, device=dev)
    dev_index = torch.accelerator.current_device_index()

    def query_rows(mode):
        mla = packed_cache(blocks, UNIT, row)
        specs = tuple(layout(t) for t in (*weights, mla))
        fakes = tuple(
            fake(shape, stride, dtype, 4 if dtype == torch.float32 else 16)
            for shape, stride, dtype in specs
        )
        kernel = compile_query(
            batch, HEADS, PITCH, fakes, dev_index, fp8=fp8, overlap=mode
        )
        shape, dtype = (HEADS, torch.uint8) if fp8 else (64, bf)
        query = torch.zeros(batch, shape, 576, device=dev, dtype=dtype)
        q_c = torch.full((batch, 1536), -7.0, device=dev, dtype=bf)
        args = (to_cute(projected), *(to_cute(w) for w in weights), to_cute(mla))
        args += (to_cute(slots, 8), to_cute(unit, 4), to_cute(unit, 4))
        args += (cutlass.Float32(EPS), to_cute(query), to_cute(q_c), stream(dev))
        kernel(*args)
        return query, q_c, mla

    serial, forked = query_rows("serial"), query_rows("fork")
    assert torch.equal(serial[0], forked[0])  # the query rows
    assert torch.equal(serial[2], forked[2])  # the KV row inserted
    assert (forked[1] == -7.0).all()  # no q_c rows

    norm = compile_norm(batch, HEADS, PITCH)
    q_c = torch.zeros(batch, 1536, device=dev, dtype=bf)
    args = (to_cute(projected), to_cute(q_norm), to_cute(q_c))
    norm(*args, cutlass.Float32(EPS), stream(dev))
    assert torch.equal(q_c.view(torch.int16), serial[1].view(torch.int16))


@pytest.mark.parametrize("fp8", [False, True], ids=["ds", "fp8"])
@pytest.mark.parametrize("batch", [1, 60])
def test_query_wait_polls_mailbox(batch, fp8):
    """QueryWait (the DCP q mailbox poll the query overlap moves off MlaQuery)
    returns only once every published chunk of the gathered heads (DCP8 x 12 in
    rows of 128) of the batch's tokens is not the sentinel, and reads nothing
    past them."""
    import time

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.query_stage import (  # noqa: E501
        compile_wait,
    )

    dev, q_heads, gathered = "cuda", 128, 8 * HEADS
    width = 576 if fp8 else 1152
    wait = compile_wait(batch, HEADS, 8, q_heads, fp8)
    mailbox = torch.full((64, q_heads, width), 0xFF, device=dev, dtype=torch.uint8)
    mailbox[:batch, :gathered] = 0x11
    mailbox[batch - 1, gathered - 1, -16:] = 0xFF  # the last published chunk
    polling, filling = torch.cuda.Stream(), torch.cuda.Stream()
    done = torch.cuda.Event()
    torch.accelerator.synchronize()
    with torch.cuda.stream(polling):
        wait(to_cute(mailbox), stream(dev))
        done.record()
    time.sleep(0.2)
    pending = not done.query()
    with torch.cuda.stream(filling):
        mailbox[batch - 1, gathered - 1, -16:] = 0x22
    deadline = time.monotonic() + 10
    while not done.query() and time.monotonic() < deadline:
        time.sleep(0.01)
    finished = done.query()
    torch.accelerator.synchronize()
    assert pending and finished, (pending, finished)


# 129-144: three 48-row tiles (no empty fourth tile).
FRONT_BATCHES = [1, 2, 3, 5, 8, 16, 17, 40, 60, 64, 65, 80, 100, 128, 129, 130, 144]
FRONT_BATCHES += [145, 160, 240, 320]


@pytest.mark.parametrize(
    "kernel_kind",
    ["mma", "mma-qc", "cluster"],
)
@pytest.mark.parametrize("batch", FRONT_BATCHES)
def test_index_front(batch, kernel_kind):
    """The fused index front (q_a RMSNorm, wq_b and DsaIndex in one kernel:
    IndexFrontMma, also from the normalized q_c rows ("mma-qc"), or the cluster
    IndexFront up to 64 rows) against the stock writers on an FP32 projection of the
    normalized q_c: the FP8 indexer query within one code step where the BF16
    projection rounds differently, the folded head weights exact up to such
    roundings, and the indexer key bytes and scales exactly as DsaIndex inserts
    them."""
    import cutlass

    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer import (
        indexer_k_norm_quant_insert,
        indexer_q_quant_fold_weights,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import (
        query_stage,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (  # noqa: E501
        weight_map,
    )

    if kernel_kind == "cluster" and batch > 64:
        pytest.skip("IndexFront serves at most 64 rows")
    torch.manual_seed(batch)
    bf, dev = torch.bfloat16, "cuda"
    projected = torch.randn(batch, PITCH, device=dev, dtype=bf) * 2
    q_norm = torch.rand(1536, device=dev, dtype=bf) + 0.5
    wq = torch.randn(32 * 128, 1536, device=dev, dtype=bf) / 30
    norm_w = torch.rand(128, device=dev) + 0.5
    norm_b = torch.randn(128, device=dev) / 10
    index = packed_cache(8, UNIT, INDEX_ROW, GAP)
    slots = torch.randperm(8 * UNIT - 1, device=dev)[:batch]
    if batch > 2:
        slots[1] = -1
    dev_index = torch.accelerator.current_device_index()
    normalized = kernel_kind == "mma-qc"
    saved = query_stage.FRONT_KERNEL
    query_stage.FRONT_KERNEL = "cluster" if kernel_kind == "cluster" else "mma"
    try:
        specs = layout(wq), layout(index)
        args = batch, HEADS, PITCH, *specs, dev_index, normalized
        kernel = query_stage.compile_front(*args)
    finally:
        query_stage.FRONT_KERNEL = saved
    q_fp8 = torch.zeros(batch, 32, 128, device=dev, dtype=torch.uint8)
    head_w = torch.zeros(batch, 32, device=dev, dtype=bf)
    q_c = rms_norm(projected[:, :1536], q_norm)
    args = (to_cute(projected),)
    weights = to_cute(wq)
    if kernel_kind != "cluster":  # the rows, and the weights' TMA map
        args += (to_cute(q_c) if normalized else to_cute(projected),)
        wq_map = weight_map(wq, 128)
        weights = cutlass.Int64(wq_map.data_ptr())
    args += (to_cute(q_norm), weights)
    args += (to_cute(norm_w, 4), to_cute(norm_b, 4), to_cute(index))
    args += (to_cute(slots, 8), to_cute(q_fp8), to_cute(head_w, 4))
    kernel(*args, cutlass.Float32(EPS), cutlass.Float32(EPS), stream(dev))

    key = 2112 + GATE
    iq = (q_c.float() @ wq.float().T).to(bf)
    ref_q, ref_w = indexer_q_quant_fold_weights(
        iq.view(batch, 32, 128),
        projected[:, key + 128 : key + 160],
        128**-0.5 * 32**-0.5,
        32,
    )
    ref_q = ref_q.view(torch.uint8)
    # Codes are compared as dequantized values (code x folded head weight, which
    # carries the head scale): within one code step of the reference value, or
    # tiny next to the head's largest. A head whose BF16 amax rounds across a power
    # of two (448 * 2^k) takes the other scale (codes move by a binade, the weight
    # by 2x), and near-zero values move by a few codes where the q_a RMSNorm rounds
    # a q_c element differently from the reference's (rare: hundreds of rows).
    ref_w = ref_w.to(bf)
    flipped = (head_w.float() == 2 * ref_w.float()) | (
        2 * head_w.float() == ref_w.float()
    )
    assert (flipped & (ref_w != 0)).float().mean() < 0.002
    f8 = torch.float8_e4m3fn
    ours = q_fp8.view(f8).float() * head_w.float().unsqueeze(-1)
    theirs = ref_q.view(f8).float() * ref_w.float().unsqueeze(-1)
    top = theirs.abs().amax(-1, keepdim=True)
    error = (ours - theirs).abs()
    assert bool((error <= theirs.abs() / 8 + top / 2**9).all())
    if normalized or batch <= 64:
        # From q_c (the reference's input) at every batch, and from the packed rows
        # up to 64 rows: every code within one step; a flipped head's codes (the
        # other power-of-two scale) within one step of the coarser scale, as
        # dequantized values (E4M3: 3 mantissa bits, subnormal step 2^-9).
        same = ~flipped.unsqueeze(-1).expand_as(q_fp8)
        assert fp8_codes_adjacent(q_fp8[same], ref_q[same])
        coarse = torch.maximum(head_w.float(), ref_w.float()).abs().unsqueeze(-1)
        bound = torch.maximum(ours.abs(), theirs.abs()) / 8 + coarse * 2**-9
        assert bool((error <= bound)[~same].all())
    # Above 64 rows the packed-row kernel's in-kernel RMSNorm (QueryNorm's order)
    # rounds some q_c elements differently from torch's: near-zero values may move
    # a few codes (bounded above by the head's largest value / 512).
    assert (q_fp8 != ref_q).float().mean() < (0.002 if normalized else 0.01)
    assert (head_w != ref_w).float().mean() < 0.02

    stock = packed_cache(8, UNIT, INDEX_ROW, GAP)
    topk = torch.empty(batch, 2048, device=dev, dtype=torch.int32)
    k = projected[:, key : key + 128]
    indexer_k_norm_quant_insert(k, norm_w, norm_b, EPS, slots, stock, topk, batch)
    live = slots[slots >= 0]
    page, entry = live // UNIT, live % UNIT
    ours, ref = index_blocks(index)[page], index_blocks(stock)[page]
    at = UNIT * 128 + 4 * entry.view(-1, 1) + torch.arange(4, device=dev)
    assert torch.equal(ours.gather(1, at), ref.gather(1, at))  # the key scales
    assert (index_blocks(index) != index_blocks(stock)).float().mean() < 1e-3


def order16(bits):
    """The order-preserving 16-bit key of BF16 bits (int32 tensor)."""
    bits = bits & 0xFFFF
    return torch.where(bits & 0x8000 != 0, bits ^ 0xFFFF, bits ^ 0x8000)


FUSED_INDEXER_CASES = {
    # (contexts, q): every row within the top-k (the bypass) or not, ragged
    # batches, rows past 16K vectors, past the shared vector maxima (TOPS: 128K keys),
    # and tie-heavy logits (integer queries and keys).
    "short": ([1500, 300], 5),
    "boundary": ([2046, 2047, 2048, 2049], 1),
    "mixed": ([5000, 300, 9000, 33000], 5),
    "uniform": ([20000] * 12, 5),
    "long": ([100000, 70000, 3000, 130000], 5),
    "q1_64": ([3000 + 977 * r for r in range(64)], 1),
    "ties": ([50000, 30000, 4000], 5),
    "far": ([600000], 1),
    # The step's last request is CUDA-graph padding: no metadata rows.
    "padded": ([40000, 3000, 0], 5),
    # Past 64 rows (the attention cores serve up to 320): R48 / R64 at q5.
    "wide240": ([9000 + 61 * r for r in range(48)], 5),
    "wide320": ([3000 + 257 * r for r in range(64)], 5),
}


def fused_indexer_inputs(contexts, q, gap, ties, seed):
    """A packed indexer cache of random FP8 keys and UE8M0 scales under a random
    block table, the per-row ends (row j of a request sees ``context + 1 + j``
    keys), the FP8 query and BF16 head weights."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    pages = [-(-(c + q) // UNIT) for c in contexts]
    blocks = sum(pages) + 3
    index = packed_cache(blocks, UNIT, INDEX_ROW, gap)
    raw = torch.randn(blocks, UNIT, 128, device="cuda", generator=gen) * 0.5
    keys = (raw.round() if ties else raw).to(torch.float8_e4m3fn)
    exps = torch.randint(120, 130, (blocks, UNIT), device="cuda", generator=gen)
    page = index_blocks(index)
    page[:, : UNIT * 128] = keys.view(torch.uint8).reshape(blocks, -1)
    scale_words = (exps.int() * 0x01010101).view(torch.uint8)
    page[:, UNIT * 128 :] = scale_words.reshape(blocks, -1)
    perm = torch.randperm(blocks, device="cuda", generator=gen).int()
    table = torch.zeros(len(contexts), max(pages), dtype=torch.int32, device="cuda")
    used = 0
    for r, n in enumerate(pages):
        table[r, :n] = perm[used : used + n]
        used += n
    ends = [c + 1 + j for c in contexts for j in range(q)]
    ends = torch.tensor(ends, dtype=torch.int32, device="cuda")
    rows = max(64, len(contexts) * q)
    query = torch.randn(rows, 32, 128, device="cuda", generator=gen)
    query = (query.round() if ties else query * 0.5).to(torch.float8_e4m3fn)
    weights = (torch.randn(rows, 32, device="cuda", generator=gen) * 0.1).bfloat16()
    return SimpleNamespace(
        index=index,
        keys=keys,
        exps=exps,
        table=table,
        ends=ends,
        query=query,
        weights=weights,
        gen=gen,
    )


def deepgemm_head_sum(score, weights):
    """Stock's (DeepGEMM SM100 paged MQA logits) head sum of ``score`` [H, L] =
    relu(q . k) in FP32 and the BF16 ``weights`` [H]: each head's value rounded
    to BF16, four BF16 accumulators (heads h = j mod 4) by BF16 FMAs, then BF16
    adds (0 + 2, 1 + 3, then the two). Returns FP32 [L] (BF16 values)."""
    bf = torch.bfloat16
    v, w = score.to(bf).double(), weights.to(bf).double()
    acc = [torch.zeros_like(v[0]) for _ in range(4)]
    for h in range(v.shape[0]):
        acc[h % 4] = (acc[h % 4] + v[h] * w[h]).float().to(bf).double()
    lo = (acc[0] + acc[2]).float().to(bf).double()
    hi = (acc[1] + acc[3]).float().to(bf).double()
    return (lo + hi).float().to(bf).float()


def check_fused_logits(got, x, t, r, length):
    """Row ``t`` (request ``r``) of the fused scorer's BF16 logits (int16 bits),
    against stock's precision (deepgemm_head_sum) over the dequantized keys: equal
    up to the FP32 q . k summation order (a BF16 rounding of a head's value or of
    an accumulator may flip)."""
    if length == 0:  # a padded row: nothing scored
        return
    pos = torch.arange(length, device="cuda")
    blk = x.table[r][pos // UNIT].long()
    k = x.keys[blk, pos % UNIT].float()
    scale = torch.exp2(x.exps[blk, pos % UNIT].float() - 127)
    score = torch.relu(x.query[t].float() @ k.T)
    ref = deepgemm_head_sum(score, x.weights[t]) * scale
    got_f = ((got.int() & 0xFFFF) << 16).view(torch.float32)
    # A few BF16 steps of the largest |term| where roundings flip; mostly equal.
    terms = (score.to(torch.bfloat16).float() * x.weights[t].float()[:, None]).abs()
    bound = terms.amax(0) * scale * 2**-6 + 1e-30
    assert bool(((got_f - ref).abs() <= bound).all()), t
    assert (got_f == ref).float().mean() > 0.95, t


def exact_topk(bits, k):
    """Key indices of the top ``k`` of BF16 ``bits`` (ties to the lowest index),
    in key order."""
    pos = torch.arange(bits.shape[0], device=bits.device)
    key = order16(bits.int()).long() << 32 | (2**32 - 1 - pos)
    return torch.topk(key, min(k, bits.shape[0])).indices.sort().values


@pytest.mark.parametrize("mla_page", [UNIT, 2048])
@pytest.mark.parametrize("gap", [GAP, FAR])
@pytest.mark.parametrize("case", list(FUSED_INDEXER_CASES))
def test_fused_indexer_selects_exact_topk(case, gap, mla_page):
    """The fused decode indexer (TP): its BF16 logits at stock's precision
    (deepgemm_head_sum) over the dequantized keys, up to BF16 rounding flips of the
    FP32 q . k order, and its selection exactly the
    top-2048 of its own logits (ties to the lowest key index), as the MLA cache's
    physical rows in key order, -1 past the count, with the counts. ``mla_page``:
    the MLA cache's tokens per block (2048: the hybrid model's attention pages,
    with 64-token indexer pages)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.indexer import (
        FusedIndexer,
    )

    contexts, q = FUSED_INDEXER_CASES[case]
    x = fused_indexer_inputs(contexts, q, gap, case == "ties", len(contexts) * 131 + q)
    rows, topk, block_rows = len(contexts) * q, 2048, mla_page + GAP
    ends, table = x.ends, x.table
    mla_pages = max(-(-(c + q) // mla_page) for c in contexts)
    mla_table = torch.randint(
        1 << 14, (len(contexts), mla_pages), device="cuda", dtype=torch.int32
    )
    if case == "padded":
        ends[-q:] = 0
    real = rows - q if case == "padded" else rows
    decode = SimpleNamespace(
        seq_lens=ends[:real].view(real, 1),
        block_table=table.repeat_interleave(q, 0)[:real],
    )
    indexer = SimpleNamespace(
        k_cache=SimpleNamespace(kv_cache=x.index),
        topk_tokens=topk,
        max_model_len=1 << 20,
    )
    cap = max(64, rows)
    fused = FusedIndexer(indexer, [(rows, q)], torch.device("cuda"), cap, page=mla_page)
    ids = torch.full((cap, topk), -7, dtype=torch.int32, device="cuda")
    fused.write_ids(ids)
    query = x.query.view(torch.uint8)
    args = query, x.weights, decode, mla_table, block_rows, rows, q
    out, valid = fused.run(*args)
    torch.accelerator.synchronize()

    tiles = [-(-c // 128) if c > topk else 0 for c in ends[q - 1 :: q].tolist()]
    for t in range(rows):
        r, j, length = t // q, t % q, int(ends[t])
        pos = torch.arange(length, device="cuda")
        if tiles[r]:
            # This row's logits in the request-compact scratch, against stock's
            # precision (deepgemm_head_sum).
            base = (sum(tiles[:r]) * q + j * tiles[r]) * 128
            got = fused.logits[base : base + length]
            check_fused_logits(got, x, t, r, length)
            if length > topk:
                pos = exact_topk(got, topk)
        n = min(length, topk)
        page = mla_table[r][pos // mla_page].long()
        physical = page * block_rows + pos % mla_page
        assert int(valid[t]) == n
        assert torch.equal(out[t, :n].long(), physical), (case, t)
        assert torch.equal(ids[t, :n].long(), pos), (case, t)
        assert bool((out[t, n:] == -1).all()) and bool((ids[t, n:] == -1).all())


DCP_INDEXER_CASES = {
    # (local contexts of one rank, q) around N = 384 local candidates: every row
    # within N (the bypass) or not, the dense and the vector-maxima paths (6K
    # keys), local indices past 2^16 (the B' bit plane), tie-heavy logits,
    # 64 rows, and a CUDA-graph padding request.
    "short": ([100, 300], 5),
    "boundary": ([382, 383, 384, 385], 1),
    "mixed": ([700, 40, 1100, 4100], 5),
    "uniform": ([4000] * 12, 5),
    "long": ([16400, 9000, 400, 6200], 5),
    "far": ([131000, 70000], 1),
    "ties": ([6000, 4000, 500], 5),
    "q1_64": ([500 + 97 * r for r in range(64)], 1),
    "padded": ([5000, 400, 0], 5),
    # Past 64 rows: 145 (the packed select's last width, one CTA per SM) and
    # 320 rows (the scorer only: the merge's DeepSelect takes the local top-N).
    "wide145": ([1100 + 37 * r for r in range(29)], 5),
    "wide320": ([2000 + 23 * r for r in range(64)], 5),
}


@pytest.mark.parametrize("mode", ["packed", "logits"])
@pytest.mark.parametrize("case", list(DCP_INDEXER_CASES))
def test_fused_indexer_dcp(case, mode):
    """The fused indexer under DCP (one rank's local keys, N = 384):

    - logits: its scorer writes DeepGEMM's paged MQA logits layout (1 KiB rows,
      key index = column), at stock's precision (deepgemm_head_sum) up to BF16
      rounding flips, which the #63 merge's
      own local top-N (DeepSelect) and pack take as they take DeepGEMM's;
    - packed: its select's B' rows are bit-exact the #63 pack kernel's of the
      same candidates, which are the exact local top-N of the logits (ties to the
      lowest key index), zero words past the count, and the counts."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.indexer import (
        FusedIndexer,
    )
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        dcp_topk_pack,
        dcp_topk_pack_indices,
    )

    contexts, q = DCP_INDEXER_CASES[case]
    x = fused_indexer_inputs(contexts, q, GAP, case == "ties", len(contexts) * 7 + q)
    rows, n = len(contexts) * q, 384
    ends, table = x.ends, x.table
    if case == "padded":
        ends[-q:] = 0
    real = rows - q if case == "padded" else rows
    decode = SimpleNamespace(
        seq_lens=ends[:real].view(real, 1),
        block_table=table.repeat_interleave(q, 0)[:real],
    )
    indexer = SimpleNamespace(
        k_cache=SimpleNamespace(kv_cache=x.index),
        topk_tokens=2048,
        max_model_len=1 << 20,
        dcp_world_size=8,
    )
    fused = FusedIndexer(
        indexer,
        [(rows, q)],
        torch.device("cuda"),
        max(64, rows),
        local_topk=n,
        packed=mode == "packed",
    )
    assert fused.pitch == 131072
    query = x.query.view(torch.uint8)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    # The packed select runs one CTA per row on its own SM: exactly up to the SM
    # count (past it, the scorer only: the "logits" case covers that).
    assert fused.packs(rows) == (mode == "packed" and rows <= sms), (rows, sms)
    if mode == "packed" and not fused.packs(rows):
        pytest.skip(f"{rows} rows exceed the {sms} SMs: no packed select")
    packed, out_logits, aux = fused.run_dcp(query, x.weights, decode, rows, q)
    assert (packed is not None) == (mode == "packed")
    # The exact merge takes the logits and the row ends in both modes.
    assert torch.equal(aux, ends[:rows])
    assert out_logits.stride(0) * 2 % 1024 == 0
    torch.accelerator.synchronize()
    logits = fused.logits[: rows * fused.pitch].view(rows, fused.pitch)
    assert out_logits.data_ptr() == logits.data_ptr()
    for t in range(rows):
        length = int(ends[t])
        check_fused_logits(logits[t, :length], x, t, t // q, length)

    bf16 = logits.view(torch.bfloat16)
    if mode == "logits":
        out = out_logits
        packed = dcp_topk_pack(out, aux, n, max_local_len=fused.max_local_len)
        torch.accelerator.synchronize()
        words = packed[:, :n]
        for t in range(rows):
            # DeepSelect may break ties otherwise: compare the selected scores.
            length = int(ends[t])
            want = order16(logits[t, exact_topk(logits[t, :length], n)].long())
            got = (words[t][words[t] != 0].long() & 0xFFFFFFFF) >> 16
            assert torch.equal(got.sort().values, want.sort().values), (case, t)
        return

    out, counts = packed, fused.valid[:rows]
    local = torch.full((rows, n), -1, dtype=torch.int32, device="cuda")
    for t in range(rows):
        length = int(ends[t])
        c = min(length, n)
        assert int(counts[t]) == c, (case, t)
        words = out[t, :n].long() & 0xFFFFFFFF
        plane = out[t, n:].long() & 0xFFFFFFFF
        assert bool((words[c:] == 0).all()) and bool((words[:c] != 0).all())
        j = torch.arange(c, device="cuda")
        high = (plane[j // 32] >> (j % 32)) & 1
        ids = (0xFFFF - (words[:c] & 0xFFFF)) | (high << 16)
        assert torch.equal(ids.sort().values, exact_topk(logits[t, :length], n))
        local[t, :c] = ids.int()
    # The same candidates (in our slot order) through #63's pack kernel.
    ref = torch.full_like(out, 7)
    dcp_topk_pack_indices(bf16, local, ref)
    torch.accelerator.synchronize()
    assert torch.equal(out, ref), case


# The first indexer page whose scale row lies 2^31 int32 words (8 GiB) past the
# cache view's base at the real page stride (64 keys x 132 bytes): a 32-bit page x
# stride offset wraps from there (the K3.1 DCP8 gsm8k IMA: valid page ids around
# 1.9M of the hybrid KV pool's index view).
HIGH_PAGE = -(-(1 << 31) // (UNIT * INDEX_ROW // 4))


def high_page_indexer_inputs(context, q, seed):
    """fused_indexer_inputs for one request whose pages lie at and past HIGH_PAGE
    of a real-stride (unpadded) indexer cache view over > 8 GiB, plus a few low
    pages, in random order. Returns (inputs as the reference reads them: compact
    page ids, the kernel's table of the real page ids)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    n = -(-(context + q) // UNIT)
    low = torch.randperm(64, device="cuda", generator=gen)[:2].int()
    high = HIGH_PAGE + torch.randperm(4 * n, device="cuda", generator=gen)[: n - 2]
    ids = torch.cat((low, high.int()))[torch.randperm(n, device="cuda", generator=gen)]
    blocks = int(ids.max()) + 2
    raw = torch.empty(blocks * UNIT * INDEX_ROW, dtype=torch.uint8, device="cuda")
    shape, stride = (blocks, UNIT, INDEX_ROW), (UNIT * INDEX_ROW, INDEX_ROW, 1)
    index = torch.as_strided(raw, shape, stride)
    keys = (torch.randn(n, UNIT, 128, device="cuda", generator=gen) * 0.5).to(
        torch.float8_e4m3fn
    )
    exps = torch.randint(120, 130, (n, UNIT), device="cuda", generator=gen)
    pages = index_blocks(index)
    pages[ids.long(), : UNIT * 128] = keys.view(torch.uint8).reshape(n, -1)
    scales = (exps.int() * 0x01010101).view(torch.uint8).reshape(n, -1)
    pages[ids.long(), UNIT * 128 :] = scales
    ends = torch.tensor(
        [context + 1 + j for j in range(q)], dtype=torch.int32, device="cuda"
    )
    query = torch.randn(64, 32, 128, device="cuda", generator=gen) * 0.5
    weights = (torch.randn(64, 32, device="cuda", generator=gen) * 0.1).bfloat16()
    reference = SimpleNamespace(
        keys=keys,
        exps=exps,
        table=torch.arange(n, device="cuda", dtype=torch.int32)[None],
        query=query.to(torch.float8_e4m3fn),
        weights=weights,
    )
    return reference, index, ids[None], ends


@pytest.mark.parametrize("mode", ["tp", "dcp_packed", "dcp_logits"])
def test_fused_indexer_high_pages(mode):
    """The fused indexer's scorer over pages past 2^31 int32 words of its cache
    view (a request whose KV blocks sit high in the hybrid KV pool, as in serving
    once prefix caching walks the pool; the tests above use low pages only): its
    logits match stock's precision over the same pages and the selection is
    exact. Before the 64-bit scale-row offsets, the scale bulk copy of such a page
    read ~8 GiB below the view (illegal address)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.indexer import (
        FusedIndexer,
    )

    dcp = mode != "tp"
    q, context = 5, 1100 if dcp else 3000
    x, index, table, ends = high_page_indexer_inputs(context, q, 11)
    assert int(table.max()) >= HIGH_PAGE
    decode = SimpleNamespace(
        seq_lens=ends.view(q, 1), block_table=table.repeat_interleave(q, 0)
    )
    indexer = SimpleNamespace(
        k_cache=SimpleNamespace(kv_cache=index),
        topk_tokens=2048,
        # DCP: a rank's keys of a context (interleave 1) size the logits rows; TP:
        # small, as the request-compact logits scale with it.
        max_model_len=1 << 20 if dcp else 1 << 13,
        dcp_world_size=8,
    )
    query = x.query.view(torch.uint8)
    device = torch.device("cuda")
    if not dcp:
        fused = FusedIndexer(indexer, [(q, q)], device, 64, page=2048)
        mla_table = torch.randint(1 << 14, (1, 2), device="cuda", dtype=torch.int32)
        out, valid = fused.run(query, x.weights, decode, mla_table, 2048 + GAP, q, q)
        torch.accelerator.synchronize()
        tiles = -(-context // 128)
        for t in range(q):
            length = int(ends[t])
            got = fused.logits[t * tiles * 128 :][:length]
            check_fused_logits(got, x, t, 0, length)
            pos = exact_topk(got, 2048)
            physical = mla_table[0][pos // 2048].long() * (2048 + GAP) + pos % 2048
            assert int(valid[t]) == 2048
            assert torch.equal(out[t].long(), physical), t
        return
    n = 384
    packs = mode == "dcp_packed"
    fused = FusedIndexer(indexer, [(q, q)], device, 64, local_topk=n, packed=packs)
    packed, _, _ = fused.run_dcp(query, x.weights, decode, q, q)
    torch.accelerator.synchronize()
    logits = fused.logits[: q * fused.pitch].view(q, fused.pitch)
    for t in range(q):
        length = int(ends[t])
        check_fused_logits(logits[t, :length], x, t, 0, length)
        if packs:
            c = min(length, n)
            assert int(fused.valid[t]) == c
            words = packed[t, :n].long() & 0xFFFFFFFF
            plane = packed[t, n:].long() & 0xFFFFFFFF
            j = torch.arange(c, device="cuda")
            high = (plane[j // 32] >> (j % 32)) & 1
            ids = (0xFFFF - (words[:c] & 0xFFFF)) | (high << 16)
            assert torch.equal(ids.sort().values, exact_topk(logits[t, :length], n))


@pytest.mark.parametrize("reqs,q", [(4, 1), (8, 5), (12, 5)])
@pytest.mark.parametrize("kernel", ["trtllm", "prims_ts", "nvfp4"])
def test_flashinfer_dcp_core_merges_to_full_attention(
    default_vllm_config, dist_init, kernel, reqs, q
):
    """DCP8 on FlashInfer sparse (per-tensor FP8): each rank's dcp_core over its
    shard of a packed cache, its gathered 96-head query padded to 128 and the
    global top-k, merged by LSE in the kernel's base, is the sparse attention
    over the whole context. ``q`` rows per request (DSpark verify), each row with
    its own top-k; request 0 lies on rank 0 only (empty shards: an LSE DcpValue
    drops, with no masking kernel). ``nvfp4``: prims_ts over an nvfp4_ds_mla
    cache."""
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.attn_fi import (
        FlashInferCore,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.prepared import (
        PreparedDsa,
    )
    from vllm.utils.flashinfer import has_flashinfer_prims_ts_sparse_mla
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseImpl,
    )

    if kernel != "trtllm" and not has_flashinfer_prims_ts_sparse_mla():
        pytest.skip("FlashInfer has no prims_ts sparse MLA")
    nvfp4 = kernel == "nvfp4"
    kernel = "prims_ts" if nvfp4 else kernel
    default_vllm_config.kernel_config.flashinfer_sparse_mla_kernel = kernel
    text = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
    object.__setattr__(
        default_vllm_config, "model_config", SimpleNamespace(hf_text_config=text)
    )
    dcp, heads, topk, block, context = 8, HEADS * 8, 2048, UNIT, 5000
    rows_n = reqs * q
    impl = FlashInferMLASparseImpl(
        num_heads=heads,
        head_size=FP8_ROW,
        scale=192**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="nvfp4_ds_mla" if nvfp4 else "fp8_e4m3",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
        kv_b_proj=ColumnParallelLinear(512, heads * 256, bias=False),
    )
    impl.need_to_return_lse_for_decode = True
    impl._workspace_buffer = torch.zeros(1 << 30, dtype=torch.int8, device="cuda")
    layer = SimpleNamespace(_q_scale_float=1.0, _k_scale_float=1.0)
    e4m3 = torch.float8_e4m3fn

    gen = torch.Generator(device="cuda").manual_seed(0)
    # Each rank's latents are offset differently and the scores are peaked, so the
    # shards' partial outputs and LSEs differ: a wrong LSE base misweights them.
    owner = torch.arange(context, device="cuda") % dcp
    if nvfp4:  # the shards' block scales differ
        kv = random_nvfp4(reqs, context, gen=gen)
        kv[..., 320:] = 0x18 + 2 * owner.view(1, -1, 1).to(torch.uint8)
        row_bytes = NVFP4_ROW
    else:
        kv = torch.rand(reqs, context, FP8_ROW, generator=gen, device="cuda") - 0.5
        kv[..., :512] += 0.2 * owner.view(1, -1, 1)
        kv = kv.to(e4m3)
        row_bytes = FP8_ROW
    query = torch.zeros(rows_n, 128, FP8_ROW, dtype=torch.uint8, device="cuda")
    rand = torch.rand(rows_n, heads, FP8_ROW, generator=gen, device="cuda")
    query[:, :heads] = (8 * (rand - 0.5)).to(e4m3).view(torch.uint8)
    tokens = torch.full((rows_n, topk), -1, dtype=torch.int32, device="cuda")
    for t in range(rows_n):
        r = t // q
        # Row t of a request sees its context up to its own position.
        end, step = context - (q - 1 - t % q), dcp if r == 0 else 1
        pool = torch.arange(0, end, step, device="cuda")
        pick = pool[torch.randperm(pool.numel(), generator=gen, device="cuda")]
        n = min(topk - 37 * (t % 9), pick.numel())
        tokens[t, :n] = pick[:n].int()

    local = -(-context // dcp)
    blocks_per_req = -(-local // block)
    rows, lses = [], []
    for rank in range(dcp):
        # FlashInfer aligns packed block strides to its 32-row pages.
        cache = packed_cache(reqs * blocks_per_req, block, row_bytes, gap=block)
        order = torch.randperm(reqs * blocks_per_req, generator=gen, device="cuda")
        table = order.view(reqs, blocks_per_req).int()
        for r in range(reqs):
            mine = torch.arange(rank, context, dcp, device="cuda")
            slot = torch.arange(mine.numel(), device="cuda")
            physical = table[r, slot // block].long()
            cache[physical, slot % block] = kv[r, mine].view(torch.uint8)
        req_ids = torch.arange(rows_n, dtype=torch.int32, device="cuda") // q
        md = SimpleNamespace(
            req_id_per_token=req_ids, block_table=table, block_size=block
        )
        box = torch.zeros(2, rows_n, 128, FP8_ROW, dtype=torch.uint8, device="cuda")
        box[0] = query
        buffers = fi_buffers(rows_n, 128)
        fi = FlashInferCore(impl, layer, 128, buffers, {rows_n: kernel}, dcp, rank)
        core = SimpleNamespace(
            fi=fi,
            indexer=SimpleNamespace(cp_kv_cache_interleave_size=1),
            parity=0,
            box=SimpleNamespace(q=box),
            # This test covers the FlashInfer decode core (not the sparse one).
            cores={(rows_n, 1): kernel},
        )
        # Stale rows and LSEs of an earlier layer must not leak into empty rows.
        buffers[2].fill_(float("nan"))
        buffers[3].fill_(1.0)
        row, lse, valid = PreparedDsa.dcp_core(core, cache, tokens, md, rows_n)
        assert row.shape == (rows_n, 128, 512) and lse.shape == (rows_n, 128)
        assert row.data_ptr() == buffers[2].data_ptr()
        assert lse.data_ptr() == buffers[3].data_ptr()
        if rank:
            assert (valid[:q] == 0).all(), "an empty shard must not count"
        # As DcpValue: the rows of no valid entries are dropped (LSE unused).
        lse = lse.masked_fill((valid == 0).view(-1, 1), -torch.inf)
        rows.append(row[:, :heads].float())
        lses.append(lse[:, :heads].float())

    lse = torch.stack(lses)
    if not impl.lse_base_on_e:
        lse = lse * torch.log(torch.tensor(2.0, device="cuda"))
    weights = torch.softmax(lse, dim=0).unsqueeze(-1)
    merged = (weights * torch.stack(rows).nan_to_num()).sum(0)

    queries = query[:, :heads].view(e4m3).float()
    for t in range(rows_n):
        picked = tokens[t][tokens[t] >= 0].long()
        keys = kv[t // q, picked]
        keys = nvfp4_rows(keys, e4m3=True) if nvfp4 else keys.float()
        probs = torch.softmax(queries[t] @ keys.T * impl.scale, dim=-1)
        ref = probs @ keys[:, :512]
        torch.testing.assert_close(merged[t], ref, atol=3e-2, rtol=3e-2)


def test_flashinfer_kernel_policy():
    """The FlashInfer sparse MLA kernel per batch: one runnable kernel is always
    taken, else the measured table per topology."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.policy import (
        batch_table,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.attn_fi import (
        FI_KERNEL,
        KERNELS,
        kernel_for,
    )

    for dcp in (False, True):
        table = batch_table(FI_KERNEL["dcp" if dcp else "tp"])
        assert set(range(1, 321)) <= set(table) and set(table.values()) <= {*KERNELS}
        for rows in (1, 5, 40, 60, 64, 320):
            assert kernel_for(rows, dcp, KERNELS) == table[rows]
            for kernel in KERNELS:
                assert kernel_for(rows, dcp, (kernel,)) == kernel
    # DCP8 batches whose KV split schedules differ between the kernels (trtllm-gen
    # splits each row up to 39 rows, prims_ts only on plans of up to 19) stay on
    # the stock decode's kernel (test_flashinfer_kernels_share_precision).
    for rows in range(20, 40):
        assert kernel_for(rows, True, KERNELS) == "trtllm"


def fi_buffers(rows, heads, topk=2048):
    """FlashInferCore's shared buffers: flat rows, counts, latent and LSE."""
    i32 = dict(dtype=torch.int32, device="cuda")
    return (
        torch.empty(rows, topk, **i32),
        torch.empty(rows, **i32),
        torch.empty(rows * heads * 512, dtype=torch.bfloat16, device="cuda"),
        torch.empty(rows * heads, dtype=torch.float32, device="cuda"),
    )


def flashinfer_impl(heads, kernel, config, kv_cache_dtype="fp8_e4m3"):
    """A FlashInfer sparse MLA impl at K3.1's MLA sizes on ``kernel`` (trtllm-gen or
    prims_ts), with a workspace."""
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseImpl,
    )

    config.kernel_config.flashinfer_sparse_mla_kernel = kernel
    text = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
    object.__setattr__(config, "model_config", SimpleNamespace(hf_text_config=text))
    impl = FlashInferMLASparseImpl(
        num_heads=heads,
        head_size=FP8_ROW,
        scale=192**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=kv_cache_dtype,
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
        kv_b_proj=ColumnParallelLinear(512, heads * 256, bias=False),
    )
    impl._workspace_buffer = torch.zeros(1 << 30, dtype=torch.int8, device="cuda")
    return impl


def random_fp8(*shape, gen=None):
    """FP8 E4M3 bytes of magnitude below 2 (no NaN codes)."""
    raw = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda", generator=gen)
    return raw & 0xBF


@pytest.mark.parametrize("context", [600, 4096, 33000])
@pytest.mark.parametrize(
    "batch,q", [(1, 1), (5, 5), (10, 5), (40, 5), (60, 5), (64, 1)]
)
@pytest.mark.parametrize("kernel", ["trtllm", "prims_ts", "nvfp4"])
def test_flashinfer_sparse_core_tp(
    default_vllm_config, dist_init, kernel, batch, q, context
):
    """TP8 (12 local heads, the whole context) on FlashInfer sparse (per-tensor
    FP8): the sparse core's logical -> physical top-k conversion and sparse decode
    of the FP8 query rows over the flat rows of a packed cache, against softmax
    attention over the gathered rows. ``q`` rows per request (DSpark verify) each
    see their context up to their own position; contexts around ``context`` (some
    shorter than the top-k, -1 padded), one row without entries. The latent lands
    in the shared buffer, as MlaValue reads it. ``nvfp4``: prims_ts over an
    nvfp4_ds_mla cache."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.attn_fi import (
        FlashInferCore,
    )
    from vllm.utils.flashinfer import has_flashinfer_prims_ts_sparse_mla

    if kernel != "trtllm" and not has_flashinfer_prims_ts_sparse_mla():
        pytest.skip("FlashInfer has no prims_ts sparse MLA")
    nvfp4 = kernel == "nvfp4"
    kv_dtype = "nvfp4_ds_mla" if nvfp4 else "fp8_e4m3"
    kernel = "prims_ts" if nvfp4 else kernel
    impl = flashinfer_impl(HEADS, kernel, default_vllm_config, kv_dtype)
    topk_buffer = torch.full((64, 2048), -1, dtype=torch.int32, device="cuda")
    layer = SimpleNamespace(_q_scale_float=1.0, _k_scale_float=1.0)

    gen = torch.Generator(device="cuda").manual_seed(batch * 7 + context)
    reqs, block = batch // q, UNIT
    lengths = context + torch.randint(-300, 300, (reqs,), generator=gen, device="cuda")
    lengths = lengths.clamp(min=q).tolist()
    pages = [-(-n // block) for n in lengths]
    # Blocks lie a block apart (other layers' pages), in a random order.
    if nvfp4:
        mla = packed_cache(sum(pages) + 2, block, NVFP4_ROW, gap=block)
        mla.copy_(random_nvfp4(*mla.shape[:2], gen=gen))
    else:
        mla = packed_cache(sum(pages) + 2, block, FP8_ROW, gap=block)
        mla.copy_(random_fp8(*mla.shape, gen=gen))
    order = torch.randperm(sum(pages) + 2, generator=gen, device="cuda").int()
    table = torch.zeros(reqs, max(pages), dtype=torch.int32, device="cuda")
    start = 0
    for r, n in enumerate(pages):
        table[r, :n] = order[start : start + n]
        start += n
    topk = torch.full((batch, 2048), -1, dtype=torch.int32, device="cuda")
    for t in range(batch):
        visible = lengths[t // q] - (q - 1 - t % q)
        n = min(visible, 2048)
        at = torch.randperm(2048, generator=gen, device="cuda")[:n]
        pick = torch.randperm(visible, generator=gen, device="cuda")[:n]
        topk[t, at] = pick.int()
    if batch > 2:
        topk[batch // 2] = -1  # a row without entries is zero
    query = random_fp8(batch, HEADS, FP8_ROW, gen=gen)
    md = SimpleNamespace(
        req_id_per_token=torch.arange(batch, dtype=torch.int32, device="cuda") // q,
        block_table=table,
        block_size=block,
    )
    topk_buffer[:batch] = topk
    buffers = fi_buffers(64, HEADS)
    fi = FlashInferCore(impl, layer, HEADS, buffers, {batch: kernel})
    latent, lse = fi.run(query, mla, topk_buffer[:batch], md)
    assert lse is None and latent.shape == (batch, HEADS, 512)
    assert latent.data_ptr() == buffers[2].data_ptr() and latent.is_contiguous()

    # The same decode replayed in a CUDA graph (a row without entries counts 0:
    # no kernel stalls on it) gives the same rows.
    eager = latent.clone()
    torch.accelerator.synchronize()
    graph, capture = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=capture):
        fi.run(query, mla, topk_buffer[:batch], md)
    latent.fill_(7)
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize()
    live = (topk >= 0).any(1)
    assert torch.equal(latent[live], eager[live])

    queries = query.view(torch.float8_e4m3fn).float()
    for t in range(batch):
        picked = topk[t][topk[t] >= 0].long()
        if picked.numel() == 0:
            continue  # a padded row: any finite or not, MlaValue's rows drop it
        rows = mla[table[t // q, picked // block].long(), picked % block]
        if nvfp4:
            keys = nvfp4_rows(rows, e4m3=True)
        else:
            keys = rows.view(torch.float8_e4m3fn).float()
        probs = torch.softmax(queries[t] @ keys.T * impl.scale, dim=-1)
        ref = probs @ keys[:, :512]
        torch.testing.assert_close(latent[t].float(), ref, atol=2e-2, rtol=2e-2)


# The sparse MLA kernels the per-batch policy mixes (FI_KERNEL) must compute at one
# precision: an E4M3 P quantized at 448 (scaled to the row max), FP32 softmax
# statistics and accumulation, BF16 split-KV partials and a BF16 output (FlashInfer:
# prims_ts mla_decode.py / helpers/math.py, trtllm-gen fmhaReduction.cu). Cases: TP8
# (12 heads) and DCP8 (96 heads padded to 128, with LSE) at the batches the policy
# gives each kernel; per-row counts of one KV tile ("tile": no rescale or split, so
# an emulation of that precision is exact), DCP8's typical ~1/8 of the top-k and the
# whole top-k; flat and peaked scores. Each batch on prims_ts runs its own plan
# rows (attn_fi.plan_rows_for), as the specialized decode plans it.
PRECISION_ROWS = {12: (10, 15, 30, 35, 64), 128: (1, 5, 12, 19, 40, 64, 160, 320)}
PRECISION_CASES = [
    (heads, rows, counts, peak)
    for heads, batches in PRECISION_ROWS.items()
    for rows in batches
    for counts in ("tile", "dcp", "full")
    for peak in (False, True)
]


def e4m3_round(x):
    """FP32 -> E4M3 (round to nearest even, saturated) -> FP32."""
    return x.clamp(-448, 448).to(torch.float8_e4m3fn).float()


def sparse_mla_emulations(queries, keys, scale):
    """Per-row sparse MLA outputs [H, 512] (FP64 math) of one row's queries [H, 576]
    and gathered keys [N, 576] (dequantized): "exact" softmax, and with P rounded
    as a kernel would round it (one KV tile: the row max is final): "e4m3" P x 448
    -> E4M3 with an FP32 sum of the unrounded P, "e4m3_sum" summing the rounded P,
    "bf16" P in BF16. Returns ({name: out}, natural-log LSE [H])."""
    s = (queries.double() @ keys.double().T).float()  # FP32 scores
    m = s.max(-1, keepdim=True).values
    p = torch.exp2((s - m) * (scale * math.log2(math.e)))  # FP32, as the kernels
    v = keys[:, :512].double()
    exact = torch.softmax(s.double() * scale, -1) @ v
    p8 = e4m3_round(p * 448).double()
    out = {
        "exact": exact,
        "e4m3": (p8 @ v) / (448 * p.double().sum(-1, keepdim=True)),
        "e4m3_sum": (p8 @ v) / p8.sum(-1, keepdim=True),
        "bf16": (p.bfloat16().double() @ v) / p.double().sum(-1, keepdim=True),
    }
    lse = torch.logsumexp(s.double() * scale, -1)
    return out, lse


@pytest.mark.parametrize(
    "heads,rows,counts,peak",
    PRECISION_CASES,
    ids=[f"h{h}-r{r}-{c}-{'peak' if p else 'flat'}" for h, r, c, p in PRECISION_CASES],
)
def test_flashinfer_kernels_share_precision(
    default_vllm_config, dist_init, heads, rows, counts, peak
):
    """trtllm-gen and prims_ts (the kernels FI_KERNEL mixes per batch) on identical
    FP8 inputs: on one KV tile both equal (to the BF16 ulp) the same emulation of
    their rounding (E4M3 P at 448, FP32 sum and accumulation, BF16 output), and on
    longer rows (online softmax, split-KV with BF16 partials) their errors against
    exact FP64 attention have the same size and no systematic difference (scale
    bias); under DCP8 also their LSEs."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.attn_fi import (
        KERNELS,
        FlashInferCore,
        kernel_for,
    )
    from vllm.utils.flashinfer import has_flashinfer_prims_ts_sparse_mla

    if not has_flashinfer_prims_ts_sparse_mla():
        pytest.skip("FlashInfer has no prims_ts sparse MLA")
    dcp = 8 if heads == 128 else 1
    # The policy runs prims_ts on this batch (the stock decode: trtllm-gen).
    assert kernel_for(rows, dcp > 1, KERNELS) == "prims_ts"
    live = HEADS * 8 if dcp > 1 else HEADS
    gen = torch.Generator(device="cuda").manual_seed(rows * 31 + len(counts) + peak)
    e4m3 = torch.float8_e4m3fn
    block, pool = UNIT, 8192
    cache = packed_cache(pool // block, block, FP8_ROW, gap=0)
    keys = torch.randn(pool, FP8_ROW, generator=gen, device="cuda")
    cache.view(-1, FP8_ROW).copy_(keys.to(e4m3).view(torch.uint8))
    flat = cache.view(-1, FP8_ROW).view(e4m3).float()
    # Score std ~1.7 sigma_q: flat ~1, peaked ~5 (much of the row underflows E4M3).
    sigma = 3.0 if peak else 0.6
    query = torch.zeros(rows, heads, FP8_ROW, dtype=torch.uint8, device="cuda")
    q = torch.randn(rows, live, FP8_ROW, generator=gen, device="cuda") * sigma
    query[:, :live] = q.to(e4m3).view(torch.uint8)
    width = 2048
    slots = torch.full((rows, width), -1, dtype=torch.int32, device="cuda")
    valid = torch.zeros(rows, dtype=torch.int32, device="cuda")
    for t in range(rows):
        if counts == "tile":
            n = 32
        elif counts == "dcp":
            n = int(torch.randint(200, 320, (1,), generator=gen, device="cuda"))
        else:
            n = width
        pick = torch.randperm(pool, generator=gen, device="cuda")[:n]
        slots[t, :n] = pick.int()
        valid[t] = n

    outs, lses = {}, {}
    for kernel in ("trtllm", "prims_ts"):
        impl = flashinfer_impl(live, kernel, default_vllm_config)
        impl.need_to_return_lse_for_decode = dcp > 1
        layer = SimpleNamespace(_q_scale_float=1.0, _k_scale_float=1.0)
        buffers = fi_buffers(rows, heads)
        fi = FlashInferCore(impl, layer, heads, buffers, {rows: kernel}, dcp, 0)
        latent, lse = fi.decode(query, cache, slots, valid, block)
        outs[kernel] = latent[:, :live].double().clone()
        if lse is not None:
            lse = lse[:, :live].double()
            lses[kernel] = lse if impl.lse_base_on_e else lse * math.log(2.0)
        scale = impl.scale

    queries = query[:, :live].view(e4m3).float()
    refs: dict[str, list] = {}
    ref_lse = []
    for t in range(rows):
        picked = slots[t, : int(valid[t])].long()
        emulated, lse = sparse_mla_emulations(queries[t], flat[picked], scale)
        for name, out in emulated.items():
            refs.setdefault(name, []).append(out)
        ref_lse.append(lse)
    ref = {name: torch.stack(v) for name, v in refs.items()}
    exact = ref["exact"]

    def ulps(out, target):
        """|out - target| in BF16 ulps of the target (rounded to BF16)."""
        t = target.bfloat16().double()
        ulp = torch.where(t == 0, 2.0**-133, t.abs()) * 2.0**-7
        exp = torch.floor(torch.log2(ulp.abs()))
        return (out - t).abs() / torch.exp2(exp)

    stats = {}
    for kernel, out in outs.items():
        err = out - exact
        rel = float(err.norm() / exact.norm())
        # Scale bias: the least-squares out ~ (1 + bias) * exact (a P rounding that
        # shrinks or inflates the rows), in units of the error.
        bias = float((err * exact).sum() / (exact * exact).sum())
        within = {
            name: float((ulps(out, r) <= 1).double().mean()) for name, r in ref.items()
        }
        stats[kernel] = dict(rel=rel, bias=bias, within=within)
        if lses:
            lse_err = lses[kernel] - torch.stack(ref_lse)
            stats[kernel]["lse"] = (
                float(lse_err.abs().max()),
                float(lse_err.mean()),
            )
    print("PRECISION", heads, rows, counts, "peak" if peak else "flat", stats)

    trt, pts = stats["trtllm"], stats["prims_ts"]
    if counts == "tile":
        # One tile: both are the E4M3-P emulation to the ulp (FP32 order and ex2
        # approximation aside), and neither is the BF16-P or exact one.
        for s in (trt, pts):
            assert s["within"]["e4m3"] >= 0.995, stats
    assert 0.8 <= pts["rel"] / trt["rel"] <= 1.25, stats
    assert abs(pts["bias"] - trt["bias"]) <= 0.1 * max(pts["rel"], trt["rel"]), stats
    if lses:
        assert abs(pts["lse"][1] - trt["lse"][1]) <= 1e-3, stats
        assert max(pts["lse"][0], trt["lse"][0]) <= 1e-2, stats


# A K3.1 DSA layer at TP8's local sizes (12 heads) on the model config the metadata
# builders read (DeepSeek-V2-Lite's MLA sizes: kv_lora 512, rope 64).
LAYER_MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"
LAYER_HIDDEN, LAYER_LEN = 1536, 8192  # 128-wide (rope-checked) heads
# (cache dtype, backend, FlashInfer sparse MLA kernel); "fi-auto": the kernel per
# batch from the policy table (the stock decode on trtllm-gen); "flashmla": the
# fp8_ds_mla cache on FlashMLA sparse (the stock fused forward's Triton norms).
LAYER_CORES = {
    "fi-trtllm": ("fp8", "FLASHINFER_MLA_SPARSE", "trtllm"),
    "fi-prims_ts": ("fp8", "FLASHINFER_MLA_SPARSE", "prims_ts"),
    "fi-auto": ("fp8", "FLASHINFER_MLA_SPARSE", "auto"),
    "fi-nvfp4": ("nvfp4_ds_mla", "FLASHINFER_MLA_SPARSE", "prims_ts"),
    "flashmla": ("fp8_ds_mla", "FLASHMLA_SPARSE", None),
}


def dsa_layer(config, topk_buffer, heads=HEADS):
    """A real KimiK3DSAAttention (``heads`` over all TP ranks) on random weights,
    its MLA weights processed."""
    from vllm.models.kimi_k3.nvidia.dsa import KimiK3DSAAttention
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    text = KimiLinearConfig(
        hidden_size=LAYER_HIDDEN,
        num_attention_heads=heads,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_nope=True,
        mla_use_output_gate=True,
        rms_norm_eps=EPS,
        dsa_config={
            "dsa_layers": [0],
            "index_topk": 2048,
            "index_head_dim": 128,
            "index_num_heads": 32,
            "index_rope_dim": 0,
            "index_k_norm_type": "layernorm",
        },
    )
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            attn = KimiK3DSAAttention(
                text,
                hidden_size=LAYER_HIDDEN,
                num_heads=heads,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                q_lora_rank=1536,
                kv_lora_rank=512,
                cache_config=config.cache_config,
                prefix="model.layers.0.self_attn",
                topk_indices_buffer=topk_buffer,
            )
    finally:
        torch.set_default_dtype(torch.float32)
    torch.manual_seed(5)
    for name, param in attn.named_parameters():
        if name.endswith("bias"):
            param.data.copy_(torch.randn_like(param) / 10)
        elif "norm" in name:
            param.data.copy_(torch.rand_like(param) + 0.5)
        else:
            param.data.copy_(torch.randn_like(param) * param.shape[-1] ** -0.5)
    attn.mla_attn.mla_attn.process_weights_after_loading()
    return attn


def paged_rows(lengths, block, row, gap, gen):
    """A packed paged cache with ``gap`` rows of other layers' pages between its
    blocks, each request's blocks in a random order, and their block table."""
    pages = [max(-(-n // block), 1) for n in lengths]
    cache = packed_cache(sum(pages) + 2, block, row, gap)
    order = torch.randperm(sum(pages) + 2, generator=gen, device="cuda").int()
    table = torch.zeros(len(lengths), max(pages), dtype=torch.int32, device="cuda")
    start = 0
    for r, n in enumerate(pages):
        table[r, :n] = order[start : start + n]
        start += n
    return cache, table


def owned_slots(table, positions, world=1, rank=0):
    """Cache slots of each request's token ``positions`` on DCP rank ``rank`` of
    ``world`` (interleave 1: position p lives on rank p % world, at p // world);
    -1 where another rank owns the token."""
    slots = []
    for r, pos in enumerate(positions):
        pos = torch.as_tensor(pos, device="cuda")
        local = pos // world
        page = table[r, (local // UNIT).clamp(max=table.shape[1] - 1)].long()
        slots.append(torch.where(pos % world == rank, page * UNIT + local % UNIT, -1))
    return torch.cat(slots)


def common_metadata(contexts, q, table, slots, world=1, rank=0):
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

    reqs = len(contexts)
    qsl = torch.arange(reqs + 1, dtype=torch.int32) * q
    seq_lens = torch.tensor([c + q for c in contexts], dtype=torch.int32)
    local = get_dcp_local_seq_lens(seq_lens, world, rank, 1) if world > 1 else None
    return CommonAttentionMetadata(
        query_start_loc=qsl.cuda(),
        query_start_loc_cpu=qsl,
        seq_lens=seq_lens.cuda(),
        num_reqs=reqs,
        num_actual_tokens=reqs * q,
        max_query_len=q,
        max_seq_len=int(seq_lens.max()),
        block_table_tensor=table,
        slot_mapping=slots,
        seq_lens_cpu_upper_bound=seq_lens,
        dcp_local_seq_lens=None if local is None else local.cuda(),
        dcp_local_seq_lens_cpu_upper_bound=local,
    )


def cache_rows(cache, table, ids, request, kv):
    """Dequantized [len(ids), 576] rows of a request's logical tokens ``ids`` in a
    ``kv`` (cache dtype) cache."""
    block = cache.shape[1]
    rows = cache[table[request, ids // block].long(), ids % block]
    if kv == "fp8":  # per-tensor FP8 (unit scale)
        return rows.view(torch.float8_e4m3fn).float()
    if kv == "nvfp4_ds_mla":
        return nvfp4_rows(rows)
    latent = rows[:, :512].view(torch.float8_e4m3fn).float().view(-1, 4, 128)
    latent = (latent * rows[:, 512:528].view(torch.float32).view(-1, 4, 1)).view(
        -1, 512
    )
    return torch.cat([latent, rows[:, 528:].view(torch.bfloat16).float()], -1)


def layer_reference(attn, projected, cache, table, topk, q, kv):
    """The gated attention rows in FP32 from the packed projection rows: the query
    unquantized, the keys and values as cached, over each row's ``topk``."""
    wrapper = attn.mla_attn
    layer = wrapper.mla_attn
    heads = layer.W_UK_T.shape[0]
    x = projected[:, :1536].float()
    eps = wrapper.q_a_layernorm.variance_epsilon
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    q_c = x * wrapper.q_a_layernorm.weight.float()
    query = (q_c @ wrapper.q_b_proj.weight.float().T).view(-1, heads, 192)
    nope = torch.einsum("thd,hdl->thl", query[..., :128], layer.W_UK_T.float())
    query = torch.cat([nope, query[..., 128:]], -1)
    latent = torch.zeros(query.shape[0], heads, 512, device=query.device)
    for t in range(query.shape[0]):
        ids = topk[t][topk[t] >= 0].long()
        keys = cache_rows(cache, table, ids, t // q, kv)
        probs = torch.softmax(query[t] @ keys.T * layer.impl.scale, -1)
        latent[t] = probs @ keys[:, :512]
    out = torch.einsum("thl,hlv->thv", latent, layer.W_UV.float())
    gate = projected[:, 2112 : 2112 + heads * 128].float()
    return out.reshape(query.shape[0], -1) * torch.sigmoid(gate)


def layer_config(core, q, world=1, index_dtype="mxfp4"):
    """The vLLM config of a DSA layer test on ``core`` (TP = DCP = ``world``);
    ``index_dtype``: the indexer cache's (the K3.1 deployments request MXFP4)."""
    from tests.v1.attention.utils import create_vllm_config
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    cache_dtype, backend, kernel = LAYER_CORES[core]
    config = create_vllm_config(
        model_name=LAYER_MODEL,
        max_model_len=LAYER_LEN,
        block_size=UNIT,
        max_num_seqs=64,
        max_num_batched_tokens=LAYER_LEN,
        hf_config_override={"index_topk": 2048},
    )
    if world > 1:  # TP = DCP across nodes (ParallelConfig checks one node's GPUs)
        parallel = config.parallel_config
        parallel.tensor_parallel_size = parallel.world_size = world
        parallel.decode_context_parallel_size = world
        parallel.cp_kv_cache_interleave_size = 1
    config.cache_config.cache_dtype = cache_dtype
    config.attention_config.backend = AttentionBackendEnum[backend]
    config.attention_config.indexer_kv_dtype = index_dtype
    if kernel is not None:
        config.kernel_config.flashinfer_sparse_mla_kernel = kernel
    if q > 1:  # DSpark verify: the indexer's decode rows take q tokens
        config.speculative_config = SimpleNamespace(
            num_speculative_tokens=q - 1,
            enable_adaptive_verification=False,
            parallel_drafting=False,
        )
    return config


# The TP query overlap modes (overlap.MODES but "wait", a DCP mode).
TP_OVERLAPS = ("serial", "fork", "front", "late", "late_front")


def fi_policy(attn, kernel, cache_dtype):
    """PreparedDsa's FlashInfer kernels for a test core: "auto" the specialized
    decode's per-batch policy (its default), else the one kernel the stock layer
    is configured to (prims_ts on the stock decode's plans), so that batch runs as
    stock."""
    if kernel in (None, "auto"):
        return {}
    impl = attn.mla_attn.mla_attn.impl
    return dict(
        kernels=(kernel,), prims_ts_rows=getattr(impl, "prims_ts_buckets", None)
    )


def run_layer_step(
    core,
    batch,
    q,
    world=1,
    overlap="serial",
    partials="bf16",
    graph=False,
    poison="rows",
    offset=0,
    index_dtype="mxfp4",
):
    """One decode step of a real DSA layer (``q`` rows per request, contexts on
    both sides of the top-k, written by the stock writers) through the stock
    KimiK3DSAAttention forward and through PreparedDsa.run, each over its own copy
    of the same caches, on the same metadata. Under DCP (``world`` ranks, this
    process one of them, interleave 1) each rank holds its shard of every context
    and PreparedDsa runs with the real q / partial-row mailboxes. ``overlap``: the
    query overlap mode (a forked query is held back on rank 0); ``partials``: the
    DcpValue exchange ("bf16", "proj" or a batch table of them). ``graph``: also
    replay the step in a CUDA graph with the fused indexer's outputs poisoned
    before every layer (``poison`` "rows": every row full, pointing at slot 0;
    "counts": only the valid counts, set to 1); ``offset``: added to every
    context (past 2048: no row takes the select's short-context bypass)."""
    from vllm import _custom_ops as ops
    from vllm.config import set_current_vllm_config
    from vllm.distributed import get_dcp_group
    from vllm.forward_context import create_forward_context, override_forward_context
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer import (
        indexer_k_norm_quant_insert,
    )
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import (
        mxfp4_k_norm_quant_insert,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.overlap import (
        side_stream,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.prepared import (
        PreparedDsa,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.providers import (
        dcp_box,
        pack_dsa_projection,
    )
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.v1.worker.block_table import get_block_table_width

    cache_dtype, _, kernel = LAYER_CORES[core]
    config = layer_config(core, q, world, index_dtype)
    rank = get_dcp_group().rank_in_group if world > 1 else 0
    row = {"fp8": FP8_ROW, "nvfp4_ds_mla": NVFP4_ROW}.get(cache_dtype, MLA_ROW)
    flashinfer = cache_dtype != "fp8_ds_mla"
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    bf = torch.bfloat16
    with set_current_vllm_config(config):
        topk_buffer = torch.full((LAYER_LEN, 2048), -1, dtype=torch.int32, device=dev)
        attn = dsa_layer(config, topk_buffer, HEADS * world)
        layer, indexer = attn.mla_attn.mla_attn, attn.dsa_indexer
        packed = pack_dsa_projection(attn)
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
                index_spec.max_num_blocks_per_req(config, LAYER_LEN), UNIT
            ),
        )

        # Every rank draws the same contexts and step; it holds its shard of them.
        reqs = batch // q
        gen = torch.Generator(device=dev).manual_seed(batch + q)
        contexts = [
            (600, 2050, 4321, 3000, 1100)[r % 5] + 37 * r + offset for r in range(reqs)
        ]
        local = [(c + q - rank + world - 1) // world for c in contexts]
        mla, mla_table = paged_rows(local, UNIT, row, UNIT, gen)
        index, index_table = paged_rows(local, UNIT, index_row(indexer), GAP, gen)
        old = [range(c) for c in contexts]
        old_mla = owned_slots(mla_table, old, world, rank)
        old_index = owned_slots(index_table, old, world, rank)
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
        # Finite logits: the zero-filled pages and the inserted keys hold no NaN.
        assert_index_nan_free(index)
        new = [range(c, c + q) for c in contexts]
        new_mla = owned_slots(mla_table, new, world, rank)
        new_index = owned_slots(index_table, new, world, rank)

        hidden = torch.randn(batch, LAYER_HIDDEN, device=dev, dtype=bf, generator=gen)
        positions = torch.tensor([p for pos in new for p in pos], device=dev)
        names = layer.layer_name, indexer.k_cache.prefix

        def step(mla_cache, index_cache):
            layer.kv_cache, indexer.k_cache.kv_cache = mla_cache, index_cache
            mla_common = common_metadata(contexts, q, mla_table, new_mla, world, rank)
            index_common = common_metadata(
                contexts, q, index_table, new_index, world, rank
            )
            md = builder.build(0, mla_common)
            index_md = index_builder.build(0, index_common)
            context = create_forward_context(
                dict(zip(names, (md, index_md))),
                config,
                slot_mapping=dict(zip(names, (new_mla, index_md.slot_mapping))),
            )
            return md, index_md, override_forward_context(context)

        # The stock forward: the rows entering o_proj.
        stock_mla, stock_index = mla.clone(), index.clone()
        entering = []
        hook = attn.o_proj.register_forward_pre_hook(
            lambda _, args: entering.append(args[0].clone())
        )
        _, _, context = step(stock_mla, stock_index)
        with context:
            attn(positions, hidden)
        hook.remove()
        stock_topk = topk_buffer[:batch].clone()
        assert indexer.use_fused_k_insert

        # The specialized decode's layer over copies of the same caches.
        projected = torch.cat(
            [
                attn.fused_qkv_a_proj(hidden)[0],
                attn.g_proj(hidden)[0],
                indexer.wkv_index(hidden)[0],
            ],
            dim=-1,
        )
        assert projected.shape[1] == packed.shape[0]
        ours_mla, ours_index = mla.clone(), index.clone()
        md, index_md, context = step(ours_mla, ours_index)
        cap = max(64, batch)  # the scratch rows of the step's widest batch
        scratch_rows = (
            torch.zeros(cap, 64, 576, device=dev, dtype=bf),
            torch.empty(cap, 1536, device=dev, dtype=bf),
            torch.empty(cap, 32 * 128, device=dev, dtype=bf),
            torch.empty(cap, 32, 128, device=dev, dtype=torch.uint8),
            torch.empty(cap, 32, device=dev, dtype=bf),
            fi_buffers(cap, 128),
        )
        policy: dict = dict(queries=(q,), overlap=overlap)
        policy.update(fi_policy(attn, kernel, cache_dtype))
        if world > 1:
            runtime = SimpleNamespace(
                dcp=world,
                tp=world,
                device=dev,
                dcp_box=None,
                dsa_dcp_box=None,
                batches=(batch,),
                wide_batches=lambda kind: (),
            )
            box = dcp_box(
                runtime, HEADS, dsa=True, flashinfer=flashinfer, partials=partials
            )
            policy.update(dcp=box)
        prepared = PreparedDsa(attn, {batch: packed.shape[0]}, scratch_rows, **policy)
        prepared.prepare_cache(ours_mla)
        # prims_ts' plans share one workspace (attn_fi._Plans): each decode runs
        # on the stream the layer runs on, never on a query overlap side stream.
        layer_streams: list = []
        prims_ts_streams: list = []
        if prepared.fi is not None:
            decode = prepared.fi.prims_ts

            def on_stream(*args, **kwargs):
                prims_ts_streams.append(
                    (torch.cuda.current_stream(), layer_streams[-1])
                )
                return decode(*args, **kwargs)

            prepared.fi.prims_ts = on_stream
        if prepared.fused is not None:
            prepared.fused.write_ids(topk_buffer)  # the fused indexer's top-k
        out = torch.empty(batch, HEADS * 128, device=dev, dtype=bf)
        merge = prepared.indexer_stage
        if world > 1 and flashinfer:
            # The one-round DCP top-k merge (#63) runs on both paths; ours also
            # writes the global ids for the top-k check below.
            assert merge.local_topk == indexer.dcp_local_topk > 0
            merge.dcp_ids = topk_buffer
            # The fused indexer's scorer and local top-N select feed it (past one
            # row per SM, wide steps: its scorer, and the merge's own select).
            assert merge.fused is not None
            assert merge.fused.packs(batch) == (
                merge.fused.packed and batch <= merge.fused.ctas
            )
            merge.fused.valid.fill_(-5)
        topk_buffer.fill_(-1)
        if overlap in ("fork", "front") and rank == 0:
            # Hold back the forked query (and under DCP rank 0's publish): the
            # core must wait for it (the join), and for every rank's (QueryWait).
            with torch.cuda.stream(side_stream(dev.index)):
                torch.cuda._sleep(40_000_000)

        def run():
            layer_streams.append(torch.cuda.current_stream())
            prepared.run(
                projected,
                out,
                cache=ours_mla,
                slots=new_mla,
                index_slots=index_md.slot_mapping,
                metadata=md,
                q=q,
            )

        with context:
            run()
            if graph:
                # Replayed back to back with the fused indexer's rows poisoned
                # first: every kernel after the select must wait for it (PDL).
                fused = prepared.fused
                assert fused is not None
                torch.accelerator.synchronize()
                g = torch.cuda.CUDAGraph()
                capture = torch.cuda.Stream()
                capture.wait_stream(torch.cuda.current_stream())
                with torch.cuda.graph(g, stream=capture):
                    for _ in range(3):
                        if poison == "rows":
                            # Every row full and pointing at slot 0.
                            fused.out.fill_(0)
                            fused.valid.fill_(fused.topk)
                        else:
                            # Only the counts: a decode reading them before its
                            # PDL wait attends one key per row.
                            fused.valid.fill_(1)
                        run()
                for _ in range(5):
                    g.replay()
        torch.accelerator.synchronize()
        off_stream = [(s, main) for s, main in prims_ts_streams if s != main]
        assert not off_stream, off_stream
        owned = None
        if merge.local_topk:
            # This rank's rows from the merge, and the filter of each global top-k.
            from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.dcp_stage import (  # noqa: E501
                local_slots,
            )
            from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view

            _, stride_rows = flat_kv_row_view(ours_mla, md.block_size)
            filtered = [
                local_slots(md, ids[:batch], world, rank, 1, stride_rows)
                for ids in (topk_buffer, stock_topk)
            ]
            merged = merge.dcp_rows[:batch], merge.dcp_valid[:batch]
            owned = (merged, *filtered)
            if merge.fused.packs(batch):
                counts = merge.fused.valid[:batch]  # the fused select ran
                assert bool((counts >= 0).all() & (counts <= merge.local_topk).all())
        # The fused indexer's top-k (TP) or the merge of its scores (DCP) against
        # the exact top-k of its own query. The stock top-k is measured against the
        # same (our) query's exact top-k (at stock's BF16 head-sum precision), so
        # its width carries the two paths' query rounding differences (printed
        # only).
        width = stock_width = None
        if prepared.fused is not None or merge.fused is not None:
            stage = prepared.query_stage
            args = index_table, contexts, q
            q_fp8, head_w = stage.q_fp8[:batch], stage.head_w[:batch]
            width = exact_topk_widths(
                ours_index, *args, q_fp8, head_w, topk_buffer[:batch], world
            )
            stock_width = exact_topk_widths(
                stock_index, *args, q_fp8, head_w, stock_topk, world
            )
    return SimpleNamespace(
        width=width,
        stock_width=stock_width,
        attn=attn,
        projected=projected,
        out=out,
        stock=entering[0],
        topk=topk_buffer[:batch].clone(),
        stock_topk=stock_topk,
        caches=(ours_mla, ours_index),
        stock_caches=(stock_mla, stock_index),
        new_slots=(new_mla, new_index),
        mla_table=mla_table,
        kv=cache_dtype,
        owned=owned,
        prims_ts_calls=len(prims_ts_streams),
    )


def exact_topk_widths(index, table, contexts, q, query, weights, ids, world=1):
    """Each row's ``ids`` (its top-k: logical, or global under DCP) against the
    exact top-k of indexer logits at stock's precision (deepgemm_head_sum) over
    the dequantized keys of the same FP8 query and BF16 head weights (``query``
    [B, 32, 128] bytes, ``weights`` [B, 32]): the largest distance of a swapped
    key's exact logit from the exact K-th one, in BF16 steps of the K-th (inf: a
    key far outside the exact top-k). Under DCP every rank scores its keys and
    the candidates are all-gathered. Exact selection up to the BF16 rounding of
    the logits keeps it near 1."""
    import torch.distributed as dist

    from vllm.distributed import get_dcp_group

    rank = get_dcp_group().rank_in_group if world > 1 else 0
    if world > 1:  # gid = local * world + rank below: DCP interleave 1 only
        from vllm.config import get_current_vllm_config

        config = get_current_vllm_config().parallel_config
        assert config.cp_kv_cache_interleave_size == 1
    topk, spare = ids.shape[1], 512
    pages, kb = index_blocks(index), key_bytes(index)
    if kb == 64:  # MXFP4 keys and query (dequantized; the scales are in the MMA)
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.index import (
            fp4_query_views,
        )

        packed, q_scales = fp4_query_views(query, query.shape[0])
        query = mxfp4_values(packed, q_scales.view(torch.uint8).view(-1, 32, 4))
    else:
        query = query.view(torch.float8_e4m3fn).float()
    vals, gids, lengths = [], [], []
    t = 0
    for r, c in enumerate(contexts):
        n = (c + q - rank + world - 1) // world
        blocks = pages[table[r, : -(-n // UNIT)].long()]
        keys = blocks[:, : UNIT * kb].reshape(-1, kb)[:n]
        sbytes = blocks[:, UNIT * kb :].reshape(-1, UNIT, 4).reshape(-1, 4)[:n]
        if kb == 64:
            keys = mxfp4_values(keys, sbytes)
            scale = torch.ones(n, device=keys.device)
        else:
            keys = keys.view(torch.float8_e4m3fn).float()
            scale = torch.exp2(sbytes[:, 0].float() - 127)
        for j in range(q):
            seen = (c + j + 1 - rank + world - 1) // world
            score = torch.relu(query[t] @ keys[:seen].T)
            # At stock's precision (the specialized scorer's as well).
            logits = deepgemm_head_sum(score, weights[t]) * scale[:seen]
            v = torch.full((topk + spare,), float("-inf"), device="cuda")
            g = torch.full((topk + spare,), -1, dtype=torch.int64, device="cuda")
            k = min(topk + spare, seen)
            top = torch.topk(logits, k)
            v[:k], g[:k] = top.values, top.indices * world + rank
            vals.append(v)
            gids.append(g)
            lengths.append(c + j + 1)
            t += 1
    vals, gids = torch.stack(vals), torch.stack(gids)
    if world > 1:
        group = get_dcp_group().device_group
        parts = [torch.empty_like(vals) for _ in range(world)]
        dist.all_gather(parts, vals, group=group)
        id_parts = [torch.empty_like(gids) for _ in range(world)]
        dist.all_gather(id_parts, gids, group=group)
        vals, gids = torch.cat(parts, 1), torch.cat(id_parts, 1)
    widest = 0.0
    for t, length in enumerate(lengths):
        got = set(ids[t][ids[t] >= 0].tolist())
        if length <= topk:
            assert got == set(range(length)), t
            continue
        top = torch.topk(vals[t], topk + spare)
        cand = gids[t][top.indices]
        kth = float(top.values[topk - 1])
        step = 2.0 ** (math.floor(math.log2(abs(kth))) - 7) if kth else 2.0**-133
        score = dict(zip(cand.tolist(), top.values.tolist()))
        exact = set(cand[:topk].tolist())
        for g in got ^ exact:
            far = g not in score
            widest = max(widest, math.inf if far else abs(score[g] - kth) / step)
    return widest


def mxfp4_values(codes, scales):
    """E2M1 bytes [..., 64] (low nibble first) times their UE8M0 block scales
    [..., 4] -> FP32 [..., 128]."""
    table = torch.tensor(E2M1, device=codes.device)
    table = torch.cat([table, -table])
    nibbles = torch.stack([codes & 15, codes >> 4], -1).flatten(-2).long()
    values = table[nibbles].unflatten(-1, (4, 32))
    return (values * torch.exp2(scales.float() - 127).unsqueeze(-1)).flatten(-2)


def fp8_codes_adjacent(ours, theirs):
    """Whether two E4M3 byte tensors differ by at most one code step (a rounding
    boundary: the same sign, adjacent magnitudes)."""
    ours, theirs = ours.int(), theirs.int()
    step = ((ours & 0x7F) - (theirs & 0x7F)).abs()
    same_sign = (ours & 0x80) == (theirs & 0x80)
    zero = ((ours | theirs) & 0x7F) <= 1  # +-0 and the smallest subnormal
    return bool(((same_sign & (step <= 1)) | zero).all())


def check_layer_step(step, q, reference=True):
    """The specialized decode's step against the stock one: the top-k, the gated
    attention rows (and, ``reference``: each against FP32 attention over its own
    top-k), the MLA rows and the indexer keys (their SM100 MX scales exact) the
    step inserts, and every other cache byte as it was."""
    # The top-k: the same keys up to near-ties at the boundary (the indexer queries
    # of the two paths round differently: their normalized q_c rows do).
    # The fused indexer scores at stock's precision (DeepGEMM's BF16 head sum), so
    # its top-k is the exact top-k of those logits up to BF16 rounding flips
    # (width <= 2 steps). The stock top-k differs from ours only by the two paths'
    # query rounding (and DeepGEMM's FP32 q . k order); the overlap bound below
    # covers that.
    if step.width is not None:
        print(
            f"top-k vs exact (BF16 steps): ours {step.width:.2f} stock (vs our "
            "query) "
            f"{step.stock_width:.2f}"
        )
        assert step.width <= 2.0, step.width
    same, overlaps = [], []
    fp4 = key_bytes(step.caches[1]) == 64
    for t in range(step.topk.shape[0]):
        a, b = set(step.stock_topk[t].tolist()), set(step.topk[t].tolist())
        # Fused vs DeepGEMM at 160-320 rows: >= 98.7% measured (both sum heads in
        # BF16; the paths' query rounding and BF16 near-ties at the K-th logit
        # differ; the width check above bounds ours). MXFP4: a query value whose
        # BF16 projection rounds differently moves by an E2M1 step (1/8 to 1/2
        # of its block's scale), not an E4M3 one, so more keys near the K-th
        # logit swap (each path is still the exact top-k of its own query).
        bound = 0.99 if step.width is None or step.topk.shape[0] <= 64 else 0.98
        if fp4:
            bound -= 0.03
        overlaps.append(len(a & b) / len(a))
        assert overlaps[-1] > bound, (t, len(a), len(a & b))
        same.append(a == b)
    print(
        f"top-k overlap with stock: min {min(overlaps):.4f} mean "
        f"{sum(overlaps) / len(overlaps):.4f}"
    )
    if fp4:
        assert sum(overlaps) / len(overlaps) > 0.985
    out, stock = step.out, step.stock
    # Rows with the same top-k attend over the same keys.
    rows = torch.tensor(same, device=out.device)
    ours, theirs = out[rows].float(), stock[rows].float()
    error = float((ours - theirs).norm() / theirs.norm().clamp(min=1e-30))
    print(f"relative error vs stock (same top-k rows): {error:.4f}")
    torch.testing.assert_close(out[rows], stock[rows], atol=2e-2, rtol=2e-2)
    if reference:
        # Each against the FP32 attention over its own top-k and its cache: the
        # specialized decode is as accurate as the stock decode.
        args = step.attn, step.projected
        ours_ref = layer_reference(
            *args, step.caches[0], step.mla_table, step.topk, q, step.kv
        )
        stock_ref = layer_reference(
            *args, step.stock_caches[0], step.mla_table, step.stock_topk, q, step.kv
        )
        error = lambda x, ref: float((x.float() - ref).norm() / ref.norm())  # noqa: E731
        ours_error, stock_error = error(out, ours_ref), error(stock, stock_ref)
        print(f"relative error vs FP32: ours {ours_error:.4f} stock {stock_error:.4f}")
        assert ours_error <= 1.1 * stock_error + 1e-3, (ours_error, stock_error)

    # The new tokens' MLA rows (latent codes differ only by one step, where the
    # RMSNorm rounds differently) and indexer keys (one step at most; the scales
    # exact); nothing else written.
    (ours_mla, ours_index), (stock_mla, stock_index) = step.caches, step.stock_caches
    new_mla, new_index = (s[s >= 0] for s in step.new_slots)
    at_new = new_mla // UNIT, new_mla % UNIT
    ours, theirs = ours_mla[at_new], stock_mla[at_new]
    if step.kv == "nvfp4_ds_mla":
        # k_pe exact; where the RMSNorm rounds differently a block scale is one
        # E4M3 code off and E2M1 codes differ (rarely), and the dequantized
        # latents are close.
        assert torch.equal(ours[:, 256:320], theirs[:, 256:320])
        scales = (ours[:, 320:].int() - theirs[:, 320:].int()).abs()
        assert bool((scales <= 1).all())
        differ = (ours[:, :256] != theirs[:, :256]).sum()
        assert differ <= 0.05 * ours[:, :256].numel()
        a, b = nvfp4_rows(ours)[:, :512], nvfp4_rows(theirs)[:, :512]
        assert (a - b).norm() <= 0.05 * b.norm() + 1e-6
    else:
        assert torch.equal(ours[:, 512:], theirs[:, 512:])
        assert fp8_codes_adjacent(ours[:, :512], theirs[:, :512])
        # (Under DCP a rank may own none of the new tokens.)
        differ = (ours[:, :512] != theirs[:, :512]).sum()
        assert differ <= 0.05 * ours[:, :512].numel()
    ours_mla[at_new] = 0
    stock_mla[at_new] = 0
    assert torch.equal(ours_mla, stock_mla)
    page, entry = new_index // UNIT, new_index % UNIT
    blocks = index_blocks(ours_index), index_blocks(stock_index)
    arange = partial(torch.arange, device=ours_index.device)
    kb = key_bytes(ours_index)
    at = UNIT * kb + 4 * entry.view(-1, 1) + arange(4)
    assert torch.equal(blocks[0][page].gather(1, at), blocks[1][page].gather(1, at))
    key_at = kb * entry.view(-1, 1) + arange(kb)
    ours_keys, stock_keys = (b[page].gather(1, key_at) for b in blocks)
    if kb == 128:
        assert fp8_codes_adjacent(ours_keys, stock_keys)
        assert (ours_keys != stock_keys).sum() <= 0.05 * ours_keys.numel()
    else:  # MXFP4: the same writer math as the stock insert, bit-exact
        assert torch.equal(ours_keys, stock_keys)
    for b in blocks:
        b[page.view(-1, 1), at] = 0
        b[page.view(-1, 1), key_at] = 0
    assert torch.equal(blocks[0], blocks[1])


@pytest.mark.parametrize("overlap", list(TP_OVERLAPS))
@pytest.mark.parametrize("batch,q", [(1, 1), (8, 1), (5, 5), (40, 5), (60, 5)])
@pytest.mark.parametrize("core", list(LAYER_CORES))
def test_prepared_dsa_run_matches_stock(
    workspace_init, dist_init, core, batch, q, overlap
):
    """PreparedDsa.run at TP (query stage, indexer, sparse core and MlaValue)
    against the stock KimiK3DSAAttention forward of the same decode step: the
    gated attention rows entering o_proj (and both against FP32 attention), the
    top-k, the MLA rows and the indexer keys (and their SM100 MX scales) the step
    inserts, and every other cache byte left as it was; serially and with the
    query forked off the indexer path (QueryNorm, then MlaQuery on a side stream)."""
    from vllm.utils.flashinfer import has_flashinfer_prims_ts_sparse_mla

    if LAYER_CORES[core][2] == "prims_ts" and not has_flashinfer_prims_ts_sparse_mla():
        pytest.skip("FlashInfer has no prims_ts sparse MLA")
    check_layer_step(run_layer_step(core, batch, q, overlap=overlap), q)


@pytest.mark.parametrize("batch,q", [(8, 1), (40, 5)])
@pytest.mark.parametrize("core", ["fi-trtllm", "fi-auto"])
def test_prepared_dsa_run_fp8_index_matches_stock(
    workspace_init, dist_init, core, batch, q
):
    """test_prepared_dsa_run_matches_stock with the FP8 indexer cache
    (indexer_kv_dtype='fp8', 132 B/token) instead of MXFP4."""
    step = run_layer_step(core, batch, q, index_dtype="fp8")
    assert step.caches[1].shape[2] == INDEX_ROW
    check_layer_step(step, q)


@pytest.mark.parametrize("batch,q", [(8, 8), (40, 8), (80, 8)])
@pytest.mark.parametrize("core", ["fi-auto"])
def test_prepared_dsa_q8_runs_stock_indexer(workspace_init, dist_init, core, batch, q):
    """DSpark k = 7 (q = 8) at TP on the MXFP4 indexer cache: the layer runs the
    stock decode indexer (DeepGEMM MXFP4 logits + top-k, faster there:
    prepared.FUSED_TP_MAX_Q) instead of the fused one, and matches stock."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.prepared import (
        FUSED_TP_MAX_Q,
    )

    assert q > FUSED_TP_MAX_Q
    step = run_layer_step(core, batch, q)
    assert step.caches[1].shape[2] == FP4_INDEX_ROW
    assert step.width is None  # no fused indexer ran
    check_layer_step(step, q)


# Past the step's 64 rows: the rows a K3.1 DSA layer's specialized projection and
# core take in a wide verify step (65-320 rows, up to providers.WIDE_ROWS): R16,
# R26 (130 rows: three IndexFrontMma row tiles), R32, R48 and R64 at q5.
WIDE_LAYER_SHAPES = [(80, 5), (130, 5), (160, 5), (240, 5), (320, 5)]


@pytest.mark.parametrize("overlap", ["serial", "fork", "front"])
@pytest.mark.parametrize("batch,q", WIDE_LAYER_SHAPES)
@pytest.mark.parametrize("core", ["fi-trtllm", "fi-prims_ts", "fi-auto"])
def test_prepared_dsa_run_wide_matches_stock(
    workspace_init, dist_init, core, batch, q, overlap
):
    """test_prepared_dsa_run_matches_stock at 80-320 rows (a wide step's DSA
    core): the specialized and the stock layer each against FP32 attention over
    its own top-k; the top-k exact up to its BF16 logits (the width check)."""
    from vllm.utils.flashinfer import has_flashinfer_prims_ts_sparse_mla

    if LAYER_CORES[core][2] == "prims_ts" and not has_flashinfer_prims_ts_sparse_mla():
        pytest.skip("FlashInfer has no prims_ts sparse MLA")
    check_layer_step(run_layer_step(core, batch, q, overlap=overlap), q)


@pytest.mark.parametrize("poison,offset", [("rows", 0), ("rows", 2048), ("counts", 0)])
@pytest.mark.parametrize("overlap", ["serial", "fork", "late", "late_front"])
@pytest.mark.parametrize("batch,q", [(5, 5), (60, 5)])
@pytest.mark.parametrize("core", ["fi-trtllm", "fi-prims_ts"])
def test_prepared_dsa_graph_waits_for_rows(
    workspace_init, dist_init, core, batch, q, overlap, poison, offset
):
    """PreparedDsa.run at TP replayed back to back in a CUDA graph, the fused
    indexer's outputs poisoned before every layer: as the stock forward, so no
    kernel reads the rows, their counts (or the query) before they are written
    (prims_ts' split-KV decode reads them before its PDL wait: the kernel before
    it must not launch it early). Poisoned rows (``offset`` 2048: every row past
    the select's short-context bypass), or only the counts at short contexts
    (rows below 2048 keys count fewer than 2048). Zero-count rows in a graph:
    test_flashinfer_sparse_core_tp."""
    from vllm.utils.flashinfer import has_flashinfer_prims_ts_sparse_mla

    if LAYER_CORES[core][2] == "prims_ts" and not has_flashinfer_prims_ts_sparse_mla():
        pytest.skip("FlashInfer has no prims_ts sparse MLA")
    args = dict(overlap=overlap, graph=True, poison=poison, offset=offset)
    step = run_layer_step(core, batch, q, **args)
    check_layer_step(step, q)


@pytest.mark.parametrize("overlap", list(TP_OVERLAPS))
@pytest.mark.parametrize("batch,q", [(15, 5), (35, 5), (64, 1)])
@pytest.mark.parametrize("core", ["fi-prims_ts", "fi-auto"])
def test_prims_ts_runs_on_the_layer_stream(
    workspace_init, dist_init, core, batch, q, overlap
):
    """Every prims_ts plan shares one workspace (attn_fi._Plans), which holds only
    while the decodes are stream ordered: PreparedDsa runs each on the stream the
    layer runs on (eager and captured), under every query overlap mode, never on
    the overlap side stream (run_layer_step checks each call)."""
    from vllm.utils.flashinfer import has_flashinfer_prims_ts_sparse_mla

    if not has_flashinfer_prims_ts_sparse_mla():
        pytest.skip("FlashInfer has no prims_ts sparse MLA")
    step = run_layer_step(core, batch, q, overlap=overlap, graph=True)
    # "auto": the policy's prims_ts batches (FI_KERNEL) run it.
    assert step.prims_ts_calls > 0


def check_prepare_allocates_nothing(core, world=1):
    """The specialized decode is built (load_model) before vLLM profiles its
    memory to size the KV cache, and prepares its cores for each bound KV cache
    (the profiling cache, then the real one) after: so PreparedDsa's constructor
    allocates every persistent buffer (the prims_ts plans' workspace, the fused
    indexer's logits, tops, select outputs and row ends, shared by every layer),
    and prepare_cache allocates nothing."""
    from vllm.config import set_current_vllm_config
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import (
        attn_fi,
        indexer,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.prepared import (
        PreparedDsa,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.providers import (
        dcp_box,
        pack_dsa_projection,
    )
    from vllm.utils.flashinfer import has_flashinfer_prims_ts_sparse_mla

    cache_dtype, _, kernel = LAYER_CORES[core]
    if kernel != "trtllm" and not has_flashinfer_prims_ts_sparse_mla():
        pytest.skip("FlashInfer has no prims_ts sparse MLA")
    q, batches = 5, (5, 10, 15, 20, 40, 60, 64)
    config = layer_config(core, q, world)
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    bf = torch.bfloat16
    with set_current_vllm_config(config):
        topk_buffer = torch.full((LAYER_LEN, 2048), -1, dtype=torch.int32, device=dev)
        attn = dsa_layer(config, topk_buffer, HEADS * world)
        layer, dsa_indexer = attn.mla_attn.mla_attn, attn.dsa_indexer
        pitch = pack_dsa_projection(attn).shape[0]
        cap = max(batches)
        scratch_rows = (
            torch.zeros(cap, 64, 576, device=dev, dtype=bf),
            torch.empty(cap, 1536, device=dev, dtype=bf),
            torch.empty(cap, 32 * 128, device=dev, dtype=bf),
            torch.empty(cap, 32, 128, device=dev, dtype=torch.uint8),
            torch.empty(cap, 32, device=dev, dtype=bf),
            fi_buffers(cap, 128),
        )
        policy: dict = dict(queries=(1, q))
        policy.update(fi_policy(attn, kernel, cache_dtype))
        if world > 1:
            runtime = SimpleNamespace(
                dcp=world,
                tp=world,
                device=dev,
                dcp_box=None,
                dsa_dcp_box=None,
                batches=batches,
                wide_batches=lambda kind: (),
            )
            box = dcp_box(runtime, HEADS, dsa=True, flashinfer=True, partials="bf16")
            policy.update(dcp=box)
        # A fresh process: no layer has reserved the shared buffers yet.
        indexer._SCRATCH.clear()
        indexer.select_outputs.cache_clear()
        attn_fi.PLANS.plans.clear()
        before = torch.cuda.memory_allocated(dev)
        pitches = dict.fromkeys(batches, pitch)
        prepared = PreparedDsa(attn, pitches, scratch_rows, **policy)
        assert torch.cuda.memory_allocated(dev) > before
        row = {"fp8": FP8_ROW, "nvfp4_ds_mla": NVFP4_ROW}[cache_dtype]
        gen = torch.Generator(device=dev).manual_seed(0)
        # The profiling run's small KV cache, then the real (larger) one (both
        # kept alive: releasing the first is not a change of prepare_cache's).
        caches = []
        for contexts in ([600] * 13, [33000, 9000, 4000] * 5):
            local = [-(-c // world) for c in contexts]
            mla, _ = paged_rows(local, UNIT, row, UNIT, gen)
            index, _ = paged_rows(local, UNIT, index_row(dsa_indexer), GAP, gen)
            caches.append((mla, index))
            layer.kv_cache, dsa_indexer.k_cache.kv_cache = mla, index
            gc.collect()  # earlier tests' garbage is not freed by prepare_cache
            bound = torch.cuda.memory_allocated(dev)
            prepared.prepare_cache(mla)
            torch.accelerator.synchronize()
            gc.collect()
            fused = prepared.fused if world == 1 else prepared.indexer_stage.fused
            assert fused is not None
            grown = torch.cuda.memory_allocated(dev) - bound
            assert grown == 0, f"prepare_cache allocated {grown} bytes"


@pytest.mark.parametrize("core", ["fi-auto", "fi-prims_ts", "fi-nvfp4"])
def test_prepare_cache_allocates_nothing(default_vllm_config, dist_init, core):
    """TP8: see check_prepare_allocates_nothing."""
    check_prepare_allocates_nothing(core)


# An 8-rank launch (2 nodes x 4 GPUs: srun -n8, one pytest per rank) sets the
# rendezvous file; the ranks then run the DCP8 layer steps together.
DCP_INIT = os.environ.get("KIMI_K3_SD_DCP_INIT")


# DCP8 layer cases: every FlashInfer core under every query overlap mode with BF16
# partials, and the W_UV-projected exchange on the default overlap policy.
DCP8_CASES = [
    *(
        (core, overlap, "bf16")
        for core in ("fi-trtllm", "fi-prims_ts", "fi-auto", "fi-nvfp4")
        for overlap in ("serial", "fork", "wait", "front")
    ),
    # W_UV-projected value rows ("proj": BF16, so checked as the lossless rows)
    *((core, None, "proj") for core in ("fi-trtllm", "fi-auto", "fi-nvfp4")),
]


@pytest.fixture(scope="module")
def dcp8_group():
    """TP = DCP = 8 over the launch's ranks (SLURM task ids), with real
    cross-node mailboxes for the specialized decode."""
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    local = int(os.environ["SLURM_LOCALID"])
    assert world == 8, "the DCP8 layer test runs on 8 ranks"
    torch.accelerator.set_device_index(local)
    with set_current_vllm_config(layer_config("fi-trtllm", 1, world)):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=local,
            distributed_init_method=f"file://{DCP_INIT}",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world, decode_context_model_parallel_size=world
        )
        init_workspace_manager(torch.device("cuda", local))
        # The exact DCP top-k merge's rescue workspace (once per process, as the
        # model's first DSA layer does): sized for the widest DSpark verify step
        # the tests run (64 requests x 5 = 320 rows).
        from vllm.v1.attention.ops.dcp_topk_merge import (
            dcp_topk_merge_max_rows,
            init_dcp_topk_merge,
        )

        wide = layer_config("fi-trtllm", 5, world)
        with set_current_vllm_config(wide):
            n = init_dcp_topk_merge(wide, 2048)
        assert n and dcp_topk_merge_max_rows(2048, n) >= 320
        yield world
        reset_workspace_manager()
    cleanup_dist_env_and_memory()


@pytest.mark.skipif(DCP_INIT is None, reason="needs an 8-rank launch (DCP8)")
@pytest.mark.skip_global_cleanup  # the DCP8 group outlives each case
@pytest.mark.parametrize("batch,q", [(1, 1), (8, 1), (5, 5), (40, 5), (60, 5)])
@pytest.mark.parametrize("core,overlap,partials", DCP8_CASES)
def test_prepared_dsa_run_dcp_matches_stock(
    dcp8_group, core, overlap, partials, batch, q
):
    """PreparedDsa.run under DCP8 (12 local heads of 96, interleave 1) with the
    real mailboxes across two nodes: the query publishes into every rank's q
    mailbox, the one-round DCP top-k merge (#63) gives this rank's rows of the
    global top-k, FlashInfer decodes the 96 gathered heads over the local shard
    and DcpValue merges this rank's heads, as the stock DCP forward of the same
    step on every rank (whose indexer runs the same merge): the gated attention
    rows entering o_proj, the global top-k and this rank's rows of it, the
    inserted rows of this rank's tokens and every other cache byte; with the q
    mailbox polled by MlaQuery ("serial"), or by QueryWait after the indexer
    ("wait") or on the forked query ("fork"). ``partials``: DcpValue's exchange
    (BF16 latent rows or "proj" value rows; both exact up to BF16 rounding)."""
    from vllm.utils.flashinfer import has_flashinfer_prims_ts_sparse_mla

    if LAYER_CORES[core][2] == "prims_ts" and not has_flashinfer_prims_ts_sparse_mla():
        pytest.skip("FlashInfer has no prims_ts sparse MLA")
    step = run_layer_step(core, batch, q, dcp8_group, overlap, partials)
    check_layer_step(step, q, reference=False)
    # The merge's rows of this rank: those of its global ids, as the stock filter
    # gives them (and those of the stock top-k on rows with the same set).
    merged, ours, stock = (
        [set(s[t, : v[t]].tolist()) for t in range(batch)] for s, v in step.owned
    )
    for t in range(batch):
        assert -1 not in merged[t] and merged[t] == ours[t], t
        if set(step.topk[t].tolist()) == set(step.stock_topk[t].tolist()):
            assert merged[t] == stock[t], t


@pytest.mark.skipif(DCP_INIT is None, reason="needs an 8-rank launch (DCP8)")
@pytest.mark.skip_global_cleanup  # the DCP8 group outlives each case
@pytest.mark.parametrize("overlap", ["serial", "fork", "wait", "front"])
@pytest.mark.parametrize("batch,q", WIDE_LAYER_SHAPES)
@pytest.mark.parametrize("core", ["fi-trtllm", "fi-prims_ts"])
def test_prepared_dsa_run_dcp_wide_matches_stock(dcp8_group, core, batch, q, overlap):
    """test_prepared_dsa_run_dcp_matches_stock at 80-320 rows (a wide DCP8 step's
    DSA core; past one row per SM the fused scorer feeds the merge's own local
    top-N; from 160 rows "proj" partials), every query overlap mode."""
    from vllm.utils.flashinfer import has_flashinfer_prims_ts_sparse_mla

    if LAYER_CORES[core][2] == "prims_ts" and not has_flashinfer_prims_ts_sparse_mla():
        pytest.skip("FlashInfer has no prims_ts sparse MLA")
    step = run_layer_step(core, batch, q, dcp8_group, overlap, "default")
    check_layer_step(step, q, reference=False)
    merged, ours, stock = (
        [set(s[t, : v[t]].tolist()) for t in range(batch)] for s, v in step.owned
    )
    for t in range(batch):
        assert -1 not in merged[t] and merged[t] == ours[t], t
        if set(step.topk[t].tolist()) == set(step.stock_topk[t].tolist()):
            assert merged[t] == stock[t], t


@pytest.mark.skipif(DCP_INIT is None, reason="needs an 8-rank launch (DCP8)")
@pytest.mark.skip_global_cleanup  # the DCP8 group outlives each case
@pytest.mark.parametrize("core", ["fi-auto", "fi-prims_ts", "fi-nvfp4"])
def test_prepare_cache_allocates_nothing_dcp(dcp8_group, core):
    """DCP8 (the fused DCP indexer's logits and B' rows): see
    check_prepare_allocates_nothing."""
    check_prepare_allocates_nothing(core, dcp8_group)


@pytest.mark.skipif(DCP_INIT is None, reason="needs an 8-rank launch (DCP8)")
@pytest.mark.skip_global_cleanup  # the DCP8 group outlives each case
@pytest.mark.parametrize("overlap", ["serial", "fork", "wait", "front"])
@pytest.mark.parametrize("batch,q", [(15, 5), (60, 5)])
def test_prims_ts_runs_on_the_layer_stream_dcp(dcp8_group, batch, q, overlap):
    """DCP8 (every batch on prims_ts): see test_prims_ts_runs_on_the_layer_stream."""
    from vllm.utils.flashinfer import has_flashinfer_prims_ts_sparse_mla

    if not has_flashinfer_prims_ts_sparse_mla():
        pytest.skip("FlashInfer has no prims_ts sparse MLA")
    step = run_layer_step("fi-prims_ts", batch, q, dcp8_group, overlap)
    assert step.prims_ts_calls > 0


def dcp_merge_vs_exact(world, saturate, repeats=1):
    """The specialized DCP decode's top-k merge (#63's dcp_topk_merge_rows, as
    IndexerStage.merge_rows calls it on DeepGEMM-layout logits) against the exact
    global top-2048 of the same scores. Every rank draws the same global scores
    (distinct up to BF16 ties) of 32K-token rows and holds its interleave-1
    shard; ``saturate``: rank 0's keys score above every other rank's, so it
    holds all 2048 of the global top-k (more than its N local candidates).
    Compares the multiset of the merged ids' scores with the exact top-2048's
    (exact selection up to ties), and each rank's rows with its owned ids.
    ``repeats``: merges the same scores that many times; every rank's rows must
    come out bit-identical, order included (the sparse decode sums in it)."""
    import torch.distributed as dist

    from vllm.config import set_current_vllm_config
    from vllm.distributed import get_dcp_group
    from vllm.v1.attention.ops import dcp_topk_merge as merge_pkg
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import dcp_topk_merge_rows

    topk, local_len, rows, block = 2048, 4096, 6, UNIT
    rank = get_dcp_group().rank_in_group
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    config = layer_config("fi-trtllm", 1, world)
    n = merge_pkg.dcp_merge_local_topk(topk, world)
    assert n == 384, n  # DCP8: 1.5x the fair share, warp-aligned
    with set_current_vllm_config(config):  # the rescue workspace (collective)
        assert merge_pkg.init_dcp_topk_merge(config, topk) == n
    assert merge_pkg.dcp_topk_merge_fits(rows, topk, n)
    gen = torch.Generator(device=dev).manual_seed(29)
    scores = torch.rand(rows, local_len * world, device=dev, generator=gen)
    if saturate:
        scores[:, 0::world] += 2  # rank 0 owns gid % world == 0
    scores = scores.bfloat16()
    # This rank's logits in DeepGEMM's layout (1 KiB-aligned BF16 rows).
    logits = scores[:, rank::world].contiguous()
    assert logits.stride(0) * 2 % 1024 == 0
    row_ends = torch.full((rows,), local_len, dtype=torch.int32, device=dev)
    req_id = torch.arange(rows, dtype=torch.int32, device=dev)
    pages = local_len // block
    table = torch.arange(rows * pages, dtype=torch.int32, device=dev)
    table = table.view(rows, pages)
    rows_out = torch.full((rows, topk), -7, dtype=torch.int32, device=dev)
    valid_out = torch.full((rows,), -7, dtype=torch.int32, device=dev)
    ids_out = torch.full((rows, topk), -7, dtype=torch.int32, device=dev)
    runs = []
    for _ in range(repeats):
        rows_out.fill_(-7)
        valid_out.fill_(-7)
        dcp_topk_merge_rows(
            logits=logits,
            row_ends=row_ends,
            max_local_len=local_len,
            local_topk=n,
            topk=topk,
            req_id=req_id,
            block_table=table,
            block_size=block,
            stride_rows=block,
            dcp_rank=rank,
            rows_out=rows_out,
            valid_out=valid_out,
            ids_out=ids_out,
        )
        runs.append((rows_out.clone(), valid_out.clone()))
    torch.accelerator.synchronize()
    for again, count in runs[1:]:
        assert torch.equal(count, runs[0][1])
        assert torch.equal(again, runs[0][0]), "the merged rows' order changed"
    exact = torch.topk(scores.float(), topk, dim=1)
    held = int((exact.indices[0] % world == 0).sum())
    assert (held > n) == saturate, held  # the construction is (not) saturated
    for t in range(rows):
        ids = ids_out[t].long()
        assert bool((ids >= 0).all()) and ids.unique().numel() == topk, t
        got = scores[t, ids].float().sort(descending=True).values
        torch.testing.assert_close(got, exact.values[t], atol=0, rtol=0)
        mine = ids[ids % world == rank] // world  # this rank's local positions
        assert int(valid_out[t]) == mine.numel(), t
        want = (table[t, mine // block].long() * block + mine % block).sort().values
        assert torch.equal(rows_out[t, : mine.numel()].long().sort().values, want), t
    dist.barrier(get_dcp_group().cpu_group)


@pytest.mark.skipif(DCP_INIT is None, reason="needs an 8-rank launch (DCP8)")
@pytest.mark.skip_global_cleanup  # the DCP8 group outlives each case
def test_dcp_topk_merge_exact_unsaturated(dcp8_group):
    """The merge the specialized DCP8 decode feeds selects exactly the global
    top-2048 when no rank holds more than its N = 384 local candidates of it
    (random scores: about 256 per rank)."""
    dcp_merge_vs_exact(dcp8_group, saturate=False)


@pytest.mark.skipif(DCP_INIT is None, reason="needs an 8-rank launch (DCP8)")
@pytest.mark.skip_global_cleanup  # the DCP8 group outlives each case
def test_dcp_topk_merge_exact_saturated(dcp8_group):
    """A saturated row (rank 0 holds all of the global top-2048, more than its N
    = 384 local candidates): the exact merge (#63: saturation detection, the
    owner's push and the rescue round) still selects the exact global top-2048,
    and rank 0's rows hold all 2048 (the specialized decode admits no
    approximate top-k)."""
    dcp_merge_vs_exact(dcp8_group, saturate=True)


@pytest.mark.skipif(DCP_INIT is None, reason="needs an 8-rank launch (DCP8)")
@pytest.mark.skip_global_cleanup  # the DCP8 group outlives each case
@pytest.mark.parametrize("saturate", [False, True])
def test_dcp_topk_merge_rows_deterministic(dcp8_group, saturate):
    """DCP8 runs to the same output every time: the merge's owned rows (the
    sparse decode's KV order, so its summation order) are the same rows in the
    same order on every run of the same scores, in the first round and in the
    rescue round of a saturated row."""
    dcp_merge_vs_exact(dcp8_group, saturate, repeats=8)


@pytest.mark.skip_global_cleanup  # pure policy check: keep a DCP8 launch's group
def test_dcp_partials_policy():
    """The DcpValue partials policy: a format for every batch ("bf16" or "proj"), a
    batch table of them or "default" (per layer kind: BF16, then "proj" from 65
    rows for MLA, from 160 rows for DSA); no lossy 8-bit partials."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import (
        partials_for,
        resolve_partials,
    )

    assert partials_for("bf16", 1) == partials_for("bf16", 64) == "bf16"
    table = "1-4:proj,20-64:proj"
    assert [partials_for(table, b) for b in (3, 8, 20)] == ["proj", "bf16", "proj"]
    assert partials_for("proj", 1) == "proj"
    assert [partials_for("1-9:bf16,10-64:proj", b) for b in (9, 10)] == [
        "bf16",
        "proj",
    ]
    mla, dsa = (resolve_partials("default", kind) for kind in ("mla", "dsa"))
    assert [partials_for(mla, b) for b in (1, 40, 64, 65, 320)] == [
        *("bf16",) * 3,
        *("proj",) * 2,
    ]
    assert [partials_for(dsa, b) for b in (1, 64, 159, 160, 320)] == [
        *("bf16",) * 3,
        *("proj",) * 2,
    ]
    assert resolve_partials(None, "dsa") == dsa
    assert resolve_partials("proj", "dsa") == "proj"
    for spec in ("fp8", "int8", "auto", "1-9:bf16,10-64:int8"):  # no 8-bit formats
        with pytest.raises((AssertionError, ValueError)):
            partials_for(spec, 10)


def dcp_value_reference(rows, lse, live, w_uv, gate, partials):
    """DcpValue's output from every rank's partial ``rows`` [DCP, B, H, 512] (FP32)
    of this rank's H heads, their base-2 ``lse`` [DCP, B, H] and ``live`` sources:
    the LSE-weighted merge ("bf16": of the BF16 rows), one BF16 rounding, W_UV
    [H, 512, 128] and the ``gate``. "proj": each source's rows through W_UV,
    rounded to BF16, then the merge and one BF16 rounding. "fp32": the merge and
    W_UV without intermediate rounding (what both BF16 exchanges approximate)."""
    if partials in ("proj", "fp32"):
        rounded = partials == "proj"
        lse = lse.masked_fill(~live, -torch.inf) * math.log(2)
        weights = torch.softmax(lse, dim=0).nan_to_num()
        rows = rows.masked_fill(~live.unsqueeze(-1), 0)
        values = torch.einsum("sbhk,hkv->sbhv", rows, w_uv.float())
        if rounded:
            values = values.bfloat16().float()
        value = (weights.unsqueeze(-1) * values).sum(0)
        if rounded:
            value = value.bfloat16().float()
        return (value * torch.sigmoid(gate)).flatten(1)
    assert partials == "bf16", partials
    lse = lse.masked_fill(~live, -torch.inf) * math.log(2)
    weights = torch.softmax(lse, dim=0).nan_to_num()
    rows = rows.masked_fill(~live.unsqueeze(-1), 0)
    merged = (weights.unsqueeze(-1) * rows).sum(0).bfloat16().float()
    value = torch.einsum("bhk,hkv->bhv", merged, w_uv.float()).bfloat16().float()
    return (value * torch.sigmoid(gate)).flatten(1)


# "proj" partials against the BF16 latent exchange (check_dcp_value): relative
# error to the unrounded FP32 merge at most this multiple of the BF16 exchange's
# (measured: at most 1.07x, both 0.30-0.36%), and relative difference to the BF16
# exchange's output below PROJ_VS_BF16_ERROR (measured: at most 0.39%, two
# independent BF16 roundings).
PROJ_VS_BF16_ACCURACY = 1.25
PROJ_VS_BF16_ERROR = 0.006


def check_dcp_value(policy, layout, batch, q, lse_base_e):
    """One DcpValue launch on every rank of the DCP8 group (real q / partial / LSE
    mailboxes across the nodes) against its torch reference. Each rank's partial
    rows and LSE are random; some local rows are empty (DSA: valid count 0; MLA: a
    zero local length), their rows and LSE garbage, and the last request is padding
    (global length 0). Every mailbox this rank read is re-armed afterwards, the
    padded q heads are untouched, and a second launch on the same parity agrees."""
    import cutlass
    import torch.distributed as dist

    from vllm.distributed import get_dcp_group
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        layout as spec,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.dcp_stage import (
        compile_dcp_value,
        publish_args,
        run_value,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import (
        DcpBox,
        compile_value,
        partials_for,
    )

    partials = partials_for(policy, batch)
    group = get_dcp_group()
    world, rank = group.world_size, group.rank_in_group
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    dsa, heads, reqs = layout == "dsa", HEADS, batch // q
    gathered = heads * world
    padded = 128 if dsa else gathered
    box = DcpBox(
        group.device_group, dev, world, rank, heads, q_heads=padded, partials=policy
    )
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    common = torch.Generator(device=dev).manual_seed(11)
    # W_UV [H, 512, 128] latent-contiguous, as MLA's processed weights.
    w_uv_all = torch.randn(gathered, 128, 512, device=dev, generator=common) / 23
    w_uv_all = w_uv_all.bfloat16().transpose(1, 2)
    w_uv = w_uv_all[rank * heads : (rank + 1) * heads]
    pitch = PITCH
    projected = torch.randn(batch, pitch, device=dev, generator=common).bfloat16()
    totals = torch.randint(1, 9000, (reqs,), device=dev, generator=common).int()
    totals[-1] = 0 if reqs > 1 else totals[-1]  # a padded request
    gen = torch.Generator(device=dev).manual_seed(100 + rank)
    # Partial rows: per-row magnitudes over several decades, some exact zeros.
    rows = torch.randn(batch, padded, 512, device=dev, generator=gen)
    rows *= torch.rand(batch, padded, 1, device=dev, generator=gen).pow(4) * 3
    rows[:, :, ::7] = 0
    rows = rows.bfloat16()
    lse = 4 * torch.rand(batch, padded, device=dev, generator=gen) + rank % 3
    if dsa:
        valid = torch.randint(0, 3, (batch,), device=dev, generator=gen).int()
        valid[: q * (rank % 2)] = 0  # odd ranks own none of request 0
        lens = valid
        live = (valid > 0).view(-1, 1)
    else:
        lens = torch.randint(0, 3, (reqs,), device=dev, generator=gen).int()
        live = (lens > 0).repeat_interleave(q).view(-1, 1)
    live = (live & (totals > 0).repeat_interleave(q).view(-1, 1)).expand(-1, padded)
    live = live.contiguous()
    # Rows not merged hold garbage: NaN rows, a NaN or infinite LSE.
    rows[~live] = torch.nan
    lse = lse.masked_fill(~live, torch.nan)
    lse[:, 0] = lse[:, 0].masked_fill(~live[:, 0], torch.inf)
    lse_e = lse_base_e and dsa
    kernel_lse = lse * math.log(2) if lse_e else lse
    out = torch.empty(batch, heads * 128, device=dev).bfloat16()
    uv, w_uv_proj = to_cute(w_uv), None
    if partials == "proj":  # every rank's heads, as DcpBox.gather_heads gives them
        uv_gathered = box.gather_heads(w_uv)
        assert uv_gathered.stride() == w_uv_all.stride()
        assert torch.equal(uv_gathered, w_uv_all)
        # run_value projects the rows through the torch W_UV (as PreparedDsa does).
        uv, w_uv_proj = to_cute(uv_gathered), uv_gathered

    if dsa:
        args = heads, world, batch, spec(w_uv), pitch, dev.index, sms, q, padded
        kernel = compile_dcp_value(*args, True, lse_e, partials)
    else:
        args = heads, world, batch, 64, spec(w_uv), pitch, dev.index, sms, q
        kernel = compile_value(*args, partials=partials)
        box.rows[:batch, 0].copy_(rows)
        box.rows_lse[:batch, 0].copy_(kernel_lse)

    def launch(parity):
        publish = publish_args(box, parity)
        box.q[parity].view(torch.uint8)[:batch, :gathered] = 7  # the core read it
        if dsa:
            core = rows, kernel_lse, valid
            args = kernel, box, parity, publish, heads, core, totals, uv
            run_value(*args, projected, out, w_uv_proj)
        else:
            mailbox, _, ids = publish
            v_args = (to_cute(box.rows[:batch]), to_cute(box.rows_lse[:batch], 4))
            mail = box.mailbox(batch)
            v_args += (to_cute(lens, 4), to_cute(mail.partial[parity]))
            v_args += (to_cute(mail.lse[parity], 4), to_cute(mail.peers, 8))
            v_args += (to_cute(mail.lse_peers, 8), mailbox, uv)
            v_args += (to_cute(projected[:, 2112:]), to_cute(totals, 4))
            v_args += (to_cute(out), cutlass.Int32(parity), ids, stream(dev))
            kernel(*v_args)
        torch.accelerator.synchronize()
        dist.barrier(group.cpu_group)
        return out.clone()

    first = launch(0)
    # Every rank's partials of this rank's heads.
    mine = slice(rank * heads, (rank + 1) * heads)
    gathered_inputs = []
    for x in (rows, lse, live):
        full = x.new_empty(world, *x.shape)
        dist.all_gather_into_tensor(full, x, group=group.device_group)
        gathered_inputs.append(
            full[:, :, mine].float() if x is rows else full[:, :, mine]
        )
    gate = projected[:, 2112 : 2112 + heads * 128].float().view(batch, heads, 128)
    padding = (totals == 0).repeat_interleave(q)

    def reference(fmt):
        ref = dcp_value_reference(*gathered_inputs, w_uv, gate, fmt)
        return ref.masked_fill(padding.view(-1, 1), 0).bfloat16().float()

    def check():
        assert torch.isfinite(first).all()
        ref = reference(partials)
        torch.testing.assert_close(first.float(), ref, atol=2e-3, rtol=2e-2)
        if partials == "proj":
            # BF16 value rows: no coarser than the BF16 latent exchange, only
            # rounded at other points (each source's value row instead of the
            # merged latent). Against the unrounded FP32 merge it is as accurate
            # as the BF16 exchange; against the BF16 exchange it differs by
            # about two independent BF16 roundings.
            ideal, exact = reference("fp32"), reference("bf16")
            ours = float((first.float() - ideal).norm() / ideal.norm())
            theirs = float((exact - ideal).norm() / ideal.norm())
            error = float((first.float() - exact).norm() / exact.norm())
            print(
                f"proj vs FP32 merge {ours:.5f} (BF16 exchange {theirs:.5f}), "
                f"vs BF16 exchange {error:.5f}"
            )
            assert ours <= PROJ_VS_BF16_ACCURACY * theirs, (ours, theirs)
            assert error < PROJ_VS_BF16_ERROR, error
        # Every mailbox this rank read is re-armed; the padded q heads untouched.
        sent = 0x80000000 - (1 << 32)  # BF16 pair (+0, -0), both formats
        mail = box.mailbox(batch)
        assert mail.format == partials
        assert (mail.partial[0].view(torch.int32) == sent).all()
        assert (mail.lse[0].view(torch.int32) == -1).all()
        q_box = box.q[0].view(torch.uint8)
        assert (q_box[:batch, :gathered] == 0xFF).all()
        assert (q_box[:, gathered:] == 0).all()

    def collective(fn):
        """``fn`` on every rank; all ranks raise if any failed (no rank is left
        waiting in the next launch)."""
        error = None
        try:
            fn()
        except AssertionError as e:
            error = e
        failed = torch.tensor([error is not None], device=dev, dtype=torch.int32)
        dist.all_reduce(failed, group=group.device_group)
        if error is not None:
            raise error
        assert not failed.item(), "another rank failed"

    collective(check)
    # The other parity, then this one again: the same output.
    launch(1)
    second = launch(0)
    collective(lambda: torch.testing.assert_close(second, first, atol=0, rtol=0))


@pytest.mark.skipif(DCP_INIT is None, reason="needs an 8-rank launch (DCP8)")
@pytest.mark.skip_global_cleanup  # the DCP8 group outlives each case
@pytest.mark.parametrize("batch,q", [(5, 5), (20, 5), (60, 5), (8, 1), (64, 1)])
@pytest.mark.parametrize("partials", ["bf16", "proj", "1-19:bf16,20-64:proj"])
@pytest.mark.parametrize(
    "layout,lse_base_e",
    [("dsa", True), ("dsa", False), ("mla", False)],
    ids=["dsa-base_e", "dsa-base2", "mla"],
)
def test_dcp_value(dcp8_group, layout, lse_base_e, partials, batch, q):
    """DcpValue on 8 ranks with the real mailboxes: the DSA layout (FlashInfer's
    128-padded rows and LSE in its base, per-row valid counts) and the MLA one
    (tokenspeed's 96 rows, local lengths), BF16 latent or "proj" value partials,
    or a batch table of both (two mailbox sets)."""
    check_dcp_value(partials, layout, batch, q, lse_base_e)


def layer_inputs(batch, fp8, seed):
    """Random packed projection rows, layer weights and caches of one DSA layer."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    bf, dev = torch.bfloat16, "cuda"

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=dev) * scale).to(bf)

    weights = (
        (torch.rand(1536, generator=gen, device=dev) + 0.5).to(bf),
        (torch.rand(512, generator=gen, device=dev) + 0.5).to(bf),
        randn(HEADS * 192, 1536, scale=1 / 40),
        randn(HEADS, 128, 512, scale=1 / 12),
        randn(HEADS, 128, 512).transpose(1, 2),
    )
    blocks = -(-(batch + 1) // UNIT) + 1
    mla = packed_cache(blocks, UNIT, FP8_ROW if fp8 else MLA_ROW)
    index = packed_cache(blocks, UNIT, INDEX_ROW)
    slots = torch.randperm(blocks * UNIT, generator=gen, device=dev)[:batch]
    slots[1::7] = -1  # padded rows (or other ranks' tokens) write nothing
    return randn(batch, PITCH), weights, mla, index, slots


@pytest.mark.parametrize("fp8", [False, True], ids=["fp8_ds_mla", "fp8"])
@pytest.mark.parametrize("batch", [80, 160, 320])
def test_wide_batches_match_64_row_kernels(batch, fp8):
    """Above 64 rows (64 requests x 5 verify tokens at most) the query, indexer
    input and value kernels compute every row as the 64-row kernels do: identical
    query rows, q_c, KV and indexer cache bytes, FP8 indexer query, head weights
    and gated outputs."""
    import cutlass

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.prepared import (
        gate_columns,
    )

    projected, weights, mla, index, slots = layer_inputs(batch, fp8, batch)
    dev, bf = "cuda", torch.bfloat16
    gen = torch.Generator(device=dev).manual_seed(1)
    iq = torch.randn(batch, 32 * 128, generator=gen, device=dev).to(bf)
    norm_w = torch.rand(128, generator=gen, device=dev) + 0.5
    norm_b = torch.randn(128, generator=gen, device=dev) / 10
    heads = HEADS if fp8 else 64
    latent = torch.randn(batch, heads, 512, generator=gen, device=dev).to(bf)
    lengths = torch.randint(0, 3, (batch,), generator=gen, device=dev).int()
    unit = torch.ones(1, device=dev, dtype=torch.float32)

    def run(rows, start, mla, index):
        """The kernels of ``rows`` rows over rows start.. (zero-padded past the
        end; padded rows have slot -1)."""
        n = min(rows, batch - start)

        def take(x, fill=0):
            out = x.new_full((rows, *x.shape[1:]), fill)
            out[:n] = x[start : start + n]
            return out

        query_kernel, index_kernel, value_kernel = kernels(
            rows, mla, index, *weights, fp8=fp8
        )
        x, s = take(projected), take(slots, -1)
        if fp8:
            query = torch.zeros(rows, HEADS, 576, device=dev, dtype=torch.uint8)
        else:
            query = torch.zeros(rows, 64, 576, device=dev, dtype=bf)
        q_c = torch.zeros(rows, 1536, device=dev, dtype=bf)
        args = (to_cute(x), *(to_cute(w) for w in weights[:4]), to_cute(mla))
        args += (to_cute(s, 8), to_cute(unit, 4), to_cute(unit, 4))
        call_query(query_kernel, args, EPS, query, q_c, weights, x)
        q_fp8 = torch.zeros(rows, 32, 128, device=dev, dtype=torch.uint8)
        head_w = torch.zeros(rows, 32, device=dev, dtype=bf)
        args = (to_cute(x), to_cute(take(iq)), to_cute(norm_w, 4), to_cute(norm_b, 4))
        args += (to_cute(index), to_cute(s, 8), to_cute(q_fp8), to_cute(head_w, 4))
        index_kernel(*args, cutlass.Float32(EPS), stream(dev))
        out = torch.zeros(rows, GATE, device=dev, dtype=bf)
        gate = to_cute(x[:, slice(*gate_columns(HEADS))])
        lat = to_cute(take(latent)[:, :HEADS])
        value_kernel(
            lat,
            to_cute(weights[4]),
            gate,
            to_cute(take(lengths), 4),
            to_cute(out),
            stream(dev),
        )
        return [t[:n] for t in (query, q_c, q_fp8, head_w, out)]

    wide_mla, wide_index = mla.clone(), index.clone()
    wide = run(batch, 0, wide_mla, wide_index)
    chunks = [run(64, start, mla, index) for start in range(0, batch, 64)]
    for ours, ref in zip(wide, (torch.cat(parts) for parts in zip(*chunks))):
        assert torch.equal(ours.view(torch.uint8), ref.view(torch.uint8))
    assert torch.equal(wide_mla, mla)
    assert torch.equal(wide_index, index)


def _dcp_mailbox_worker(rank, size, port, batch, q, dsa, partials="bf16"):
    """One DCP rank: MlaQuery publishes this rank's query heads into every rank's
    q mailbox, then DcpValue merges every rank's partial rows of this rank's heads
    (exchanged as ``partials``), checked against the all-gathered tensors."""
    import cutlass
    import torch.distributed as dist

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import (
        prepared as dsa_prepared,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla import (
        prepared as mla_prepared,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import DcpBox

    torch.accelerator.set_device_index(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=size,
        device_id=dev,
    )
    group, bf, f32 = dist.group.WORLD, torch.bfloat16, torch.float32
    q_heads = 64 if dsa else size * HEADS  # DSA: padded to the sparse decode's
    box = DcpBox(
        group, dev, size, rank, HEADS, cap=320, q_heads=q_heads, partials=partials
    )
    projected, weights, mla, index, slots = layer_inputs(batch, True, 100 + rank)
    weights = (*weights[:4], weights[4] / 16)  # W_UV outputs of unit scale
    assert weights[4].stride() == (128 * 512, 1, 512)
    unit = torch.ones(1, device=dev, dtype=f32)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    parity, s = 0, stream(dev)
    mailbox, mc = to_cute(box.q[parity]), box.mc(parity)
    ids = cutlass.Int32(parity), cutlass.Int32(rank)

    def query_and_value(rows):
        if dsa:
            specs = tuple(layout(w) for w in (*weights, mla, index))
            args = (rows, HEADS, PITCH, specs, rank, q, size, q_heads, sms)
            mq, _, value = dsa_prepared.compile_core(
                *args, fp8=True, cap=box.cap, partials=partials
            )
        else:
            specs = tuple(layout(w) for w in (*weights, unit, unit, mla))
            args = (rows, HEADS, PITCH, specs, rank, size, sms, q, box.cap)
            mq, value = mla_prepared.compile_core(*args, partials)
        return mq, value

    # The query: every rank's box holds every rank's query rows (bit-identical).
    mq, value = query_and_value(batch)
    query = torch.zeros(batch, HEADS, 576, device=dev, dtype=torch.uint8)
    q_c = torch.zeros(batch, 1536, device=dev, dtype=bf)
    x = projected if dsa else projected[:, :2112]
    args = (to_cute(x), *(to_cute(w) for w in weights[:4]), to_cute(mla))
    args += (
        to_cute(slots, 8),
        to_cute(unit, 4),
        to_cute(unit, 4),
        cutlass.Float32(EPS),
    )
    args += (to_cute(query),) + ((to_cute(q_c),) if dsa else ())
    mq(*args, mailbox, mc, ids[1], s)
    queries = [torch.empty_like(query) for _ in range(size)]
    dist.all_gather(queries, query)
    gathered = box.q[parity, :batch]
    assert torch.equal(gathered[:, : size * HEADS], torch.cat(queries, 1))
    assert (gathered[:, size * HEADS :] == 0).all()

    # The value: partial rows / LSEs of every rank's heads over the local shard.
    gen = torch.Generator(device=dev).manual_seed(rank)
    reqs = batch // q
    rows = torch.randn(batch, q_heads, 512, generator=gen, device=dev).to(bf)
    lse = torch.randn(batch, q_heads, generator=gen, device=dev) * 3
    lse[torch.rand(batch, q_heads, generator=gen, device=dev) < 0.05] = float("-inf")
    lens = torch.randint(0, 4, (reqs,), generator=gen, device=dev).int()
    lens_rows = lens.repeat_interleave(q)
    shared = torch.Generator(device=dev).manual_seed(7)  # the same on every rank
    totals = torch.randint(0, 3, (reqs,), generator=shared, device=dev).int()
    w_uv = weights[4]
    # "proj": every rank's W_UV (the layer providers gather it at construction).
    uv = to_cute(w_uv) if partials == "bf16" else box.value_weights(w_uv)[1]

    def merged(rows_n, start, n):
        out = torch.zeros(n, GATE, device=dev, dtype=bf)
        heads = size * HEADS
        view = rows[start : start + n].as_strided(
            (n, 1, heads, 512), (q_heads * 512, 0, 512, 1)
        )
        lse_view = lse[start : start + n].as_strided((n, 1, heads), (q_heads, 0, 1))
        r0, r1 = start // q, (start + n) // q
        # DSA: the sparse decode's per-row valid counts; MLA: per-request lengths.
        counts = lens_rows[start : start + n] if dsa else lens[r0:r1]
        mail = box.mailbox(n)
        v_args = (to_cute(view), to_cute(lse_view, 4), to_cute(counts, 4))
        v_args += (to_cute(mail.partial[parity]), to_cute(mail.lse[parity], 4))
        v_args += (to_cute(mail.peers, 8), to_cute(mail.lse_peers, 8), mailbox)
        gate = to_cute(projected[start : start + n, 2112 : 2112 + GATE])
        v_args += (uv, gate, to_cute(totals[r0:r1], 4), to_cute(out))
        rows_n(*v_args, *ids, s)
        torch.accelerator.synchronize()
        dist.barrier()
        return out

    out = merged(value, 0, batch)
    assert (box.q[parity, :batch, : size * HEADS].view(torch.int32) == -1).all()

    everything = [torch.empty_like(t) for t in (rows, lse, lens) for _ in range(size)]
    for i, t in enumerate((rows, lse, lens)):
        dist.all_gather(everything[i * size : (i + 1) * size], t)
    all_rows = torch.stack(everything[:size])[:, :, rank * HEADS : (rank + 1) * HEADS]
    all_lse = torch.stack(everything[size : 2 * size])[
        :, :, rank * HEADS : (rank + 1) * HEADS
    ]
    all_lens = torch.stack(everything[2 * size :]).repeat_interleave(q, 1)
    scale = 1.4426950408889634 if dsa else 1.0  # DSA (FlashInfer): base e
    live = (all_lens > 0).unsqueeze(-1) & torch.isfinite(all_lse)
    lse2 = torch.where(live, all_lse * scale, torch.full_like(all_lse, -3.0e38))
    weight = torch.exp2(lse2 - lse2.amax(0))
    weight = torch.where(live, weight, torch.zeros_like(weight))
    weight = weight / weight.sum(0).clamp_min(1e-30)
    latent = (weight.unsqueeze(-1) * all_rows.float()).sum(0).to(bf)
    value_ref = torch.einsum("bhk,hkv->bhv", latent.float(), w_uv.float()).to(bf)
    gate = projected[:, 2112 : 2112 + GATE].float().view(batch, HEADS, 128)
    ref = (value_ref.float() * torch.sigmoid(gate)).reshape(batch, GATE)
    ref[(totals.repeat_interleave(q) == 0)] = 0
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)

    # Every row as 40-row (8 requests) kernels compute it, bit for bit.
    _, chunk_value = query_and_value(40)
    chunks = [merged(chunk_value, start, 40) for start in range(0, batch, 40)]
    assert torch.equal(out, torch.cat(chunks))
    dist.destroy_process_group()


@pytest.mark.skipif(torch.accelerator.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize("dsa", [False, True], ids=["mla", "dsa_flashinfer"])
@pytest.mark.parametrize("batch", [40, 80, 160, 320])
@pytest.mark.parametrize("partials", ["bf16", "proj"])
def test_dcp_mailboxes_wide_batches(batch, dsa, partials):
    """DCP4 over one node's NVLink multicast at up to 320 rows (64 requests x 5
    verify tokens): the gathered q mailbox, DcpValue's LSE merge, W_UV and gate,
    and the mailboxes re-armed for the next layer."""
    import torch.multiprocessing as mp

    from vllm.utils.network_utils import get_open_port

    args = (4, get_open_port(), batch, 5, dsa, partials)
    mp.spawn(_dcp_mailbox_worker, args=args, nprocs=4)


def test_step_caps():
    """The specialized step takes batches up to the measured specialized-vs-stock
    crossover of every attention layer kind of the model, within what the step's
    providers serve; the DSA cores' own crossovers stay within the cores' rows."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.providers import (
        ATTENTION_MAX,
        PREPARED_MAX,
        SPECIALIZED_ROWS,
        step_cap,
    )

    assert set(SPECIALIZED_ROWS) == {(k, d) for k in ("mla", "dsa") for d in (0, 1)}
    assert all(0 < rows <= ATTENTION_MAX for rows in SPECIALIZED_ROWS.values())
    for dcp in (False, True):
        mla = SPECIALIZED_ROWS["mla", dcp]
        assert step_cap(False, dcp) == min(PREPARED_MAX, mla)
        dsa = min(mla, SPECIALIZED_ROWS["dsa", dcp])
        assert step_cap(True, dcp) == min(PREPARED_MAX, dsa)


def test_wide_caps():
    """Wide steps: each layer kind's projection and core run specialized up to its
    own crossover (DSA: the attention layer's, SPECIALIZED_ROWS), within the cores'
    rows (fixed in code: no deployment override)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.providers import (
        ATTENTION_MAX,
        KDA_WIDE_ROWS,
        SPECIALIZED_ROWS,
        WIDE_ROWS,
        wide_caps,
    )

    kinds = ("mla", "dsa")
    assert set(WIDE_ROWS) == {(k, d) for k in kinds for d in (0, 1)}
    rows = (*WIDE_ROWS.values(), *KDA_WIDE_ROWS.values())
    assert all(0 <= r <= ATTENTION_MAX for r in rows)
    # KDA: the RecoverSSM verify runs the specialized core at every wide row count.
    assert KDA_WIDE_ROWS[True] == ATTENTION_MAX
    for dcp in (False, True):
        assert WIDE_ROWS["dsa", dcp] == SPECIALIZED_ROWS["dsa", dcp]
        for recoverssm in (False, True):
            caps = wide_caps(dcp, recoverssm)
            expected = {k: WIDE_ROWS[k, dcp] for k in kinds}
            assert caps == {**expected, "kda": KDA_WIDE_ROWS[recoverssm]}


@pytest.mark.parametrize("pitch", [PITCH, 6288 // 8 * 8])
@pytest.mark.parametrize("batch", [80, 160, 240, 320])
def test_projection_wide_batches(pitch, batch):
    """The split-K input projection at the wide steps' rows (64-row N tiles) against
    a BF16 matmul of the same weight."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (
        PreparedProjection,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.providers import (
        projection_policies,
    )

    gen = torch.Generator(device="cuda").manual_seed(batch)
    weight = torch.randn(pitch, 7168, device="cuda", generator=gen).bfloat16()
    x = torch.randn(batch, 7168, device="cuda", generator=gen).bfloat16()
    policies = projection_policies(8, pitch, (batch,))
    assert policies[batch][1] == 64  # the 64-row N tile
    projection = PreparedProjection(weight, policies).prepare()
    out = torch.full((batch, pitch), float("nan"), device="cuda", dtype=torch.bfloat16)
    projection.run(x, out)
    ref = x.float() @ weight.float().T
    torch.testing.assert_close(out.float(), ref, atol=0.25, rtol=2e-2)


def test_front_overlap_rows():
    """The front query overlap serves 1-320 rows: the cluster IndexFront (it
    stages every row in shared memory) below FRONT_MMA_FROM rows and never past
    CLUSTER_FRONT_MAX (forcing it there is rejected), IndexFrontMma from there."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import (
        query_stage,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.overlap import (
        overlap_for,
    )

    assert overlap_for(320, "front") == "front"
    kinds = [query_stage.front_kernel(b) for b in range(1, 321)]
    assert set(kinds[: query_stage.FRONT_MMA_FROM - 1]) == {"cluster"}
    assert set(kinds[query_stage.FRONT_MMA_FROM - 1 :]) == {"mma"}
    saved = query_stage.FRONT_KERNEL
    query_stage.FRONT_KERNEL = "cluster"
    try:
        assert query_stage.front_kernel(query_stage.CLUSTER_FRONT_MAX) == "cluster"
        with pytest.raises(ValueError):
            query_stage.front_kernel(query_stage.CLUSTER_FRONT_MAX + 1)
    finally:
        query_stage.FRONT_KERNEL = saved


# --- Stock precision parity: bit for bit against the stock ops -------------------
# Inputs make every sum the two sides reduce in a different order exact (few
# mantissa bits, positive or sparse rows), so any difference is a rounding point.


def exact_rows(rows, cols, gen, *, signed=True, nonzero=None):
    """[rows, cols] BF16 values +-k/16 (k in 16..31): their squares sum exactly in
    any order. ``nonzero``: only that many leading columns are nonzero."""
    k = torch.randint(16, 32, (rows, cols), generator=gen, device="cuda")
    x = k.float() / 16
    if signed:
        x *= torch.randint(0, 2, (rows, cols), generator=gen, device="cuda") * 2 - 1
    if nonzero is not None:
        x[:, nonzero:] = 0
    return x.bfloat16()


@pytest.mark.parametrize("batch", [3, 64])
def test_query_norm_matches_stock(batch):
    """IDX-2 / ATTN-10 / MLA-QNORM-DIV: the q_a RMSNorm (QueryNorm, MlaQuery's own
    rows: test_query_overlap_kernels) is the stock layer's, bit for bit: vllm_c's
    rms_norm (IEEE divide)."""
    import cutlass

    from vllm import ir
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.query_stage import (  # noqa: E501
        compile_norm,
    )
    from vllm.platforms import current_platform

    current_platform.import_kernels()
    import vllm.kernels.vllm_c  # noqa: F401  (registers the vllm_c provider)

    gen = torch.Generator(device="cuda").manual_seed(batch)
    dev, bf = "cuda", torch.bfloat16
    norm = compile_norm(batch, HEADS, PITCH)
    q_norm = (torch.rand(1536, generator=gen, device=dev) + 0.5).to(bf)
    mismatched = 0
    for _ in range(4096 // batch):
        projected = torch.zeros(batch, PITCH, device=dev, dtype=bf)
        projected[:, :2048] = exact_rows(batch, 2048, gen)
        q_c = torch.zeros(batch, 1536, device=dev, dtype=bf)
        args = (to_cute(projected), to_cute(q_norm), to_cute(q_c))
        norm(*args, cutlass.Float32(EPS), stream(dev))
        q_a = projected[:, :1536]
        with ir.ops.rms_norm.set_priority(["vllm_c", "native"]):
            refs = [ir.ops.rms_norm(q_a.contiguous(), q_norm, EPS)]
        for ref in refs:
            mismatched += int((q_c.view(torch.int16) != ref.view(torch.int16)).sum())
    assert mismatched == 0


def test_fused_indexer_reservation_bounds(monkeypatch):
    """The fused indexer's TP logits are reserved (at construction) only where the
    fused indexer can run every prepared shape as far as is known before the KV
    cache exists (static_unsupported: top-k, rows and q), sized by the longest
    context the KV cache holds when its size is already known (kv_cache_blocks)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import (
        indexer,
        prepared,
    )
    from vllm.utils.deep_gemm import mqa_logits_use_mx_scales

    if not mqa_logits_use_mx_scales():
        pytest.skip("the fused indexer needs the SM100 MX-scaled indexer cache")
    layer = SimpleNamespace(topk_tokens=2048, max_model_len=1 << 20)
    assert indexer.static_unsupported(layer, 320, 5) is None
    assert indexer.static_unsupported(layer, 64, 1) is None
    for rows, q in ((320, 3), (9, 9), (64 * 9, 1)):
        assert indexer.static_unsupported(layer, rows, q) is not None, (rows, q)
    for topk in (1000, 8192):
        layer.topk_tokens = topk
        assert "top-k" in indexer.static_unsupported(layer, 64, 1)
    layer.topk_tokens = 2048

    import vllm.config

    cache = SimpleNamespace(num_gpu_blocks=None, num_gpu_blocks_override=None)
    cache.block_size = 2048
    config = SimpleNamespace(cache_config=cache)
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    assert prepared.kv_cache_blocks() is None
    cache.num_gpu_blocks_override = 10
    assert prepared.kv_cache_blocks() == 10 * 2048 // 64
    cache.num_gpu_blocks = 3
    assert prepared.kv_cache_blocks() == 3 * 2048 // 64

    dev = torch.device("cuda", torch.accelerator.current_device_index())
    for blocks, context in ((None, 1 << 20), (96, 96 * 64)):
        indexer._SCRATCH.clear()
        logits = indexer.reserve(layer, dev, 320, blocks=blocks)[0]
        assert logits.numel() == 320 * (-(-context // 128) * 128 + 128)
    indexer._SCRATCH.clear()
    indexer.select_outputs.cache_clear()


def tile_scales_kernel():
    """A kernel writing ds_tile_scale of each of 128 * n amax values."""
    import cutlass.cute as cute

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (  # noqa: E501
        ds_tile_scale,
        launch,
    )

    class TileScales:
        @cute.jit
        def __call__(self, *args):
            launch(self.kernel, args, args[0].shape[0] // 128, 128, pdl=False)

        @cute.kernel
        def kernel(self, args):
            amax, out = args
            i = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
            out[i] = ds_tile_scale(amax[i])

    return TileScales()


def tile_scales(amax):
    import cutlass.cute as cute

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )

    n = -(-amax.numel() // 128) * 128
    padded = torch.ones(n, device="cuda")
    padded[: amax.numel()] = amax.flatten()
    out = torch.zeros(n, device="cuda")
    args = to_cute(padded, 4), to_cute(out, 4), stream(amax.device)
    cute.compile(tile_scales_kernel(), *args)(*args)
    return out[: amax.numel()].view(amax.shape)


def test_ds_tile_scale_matches_stock_writers():
    """ATTN-9: the fp8_ds_mla tile scale is the stock writer's exp2f(ceilf(log2f(
    max(amax / 448, 1e-4)))): concat_and_cache_ds_mla (IEEE divide) over every
    finite non-negative BF16 amax."""
    from vllm import _custom_ops as ops

    dev, bf = "cuda", torch.bfloat16
    bits = torch.arange(0, 0x7F80, dtype=torch.int32, device=dev)
    amax = bits.to(torch.int16).view(bf)
    rows = -(-amax.numel() // 4)
    kv = torch.zeros(rows * 4, 128, device=dev, dtype=bf)
    kv[: amax.numel(), 0] = amax
    kv = kv.view(rows, 512)
    blocks = -(-rows // UNIT)
    cache = packed_cache(blocks, UNIT, MLA_ROW)
    slots = torch.arange(rows, device=dev)
    k_pe = torch.zeros(rows, 64, device=dev, dtype=bf)
    unit = torch.ones(1, device=dev)
    ops.concat_and_cache_mla(kv, k_pe, cache, slots, "fp8_ds_mla", unit)
    stock = cache.reshape(-1, MLA_ROW)[:rows, 512:528].contiguous().view(torch.float32)
    ours = tile_scales(amax.float())
    assert torch.equal(ours.view(torch.int32), stock.flatten()[: amax.numel()].view(
        torch.int32
    ))  # fmt: skip


@pytest.mark.parametrize("batch", [5, 64])
def test_fp8_cache_insert_divides_by_k_scale(batch):
    """ATTN-8: the DSA per-tensor FP8 KV row is the stock concat_and_cache_mla
    "fp8" row (each value divided by k_scale), bit for bit, with a k_scale that is
    not a power of two and k_pe values exactly halfway between two E4M3 codes after
    the divide (multiplying by the rounded inverse misses the tie)."""
    from vllm import _custom_ops as ops
    from vllm import ir
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        to_cute,
    )
    from vllm.platforms import current_platform

    current_platform.import_kernels()
    import vllm.kernels.vllm_c  # noqa: F401  (registers the vllm_c provider)

    gen = torch.Generator(device="cuda").manual_seed(batch)
    bf, dev = torch.bfloat16, "cuda"
    projected = torch.randn(batch, PITCH, generator=gen, device=dev).to(bf)
    projected[:, 1536:2048] = exact_rows(batch, 512, gen)
    # k_pe = m * k_scale for m halfway between two E4M3 codes, where m * k_scale
    # times the FP32 inverse of k_scale is not m.
    k_scale = torch.full((1,), 0.875, device=dev)
    ties = [0.78125, 0.90625, 1.5625, -1.8125, 3.125, -3.625, 0.390625, 0.453125]
    ties = torch.tensor(ties, device=dev)
    pick = torch.randint(0, ties.numel(), (batch, 64), generator=gen, device=dev)
    projected[:, 2048:2112] = (ties[pick] * 0.875).to(bf)
    assert torch.equal(projected[:, 2048:2112].float(), ties[pick] * 0.875)
    weights = (
        (torch.rand(1536, generator=gen, device=dev) + 0.5).to(bf),
        (torch.rand(512, generator=gen, device=dev) + 0.5).to(bf),
        (torch.randn(HEADS * 192, 1536, generator=gen, device=dev) / 40).to(bf),
        (torch.randn(HEADS, 128, 512, generator=gen, device=dev) / 12).to(bf),
        (torch.randn(HEADS, 128, 512, generator=gen, device=dev) / 12)
        .to(bf)
        .transpose(1, 2),
    )
    blocks = max(2, -(-batch // UNIT))
    mla = packed_cache(blocks, UNIT, FP8_ROW)
    index = packed_cache(2, UNIT, INDEX_ROW)
    slots = torch.randperm(blocks * UNIT, generator=gen, device=dev)[:batch]
    query_kernel, _, _ = kernels(batch, mla, index, *weights, fp8=True)
    query = torch.zeros(batch, HEADS, 576, device=dev, dtype=torch.uint8)
    q_c = torch.zeros(batch, 1536, device=dev, dtype=bf)
    unit = torch.ones(1, device=dev)
    args = (to_cute(projected), *(to_cute(w) for w in weights[:4]), to_cute(mla))
    args += (to_cute(slots, 8), to_cute(unit, 4), to_cute(k_scale, 4))
    call_query(query_kernel, args, EPS, query, q_c, weights, projected)

    stock = packed_cache(blocks, UNIT, FP8_ROW)
    with ir.ops.rms_norm.set_priority(["vllm_c", "native"]):
        kv = ir.ops.rms_norm(projected[:, 1536:2048].contiguous(), weights[1], EPS)
    k_pe = projected[:, 2048:2112].contiguous()
    ops.concat_and_cache_mla(kv, k_pe, stock, slots, "fp8", k_scale)
    ours, ref = (c[slots // UNIT, slots % UNIT] for c in (mla, stock))
    assert torch.equal(ours, ref)


def final_attn_res_inputs(tokens, nb, gen):
    """Final AttnRes inputs on 128 nonzero columns with positive values in [1, 2):
    every sum of squares and logit dot product is exact, and the prefix row's logit
    is below block 0's (a single softmax maximum on both sides)."""
    dev, bf, cols = "cuda", torch.bfloat16, 128

    def values(rows):
        k = torch.randint(128, 256, (*rows, 7168), generator=gen, device=dev)
        x = k.float() / 128
        x[..., cols:] = 0
        return x

    query_w = torch.zeros(7168, device=dev)
    query_w[:cols] = torch.randint(-4, 5, (cols,), generator=gen, device=dev) / 8
    blocks = values((tokens, nb))
    up, down = (query_w > 0)[:cols], (query_w < 0)[:cols]
    blocks[:, 0, :cols][:, up] = 1.875
    blocks[:, 0, :cols][:, down] = 1.0
    prefix = values((tokens,))
    prefix[:, :cols][:, up] = 1.0
    prefix[:, :cols][:, down] = 1.25
    delta = torch.zeros(tokens, 7168, device=dev)
    delta[:, :cols] = torch.randint(0, 64, (tokens, cols), generator=gen, device=dev)
    delta[:, :cols] /= 128  # prefix + delta stays in [1, 2) on 7 bits: exact
    norm_w = torch.ones(7168, device=dev)
    out_w = torch.rand(7168, generator=gen, device=dev) + 0.5
    return tuple(t.to(bf) for t in (blocks, prefix, delta, norm_w, query_w, out_w))


@pytest.mark.parametrize("nb", [1, 3])
def test_final_attn_res_matches_stock(nb):
    """NA-7: the last AttnRes and the backbone norm (applied in the specialized
    decode's last AttnRes) are the stock attn_res (no output norm: acc * rcp(s) in
    attn_res_kernel.cu's fast-math forms) followed by vllm_c's RMSNorm (IEEE
    divide), bit for bit, as is the updated prefix row."""
    from vllm import ir
    from vllm.models.kimi_k3.nvidia.ops.attn_res import attn_res
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.attn_res.prepared import (  # noqa: E501
        PreparedAttnRes,
    )
    from vllm.platforms import current_platform

    current_platform.import_kernels()
    import vllm.kernels.vllm_c  # noqa: F401  (registers the vllm_c provider)

    if not hasattr(torch.ops._C, "kimi_k3_attn_res"):
        pytest.skip("no native Kimi K3 AttnRes kernel in this build")
    tokens, eps = 2048, 1e-6
    gen = torch.Generator(device="cuda").manual_seed(nb)
    blocks, prefix, delta, norm_w, query_w, out_w = final_attn_res_inputs(
        tokens, nb, gen
    )
    stock_prefix = prefix.clone()
    mixed = attn_res(
        stock_prefix, delta, blocks, norm_w, query_w, None, nb, -1, eps, 0.0
    )
    with ir.ops.rms_norm.set_priority(["vllm_c", "native"]):
        assert ir.ops.rms_norm.dispatch(mixed, out_w, eps).provider == "vllm_c"
        ref = ir.ops.rms_norm(mixed, out_w, eps)

    key = nb, True, -1, True
    prepared = PreparedAttnRes((norm_w, query_w, out_w), key, (eps, eps))
    ours_prefix, out = prefix.clone(), torch.empty_like(prefix)
    prepared.run(delta, ours_prefix, blocks, out)
    assert torch.equal(ours_prefix, stock_prefix)
    differ = out.view(torch.int16) != ref.view(torch.int16)
    assert not differ.any(), f"{int(differ.sum())} of {differ.numel()} values differ"


@pytest.mark.parametrize("heads,batch", [(HEADS, 25), (128, 5)], ids=["tp", "dcp"])
def test_prims_ts_plan_matches_stock(default_vllm_config, dist_init, heads, batch):
    """ATTN-2: prims_ts runs a batch on the stock decode's plan (the smallest row
    capacity that holds it: its split-KV schedule fixes the BF16 partial
    roundings), so the latent and LSE are the stock _forward_prims_ts's, bit for
    bit, also where the specialized decode used to plan wider (21-31 TP rows on
    32, 3-19 DCP8 rows on 20)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.attn_fi import (
        FlashInferCore,
        plan_rows_for,
    )
    from vllm.utils.flashinfer import has_flashinfer_prims_ts_sparse_mla

    if not has_flashinfer_prims_ts_sparse_mla():
        pytest.skip("FlashInfer has no prims_ts sparse MLA")
    # The stock capacities with DSpark k=4 graph sizes 5..60 (as the recipes).
    capacities = tuple(sorted({1 << i for i in range(13)} | set(range(5, 61, 5))))
    assert plan_rows_for(batch, capacities) == batch
    impl = flashinfer_impl(heads, "prims_ts", default_vllm_config)
    impl.prims_ts_buckets = capacities
    impl.need_to_return_lse_for_decode = dcp = heads > 64
    layer = SimpleNamespace(_q_scale_float=1.0, _k_scale_float=1.0)
    gen = torch.Generator(device="cuda").manual_seed(batch)
    blocks = 300
    mla = packed_cache(blocks, UNIT, FP8_ROW, gap=UNIT)
    mla.copy_(random_fp8(*mla.shape, gen=gen))
    rows_total = blocks * UNIT
    query = random_fp8(batch, heads, FP8_ROW, gen=gen)
    valid = torch.randint(64, 2049, (batch,), generator=gen, device="cuda").int()
    flat = torch.randint(0, rows_total, (batch, 2048), generator=gen, device="cuda")
    flat = flat.int()
    # Flat rows of the packed blocks (a block stride of two blocks' rows).
    flat = flat // UNIT * 2 * UNIT + flat % UNIT
    buffers = fi_buffers(64, heads)
    fi = FlashInferCore(
        impl, layer, heads, buffers, {batch: "prims_ts"}, dcp=8 if dcp else 1,
        capacities=capacities,
    )  # fmt: skip
    q8 = query.view(torch.float8_e4m3fn)
    latent, lse = fi.decode(q8, mla, flat, valid, UNIT)
    out, stock_lse = impl._forward_prims_ts(q8, mla, flat, valid, layer, UNIT)
    assert torch.equal(latent.view(torch.int16), out.view(torch.int16))
    if dcp:
        assert torch.equal(lse.view(torch.int32), stock_lse.view(torch.int32))


def test_dsa_capability_gates():
    """IDX-3: the query and index fronts implement the K3.1 indexer: indexer RoPE or
    RMSNorm keys are refused with a reason (fp8_ds_mla: no FlashInfer kernel)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import (
        dsa_unsupported,
    )

    dsa = SimpleNamespace(index_rope_dim=0, index_k_norm_type="layernorm")
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(dsa_config=dsa)),
        cache_config=SimpleNamespace(cache_dtype="fp8_ds_mla"),
    )
    assert dsa_unsupported(config) is None
    for attr, value, expected in (
        ("index_rope_dim", 64, "RoPE"),
        ("index_k_norm_type", "rmsnorm", "LayerNorm"),
    ):
        saved = getattr(dsa, attr)
        setattr(dsa, attr, value)
        reason = dsa_unsupported(config)
        setattr(dsa, attr, saved)
        assert reason is not None and expected in reason, (attr, reason)


def check_dcp_value_merge_stock(layout, batch, q):
    """DcpValue's BF16 latent merge against the stock direct A2A combine
    (dcp_direct_a2a_lse_reduce: weight * (1 / sum), sources in rank order), bit for
    bit: two live sources per row (their sum is the same in either order) with
    fractional base-2 LSEs, W_UV picking latent column 4c into value column c and a
    gate of 30 (sigmoid 1), so DcpValue's output is its merged BF16 latent."""
    import cutlass
    import torch.distributed as dist

    from vllm.distributed import get_dcp_group
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        layout as spec,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.dcp_stage import (
        compile_dcp_value,
        publish_args,
        run_value,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import (
        DcpBox,
        compile_value,
    )
    from vllm.v1.attention.ops.dcp import DirectDCPA2AWorkspace

    group = get_dcp_group()
    world, rank = group.world_size, group.rank_in_group
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    dsa, heads, reqs = layout == "dsa", HEADS, batch // q
    gathered = heads * world
    padded = 128 if dsa else gathered
    box = DcpBox(group.device_group, dev, world, rank, heads, q_heads=padded)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    pick = torch.zeros(gathered, 128, 512, device=dev)
    pick[:, torch.arange(128), 4 * torch.arange(128)] = 1
    w_uv = pick.bfloat16().transpose(1, 2)[rank * heads : (rank + 1) * heads]
    projected = torch.zeros(batch, PITCH, device=dev, dtype=torch.bfloat16)
    projected[:, 2112 : 2112 + heads * 128] = 30
    totals = torch.full((reqs,), 4096, device=dev, dtype=torch.int32)
    gen = torch.Generator(device=dev).manual_seed(200 + rank)
    rows = torch.randn(batch, padded, 512, device=dev, generator=gen).bfloat16()
    lse = 6 * torch.rand(batch, padded, device=dev, generator=gen) - 3
    # Request r's live sources: ranks r % world and (r + 3) % world.
    owners = torch.arange(reqs, device=dev)
    owned = ((owners % world == rank) | ((owners + 3) % world == rank)).int()
    if dsa:
        valid = owned.repeat_interleave(q)
        live = (valid > 0).view(-1, 1)
    else:
        lens = owned
        live = (lens > 0).repeat_interleave(q).view(-1, 1)
    live = live.expand(-1, padded).contiguous()
    rows[~live] = torch.nan
    lse = lse.masked_fill(~live, torch.nan)
    out = torch.empty(batch, heads * 128, device=dev).bfloat16()
    uv = to_cute(w_uv)
    if dsa:
        args = heads, world, batch, spec(w_uv), PITCH, dev.index, sms, q, padded
        kernel = compile_dcp_value(*args, True, False, "bf16")
        publish = publish_args(box, 0)
        box.q[0].view(torch.uint8)[:batch, :gathered] = 7  # the core read it
        args = kernel, box, 0, publish, heads, (rows, lse, valid), totals, uv
        run_value(*args, projected, out)
    else:
        args = heads, world, batch, 64, spec(w_uv), PITCH, dev.index, sms, q
        kernel = compile_value(*args, partials="bf16")
        box.rows[:batch, 0].copy_(rows)
        box.rows_lse[:batch, 0].copy_(lse)
        mailbox, _, ids = publish_args(box, 0)
        box.q[0].view(torch.uint8)[:batch, :gathered] = 7
        mail = box.mailbox(batch)
        v_args = (to_cute(box.rows[:batch]), to_cute(box.rows_lse[:batch], 4))
        v_args += (to_cute(lens, 4), to_cute(mail.partial[0]))
        v_args += (to_cute(mail.lse[0], 4), to_cute(mail.peers, 8))
        v_args += (to_cute(mail.lse_peers, 8), mailbox, uv)
        v_args += (to_cute(projected[:, 2112:]), to_cute(totals, 4))
        v_args += (to_cute(out), cutlass.Int32(0), ids, stream(dev))
        kernel(*v_args)
    torch.accelerator.synchronize()
    dist.barrier(group.cpu_group)

    workspace = DirectDCPA2AWorkspace(group.device_group, dev, 64, heads, 512)
    partial = rows[:, :gathered].contiguous()
    stock = workspace.lse_reduce(partial, lse[:, :gathered].contiguous(), False)
    torch.accelerator.synchronize()
    ours = out.view(batch, heads, 128)
    ref = stock[:, :, ::4]
    differ = ours.view(torch.int16) != ref.view(torch.int16)
    error = None if not differ.any() else f"{int(differ.sum())} values differ"
    failed = torch.tensor([error is not None], device=dev, dtype=torch.int32)
    dist.all_reduce(failed, group=group.device_group)
    assert error is None, error
    assert not failed.item(), "another rank failed"


@pytest.mark.skipif(DCP_INIT is None, reason="needs an 8-rank launch (DCP8)")
@pytest.mark.skip_global_cleanup  # the DCP8 group outlives each case
@pytest.mark.parametrize("batch,q", [(20, 5), (64, 1)])
@pytest.mark.parametrize("layout", ["dsa", "mla"])
def test_dcp_value_merge_matches_stock(dcp8_group, layout, batch, q):
    """ATTN-5 / MLA-DCP-MERGE-WEIGHT: DcpValue's merge weights are the stock
    combine's weight * (1 / sum) (check_dcp_value_merge_stock)."""
    check_dcp_value_merge_stock(layout, batch, q)


@pytest.mark.parametrize("overlap", ["fork", "front"])
@pytest.mark.parametrize("core", ["fi-trtllm", "fi-prims_ts"])
def test_prepared_dsa_compiles_once_per_kernel(
    default_vllm_config, workspace_init, dist_init, monkeypatch, core, overlap
):
    """Startup compiles: two DSA layers over caches of the same layout compile each
    CuTe kernel once (the cores, the fused indexer and the overlap kernels are
    shared by every layer), and of a memory profiling cache and the real one only
    the real one is prepared (SpecializedDecode.prepare_kv): every cute.compile is a
    distinct kernel."""
    import cutlass.cute as cute

    from vllm.config import set_current_vllm_config
    from vllm.models.kimi_k3.nvidia.model_state import KimiK3ModelState
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import (
        indexer,
        prepared,
        query_stage,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.decode import (
        SpecializedDecode,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.providers import (
        pack_dsa_projection,
    )
    from vllm.utils.flashinfer import has_flashinfer_prims_ts_sparse_mla
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    if LAYER_CORES[core][2] == "prims_ts" and not has_flashinfer_prims_ts_sparse_mla():
        pytest.skip("FlashInfer has no prims_ts sparse MLA")
    for memo in (
        prepared.compile_core,
        query_stage.compile_norm,
        query_stage.compile_wait,
        query_stage._compile_front,
        indexer.compile_indexer,
    ):
        memo.cache_clear()
    compiles: list = []

    class Counting:  # cute.compile, counted (cute.compile[...] passes through)
        def __init__(self, compile_):
            self.compile_ = compile_

        def __call__(self, kernel, *args, **kwargs):
            compiles.append(type(kernel).__name__)
            return self.compile_(kernel, *args, **kwargs)

        def __getitem__(self, key):
            return self.compile_[key]

    monkeypatch.setattr(cute, "compile", Counting(cute.compile))
    config = layer_config(core, 5)
    dev, bf = torch.device("cuda"), torch.bfloat16
    batches = (5, 8, 10, 40)
    cap = max(batches)
    layers = []
    for _ in range(2):  # each its own config (both are named layers.0)
        layer_cfg = layer_config(core, 5)
        with set_current_vllm_config(layer_cfg):
            topk = torch.full((LAYER_LEN, 2048), -1, dtype=torch.int32, device=dev)
            attn = dsa_layer(layer_cfg, topk)
            pitch = pack_dsa_projection(attn).shape[0]
            scratch_rows = (
                torch.zeros(cap, 64, 576, device=dev, dtype=bf),
                torch.empty(cap, 1536, device=dev, dtype=bf),
                torch.empty(cap, 32 * 128, device=dev, dtype=bf),
                torch.empty(cap, 32, 128, device=dev, dtype=torch.uint8),
                torch.empty(cap, 32, device=dev, dtype=bf),
                fi_buffers(cap, 128),
            )
            pitches = dict.fromkeys(batches, pitch)
            args = attn, pitches, scratch_rows
            dsa = prepared.PreparedDsa(*args, queries=(1, 5), overlap=overlap)
            layers.append((attn, dsa))
    with set_current_vllm_config(config):
        compiles.clear()  # the FlashInfer plans of construction aside
        per_layer: list = []

        def prepare(blocks):
            # One KV cache: every layer's view of it has the same layout.
            for attn, dsa in layers:
                index = packed_cache(blocks, UNIT, INDEX_ROW)
                attn.dsa_indexer.k_cache.kv_cache = index
                dsa.prepare_cache(packed_cache(blocks, UNIT, FP8_ROW))
                per_layer.append(len(compiles))

        # prepare_kv prepares only the real cache (the worker records its block
        # count first); the profiling cache is bound unprepared.
        monkeypatch.setattr(
            MambaHybridModelState, "initialize_kv_cache", lambda *a: None
        )
        bound: list = []
        prepare_layout = lambda caches: (bound.append("prepared"), prepare(16))
        cache_config = SimpleNamespace(num_gpu_blocks=None)
        decode = SpecializedDecode.__new__(SpecializedDecode)
        decode.cache_config, decode.kv_prepared = cache_config, True
        decode.layers = (SimpleNamespace(layer_idx=0, self_attn=None),)
        decode.attention = {0: SimpleNamespace(prepare_layout=prepare_layout)}
        state = KimiK3ModelState.__new__(KimiK3ModelState)
        state.language_model = SimpleNamespace(specialized_decode=decode)
        initialize = KimiK3ModelState.initialize_kv_cache
        initialize(state, SimpleNamespace(num_blocks=4), None)  # profiling
        assert not bound and not decode.kv_prepared and not compiles
        cache_config.num_gpu_blocks = 16  # gpu_worker.initialize_from_config
        initialize(state, SimpleNamespace(num_blocks=16), None)
        assert bound == ["prepared"] and decode.kv_prepared
        # The second layer compiles nothing: no kernel compiled twice.
        first = per_layer[0]
        assert first > 0 and per_layer == [first, first], (per_layer, compiles)
        assert (
            "QueryNorm" in compiles
            or "IndexFront" in compiles
            or ("IndexFrontMma" in compiles)
        ), compiles
    # Each layer holds the same kernel objects.
    (_, a), (_, b) = layers
    assert a.kernels.keys() == b.kernels.keys()
    for shape, kernels in a.kernels.items():
        assert all(x is y for x, y in zip(kernels, b.kernels[shape])), shape
    for shape, kernels in a.overlap_kernels.items():
        assert all(x is y for x, y in zip(kernels, b.overlap_kernels[shape])), shape
