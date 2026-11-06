# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode with the MXFP4 DSA indexer cache (68 B/token,
indexer_kv_dtype="mxfp4" on SM100): the decode's FP4 writers against the stock writers
(bit-exact), and the fused FP4 scorer against DeepGEMM's FP4 paged MQA logits
(bit-exact: the same block-scaled MXF4 MMA and BF16 head sum), with the exact
top-k of its own logits (TP) and the DCP logits layout and local select."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels need SM100/SM103",
)

UNIT, MLA_ROW, FP4_ROW = 64, 656, 68  # indexer page tokens; cache row bytes
GAP = 36  # rows of the other layers' pages between two blocks of a packed cache
HEADS = 12  # TP8 attention heads
GATE = HEADS * 128
PITCH = 2112 + GATE + 128 + 32  # packed [q_a | kv_a | gate | wk | wv] row
KEY = 2112 + GATE  # wk column offset in the packed row
EPS = 1e-6
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def packed_cache(blocks, block, row, gap=GAP):
    stride = (block + gap) * row
    raw = torch.zeros(blocks * stride, dtype=torch.uint8, device="cuda")
    return torch.as_strided(raw, (blocks, block, row), (stride, row, 1))


def pages(index):
    """Each indexer page's bytes: [64 x 64 B E2M1 | 64 x 4 UE8M0]."""
    return torch.as_strided(
        index, (index.shape[0], index.shape[1] * FP4_ROW), (index.stride(0), 1)
    )


def dequant(codes, scales):
    """E2M1 bytes [..., 64] (low nibble first) and UE8M0 bytes [..., 4] ->
    FP32 [..., 128]."""
    table = torch.tensor(E2M1, device=codes.device)
    table = torch.cat([table, -table])
    nib = torch.stack([codes & 15, codes >> 4], -1).flatten(-2).long()
    values = table[nib].unflatten(-1, (4, 32))
    scale = torch.exp2(scales.float() - 127).unsqueeze(-1)
    return (values * scale).flatten(-2)


def kernels(batch, mla, index, *weights):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        layout,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.prepared import (
        compile_core,
    )

    specs = tuple(layout(w) for w in (*weights, mla, index))
    device = torch.accelerator.current_device_index()
    return compile_core(batch, HEADS, PITCH, specs, device, 1)


def adversarial_heads(iq):
    """Heads whose block maxima sit at and next to 6 x 2^k (where the stock
    writer's ceil(log2(amax / 6)) changes) and an all-zero head."""
    rows = iq.view(iq.shape[0], 32, 4, 32)
    for k, head in enumerate(range(1, 9)):
        base = 6.0 * 2.0 ** (k - 4)
        rows[:, head] *= 0.05 * 2.0 ** (k - 4)  # the planted value is the max
        for b, ulps in enumerate((0, 1, -1, 2)):
            value = torch.tensor(base, dtype=torch.bfloat16)
            bits = value.view(torch.int16) + ulps
            rows[:, head, b, 7] = bits.view(torch.bfloat16).item()
    rows[0, 0] = 0


@pytest.mark.parametrize("gap", [0, GAP])
@pytest.mark.parametrize("batch", [1, 5, 64, 320])
def test_fp4_indexer_inputs(batch, gap):
    """DsaIndex on an MXFP4 indexer cache writes the stock writers' bytes:
    mxfp4_q_quant's E2M1 query and UE8M0 scales (in the query rows'
    FP4 layout) and its BF16 head weights, and mxfp4_k_norm_quant_insert's
    MXFP4 key bytes and scales (the stock writer's log2 rounding included)."""
    import cutlass

    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import (
        mxfp4_k_norm_quant_insert,
        mxfp4_q_quant,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.index import (
        fp4_query_views,
    )

    torch.manual_seed(batch)
    bf, dev = torch.bfloat16, "cuda"
    projected = torch.randn(batch, PITCH, device=dev, dtype=bf)
    projected[:, KEY : KEY + 128] *= torch.logspace(-3, 3, batch, device=dev).to(bf)[
        :, None
    ]
    iq = torch.randn(batch, 32 * 128, device=dev, dtype=bf)
    adversarial_heads(iq)
    norm_w = torch.rand(128, device=dev) + 0.5
    norm_b = torch.randn(128, device=dev) / 10
    mla = packed_cache(8, UNIT, MLA_ROW)
    blocks = -(-batch // UNIT) + 8
    index = packed_cache(blocks, UNIT, FP4_ROW, gap)
    slots = torch.randperm(blocks * UNIT - 1, device=dev)[:batch]
    slots[0] = blocks * UNIT - 1
    if batch > 2:
        slots[1] = -1  # a padded row (or a token another DCP rank owns)
    weights = (
        torch.ones(1536, device=dev, dtype=bf),
        torch.ones(512, device=dev, dtype=bf),
        torch.zeros(HEADS * 192, 1536, device=dev, dtype=bf),
        torch.zeros(HEADS, 128, 512, device=dev, dtype=bf),
        torch.zeros(HEADS, 128, 512, device=dev, dtype=bf).transpose(1, 2),
    )
    _, index_kernel, _ = kernels(batch, mla, index, *weights)
    q_rows = torch.full((batch, 32, 128), 0x77, device=dev, dtype=torch.uint8)
    head_w = torch.zeros(batch, 32, device=dev, dtype=bf)
    args = (to_cute(projected), to_cute(iq), to_cute(norm_w, 4), to_cute(norm_b, 4))
    args += (to_cute(index), to_cute(slots, 8), to_cute(q_rows), to_cute(head_w, 4))
    index_kernel(*args, cutlass.Float32(EPS), stream(dev))

    ref_q, ref_s, ref_w = mxfp4_q_quant(
        iq.view(batch, 32, 128),
        projected[:, KEY + 128 : KEY + 160],
        128**-0.5 * 32**-0.5,
        32,
    )
    packed, scales = fp4_query_views(q_rows, batch)
    assert torch.equal(packed, ref_q)
    assert torch.equal(scales, ref_s)
    # The stock decode casts the FP32 weights to BF16 for SM100's MQA logits.
    assert torch.equal(head_w, ref_w.to(bf))

    stock = packed_cache(blocks, UNIT, FP4_ROW, gap)
    topk = torch.empty(batch, 2048, device=dev, dtype=torch.int32)
    k = projected[:, KEY : KEY + 128]
    mxfp4_k_norm_quant_insert(k, norm_w, norm_b, EPS, slots, stock, topk, batch)
    assert torch.equal(pages(index), pages(stock))


FRONT_BATCHES = [1, 16, 64, 80, 320]


@pytest.mark.parametrize("kernel_kind", ["mma", "mma-qc", "cluster"])
@pytest.mark.parametrize("batch", FRONT_BATCHES)
def test_fp4_index_front(batch, kernel_kind):
    """The fused index fronts on an MXFP4 indexer cache: the key bytes exactly
    the stock insert's, the head weights exactly bf16(wv / 64), and the FP4
    query as mxfp4_q_quant of the front's own projection: its codes
    equal the stock writer's of an FP32 projection of the same q_c except where
    the BF16 projection rounds differently (dequantized within one E2M1 step of
    the block's scale)."""
    import cutlass

    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import (
        mxfp4_k_norm_quant_insert,
        mxfp4_q_quant,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import (
        query_stage,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.index import (
        fp4_query_views,
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
    blocks = -(-batch // UNIT) + 8
    index = packed_cache(blocks, UNIT, FP4_ROW, GAP)
    slots = torch.randperm(blocks * UNIT - 1, device=dev)[:batch]
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
    q_rows = torch.zeros(batch, 32, 128, device=dev, dtype=torch.uint8)
    head_w = torch.zeros(batch, 32, device=dev, dtype=bf)
    var = projected[:, :1536].float()
    q_c = var * torch.rsqrt(var.pow(2).mean(-1, keepdim=True) + EPS)
    q_c = (q_c * q_norm.float()).to(bf)
    args = (to_cute(projected),)
    weights = to_cute(wq)
    if kernel_kind != "cluster":
        args += (to_cute(q_c) if normalized else to_cute(projected),)
        wq_map = weight_map(wq, 128)
        weights = cutlass.Int64(wq_map.data_ptr())
    args += (to_cute(q_norm), weights)
    args += (to_cute(norm_w, 4), to_cute(norm_b, 4), to_cute(index))
    args += (to_cute(slots, 8), to_cute(q_rows), to_cute(head_w, 4))
    kernel(*args, cutlass.Float32(EPS), cutlass.Float32(EPS), stream(dev))

    wv = projected[:, KEY + 128 : KEY + 160]
    assert torch.equal(head_w, (wv.float() / 64).to(bf))
    iq = (q_c.float() @ wq.float().T).to(bf)
    ref_q, ref_s, _ = mxfp4_q_quant(iq.view(batch, 32, 128), wv, 1 / 64, 32)
    packed, scales = fp4_query_views(q_rows, batch)
    ours = dequant(packed, scales.view(torch.uint8).view(batch, 32, 4))
    theirs = dequant(ref_q, ref_s.view(torch.uint8).view(batch, 32, 4))
    step = torch.exp2(
        torch.maximum(
            scales.view(torch.uint8).view(batch, 32, 4),
            ref_s.view(torch.uint8).view(batch, 32, 4),
        ).float()
        - 127
    )
    bound = 2 * step.repeat_interleave(32, -1)  # one E2M1 step at the top binade
    assert bool(((ours - theirs).abs() <= bound).all())
    assert (packed != ref_q).float().mean() < 0.05
    assert (scales != ref_s).float().mean() < 0.01

    stock = packed_cache(blocks, UNIT, FP4_ROW, GAP)
    topk = torch.empty(batch, 2048, device=dev, dtype=torch.int32)
    k = projected[:, KEY : KEY + 128]
    mxfp4_k_norm_quant_insert(k, norm_w, norm_b, EPS, slots, stock, topk, batch)
    assert torch.equal(pages(index), pages(stock))


def order16(bits):
    bits = bits & 0xFFFF
    return torch.where(bits & 0x8000 != 0, bits ^ 0xFFFF, bits ^ 0x8000)


def exact_topk(bits, k):
    """Key indices of the top ``k`` of BF16 ``bits`` (ties to the lowest index),
    in key order."""
    pos = torch.arange(bits.shape[0], device=bits.device)
    key = order16(bits.int()).long() << 32 | (2**32 - 1 - pos)
    return torch.topk(key, min(k, bits.shape[0])).indices.sort().values


def fp4_inputs(contexts, q, seed, gap=GAP):
    """A packed MXFP4 indexer cache of random E2M1 codes and UE8M0 scales under a
    random block table, per-row ends (row j of a request sees context + 1 + j
    keys), the MXFP4 query in the query rows' layout and BF16 head weights."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.index import (
        fp4_query_views,
    )

    gen = torch.Generator(device="cuda").manual_seed(seed)
    need = [-(-(c + q) // UNIT) for c in contexts]
    blocks = sum(need) + 3
    index = packed_cache(blocks, UNIT, FP4_ROW, gap)
    page = pages(index)
    page[:, : UNIT * 64] = torch.randint(
        0, 256, (blocks, UNIT * 64), device="cuda", generator=gen, dtype=torch.uint8
    )
    page[:, UNIT * 64 :] = torch.randint(
        118, 136, (blocks, UNIT * 4), device="cuda", generator=gen
    ).to(torch.uint8)
    perm = torch.randperm(blocks, device="cuda", generator=gen).int()
    table = torch.zeros(len(contexts), max(need), dtype=torch.int32, device="cuda")
    used = 0
    for r, n in enumerate(need):
        table[r, :n] = perm[used : used + n]
        used += n
    ends = [c + 1 + j for c in contexts for j in range(q)]
    ends = torch.tensor(ends, dtype=torch.int32, device="cuda")
    rows = max(64, len(contexts) * q)
    q_rows = torch.zeros(rows, 32, 128, dtype=torch.uint8, device="cuda")
    packed, scales = fp4_query_views(q_rows, rows)
    packed.copy_(
        torch.randint(0, 256, packed.shape, device="cuda", generator=gen).to(
            torch.uint8
        )
    )
    exps = torch.randint(118, 136, (rows, 32, 4), device="cuda", generator=gen)
    scales.copy_(exps.to(torch.uint8).view(torch.int32).view(rows, 32))
    weights = (torch.randn(rows, 32, device="cuda", generator=gen) * 0.1).bfloat16()
    return SimpleNamespace(
        index=index, table=table, ends=ends, q_rows=q_rows, weights=weights
    )


def deepgemm_logits(x, rows, q, max_len):
    """Stock's decode logits (DeepGEMM SM100 FP4 paged MQA logits, varlen as the
    indexer metadata builds them): BF16 [rows, max_len]."""
    from vllm.model_executor.layers.sparse_attn_indexer import (
        kv_cache_as_quant_view,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.index import (
        fp4_query_views,
    )
    from vllm.utils.deep_gemm import (
        fp8_fp4_paged_mqa_logits,
        get_num_sms,
        get_paged_mqa_logits_metadata,
        mqa_logits_mx_inputs,
    )

    packed, scales = fp4_query_views(x.q_rows, rows)
    query = packed.contiguous().view(torch.int8).view(rows, 1, 32, 64)
    q_scale = scales.contiguous().view(rows, 1, 32)
    lens = x.ends[:rows].view(rows, 1).contiguous()
    table = x.table.repeat_interleave(q, 0)[:rows].contiguous()
    indices = torch.arange(rows, dtype=torch.int32, device="cuda")
    schedule = get_paged_mqa_logits_metadata(lens, UNIT, get_num_sms(), indices)
    q_scale, weights = mqa_logits_mx_inputs(
        query, q_scale, x.weights[:rows], torch.bfloat16
    )
    return fp8_fp4_paged_mqa_logits(
        (query, q_scale),
        kv_cache_as_quant_view(x.index, 128, True),
        weights,
        lens,
        table,
        schedule,
        max_model_len=max_len,
        logits_dtype=torch.bfloat16,
        indices=indices,
    )


def same_bits(got, ref):
    """BF16 bits equal (int16 tensors), -0 as +0."""
    got, ref = got.int() & 0xFFFF, ref.int() & 0xFFFF
    got = torch.where(got == 0x8000, 0, got)
    ref = torch.where(ref == 0x8000, 0, ref)
    return torch.equal(got, ref)


FP4_TP_CASES = {
    "short": ([1500, 300], 5),
    "boundary": ([2046, 2047, 2048, 2049], 1),
    "mixed": ([5000, 300, 9000, 33000], 5),
    "long": ([100000, 70000, 3000, 130000], 5),
    "q1_64": ([3000 + 977 * r for r in range(64)], 1),
    "padded": ([40000, 3000, 0], 5),
    "q7": ([6000, 20000, 2500], 7),
    "q8": ([6000, 20000, 2500], 8),
    "wide320": ([3000 + 257 * r for r in range(64)], 5),
}


@pytest.mark.parametrize("case", list(FP4_TP_CASES))
def test_fused_fp4_indexer_tp(case):
    """The fused decode indexer (TP) on MXFP4 keys: its BF16 logits bit-exact
    DeepGEMM's FP4 paged MQA logits (the stock decode's), and its selection
    exactly the top-2048 of them (ties to the lowest key), as the MLA cache's
    physical rows in key order with the counts."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.indexer import (
        FusedIndexer,
    )

    contexts, q = FP4_TP_CASES[case]
    x = fp4_inputs(contexts, q, len(contexts) * 131 + q)
    rows, topk, mla_page = len(contexts) * q, 2048, UNIT
    block_rows = mla_page + GAP
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
        topk_tokens=topk,
        max_model_len=1 << 20,
    )
    cap = max(64, rows)
    fused = FusedIndexer(indexer, [(rows, q)], torch.device("cuda"), cap)
    assert fused.fp4
    mla_pages = max(-(-(c + q) // mla_page) for c in contexts)
    mla_table = torch.randint(
        1 << 14, (len(contexts), mla_pages), device="cuda", dtype=torch.int32
    )
    ids = torch.full((cap, topk), -7, dtype=torch.int32, device="cuda")
    fused.write_ids(ids)
    out, valid = fused.run(x.q_rows, x.weights, decode, mla_table, block_rows, rows, q)
    ref = deepgemm_logits(x, rows, q, int(ends.max()) + 1).view(torch.int16)
    torch.accelerator.synchronize()

    tiles = [-(-c // 128) if c > topk else 0 for c in ends[q - 1 :: q].tolist()]
    for t in range(rows):
        r, j, length = t // q, t % q, int(ends[t])
        pos = torch.arange(length, device="cuda")
        if tiles[r]:
            base = (sum(tiles[:r]) * q + j * tiles[r]) * 128
            got = fused.logits[base : base + length]
            assert same_bits(got, ref[t, :length]), (case, t)
            if length > topk:
                pos = exact_topk(got, topk)
        n = min(length, topk)
        page = mla_table[r][pos // mla_page].long()
        physical = page * block_rows + pos % mla_page
        assert int(valid[t]) == n
        assert torch.equal(out[t, :n].long(), physical), (case, t)
        assert torch.equal(ids[t, :n].long(), pos), (case, t)
        assert bool((out[t, n:] == -1).all()) and bool((ids[t, n:] == -1).all())


FP4_DCP_CASES = {
    "mixed": ([700, 40, 1100, 4100], 5),
    "long": ([16400, 9000, 400, 6200], 5),
    "far": ([131000, 70000], 1),
    "q1_64": ([500 + 97 * r for r in range(64)], 1),
    "q8": ([3000, 900], 8),
    "wide320": ([2000 + 23 * r for r in range(64)], 5),
}


@pytest.mark.parametrize("mode", ["packed", "logits"])
@pytest.mark.parametrize("case", list(FP4_DCP_CASES))
def test_fused_fp4_indexer_dcp(case, mode, monkeypatch):
    """The fused indexer under DCP on MXFP4 keys (one rank's local keys, N =
    384): DeepGEMM's logits layout, bit-exact its FP4 paged MQA logits, and
    (packed) the local select's B' rows decode to the exact local top-N, also
    past the SM count (wide320: rows share SMs, as the NVLink exchange runs
    them there)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.indexer import (
        FusedIndexer,
    )

    if mode == "packed":
        monkeypatch.setattr(FusedIndexer, "packs", lambda self, rows: self.packed)

    contexts, q = FP4_DCP_CASES[case]
    x = fp4_inputs(contexts, q, len(contexts) * 7 + q)
    rows, n = len(contexts) * q, 384
    decode = SimpleNamespace(
        seq_lens=x.ends[:rows].view(rows, 1),
        block_table=x.table.repeat_interleave(q, 0)[:rows],
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
    packed, logits, ends = fused.run_dcp(x.q_rows, x.weights, decode, rows, q)
    ref = deepgemm_logits(x, rows, q, int(x.ends.max()) + 1).view(torch.int16)
    torch.accelerator.synchronize()
    bits = logits.view(torch.int16)
    for t in range(rows):
        length = int(ends[t])
        assert same_bits(bits[t, :length], ref[t, :length]), (case, t)
    if mode == "logits":
        return
    counts = fused.valid[:rows]
    for t in range(rows):
        length = int(ends[t])
        c = min(length, n)
        assert int(counts[t]) == c, (case, t)
        words = packed[t, :n].long() & 0xFFFFFFFF
        plane = packed[t, n:].long() & 0xFFFFFFFF
        j = torch.arange(c, device="cuda")
        high = (plane[j // 32] >> (j % 32)) & 1
        ids = (0xFFFF - (words[:c] & 0xFFFF)) | (high << 16)
        assert torch.equal(ids.sort().values, exact_topk(bits[t, :length], n))


@pytest.mark.parametrize("q", [1, 5])
def test_prefill_cache_round_trip(q):
    """P/D cache round trip: a prefill engine writes the MXFP4 indexer keys
    through the stock model's call (mxfp4_fused_prologue, 16K-token chunks into
    packed block rows under a block table), the decode's fused scorer reads them:
    its BF16 logits bit-exact DeepGEMM's FP4 paged MQA logits (the stock decode's
    read) and its selection the exact top-2048; the stock prefill's gather of the
    same cache returns the written bytes, which dequantize to the LayerNorm'd
    keys within half an E2M1 step of their block."""
    import torch.nn.functional as F

    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import (
        mxfp4_fused_prologue,
        mxfp4_q_quant,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.index import (
        fp4_query_views,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.indexer import (
        FusedIndexer,
    )

    contexts, topk, chunk = [1500, 2047, 9000, 40000, 70000], 2048, 16384
    gen = torch.Generator(device="cuda").manual_seed(17 + q)
    need = [-(-(c + q) // UNIT) for c in contexts]
    blocks = sum(need) + 3
    index = packed_cache(blocks, UNIT, FP4_ROW)
    perm = torch.randperm(blocks, device="cuda", generator=gen).int()
    table = torch.zeros(len(contexts), max(need), dtype=torch.int32, device="cuda")
    used = 0
    for r, n in enumerate(need):
        table[r, :n] = perm[used : used + n]
        used += n

    # Prefill: every request's context and its q draft/verify positions.
    bf = torch.bfloat16
    lens = [c + q - 1 for c in contexts]
    req = torch.cat([torch.full((n,), r, device="cuda") for r, n in enumerate(lens)])
    pos = torch.cat([torch.arange(n, device="cuda") for n in lens])
    slots = (table[req, pos // UNIT].long() * UNIT + pos % UNIT).contiguous()
    tokens = slots.shape[0]
    keys = torch.randn(tokens, 128, device="cuda", dtype=bf, generator=gen) * 3
    k_w = torch.rand(128, device="cuda", generator=gen) + 0.5
    k_b = torch.randn(128, device="cuda", generator=gen) * 0.1
    q_w = torch.ones(1536, device="cuda", dtype=bf)
    kv_w = torch.ones(512, device="cuda", dtype=bf)
    for start in range(0, tokens, chunk):
        end = min(start + chunk, tokens)
        n = end - start
        q_c = torch.randn(n, 1536, device="cuda", dtype=bf, generator=gen)
        kv_c = torch.randn(n, 512, device="cuda", dtype=bf, generator=gen)
        k_pe = torch.randn(n, 64, device="cuda", dtype=bf, generator=gen)
        scratch = torch.empty(n, topk, dtype=torch.int32, device="cuda")
        mxfp4_fused_prologue(
            q_c, kv_c, k_pe, keys[start:end], q_w, kv_w, k_w, k_b, EPS, scratch,
            None, None, index, slots[start:end],
        )  # fmt: skip

    # The written keys: the LayerNorm'd BF16 keys within half an E2M1 step.
    page = pages(index)
    codes = page[:, : UNIT * 64].reshape(-1, 64)
    scales = page[:, UNIT * 64 :].reshape(-1, 4)
    got = dequant(codes[slots], scales[slots])
    ref = F.layer_norm(keys.float(), (128,), k_w, k_b, EPS).to(bf).float()
    step = torch.exp2(scales[slots].float() - 127).repeat_interleave(32, -1)
    assert bool(((got - ref).abs() <= step).all())  # one step at the 4-6 code gap
    # The stock prefill's read: the gathered rows are the written bytes.
    k_quant = torch.empty(lens[-1], 64, dtype=torch.uint8, device="cuda")
    k_scale = torch.empty(lens[-1], 4, dtype=torch.uint8, device="cuda")
    ops.cp_gather_indexer_k_quant_cache(
        index,
        k_quant,
        k_scale,
        table[-1:].contiguous(),
        torch.tensor([0, lens[-1]], dtype=torch.int32, device="cuda"),
    )
    last = slots[req == len(contexts) - 1]
    assert torch.equal(k_quant, codes[last]) and torch.equal(k_scale, scales[last])

    # Decode: the MXFP4 query (as the stock writer makes it) in the query rows.
    rows = len(contexts) * q
    iq = torch.randn(rows, 32, 128, device="cuda", dtype=bf, generator=gen)
    wv = torch.randn(rows, 32, device="cuda", dtype=bf, generator=gen)
    q_packed, q_scale, weights = mxfp4_q_quant(iq, wv, 1 / 64, 32)
    q_rows = torch.zeros(max(64, rows), 32, 128, dtype=torch.uint8, device="cuda")
    packed, scale_words = fp4_query_views(q_rows, rows)
    packed.copy_(q_packed)
    scale_words.copy_(q_scale)
    ends = torch.tensor(
        [c + 1 + j for c in contexts for j in range(q)],
        dtype=torch.int32,
        device="cuda",
    )
    x = SimpleNamespace(
        index=index, table=table, ends=ends, q_rows=q_rows, weights=weights.to(bf)
    )
    decode = SimpleNamespace(
        seq_lens=ends.view(rows, 1), block_table=table.repeat_interleave(q, 0)
    )
    indexer = SimpleNamespace(
        k_cache=SimpleNamespace(kv_cache=index), topk_tokens=topk, max_model_len=1 << 20
    )
    fused = FusedIndexer(indexer, [(rows, q)], torch.device("cuda"), max(64, rows))
    assert fused.fp4
    mla_table = torch.randint(
        1 << 14, (len(contexts), max(need)), device="cuda", dtype=torch.int32
    )
    ids = torch.full((max(64, rows), topk), -7, dtype=torch.int32, device="cuda")
    fused.write_ids(ids)
    block_rows = UNIT + GAP
    out, valid = fused.run(q_rows, x.weights, decode, mla_table, block_rows, rows, q)
    ref_logits = deepgemm_logits(x, rows, q, int(ends.max()) + 1).view(torch.int16)
    torch.accelerator.synchronize()

    tiles = [-(-c // 128) if c > topk else 0 for c in ends[q - 1 :: q].tolist()]
    for t in range(rows):
        r, j, length = t // q, t % q, int(ends[t])
        sel = torch.arange(length, device="cuda")
        if tiles[r]:
            base = (sum(tiles[:r]) * q + j * tiles[r]) * 128
            logits = fused.logits[base : base + length]
            assert same_bits(logits, ref_logits[t, :length]), t
            if length > topk:
                sel = exact_topk(logits, topk)
        n = min(length, topk)
        physical = mla_table[r][sel // UNIT].long() * block_rows + sel % UNIT
        assert int(valid[t]) == n
        assert torch.equal(out[t, :n].long(), physical), t
        assert torch.equal(ids[t, :n].long(), sel), t


# --------------------------------------------------------------------------
# DCP top-k exchange (topk_exchange): 8 simulated ranks on one GPU
# --------------------------------------------------------------------------
def _local_select(rows, n, pitch, world):
    """IndexerLocalTopk with its exchange push, compiled for [rows] rows."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        dynamic_fake,
        fake,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.indexer import (
        IndexerLocalTopk,
    )

    width = n + n // 32
    args = (
        fake((rows * pitch,), (1,), cutlass.Int16),
        fake((rows,), (1,), torch.int32, 4),
        fake((rows, width), (width, 1), torch.int32, 4),
        fake((rows,), (1,), torch.int32, 4),
        dynamic_fake(1, torch.int64),
        dynamic_fake(1, torch.int32),
        cutlass.Int32(0),
        cutlass.Int32(1),
        make_fake_stream(),
    )
    return cute.compile(IndexerLocalTopk(rows, n, pitch, world), *args)


def _local_workspace(world, rank, max_rows, topk, n):
    """A rescue workspace whose W peers are this GPU's own buffers."""
    from vllm.v1.attention.ops.dcp_topk_merge import packed_row_words, rescue_slots
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import DCPTopkRescueWorkspace

    ws = DCPTopkRescueWorkspace.__new__(DCPTopkRescueWorkspace)
    slots = rescue_slots(topk, n)
    recv = torch.zeros(
        max_rows, slots, packed_row_words(topk), dtype=torch.int32, device="cuda"
    )
    flags = torch.zeros(world, max_rows, dtype=torch.int32, device="cuda")
    peer_recv, peer_flags = (
        torch.tensor([t.data_ptr()] * world, device="cuda") for t in (recv, flags)
    )
    ws._init(recv, flags, peer_recv, peer_flags, max_rows, topk, n)
    ws.world_size, ws.rank, ws.device_group = world, rank, None
    return ws


EXCHANGE_CASES = {
    # (global contexts, q, logits kind)
    "a1": ([136505], 5, "random"),
    "a4": ([136505, 120000, 90000, 136505], 5, "random"),
    "r12": ([32773 + 1000 * r for r in range(12)], 5, "random"),
    "short": ([100, 2040, 3000, 9000], 5, "random"),
    "tied": ([136505, 32773], 5, "zero"),
    "dups": ([32773, 136505, 5000], 5, "dups"),
    "coarse": ([32773, 136505, 5000], 5, "coarse"),
    "sat0": ([32773, 136505, 60000], 5, "sat0"),
}


@pytest.mark.parametrize("case", list(EXCHANGE_CASES))
def test_dcp_topk_exchange_merge(case):
    """8 simulated DCP ranks (interleave 1) on one GPU: each rank's
    IndexerLocalTopk pushes its B' rows into the exchange (every peer address
    this GPU's slots), ExchangeMerge (rank 0) merges them; ids, rows and
    counts bit-equal, in the same order, the #63 merge's select + rescue over
    the same B' rows (dcp_topk_select). Two calls (both epoch parities).
    "sat0": rank 0 holds most of the top-K of some rows (its rescue push);
    "coarse" / "dups" / "tied": tens, hundreds and thousands of ties at the
    threshold score (the merge's tie ranking and its radix fallback)."""
    from types import SimpleNamespace as NS

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        dynamic_view,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.topk_exchange import (  # noqa: E501
        ExchangeMerge,
        ll_pairs,
    )
    from vllm.v1.attention.ops.dcp_topk_merge import dcp_merge_local_topk
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        _RowsOutput,
        dcp_topk_select,
    )

    contexts, q, kind = EXCHANGE_CASES[case]
    world, topk, rank0 = 8, 2048, 0
    n = dcp_merge_local_topk(topk, world)
    rows = len(contexts) * q
    ends = torch.tensor(
        [c + 1 + j for c in contexts for j in range(q)], dtype=torch.int32
    ).cuda()
    max_local = -(-int(ends.max()) // world)
    pitch = -(-max_local // 512) * 512
    width = n + n // 32
    local = _local_select(rows, n, pitch, world)
    pairs = ll_pairs(n)
    recv = torch.zeros(2, world, rows, 4 * pairs, dtype=torch.int32, device="cuda")
    ex = NS(
        recv=recv,
        peers=torch.tensor([recv.data_ptr()] * world, device="cuda"),
        epoch=torch.zeros(rows, dtype=torch.int32, device="cuda"),
        rows=rows,
        world=world,
        rank=rank0,
        local_topk=n,
    )
    ws_new = _local_workspace(world, rank0, rows, topk, n)
    ws_ref = _local_workspace(world, rank0, rows, topk, n)
    merge = ExchangeMerge(ex, ws_new, rows, topk, UNIT, emit_ids=True)
    # This rank's MLA cache rows: reversed page ids per request.
    mla_blocks = -(-max_local // UNIT) + 1
    mla_table = torch.arange(len(contexts) * mla_blocks, dtype=torch.int32).cuda()
    mla_table = mla_table.flip(0).view(len(contexts), mla_blocks).contiguous()
    req_id = torch.arange(len(contexts), dtype=torch.int32).repeat_interleave(q)
    req_id = req_id.cuda()
    gen = torch.Generator(device="cuda").manual_seed(len(case))
    for call in range(2):
        span = world * pitch
        if kind == "zero":
            g = torch.zeros(rows, span, device="cuda")
        elif kind == "dups":
            vals = torch.tensor([-1.5, 0.25, 0.25, 3.0], device="cuda")
            g = vals[torch.randint(0, 4, (rows, span), device="cuda", generator=gen)]
        else:
            g = torch.randn(rows, span, device="cuda", generator=gen)
        if kind == "coarse":  # ~40 values: tens of ties at the threshold score
            g = (g * 4).round() / 4
        if kind == "sat0":  # rank 0's keys of every other request's rows lead
            boost = (torch.arange(span, device="cuda") % world == 0).float() * 8
            g[::2] += boost
        g = g.bfloat16()
        packs, logits_r, ends_r = [], [], []
        for r in range(world):
            lg = torch.empty(rows, pitch, dtype=torch.bfloat16, device="cuda")
            lg.copy_(g[:, r::world][:, :pitch])
            le = ((ends - r + world - 1) // world).clamp_min(0).int()
            out = torch.zeros(rows, width, dtype=torch.int32, device="cuda")
            valid = torch.zeros(rows, dtype=torch.int32, device="cuda")
            local(
                to_cute(lg.view(torch.int16).view(-1)),
                to_cute(le, 4),
                to_cute(out, 4),
                to_cute(valid, 4),
                dynamic_view(ex.peers),
                dynamic_view(ex.epoch),
                cutlass_i32(r),
                cutlass_i32(rows),
                stream(torch.device("cuda")),
            )
            packs.append(out)
            logits_r.append(lg[:, :max_local])
            ends_r.append(le)
        got = [torch.full((rows, topk), -7, dtype=torch.int32, device="cuda")]
        got += [torch.full((rows,), -7, dtype=torch.int32, device="cuda")]
        got += [torch.full((rows, topk), -7, dtype=torch.int32, device="cuda")]
        merge(logits_r[0], ends_r[0], req_id, mla_table, UNIT, got[0], got[1], got[2])
        ref = [torch.full_like(t, -9) for t in got]
        rows_ref = _RowsOutput(
            req_id=req_id,
            block_table=mla_table,
            rows_out=ref[0],
            valid_out=ref[1],
            dcp_rank=rank0,
            block_size=UNIT,
            stride_rows=UNIT,
        )
        dcp_topk_select(
            torch.stack(packs),
            n,
            ref[2],
            topk=topk,
            rows=rows_ref,
            exact=ws_ref.args(logits_r[0], ends_r[0]),
        )
        torch.accelerator.synchronize()
        for a, b, what in zip(got, ref, ("rows", "counts", "ids")):
            assert torch.equal(a, b), (case, call, what)
        assert torch.equal(ex.epoch, torch.full_like(ex.epoch, call + 1))
        assert torch.equal(ws_new.seq2, ws_ref.seq2), (case, call)
        if kind == "sat0":
            assert int((ws_ref.sat_word >> 16).ne(0).sum()) > 0, "no saturated row"
        count = (got[2] >= 0).sum(1)
        want = ends.clamp(max=topk) if kind != "short" else ends.clamp(max=topk)
        assert torch.equal(count.int(), want), (case, call)


def cutlass_i32(v):
    import cutlass

    return cutlass.Int32(v)
