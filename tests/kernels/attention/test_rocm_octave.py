# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Octave HIP kernels against the PyTorch reference in
vllm/v1/attention/ops/rocm_octave.py."""

import math

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.platforms import current_platform
from vllm.v1.attention.backends.rocm_octave_attn import RocmOctaveAttentionBackend
from vllm.v1.attention.ops import rocm_octave as sq
from vllm.v1.kv_cache_interface import AttentionSpec, get_kv_quant_mode


def _arch() -> str | None:
    if not current_platform.is_rocm():
        return None
    from vllm.platforms.rocm import get_cdna_version, get_rdna_version

    rdna = get_rdna_version()
    if rdna in (3, 4):
        return f"rdna{rdna}"
    return "cdna3" if get_cdna_version() == 3 else None


ARCH = _arch()
pytestmark = pytest.mark.skipif(
    ARCH is None or (ARCH != "rdna4" and not hasattr(torch.ops._C, "octave_decode")),
    reason="Octave kernels need RDNA3, RDNA4 or CDNA3",
)

HEAD, ROPE, BLOCK = 256, 64, 16
CACHE_DTYPES = ["octave_k3v4", "octave_k3v3", "octave_k3v3_compact"]


@pytest.fixture(scope="module")
def kernel():
    if ARCH != "rdna4":
        return None
    from vllm.v1.attention.ops.rocm_octave_rdna4 import TunedHipOctave

    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    return TunedHipOctave(HEAD, ROPE, arch)


@pytest.fixture(scope="module")
def sparse_kernel():
    if ARCH != "rdna4":
        return None
    from vllm.v1.attention.ops.rocm_octave_rdna4.hip import get_geometry_engine

    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    return get_geometry_engine(HEAD, ROPE, arch)


def _format(cache_dtype: str) -> sq.OctaveFormat:
    return sq.OctaveFormat.from_cache_dtype(cache_dtype, HEAD, ROPE)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _token_slots(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """(blocks, H, block_size, slot) cache -> (T, H, slot) for flat slots."""
    flat = cache.permute(0, 2, 1, 3).reshape(-1, cache.shape[1], cache.shape[3])
    return flat[slots]


def _cache_dtype(fmt):
    return "octave_k3v3_compact" if fmt.compact else f"octave_k3v{fmt.v_bits}"


def _store(k, v, cache, slots, k_signs, v_signs, fmt, kernel):
    if kernel is not None:
        kernel.store(k, v, cache, slots, k_signs, _cache_dtype(fmt))
    else:
        torch.ops._C.octave_cache_store(
            k, v, cache, slots, k_signs, v_signs, fmt.kernel_code
        )


def _fill(cache, block_ids, k, v, k_signs, v_signs, fmt, kernel=None):
    n = k.shape[0]
    slots = (
        block_ids.long().repeat_interleave(BLOCK)[:n] * BLOCK
        + torch.arange(n, device=k.device) % BLOCK
    )
    _store(k, v, cache, slots, k_signs, v_signs, fmt, kernel)
    return slots


@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_store_matches_reference(cache_dtype, dtype, kernel):
    torch.manual_seed(0)
    dev, fmt = "cuda", _format(cache_dtype)
    ksg = vsg = sq.sign_bits(HEAD).to(dev)
    t, hkv = 1024, 2
    k = torch.randn(t, hkv, HEAD, device=dev)
    k[..., 100] *= 8  # an outlier channel is spread by the rotation
    k = k.to(dtype)
    v = torch.randn(t, hkv, HEAD, device=dev).to(dtype)
    nblk = t // BLOCK + 4
    cache = torch.zeros(nblk, hkv, BLOCK, fmt.slot_bytes, dtype=torch.uint8, device=dev)
    slots = _fill(cache, torch.randperm(nblk, device=dev), k, v, ksg, vsg, fmt, kernel)

    k_got, v_got = sq.reference_dequantize(_token_slots(cache, slots), fmt)
    ref = sq.reference_quantize(k.float(), v.float(), fmt)
    k_ref, v_ref = sq.reference_dequantize(ref, fmt)
    # Same codes up to rounding ties from a different summation order.
    assert _rel(k_got, k_ref) < 1e-2 and _rel(v_got, v_ref) < 1e-2
    # Lloyd-Max on Gaussian data: 3-bit codes (4-bit for block K's RoPE
    # block) keep ~19% of the norm as error, 4-bit ~10%.
    assert _rel(k_got, k) < 0.2
    assert _rel(v_got, v) < (0.12 if fmt.v_bits == 4 else 0.22)


@pytest.mark.parametrize(
    "cache_dtype,slot_bytes",
    [("octave_k3v4", 248), ("octave_k3v3", 216), ("octave_k3v3_compact", 196)],
)
def test_slot_bytes_match_kernel_layout(cache_dtype, slot_bytes):
    """The cache is sized from the Python format; the kernels index it with
    octave::Format in csrc/attention/octave_format.cuh."""
    assert _format(cache_dtype).slot_bytes == slot_bytes


@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES)
def test_store_skips_padding_slots(cache_dtype, kernel):
    """Tokens with slot -1 (cudagraph padding) leave every byte unchanged."""
    torch.manual_seed(0)
    dev, fmt = "cuda", _format(cache_dtype)
    ksg = vsg = sq.sign_bits(HEAD).to(dev)
    t, hkv = 8, 2
    k = torch.randn(t, hkv, HEAD, device=dev).bfloat16()
    v = torch.randn(t, hkv, HEAD, device=dev).bfloat16()
    cache = torch.randint(
        0, 256, (2, hkv, BLOCK, fmt.slot_bytes), dtype=torch.uint8, device=dev
    )
    before = cache.clone()
    slots = torch.full((t,), -1, dtype=torch.long, device=dev)
    slots[3] = 21
    _store(k, v, cache, slots, ksg, vsg, fmt, kernel)
    written = torch.zeros_like(cache, dtype=torch.bool)
    written[1, :, 5] = True
    assert torch.equal(cache[~written], before[~written])
    assert not torch.equal(cache[written], before[written])


@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES)
@pytest.mark.parametrize("query_group", [1, 4])
@pytest.mark.parametrize("num_splits", [1, 37])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_wmma", [True] if ARCH == "rdna4" else [True, False])
def test_decode_matches_reference(
    cache_dtype, query_group, num_splits, dtype, use_wmma, kernel
):
    """Covers MTP verification: up to 4 consecutive query tokens per request,
    each with its own causal length, plus a padded query with no request.
    `use_wmma=False` checks the portable RDNA3/CDNA3 tier. RDNA4 uses
    its independent native runtime."""
    torch.manual_seed(1)
    dev, fmt = "cuda", _format(cache_dtype)
    ksg = vsg = sq.sign_bits(HEAD).to(dev)
    hkv, group = 1, 6
    lens = [1, 16, 37, 700, 4096]
    cache = torch.zeros(512, hkv, BLOCK, fmt.slot_bytes, dtype=torch.uint8, device=dev)
    max_blocks = max(math.ceil(n / BLOCK) for n in lens)
    block_table = torch.zeros(len(lens), max_blocks, dtype=torch.int32, device=dev)
    free = torch.randperm(512, device=dev)
    q_to_req, q_to_klen, used = [], [], 0
    for i, n in enumerate(lens):
        nb = math.ceil(n / BLOCK)
        block_table[i, :nb] = free[used : used + nb].int()
        used += nb
        k = torch.randn(n, hkv, HEAD, device=dev).bfloat16()
        v = torch.randn(n, hkv, HEAD, device=dev).bfloat16()
        _fill(cache, block_table[i, :nb], k, v, ksg, vsg, fmt, kernel)
        nq = min(query_group, n)
        for j in range(nq):
            q_to_req.append(i)
            q_to_klen.append(n - nq + 1 + j)
    q_to_req.append(len(lens) + 3)  # cudagraph padding: no request
    q_to_klen.append(0)
    q_to_req = torch.tensor(q_to_req, dtype=torch.int32, device=dev)
    q_to_klen = torch.tensor(q_to_klen, dtype=torch.int32, device=dev)

    num_q = q_to_req.numel()
    q = (torch.randn(num_q, hkv * group, HEAD, device=dev) * 2).to(dtype)
    out = _decode(
        cache,
        q,
        block_table,
        q_to_req,
        q_to_klen,
        ksg,
        fmt,
        use_wmma,
        kernel,
        num_splits=num_splits,
        query_group=query_group,
    )
    ref = sq.reference_attention(
        q.float(),
        cache,
        block_table,
        q_to_req.clamp(max=len(lens) - 1),
        q_to_klen,
        fmt,
        1 / 16,
    )
    # Only the kernel's query precision (fp16 or int8) differs.
    assert _rel(out[:-1], ref[:-1]) < 2e-2
    assert out[-1].abs().max().item() == 0


@pytest.mark.skipif(ARCH != "rdna4", reason="RDNA4 live split count")
def test_rdna4_decode_graph_matches_reference(kernel):
    """Follow live lengths over shuffled pages and ignore inactive scratch."""
    torch.manual_seed(17)
    cache_dtype, dtype = "octave_k3v4", torch.bfloat16
    batch, page = 1, 7
    dev, fmt, context, splits = "cuda", _format(cache_dtype), 2048, 256
    signs = sq.sign_bits(HEAD).to(dev)
    pages_per_request = math.ceil(context / page)
    pages = batch * pages_per_request
    cache = torch.zeros(
        pages + 1, 1, page, fmt.slot_bytes, dtype=torch.uint8, device=dev
    )
    table = torch.zeros(
        batch, max(82, pages_per_request), dtype=torch.int32, device=dev
    )
    blocks = torch.arange(pages, 0, -1, device=dev).view(batch, pages_per_request)
    table[:, :pages_per_request] = blocks
    positions = torch.arange(context, device=dev)
    for request in range(batch):
        key = torch.randn(context, 1, HEAD, device=dev).to(dtype)
        value = torch.randn_like(key)
        slots = blocks[request, positions // page] * page + positions % page
        _store(key, value, cache, slots, signs, signs, fmt, kernel)
    query = torch.randn(batch, 6, HEAD, device=dev).to(dtype)
    normal_query = query.clone()
    mapping = torch.arange(batch, dtype=torch.int32, device=dev)
    lengths = torch.full((batch,), context, dtype=torch.int32, device=dev)
    output = torch.empty(query.shape, dtype=torch.float32, device=dev)
    scratch = torch.empty(batch, 6, splits, HEAD + 2, device=dev)

    def decode():
        kernel.decode(
            query,
            cache,
            table,
            mapping,
            lengths,
            signs,
            cache_dtype,
            scale=1 / 16,
            splits=splits,
            out=output,
            scratch=scratch,
        )

    decode()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode()
    for length, unsafe in (
        (context, False),
        (127, False),
        (128, False),
        (1, False),
        (0, False),
        (context, True),
        (context, False),
    ):
        lengths.fill_(length)
        query.copy_(normal_query)
        if unsafe:
            query[0].fill_(1000)
        mapping.copy_(torch.arange(batch, dtype=torch.int32, device=dev))
        if length == 0:
            mapping.fill_(batch + 10)  # Padding must not dereference a request.
        output.fill_(float("nan"))
        scratch.fill_(float("nan"))
        graph.replay()
        if length == 0:
            assert torch.count_nonzero(output).item() == 0
            continue
        reference = sq.reference_attention(
            query.float(), cache, table, mapping, lengths, fmt, 1 / 16
        )
        assert torch.isfinite(output).all()
        error = (output - reference).flatten(1).norm(dim=1)
        norm = reference.flatten(1).norm(dim=1)
        assert (error / norm.clamp_min(1e-20)).max().item() < 1e-3


@pytest.mark.skipif(ARCH != "rdna4", reason="RDNA4 MTP decode path")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_splits", [1, 37])
def test_rdna4_mtp_four_token_k3v4_matches_reference(dtype, num_splits, kernel):
    """Cover the native four-row MTP WMMA path for K3V4 GQA6."""
    torch.manual_seed(13)
    dev, fmt, hkv, group, length = "cuda", _format("octave_k3v4"), 1, 6, 1024
    signs = sq.sign_bits(HEAD).to(dev)
    nblocks = math.ceil(length / BLOCK)
    cache = torch.zeros(
        nblocks, hkv, BLOCK, fmt.slot_bytes, dtype=torch.uint8, device=dev
    )
    pages = torch.arange(nblocks, dtype=torch.int32, device=dev)
    table = pages.view(1, -1).contiguous()
    key = torch.randn(length, hkv, HEAD, device=dev).to(dtype)
    value = torch.randn_like(key)
    _fill(cache, pages, key, value, signs, signs, fmt, kernel)

    q = torch.randn(4, hkv * group, HEAD, device=dev).to(dtype)
    q_to_req = torch.zeros(4, dtype=torch.int32, device=dev)
    q_to_klen = torch.arange(length - 3, length + 1, dtype=torch.int32, device=dev)
    out = _decode(
        cache,
        q,
        table,
        q_to_req,
        q_to_klen,
        signs,
        fmt,
        True,
        kernel,
        num_splits=num_splits,
        query_group=4,
    )
    ref = sq.reference_attention(
        q.float(), cache, table, q_to_req, q_to_klen, fmt, 1 / 16
    )
    assert _rel(out, ref) < 2e-2


@pytest.mark.skipif(ARCH != "rdna4", reason="RDNA4 Octave geometry contract")
@pytest.mark.parametrize(
    "head_dim,rope_dim",
    [(128, 32), (192, 48), (512, 128), (128, 64), (256, 32), (256, 128)],
)
def test_rdna4_geometry_rejects_unsupported_head_and_rope(head_dim, rope_dim):
    """RDNA4 Octave must reject every geometry outside the shipped 256/64 ABI."""
    from vllm.v1.attention.ops.rocm_octave_rdna4 import OctaveGeometry, TunedHipOctave

    with pytest.raises(ValueError, match="only head dimension 256 with 64 RoPE"):
        OctaveGeometry(head_dim, rope_dim)
    with pytest.raises(ValueError, match="only head dimension 256 with 64 RoPE"):
        TunedHipOctave(head_dim, rope_dim, "gfx1201")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "cache_dtype,query_lens,group,path",
    [
        (cache_dtype, query_lens, group, "packed")
        for cache_dtype in CACHE_DTYPES
        for query_lens in ([256], [512], [129, 1, 517])
        for group in (1, 2, 6)
    ]
    + [
        ("octave_k3v3_compact", [256], 4, "packed"),
        ("octave_k3v3_compact", [256], 8, "packed"),
    ]
    + (
        [
            (cache_dtype, query_lens, group, path)
            for cache_dtype, query_lens, group in (
                [(fmt, [129, 1, 517], 6) for fmt in CACHE_DTYPES]
                + [
                    ("octave_k3v3", [7, 1, 13], 1),
                    ("octave_k3v4", [19, 1, 35], 4),
                    ("octave_k3v3", [31, 1, 33], 5),
                    ("octave_k3v3_compact", [17, 1, 31], 8),
                    ("octave_k3v4", [3, 1, 5], 321),
                ]
            )
            for path in ("expanded", "streamed")
        ]
        if ARCH == "rdna4"
        else []
    ),
)
def test_prefill_matches_reference(cache_dtype, query_lens, dtype, group, path, kernel):
    """Chunked prefill: each request's chunk attends causally to itself
    (unquantized K/V) and to its cached prefix (packed cache), in the rotated
    space the backend runs it in."""
    torch.manual_seed(3)
    dev, fmt = "cuda", _format(cache_dtype)
    ksg = vsg = sq.sign_bits(HEAD).to(dev)
    hkv = 2 if path != "packed" and group != 6 else 1
    ctx_lens = ([67, 0, 97] if hkv > 1 else [4096, 0, 4096])[: len(query_lens)]
    seq_lens = [c + n for c, n in zip(ctx_lens, query_lens)]
    max_blocks = max(math.ceil(n / BLOCK) for n in seq_lens)
    nblk = sum(math.ceil(n / BLOCK) for n in seq_lens) + 8
    cache = torch.zeros(nblk, hkv, BLOCK, fmt.slot_bytes, dtype=torch.uint8, device=dev)
    block_table = torch.zeros(len(seq_lens), max_blocks, dtype=torch.int32, device=dev)
    free, used = torch.randperm(nblk, device=dev), 0
    ks, vs = [], []
    for i, n in enumerate(seq_lens):
        nb = math.ceil(n / BLOCK)
        block_table[i, :nb] = free[used : used + nb].int()
        used += nb
        k = torch.randn(n, hkv, HEAD, device=dev).to(dtype)
        v = torch.randn(n, hkv, HEAD, device=dev).to(dtype)
        _fill(cache, block_table[i, :nb], k, v, ksg, vsg, fmt, kernel)
        ks.append(k)
        vs.append(v)

    qsl = torch.tensor([0, *query_lens], device=dev).cumsum(0).int()
    total = int(qsl[-1])
    q = (torch.randn(total, hkv * group, HEAD, device=dev) * 2).to(dtype)
    if path != "packed":
        q[0, 0, HEAD // 2 :].fill_(1000)  # Exercise the packed FP32 repair.
    k_new = torch.cat([k[c:] for k, c in zip(ks, ctx_lens)])
    v_new = torch.cat([v[c:] for v, c in zip(vs, ctx_lens)])
    out = _prefill(
        cache,
        q,
        k_new,
        v_new,
        block_table,
        qsl,
        torch.tensor(seq_lens, dtype=torch.int32, device=dev),
        ksg,
        fmt,
        kernel,
        path=path,
    )

    ref = torch.empty(total, hkv * group, HEAD, device=dev)
    for i, (c, n) in enumerate(zip(ctx_lens, query_lens)):
        k_all, v_all = ks[i][c:].float(), vs[i][c:].float()
        if c > 0:
            pos = torch.arange(c, device=dev)
            slots = cache[block_table[i, pos // BLOCK].long(), :, pos % BLOCK]
            k_pre, v_pre = sq.reference_dequantize(slots, fmt)
            k_all, v_all = torch.cat([k_pre, k_all]), torch.cat([v_pre, v_all])
        k_all = k_all.repeat_interleave(group, 1)
        v_all = v_all.repeat_interleave(group, 1)
        qi = q[int(qsl[i]) : int(qsl[i + 1])].float()
        s = torch.einsum("qhd,thd->hqt", qi, k_all) / 16
        mask = (
            torch.arange(c + n, device=dev)[None]
            > (c + torch.arange(n, device=dev))[:, None]
        )
        p = torch.softmax(s.masked_fill(mask, float("-inf")), -1)
        ref[int(qsl[i]) : int(qsl[i + 1])] = torch.einsum("hqt,thd->qhd", p, v_all)
    # Only attention arithmetic and output rounding differ.
    assert _rel(out, ref) < (1e-3 if path != "packed" else 2e-2)
    if path != "packed":
        assert _rel(out[0, 0], ref[0, 0]) < 1e-3


@pytest.mark.skipif(ARCH != "rdna4", reason="RDNA4 cached prefill")
@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens", [129, 513])
def test_cached_prefill_sharp_keys_preserve_attention_weights(
    cache_dtype, dtype, tokens, kernel
):
    """FP16 rounding of QK can change the weights of competing cached keys."""
    torch.manual_seed(127421)
    device, prefix, heads = "cuda", 1027, 6
    fmt = kernel.layout(cache_dtype)
    signs = sq.sign_bits(HEAD).to(device)
    pages = math.ceil((prefix + tokens) / BLOCK)
    cache = torch.empty(
        pages, 1, BLOCK, fmt.slot_bytes, dtype=torch.uint8, device=device
    )
    table = torch.randperm(pages, device=device).int()[None]
    k = (8 * torch.randn(prefix + tokens, 1, HEAD, device=device)).to(dtype)
    v = torch.randn(prefix + tokens, 1, HEAD, device=device).to(dtype)
    q = torch.randn(tokens, heads, HEAD, device=device).to(dtype)
    pk, pv = sq.reference_dequantize(
        _token_slots(
            cache,
            _fill(cache, table[0], k[:prefix], v[:prefix], signs, signs, fmt, kernel),
        ),
        fmt,
    )
    starts = torch.tensor([0, tokens], device=device, dtype=torch.int32)
    lengths = starts[-1:] + prefix
    path, shapes = kernel.prefill_workspace(
        q, cache, table, prefix + tokens, 1 / 16, 2 << 30, prefix
    )
    scratch = [torch.empty(sh, dtype=dt, device=device) for sh, dt in shapes]
    out = torch.empty_like(q, dtype=torch.float32)
    args = (
        q,
        k[prefix:],
        v[prefix:],
        cache,
        table,
        starts,
        lengths,
        signs,
        cache_dtype,
    )
    if path == "streamed":
        kernel.prefill_streamed(*args, buffers=scratch, out=out, scale=1 / 16)
    else:
        assert path == "expanded"
        kt, vt, qt, flags, correction = scratch
        kernel.prefill_expanded(
            *args,
            key_tmp=kt,
            value_tmp=vt,
            query_tmp=qt,
            repair=flags,
            correction=correction,
            out=out,
            scale=1 / 16,
        )
    keys = torch.cat((pk[:, 0].double(), k[prefix:, 0].double()))
    values = torch.cat((pv[:, 0].double(), v[prefix:, 0].double()))
    causal = torch.arange(len(keys), device=device) > (
        prefix + torch.arange(tokens, device=device)[:, None]
    )
    scores = (q.double() @ keys.T) / 16
    ref = scores.masked_fill(causal[:, None], -torch.inf).softmax(-1) @ values
    errors = (out - ref).norm(dim=-1) / ref.norm(dim=-1)
    assert errors.amax().item() < 1.2e-3


@pytest.mark.skipif(ARCH != "rdna4", reason="RDNA4 expanded prefill")
@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES)
def test_expanded_prefill_keeps_small_weights_on_large_values(cache_dtype, kernel):
    device, prefix, tokens = "cuda", 1025, 32
    fmt = kernel.layout(cache_dtype)
    signs = sq.sign_bits(HEAD).to(device)
    pages = math.ceil((prefix + tokens) / BLOCK)
    cache = torch.empty(
        pages, 1, BLOCK, fmt.slot_bytes, dtype=torch.uint8, device=device
    )
    table = torch.arange(pages, device=device, dtype=torch.int32)[None]
    k = torch.zeros(prefix, 1, HEAD, device=device, dtype=torch.float16)
    k[0] = 1
    v = torch.full_like(k, 4096)
    v[0] = 1
    slots = _fill(cache, table[0], k, v, signs, signs, fmt, kernel)
    keys, values = sq.reference_dequantize(_token_slots(cache, slots[:2]), fmt)
    # Put the winning key first. Later tiles must retain the other keys'
    # small probabilities to account for their large contribution to PV.
    first = keys[0, 0].double()
    q0 = (first * (17 * 16 / first.square().sum())).half()
    q = q0.repeat(tokens, 6, 1)
    live = torch.zeros_like(k[:tokens])
    starts = torch.tensor([0, tokens], device=device, dtype=torch.int32)
    out = _prefill(
        cache,
        q,
        live,
        live,
        table,
        starts,
        starts[-1:] + prefix,
        signs,
        fmt,
        kernel,
        path="expanded",
    )
    weights = (keys[:, 0].double() @ q0.double() / 16).exp()
    weights[1] *= prefix - 1
    mass = weights.sum() + torch.arange(1, tokens + 1, device=device)
    ref = (weights @ values[:, 0].double()) / mass.double()[:, None]
    error = (out.double() - ref[:, None]).norm(dim=-1) / ref.norm(dim=-1)[:, None]
    assert error.amax().item() < 1.5e-3


@pytest.mark.skipif(ARCH != "rdna4", reason="RDNA4 prefill dispatch")
@pytest.mark.parametrize(
    "group,batch,prefix,budget,path",
    [
        (1, 1, 512, 2 << 30, "streamed"),
        (4, 1, 20480, 2 << 30, "streamed"),
        (5, 1, 0, 2 << 30, "streamed"),
        (8, 1, 0, 2 << 30, "streamed"),
        (8, 1, 20480, 2 << 30, "expanded"),
        (4, 4, 20480, 2 << 30, "expanded"),
        (8, 1, 20480, 16 << 20, "streamed"),
        (8, 1, 20480, 1 << 20, "packed"),
    ],
)
def test_prefill_dispatch_by_work_and_budget(
    group, batch, prefix, budget, path, kernel
):
    from vllm.utils.math_utils import round_up

    chunk, kv = 256, 4
    pages = math.ceil((chunk + prefix) / BLOCK)
    query = torch.empty((batch * chunk, kv * group, HEAD), device="meta")
    slot = kernel.layout("octave_k3v4").slot_bytes
    cache = torch.empty((pages * batch, kv, BLOCK, slot), device="meta")
    table = torch.empty((batch, pages), device="meta")
    selected, buffers = kernel.prefill_workspace(
        query, cache, table, chunk + prefix, 1 / 16, budget, prefix
    )
    assert selected == path
    assert (
        sum(
            round_up(math.prod(shape) * dtype.itemsize, 256) for shape, dtype in buffers
        )
        <= budget
    )


@pytest.mark.skipif(ARCH != "rdna4", reason="RDNA4 streamed prefill")
def test_startup_budget_preserves_single_head_split_count(kernel):
    from vllm.v1.attention.ops.rocm_octave_rdna4.tuned import startup_scratch_bytes

    tokens, capacity = 160, 8192
    q = torch.empty((tokens, 1, HEAD), device="meta")
    cache = torch.empty(
        (capacity // BLOCK, 1, BLOCK, kernel.layout("octave_k3v4").slot_bytes),
        device="meta",
    )
    table = torch.empty((1, capacity // BLOCK), device="meta")
    budget = startup_scratch_bytes(
        tokens=tokens, heads=1, kv=1, dim=HEAD, requests=1, capacity=capacity
    )
    path, buffers = kernel.prefill_workspace(q, cache, table, capacity, 1 / 16, budget)
    assert path == "streamed"
    partials, *_ = buffers
    assert partials[0][0] == 32


@pytest.mark.skipif(ARCH != "rdna4", reason="RDNA4 graph workspace")
def test_serving_reserves_bounded_workspace_before_capture(
    tmp_path, vllm_runner, enable_pickle
):
    from transformers import LlamaConfig

    from vllm import SamplingParams

    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        head_dim=HEAD,
        num_attention_heads=16,
        num_key_value_heads=4,
        num_hidden_layers=1,
        max_position_embeddings=4096,
        partial_rotary_factor=ROPE / HEAD,
        vocab_size=128,
    ).save_pretrained(tmp_path)

    def workspace_bytes(_model):
        from vllm.v1.worker.workspace import current_workspace_manager

        (view,) = current_workspace_manager().get_simultaneous(((1,), torch.uint8))
        return view.untyped_storage().nbytes()

    with vllm_runner(
        str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        attention_backend="ROCM_OCTAVE",
        kv_cache_dtype="octave_k3v4",
        max_model_len=4096,
        max_num_batched_tokens=1024,
        max_num_seqs=2,
        kv_cache_memory_bytes=64 << 20,
        gpu_memory_utilization=0.5,
        enable_chunked_prefill=True,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "decode_cudagraph_capture_request_sizes": [1, 2],
        },
    ) as runner:
        (before,) = runner.llm.apply_model(workspace_bytes)
        # A shared reservation below 300 MiB covers both prefill routes and decode.
        assert 0 < before <= 300 << 20
        for length in (250, 694):
            (result,) = runner.llm.generate(
                [{"prompt_token_ids": [7] * length}],
                SamplingParams(
                    temperature=0, max_tokens=3, ignore_eos=True, detokenize=False
                ),
            )
            assert len(result.outputs[0].token_ids) == 3
        assert runner.llm.apply_model(workspace_bytes) == [before]


@pytest.mark.skipif(ARCH != "rdna4", reason="RDNA4 prefill workspace")
@pytest.mark.parametrize(
    "tokens,requests,pages,hint,budget,path",
    [
        (8, 1, 33, None, 2 << 30, "streamed"),
        (159, 16, 73, 1153, 2 << 30, "streamed"),
        # Expansion can fit even when one streamed split is too large.
        (128, 1, 8, 128, 3 << 20, "expanded"),
        (256, 1, 1296, 20736, 1 << 20, "packed"),
        (32768, 128, 4096, 65536, 2 << 30, "packed"),
    ],
)
def test_prefill_fallback_needs_workspace(
    tokens, requests, pages, hint, budget, path, kernel
):
    heads = 64 if tokens == 32768 else 32
    q = torch.empty((tokens, heads, HEAD), device="meta")
    cache = torch.empty(
        (pages, 4, BLOCK, kernel.layout("octave_k3v4").slot_bytes), device="meta"
    )
    table = torch.empty((requests, pages), device="meta")
    selected, _ = kernel.prefill_workspace(q, cache, table, hint, 1 / 16, budget)
    assert selected == path
    if path == "packed":
        assert (
            kernel.suggest_splits(
                q,
                cache,
                table,
                "octave_k3v4",
                prefill=True,
                context_capacity=hint,
                budget=budget,
            )
            == 1
        )


@pytest.mark.skipif(ARCH != "rdna4", reason="RDNA4 FP32 prefill repair")
@pytest.mark.parametrize("path", ["streamed", "expanded"])
def test_prefill_preserves_finite_bf16_values_above_fp16_range(path, kernel):
    fmt = kernel.layout("octave_k3v4")
    tokens = 129
    q = torch.zeros((tokens, 4, HEAD), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q[:, :1])
    v = torch.full_like(k, 131072)
    pages = math.ceil(tokens / BLOCK)
    cache = torch.zeros(
        (pages, 1, BLOCK, fmt.slot_bytes), device="cuda", dtype=torch.uint8
    )
    table = torch.arange(pages, device="cuda", dtype=torch.int32)[None]
    starts = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    out = _prefill(
        cache,
        q,
        k,
        v,
        table,
        starts,
        starts[1:],
        sq.sign_bits(HEAD).to("cuda"),
        fmt,
        kernel,
        path=path,
    )
    torch.testing.assert_close(out, torch.full_like(out, 131072), rtol=1e-6, atol=0)


@pytest.mark.skipif(ARCH != "rdna4", reason="RDNA4 packed-prefix dispatch")
def test_uniform_cached_prefill_dispatch_uses_shortest_prefix(kernel):
    from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
    from vllm.config import SchedulerConfig, VllmConfig
    from vllm.v1.attention.backends.rocm_octave_rdna4_attn import (
        RocmOctaveRdna4MetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import AttentionSpec

    cfg = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_model_len=32768, max_num_batched_tokens=8192, is_encoder_decoder=False
        )
    )
    spec = AttentionSpec(
        block_size=3264, num_kv_heads=2, head_size=HEAD, dtype=torch.bfloat16
    )
    builder = RocmOctaveRdna4MetadataBuilder(spec, ["layer"], cfg, torch.device("cpu"))
    slot_bytes = kernel.layout("octave_k3v4").slot_bytes
    cache = torch.empty((12, 2, 3264, slot_bytes), device="meta", dtype=torch.uint8)
    for chunks, lengths, prefix, path in (
        ([1, 2112, 2112], [16, 18432, 18432], 16320, "expanded"),
        ([1, 512, 512], [16, 131584, 131584], 131072, "streamed"),
        ([2112, 2112], [2112, 18432], 0, "expanded"),
        ([2111, 2112], [18432, 18432], 0, "expanded"),
    ):
        common = create_common_attn_metadata(BatchSpec(lengths, chunks), 3264, "cpu")
        md = builder.build(0, common)
        assert md.uniform_prefill_prefix == prefix
        query = torch.empty(
            (sum(chunks) - md.num_decode_tokens, 12, HEAD), device="meta"
        )
        table = md.block_table[md.num_decodes :]
        assert (
            kernel.prefill_workspace(
                query,
                cache,
                table,
                md.max_seq_len,
                1 / 16,
                2 << 30,
                md.uniform_prefill_prefix,
            )[0]
            == path
        )


@pytest.mark.skipif(ARCH != "rdna4", reason="RDNA4 chunked prefill")
@pytest.mark.parametrize("path", ["expanded", "streamed"])
@pytest.mark.parametrize("heads", [5, 6])
def test_prefill_replay_changes_live_prefix_and_repair_rows(path, heads, kernel):
    torch.manual_seed(22)
    device, dtype = "cuda", torch.bfloat16
    fmt = kernel.layout("octave_k3v4")
    tokens, max_prefix = 64, 528
    signs = sq.sign_bits(HEAD).to(device)

    # Strided activations and padded table rows, as in mixed batches.
    def sample(n, h):
        return torch.randn(n, h + 1, HEAD, device=device, dtype=dtype)[:, :h]

    q, k, v = sample(tokens, heads), sample(tokens, 1), sample(tokens, 1)
    pages = (
        max_prefix // BLOCK
        if path == "streamed"
        else math.ceil((max_prefix + tokens) / BLOCK)
    )
    table = torch.randperm(2 * (pages + 1), device=device).int().view(2, -1)[:, :pages]
    cache = torch.zeros(
        2 * (pages + 1), 1, BLOCK, fmt.slot_bytes, dtype=torch.uint8, device=device
    )
    pos = torch.arange(max_prefix, device=device)
    prefix_kv = []
    for row in table:
        slots = row[pos // BLOCK].long() * BLOCK + pos % BLOCK
        kernel.store(
            sample(max_prefix, 1),
            sample(max_prefix, 1),
            cache,
            slots,
            signs,
            "octave_k3v4",
        )
        prefix_kv.append(
            sq.reference_dequantize(cache[row[pos // BLOCK], :, pos % BLOCK], fmt)
        )

    starts = torch.tensor([0, 24, tokens], device=device, dtype=torch.int32)
    lengths = torch.tensor([24, tokens - 24], device=device, dtype=torch.int32)
    out = torch.empty(q.shape, device=device, dtype=torch.float32)
    args = (q, k, v, cache, table, starts, lengths, signs, "octave_k3v4")
    if path == "expanded":
        # The second replay exceeds this capacity for only one request.
        tmp = torch.empty((2, 1, 128, HEAD), device=device, dtype=torch.float16)
        flags = torch.empty((tokens, heads, 2), device=device, dtype=torch.uint8)
        kw = dict(
            key_tmp=tmp,
            value_tmp=torch.empty_like(tmp),
            repair=flags,
            query_tmp=torch.empty(q.shape, device=device, dtype=torch.float16),
        )
        op = kernel.prefill_expanded
    else:
        shapes = kernel.streamed_workspace(q, cache, 2, max_prefix, 2 << 30)
        buffers = [torch.empty(sh, dtype=dt, device=device) for sh, dt in shapes]
        flags = buffers[-1]
        kw = dict(buffers=buffers)
        op = kernel.prefill_streamed

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*args, out=out, scale=1 / 16, **kw)
    replays = [(24, [33, 0]), (31, [0, 519])]
    if path == "streamed":
        # Row padding has a valid physical page outside the logical table.
        # Both the WMMA result and the repaired row must stop at its width.
        replays.append((23, [max_prefix + 11, -7]))
    replays.append((17, [30, 7]))
    for cut, prefixes in replays:
        starts[1] = cut
        lengths.copy_(
            torch.tensor(
                [cut + prefixes[0], tokens - cut + prefixes[1]],
                device=device,
                dtype=torch.int32,
            )
        )
        # A previously valid finite output and stale partials must be ignored.
        out.fill_(7)
        sharp = cut in (31, 23)
        q[0, 0, HEAD // 2 :] = 1000 if sharp else 1
        graph.replay()
        if sharp:
            assert flags[0, 0].any()
        else:
            assert not flags[0, 0].any()
        for i, (begin, end) in enumerate(((0, cut), (cut, tokens))):
            prefix = max(0, min(max_prefix, prefixes[i]))
            pk, pv = prefix_kv[i]
            keys = torch.cat((pk[:prefix, 0], k[begin:end, 0].float()))
            vals = torch.cat((pv[:prefix, 0], v[begin:end, 0].float()))
            logits = q[begin:end].float() @ keys.T / 16
            causal = torch.arange(len(keys), device=device) > (
                prefix + torch.arange(end - begin, device=device)[:, None]
            )
            ref = logits.masked_fill(causal[:, None], -torch.inf).softmax(-1) @ vals
            assert _rel(out[begin:end], ref) < 1e-3

    # A genuine data NaN survives numerical repair; it is never the route flag.
    v[0, 0, 3] = torch.nan
    graph.replay()
    assert not flags[0].any()
    assert out[0, :, 3].isnan().all()
    assert out[0, :, 4:].isfinite().all()


def _decode(
    cache,
    q,
    block_table,
    q_to_req,
    q_to_klen,
    signs,
    fmt,
    use_wmma,
    kernel=None,
    *,
    num_splits=8,
    query_group=1,
):
    out = torch.zeros_like(q)
    mid = torch.empty(
        *q.shape[:2], num_splits, HEAD + 2, dtype=torch.float32, device=q.device
    )
    if kernel is not None:
        return kernel.decode(
            q,
            cache,
            block_table,
            q_to_req,
            q_to_klen,
            signs,
            _cache_dtype(fmt),
            scale=1 / 16,
            splits=num_splits,
            scratch=mid,
        ).to(q.dtype)
    torch.ops._C.octave_decode(
        out,
        q,
        cache,
        block_table,
        q_to_req,
        q_to_klen,
        mid,
        signs,
        signs,
        1 / 16,
        num_splits,
        fmt.kernel_code,
        query_group,
        use_wmma,
    )
    return out


def _prefill(
    cache,
    q,
    k,
    v,
    block_table,
    qsl,
    seq_lens,
    signs,
    fmt,
    kernel=None,
    *,
    path="packed",
):
    if kernel is not None:
        cache_dtype = _cache_dtype(fmt)
        if path == "streamed":
            shapes = kernel.streamed_workspace(q, cache, len(seq_lens), 4096, 2 << 30)
            buffers = [torch.empty(s, device=q.device, dtype=d) for s, d in shapes]
            return kernel.prefill_streamed(
                q,
                k,
                v,
                cache,
                block_table,
                qsl,
                seq_lens,
                signs,
                cache_dtype,
                buffers=buffers,
                out=torch.empty_like(q, dtype=torch.float32),
                scale=1 / 16,
            )
        if path == "expanded":
            capacity = math.ceil(block_table.shape[1] * cache.shape[2] / 32) * 32
            shape = (block_table.shape[0], cache.shape[1], capacity, HEAD)
            key_tmp = torch.empty(shape, device=q.device, dtype=torch.float16)
            return kernel.prefill_expanded(
                q,
                k,
                v,
                cache,
                block_table,
                qsl,
                seq_lens,
                signs,
                cache_dtype,
                key_tmp=key_tmp,
                value_tmp=torch.empty_like(key_tmp),
                query_tmp=torch.empty_like(q, dtype=torch.float16),
                out=torch.empty_like(q, dtype=torch.float32),
                scale=1 / 16,
            )
        return kernel.prefill(
            q,
            k,
            v,
            cache,
            block_table,
            qsl,
            seq_lens,
            signs,
            cache_dtype,
            scale=1 / 16,
            splits=kernel.suggest_splits(
                q, cache, block_table, cache_dtype, prefill=True
            ),
        ).to(q.dtype)
    q, k, v = q.clone(), k.clone(), v.clone()
    torch.ops._C.octave_rotate(q, signs, not fmt.compact, False)
    torch.ops._C.octave_rotate(k, signs, not fmt.compact, False)
    torch.ops._C.octave_rotate(v, signs, False, False)
    out = torch.empty_like(q)
    torch.ops._C.octave_prefill(
        out,
        q,
        k,
        v,
        cache,
        block_table,
        qsl,
        seq_lens,
        int((qsl[1:] - qsl[:-1]).max()),
        1 / 16,
        fmt.kernel_code,
    )
    torch.ops._C.octave_rotate(out, signs, False, True)
    return out


@pytest.mark.skipif(ARCH != "rdna4", reason="concurrent test targets RDNA4 runtime")
def test_rdna4_concurrent_decode_has_per_call_state(kernel):
    """Two HIP streams can use the shared geometry engine without state leaks."""
    torch.manual_seed(11)
    dev, fmt, hkv, group, n = "cuda", _format("octave_k3v4"), 1, 6, 73
    signs = sq.sign_bits(HEAD).to(dev)
    cache = torch.zeros(8, hkv, BLOCK, fmt.slot_bytes, dtype=torch.uint8, device=dev)
    key = torch.randn(n, hkv, HEAD, device=dev).bfloat16()
    value = torch.randn_like(key)
    pages = torch.arange(cache.shape[0], dtype=torch.int32, device=dev)
    _fill(cache, pages, key, value, signs, signs, fmt, kernel)
    table = pages[: math.ceil(n / BLOCK)].view(1, -1).contiguous()
    req, lengths = (
        torch.zeros(1, dtype=torch.int32, device=dev),
        torch.tensor([n], dtype=torch.int32, device=dev),
    )
    queries = [
        torch.randn(1, hkv * group, HEAD, device=dev).to(dtype)
        for dtype in (torch.float16, torch.bfloat16)
    ]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    torch.accelerator.synchronize()
    for stream, query in zip(streams, queries):
        with torch.cuda.stream(stream):
            outputs.append(
                _decode(
                    cache,
                    query,
                    table,
                    req,
                    lengths,
                    signs,
                    fmt,
                    True,
                    kernel,
                    num_splits=37,
                )
            )
    for stream in streams:
        stream.synchronize()
    for query, actual in zip(queries, outputs):
        expected = sq.reference_attention(
            query.float(), cache, table, req, lengths, fmt, 1 / 16
        )
        assert _rel(actual, expected) < 2e-2


@pytest.mark.skipif(ARCH != "rdna4", reason="RDNA4-only Octave ops")
def test_rdna4_precompiled_runtime_never_invokes_a_compiler(monkeypatch):
    """Native store/attention/prefill and replay work with compilers disabled."""
    import subprocess

    from vllm.v1.attention.ops.rocm_octave_rdna4 import TunedHipOctave

    def compiler_forbidden(*args, **kwargs):
        raise AssertionError("runtime Octave must not launch a compiler")

    monkeypatch.setattr(subprocess, "run", compiler_forbidden)
    monkeypatch.setattr(subprocess, "Popen", compiler_forbidden)
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    runtime = TunedHipOctave(HEAD, ROPE, arch)

    torch.manual_seed(12)
    dev, fmt, cache_dtype = "cuda", _format("octave_k3v4"), "octave_k3v4"
    signs = sq.sign_bits(HEAD).to(dev)
    cache = torch.zeros(4, 1, BLOCK, fmt.slot_bytes, dtype=torch.uint8, device=dev)
    table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=dev)
    key, value = (
        torch.randn(32, 1, HEAD, device=dev).bfloat16(),
        torch.randn(32, 1, HEAD, device=dev).bfloat16(),
    )
    _store(key, value, cache, torch.arange(32, device=dev), signs, signs, fmt, runtime)
    q = torch.randn(1, 6, HEAD, device=dev).bfloat16()
    req = torch.zeros(1, dtype=torch.int32, device=dev)
    lengths = torch.tensor([32], dtype=torch.int32, device=dev)
    eager = runtime.decode(
        q, cache, table, req, lengths, signs, cache_dtype, scale=1 / 16, splits=37
    )
    ref = sq.reference_attention(q.float(), cache, table, req, lengths, fmt, 1 / 16)
    assert _rel(eager, ref) < 2e-2

    k_new, v_new = (
        torch.randn(2, 1, HEAD, device=dev).bfloat16(),
        torch.randn(2, 1, HEAD, device=dev).bfloat16(),
    )
    prefill = runtime.prefill(
        q.expand(2, -1, -1).contiguous(),
        k_new,
        v_new,
        cache,
        table,
        torch.tensor([0, 2], dtype=torch.int32, device=dev),
        torch.tensor([34], dtype=torch.int32, device=dev),
        signs,
        cache_dtype,
        scale=1 / 16,
    )
    assert prefill.shape == (2, 6, HEAD)
    assert torch.isfinite(prefill).all()

    out = torch.empty((1, 6, HEAD), dtype=torch.float32, device=dev)
    scratch = torch.empty((1, 6, 37, HEAD + 2), dtype=torch.float32, device=dev)

    def graph_decode():
        runtime.decode(
            q,
            cache,
            table,
            req,
            lengths,
            signs,
            cache_dtype,
            scale=1 / 16,
            splits=37,
            out=out,
            scratch=scratch,
        )

    graph_decode()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_decode()
    q.add_(0.5)
    graph.replay()
    ref = sq.reference_attention(q.float(), cache, table, req, lengths, fmt, 1 / 16)
    assert _rel(out, ref) < 2e-2


@pytest.mark.parametrize(
    "layout", RocmOctaveAttentionBackend.supported_kv_cache_layouts()
)
def test_kernels_follow_declared_cache_layouts(layout, kernel):
    """Store, decode and prefill on the cache view vLLM allocates for each
    declared layout match a contiguous cache bit for bit. Two KV heads, so the
    head and token strides differ between layouts."""
    torch.manual_seed(4)
    dev, cache_dtype = "cuda", "octave_k3v3"
    fmt, hkv, group, nblk = _format(cache_dtype), 2, 6, 80
    signs = sq.sign_bits(HEAD).to(dev)
    sq.register_rope_dim(HEAD, ROPE)
    spec = RocmOctaveAttentionBackend.customize_spec(
        AttentionSpec(
            block_size=BLOCK,
            num_kv_heads=hkv,
            head_size=HEAD,
            dtype=torch.uint8,
            kv_quant_mode=get_kv_quant_mode(cache_dtype),
        )
    )
    raw = torch.zeros(nblk * spec.page_size_bytes, dtype=torch.int8, device=dev)
    cache = dense_kv_cache_views(raw, spec, nblk, 1, layout)[0]
    dense = torch.zeros(nblk, hkv, BLOCK, fmt.slot_bytes, dtype=torch.uint8, device=dev)
    assert cache.shape == dense.shape and cache.dtype == dense.dtype

    seq_lens, query_lens = [700, 300], [100, 300]
    max_blocks = max(math.ceil(n / BLOCK) for n in seq_lens)
    block_table = torch.zeros(len(seq_lens), max_blocks, dtype=torch.int32, device=dev)
    free, used, ks, vs = torch.randperm(nblk, device=dev), 0, [], []
    for i, n in enumerate(seq_lens):
        nb = math.ceil(n / BLOCK)
        block_table[i, :nb] = free[used : used + nb].int()
        used += nb
        ks.append(torch.randn(n, hkv, HEAD, device=dev).half())
        vs.append(torch.randn(n, hkv, HEAD, device=dev).half())
        for c in (cache, dense):
            _fill(c, block_table[i, :nb], ks[i], vs[i], signs, signs, fmt, kernel)
    assert torch.equal(cache, dense)

    q_to_req = torch.arange(len(seq_lens), dtype=torch.int32, device=dev)
    q_to_klen = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    q = torch.randn(len(seq_lens), hkv * group, HEAD, device=dev).half()
    for use_wmma in (True,) if kernel is not None else (True, False):
        out, ref = (
            _decode(
                c, q, block_table, q_to_req, q_to_klen, signs, fmt, use_wmma, kernel
            )
            for c in (cache, dense)
        )
        assert out.abs().sum() > 0
        assert torch.equal(out, ref)

    qsl = torch.tensor([0, *query_lens], device=dev).cumsum(0).int()
    q = torch.randn(int(qsl[-1]), hkv * group, HEAD, device=dev).half()
    ctx = [n - m for n, m in zip(seq_lens, query_lens)]
    k = torch.cat([x[c:] for x, c in zip(ks, ctx)])
    v = torch.cat([x[c:] for x, c in zip(vs, ctx)])
    out, ref = (
        _prefill(c, q, k, v, block_table, qsl, q_to_klen, signs, fmt, kernel)
        for c in (cache, dense)
    )
    assert torch.equal(out, ref)


def _window_tables(entries, hkv, dev, dtype=torch.float16):
    return (
        torch.zeros(entries, hkv, 2 * HEAD, dtype=dtype, device=dev),
        torch.full((entries,), -1, dtype=torch.int32, device=dev),
        torch.zeros(entries, dtype=torch.int32, device=dev),
    )


@pytest.mark.skipif(
    ARCH not in ("rdna3", "rdna4"), reason="sparse decode needs RDNA3 or RDNA4"
)
@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES)
@pytest.mark.parametrize(
    "hkv,group,splits,window,dtype",
    [
        (hkv, 6, 64, window, torch.float16)
        for hkv in (1, 2)
        for window in ("off", "on", "evicted")
    ]
    + (
        [(1, 12, 32, "on", torch.bfloat16), (1, 12, 1, "evicted", torch.bfloat16)]
        if ARCH == "rdna4"
        else []
    ),
)
def test_sparse_decode_matches_reference(
    cache_dtype, hkv, group, splits, window, dtype, sparse_kernel
):
    """Sparse (top-k) decode as qwen4_exp QSA runs it: each query reads the
    logical positions in its index row, -1 entries are holes. With the window
    on, the first 4 and the last 64 positions come from FP16/BF16 tables;
    an entry whose tag was overwritten falls back to the packed cache."""
    torch.manual_seed(4)
    kernel = sparse_kernel
    dev, fmt = "cuda", _format(cache_dtype)
    signs = sq.sign_bits(HEAD).to(dev)
    lens, max_blocks, nblk = [1500, 2600, 777], 192, 1024
    cache = torch.zeros(nblk, hkv, BLOCK, fmt.slot_bytes, dtype=torch.uint8, device=dev)
    block_table = (
        torch.randperm(nblk, device=dev)[: len(lens) * max_blocks]
        .view(len(lens), max_blocks)
        .int()
    )
    wtab, wtags, wlocks = _window_tables(4096, hkv, dev, dtype)
    stab, stags, slocks = _window_tables(256, hkv, dev, dtype)
    ks, vs = [], []
    for r, n in enumerate(lens):
        k = torch.randn(n, hkv, HEAD, device=dev).to(dtype)
        v = torch.randn_like(k)
        pos = torch.arange(n, device=dev)
        slots = block_table[r, pos // BLOCK].long() * BLOCK + pos % BLOCK
        _store(k, v, cache, slots, signs, signs, fmt, kernel)
        window_store = (
            kernel.window_store
            if kernel is not None
            else torch.ops._C.octave_window_store
        )
        window_store(k, v, slots, pos.int(), wtab, wtags, wlocks, stab, stags, slocks)
        ks.append(k)
        vs.append(v)
    if window == "evicted":
        wtags[::2] = -7
        stags[::2] = -7

    q_to_req = torch.tensor([0, 1, 1, 2, 0, 1, 2], dtype=torch.int32, device=dev)
    num_q, topk = q_to_req.numel(), 512
    qpos = torch.tensor(
        [lens[r] - 1 - 37 * i for i, r in enumerate(q_to_req.tolist())],
        dtype=torch.int32,
        device=dev,
    )
    indices = torch.full((num_q, topk), -1, dtype=torch.int32, device=dev)
    for i in range(num_q):
        n = int(qpos[i]) + 1
        cand = torch.cat(
            [
                torch.arange(max(0, n - 80), n, device=dev),
                torch.arange(4, device=dev),
                torch.randperm(n, device=dev)[:300],
            ]
        ).unique()
        indices[i, : cand.numel()] = cand.int()
        indices[i] = indices[i][torch.randperm(topk, device=dev)]

    q = torch.randn(num_q, hkv * group, HEAD, device=dev).to(dtype)
    if group == 12 and window == "on":
        # Mix a row outside the WMMA input range with ordinary rows.
        q[0].mul_(48)
    out = torch.empty_like(q)
    mid = torch.empty(
        num_q, hkv * group, splits, HEAD + 2, dtype=torch.float32, device=dev
    )
    scale = HEAD**-0.5
    window_args = () if window == "off" else (qpos, wtab, wtags, stab, stags)
    if kernel is None:
        torch.ops._C.octave_decode_sparse(
            out,
            q,
            cache,
            block_table,
            q_to_req,
            indices,
            mid,
            signs,
            signs,
            scale,
            splits,
            fmt.kernel_code,
            *window_args,
        )
    else:
        fp32_out = torch.empty_like(q, dtype=torch.float32)
        exact_kwargs = (
            {}
            if window == "off"
            else dict(positions=qpos, wtab=wtab, wtags=wtags, stab=stab, stags=stags)
        )

        def call():
            out.copy_(
                kernel.decode_sparse(
                    q,
                    cache,
                    block_table,
                    q_to_req,
                    indices,
                    signs,
                    cache_dtype,
                    scale=scale,
                    splits=splits,
                    out=fp32_out,
                    scratch=mid,
                    **exact_kwargs,
                )
            )

        call()

    ref = torch.zeros(num_q, hkv * group, HEAD, device=dev)
    exact_hits = 0
    for i in range(num_q):
        r, p = int(q_to_req[i]), int(qpos[i])
        sel = indices[i][indices[i] >= 0].long()
        blocks = block_table[r, sel // BLOCK].long()
        k, v = sq.reference_dequantize(cache[blocks, :, sel % BLOCK], fmt)
        if window != "off":
            slot = blocks * BLOCK + sel % BLOCK
            exact = ((p - sel) < 64) & (wtags[slot % 4096].long() == slot)
            exact |= (sel < 4) & (stags[slot % 256].long() == slot)
            exact_hits += exact.sum().item()
            k[exact] = ks[r][sel[exact]].float()
            v[exact] = vs[r][sel[exact]].float()
        k = k.repeat_interleave(group, 1)
        v = v.repeat_interleave(group, 1)
        p_attn = torch.softmax(torch.einsum("hd,thd->ht", q[i].float(), k) * scale, -1)
        ref[i] = torch.einsum("ht,thd->hd", p_attn, v)
    assert _rel(out, ref) < 2e-2
    if window != "off":
        assert exact_hits > 0
    if (
        kernel is not None
        and group == 12
        and window == "on"
        and cache_dtype == "octave_k3v4"
    ):
        eager = out.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        out.fill_(float("nan"))
        graph.replay()
        assert torch.equal(out, eager)
        indices.fill_(-1)
        graph.replay()
        assert torch.count_nonzero(out).item() == 0
        # The only selected value is exact BF16, outside the FP16 tile range.
        r, p = int(q_to_req[1]), int(qpos[1])
        slot = int(block_table[r, p // BLOCK]) * BLOCK + p % BLOCK
        entry = slot % wtab.shape[0]
        exact_v = wtab[entry, :, HEAD:]
        exact_v.zero_()
        exact_v[:, 0] = 2**21
        wtags[entry] = slot
        indices[1, 0] = p
        graph.replay()
        torch.testing.assert_close(out[1], exact_v.expand_as(out[1]), rtol=0, atol=0)
