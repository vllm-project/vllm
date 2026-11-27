# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY V4 Flash IndexPool and Engram kernels match reference math.

The NVIDIA fused kernels, the ROCm IndexPool ops and the shared Engram kernels
are each checked on the platform that runs them.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype, set_random_seed

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="NVIDIA HY V4 Flash kernels require CUDA"
)
requires_rocm = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="AMD HY V4 Flash kernels require ROCm"
)
requires_gpu = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="CUDA or ROCm required"
)

_ROPE = {
    "mrope_interleaved": True,
    "mrope_section": [12, 10, 10],
    "rope_theta": 10000000,
    "rope_type": "default",
}


@pytest.fixture(autouse=True)
def _seed():
    set_random_seed(0)


def _rotate_reference(x, positions):
    inverse = _ROPE["rope_theta"] ** (
        -torch.arange(0, 64, 2, device=x.device, dtype=torch.float32) / 64
    )
    if positions.ndim == 1:
        frequency = positions.float()[:, None] * inverse
    else:
        planes = positions.float()[:, :, None] * inverse
        lanes = torch.arange(32, device=x.device)
        frequency = torch.where((lanes % 3 == 1) & (lanes < 30), planes[1], planes[0])
        frequency = torch.where((lanes % 3 == 2) & (lanes < 30), planes[2], frequency)
    nope, pe = x.split((64, 64), dim=-1)
    pe = pe.reshape(*pe.shape[:-1], 32, 2).transpose(-2, -1).reshape(pe.shape)
    frequency = torch.cat((frequency, frequency), -1)
    if pe.ndim == 3:
        frequency = frequency[:, None]
    left, right = pe.float().chunk(2, dim=-1)
    rotated = (
        pe.float() * frequency.cos() + torch.cat((-right, left), -1) * frequency.sin()
    )
    return torch.cat((nope, rotated.to(x.dtype)), dim=-1)


def _fwht_quantize_reference(vectors):
    x = vectors.float()
    stride = 1
    while stride < 128:
        pairs = x.reshape(-1, 2, stride)
        left, right = pairs[:, 0], pairs[:, 1]
        x = torch.stack((left + right, left - right), dim=1).reshape(vectors.shape)
        stride *= 2
    x = x * 128**-0.5
    scale = torch.exp2(torch.ceil(torch.log2(x.abs().amax(-1).clamp_min(1e-4) / 448)))
    quant = (x / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quant, scale


@requires_cuda
@pytest.mark.parametrize("mx_scales,three_axis", [(True, True), (False, False)])
def test_pool_quantize_write(mx_scales, three_axis):
    from vllm.models.hy_v4_flash.nvidia.indexpool_ops import pool_quantize_write

    slots, ratio, page, blocks = 29, 4, 64, 64
    # Values and gates are the two halves of one fused projection.
    projection = torch.randn(slots, ratio, 256, device="cuda")
    values, gates = projection[..., :128], projection[..., 128:]
    ape = torch.randn(ratio, 128, device="cuda")
    norm_weight = torch.rand(128, device="cuda") + 0.5
    positions = torch.randint(
        0, 50000, (3, slots) if three_axis else (slots,), device="cuda"
    )
    table = torch.randperm(blocks, device="cuda").int().reshape(4, 16)
    owners = torch.randint(0, 4, (slots,), device="cuda").int()
    groups = torch.arange(slots, device="cuda") * 7
    active = torch.rand(slots, device="cuda") > 0.2

    probabilities = (gates + ape).softmax(1)
    pooled = F.rms_norm((values * probabilities).sum(1), (128,), norm_weight, 1e-5)
    quant, scale = _fwht_quantize_reference(
        _rotate_reference(pooled.bfloat16(), positions)
    )
    scale_bits = scale.view(torch.int32)
    if mx_scales:
        scale_bits = (scale_bits >> 23) * 0x01010101
    expected = torch.zeros(blocks, page, 132, dtype=torch.uint8, device="cuda")
    for slot in active.nonzero().flatten().tolist():
        group = int(groups[slot])
        block = int(table[owners[slot], group // page])
        row = group % page
        # A page holds `page` rows of 128 FP8 bytes, then `page` 4-byte scales.
        flat = expected[block].view(-1)
        flat[row * 128 : (row + 1) * 128] = quant[slot].view(torch.uint8)
        flat[page * 128 :].view(torch.int32)[row] = scale_bits[slot]

    cache = torch.zeros_like(expected)
    pool_quantize_write(
        projection,
        ape,
        norm_weight,
        1e-5,
        positions,
        _ROPE,
        cache,
        table,
        owners,
        groups,
        active,
        mx_scales,
    )
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("width", [4096, 200])
@pytest.mark.parametrize("backend", ["auto", "per_row"])
def test_select_expand_groups(width, backend):
    from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk
    from vllm.models.hy_v4_flash.nvidia.indexpool_ops import select_expand_groups

    rows, budget, ratio = 40, 511, 4
    logits = torch.randn(rows, width, device="cuda")
    topk = SparseIndexerTopk(backend)
    try:
        topk.resolve_backend(logits, min(budget, width), rows)
    except RuntimeError as e:
        pytest.skip(str(e))
    logical = torch.randint(-1, width * ratio, (rows,), device="cuda")
    # A row sees at most the width of its logits.
    upper = (logical // ratio).clamp_min(0).int()
    active = torch.rand(rows, device="cuda") > 0.1

    columns = torch.arange(width, device="cuda")
    scores = logits.masked_fill(columns[None] >= upper[:, None], -torch.inf)
    values, indices = scores.topk(min(budget, width), dim=-1)
    sentinel = torch.iinfo(torch.int32).max
    groups = torch.where(values != -torch.inf, indices, sentinel).int().sort(-1).values
    groups = F.pad(
        groups.masked_fill(groups == sentinel, -1),
        (0, budget - groups.shape[1]),
        value=-1,
    )
    live = (logical >= 0) & active
    history = groups[..., None] * ratio + torch.arange(ratio, device="cuda")
    valid = (groups >= 0) & (groups < upper[:, None]) & live[:, None]
    history = torch.where(valid[..., None], history, -1)
    tail = upper[:, None] * ratio + torch.arange(ratio, device="cuda")
    tail = torch.where(live[:, None] & (tail <= logical[:, None]), tail, -1)
    candidates = torch.cat((history.flatten(1), tail), -1).int()
    order = torch.argsort((candidates < 0).int(), dim=-1, stable=True)
    expected = candidates.gather(-1, order)

    out = torch.empty(rows, budget * ratio + ratio, dtype=torch.int32, device="cuda")
    select_expand_groups(logits, upper, budget, logical, active, ratio, out, topk)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@requires_cuda
def test_prepare_groups_query_metadata():
    """Per-query scoring inputs replace the eager clamp, cast and table gather."""
    query_lens = [5, 0, 300, 2, 1]
    from vllm.models.hy_v4_flash.nvidia.indexpool_prepare import prepare_groups

    ratio, cols = 4, 70
    batch = len(query_lens)
    lens = torch.tensor(query_lens)
    starts = torch.cat((torch.zeros(1, dtype=torch.long), lens.cumsum(0)))
    seq_lens = lens + torch.randint(0, 2000, (batch,))
    tokens = int(starts[-1])
    keys = torch.randn(tokens, 128, device="cuda").bfloat16()
    tail = torch.zeros(batch, ratio, 132, device="cuda")
    tail_table = torch.arange(batch, device="cuda").int()[:, None]
    pool_table = torch.randint(0, 1000, (batch, cols), device="cuda").int()
    (*_, owner, logical, upper, score_lengths, token_table, live) = prepare_groups(
        keys,
        torch.arange(tokens, device="cuda"),
        tail,
        tail_table,
        pool_table,
        starts.int().cuda(),
        seq_lens.int().cuda(),
        tokens,
        ratio,
    )

    request = torch.repeat_interleave(torch.arange(batch), lens)
    context = seq_lens - lens
    expected_logical = context[request] + torch.arange(tokens) - starts[request]
    expected_upper = (expected_logical // ratio).int().cuda()
    request = request.cuda()
    assert torch.equal(owner, request.int())
    assert torch.equal(logical.cpu(), expected_logical)
    assert torch.equal(upper, expected_upper)
    assert torch.equal(score_lengths[:, 0], expected_upper.clamp_min(1))
    assert torch.equal(token_table, pool_table[request])
    assert bool(live.all())


@requires_cuda
def test_window_indices():
    from vllm.models.hy_v4_flash.nvidia.swa import _window_indices_kernel
    from vllm.triton_utils import triton

    query_lens, window, width = [5, 1, 300, 2, 1], 512, 640
    lens = torch.tensor(query_lens)
    starts = torch.cat((torch.zeros(1, dtype=torch.long), lens.cumsum(0)))
    starts = starts.int().cuda()
    seq_lens = (lens + torch.randint(0, 2000, (len(query_lens),))).int().cuda()
    num_tokens = int(starts[-1])
    rows = num_tokens + 9

    row = torch.arange(rows, device="cuda")
    request = torch.searchsorted(starts[1:], row, right=True)
    request = request.clamp_max(len(query_lens) - 1).long()
    position = seq_lens[request] - (starts[request + 1] - starts[request])
    position = torch.where(row < num_tokens, position + row - starts[request], -1)
    column = torch.arange(width, device="cuda")
    index = position[:, None] - window + 1 + column
    keep = (position[:, None] >= 0) & (column < window) & (index >= 0)
    expected = torch.where(keep, index, -1).int()

    out = torch.empty(rows, width, dtype=torch.int32, device="cuda")
    _window_indices_kernel[(rows,)](
        out,
        starts,
        seq_lens,
        len(query_lens),
        num_tokens,
        out.stride(0),
        window,
        width,
        triton.next_power_of_2(width),
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@requires_gpu
def test_engram_fused_update():
    from vllm.models.hy_v4_flash.common.engram_ops import gated_update, lookup_norm

    tokens, hc, dim, heads, head_dim, cols = 300, 4, 3072, 4, 96, 16
    eps = torch.finfo(torch.float32).eps
    hidden = torch.randn(tokens, hc, dim, device="cuda").bfloat16()
    # This rank owns buckets [vocab_start, vocab_end); others read as zero.
    vocab_start, vocab_end = 100, 900
    table = (torch.randn(vocab_end - vocab_start, head_dim, device="cuda") * 50).to(
        torch.float8_e4m3fn
    )
    table_scales = (
        torch.rand(vocab_end - vocab_start, device="cuda") * 0.02
    ).bfloat16()
    hashes = torch.randint(-1, 1000, (tokens, cols), device="cuda", dtype=torch.int32)
    hashes[-1] = -1
    local_hashes = hashes[:, 4:8]
    found = (local_hashes >= vocab_start) & (local_hashes < vocab_end)
    rows = (local_hashes - vocab_start).clamp(0, vocab_end - vocab_start - 1).long()
    gamma = (torch.rand(heads, head_dim, device="cuda") + 0.5).bfloat16()
    weight = (
        torch.randn((hc + 1) * dim, heads * head_dim, device="cuda") * 0.05
    ).bfloat16()
    bias = (torch.randn((hc + 1) * dim, device="cuda") * 0.1).bfloat16()
    key_norm, query_norm = (
        (torch.rand(hc, dim, device="cuda") + 0.5).bfloat16() for _ in range(2)
    )
    pre_gamma = (torch.rand(dim, device="cuda") + 0.5).bfloat16()
    hc_scales = (torch.randn(hc, dim, device="cuda") * 0.3).bfloat16()

    x = (table[rows].float() * table_scales[rows].float()[..., None]).masked_fill(
        ~found[..., None], 0
    )
    x = (x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * gamma.float()).to(
        torch.bfloat16
    )
    x = x.flatten(1)
    proj = F.linear(x, weight, bias)
    keys, value = proj[:, : hc * dim].unflatten(1, (hc, dim)), proj[:, hc * dim :]
    gate = torch.stack(
        [
            (
                F.rms_norm(keys[:, c], (dim,), key_norm[c], eps)
                * F.rms_norm(hidden[:, c], (dim,), query_norm[c], eps)
            ).sum(-1)
            for c in range(hc)
        ],
        dim=1,
    )
    gate = gate * dim**-0.5
    gate = (gate.abs().clamp_min(1e-6).sqrt() * gate.sign()).sigmoid()
    value = value.float()
    signal = pre_gamma.float() * value * (value.square().mean(-1, True) + eps).rsqrt()
    update = ((gate.float()[..., None] + hc_scales.float()) * signal[:, None]).to(
        torch.bfloat16
    )
    expected = hidden + update.masked_fill((hashes < 0).all(-1)[:, None, None], 0)

    fused_x = lookup_norm(
        table, table_scales, local_hashes, vocab_start, vocab_end, gamma, torch.bfloat16
    )
    torch.testing.assert_close(fused_x, x, rtol=0, atol=0)
    out = gated_update(hidden, proj, hashes, key_norm, query_norm, pre_gamma, hc_scales)
    torch.testing.assert_close(out[-1], hidden[-1], rtol=0, atol=0)
    torch.testing.assert_close(out, expected, rtol=0, atol=0.02)


def _ngram_hashes_reference(
    input_ids, lookback, starts, seq_lens, num_live, tables, pad_token_id, visual_ids
):
    """The unfused Flash hash path: the shared DSV4.1 hash kernel plus torch glue."""
    from vllm.models.deepseek_v41.common.engram import _hash_ids_kernel
    from vllm.triton_utils import triton

    token_map, multipliers, primes, offsets, pad_id = tables
    vocab = token_map.numel()
    layers, max_ngram = multipliers.shape
    heads = primes.shape[-1]
    tokens = input_ids.numel()
    rows = torch.arange(tokens, device="cuda")
    requests = torch.searchsorted(starts[1:], rows, right=True)
    requests = requests.clamp_max(starts.numel() - 2).long()
    logical = (
        seq_lens[requests] - (starts[requests + 1] - starts[requests]) + rows
    ) - starts[requests]
    dead = (input_ids < 0) | (input_ids >= vocab)
    ids = input_ids.clamp(0, vocab - 1)
    positions = logical.long()
    window = lookback.clamp(-1, vocab - 1)
    window_dead = lookback >= vocab
    hashes = torch.empty(
        tokens, layers, (max_ngram - 1) * heads, dtype=torch.int32, device="cuda"
    )
    _hash_ids_kernel[(triton.cdiv(tokens, 32), layers)](
        ids,
        token_map,
        dead,
        positions,
        starts,
        multipliers,
        primes,
        offsets,
        window,
        window_dead,
        hashes,
        tokens,
        pad_id,
        input_stride=ids.stride(0),
        mask_stride=dead.stride(0),
        position_stride=positions.stride(0),
        query_stride=starts.stride(0),
        num_query_rows=starts.numel() - 1,
        MAX_NGRAM=max_ngram,
        num_heads=heads,
        BLOCK_T=32,
        BLOCK_H=triton.next_power_of_2(heads),
        dead_id=-1,
        lookback_depth=window.shape[1],
        lookback_row_stride=window.stride(0),
        lookback_col_stride=window.stride(1),
        lookback_mask_row_stride=window_dead.stride(0),
        lookback_mask_col_stride=window_dead.stride(1),
        num_warps=4,
    )
    flat_primes = primes.flatten(1).unsqueeze(0)
    hashes = torch.where(
        hashes < offsets.unsqueeze(0), hashes + flat_primes, hashes
    ).int()
    no_hash = dead | (rows >= num_live) | (rows >= starts[-1])
    for token_id in (pad_token_id, *visual_ids):
        no_hash |= input_ids == token_id
    hashes.masked_fill_(no_hash[:, None, None], -1)
    return hashes.permute(1, 0, 2).contiguous()


@requires_gpu
def test_ngram_hashes():
    """The fused hash kernel matches the unfused path on a mixed batch."""
    from vllm.models.hy_v4_flash.common.engram_ops import ngram_hashes

    layers, max_ngram, heads, vocab = 2, 3, 4, 500
    primes = torch.randint(1000, 5000, (layers, max_ngram - 1, heads), device="cuda")
    flat = primes.flatten(1)
    offsets = torch.cumsum(flat, 1) - flat
    token_map = torch.randint(0, 300, (vocab,), device="cuda", dtype=torch.int32)
    multipliers = torch.randint(0, 2**40, (layers, max_ngram), device="cuda") * 2 + 1
    pad_id = 7
    # Fresh prefill, decode, chunked prefill, one-token context, then a
    # padded empty request and two padding rows.
    starts = torch.tensor([0, 5, 6, 9, 10, 10], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([5, 21, 5, 2, 0], device="cuda", dtype=torch.int32)
    num_live, tokens = 10, 12
    pad_token_id, visual_ids = 3, (11, 12)
    input_ids = torch.randint(-2, vocab + 3, (tokens,), device="cuda")
    input_ids[[1, 4, 7]] = torch.tensor([pad_token_id, 11, 12], device="cuda")
    lookback = torch.randint(
        -3, vocab + 3, (5, max_ngram - 1), device="cuda", dtype=torch.int32
    )

    expected = _ngram_hashes_reference(
        input_ids,
        lookback,
        starts,
        seq_lens,
        num_live,
        (token_map, multipliers, primes, offsets, pad_id),
        pad_token_id,
        visual_ids,
    )
    out = torch.empty_like(expected)
    ngram_hashes(
        input_ids,
        lookback,
        starts,
        seq_lens,
        num_live,
        token_map,
        multipliers,
        flat,
        offsets,
        pad_id,
        (pad_token_id, *visual_ids),
        out,
    )
    assert (expected >= 0).any() and (expected == -1).any()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


_ENGRAM_VOCAB = 64
_ENGRAM_LAYERS = (1, 3)
_ENGRAM_PRIMES = {
    "1": [[31, 37], [41, 43]],
    "3": [[47, 53], [59, 61]],
}


def _engram_config(**overrides):
    from vllm.models.hy_v4_flash.configs import HYV4FlashConfig

    config = dict(
        engram_layer_ids=list(_ENGRAM_LAYERS),
        engram_max_ngram_size=3,
        engram_n_head_per_ngram=2,
        engram_n_embed_per_ngram=16,
        engram_pad_id=5,
        engram_seed=0,
        engram_layer_head_vocab_sizes=_ENGRAM_PRIMES,
        engram_fp8_embedding=True,
        vocab_size=_ENGRAM_VOCAB,
        hidden_size=96,
        hc_mult=4,
        enable_ihc=True,
        num_hidden_layers=max(_ENGRAM_LAYERS),
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config.update(overrides)
    return HYV4FlashConfig(**config)


def _hash_state():
    from vllm.models.hy_v4_flash.common.engram import FlashNgramHashState

    with torch.device("cuda"):
        state = FlashNgramHashState(_engram_config())
    # A short map: ids past it hash as the pad id.
    state.load_lookup(torch.randint(0, 40, (_ENGRAM_VOCAB - 4,), dtype=torch.int32))
    state.visual_token_ids = (11, 12)
    return state


@requires_gpu
def test_engram_prepare_hashes():
    """`prepare_hashes` feeds the checkpoint token map, the pad and visual
    skip ids and the batch metadata to the hash kernel: chunked prefill into
    lookback, out-of-vocab lookback, an empty request and padding rows."""
    state = _hash_state()
    query_lens = [7, 6, 1, 1, 1, 0]
    seq_lens = [7, 20, 50, 2, 9, 0]
    lookback = [
        [-1, -1],
        [17, 61],
        [62, 3],
        [9, -1],
        [_ENGRAM_VOCAB + 1, 4],
        [-1, -1],
    ]
    starts = [0]
    for query_len in query_lens:
        starts.append(starts[-1] + query_len)
    num_live, rows = starts[-1] - 1, starts[-1] + 3
    input_ids = torch.randint(0, _ENGRAM_VOCAB, (rows,), device="cuda")
    input_ids[1] = state.layout.pad_token_id
    input_ids[3], input_ids[4] = 11, 12
    input_ids[8], input_ids[9] = -1, _ENGRAM_VOCAB + 2
    input_ids[10] = _ENGRAM_VOCAB - 1
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor(starts, dtype=torch.int32, device="cuda"),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        num_actual_tokens=num_live,
    )
    lookback = torch.tensor(lookback, dtype=torch.int32, device="cuda")
    out = torch.zeros(len(_ENGRAM_LAYERS), rows, 4, dtype=torch.int32, device="cuda")

    state.prepare_hashes(input_ids, lookback, metadata, out)

    layers, max_ngram = state.multipliers.shape
    expected = _ngram_hashes_reference(
        input_ids,
        lookback,
        metadata.query_start_loc,
        metadata.seq_lens,
        num_live,
        (
            state.token_map,
            state.multipliers,
            state.primes.reshape(layers, max_ngram - 1, -1),
            state.offsets,
            state.pad_id,
        ),
        state.layout.pad_token_id,
        state.visual_token_ids,
    )
    assert (expected >= 0).any() and (expected[:, :num_live] == -1).any()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@requires_gpu
def test_engram_hashes_without_metadata_are_dead():
    state = _hash_state()
    out = torch.zeros(len(_ENGRAM_LAYERS), 3, 4, dtype=torch.int32, device="cuda")
    ids = torch.ones(3, dtype=torch.long, device="cuda")
    state.prepare_hashes(ids, None, None, out)
    assert bool((out == -1).all())


@pytest.mark.parametrize("sizes", [[5] * 8, [7, 1, 1, 9, 3, 3, 2, 8], [4, 4]])
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_engram_head_range_partitions_heads(sizes, tp):
    """TP ranks own disjoint, non-empty, contiguous hash-head ranges that
    cover every head."""
    from vllm.models.hy_v4_flash.common.engram import head_range

    if tp > len(sizes):
        with pytest.raises(ValueError):
            head_range(sizes, tp, 0)
        return
    ranges = [head_range(sizes, tp, rank) for rank in range(tp)]
    assert ranges[0][0] == 0 and ranges[-1][1] == len(sizes)
    assert all(lo < hi for lo, hi in ranges)
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))


def test_engram_lookup_hashes_ids_past_the_map_as_pad():
    """Ids past the frozen map read as the pad id, as in the reference."""
    from vllm.models.hy_v4_flash.common.engram import FlashEngramLayout

    layout = FlashEngramLayout(_engram_config())
    lookup = torch.arange(_ENGRAM_VOCAB - 4)
    assert layout.load_lookup(lookup)

    pad = layout.token_map[layout.pad_token_id]
    assert layout.token_map == lookup.tolist() + [pad] * 4
    bound = torch.iinfo(torch.int64).max
    assert max(layout.token_map) * int(layout.multipliers.max()) <= bound


def _engram_checkpoint(config, sizes, head_dim, fp8):
    hidden, hc = config.hidden_size, config.hc_mult
    width = len(sizes) * head_dim
    rows = sum(sizes)
    table = torch.randn(rows, head_dim, device="cuda")
    weights = {}
    if fp8:
        weights["multi_head_embedding.embedding.weight"] = (table * 50).to(
            torch.float8_e4m3fn
        )
        weights["multi_head_embedding.embedding.weight_scale_inv"] = (
            torch.rand(rows, device="cuda") * 0.02 + 0.001
        ).bfloat16()
    else:
        weights["multi_head_embedding.embedding.weight"] = table.bfloat16()

    def bf16(*shape, scale=1.0, shift=0.0):
        return (torch.randn(*shape, device="cuda") * scale + shift).bfloat16()

    for c in range(hc):
        weights[f"key_projs.{c}.weight"] = bf16(hidden, width, scale=0.05)
        weights[f"key_projs.{c}.bias"] = bf16(hidden, scale=0.1)
        weights[f"norm_key.{c}.weight"] = bf16(hidden, scale=0.1, shift=1.0)
        weights[f"norm_query.{c}.weight"] = bf16(hidden, scale=0.1, shift=1.0)
        weights[f"unified_per_hc_scales.{c}"] = bf16(hidden, scale=0.3)
    weights["value_proj.weight"] = bf16(hidden, width, scale=0.05)
    weights["value_proj.bias"] = bf16(hidden, scale=0.1)
    weights["unified_pre_gamma"] = bf16(hidden, scale=0.1, shift=1.0)
    for head in range(len(sizes)):
        weights[f"lookup_per_head_norms.{head}.weight"] = bf16(
            head_dim, scale=0.1, shift=1.0
        )
    return weights


def _engram_reference(weights, hidden, hashes, hc, head_dim):
    """Eager FP32 Engram: per-head table lookup and RMSNorm, key/value
    projections, sqrt-signed sigmoid gate per stream, normalized value."""
    eps = torch.finfo(torch.float32).eps
    table = weights["multi_head_embedding.embedding.weight"].float()
    scales = weights.get("multi_head_embedding.embedding.weight_scale_inv")
    if scales is not None:
        table = table * scales.float()[:, None]
    found = (hashes >= 0) & (hashes < table.shape[0])
    x = table[hashes.clamp(0, table.shape[0] - 1).long()]
    x = x.masked_fill(~found[..., None], 0)
    gamma = torch.stack(
        [
            weights[f"lookup_per_head_norms.{h}.weight"].float()
            for h in range(hashes.shape[1])
        ]
    )
    x = (x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * gamma).bfloat16()
    x = x.flatten(1).float()

    def rms(t, w):
        return t * (t.square().mean(-1, keepdim=True) + eps).rsqrt() * w.float()

    value = F.linear(
        x, weights["value_proj.weight"].float(), weights["value_proj.bias"].float()
    )
    signal = rms(value, weights["unified_pre_gamma"])
    out = []
    for c in range(hc):
        key = F.linear(
            x,
            weights[f"key_projs.{c}.weight"].float(),
            weights[f"key_projs.{c}.bias"].float(),
        )
        query = hidden[:, c].float()
        dot = (
            rms(key, weights[f"norm_key.{c}.weight"])
            * rms(query, weights[f"norm_query.{c}.weight"])
        ).sum(-1) * key.shape[-1] ** -0.5
        gate = (dot.abs().clamp_min(1e-6).sqrt() * dot.sign()).sigmoid()
        scale = weights[f"unified_per_hc_scales.{c}"].float()
        out.append(query + (gate[:, None] + scale) * signal)
    out = torch.stack(out, 1)
    live = (hashes >= 0).any(-1)
    return torch.where(live[:, None, None], out, hidden.float())


@requires_gpu
@pytest.mark.parametrize("num_tokens", [0, 1, 37])
@pytest.mark.parametrize(
    "fp8,cpu_offload",
    [(True, False), (False, False), (True, True)],
    ids=["fp8", "bf16", "fp8-uva"],
)
def test_engram_layer_matches_reference(
    num_tokens, fp8, cpu_offload, dist_init, tmp_path
):
    """A checkpoint-loaded Engram layer matches the eager math, passes
    rows without live buckets through bit-exactly, and zeroes dead heads."""
    from vllm.config import (
        EngramConfig,
        ModelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.models.hy_v4_flash.common.engram import FlashEngram, FlashEngramLayout
    from vllm.utils.platform_utils import is_uva_available

    if cpu_offload and not is_uva_available():
        pytest.skip("Engram CPU offload requires UVA")
    checkpoint_config = _engram_config(engram_fp8_embedding=fp8)
    checkpoint_config.architectures = ["HYV4FlashForCausalLM"]
    checkpoint_config.save_pretrained(tmp_path)
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path), dtype="bfloat16", skip_tokenizer_init=True
        ),
        engram_config=EngramConfig(cpu_offload=cpu_offload),
    )
    config = vllm_config.model_config.hf_text_config
    layout = FlashEngramLayout(config)
    layer_id = _ENGRAM_LAYERS[1]
    sizes = [p for order in _ENGRAM_PRIMES[str(layer_id)] for p in order]
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        layer = FlashEngram(vllm_config, layout, layer_id, "engram")
    weights = _engram_checkpoint(config, sizes, layout.head_dim, fp8)
    for name, value in weights.items():
        source = value.cpu() if name.startswith("multi_head_embedding.") else value
        assert layer.load_weight(name, source) is not None

    hidden = torch.randn(num_tokens, config.hc_mult, config.hidden_size, device="cuda")
    hidden = hidden.bfloat16()
    offset = layout.offsets[_ENGRAM_LAYERS.index(layer_id)].tolist()
    hashes = torch.stack(
        [
            torch.randint(lo, lo + size, (num_tokens,), device="cuda")
            for lo, size in zip(offset, sizes)
        ],
        dim=1,
    ).int()
    if num_tokens > 2:
        # Edge buckets, a token with one dead head and a token with none live.
        hashes[0, 0], hashes[0, -1] = 0, sum(sizes) - 1
        hashes[1, 2] = -1
        hashes[2] = -1

    out = layer(hidden, hashes)

    expected = _engram_reference(
        weights, hidden, hashes, config.hc_mult, layout.head_dim
    )
    assert out.shape == hidden.shape and out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=3e-2)
    dead = (hashes < 0).all(-1)
    torch.testing.assert_close(out[dead], hidden[dead], rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_engram_disk_checkpoint_matches_pinned(fp8, dist_init, tmp_path):
    """Keep packed data on disk and preserve duplicate and masked lookups."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    from vllm.config import EngramConfig, ModelConfig, VllmConfig
    from vllm.models.hy_v4_flash.common.engram import FlashEngram, FlashEngramLayout

    checkpoint = _engram_config(engram_fp8_embedding=fp8)
    checkpoint.architectures = ["HYV4FlashForCausalLM"]
    checkpoint.save_pretrained(tmp_path)
    model = ModelConfig(
        model=str(tmp_path),
        dtype="bfloat16",
        enforce_eager=True,
        skip_tokenizer_init=True,
    )
    config = model.hf_text_config
    layout = FlashEngramLayout(config)
    layer_id = _ENGRAM_LAYERS[1]
    sizes = [p for order in layout.primes[1] for p in order]
    weights = _engram_checkpoint(config, sizes, layout.head_dim, fp8)
    shard = tmp_path / "model.safetensors"
    save_file(
        {name: value.cpu().contiguous() for name, value in weights.items()}, shard
    )

    layers = []
    for disk in (False, True):
        runtime = VllmConfig(
            model_config=model, engram_config=EngramConfig(disk_offload=disk)
        )
        with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
            layer = FlashEngram(runtime, layout, layer_id, "engram")
        with safe_open(shard, framework="pt") as source:
            # Real checkpoints may supply the scale first.
            for name in reversed(weights):
                layer.load_weight(name, source.get_tensor(name))
        layers.append(layer)

    offsets = layout.offsets[1].tolist()
    edge = [lo + size - 1 for lo, size in zip(offsets, sizes)]
    hashes = torch.tensor([offsets, edge, offsets, [-1] * len(sizes)], device="cuda")
    hashes[1, 1] = -1
    hidden = torch.randn(
        (4, config.hc_mult, config.hidden_size), device="cuda"
    ).bfloat16()
    pinned, disk = layers
    torch.testing.assert_close(
        disk(hidden, hashes), pinned(hidden, hashes), rtol=0, atol=0
    )
    torch.testing.assert_close(
        disk(hidden[-1:], hashes[-1:]), hidden[-1:], rtol=0, atol=0
    )
    assert disk.multi_head_embedding.embedding.weight.numel() == 0


def _hadamard(n, device):
    """Dense Sylvester Hadamard matrix, `H[i][j] = (-1)^popcount(i & j)`."""
    index = torch.arange(n, device=device)
    bits = index[:, None] & index[None]
    parity = torch.zeros_like(bits)
    while bool(bits.any()):
        parity ^= bits & 1
        bits >>= 1
    return 1 - 2 * parity.double()


@requires_rocm
@pytest.mark.parametrize("rows", [0, 1, 37, 1024])
@pytest.mark.parametrize("strided", [False, True], ids=["bf16", "strided_fp32"])
def test_amd_fwht_quantize_matches_dense_hadamard(rows, strided):
    """FP8 codes and power-of-two scales round the orthonormal Hadamard
    transform to within half an e4m3 ulp, bitwise equal to the NVIDIA reference.
    """
    from vllm.models.hy_v4_flash.amd.indexpool_ops import fwht_quantize

    magnitude = 10 ** (torch.rand(rows, 1, device="cuda") * 6 - 3)
    data = torch.randn(rows, 256, device="cuda") * magnitude
    if rows > 1:
        data[0] = 0
        data[1, 5] = 3e4
    vectors = data[:, 64:192] if strided else data[:, :128].bfloat16()

    quant, scale = fwht_quantize(vectors)

    exact = vectors.double() @ _hadamard(128, "cuda") * 128**-0.5
    amax = exact.abs().amax(-1).clamp_min(1e-4)
    assert torch.equal(scale.double(), torch.exp2(torch.ceil(torch.log2(amax / 448))))
    scale = scale.double()[:, None]
    error = (quant.double() * scale - exact).abs()
    assert bool((error <= (2**-4 + 1e-5) * exact.abs() + 2**-10 * scale).all())
    expected_quant, expected_scale = _fwht_quantize_reference(vectors)
    assert torch.equal(quant.view(torch.uint8), expected_quant.view(torch.uint8))
    assert torch.equal(scale.float().flatten(), expected_scale)


def _rope_planes(kind):
    """MRoPE axis feeding each of the 32 rotary frequency lanes."""
    if kind == "interleaved":
        return [
            1 if i % 3 == 1 and i < 30 else 2 if i % 3 == 2 else 0 for i in range(32)
        ]
    return [0] * 12 + [1] * 10 + [2] * 10


@requires_rocm
@pytest.mark.parametrize("heads", [None, 16, 64], ids=["keys", "q16", "q64"])
@pytest.mark.parametrize("kind", ["1d", "interleaved", "sectioned"])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
def test_amd_rotate_indexer_matches_complex_rotation(heads, kind, strided):
    """Each adjacent pair `(2i, 2i + 1)` of the 64 rotary channels is rotated
    as a complex number and written to lanes `i` and `i + 32`; the 64 NoPE
    channels pass through untouched."""
    from vllm.models.hy_v4_flash.amd.indexpool_ops import (
        rope_inverse_frequencies,
        rotate_indexer,
    )

    tokens = 37
    parameters = {**_ROPE, "mrope_interleaved": kind != "sectioned"}
    shape = (tokens, 256) if heads is None else (tokens, heads, 256)
    data = torch.randn(shape, device="cuda").bfloat16()
    x = data[..., 64:192] if strided else data[..., :128].contiguous()
    if kind == "1d":
        positions = torch.randint(0, 2**20, (tokens,), device="cuda")
    else:
        # The indexer slices padded positions, so the planes are strided.
        positions = torch.randint(0, 2**20, (3, tokens + 5), device="cuda")
        positions = positions[:, :tokens]
    inverse = rope_inverse_frequencies(64, parameters, "cuda")

    out = rotate_indexer(x, positions, 64, parameters, inverse)

    if kind == "1d":
        angle = positions.float()[:, None] * inverse
    else:
        planes = torch.tensor(_rope_planes(kind), device="cuda")
        angle = positions.float()[planes].T * inverse
    if heads is not None:
        angle = angle[:, None]
    pairs = torch.view_as_complex(x[..., 64:].double().unflatten(-1, (32, 2)))
    rotated = pairs * torch.polar(torch.ones_like(angle.double()), angle.double())
    assert torch.equal(out[..., :64], x[..., :64])
    torch.testing.assert_close(
        out[..., 64:].double(),
        torch.cat((rotated.real, rotated.imag), -1),
        rtol=2**-8,
        atol=1e-5,
    )
    assert torch.equal(rotate_indexer(x, positions, 64, parameters), out)


@requires_rocm
@pytest.mark.parametrize("slots", [0, 1, 37, 1025])
def test_amd_gated_pool_matches_float64_reference(slots):
    """Each group is the softmax(gate + APE)-weighted sum of its keys, then
    RMS-normalized; values and gates are the two strided halves of one GEMM."""
    from vllm.models.hy_v4_flash.amd.indexpool_ops import gated_pool

    ratio, eps = 4, 1e-5
    projection = torch.randn(slots, ratio, 256, device="cuda") * 3
    values, gates = projection[..., :128], projection[..., 128:]
    ape = torch.randn(ratio, 128, device="cuda")
    norm_weight = torch.rand(128, device="cuda") + 0.5

    out = gated_pool(values, gates, ape, norm_weight, eps)

    weights = (gates.double() + ape.double()).softmax(1)
    pooled = (values.double() * weights).sum(1)
    rms = (pooled.square().mean(-1, keepdim=True) + eps).rsqrt()
    expected = pooled * rms * norm_weight.double()
    assert out.dtype == torch.float32 and out.shape == (slots, 128)
    torch.testing.assert_close(out.double(), expected, rtol=1e-5, atol=1e-6)


@requires_rocm
@pytest.mark.parametrize("width,budget", [(200, 511), (4096, 511), (4096, 7)])
def test_amd_select_groups_matches_per_row_topk(width, budget):
    """Rows keep the highest-scoring groups below their bound in ascending
    order, ignore scores at or past the bound, and pad with -1 to the budget."""
    from vllm.models.hy_v4_flash.amd.indexpool_ops import select_groups

    rows = 40
    upper = torch.randint(0, width + 50, (rows,), device="cuda").int()
    upper[:6] = torch.tensor([0, 1, budget - 1, budget, budget + 1, width + 50])
    logits = torch.randn(rows, width, device="cuda")
    columns = torch.arange(width, device="cuda")
    # The paged logits kernel leaves arbitrary scores past each row's bound.
    logits.masked_fill_(columns[None] >= upper[:, None], 1e30)

    out = select_groups(logits, upper, budget)

    expected = torch.full((rows, budget), -1, dtype=torch.int32)
    for row in range(rows):
        bound = min(int(upper[row]), width)
        order = logits[row, :bound].cpu().double().argsort(descending=True)
        chosen = order[:budget].sort().values
        expected[row, : chosen.numel()] = chosen.int()
    assert out.dtype == torch.int32
    assert torch.equal(out.cpu(), expected)
    assert select_groups(logits[:0], upper[:0], budget).shape == (0, budget)


@requires_rocm
@pytest.mark.parametrize("with_active", [False, True])
def test_amd_expand_groups_matches_per_row_reference(with_active):
    """Selected groups expand to their member tokens, then the current partial
    group's tokens up to the query position; live rows are front-packed and
    padding, inactive and stale (>= bound) groups contribute nothing."""
    from vllm.models.hy_v4_flash.amd.indexpool_ops import expand_groups

    rows, budget, ratio = 48, 7, 4
    positions = torch.randint(0, 80, (rows,))
    positions[:4] = torch.tensor([-1, 0, 3, 4])
    active = torch.rand(rows) > 0.25 if with_active else None
    group_ids = torch.full((rows, budget), -1, dtype=torch.int32)
    for row in range(rows):
        # Includes groups at and past the row's bound, which must be dropped.
        pool = torch.randperm(int(positions[row]) // ratio + 3)
        picked = pool[: int(torch.randint(0, budget + 1, ()))].sort().values
        group_ids[row, : picked.numel()] = picked.int()

    out = expand_groups(
        group_ids.cuda(),
        positions.cuda(),
        ratio,
        active=None if active is None else active.cuda(),
    )

    expected = torch.full((rows, budget * ratio + ratio), -1, dtype=torch.int32)
    for row in range(rows):
        position = int(positions[row])
        if position < 0 or (active is not None and not active[row]):
            continue
        bound = position // ratio
        members = [
            group * ratio + k
            for group in group_ids[row].tolist()
            if 0 <= group < bound
            for k in range(ratio)
        ]
        members += [bound * ratio + k for k in range(position % ratio + 1)]
        expected[row, : len(members)] = torch.tensor(members)
    assert torch.equal(out.cpu(), expected)


@requires_rocm
@pytest.mark.parametrize("slots", [0, 1, 29])
def test_amd_write_pool_matches_page_layout(slots):
    """Active rows land as 128 FP8 bytes at `row * 128` and a 4-byte FP32 scale
    after the page's code rows; inactive rows and other bytes stay untouched."""
    from vllm.models.hy_v4_flash.amd.indexpool_ops import write_pool

    page, blocks, owners_count, columns = 64, 64, 4, 16
    cache = torch.randint(0, 256, (blocks, page, 132), dtype=torch.uint8).cuda()
    table = torch.randperm(blocks).int().reshape(owners_count, columns).cuda()
    owners = torch.randint(0, owners_count, (slots,)).int().cuda()
    groups = torch.randperm(page * columns)[:slots].cuda()
    active = (torch.rand(slots) > 0.2).cuda()
    # Inactive rows may name groups past the table; they must not be written.
    groups = torch.where(active, groups, groups + page * columns)
    codes = torch.randint(0, 256, (slots, 256), dtype=torch.uint8).cuda()
    quant = codes[:, 64:192].view(torch.float8_e4m3fn)
    scales = torch.rand(slots, device="cuda") + 0.1

    expected = cache.clone()
    write_pool(cache, table, owners, groups, active, quant, scales)

    for slot in active.nonzero().flatten().tolist():
        group = int(groups[slot])
        flat = expected[int(table[owners[slot], group // page])].view(-1)
        row = group % page
        flat[row * 128 : (row + 1) * 128] = codes[slot, 64:192]
        flat[page * 128 + row * 4 : page * 128 + (row + 1) * 4] = scales[
            slot : slot + 1
        ].view(torch.uint8)
    assert torch.equal(cache, expected)


# (query_len, accepted) for requests A, B and C, then the query length of a
# padded request D whose seq_len is 0, then trailing padding tokens. Rejected
# tokens carry garbage keys and must not survive into later groups.
_PREPARE_STEPS = [
    ([(0, 0), (0, 0), (0, 0)], 0, 0),
    ([(7, 7), (1, 1), (13, 13)], 0, 0),
    ([(4, 2), (6, 6), (0, 0)], 2, 3),
    ([(4, 4), (1, 1), (4, 1)], 0, 0),
    ([(1, 1), (4, 3), (3, 3)], 0, 2),
    ([(4, 4), (1, 1), (1, 1)], 0, 0),
]


def _random_positions(three_axis, tokens):
    shape = (3, tokens) if three_axis else (tokens,)
    return torch.randint(0, 2**20, shape, device="cuda")


@requires_rocm
@pytest.mark.parametrize("three_axis", [False, True], ids=["1d", "mrope"])
def test_amd_prepare_groups_follows_accepted_sequence(three_axis):
    """Across prefill chunks, idle steps and speculative steps with rejected
    drafts, each step emits exactly the groups its tokens complete, built from
    accepted keys carried in the tail ring plus this step's keys, and the
    per-token metadata names the owning request and logical position."""
    from vllm.models.hy_v4_flash.amd.indexpool_prepare import (
        prepare_groups,
        write_tails,
    )
    from vllm.models.hy_v4_flash.common.indexpool_cache import tail_ring_size

    ratio, total = 4, 64
    ring = tail_ring_size(ratio, 3)
    true_keys = torch.randn(3, total, 128, device="cuda").bfloat16()
    true_positions = [_random_positions(three_axis, total) for _ in range(3)]
    tail = torch.randn(12, ring, 131, device="cuda")
    table = torch.randperm(12).int().reshape(4, 3).cuda()
    contexts = [0, 0, 0]

    for requests, padded, trailing in _PREPARE_STEPS:
        step_keys, step_positions, query_lens = [], [], []
        for request, (query, accepted) in enumerate(requests):
            begin = contexts[request]
            keys = torch.randn(query, 128, device="cuda").bfloat16()
            keys[:accepted] = true_keys[request, begin : begin + accepted]
            positions = _random_positions(three_axis, query)
            positions[..., :accepted] = true_positions[request][
                ..., begin : begin + accepted
            ]
            step_keys.append(keys)
            step_positions.append(positions)
            query_lens.append(query)
        query_lens.append(padded)
        tokens = sum(query_lens) + trailing
        rows = tokens + 5
        storage = torch.randn(rows, 256, device="cuda").bfloat16()
        keys = storage[:, :128]
        positions = _random_positions(three_axis, rows)
        start = 0
        for request, query in enumerate(query_lens[:3]):
            keys[start : start + query] = step_keys[request]
            positions[..., start : start + query] = step_positions[request]
            start += query
        starts = torch.tensor([0, *query_lens], device="cuda").cumsum(0).int()
        lengths = [context + query for context, query in zip(contexts, query_lens)]
        lengths = torch.tensor([*lengths, 0], device="cuda").int()

        (
            raw,
            group_positions,
            owners,
            groups,
            group_live,
            owner,
            logical,
            upper,
            live,
        ) = prepare_groups(keys, positions, tail, table, starts, lengths, tokens, ratio)

        expected = []
        for request, query in enumerate(query_lens[:3]):
            context = contexts[request]
            history = torch.cat((true_keys[request, :context], step_keys[request]))
            places = torch.cat(
                (true_positions[request][..., :context], step_positions[request]), -1
            )
            for group in range(context // ratio, (context + query) // ratio):
                members = slice(group * ratio, (group + 1) * ratio)
                expected.append(
                    (request, group, history[members], places[..., group * ratio])
                )
        count = len(expected)
        assert raw.shape[0] == (tokens + 4 * (ratio - 1)) // ratio
        assert bool(group_live[:count].all()) and not bool(group_live[count:].any())
        assert owners[:count].tolist() == [entry[0] for entry in expected]
        assert groups[:count].tolist() == [entry[1] for entry in expected]
        if count:
            assert torch.equal(raw[:count], torch.stack([e[2] for e in expected]))
            assert torch.equal(
                group_positions[..., :count], torch.stack([e[3] for e in expected], -1)
            )
        # Padding slots must be finite zeros so the pooled GEMM stays finite.
        assert not bool(raw[count:].any())
        assert not bool(group_positions[..., count:].any())

        request_of = (
            torch.repeat_interleave(torch.arange(4), torch.tensor(query_lens)).tolist()
            + [3] * trailing
        )
        starts_list = starts.tolist()
        for token, request in enumerate(request_of):
            is_live = request < 3 and token < starts_list[-1]
            assert int(owner[token]) == request
            assert bool(live[token]) == is_live
            if is_live:
                position = contexts[request] + token - starts_list[request]
                assert int(logical[token]) == position
                assert int(upper[token]) == position // ratio
            else:
                assert int(upper[token]) == 0

        write_tails(keys, positions, tail, table, starts, lengths)
        for request, (_, accepted) in enumerate(requests):
            contexts[request] += accepted


def _prepare_batch(batch, tokens, padding):
    extra = torch.randint(0, batch, (tokens - batch,))
    query_lens = 1 + extra.bincount(minlength=batch)
    starts = torch.cat((torch.zeros(1, dtype=torch.long), query_lens.cumsum(0)))
    lengths = query_lens + torch.randint(0, 3000, (batch,))
    return (
        torch.randn(tokens + padding, 128, device="cuda").bfloat16(),
        torch.randint(0, 2**20, (3, tokens + padding), device="cuda"),
        starts.int().cuda(),
        lengths.int().cuda(),
    )


@requires_rocm
def test_amd_prepare_groups_reuses_jit_across_batch_shapes(monkeypatch):
    """Changing token, slot and padded-position counts within one batch-size
    bucket reuses the compiled IndexPool prepare kernels (no per-step JIT)."""
    from vllm.models.hy_v4_flash.amd import indexpool_prepare
    from vllm.triton_utils import triton

    ratio = 4
    tail = torch.randn(8, 8, 131, device="cuda")
    table = torch.arange(8, device="cuda").int().reshape(8, 1).expand(8, 4)

    def run(batch, tokens, padding):
        keys, positions, starts, lengths = _prepare_batch(batch, tokens, padding)
        indexpool_prepare.prepare_groups(
            keys, positions, tail, table, starts, lengths, tokens, ratio
        )
        indexpool_prepare.write_tails(keys, positions, tail, table, starts, lengths)

    # Batch sizes 5 and 6 share the power-of-two prefix block and search depth.
    run(5, 7, 1)
    torch.accelerator.synchronize()
    compiled = []
    monkeypatch.setattr(
        triton.knobs.runtime,
        "jit_post_compile_hook",
        lambda **kwargs: compiled.append(kwargs.get("fn")),
    )
    for batch, tokens, padding in [
        (6, 6, 0),
        (5, 17, 3),
        (6, 64, 16),
        (5, 301, 7),
        (6, 1000, 24),
    ]:
        run(batch, tokens, padding)
    torch.accelerator.synchronize()
    assert not compiled, compiled


def _capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


@requires_rocm
def test_amd_pool_write_replays_under_cuda_graph():
    """The ROCm IndexPool write path (prepare, pool, rotate, quantize, write,
    retain tails) captured for a decode batch replays new inputs bitwise
    equal to eager execution."""
    from vllm.models.hy_v4_flash.amd.indexpool_ops import (
        fwht_quantize,
        gated_pool,
        rope_inverse_frequencies,
        rotate_indexer,
        write_pool,
    )
    from vllm.models.hy_v4_flash.amd.indexpool_prepare import (
        prepare_groups,
        write_tails,
    )

    batch, ratio, padding = 6, 4, 2
    value_weight, gate_weight = torch.randn(2, 128, 128, device="cuda") * 0.1
    ape = torch.randn(ratio, 128, device="cuda")
    norm_weight = torch.rand(128, device="cuda") + 0.5
    inverse = rope_inverse_frequencies(64, _ROPE, "cuda")
    pool_table = torch.randperm(64).int().reshape(batch + 2, 8)[:batch].cuda()
    tail_table = torch.randperm(batch).int()[:, None].cuda()

    def inputs():
        lengths = torch.randint(1, 8 * 64 * ratio, (batch,)).int()
        # Several requests complete a group on this decode step.
        lengths[:3] = torch.tensor([4, 8, 4 * 37])
        return {
            "keys": torch.randn(batch + padding, 128, device="cuda").bfloat16(),
            "positions": torch.randint(0, 2**20, (3, batch + padding), device="cuda"),
            "lengths": lengths.cuda(),
            "cache": torch.randint(0, 256, (64, 64, 132), dtype=torch.uint8).cuda(),
            "tail": torch.randn(batch, ratio * 2, 131, device="cuda"),
        }

    def step(state):
        starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
        raw, coordinates, owners, groups, group_live, *_ = prepare_groups(
            state["keys"],
            state["positions"],
            state["tail"],
            tail_table,
            starts,
            state["lengths"],
            batch,
            ratio,
        )
        pooled = gated_pool(
            F.linear(raw.float(), value_weight),
            F.linear(raw.float(), gate_weight),
            ape,
            norm_weight,
            1e-5,
        ).bfloat16()
        quant, scales = fwht_quantize(
            rotate_indexer(pooled, coordinates, 64, _ROPE, inverse)
        )
        write_pool(
            state["cache"], pool_table, owners, groups, group_live, quant, scales
        )
        write_tails(
            state["keys"],
            state["positions"],
            state["tail"],
            tail_table,
            starts,
            state["lengths"],
        )

    static = inputs()
    graph, _ = _capture(lambda: step(static))
    for _ in range(2):
        fresh = inputs()
        for name, tensor in fresh.items():
            static[name].copy_(tensor)
        graph.replay()
        step(fresh)
        torch.accelerator.synchronize()
        assert torch.equal(static["cache"], fresh["cache"])
        assert torch.equal(static["tail"], fresh["tail"])


@requires_rocm
def test_amd_group_selection_replays_under_cuda_graph():
    """`select_groups` plus `expand_groups` captured with static shapes replay
    new logits, bounds and liveness bitwise equal to eager execution."""
    from vllm.models.hy_v4_flash.amd.indexpool_ops import (
        expand_groups,
        select_groups,
    )

    rows, width, budget, ratio = 32, 1024, 511, 4

    def inputs():
        logical = torch.randint(-1, width * ratio, (rows,), device="cuda")
        return {
            "logits": torch.randn(rows, width, device="cuda"),
            "upper": (logical // ratio).clamp_min(0).int(),
            "logical": logical,
            "live": torch.rand(rows, device="cuda") > 0.1,
        }

    def select(state):
        recalled = select_groups(state["logits"], state["upper"], budget)
        return expand_groups(recalled, state["logical"], ratio, state["live"])

    static = inputs()
    graph, out = _capture(lambda: select(static))
    for _ in range(2):
        fresh = inputs()
        for name, tensor in fresh.items():
            static[name].copy_(tensor)
        graph.replay()
        assert torch.equal(out, select(fresh))
