# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY V4 Flash fused Triton kernels match the eager math they replaced."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="HY V4 Flash kernels require CUDA"
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


def test_engram_fused_update():
    from vllm.models.hy_v4_flash.nvidia.engram_ops import gated_update, lookup_norm

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


def test_ngram_hashes():
    """The fused hash kernel matches the unfused path on a mixed batch."""
    from vllm.models.hy_v4_flash.nvidia.engram_ops import ngram_hashes

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
