# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 SM80 kernels against their PyTorch references.

* the FP4 quantizers (indexer queries, index keys, NVFP4 compressed records)
  and the BF16x2 FP8/FP4 decoders bitwise;
* the indexer logits, ragged and paged, up to the FP32 summation order, the
  paged ones split over CTAs;
* the decode top-k and candidate-block selection exactly, ties included;
* a candidate consumer's compact logits bitwise against the dense ones, and
  its top-k the same keys as over the masked dense row; on SM90's FP8 cache
  with unordered candidate blocks, the logits up to the FP32 summation order
  and the top-k the same set;
* sparse MLA within BF16 rounding of the probabilities, with a key list
  split over CTAs or walked by one (several chunks per CTA), empty rows,
  unselected (-1) entries, window-only layers, padded cache pages and head
  counts that do not fill a CTA's 8 or 16 heads; and its V4-record variant
  (SM90's caches) without the inverse RoPE.

The kernels use SM80 instructions only, so any SM80-or-newer GPU runs them;
about a minute, most of it compilation.
"""

import pytest
import torch

from vllm.platforms import current_platform

from . import reference

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(80),
    reason="needs an SM80-or-newer CUDA GPU",
)

DEVICE = "cuda"


def _paged(blocks: int, block_size: int, record: int, pad: int) -> torch.Tensor:
    """A zeroed uint8 cache `[blocks, block_size, record]` whose pages are
    `pad` bytes longer than their records."""
    raw = torch.zeros(
        blocks, block_size * record + pad, dtype=torch.uint8, device=DEVICE
    )
    return raw[:, : block_size * record].view(blocks, block_size, record)


@pytest.fixture
def cos_sin() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4096, 64, device=DEVICE)


def test_bf16x2_decoders():
    """The pair decoders bitwise against the FP32 reference decoders rounded
    to BF16, over every E4M3/E2M1 code and scales down to ones whose products
    are subnormal or zero."""
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64

    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import codecs

    @cute.kernel
    def kernel(words: Int64, codes: Int64, out: Int64):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        item = Int64(block * 128 + tid)
        word = codecs.load_u32(words + item * 4)
        code = codecs.load_u32(codes + item * 4) & 0xFF
        row = out + item * 24
        ue8m0 = codecs.ue8m0_bf16x2(code)
        e4m3 = codecs.e4m3_bf16x2(code)
        for i in cutlass.range_constexpr(2):
            codecs.store_u32(row + 4 * i, codecs.e4m3x2_bf16(word, i, ue8m0))
        for i in cutlass.range_constexpr(4):
            codecs.store_u32(row + 8 + 4 * i, codecs.e2m1x2_bf16(word, i, e4m3))

    @cute.jit
    def launch(words: Int64, codes: Int64, out: Int64, blocks: Int32):
        kernel(words, codes, out).launch(grid=(blocks, 1, 1), block=[128, 1, 1])

    torch.manual_seed(0)
    # Every byte value at every position of a word; NaN E4M3 codes excluded.
    values = torch.arange(256, device=DEVICE, dtype=torch.int32)
    values = values[(values & 0x7F) != 0x7F]
    words = values[torch.randint(0, values.numel(), (128 * 64, 4), device=DEVICE)]
    words[: values.numel(), 0] = values
    words = words.to(torch.uint8).view(torch.int32).flatten()
    # Codes 0..126: UE8M0 scales 2^-127..2^-1 and positive E4M3 scales (not
    # NaN); most rows use 2^-27..2^-1.
    codes = torch.randint(100, 127, (words.numel(),), device=DEVICE, dtype=torch.int32)
    codes[: values.numel() // 2] = values[: values.numel() // 2]
    out = torch.zeros(words.numel(), 6, dtype=torch.int32, device=DEVICE)
    launch(
        Int64(words.data_ptr()),
        Int64(codes.data_ptr()),
        Int64(out.data_ptr()),
        Int32(words.numel() // 128),
    )
    torch.accelerator.synchronize()
    data = words.view(torch.uint8).view(-1, 4)
    want_e4m3 = reference.e4m3_to_f32(data) * reference.ue8m0_to_f32(codes)[:, None]
    nibbles = torch.stack((data & 0xF, data >> 4), dim=-1).flatten(-2)
    want_e2m1 = (
        reference.e2m1_to_f32(nibbles)
        * reference.e4m3_to_f32(codes.to(torch.uint8))[:, None]
    )
    got = out.view(torch.int16).view(-1, 12)
    assert torch.equal(got[:, :4], want_e4m3.bfloat16().view(torch.int16))
    assert torch.equal(got[:, 4:], want_e2m1.bfloat16().view(torch.int16))


def test_indexer_query(cos_sin):
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import quantize

    tokens, heads = 37, 64
    magnitude = torch.logspace(-3, 2, heads, device=DEVICE)[None, :, None]
    q = (torch.randn(tokens, heads, 128, device=DEVICE) * magnitude).bfloat16()
    q[0, 0, :32] = 0
    positions = torch.randint(0, 4096, (tokens,), device=DEVICE)
    weights = torch.randn(tokens, heads, device=DEVICE).bfloat16()
    want = reference.indexer_q_rope_quant(positions, q, cos_sin, weights, 0.0123)
    got = quantize.indexer_q_rope_quant(positions, q, cos_sin, weights, 0.0123)
    for w, g in zip(want, got):
        torch.testing.assert_close(g, w, rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [1, 2])
def test_index_keys(cos_sin, ratio):
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import quantize

    block_size, blocks, tokens = 64, 8, 100
    want = _paged(blocks, block_size, 68, 32)
    got = _paged(blocks, block_size, 68, 32)
    k_pre = (torch.randn(tokens, 128, device=DEVICE) * 3).bfloat16()
    positions = torch.arange(tokens, device=DEVICE) + 7
    slots = torch.randperm(blocks * block_size, device=DEVICE)[:tokens]
    slots[5] = -1
    norm_weight = (torch.rand(128, device=DEVICE) + 0.5).bfloat16()
    values, scales = reference.indexer_k_norm_rope(
        k_pre, positions, cos_sin, norm_weight, 1e-6, ratio
    )
    reference.store_index_keys(values, scales, positions, slots, ratio, want)
    quantize.store_index_keys(
        k_pre, positions, slots, norm_weight, 1e-6, cos_sin, ratio, got
    )
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [1, 2])
def test_compressed_records(cos_sin, ratio):
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import quantize

    block_size, blocks, tokens = 32, 10, 200
    want = _paged(blocks, block_size, 288, 64)
    got = _paged(blocks, block_size, 288, 64)
    magnitude = torch.logspace(-4, 3, 512, device=DEVICE)
    latent = (torch.randn(tokens, 512, device=DEVICE) * magnitude).bfloat16()
    latent[3] = 0
    positions = torch.arange(tokens, device=DEVICE) * 3 + 1
    slots = torch.randperm(blocks * block_size, device=DEVICE)[:tokens]
    slots[7] = -1
    keep = (slots >= 0) & ((positions + 1) % ratio == 0)
    records = reference.nvfp4_kv_record(latent[keep], positions[keep], cos_sin, ratio)
    kept = slots[keep]
    want[kept // block_size, kept % block_size] = records
    quantize.store_compressed_records(latent, positions, slots, cos_sin, ratio, got)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def _assert_rows_close(got, want, lens, rtol):
    for row in range(want.shape[0]):
        n = int(lens[row])
        if n:
            scale = want[row, :n].abs().max().clamp_min(1e-6)
            assert ((got[row, :n] - want[row, :n]).abs().max() / scale) < rtol


@pytest.mark.parametrize("heads", [32, 64])
def test_ragged_logits(heads):
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import mqa_logits

    torch.manual_seed(0)
    rows, keys = 37, 900
    q_values, q_scales = reference.mxfp4_quantize(
        torch.randn(rows, heads, 128, device=DEVICE)
    )
    k_values, k_scales = reference.mxfp4_quantize(torch.randn(keys, 128, device=DEVICE))
    weights = torch.randn(rows, heads, device=DEVICE)
    starts = torch.randint(0, 300, (rows,), device=DEVICE, dtype=torch.int32)
    lengths = torch.randint(0, 600, (rows,), device=DEVICE, dtype=torch.int32)
    ends = (starts + lengths).clamp(max=keys)
    full = reference.mqa_logits(q_values, q_scales, k_values, k_scales, weights)
    max_len = int((ends - starts).max())
    got = mqa_logits.mqa_logits(
        q_values, q_scales, k_values, k_scales, weights, starts, ends, max_len
    )
    want = torch.zeros(rows, max_len, device=DEVICE)
    for row in range(rows):
        want[row, : ends[row] - starts[row]] = full[row, starts[row] : ends[row]]
    _assert_rows_close(got, want, ends - starts, 1e-4)


def _index_key_cache(blocks: int, block_size: int) -> torch.Tensor:
    """A paged index key cache of random MXFP4 keys."""
    cache = _paged(blocks, block_size, 68, 32)
    values, scales = reference.mxfp4_quantize(
        torch.randn(blocks * block_size, 128, device=DEVICE)
    )
    positions = torch.arange(blocks * block_size, device=DEVICE)
    reference.store_index_keys(values, scales, positions, positions, 1, cache)
    return cache


@pytest.mark.parametrize(
    "heads,block_size,splits,group",
    [
        (32, 64, None, 1),
        (64, 64, None, 1),
        (32, 64, 1, 1),
        (32, 32, 1, 1),
        (32, 64, None, 3),
        (64, 64, None, 3),
        (32, 32, 1, 3),
    ],
)
def test_paged_logits(heads, block_size, splits, group):
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import mqa_logits

    torch.manual_seed(0)
    blocks, next_n, requests = 1200 * 64 // block_size, 3, 4
    cache = _index_key_cache(blocks, block_size)
    table = torch.randperm(blocks, device=DEVICE, dtype=torch.int32).view(requests, -1)
    rows = requests * next_n
    max_len = table.shape[1] * block_size
    lens = torch.randint(0, max_len + 1, (rows,), device=DEVICE, dtype=torch.int32)
    lens[0] = max_len
    q_values, q_scales = reference.mxfp4_quantize(
        torch.randn(rows, heads, 128, device=DEVICE)
    )
    weights = torch.randn(rows, heads, device=DEVICE)
    args = (q_values, q_scales, cache, weights, lens, table, next_n, max_len)
    got = mqa_logits.paged_mqa_logits(*args, splits, group)
    if group > 1:
        # The rows of a group share the decoded keys, not the arithmetic.
        alone = mqa_logits.paged_mqa_logits(*args, splits)
        for row in range(rows):
            n = int(lens[row])
            assert torch.equal(got[row, :n], alone[row, :n])
    want = torch.zeros(rows, max_len, device=DEVICE)
    for row in range(rows):
        n = int(lens[row])
        if n:
            k_values, k_scales = reference.gather_index_keys(
                cache, table[row // next_n], n
            )
            want[row, :n] = reference.mqa_logits(
                q_values[row : row + 1],
                q_scales[row : row + 1],
                k_values,
                k_scales,
                weights[row : row + 1],
            )[0]
    _assert_rows_close(got, want, lens, 1e-4)


@pytest.mark.parametrize("block_size,splits", [(64, None), (32, 1), (64, 3)])
def test_candidate_logits(block_size, splits):
    """A candidate consumer: compact logits bitwise against the dense ones, and
    its top-k the keys the dense row selects with the other keys masked."""
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import (
        mqa_logits,
        topk,
    )

    torch.manual_seed(0)
    heads, count, block, k = 32, 2048, 8, 512
    # Rows of 19200 keys hold more than 2048 blocks of 8.
    blocks = 1200 * 64 // block_size
    cache = _index_key_cache(blocks, block_size)
    table = torch.randperm(blocks, device=DEVICE, dtype=torch.int32).view(4, -1)
    max_len = table.shape[1] * block_size
    # Empty, below k, k, one past it, every block listed, a partial newest
    # block, and lists of 2048 of more blocks.
    lens = [0, 300, 512, 513, 9000, count * block + 5, max_len - 3, max_len]
    rows = len(lens)
    lens = torch.tensor(lens, dtype=torch.int32, device=DEVICE)
    table = table[torch.arange(rows, device=DEVICE) % 4]
    q_values, q_scales = reference.mxfp4_quantize(
        torch.randn(rows, heads, 128, device=DEVICE)
    )
    weights = torch.randn(rows, heads, device=DEVICE)
    # As the candidate source lists them: ascending, the newest block last.
    candidates = torch.full((rows, count), -1, dtype=torch.int32, device=DEVICE)
    for row in range(rows):
        n = (int(lens[row]) + block - 1) // block
        chosen = torch.arange(n, device=DEVICE)
        if n > count:
            older = torch.randperm(n - 1, device=DEVICE)[: count - 1]
            chosen = torch.cat([older.sort().values, chosen[-1:]])
        candidates[row, : chosen.numel()] = chosen.int()
    got, compact_lens = mqa_logits.candidate_mqa_logits(
        q_values, q_scales, cache, weights, lens, table, 1, candidates, block, splits
    )
    dense = mqa_logits.paged_mqa_logits(
        q_values, q_scales, cache, weights, lens, table, 1, max_len
    )
    for row in range(rows):
        n = int(lens[row])
        listed = candidates[row][candidates[row] >= 0].long()
        keys = (listed[:, None] * block + torch.arange(block, device=DEVICE)).flatten()
        assert int(compact_lens[row]) == keys.numel()
        valid = keys < n
        assert torch.equal(got[row, : keys.numel()][valid], dense[row, keys[valid]])
        assert (got[row, : keys.numel()][~valid] == float("-inf")).all()
    out = torch.full((rows, k + 16), 77, dtype=torch.int32, device=DEVICE)
    topk.decode_topk(got, compact_lens, out[:, :k], 1, candidates, block, lens)
    masked = reference.candidate_mask(dense, lens, candidates, block)
    want = reference.decode_topk(masked, lens, k)
    torch.testing.assert_close(out[:, :k], want, rtol=0, atol=0)
    assert (out[:, k:] == 77).all()


def _fp8_index_key_cache(blocks: int, block_size: int):
    """A paged SM90 FP8 index key cache (each page's E4M3 keys, then their
    FP32 scales) and its keys decoded, `[blocks * block_size, 128]`."""
    cache = _paged(blocks, block_size, 132, 32)
    keys = torch.randn(blocks, block_size, 128, device=DEVICE)
    scales = keys.abs().amax(-1) / 448
    values = (keys / scales[..., None]).to(torch.float8_e4m3fn)
    page = cache.view(blocks, block_size * 132)
    page[:, : block_size * 128] = values.view(torch.uint8).view(blocks, -1)
    page[:, block_size * 128 :] = scales.view(torch.uint8)
    return cache, (values.float() * scales[..., None]).view(-1, 128)


@pytest.mark.parametrize("next_n", [1, 2])
def test_candidate_logits_fp8(next_n):
    """A candidate consumer on SM90's FP8 cache, its candidate blocks in any
    order: compact logits against the FP32 reference, `-inf` past a row's
    length, and the top-k the best keys of each row (all of a short row's)."""
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import (
        mqa_logits,
        topk,
    )

    torch.manual_seed(0)
    heads, count, block, k, block_size = 64, 2048, 8, 512, 64
    blocks = 1200
    cache, keys = _fp8_index_key_cache(blocks, block_size)
    requests = 8 // next_n
    table = torch.randperm(blocks, device=DEVICE, dtype=torch.int32)
    table = table.view(4, -1)[torch.arange(requests, device=DEVICE) % 4]
    max_len = table.shape[1] * block_size
    lens = [0, 300, 512, 513, 9000, count * block + 5, max_len - 3, max_len]
    if next_n > 1:
        # A request's tokens: lengths up to its last token's.
        lens = [
            max(n - (next_n - 1 - i), 0) for n in lens[1::next_n] for i in range(next_n)
        ]
    rows = len(lens)
    lens = torch.tensor(lens, dtype=torch.int32, device=DEVICE)
    q = torch.randn(rows, heads, 128, device=DEVICE).to(torch.float8_e4m3fn)
    weights = torch.randn(rows, heads, device=DEVICE)
    candidates = torch.full((rows, count), -1, dtype=torch.int32, device=DEVICE)
    for row in range(rows):
        n = (int(lens[row]) + block - 1) // block
        chosen = torch.arange(n, device=DEVICE)
        if n > count:
            older = torch.randperm(n - 1, device=DEVICE)[: count - 1]
            chosen = torch.cat([older, chosen[-1:]])
        chosen = chosen[torch.randperm(chosen.numel(), device=DEVICE)]
        candidates[row, : chosen.numel()] = chosen.int()
    got, compact_lens = mqa_logits.candidate_mqa_logits(
        q, None, cache, weights, lens, table, next_n, candidates, block
    )
    out = torch.full((rows, k + 16), 77, dtype=torch.int32, device=DEVICE)
    topk.decode_topk(got, compact_lens, out[:, :k], 1, candidates, block, lens)
    assert (out[:, k:] == 77).all()
    for row in range(rows):
        n = int(lens[row])
        listed = candidates[row][candidates[row] >= 0].long()
        positions = listed[:, None] * block + torch.arange(block, device=DEVICE)
        positions = positions.flatten()
        assert int(compact_lens[row]) == positions.numel()
        scores = got[row, : positions.numel()]
        valid = positions < n
        assert (scores[~valid] == float("-inf")).all()
        pages = table[row // next_n].long()[positions[valid] // block_size]
        k_rows = keys[pages * block_size + positions[valid] % block_size]
        dots = torch.relu(q[row].float() @ k_rows.T)
        want = weights[row] @ dots
        if n:
            scale = want.abs().max().clamp_min(1e-6)
            assert ((scores[valid] - want).abs().max() / scale) < 1e-4
        chosen = out[row, :k]
        if n <= k:
            want_out = torch.full((k,), -1, dtype=torch.int32, device=DEVICE)
            want_out[:n] = torch.arange(n, device=DEVICE)
            assert torch.equal(chosen, want_out)
        else:
            best = positions[scores.topk(k).indices].int()
            assert torch.equal(chosen.sort().values, best.sort().values)


@pytest.mark.parametrize("k,group", [(512, 1), (2048, 8)])
@pytest.mark.parametrize("ties", [False, True])
# 2^18 takes the two-launch selection; 2^14 fits one chunk (one launch, for
# both block sizes); 2^16 is one chunk of blocks exactly but four of scores.
@pytest.mark.parametrize("width", [1 << 18, 1 << 16, 1 << 14])
def test_decode_topk(k, group, ties, width):
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import topk

    torch.manual_seed(0)
    # Empty, short, exactly k, one past k, and long rows (beyond 1024 * 4 per
    # thread run, and the full width).
    lens = [0, 7, k * group, k * group + 1, 9001, 70000, width, 100000]
    logits = torch.randn(len(lens), width, device=DEVICE)
    if ties:
        logits = torch.round(logits * 4) / 4
    logits[5, :20000] = 0.0
    lens = torch.tensor(lens, dtype=torch.int32, device=DEVICE)
    out = torch.full((len(lens), k + 16), 77, dtype=torch.int32, device=DEVICE)
    topk.decode_topk(logits, lens, out[:, :k], group)
    want = reference.decode_topk(logits, lens, k, group)
    torch.testing.assert_close(out[:, :k], want, rtol=0, atol=0)
    assert (out[:, k:] == 77).all()


def test_decode_topk_matches_vllm():
    """The same sets as vLLM's decode top-k at the shapes of a decode step:
    logits as wide as the model length, rows far shorter."""
    import vllm._custom_ops  # noqa: F401  (loads the vLLM ops)
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import topk

    if not hasattr(torch.ops._C, "persistent_topk"):
        pytest.skip("this vLLM build has no persistent_topk")
    torch.manual_seed(0)
    width = 1 << 20
    for rows, length in ((6, 9000), (48, 4500), (192, 9000)):
        logits = torch.randn(rows, width, device=DEVICE)
        lens = length - torch.arange(rows, device=DEVICE, dtype=torch.int32) % 6
        want = torch.empty(rows, 512, dtype=torch.int32, device=DEVICE)
        workspace = torch.empty(1 << 20, dtype=torch.uint8, device=DEVICE)
        torch.ops._C.persistent_topk(
            logits, lens.view(-1, 1), want, workspace, 512, width
        )
        got = torch.empty_like(want)
        topk.decode_topk(logits, lens, got)
        assert torch.equal(got, want.sort(-1).values)
        del logits


def _fill_window_cache(cache: torch.Tensor) -> None:
    """Random MXFP8 window records, some values E4M3 subnormals."""
    tokens = cache.shape[0] * cache.shape[1]
    x = torch.randn(tokens, 512, device=DEVICE) * 2
    x[:, :8] *= 1e-3
    blocks = x.unflatten(-1, (16, 32))
    amax = blocks.abs().amax(-1).clamp_min(1e-30)
    exponent = torch.ceil(torch.log2(amax / 448)).clamp(-127, 127)
    values = (blocks / torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn)
    records = torch.cat(
        [values.view(torch.uint8).flatten(-2), (exponent + 127).to(torch.uint8)], -1
    )
    index = torch.arange(tokens, device=DEVICE)
    cache[index // cache.shape[1], index % cache.shape[1]] = records


@pytest.mark.parametrize(
    "tokens,heads,splits,empty,window_only",
    [
        (4, 8, None, False, False),
        (4, 8, 1, False, False),
        (3, 16, None, True, False),
        (70, 8, None, False, False),
        (5, 20, None, False, False),
        (5, 20, 1, False, False),
        (6, 4, 3, False, False),
        (6, 8, None, True, True),
        (40, 8, 1, False, True),
    ],
)
def test_sparse_mla(cos_sin, tokens, heads, splits, empty, window_only):
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import sparse_mla

    torch.manual_seed(0)
    window, top_k = 128, 512
    swa = _paged(40, 32, 528, 512 - (32 * 528) % 512)
    _fill_window_cache(swa)
    compressed = _paged(20, 64, 288, 64)
    latents = torch.randn(20 * 64, 512, device=DEVICE).bfloat16()
    index = torch.arange(20 * 64, device=DEVICE)
    compressed[index // 64, index % 64] = reference.nvfp4_kv_record(
        latents, index, cos_sin, 1
    )
    q = torch.randn(tokens, heads, 512, device=DEVICE).bfloat16()
    swa_indices = torch.randint(
        0, 40 * 32, (tokens, window), device=DEVICE, dtype=torch.int32
    )
    swa_lens = torch.randint(
        0 if empty else window // 2,
        window + 1,
        (tokens,),
        device=DEVICE,
        dtype=torch.int32,
    )
    compressed_indices = torch.randint(
        0, 20 * 64, (tokens, top_k), device=DEVICE, dtype=torch.int32
    )
    compressed_lens = torch.randint(
        0, top_k + 1, (tokens,), device=DEVICE, dtype=torch.int32
    )
    if empty:
        swa_lens[0] = 0
        compressed_lens[0] = 0
    # Unselected entries inside a row's length are -1 and skipped.
    compressed_indices[:, 7::50] = -1
    sink = torch.randn(heads, device=DEVICE)
    positions = torch.randint(0, 4096, (tokens,), device=DEVICE)
    args = (
        q,
        swa,
        swa_indices,
        swa_lens,
        None if window_only else compressed,
        None if window_only else compressed_indices,
        None if window_only else compressed_lens,
        sink,
        512**-0.5,
        positions,
        cos_sin,
    )
    want = reference.sparse_mla(*args)
    got = torch.full_like(q, float("nan"))
    sparse_mla.sparse_mla(*args, got, splits=splits)
    assert not got.isnan().any()
    assert (got.float() - want.float()).abs().max() < 2e-2 * want.float().abs().max()


def _v4_cache(blocks: int, block_size: int) -> torch.Tensor:
    """A V4-record cache of random keys, each 64-dim block at its own power
    of two (so every scale byte matters; some values E4M3 subnormals), its
    pages padded to a multiple of 576 bytes as vLLM allocates them."""
    cache = _paged(blocks, block_size, 584, -(block_size * 584) % 576)
    x = torch.randn(blocks * block_size, 8, 64, device=DEVICE)
    x *= torch.exp2(torch.randint(-3, 4, (len(x), 8, 1), device=DEVICE))
    x = x.flatten(1)
    x[:, :8] *= 1e-3
    reference.store_v4_records(cache, torch.arange(len(x), device=DEVICE), x)
    return cache


@pytest.mark.parametrize(
    "tokens,heads,splits,empty,window_only",
    [
        (6, 8, None, False, False),
        (6, 8, 1, True, False),
        (48, 8, None, False, False),
        (5, 20, None, False, False),
        (6, 8, None, True, True),
    ],
)
def test_sparse_mla_v4_records(tokens, heads, splits, empty, window_only):
    """The V4-record variant without the inverse RoPE, as SM90 runs it."""
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import sparse_mla

    torch.manual_seed(0)
    window, top_k = 128, 512
    swa, compressed = _v4_cache(24, 64), _v4_cache(40, 32)
    q = torch.randn(tokens, heads, 512, device=DEVICE).bfloat16()
    swa_indices = torch.randint(
        0, 24 * 64, (tokens, window), device=DEVICE, dtype=torch.int32
    )
    swa_lens = torch.randint(
        window // 2, window + 1, (tokens,), device=DEVICE, dtype=torch.int32
    )
    compressed_indices = torch.randint(
        0, 40 * 32, (tokens, top_k), device=DEVICE, dtype=torch.int32
    )
    compressed_lens = torch.randint(
        0, top_k + 1, (tokens,), device=DEVICE, dtype=torch.int32
    )
    if empty:
        swa_lens[0] = 0
        compressed_lens[0] = 0
    compressed_indices[:, 7::50] = -1
    sink = torch.randn(heads, device=DEVICE)
    args = (
        q,
        swa,
        swa_indices,
        swa_lens,
        None if window_only else compressed,
        None if window_only else compressed_indices,
        None if window_only else compressed_lens,
        sink,
        512**-0.5,
        None,
        None,
    )
    want = reference.sparse_mla(*args, v4_records=True)
    got = torch.full_like(q, float("nan"))
    sparse_mla.sparse_mla(*args, got, splits=splits, v4_records=True)
    assert not got.isnan().any()
    assert (got.float() - want.float()).abs().max() < 2e-2 * want.float().abs().max()
