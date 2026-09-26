# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""mm_prefix query-range metadata and the FA4 mask_mod that consumes it.

The metadata tests run anywhere. The FA4 tests need SM100 and run the real
kernel against a dense float32 reference, which is the only way to check what
cannot be reasoned about from the Python side:

* ``q_ranges[token_idx, 0]`` indexing an aux tensor with a runtime ``Int32``.
* ``cu_seqlens_q[b] + q_local`` matching FA4's own packing of a varlen batch.

The mask_mod is the *entire* mask (``(causal ∧ window) ∨ mm_prefix``), so
callers must pass ``causal=False`` and no FA-layer window. FA #155 stopped
auto-clearing those when ``mask_mod`` is set; leaving ``causal=True`` would
short out the mask_mod on SM90 and clip bidirectional ranges on SM100.
"""

from typing import Any

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import (
    fill_mm_prefix_query_ranges,
)
from vllm.v1.kv_cache_interface import KVCacheLayout


def _fa4_available() -> bool:
    if not current_platform.is_cuda():
        return False
    from vllm.v1.attention.backends.fa_utils import is_fa_version_supported

    return is_fa_version_supported(4)


requires_fa4 = pytest.mark.skipif(
    not _fa4_available(), reason="FA4 mm_prefix mask_mod requires FA4"
)

# Imported conditionally because these pull in CuTe / FA4 build artifacts that
# are absent wherever requires_fa4 skips.
if _fa4_available():
    from types import MethodType

    from tests.v1.attention.test_attention_backends import (
        MockAttentionLayer,
        create_and_prepopulate_kv_cache,
    )
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_standard_kv_cache_spec,
        create_vllm_config,
    )
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionMetadataBuilder,
        _make_mm_prefix_mask_mod,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec, get_kv_quant_mode
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
HEAD_SIZE = 128
NUM_HEADS = 8
NUM_KV_HEADS = 8
SLIDING_WINDOW_LEFT = 129


# --------------------------------------------------------------------------- #
# Metadata construction
# --------------------------------------------------------------------------- #

# Stands in for the builder's persistent (max_num_batched_tokens, 2) buffer.
STAGING_CAPACITY = 64


def _query_ranges(mm_ranges, query_lens, seq_lens):
    """Fill a staging buffer and return the written rows, or None if empty."""
    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32
    )
    # Poison the buffer so a stale row surviving the fill is visible.
    out = np.full((STAGING_CAPACITY, 2), 12345, dtype=np.int32)
    num_tokens = fill_mm_prefix_query_ranges(
        out,
        mm_ranges,
        query_start_loc,
        torch.tensor(seq_lens, dtype=torch.int32),
    )
    if num_tokens == 0:
        return None
    return torch.from_numpy(out[:num_tokens])


def test_matches_range_scan_semantics_with_context_offset():
    """Pin the equivalence the O(1) lookup rests on.

    The kernel checks ``r_start <= kv_idx <= r_end`` for the range containing
    the query token, which is only equal to the old scan's
    ``any(q in r and kv in r)`` because ranges never overlap.  Request 1 carries
    a context offset so the local-to-absolute query mapping is exercised too.
    """
    mm_ranges = {0: [(1, 3), (5, 7)], 1: [(2, 4), (9, 12)]}
    query_lens = [8, 6]
    seq_lens = [8, 13]

    query_ranges = _query_ranges(mm_ranges, query_lens, seq_lens)
    assert query_ranges is not None

    token_start = 0
    for req_idx, query_len in enumerate(query_lens):
        context_len = seq_lens[req_idx] - query_len
        for q_local in range(query_len):
            q_abs = context_len + q_local
            r_start, r_end = query_ranges[token_start + q_local].tolist()
            for kv_idx in range(seq_lens[req_idx]):
                old_scan_keep = any(
                    start < end and start <= q_abs <= end and start <= kv_idx <= end
                    for start, end in mm_ranges[req_idx]
                )
                new_keep = r_start <= kv_idx <= r_end
                assert new_keep == old_scan_keep, (req_idx, q_abs, kv_idx)
        token_start += query_len


def test_ranges_beyond_scheduled_chunk_are_clipped():
    """Chunked prefill must not error or produce a wrong mask.

    ``disable_chunked_mm_input`` keeps a single mm item intact but still splits
    a prompt across steps, so a request's ranges routinely sit entirely past the
    tokens scheduled so far.  Sizing by query token makes the out-of-chunk part
    a no-op rather than an out-of-bounds condition.
    """
    # Step 1 of "TTTT IIIIII": only the 4 text tokens are scheduled, so the
    # range is entirely ahead of the chunk. The old bounds check raised here.
    assert _query_ranges({0: [(4, 9)]}, query_lens=[4], seq_lens=[4]) is None

    # A range starting mid-chunk and running past its end keeps its absolute
    # bounds, so the bidirectional block still reaches the range's later keys
    # once they are cached. Query tokens here are absolute positions 2..5.
    query_ranges = _query_ranges({0: [(4, 9)]}, query_lens=[4], seq_lens=[6])
    assert query_ranges is not None
    expected = torch.tensor([[-1, -1], [-1, -1], [4, 9], [4, 9]], dtype=torch.int32)
    torch.testing.assert_close(query_ranges, expected)


def test_buffer_reuse_does_not_leak_previous_rows():
    """The staging buffer persists across steps, so every reported row must be
    rewritten. Guards against dropping the ``-1`` fill as an optimization: a
    step whose ranges shrink would otherwise expose the prior step's bounds and
    silently widen the bidirectional mask.
    """
    out = np.zeros((STAGING_CAPACITY, 2), dtype=np.int32)
    query_start_loc = torch.tensor([0, 8], dtype=torch.int32)
    seq_lens = torch.tensor([8], dtype=torch.int32)

    num_tokens = fill_mm_prefix_query_ranges(
        out, {0: [(0, 7)]}, query_start_loc, seq_lens
    )
    assert num_tokens == 8
    np.testing.assert_array_equal(out[:8], np.tile([0, 7], (8, 1)))

    # Same rows scheduled, but now only tokens 2..3 sit inside a range.
    num_tokens = fill_mm_prefix_query_ranges(
        out, {0: [(2, 3)]}, query_start_loc, seq_lens
    )
    assert num_tokens == 8
    expected = np.full((8, 2), -1, dtype=np.int32)
    expected[2:4] = (2, 3)
    np.testing.assert_array_equal(out[:8], expected)


def test_returns_none_when_no_range_covers_a_query_token():
    """No aux tensor means forward() skips the mask_mod entirely.

    Returning None rather than an all-``-1`` tensor is what keeps text-only and
    decode-only batches off the fill path.
    """
    # Text-only batch on an mm_prefix model: every request present, no ranges.
    assert _query_ranges({0: [], 1: []}, query_lens=[2, 2], seq_lens=[2, 2]) is None
    # Degenerate single-token ranges are skipped, matching the Triton path's
    # `start < end` validity check.
    assert _query_ranges({0: [(1, 1)]}, query_lens=[4], seq_lens=[4]) is None
    # Decode rows: the query token is generated, so it is in no range.
    assert _query_ranges({0: [(1, 3)]}, query_lens=[1], seq_lens=[9]) is None


# --------------------------------------------------------------------------- #
# FA4 kernel
# --------------------------------------------------------------------------- #


def _cu_seqlens(lens: list[int]) -> torch.Tensor:
    out = torch.zeros(len(lens) + 1, dtype=torch.int32)
    out[1:] = torch.tensor(lens, dtype=torch.int32).cumsum(0)
    return out


def _dense_reference(
    q,
    k_per_req,
    v_per_req,
    query_lens,
    seq_lens,
    mm_ranges,
    sliding_window_left,
    scale,
    mm_clamp_sw=0,
):
    """Dense float32 ``(causal AND window) OR mm_prefix`` per request.

    ``mm_clamp_sw`` confines the bidirectional block to the window, reproducing
    Gemma4's ``mm_prefix_clamp_sliding_window``. Only the past bound applies, so
    future keys inside a range still pass.
    """
    out = torch.empty_like(q, dtype=torch.float32)
    repeats = q.shape[1] // k_per_req[0].shape[1]
    q_off = 0
    for req_idx, (q_len, k_len) in enumerate(zip(query_lens, seq_lens)):
        ctx = k_len - q_len
        q_pos = torch.arange(q_len, device=q.device) + ctx
        k_pos = torch.arange(k_len, device=q.device)
        delta = q_pos[:, None] - k_pos[None, :]

        keep = delta >= 0
        if sliding_window_left is not None:
            keep &= delta < sliding_window_left

        for start, end in mm_ranges.get(req_idx, []):
            if start >= end:
                continue
            q_in = (q_pos >= start) & (q_pos <= end)
            k_in = (k_pos >= start) & (k_pos <= end)
            mm = q_in[:, None] & k_in[None, :]
            if mm_clamp_sw > 0:
                mm &= delta < mm_clamp_sw
            keep |= mm

        q_i = q[q_off : q_off + q_len].float().transpose(0, 1)
        k_i = k_per_req[req_idx].float().transpose(0, 1)
        v_i = v_per_req[req_idx].float().transpose(0, 1)
        if repeats > 1:
            k_i = k_i.repeat_interleave(repeats, dim=0)
            v_i = v_i.repeat_interleave(repeats, dim=0)

        scores = (q_i @ k_i.transpose(-1, -2)) * scale
        scores = scores.masked_fill(~keep[None], float("-inf"))
        out[q_off : q_off + q_len] = (scores.softmax(-1) @ v_i).transpose(0, 1)
        q_off += q_len
    return out


def _split(packed: torch.Tensor, lens: list[int]) -> list[torch.Tensor]:
    return list(packed.split(lens))


def _reference(q, k, v, query_lens, seq_lens, mm_ranges, sw_left, scale, **kwargs):
    return _dense_reference(
        q,
        _split(k, seq_lens),
        _split(v, seq_lens),
        query_lens,
        seq_lens,
        mm_ranges,
        sw_left,
        scale,
        **kwargs,
    )


def _run_kernel(
    q, k, v, query_lens, seq_lens, mm_ranges, sliding_window_left, scale, mm_clamp_sw=0
):
    cu_q = _cu_seqlens(query_lens)
    cu_k = _cu_seqlens(seq_lens)

    staging = torch.full((int(cu_q[-1]), 2), 12345, dtype=torch.int32).numpy()
    num_rows = fill_mm_prefix_query_ranges(
        staging, mm_ranges, cu_q, torch.tensor(seq_lens, dtype=torch.int32)
    )
    assert num_rows > 0, "test case must have at least one in-range query token"
    q_ranges = torch.from_numpy(staging[:num_rows]).to(DEVICE)
    cu_q_gpu = cu_q.to(DEVICE)

    out = torch.empty_like(q)
    flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_q_gpu,
        cu_seqlens_k=cu_k.to(DEVICE),
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(seq_lens),
        softmax_scale=scale,
        causal=False,
        fa_version=4,
        mask_mod=_make_mm_prefix_mask_mod(
            sliding_window=mm_clamp_sw, sliding_window_left=sliding_window_left
        ),
        aux_tensors=[q_ranges, cu_q_gpu],
    )
    return out


def _randn_qkv(query_lens, seq_lens):
    torch.manual_seed(0)
    q = torch.randn(sum(query_lens), NUM_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    k = torch.randn(sum(seq_lens), NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    v = torch.randn(sum(seq_lens), NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    return q, k, v, HEAD_SIZE**-0.5


# (name, query_lens, seq_lens, mm_ranges). One shape per index computation the
# mask_mod performs; shapes that differ only in request count are omitted.
CASES = [
    # cu_seqlens_q[b] + q_local: unequal query lens across the batch, the
    # packing the pooling workload never produced.
    (
        "varlen_batch",
        [128, 64, 200],
        [128, 64, 200],
        {0: [(8, 71)], 1: [(0, 31), (40, 55)], 2: [(64, 191)]},
    ),
    # q_abs = q_idx + (seqlen_k - seqlen_q): ranges are absolute prompt
    # positions, so part of each range sits in the context.
    (
        "context_offset",
        [64],
        [320],
        {0: [(32, 95), (200, 287)]},
    ),
    # Decode rows: the query token is generated, so it is in no range and must
    # get the (-1, -1) sentinel. One prefill row keeps the batch mixed.
    (
        "mixed_prefill_decode",
        [160, 1, 1, 1],
        [160, 512, 300, 97],
        {0: [(32, 127)], 1: [(4, 259)], 2: [(8, 71)], 3: [(1, 64)]},
    ),
]

# A 320-token range, wider than the window. Every CASES range is narrower than
# the window and therefore clamp-insensitive, so the window modes are exercised
# on this shape instead: images larger than the window are exactly why
# mm_prefix_clamp_sliding_window exists.
WIDE_RANGE_CASE = ([384], [384], {0: [(32, 351)]})

# (sliding_window_left, mm_clamp_sw). sliding_window_left selects between the
# two mask_mod variants; mm_clamp_sw gates the Gemma4 clamp inside the sliding
# one. Gemma4 sets mm_clamp_sw == sw_val, and the unclamped variant ignores
# mm_clamp_sw, so a clamp without a window is not a reachable configuration.
WINDOW_MODES = [
    (None, 0),
    (SLIDING_WINDOW_LEFT, 0),
    (SLIDING_WINDOW_LEFT, SLIDING_WINDOW_LEFT),
]
WINDOW_IDS = ["full_causal", "window", "window_clamped"]


@requires_fa4
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_mm_prefix_mask_mod_matches_dense_reference(case):
    _, query_lens, seq_lens, mm_ranges = case
    q, k, v, scale = _randn_qkv(query_lens, seq_lens)
    args = (q, k, v, query_lens, seq_lens, mm_ranges, SLIDING_WINDOW_LEFT, scale)
    torch.testing.assert_close(
        _run_kernel(*args).float(), _reference(*args), atol=2e-2, rtol=2e-2
    )


@requires_fa4
@pytest.mark.parametrize("window_mode", WINDOW_MODES, ids=WINDOW_IDS)
def test_window_modes_on_a_range_wider_than_the_window(window_mode):
    """Cover the three reachable window configurations.

    The clamped mode additionally asserts the clamp changes the mask at all,
    without which a range narrower than the window would leave the clamp branch
    unexercised while the test still passed.
    """
    sliding_window_left, mm_clamp_sw = window_mode
    query_lens, seq_lens, mm_ranges = WIDE_RANGE_CASE
    q, k, v, scale = _randn_qkv(query_lens, seq_lens)
    args = (q, k, v, query_lens, seq_lens, mm_ranges, sliding_window_left, scale)

    expected = _reference(*args, mm_clamp_sw=mm_clamp_sw)
    if mm_clamp_sw:
        assert not torch.allclose(
            expected, _reference(*args, mm_clamp_sw=0), atol=2e-2, rtol=2e-2
        ), "range is not wide enough to exercise the clamp branch"

    actual = _run_kernel(*args, mm_clamp_sw=mm_clamp_sw)
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)


@requires_fa4
def test_ranges_outside_chunk_degrade_to_causal():
    """A range fully behind the scheduled chunk must not widen the mask.

    Under chunked prefill a request's ranges routinely sit entirely in the
    already-computed context. Indexing by query token makes those a no-op; the
    result must be identical to running with no ranges at all.
    """
    query_lens, seq_lens = [64], [512]
    q, k, v, scale = _randn_qkv(query_lens, seq_lens)

    # Range (16, 79) is entirely below the chunk start (448), plus one live
    # range so the fill still reports rows.
    args = (q, k, v, query_lens, seq_lens)
    actual = _run_kernel(*args, {0: [(16, 79), (448, 511)]}, None, scale)
    expected = _reference(*args, {0: [(448, 511)]}, None, scale)
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------- #
# FA4 through the metadata builder and the paged KV cache
# --------------------------------------------------------------------------- #

# Stock google/gemma-4-E2B-it ships text_config.use_bidirectional_attention as
# null, so is_mm_prefix_lm is False out of the box. Tests that exercise the
# builder must flip it via `_enable_mm_prefix`. Only config.json / tokenizer
# are read; weights are never loaded.
MODEL = "google/gemma-4-E2B-it"

BLOCK_SIZE = 16
SLIDING_WINDOW = 256

# Request 0 is a prefill chunk whose range starts mid-context; requests 1-3 are
# decodes whose generated token sits outside every range.
PAGED_SEQ_LENS = [352, 513, 200, 97]
PAGED_QUERY_LENS = [96, 1, 1, 1]
PAGED_MM_RANGES = {
    0: [(224, 287), (300, 351)],
    1: [(4, 259)],
    2: [(0, 63), (80, 143)],
    3: [(1, 64)],
}


def _enable_mm_prefix(vllm_config):
    """Flip stock Gemma4's null use_bidirectional_attention to "vision".

    ``ModelConfig.__init__`` derives ``model_arch_config`` once from
    ``hf_text_config``, and ``is_mm_prefix_lm`` is a ``cached_property`` that
    may already hold False from that first derivation, so re-derive and drop
    the cache.
    """
    model_config = vllm_config.model_config
    model_config.hf_text_config.use_bidirectional_attention = "vision"
    model_config.model_arch_config = model_config.get_model_arch_config()
    model_config.__dict__.pop("is_mm_prefix_lm", None)
    return vllm_config


@requires_fa4
def test_decode_only_batch_reports_no_ranges():
    """Decode-only steps must carry no mm_prefix metadata at all.

    This is what makes FULL CUDA graph capture consistent for this feature.
    Capture runs through ``_dummy_run`` with no multimodal requests, so no
    mask_mod is attached to the captured graph; a decode-only replay must
    therefore also want no mask_mod, or the captured graph would silently drop
    it. Indexing by query token gives that for free, because a generated token
    is always past every prompt range.

    The range-id scheme keyed off the request having ranges at all, so it
    attached a mask_mod to decode steps and did not have this property.
    """
    vllm_config = _enable_mm_prefix(
        create_vllm_config(
            model_name=MODEL,
            max_model_len=1024,
            block_size=BLOCK_SIZE,
            num_gpu_blocks=256,
        )
    )
    decode_batch = BatchSpec(
        seq_lens=[513, 200, 97], query_lens=[1, 1, 1], name="decode_only"
    )
    common = create_common_attn_metadata(decode_batch, BLOCK_SIZE, DEVICE)
    common.mm_req_doc_ranges = {0: [(4, 259)], 1: [(0, 63)], 2: [(1, 64)]}

    builder = FlashAttentionMetadataBuilder(
        create_standard_kv_cache_spec(vllm_config),
        ["model.layers.0.self_attn.attn"],
        vllm_config,
        DEVICE,
    )
    md = builder.build(common_prefix_len=0, common_attn_metadata=common)
    assert md.mm_prefix_query_range_tensor is None


@requires_fa4
@pytest.mark.parametrize(
    "head_size",
    [
        # 128: FA4's default SM100 path (not the dedicated hd256 kernel), so
        # paged KV + seqused_k works on B200 and H100. Non-paged mm_prefix
        # cases in this file already use HEAD_SIZE=128.
        pytest.param(128, id="hd128"),
        # 512: Gemma4 global head dim; FA4 resolves on H100, skips on
        # Blackwell (TMEM).
        pytest.param(512, id="global_512"),
    ],
)
def test_mm_prefix_kv_cache_path(head_size: int):
    """FA4 + mask_mod + block_table matches a dense reference.

    Use head_size=128 on B200/H100 (FA4 + paged/`seqused_k` supported) and
    512 on H100 only (Blackwell FA4 rejects 512 via TMEM). Avoid 256: SM100's
    dedicated hd256 2CTA kernel asserts away ``seqused_k``, which paged KV
    requires.

    Tensors, kv_cache_spec, and FlashAttentionImpl must all use the same
    ``head_size`` — ``run_attention_backend`` / ``create_standard_kv_cache_spec``
    read ``get_head_size()`` (=512 for Gemma4), so this test builds them
    explicitly. The sliding window is set on the impl; mm_prefix encodes it in
    mask_mod and clears FA's built-in window at forward time.
    """
    torch.manual_seed(0)
    vllm_config = _enable_mm_prefix(
        create_vllm_config(
            model_name=MODEL,
            max_model_len=max(PAGED_SEQ_LENS),
            block_size=BLOCK_SIZE,
            num_gpu_blocks=2048,
        )
    )
    assert vllm_config.model_config.is_mm_prefix_lm

    mc = vllm_config.model_config
    # Keep every get_head_size() reader (builder helpers, etc.) on the same dim.
    mc.get_head_size = MethodType(lambda self: head_size, mc)

    # Skip under the same config Gemma4 forces (flash_attn_version=4), so the
    # check matches FlashAttentionImpl's get_flash_attn_version call.
    with set_current_vllm_config(vllm_config):
        if get_flash_attn_version(head_size=head_size) != 4:
            pytest.skip(f"FA4 does not support head_size={head_size} on this device")

    num_heads = mc.get_num_attention_heads(vllm_config.parallel_config)
    num_kv_heads = mc.get_num_kv_heads(vllm_config.parallel_config)
    scale = head_size**-0.5

    batch = BatchSpec(
        seq_lens=PAGED_SEQ_LENS, query_lens=PAGED_QUERY_LENS, name="mm_prefix_mixed"
    )
    kv_cache_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=mc.dtype,
        sliding_window=SLIDING_WINDOW,
        kv_quant_mode=get_kv_quant_mode(vllm_config.cache_config.cache_dtype),
    )
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    common.mm_req_doc_ranges = PAGED_MM_RANGES

    qs, new_ks, new_vs, k_fulls, v_fulls, k_ctxs, v_ctxs = [], [], [], [], [], [], []
    for q_len, s_len in zip(PAGED_QUERY_LENS, PAGED_SEQ_LENS):
        ctx = s_len - q_len
        shape = (num_kv_heads, head_size)
        qs.append(torch.randn(q_len, num_heads, head_size, dtype=DTYPE, device=DEVICE))
        k_full = torch.randn(s_len, *shape, dtype=DTYPE, device=DEVICE)
        v_full = torch.randn(s_len, *shape, dtype=DTYPE, device=DEVICE)
        k_fulls.append(k_full)
        v_fulls.append(v_full)
        k_ctxs.append(k_full[:ctx])
        v_ctxs.append(v_full[:ctx])
        new_ks.append(k_full[ctx:])
        new_vs.append(v_full[ctx:])

    query = torch.cat(qs)
    key = torch.cat(new_ks)
    value = torch.cat(new_vs)
    kv_cache = create_and_prepopulate_kv_cache(
        k_contexts=k_ctxs,
        v_contexts=v_ctxs,
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=DTYPE,
        device=DEVICE,
        num_blocks=2048,
        common_attn_metadata=common,
        layout=KVCacheLayout.LBNHC,
        randomize_blocks=True,
    )

    # Packed KV layout is [..., 2 * head_size]; impl.split(head_size) needs match.
    assert kv_cache.shape[-1] == 2 * head_size

    layer_names = ["model.layers.0.self_attn.attn"]
    with set_current_vllm_config(vllm_config):
        builder = FlashAttentionMetadataBuilder(
            kv_cache_spec, layer_names, vllm_config, DEVICE
        )
        attn_metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)
        # head_size=128 keeps FA4 under local attention on Blackwell (the
        # FA4→FA2 demotion is only for local+256). Window is still encoded in
        # mask_mod at forward; FA's built-in window is cleared there.
        impl = FlashAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=SLIDING_WINDOW,
            kv_cache_dtype="auto",
        )
        assert impl.vllm_flash_attn_version == 4, (
            f"expected FA4, got FA{impl.vllm_flash_attn_version} for "
            f"head_size={head_size}"
        )
        assert impl.head_size == head_size == kv_cache_spec.head_size

        mock_layer = MockAttentionLayer(DEVICE)
        output = torch.empty_like(query)
        impl.do_kv_cache_update(
            mock_layer, key, value, kv_cache, attn_metadata.slot_mapping
        )
        actual = impl.forward(
            mock_layer, query, key, value, kv_cache, attn_metadata, output=output
        )

    ref_args = (query, k_fulls, v_fulls, PAGED_QUERY_LENS, PAGED_SEQ_LENS)
    expected = _dense_reference(*ref_args, PAGED_MM_RANGES, SLIDING_WINDOW, scale)
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)

    # Without this the assertion above would also pass if the mask_mod were
    # silently dropped, since the batch would then be plain causal + window.
    causal_only = _dense_reference(*ref_args, {}, SLIDING_WINDOW, scale)
    assert not torch.allclose(expected, causal_only, atol=2e-2, rtol=2e-2), (
        "test batch does not actually exercise the mm_prefix branch"
    )


# --------------------------------------------------------------------------- #
# FlashInfer JIT-variant path
# --------------------------------------------------------------------------- #
# The FlashInfer backend serves mm-prefix through FlashInfer's own wrapper,
# which consumes the per-query [start, end] rows directly, so none of these
# tests depend on FA4.

FLASHINFER_MM_SLIDING_WINDOW = 32
"""Narrower than PAGED_MM_RANGES' widest range (224, 287), so the clamp-on
and clamp-off window formulas produce different outputs on this batch."""


@pytest.mark.parametrize(
    ("capability", "head_size", "block_size", "kv_dtype", "sink", "jit", "reason"),
    [
        # Plain NVFP4 on pre-SM100 runs on fa2 through the scale-factor tensors.
        ((8, 0), 128, 16, "nvfp4", False, True, None),
        # SM100 serves NVFP4 with trtllm-gen, which cannot run a custom variant.
        ((10, 0), 128, 16, "nvfp4", False, True, "SM100"),
        # The store-time scale search exists only in trtllm-gen.
        ((8, 0), 128, 16, "nvfp4_4over6", False, True, "nvfp4_4over6"),
        ((8, 0), 128, 16, "fp8", False, True, "fp8"),
        ((8, 0), 128, 16, None, True, True, "sink"),
        # trtllm-only page sizes fail at selection, not at the first build().
        ((8, 0), 128, 128, None, False, True, "page size 128"),
        ((8, 0), 128, 64, None, False, True, None),
        # Hybrid models select per group, so the probe sees the group's head size.
        ((8, 0), 256, 16, "float16", False, True, None),
        ((8, 0), 512, 16, "float16", False, True, None),
        ((8, 0), 128, 16, None, False, False, "mm-prefix attention"),
    ],
)
def test_flashinfer_mm_prefix_supports_combination(
    monkeypatch, capability, head_size, block_size, kv_dtype, sink, jit, reason
):
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends import flashinfer as fi

    probed: list[tuple] = []

    def probe(*args) -> bool:
        probed.append(args)
        return jit

    monkeypatch.setattr(fi, "_mm_prefix_jit_available", probe)

    result = fi.FlashInferBackend.supports_combination(
        head_size=head_size,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_dtype,
        block_size=block_size,
        use_mla=False,
        has_sink=sink,
        use_sparse=False,
        use_mm_prefix=True,
        device_capability=DeviceCapability(*capability),
    )

    if reason is not None:
        assert reason in result
        return
    assert result is None
    expected_kv = {"nvfp4": torch.uint8, "float16": torch.float16}.get(kv_dtype)
    if expected_kv is not None:
        assert probed[-1] == (torch.bfloat16, expected_kv, torch.bfloat16, head_size)


@pytest.mark.parametrize(
    ("use_dcp", "use_rswa", "reason"),
    [(True, False, "DCP"), (False, True, "R-SWA not supported"), (False, False, None)],
)
def test_flashinfer_mm_prefix_validate_configuration(use_dcp, use_rswa, reason):
    """DCP is rejected here; R-SWA is decided by the base class, so the
    override has to pass the flag through."""
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    reasons = FlashInferBackend.validate_configuration(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype=None,
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=True,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(8, 0),
        attn_type="decoder",
        use_dcp=use_dcp,
        use_rswa=use_rswa,
    )

    if reason is None:
        assert not any("DCP" in r or "R-SWA" in r for r in reasons), reasons
    else:
        assert any(reason in r for r in reasons), reasons


def _flashinfer_builder_env(sliding_window: int | None):
    """vllm_config + kv_cache_spec + per-layer-parameter mock for FlashInfer."""
    import unittest.mock

    from tests.v1.attention.utils import create_vllm_config
    from vllm.v1.attention.backends.utils import PerLayerParameters
    from vllm.v1.kv_cache_interface import FullAttentionSpec, get_kv_quant_mode

    vllm_config = _enable_mm_prefix(
        create_vllm_config(
            model_name=MODEL,
            max_model_len=max(PAGED_SEQ_LENS),
            block_size=BLOCK_SIZE,
            num_gpu_blocks=2048,
        )
    )
    mc = vllm_config.model_config
    kv_cache_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=mc.get_num_kv_heads(vllm_config.parallel_config),
        head_size=HEAD_SIZE,
        dtype=mc.dtype,
        sliding_window=sliding_window,
        kv_quant_mode=get_kv_quant_mode(vllm_config.cache_config.cache_dtype),
    )
    window_left = sliding_window - 1 if sliding_window is not None else -1
    scale = HEAD_SIZE**-0.5

    def mock_params(vllm_config, layer_names, impl_cls, soft_cap=0.0):
        return {
            name: PerLayerParameters(
                window_left=window_left,
                logits_soft_cap=soft_cap,
                sm_scale=scale,
                has_sinks=False,
            )
            for name in layer_names
        }

    patch = unittest.mock.patch(
        "vllm.v1.attention.backends.flashinfer.get_per_layer_parameters",
        mock_params,
    )
    return vllm_config, kv_cache_spec, patch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_flashinfer_mm_prefix_builder_is_fail_closed():
    """soft-cap and explicitly forced trtllm must die at engine start."""
    import unittest.mock
    from functools import partial

    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    vllm_config, kv_cache_spec, patch = _flashinfer_builder_env(SLIDING_WINDOW)
    layer_names = ["model.layers.0.self_attn.attn"]

    mock_params = patch.new
    with set_current_vllm_config(vllm_config):
        with (
            unittest.mock.patch(
                "vllm.v1.attention.backends.flashinfer.get_per_layer_parameters",
                partial(mock_params, soft_cap=30.0),
            ),
            pytest.raises(NotImplementedError, match="logits_soft_cap"),
        ):
            FlashInferMetadataBuilder(
                kv_cache_spec, layer_names, vllm_config, torch.device("cuda:0")
            )

        vllm_config.attention_config.use_trtllm_attention = True
        with patch, pytest.raises(NotImplementedError, match="trtllm"):
            FlashInferMetadataBuilder(
                kv_cache_spec, layer_names, vllm_config, torch.device("cuda:0")
            )
        vllm_config.attention_config.use_trtllm_attention = None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_flashinfer_mm_prefix_text_only_and_decode_only_keep_causal():
    """Empty range dicts and decode-only batches must not select the mm path.

    Neither case may construct the mm wrapper, but the builder's fail-fast
    backstop still requires the capability to be available here.
    """
    from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.flashinfer import (
        FlashInferMetadataBuilder,
        _mm_prefix_jit_available,
    )

    vllm_config, kv_cache_spec, patch = _flashinfer_builder_env(SLIDING_WINDOW)
    mc = vllm_config.model_config
    if not _mm_prefix_jit_available(mc.dtype, mc.dtype, mc.dtype, HEAD_SIZE):
        pytest.skip("FlashInfer mm-prefix attention unavailable here")
    layer_names = ["model.layers.0.self_attn.attn"]
    device = torch.device("cuda:0")

    with set_current_vllm_config(vllm_config), patch:
        builder = FlashInferMetadataBuilder(
            kv_cache_spec, layer_names, vllm_config, device
        )

        # Text-only: ranges dict present but every list empty.
        batch = BatchSpec(seq_lens=[352, 200], query_lens=[96, 60], name="text_only")
        common = create_common_attn_metadata(batch, BLOCK_SIZE, device)
        common.mm_req_doc_ranges = {0: [], 1: []}
        md = builder.build(common_prefix_len=0, common_attn_metadata=common)
        assert md.mm_prefix_query_range_tensor is None
        assert md.prefill is not None and md.prefill.mm_wrapper is None

        # Decode-only: every query token sits past every prompt range.
        batch = BatchSpec(seq_lens=[513, 200, 97], query_lens=[1, 1, 1], name="decode")
        common = create_common_attn_metadata(batch, BLOCK_SIZE, device)
        common.mm_req_doc_ranges = {0: [(4, 259)], 1: [(0, 63)], 2: [(1, 64)]}
        md = builder.build(common_prefix_len=0, common_attn_metadata=common)
        assert md.mm_prefix_query_range_tensor is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_flashinfer_mm_prefix_kv_cache_path():
    """FlashInfer mm-prefix attention + paged KV matches the dense reference.

    The sliding window (32) is narrower than the widest range in
    PAGED_MM_RANGES, so the clamp-off and clamp-on formulas genuinely differ
    on this batch and both are checked against their own reference. The
    cleared-metadata fallback (Gemma4 clears the field for its full-attention
    groups at forward time) must reproduce the plain causal wrapper.
    """
    from tests.v1.attention.test_attention_backends import (
        MockAttentionLayer,
        create_and_prepopulate_kv_cache,
    )
    from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.flashinfer import (
        FlashInferImpl,
        FlashInferMetadataBuilder,
        _mm_prefix_jit_available,
    )
    from vllm.v1.kv_cache_interface import KVCacheLayout

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    sliding_window = FLASHINFER_MM_SLIDING_WINDOW
    vllm_config, kv_cache_spec, patch = _flashinfer_builder_env(sliding_window)
    mc = vllm_config.model_config
    if not _mm_prefix_jit_available(mc.dtype, mc.dtype, mc.dtype, HEAD_SIZE):
        pytest.skip("FlashInfer mm-prefix attention unavailable here")

    from types import MethodType

    mc.get_head_size = MethodType(lambda self: HEAD_SIZE, mc)
    num_heads = mc.get_num_attention_heads(vllm_config.parallel_config)
    num_kv_heads = mc.get_num_kv_heads(vllm_config.parallel_config)
    scale = HEAD_SIZE**-0.5

    batch = BatchSpec(
        seq_lens=PAGED_SEQ_LENS, query_lens=PAGED_QUERY_LENS, name="fi_mm_mixed"
    )
    common = create_common_attn_metadata(batch, BLOCK_SIZE, device)
    common.mm_req_doc_ranges = PAGED_MM_RANGES

    qs, new_ks, new_vs, k_fulls, v_fulls, k_ctxs, v_ctxs = [], [], [], [], [], [], []
    for q_len, s_len in zip(PAGED_QUERY_LENS, PAGED_SEQ_LENS, strict=True):
        ctx = s_len - q_len
        shape = (num_kv_heads, HEAD_SIZE)
        qs.append(torch.randn(q_len, num_heads, HEAD_SIZE, dtype=DTYPE, device=device))
        k_full = torch.randn(s_len, *shape, dtype=DTYPE, device=device)
        v_full = torch.randn(s_len, *shape, dtype=DTYPE, device=device)
        k_fulls.append(k_full)
        v_fulls.append(v_full)
        k_ctxs.append(k_full[:ctx])
        v_ctxs.append(v_full[:ctx])
        new_ks.append(k_full[ctx:])
        new_vs.append(v_full[ctx:])

    query, key, value = torch.cat(qs), torch.cat(new_ks), torch.cat(new_vs)
    kv_cache = create_and_prepopulate_kv_cache(
        k_contexts=k_ctxs,
        v_contexts=v_ctxs,
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
        device=device,
        num_blocks=2048,
        common_attn_metadata=common,
        layout=KVCacheLayout.LBNHC,
        randomize_blocks=True,
    )

    layer_names = ["model.layers.0.self_attn.attn"]
    with set_current_vllm_config(vllm_config), patch:
        builder = FlashInferMetadataBuilder(
            kv_cache_spec, layer_names, vllm_config, device
        )
        attn_metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)
        assert attn_metadata.mm_prefix_query_range_tensor is not None
        assert attn_metadata.prefill is not None
        assert attn_metadata.prefill.mm_wrapper is not None

        impl = FlashInferImpl(
            num_heads=num_heads,
            head_size=HEAD_SIZE,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=sliding_window,
            kv_cache_dtype="auto",
        )
        # mm_prefix_clamp_sliding_window is read off the layer with getattr(),
        # so the mock grows it here; the annotation keeps mypy from rejecting
        # an attribute the stub class does not declare.
        mock_layer: Any = MockAttentionLayer(device)
        impl.do_kv_cache_update(
            mock_layer, key, value, kv_cache, attn_metadata.slot_mapping
        )

        ref_args = (query, k_fulls, v_fulls, PAGED_QUERY_LENS, PAGED_SEQ_LENS)
        expected_by_clamp = {}
        for clamp in (False, True):
            mock_layer.mm_prefix_clamp_sliding_window = clamp
            output = torch.empty_like(query)
            actual = impl.forward(
                mock_layer, query, key, value, kv_cache, attn_metadata, output=output
            )
            expected = _dense_reference(
                *ref_args,
                PAGED_MM_RANGES,
                sliding_window,
                scale,
                mm_clamp_sw=sliding_window if clamp else 0,
            )
            expected_by_clamp[clamp] = expected
            torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)

        # The two window formulas must actually differ on this batch, and
        # both must differ from plain causal + window.
        causal_only = _dense_reference(*ref_args, {}, sliding_window, scale)
        assert not torch.allclose(
            expected_by_clamp[False], expected_by_clamp[True], atol=2e-2, rtol=2e-2
        )
        assert not torch.allclose(
            expected_by_clamp[False], causal_only, atol=2e-2, rtol=2e-2
        )

        # Cleared metadata field -> plain causal wrapper.
        attn_metadata.mm_prefix_query_range_tensor = None
        mock_layer.mm_prefix_clamp_sliding_window = False
        output = torch.empty_like(query)
        actual = impl.forward(
            mock_layer, query, key, value, kv_cache, attn_metadata, output=output
        )
        torch.testing.assert_close(actual.float(), causal_only, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_flashinfer_mm_prefix_short_extend_stays_a_prefill():
    """A short chunked-prefill tail inside a range must not become a decode.

    With the default treat_short_extends_as_decodes=True a query_len==1
    still-prefilling request is routed to the causal decode path, losing
    access to the future tokens of its bidirectional range. On mm batches
    the builder reclassifies via is_prefilling.
    """
    from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.flashinfer import (
        FlashInferMetadataBuilder,
        _mm_prefix_jit_available,
    )

    device = torch.device("cuda:0")
    vllm_config, kv_cache_spec, patch = _flashinfer_builder_env(SLIDING_WINDOW)
    mc = vllm_config.model_config
    if not _mm_prefix_jit_available(mc.dtype, mc.dtype, mc.dtype, HEAD_SIZE):
        pytest.skip("FlashInfer mm-prefix attention unavailable here")

    # Reordered batches put decodes first. Request 0: a plain decode.
    # Request 1: query_len 1 but still prefilling (chunked tail at absolute
    # position 128) inside range (100, 200).
    batch = BatchSpec(seq_lens=[300, 129], query_lens=[1, 1], name="short_extend")
    common = create_common_attn_metadata(batch, BLOCK_SIZE, device)
    common.mm_req_doc_ranges = {1: [(100, 200)]}
    common.is_prefilling = torch.tensor([False, True])

    layer_names = ["model.layers.0.self_attn.attn"]
    with set_current_vllm_config(vllm_config), patch:
        builder = FlashInferMetadataBuilder(
            kv_cache_spec, layer_names, vllm_config, device
        )
        md = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert md.mm_prefix_query_range_tensor is not None
    # Reordered batches put decodes first: the still-prefilling tail must be
    # counted as a prefill even though its query length is 1.
    assert md.num_prefills == 1
    assert md.num_decodes == 1
    assert md.prefill is not None and md.prefill.mm_wrapper is not None
    assert md.prefill.mm_prefill_ranges is not None
    # The prefill slice holds exactly the tail row: its range, inclusive.
    # The wrapper takes (N, 2) rows and rejects a non-contiguous tensor, so
    # both are part of the contract this slice has to satisfy.
    assert md.prefill.mm_prefill_ranges.shape == (1, 2)
    assert md.prefill.mm_prefill_ranges.is_contiguous()
    assert md.prefill.mm_prefill_ranges.tolist() == [[100, 200]]


def test_flashinfer_mm_prefix_restricts_kernel_block_sizes(monkeypatch):
    """Automatic page-size selection must stay off trtllm-only pages.

    On Blackwell GQA the backend normally advertises kernel block sizes up
    to 1024, which only run on trtllm-gen; an mm-prefix model must keep the
    fa2-servable [16, 32, 64] so selection never picks a page the mm path
    dies on. A non-mm model under the same conditions keeps the large pages.
    """
    from types import MethodType

    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends import flashinfer as fi
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    vllm_config, _, _ = _flashinfer_builder_env(SLIDING_WINDOW)
    mc = vllm_config.model_config
    assert mc.is_mm_prefix_lm

    # Blackwell GQA conditions that advertise large pages on non-mm models.
    monkeypatch.setattr(
        fi.current_platform,
        "is_device_capability_family",
        lambda family: family == 100,
    )
    monkeypatch.setattr(fi, "can_use_trtllm_attention", lambda q, kv: True)
    monkeypatch.setattr(
        mc, "get_num_attention_heads", MethodType(lambda self, pc: 8, mc)
    )
    monkeypatch.setattr(mc, "get_num_kv_heads", MethodType(lambda self, pc: 2, mc))

    with set_current_vllm_config(vllm_config):
        assert FlashInferBackend.get_supported_kernel_block_sizes() == [16, 32, 64]

        # Same conditions without mm-prefix: large pages stay advertised.
        mc.__dict__["is_mm_prefix_lm"] = False
        try:
            sizes = FlashInferBackend.get_supported_kernel_block_sizes()
        finally:
            mc.__dict__.pop("is_mm_prefix_lm", None)
        assert sizes == [16, 32, 64, 128, 256, 512, 1024]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA backend imports")
@pytest.mark.parametrize(
    "query_len,seq_len,spans,expected,unclamped_expected",
    [
        (16, 32, [(20, 28)], True, True),
        (16, 32, [(0, 16)], False, True),
        (16, 32, [(0, 17)], True, True),
        (16, 32, [(32, 40)], False, False),
        (16, 32, [(31, 40)], False, True),
        (1, 24, [(20, 28)], False, True),
        (0, 24, [(20, 28)], False, False),
        (16, 32, [(20, 20)], False, False),
        (16, 64, [(20, 28)], False, False),
    ],
)
def test_composite_routes_queries_that_need_image_masking(
    query_len, seq_len, spans, expected, unclamped_expected
):
    """Historical images and single-query steps must use the causal graph path."""
    from types import SimpleNamespace

    from vllm.v1.attention.backends.composite import requires_mm_prefix

    common = SimpleNamespace(
        max_query_len=query_len,
        mm_req_doc_ranges={0: spans},
        query_start_loc_cpu=torch.tensor([0, query_len]),
        seq_lens_cpu_upper_bound=torch.tensor([seq_len]),
    )
    assert requires_mm_prefix(common) is expected
    assert requires_mm_prefix(common, unclamped_window=True) is unclamped_expected


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA attention kernels")
@pytest.mark.parametrize("head_size", [256, 512])
@pytest.mark.parametrize(
    "reverse_children,cache_dtype",
    [(False, "auto"), (True, "auto"), (False, "fp8_e4m3")],
)
@pytest.mark.parametrize("causal_backend_name", ["FLASHINFER", "FLASH_ATTN"])
def test_composite_shared_cache_across_image_and_causal_steps(
    head_size, reverse_children, cache_dtype, causal_backend_name
):
    """Changing routes must preserve KV writes, image masking and graph replay."""
    if causal_backend_name == "FLASHINFER":
        if not current_platform.is_device_capability_family(100):
            pytest.skip("The Triton/FlashInfer composite supports Blackwell")
    else:
        if not current_platform.is_device_capability(90):
            pytest.skip("The Triton/FlashAttention composite supports Hopper")
        if reverse_children or cache_dtype != "auto":
            pytest.skip("Reverse-child and FP8 coverage is backend-independent")

    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.layers.attention import Attention
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
    from vllm.v1.attention.backends.composite import (
        MMPrefixAttentionRouting,
        create_composite_attention_backend,
    )
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionMetadata,
    )
    from vllm.v1.attention.backends.flashinfer import (
        FlashInferBackend,
        FlashInferMetadata,
        FlashInferTrtllmAPIDecode,
        TRTLLMPrefill,
    )
    from vllm.v1.attention.backends.triton_attn import (
        TritonAttentionBackend,
        TritonAttentionMetadata,
    )
    from vllm.v1.attention.backends.triton_flash_attn import (
        TritonFlashAttentionBackend,
    )
    from vllm.v1.attention.backends.triton_flashinfer import (
        TritonFlashInferBackend,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    if causal_backend_name == "FLASHINFER":
        backend = TritonFlashInferBackend
        causal_metadata_type = FlashInferMetadata
        block_size = backend.get_supported_kernel_block_sizes()[0]
    else:
        backend = TritonFlashAttentionBackend
        causal_metadata_type = FlashAttentionMetadata
        block_size = backend.get_preferred_block_size(16)
    num_blocks = (209 + block_size - 1) // block_size
    torch.manual_seed(42)
    attention_config = {"backend": backend.get_name()}
    if causal_backend_name == "FLASH_ATTN":
        attention_config["flash_attn_version"] = 4
    cfg = EngineArgs(
        model="google/gemma-4-31B-it",
        dtype="bfloat16",
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        block_size=block_size,
        kv_cache_dtype=cache_dtype,
        enforce_eager=True,
        attention_config=attention_config,
    ).create_engine_config()
    cfg.cache_config.kv_cache_layout = "LBHNC"
    window = 128 if head_size == 256 else None
    storage_dtype = torch.uint8 if cache_dtype == "fp8_e4m3" else DTYPE
    spec_kwargs = dict(
        block_size=block_size,
        num_kv_heads=16,
        head_size=head_size,
        dtype=storage_dtype,
        kv_quant_mode=get_kv_quant_mode(cache_dtype),
    )
    spec = (
        SlidingWindowSpec(**spec_kwargs, sliding_window=window)
        if window
        else FullAttentionSpec(**spec_kwargs)
    )
    if reverse_children:
        # Swapping the children and policy must preserve outputs and KV writes;
        # dispatch cannot depend on a particular child's class or position.
        class ReversedRouting(MMPrefixAttentionRouting):
            capture_variant = 0

            def select(self, metadata):
                return 1 - super().select(metadata)

            @staticmethod
            def variant_uses_mm_prefix(variant):
                return variant == 1

        backend = create_composite_attention_backend(
            FlashInferBackend,
            TritonAttentionBackend,
            name="ReversedCompositeBackend",
            backend_name="CUSTOM",
            module=__name__,
            routing_policy=ReversedRouting,
        )
    with set_current_vllm_config(cfg), set_default_torch_dtype(DTYPE):
        layer = Attention(
            32,
            head_size,
            head_size**-0.5,
            num_kv_heads=16,
            cache_config=cfg.cache_config,
            prefix="composite_test",
            per_layer_sliding_window=window,
            attn_backend=backend,
        ).to(DEVICE)
        layer.mm_prefix_clamp_sliding_window = window is not None
        assert backend.get_builder_cls().get_cudagraph_support(cfg, spec) == (
            AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        )
        if window:
            layer.mm_prefix_clamp_sliding_window = False
            assert backend.get_builder_cls().get_cudagraph_support(cfg, spec) == (
                AttentionCGSupport.NEVER
            )
            layer.mm_prefix_clamp_sliding_window = True
        builder = backend.get_builder_cls()(spec, ["composite_test"], cfg, DEVICE)
        builder.set_kernel_block_size(block_size)
        # Match production, where unwritten page tails are zero-initialized.
        cache = torch.zeros(
            num_blocks,
            16,
            block_size,
            2 * head_size,
            dtype=storage_dtype,
            device=DEVICE,
        )
        keys = torch.randn(209, 16, head_size, dtype=DTYPE, device=DEVICE)
        values = torch.randn_like(keys)
        if cache_dtype == "fp8_e4m3":
            keys = keys.to(torch.float8_e4m3fn).to(DTYPE)
            values = values.to(torch.float8_e4m3fn).to(DTYPE)
        table = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).unsqueeze(0)
        previous = 0
        for end, spans, metadata_type in [
            (128, [(32, 111)], TritonAttentionMetadata),
            (129, [(32, 111)], causal_metadata_type),
            (145, [(32, 111)], causal_metadata_type),
            (209, [(32, 111), (160, 207)], TritonAttentionMetadata),
        ]:
            qlen = end - previous
            starts = torch.tensor([0, qlen], dtype=torch.int32)
            lengths = torch.tensor([end], dtype=torch.int32)
            common = CommonAttentionMetadata(
                query_start_loc=starts.to(DEVICE),
                query_start_loc_cpu=starts,
                seq_lens=lengths.to(DEVICE),
                seq_lens_cpu_upper_bound=lengths,
                num_reqs=1,
                num_actual_tokens=qlen,
                max_query_len=qlen,
                max_seq_len=end,
                block_table_tensor=table,
                slot_mapping=torch.arange(previous, end, device=DEVICE),
                mm_req_doc_ranges={0: spans},
            )
            metadata = builder.build(0, common)
            assert isinstance(metadata, metadata_type)
            if isinstance(metadata, FlashInferMetadata):
                if qlen == 1:
                    assert isinstance(metadata.decode, FlashInferTrtllmAPIDecode)
                else:
                    assert isinstance(metadata.prefill, TRTLLMPrefill)
            query = torch.randn(qlen, 32, head_size, dtype=DTYPE, device=DEVICE)
            if cache_dtype == "fp8_e4m3":
                query = query.to(torch.float8_e4m3fn).to(DTYPE)
            key, value = keys[previous:end], values[previous:end]
            output = torch.empty_like(query)
            layer.impl.do_kv_cache_update(layer, key, value, cache, common.slot_mapping)
            layer.impl.forward(layer, query, key, value, cache, metadata, output)
            expected = _dense_reference(
                query,
                [keys[:end]],
                [values[:end]],
                [qlen],
                [end],
                {0: spans},
                window,
                head_size**-0.5,
                mm_clamp_sw=window or 0,
            )
            torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)
            if qlen == 1:
                metadata = builder.build_for_cudagraph_capture(common)
                assert isinstance(metadata, metadata_type)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    layer.impl.forward(
                        layer, query, key, value, cache, metadata, output
                    )
                output.zero_()
                graph.replay()
                torch.testing.assert_close(
                    output.float(), expected, atol=2e-2, rtol=2e-2
                )
            previous = end
