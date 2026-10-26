# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_FA output vs a float32 reference over flagship model shapes.

Dimensions: model shape (head size, H/KV, TP-sharded heads, sliding window)
x phase (prefill / extend / decode / mixed) x tokens per request (1, 2048,
4096, 8192) x requests per batch (1, 16, 128) x block size (16, 32) x KV
dtype (auto, fp8). Pre-merge runs a covering set where every value of each
dimension meets every phase; ``-m slow_test`` runs the full product.

Prefill and extend batches are capped at ``TOKEN_BUDGET`` new tokens, as the
scheduler's max_num_batched_tokens would cap them; requests beyond the budget
become decodes over the same context, which is what a running batch holds.
"""

import itertools

import pytest
import torch

from tests.v1.attention.backends.rocm_aiter_fa.fa_harness import FaCase, blocks_for
from tests.v1.attention.model_shapes import (
    GEMMA4_31B_SLIDING,
    MUSE_SPARK_SLIDING,
    AttnShape,
    dense_rank_shapes,
    shapes_supported_by,
)

SUPPORTED_HEAD_SIZES = [64, 128, 256]
SHAPES = shapes_supported_by(SUPPORTED_HEAD_SIZES)
PHASES = ["prefill", "extend", "decode", "mixed"]
TOKENS = [1, 2048, 4096, 8192]
BATCHES = [1, 16, 128]
BLOCK_SIZES = [16, 32]
KV_DTYPES = ["auto", "fp8"]
TOKEN_BUDGET = 16384
EXTEND_CHUNK = 512


def make_lens(phase: str, tokens: int, batch: int) -> tuple[list[int], list[int]]:
    """(query_lens, seq_lens) in reorder order: decodes, extends, prefills."""
    if phase == "decode":
        return [1] * batch, [tokens] * batch
    if phase == "prefill":
        n = max(1, min(batch, TOKEN_BUDGET // tokens))
        return [1] * (batch - n) + [tokens] * n, [tokens] * batch
    if phase == "extend":
        chunk = max(1, min(EXTEND_CHUNK, tokens - 1, TOKEN_BUDGET // batch))
        if tokens == 1:
            return [1] * batch, [1] * batch
        return [chunk] * batch, [tokens] * batch
    # mixed: half decodes, a quarter extends, the rest prefills (>= 1 each).
    n_dec = max(1, batch // 2) if batch > 1 else 0
    n_ext = max(1, batch // 4) if batch > 2 else 0
    n_pre = batch - n_dec - n_ext
    pre_len = max(1, min(tokens, TOKEN_BUDGET // (2 * max(n_pre, 1))))
    ext_q = max(1, min(EXTEND_CHUNK, tokens - 1)) if tokens > 1 else 1
    q = [1] * n_dec + [ext_q] * n_ext + [pre_len] * n_pre
    s = [tokens] * n_dec + [tokens] * n_ext + [pre_len] * n_pre
    if tokens == 1:
        s = [max(x, 2) if i < n_dec else x for i, x in enumerate(s)]
    return q, s


def _run(
    shape: AttnShape,
    phase,
    tokens,
    batch,
    block_size,
    kv_cache_dtype,
    dtype=torch.bfloat16,
):
    query_lens, seq_lens = make_lens(phase, tokens, batch)
    case = FaCase(
        shape,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        block_size=block_size,
        sliding_window=shape.sliding_window,
        max_model_len=max(16384, max(seq_lens)),
        max_num_batched_tokens=max(TOKEN_BUDGET, sum(query_lens)),
    ).setup(blocks_for(seq_lens, block_size))
    batch_ = case.batch(query_lens, seq_lens)
    query = case.query(batch_.num_tokens)
    case.check(batch_, query, case.run(batch_, query))


def _covering_set():
    """Each (tokens, batch) pair with each phase once; the shape, block size
    and KV dtype rotate so each of them appears across the set."""
    cases = []
    for i, (phase, tokens, batch) in enumerate(
        itertools.product(PHASES, TOKENS, BATCHES)
    ):
        shape = SHAPES[i % len(SHAPES)]
        cases.append(
            pytest.param(
                shape,
                phase,
                tokens,
                batch,
                BLOCK_SIZES[i % 2],
                KV_DTYPES[(i // 2) % 2],
                id=f"{shape.id}-{phase}-t{tokens}-b{batch}-bs{BLOCK_SIZES[i % 2]}"
                f"-{KV_DTYPES[(i // 2) % 2]}",
            )
        )
    return cases


@pytest.mark.parametrize(
    "shape, phase, tokens, batch, block_size, kv_cache_dtype", _covering_set()
)
def test_covering_set(shape, phase, tokens, batch, block_size, kv_cache_dtype):
    _run(shape, phase, tokens, batch, block_size, kv_cache_dtype)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.id)
@pytest.mark.parametrize("kv_cache_dtype", KV_DTYPES)
def test_every_model_shape_mixed_batch(shape, kv_cache_dtype):
    _run(shape, "mixed", 2048, 16, 16, kv_cache_dtype)


@pytest.mark.parametrize(
    "shape",
    [s for s in dense_rank_shapes() if s.head_size in SUPPORTED_HEAD_SIZES],
    ids=lambda s: s.id,
)
@pytest.mark.parametrize("kv_cache_dtype", KV_DTYPES)
def test_every_deployed_rank_shape(shape, kv_cache_dtype):
    """Per-rank shapes of every valid TP/DP/EP/PP layout on the maintainers'
    GPU counts (model_shapes.PARALLEL_LAYOUTS)."""
    _run(shape, "mixed", 2048, 16, 16, kv_cache_dtype)


@pytest.mark.parametrize("phase", PHASES)
def test_float16_model(phase):
    _run(SHAPES[0], phase, 2048, 16, 16, "auto", dtype=torch.float16)


@pytest.mark.parametrize(
    "seq_len",
    # Around block (16/32), partition (256) and window (1024 / 2048) edges.
    [15, 16, 17, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049],
)
@pytest.mark.parametrize("phase", ["decode", "extend"])
@pytest.mark.parametrize(
    "shape",
    [GEMMA4_31B_SLIDING, MUSE_SPARK_SLIDING],
    ids=lambda s: s.id,
)
def test_boundaries(shape, phase, seq_len):
    batch = 3
    if phase == "decode":
        query_lens, seq_lens = [1] * batch, [seq_len - 1, seq_len, seq_len + 1]
    else:
        q = min(7, seq_len - 1)
        query_lens, seq_lens = [q] * batch, [seq_len - 1, seq_len, seq_len + 1]
    case = FaCase(shape, sliding_window=shape.sliding_window).setup(
        blocks_for(seq_lens, 16)
    )
    batch_ = case.batch(query_lens, seq_lens)
    query = case.query(batch_.num_tokens)
    case.check(batch_, query, case.run(batch_, query))


@pytest.mark.slow_test
@pytest.mark.parametrize("kv_cache_dtype", KV_DTYPES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.id)
def test_full_matrix(shape, phase, tokens, batch, block_size, kv_cache_dtype):
    _run(shape, phase, tokens, batch, block_size, kv_cache_dtype)
