# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_MLA_SPARSE numerics over the deployed per-rank shapes.

Every per-rank shape of DeepSeek V3.2, GLM-5.2/5.3 and GLM-5.3-Flash at TP
1/2/4/8 runs every phase. Pre-merge runs a covering set (each shape with each
phase; KV dtype and block size rotate so each meets each phase); the nightly
``slow_test`` runs the full product.
"""

import itertools

import pytest
import torch

from tests.v1.attention.backends.rocm_aiter_mla_sparse.sparse_harness import (
    make_case,
)
from tests.v1.attention.model_shapes import SPARSE_MLA_SHAPES

RANK_SHAPES = [s.shard(tp) for s in SPARSE_MLA_SHAPES for tp in (1, 2, 4, 8)]
KV_DTYPES = ["auto", "fp8"]
BLOCK_SIZES = [1, 16, 64]

# (query_lens, seq_lens); topk is 256, so rows longer than 256 are sparse.
PHASES = {
    "decode": ([1, 1, 1, 1], [1, 255, 257, 3000]),
    "spec_uniform": ([4, 4], [256, 3000]),
    "spec_ragged": ([2, 4, 3], [5, 257, 1000]),
    "extend": ([64, 200], [500, 1200]),
    "prefill_dense": ([100, 256], [100, 256]),
    "prefill_sparse": ([700], [700]),
    "mixed": ([1, 3, 120, 400], [900, 255, 300, 400]),
}


def _covering_set():
    cases = []
    rotation = itertools.cycle(itertools.product(KV_DTYPES, BLOCK_SIZES))
    for shape in RANK_SHAPES:
        for phase in PHASES:
            kv, bs = next(rotation)
            cases.append(
                pytest.param(shape, phase, kv, bs, id=f"{shape.id}-{phase}-{kv}-bs{bs}")
            )
    return cases


def _product():
    return [
        pytest.param(
            shape,
            phase,
            kv,
            bs,
            id=f"{shape.id}-{phase}-{kv}-bs{bs}",
            marks=pytest.mark.slow_test,
        )
        for shape, phase, kv, bs in itertools.product(
            RANK_SHAPES, PHASES, KV_DTYPES, BLOCK_SIZES
        )
    ]


def _run(shape, phase, kv_cache_dtype, block_size, **kw):
    query_lens, seq_lens = PHASES[phase] if isinstance(phase, str) else phase
    case = make_case(
        shape,
        seq_lens,
        kv_cache_dtype=kv_cache_dtype,
        block_size=block_size,
        spec_tokens=3,
        **kw,
    )
    batch = case.batch(query_lens, seq_lens)
    query = case.query(batch.num_tokens)
    indices = case.indices(batch)
    out = case.run(batch, query, indices)
    case.check(batch, query, indices, out[case.names[0]])


@pytest.mark.parametrize("shape, phase, kv_cache_dtype, block_size", _covering_set())
def test_shape_phase_covering_set(env, shape, phase, kv_cache_dtype, block_size):
    _run(shape, phase, kv_cache_dtype, block_size)


@pytest.mark.parametrize("shape, phase, kv_cache_dtype, block_size", _product())
def test_shape_phase_product(env, shape, phase, kv_cache_dtype, block_size):
    _run(shape, phase, kv_cache_dtype, block_size)


@pytest.mark.parametrize("phase", ["decode", "spec_ragged", "extend", "mixed"])
def test_float16_model(env, phase):
    # fp16 with an fp8 cache is refused (see test_capabilities).
    _run(RANK_SHAPES[3], phase, "auto", 64, dtype=torch.float16)


# Tokens per request 1 / 2048 / 4096 / 8192 at batch 1 / 16 / 128 with the
# models' real topk of 2048.
@pytest.mark.parametrize(
    "batch_size, seq_len, q_len",
    [
        (1, 8192, 1),
        (16, 4096, 1),
        (128, 2048, 1),
        (1, 8192, 8192),
        (1, 4096, 2048),
        (16, 2049, 1),
    ],
    ids=lambda v: str(v),
)
@pytest.mark.parametrize("kv_cache_dtype", KV_DTYPES)
def test_real_topk_lengths(env, batch_size, seq_len, q_len, kv_cache_dtype):
    shape = SPARSE_MLA_SHAPES[0].shard(8)
    _run(
        shape,
        ([q_len] * batch_size, [seq_len] * batch_size),
        kv_cache_dtype,
        64,
        topk=2048,
        max_num_batched_tokens=max(8192, batch_size * q_len),
    )


@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_block_and_topk_boundaries(env, block_size, delta):
    """Seq lens at, one below and one above a block edge and the topk."""
    seq_lens = [block_size * 4 + delta, 256 + delta, 256 + block_size + delta]
    _run(SPARSE_MLA_SHAPES[1].shard(8), ([1, 1, 1], seq_lens), "auto", block_size)
    _run(SPARSE_MLA_SHAPES[1].shard(8), (seq_lens, seq_lens), "auto", block_size)
