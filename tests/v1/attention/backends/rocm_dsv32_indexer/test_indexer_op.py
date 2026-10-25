# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm DSA indexer op (``SparseAttnIndexer.forward_hip`` ->
``rocm_aiter_sparse_attn_indexer``) for DeepSeek V3.2 and GLM-5.x.

Every test checks the top-k rows against float32 scores and the output
contract ROCM_AITER_MLA_SPARSE relies on (the first ``min(pos + 1, topk)``
entries valid, the rest -1), and asserts the kernels that ran.

| Condition                                        | Kernels                    |
| ------------------------------------------------ | -------------------------- |
| every step (unless the model writes K itself)    | K quant + cache write      |
| prefill chunk                                    | K gather, fp8 MQA logits   |
|   aiter top-k supported for the chunk            | aiter prefill top-k        |
|   otherwise                                      | ``_C.top_k_per_row_prefill``|
| decode (uniform next_n, or flattened)            | fp8 paged MQA logits       |
|   (the pack / unpack branch for non-uniform      |                            |
|   native rows is unreachable: the native split   |                            |
|   requires uniform decode rows)                  |                            |
|   aiter top-k supported and not DSV4-preferred   | aiter decode top-k         |
|   otherwise                                      | ``_C.top_k_per_row_decode`` |
"""

import pytest

from tests.v1.attention.backends.indexer_harness import check_topk_rows
from tests.v1.attention.backends.rocm_dsv32_indexer.idx_harness import (
    AITER_TOPK_DECODE,
    AITER_TOPK_PREFILL,
    DECODE_LOGITS,
    K_GATHER,
    K_WRITE,
    PACK,
    PREFILL_LOGITS,
    TOPK_DECODE,
    TOPK_PREFILL,
    make_idx_case,
)
from tests.v1.attention.model_shapes import DEEPSEEK_V3_2, GLM_5_2
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops

SHAPES = [DEEPSEEK_V3_2, GLM_5_2]

PHASES = {
    "decode": ([1, 1, 1, 1], [1, 255, 257, 3000]),
    "prefill_short": ([100, 7], [100, 7]),
    "prefill_long": ([700, 300], [700, 300]),
    "extend": ([64, 200], [500, 1200]),
    "mixed": ([1, 1, 120, 400], [900, 255, 300, 400]),
}


def _check(case, phase, spy=None):
    query_lens, seq_lens = phase
    batch = case.batch(query_lens, seq_lens)
    q_fp8, weights = case.inputs(batch.num_tokens)
    topk = case.run(batch, q_fp8, weights)
    check_topk_rows(topk, case.logits(batch, q_fp8, weights), case.topk)
    return batch


@pytest.mark.parametrize("phase", list(PHASES))
@pytest.mark.parametrize("block_size", [1, 16, 64])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.model)
def test_topk_matches_reference(env, spy, shape, block_size, phase):
    query_lens, seq_lens = PHASES[phase]
    case = make_idx_case(shape, seq_lens, block_size=block_size)
    _check(case, PHASES[phase])
    names = spy.names()
    assert K_WRITE in names
    has_prefill = any(q > 1 for q in query_lens)
    has_decode = any(q == 1 for q in query_lens)
    assert (PREFILL_LOGITS in names) == has_prefill
    assert (K_GATHER in names) == has_prefill
    assert (DECODE_LOGITS in names) == has_decode
    if has_prefill:
        assert AITER_TOPK_PREFILL in names or TOPK_PREFILL in names
    if has_decode:
        assert AITER_TOPK_DECODE in names or TOPK_DECODE in names


@pytest.mark.parametrize("topk", [2048])
@pytest.mark.parametrize(
    "phase",
    [([1] * 16, [4096] * 16), ([1], [8192]), ([4096], [4096]), ([2048], [6000])],
    ids=["decode-bs16-4k", "decode-8k", "prefill-4k", "extend-6k"],
)
def test_real_topk(env, phase, topk):
    case = make_idx_case(DEEPSEEK_V3_2, phase[1], topk=topk)
    _check(case, phase)


@pytest.mark.parametrize("supported", [True, False])
def test_topk_kernel_choice(env, spy, monkeypatch, supported):
    """Aiter top-k when supported for the shape, vLLM's own kernel otherwise."""
    monkeypatch.setattr(
        rocm_aiter_ops, "is_indexer_top_k_supported", lambda **kw: supported
    )
    phase = PHASES["mixed"]
    case = make_idx_case(DEEPSEEK_V3_2, phase[1])
    _check(case, phase)
    names = spy.names()
    if supported:
        assert AITER_TOPK_PREFILL in names and TOPK_PREFILL not in names
    else:
        assert TOPK_PREFILL in names and AITER_TOPK_PREFILL not in names
        assert TOPK_DECODE in names and AITER_TOPK_DECODE not in names


@pytest.mark.parametrize(
    "spec_tokens, query_lens, num_decodes",
    [
        (1, [2, 2, 2], 3),  # native next_n 2
        # Native decode needs uniform rows: the split stops at the first
        # differing row and the rest run as prefills.
        (1, [2, 1, 2], 1),
        (3, [4, 4, 4], 3),  # flattened per token
        (3, [4, 2, 3], 3),  # flattened, ragged
    ],
    ids=["native", "native-ragged", "flat", "flat-ragged"],
)
def test_spec_decode_rows(env, spy, spec_tokens, query_lens, num_decodes):
    seq_lens = [300, 9, 2000]
    case = make_idx_case(DEEPSEEK_V3_2, seq_lens, spec_tokens=spec_tokens)
    md = case.build(case.batch(query_lens, seq_lens))
    assert md.num_decodes == num_decodes
    _check(case, (query_lens, seq_lens))
    names = spy.names()
    assert DECODE_LOGITS in names
    assert (PREFILL_LOGITS in names) == (num_decodes < len(query_lens))
    # Uniform native rows reshape; flattened rows are one token each.
    assert PACK not in names


def test_prefill_is_chunked_under_the_logits_budget(env, spy, monkeypatch):
    """A tiny logits budget splits prefills into several chunks, and one
    long request into query sub-chunks; the result must not change."""
    monkeypatch.setattr(envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 1)
    phase = ([700, 300, 1000], [700, 300, 1000])
    case = make_idx_case(DEEPSEEK_V3_2, phase[1])
    md = case.build(case.batch(*phase))
    assert len(md.prefill.chunks) > 2
    assert any(c.skip_kv_gather for c in md.prefill.chunks)
    _check(case, phase)
    assert spy.names().count(PREFILL_LOGITS) == len(md.prefill.chunks)
    # Query sub-chunks reuse their request group's gathered K.
    gathers = sum(not c.skip_kv_gather for c in md.prefill.chunks)
    assert spy.names().count(K_GATHER) == gathers


def test_padded_tokens_are_not_selected(env):
    """Graph-padded tokens (slot -1) never write the cache."""
    phase = ([1, 1], [300, 900])
    case = make_idx_case(DEEPSEEK_V3_2, phase[1])
    batch = case.batch(*phase, num_padded_tokens=8)
    q_fp8, weights = case.inputs(8)
    before = case.k_cache.kv_cache.clone()
    topk = case.run(batch, q_fp8, weights)
    check_topk_rows(topk, case.logits(batch, q_fp8, weights), case.topk)
    changed = (case.k_cache.kv_cache != before).flatten(1).any(1).sum().item()
    assert changed <= 2  # at most the blocks of the two real tokens
