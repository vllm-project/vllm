# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("the MiniMax-M3 MSA indexer is AMD-only", allow_module_level=True)

from vllm.models.minimax_m3.amd.indexer_msa import (  # noqa: E402
    MAX_SUPPORTED_BLOCKS,
    MSA_INDEX_HEAD_DIM,
    MSA_SCORE_TYPE,
    MSA_SPARSE_BLOCK_SIZE,
    MSA_TOPK_BLOCKS,
    SLOTS_MAX,
    WAVE_SIZE,
)

# No head count: the gate takes one only to describe a CP shape, and the
# default cp_world of 1 is the unsharded indexer, which does not turn on one.
_ADMITTED = dict(
    topk_blocks=MSA_TOPK_BLOCKS,
    sparse_block_size=MSA_SPARSE_BLOCK_SIZE,
    index_head_dim=MSA_INDEX_HEAD_DIM,
    indexer_kv_dtype="fp8",
    max_model_len=8192,
    score_type=MSA_SCORE_TYPE,
)

# The same config sharded, at MiniMax-M3's own shape: 4 index heads over 4
# ranks is one apiece, and 4 x top-16 fills the merge exactly.
_ADMITTED_CP = dict(
    _ADMITTED,
    cp_world=4,
    num_index_heads=4,
    num_kv_heads=1,
    max_query_len=1,
    dcp_size=1,
)

# The longest context the top-k has register slots for.
_MAX_LEN = MAX_SUPPORTED_BLOCKS * MSA_SPARSE_BLOCK_SIZE


@pytest.fixture
def aiter_msa_indexer_gate(monkeypatch):
    import vllm.models.minimax_m3.amd.indexer_msa as indexer_msa_mod
    import vllm.platforms.rocm as rocm_mod

    def probe(rocm=True, gfx950=True, aiter_attend=True, base=_ADMITTED, **overrides):
        monkeypatch.setattr(indexer_msa_mod.current_platform, "is_rocm", lambda: rocm)
        monkeypatch.setattr(
            indexer_msa_mod,
            "_minimax_m3_aiter_sparse_pa_requested",
            lambda: aiter_attend,
        )
        monkeypatch.setattr(rocm_mod, "on_gfx950", lambda: gfx950)
        return indexer_msa_mod.msa_indexer_unsupported_reason(**{**base, **overrides})

    return probe


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({}, None),
        ({"rocm": False}, "ROCm"),
        ({"gfx950": False}, "gfx950"),
        ({"aiter_attend": False}, "AITER sparse PA attend"),
        ({"indexer_kv_dtype": "bf16"}, "fp8 e4m3 index cache"),
        ({"score_type": "sum"}, "score_type"),
        ({"topk_blocks": MSA_TOPK_BLOCKS * 2}, "topk_blocks"),
        ({"sparse_block_size": MSA_SPARSE_BLOCK_SIZE // 2}, "sparse_block_size"),
        ({"index_head_dim": MSA_INDEX_HEAD_DIM // 2}, "index_head_dim"),
        ({"max_model_len": _MAX_LEN}, None),
        ({"max_model_len": _MAX_LEN + 1}, "the top-k is compiled for"),
    ],
)
def test_msa_indexer_gate_rejects_unsupported_configs(
    aiter_msa_indexer_gate, override, expected
):
    reason = aiter_msa_indexer_gate(**override)
    if expected is None:
        assert reason is None, f"{override} should be admitted, got {reason!r}"
    else:
        assert reason is not None, f"{override} should be refused"
        assert expected in reason, f"{override} refused for the wrong reason: {reason}"


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({}, None),
        # Two ranks owning two index heads each, matching the model's kv split.
        ({"cp_world": 2, "num_kv_heads": 2}, None),
        # Sharding does not excuse the unsharded contract: the same checks run
        # first, so one gate cannot admit at one width what it refuses at the
        # other.
        ({"indexer_kv_dtype": "bf16"}, "fp8 e4m3 index cache"),
        ({"gfx950": False}, "gfx950"),
        ({"max_model_len": _MAX_LEN + 1}, "the top-k is compiled for"),
        # And then the conditions only sharding runs into.
        ({"dcp_size": 2}, "decode_context_parallel_size"),
        ({"cp_world": 3}, "divisible"),
        ({"cp_world": 8, "num_index_heads": 8}, "candidates the merge holds"),
        ({"max_query_len": 5}, "MFMA columns"),
        ({"num_kv_heads": 2}, "1:1"),
    ],
)
def test_msa_indexer_gate_rejects_unshardable_configs(
    aiter_msa_indexer_gate, override, expected
):
    reason = aiter_msa_indexer_gate(base=_ADMITTED_CP, **override)
    if expected is None:
        assert reason is None, f"{override} should be admitted, got {reason!r}"
    else:
        assert reason is not None, f"{override} should be refused"
        assert expected in reason, f"{override} refused for the wrong reason: {reason}"


def test_msa_indexer_cp_limits_are_aiters(aiter_msa_indexer_gate):
    # Both CP bounds belong to the kernels, so the gate reads them from AITER
    # rather than restating them. Pinned from the outside here -- admitted at
    # the limit and refused one past it -- so that re-hardcoding either would
    # fail as soon as the kernels moved.
    msa_block_select = pytest.importorskip("aiter.ops.msa_block_select")

    # Every shard's whole top-k has to fit the merge's wave at once.
    world = msa_block_select.TOPK_CP_MAX_CAND // MSA_TOPK_BLOCKS
    at_cap = dict(cp_world=world, num_index_heads=world)
    assert aiter_msa_indexer_gate(base=_ADMITTED_CP, **at_cap) is None
    over_cap = aiter_msa_indexer_gate(
        base=_ADMITTED_CP, cp_world=world + 1, num_index_heads=world + 1
    )
    assert over_cap is not None and "candidates the merge holds" in over_cap

    # One (query token, index head) pair per column of the score pass's MFMA.
    heads = _ADMITTED_CP["num_index_heads"]
    query_len = msa_block_select.SCORE_MFMA_COLS // heads
    assert aiter_msa_indexer_gate(base=_ADMITTED_CP, max_query_len=query_len) is None
    over_cols = aiter_msa_indexer_gate(base=_ADMITTED_CP, max_query_len=query_len + 1)
    assert over_cols is not None and "MFMA columns" in over_cols


def test_msa_slot_cap_matches_what_aiter_compiled():
    msa_block_select = pytest.importorskip("aiter.ops.msa_block_select")

    assert SLOTS_MAX == msa_block_select.TOPK_MAX_SLOTS
    assert WAVE_SIZE == msa_block_select.WAVE_SIZE

    def aiter_accepts(blocks: int) -> bool:
        strips = -(-blocks // WAVE_SIZE)
        return 1 << (strips - 1).bit_length() <= SLOTS_MAX

    assert aiter_accepts(MAX_SUPPORTED_BLOCKS)
    assert not aiter_accepts(MAX_SUPPORTED_BLOCKS + 1)
