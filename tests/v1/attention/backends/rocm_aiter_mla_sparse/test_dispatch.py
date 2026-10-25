# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One test per dispatch branch of ROCM_AITER_MLA_SPARSE.

Each test asserts the kernel that ran (``spy``) and checks the output against
a float32 absorbed-MLA reference over the same top-k index set.

Layer (``MLAAttention.forward_impl`` with a sparse impl):

| Condition                                              | Path                 |
| ------------------------------------------------------ | -------------------- |
| decode tokens (q_len <= 1 + spec tokens)               | ``forward_mqa``      |
| prefill, max seq <= topk, prefill backend, not forced  | ``forward_mha``      |
|   (dense MHA prefill: every visible key is in top-k)   |                      |
| prefill otherwise (seq > topk, sinks, force_mqa)       | ``forward_mqa``      |

``ROCMAiterMLASparseImpl.forward_mqa`` / ``_forward_mla``:

| Condition                                              | Kernel               |
| ------------------------------------------------------ | -------------------- |
| q tuple, bf16 with rope                                | ``concat_mla_q``     |
| layer owns the indexer, or a new top-k epoch           | ``fit_kpool`` + remap|
| shared layer, same epoch and buffer                    | no remap             |
| top-k buffer wider than topk (kpool tail)              | fit keeps live tail  |
| ``use_aiter_sparse_mla`` (env on gfx950, or NoPE fp8)  | triton_sparse_mla_fwd|
| bf16 NoPE (head == kv_lora_rank) or fp16 q, no fp8 KV:|                      |
|   decode-only, splits > 1                              | decode_bf16 (Triton) |
|   otherwise                                            | sparse_attn_prefill  |
| sinks with fp16 / >128 bf16 heads / gfx942 33-64 heads | Triton with sinks    |
| gfx942, no sinks, long pure-bf16 prefill               | sparse_attn_prefill  |
| sinks (bf16 or fp8 head buckets)                       | mla_decode_fwd_lse   |
| default                                                | mla_decode_fwd       |
|   + persistent metadata, except a pure bf16 prefill    | get_mla_metadata_v1  |
|     with rows longer than topk (non-persistent ASM)    |                      |
"""

from dataclasses import replace

import pytest
import torch

from tests.v1.attention.backends.mla_harness import sparse_topk_indices
from tests.v1.attention.backends.rocm_aiter_mla_sparse.sparse_harness import (
    AITER_FP8_LSE_STALE_ROWS,
    AITER_TRITON,
    CONCAT_Q,
    DENSE_PREFILL,
    DEVICE,
    FIT_KPOOL,
    MLA_DECODE,
    MLA_DECODE_LSE,
    MLA_METADATA,
    REMAP,
    SPARSE_MQA,
    TRITON_DECODE,
    TRITON_PREFILL,
    make_case,
)
from tests.v1.attention.model_shapes import (
    DEEPSEEK_V3_2,
    GLM_5_2,
    GLM_5_3_FLASH,
)
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms.rocm import on_gfx950
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    rocm_sparse_decode_bf16_num_splits,
)

DSV32 = DEEPSEEK_V3_2.shard(8)
GLM = GLM_5_2.shard(8)
GLM_FLASH = GLM_5_3_FLASH.shard(8)

# (query_lens, seq_lens) per phase; topk is 256 unless a test says otherwise.
DECODE = ([1, 1, 1], [5, 300, 700])
SPEC = ([4, 4, 4], [8, 300, 700])
SPEC_RAGGED = ([2, 4, 3], [8, 300, 700])
SHORT_PREFILL = ([40, 100], [40, 200])  # every row fits in topk
LONG_PREFILL = ([300, 500], [300, 600])  # rows exceed topk
MIXED = ([1, 1, 64, 300], [9, 700, 400, 300])


def _watch_layer(spy, case):
    for layer in case.layers.values():
        spy.watch(layer.impl, "forward_mha", DENSE_PREFILL)
        spy.watch(layer.impl, "forward_mqa", SPARSE_MQA)


def _run(case, spy, phase, *, dense_indices=False):
    query_lens, seq_lens = phase
    batch = case.batch(query_lens, seq_lens)
    query = case.query(batch.num_tokens)
    indices = case.indices(batch, dense=dense_indices)
    _watch_layer(spy, case)
    out = case.run(batch, query, indices)
    for name, layer in case.layers.items():
        case.check(batch, query, indices, out[name], layer=layer)
    return batch


@pytest.mark.parametrize("phase", [DECODE, SPEC, MIXED, LONG_PREFILL])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_default_path_is_asm_decode(env, spy, phase, kv_cache_dtype):
    case = make_case(DSV32, phase[1], kv_cache_dtype=kv_cache_dtype, spec_tokens=3)
    _run(case, spy, phase)
    names = spy.names()
    assert MLA_DECODE in names and SPARSE_MQA in names
    assert TRITON_PREFILL not in names and TRITON_DECODE not in names
    # bf16 concatenates [ql_nope; q_pe]; fp8 concatenates and quantizes in
    # one fused op on the layer.
    assert (CONCAT_Q in names) == (kv_cache_dtype == "auto")
    (kwargs,) = spy.kwargs_of(MLA_DECODE)
    persistent = not (phase is LONG_PREFILL and kv_cache_dtype == "auto")
    assert (MLA_METADATA in names) == persistent
    assert ("work_meta_data" in kwargs) == persistent


def test_short_prefill_takes_dense_mha(env, spy):
    case = make_case(DSV32, SHORT_PREFILL[1])
    _run(case, spy, SHORT_PREFILL, dense_indices=True)
    assert spy.names().count(DENSE_PREFILL) == 1
    assert SPARSE_MQA not in spy.names()


def test_mixed_batch_splits_dense_prefill_and_sparse_decode(env, spy):
    phase = ([1, 1, 40], [300, 700, 100])
    case = make_case(DSV32, phase[1])
    batch = case.batch(*phase)
    query = case.query(batch.num_tokens)
    # Decode rows sparse, the short prefill row dense (as the indexer gives).
    indices = case.indices(batch)
    _watch_layer(spy, case)
    out = case.run(batch, query, indices)
    case.check(batch, query, indices, out[case.names[0]])
    assert spy.names().count(DENSE_PREFILL) == 1
    assert spy.names().count(SPARSE_MQA) == 1
    (mqa_args,) = spy.args_of(SPARSE_MQA)
    assert mqa_args[0][0].shape[0] == 2  # only the two decode tokens


def test_force_mqa_routes_short_prefill_sparse(env, spy):
    case = make_case(DSV32, SHORT_PREFILL[1], force_mqa=True)
    _run(case, spy, SHORT_PREFILL)
    assert DENSE_PREFILL not in spy.names()
    assert spy.names().count(SPARSE_MQA) == 1


@pytest.mark.parametrize("phase", [DECODE, SPEC_RAGGED, MIXED, LONG_PREFILL])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_nope_glm_flash(env, spy, phase, kv_cache_dtype):
    """GLM-5.3-Flash has no rope dims: bf16 runs vLLM's Triton kernels, fp8
    runs AITER's Triton sparse MLA (the only kernel with a NoPE fp8 path)."""
    case = make_case(GLM_FLASH, phase[1], kv_cache_dtype=kv_cache_dtype, spec_tokens=3)
    _run(case, spy, phase)
    names = spy.names()
    assert MLA_DECODE not in names and MLA_METADATA not in names
    assert CONCAT_Q not in names  # rope width 0: a plain copy
    if kv_cache_dtype == "fp8":
        assert case.layer.impl.use_aiter_sparse_mla
        assert AITER_TRITON in names
        return
    num_tokens = sum(phase[0])
    decode_only = all(q <= 4 for q in phase[0])
    splits = rocm_sparse_decode_bf16_num_splits(
        num_tokens, 16, min(max(phase[1]), case.topk)
    )
    expect = TRITON_DECODE if decode_only and splits > 1 else TRITON_PREFILL
    assert expect in names, names


@pytest.mark.parametrize("phase", [DECODE, MIXED])
def test_fp16_runs_triton(env, spy, phase):
    """AITER's ASM decode has no fp16 kernel."""
    case = make_case(DSV32, phase[1], dtype=torch.float16)
    _run(case, spy, phase)
    names = spy.names()
    assert MLA_DECODE not in names and MLA_METADATA not in names
    assert TRITON_DECODE in names or TRITON_PREFILL in names


def test_nope_bf16_single_decode_split_choice(env, spy):
    """A decode-only batch that fits one split still uses the prefill kernel."""
    phase = ([1], [8])
    case = make_case(GLM_FLASH, phase[1])
    _run(case, spy, phase)
    assert TRITON_PREFILL in spy.names()


@pytest.mark.skipif(not on_gfx950(), reason="AITER Triton sparse MLA is gfx950")
@pytest.mark.parametrize("phase", [DECODE, MIXED, LONG_PREFILL])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("shape", [DSV32, GLM], ids=lambda s: s.id)
def test_aiter_triton_sparse_mla_env(
    env, spy, monkeypatch, phase, kv_cache_dtype, shape
):
    monkeypatch.setattr(rocm_aiter_ops, "_TRITON_SPARSE_MLA", True)
    case = make_case(shape, phase[1], kv_cache_dtype=kv_cache_dtype)
    assert case.layer.impl.use_aiter_sparse_mla
    _run(case, spy, phase)
    names = spy.names()
    assert AITER_TRITON in names
    assert MLA_DECODE not in names and MLA_METADATA not in names


def test_aiter_triton_env_rejects_fp16_model(env, monkeypatch):
    monkeypatch.setattr(rocm_aiter_ops, "_TRITON_SPARSE_MLA", True)
    case = make_case(DSV32, DECODE[1], dtype=torch.float16)
    assert not case.layer.impl.use_aiter_sparse_mla


@pytest.mark.parametrize("phase", [DECODE, MIXED, LONG_PREFILL])
@pytest.mark.parametrize(
    "kv_cache_dtype", ["auto", pytest.param("fp8", marks=AITER_FP8_LSE_STALE_ROWS)]
)
def test_sinks_take_lse_decode(env, spy, phase, kv_cache_dtype):
    case = make_case(DSV32, phase[1], kv_cache_dtype=kv_cache_dtype, use_sinks=True)
    # A sink layer has no dense-MHA prefill backend: all tokens are sparse.
    assert case.layer.prefill_backend is None
    _run(case, spy, phase)
    names = spy.names()
    assert MLA_DECODE_LSE in names and MLA_DECODE not in names
    assert DENSE_PREFILL not in names
    assert MLA_METADATA not in names


def test_sinks_with_fp16_fall_back_to_triton(env, spy):
    case = make_case(DSV32, DECODE[1], dtype=torch.float16, use_sinks=True)
    _run(case, spy, DECODE)
    names = spy.names()
    assert MLA_DECODE_LSE not in names
    assert TRITON_DECODE in names or TRITON_PREFILL in names


def test_sinks_with_unsupported_head_bucket_fall_back_to_triton(env, spy):
    """bf16 sinks above 128 padded heads have no AITER LSE kernel (synthetic
    256-head shape: no deployed layout has more than 128 heads per rank)."""
    shape = replace(DEEPSEEK_V3_2, num_heads=256)
    case = make_case(shape, DECODE[1], use_sinks=True)
    _run(case, spy, DECODE)
    assert MLA_DECODE_LSE not in spy.names()


def test_gfx942_long_bf16_prefill_uses_triton(env, spy, monkeypatch):
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx942", lambda: True)
    case = make_case(DSV32, LONG_PREFILL[1])
    _run(case, spy, LONG_PREFILL)
    names = spy.names()
    assert TRITON_PREFILL in names
    assert MLA_DECODE not in names and MLA_METADATA not in names


def test_every_owner_forward_remaps(env, spy):
    case = make_case(DSV32, DECODE[1])
    batch = case.batch(*DECODE)
    query = case.query(batch.num_tokens)
    indices = case.indices(batch)
    metadata = case.build(batch)
    case.run(batch, query, indices, metadata)
    case.run(batch, query, indices, metadata)
    assert spy.names().count(REMAP) == 2


def test_shared_layers_remap_once_per_indexer_write(env, spy):
    """GLM-5.2's "shared" layers reuse the last indexer layer's top-k."""
    case = make_case(GLM, MIXED[1], num_layers=4, force_mqa=True)
    _run(case, spy, MIXED)
    # Layer 0 owns the indexer and remaps; layers 1-3 reuse its remap.
    assert spy.names().count(REMAP) == 1
    assert spy.names().count(SPARSE_MQA) == 4


def test_shared_layer_remaps_after_a_new_indexer_write(env, spy):
    case = make_case(GLM, DECODE[1], num_layers=2)
    batch = case.batch(*DECODE)
    query = case.query(batch.num_tokens)
    metadata = case.build(batch)
    first = case.indices(batch)
    case.run(batch, query, first, metadata)
    # A new metadata object starts a new epoch even with the same buffer.
    second = sparse_topk_indices(batch, case.topk, seed=1, device=DEVICE)
    out = case.run(batch, query, second, case.build(batch))
    shared = case.names[1]
    case.check(batch, query, second, out[shared], layer=case.layers[shared])
    assert spy.names().count(REMAP) == 2


def test_wide_kpool_buffer_keeps_live_tail(env, spy):
    """A kpool indexer writes topk history columns plus a tail of live rows."""
    case = make_case(GLM_FLASH, LONG_PREFILL[1], topk_width=256 + 64)
    batch = case.batch(*LONG_PREFILL)
    query = case.query(batch.num_tokens)
    indices = case.indices(batch)
    # Move the last 32 history entries of every full row into the tail.
    full = (indices[:, :256] >= 0).sum(-1) == 256
    indices[full, 256:288] = indices[full, 224:256]
    indices[full, 224:256] = -1
    _watch_layer(spy, case)
    out = case.run(batch, query, indices)
    case.check(batch, query, indices, out[case.names[0]])
    (fit_args,) = spy.args_of(FIT_KPOOL)
    assert fit_args[0].shape[1] == 256 + 64
