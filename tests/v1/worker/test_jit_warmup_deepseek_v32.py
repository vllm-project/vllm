# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the deepseek_v32 (GLM-5.2/5.3 DSA) JIT-warmup kernel owners against
runtime dispatch.

Each owner's ``compile``/``__call__`` launches a real Triton kernel, but
``dispatch`` and ``get_warmup_keys`` only compute the ``CompileKey`` compile-key
space -- no GPU work -- so these are fast CPU-side contract checks (guarded to
run only where the kernel module imports, i.e. CUDA-alike). They pin two things:

  * ``dispatch(**runtime_kwargs)`` produces the exact specialized compile key a
    runtime launch would use (golden expected keys, re-derived independently of
    ``dispatch``), and
  * ``get_warmup_keys(**register_kwargs)`` covers that runtime key with no
    extraneous keys -- the warmup-covers-runtime contract the migration relies
    on.

Companion of ``tests/v1/worker/test_jit_warmup_migration.py`` (which covers the
``ComputeSlotMappingKernel`` reference owner).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip(
        "deepseek_v32 dispatch tests require the CUDA/ROCm Triton kernel module",
        allow_module_level=True,
    )

from vllm.models.deepseek_v32.common import kernels as K

# The CuTeDSL fused-Q owner lives in a module that imports cutlass at module
# scope, so guard it: the Triton dispatch checks above stay importable on builds
# without the CuTeDSL toolchain, and only the CuTeDSL cases below are skipped.
try:
    from vllm.models.deepseek_v32.nvidia.ops import fused_q_cutedsl as FQC

    _HAS_CUTEDSL = True
except ImportError:
    _HAS_CUTEDSL = False

requires_cutedsl = pytest.mark.skipif(
    not _HAS_CUTEDSL, reason="fused_q_cutedsl requires the CuTeDSL (cutlass) toolchain"
)

# GLM-5.2 / DeepSeek-V3.2 shapes at TP8 (mirrors the correctness-test fixtures in
# tests/kernels/test_fused_deepseek_v32_norm_rope.py). All of Q_LORA, KV_LORA and
# INDEX_HEAD_DIM are exact powers of two, so triton.next_power_of_2 is the
# identity on them and the *_BLOCK_SIZE fields below equal their dims -- written
# as literals to keep the golden keys independent of dispatch()/triton.
Q_LORA = 2048
KV_LORA = 512
ROPE_DIM = 64
HALF_ROT = ROPE_DIM // 2  # 32
NUM_HEADS = 8  # num_local_heads at TP8
INDEX_HEADS = 32  # indexer.n_head
INDEX_HEAD_DIM = 128
TOPK = 2048
HIDDEN = 6144  # MTP draft hidden size
BLOCK_SIZE = 64  # kv-cache page size


# ── FusedNormRopeKernel ──────────────────────────────────────────────────────


def _norm_rope_kwargs(**overrides):
    kwargs = dict(
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE_DIM,
        index_head_dim=INDEX_HEAD_DIM,
        topk=TOPK,
        use_pcp=False,
        has_indexer=True,
        index_rope_interleave=True,
        use_pdl=True,
        mla_kv_cache_dtype="auto",
        block_size=BLOCK_SIZE,
        act_dtype=torch.bfloat16,
        cos_sin_dtype=torch.bfloat16,
        topk_dtype=torch.int32,
    )
    kwargs.update(overrides)
    return kwargs


def test_fused_norm_rope_dispatch_matches_runtime_indexer_layer() -> None:
    """Canonical indexer layer (bf16 MLA cache, no PCP): the two attn_metadata
    branches (has_cache True/False) map to two golden keys, and get_warmup_keys
    covers exactly those two."""
    kernel = K.FusedNormRopeKernel()
    warmup_kwargs = _norm_rope_kwargs()

    common = dict(
        Q_DIM=Q_LORA,
        Q_BLOCK_SIZE=Q_LORA,
        KV_DIM=KV_LORA,
        KPE_HALF_ROT_DIM=HALF_ROT,
        INDEX_K_DIM=INDEX_HEAD_DIM,
        INDEX_K_BLOCK_SIZE=INDEX_HEAD_DIM,
        INDEX_K_HALF_ROT_DIM=HALF_ROT,
        MLA_CACHE_FP8=False,
        MLA_CACHE_DS_MLA=False,
        MLA_NUM_TILES=1,
        MLA_TILE_DIM=1,
        TOPK=TOPK,
        TOPK_BLOCK_SIZE=1024,
        HAS_INDEXER=True,
        INDEX_ROPE_INTERLEAVE=True,
        USE_PDL=True,
        kv_out_present=False,
        kpe_out_present=False,
        index_k_out_present=False,
        block_size=BLOCK_SIZE,
        act_dtype=torch.bfloat16,
        cos_sin_dtype=torch.bfloat16,
        topk_dtype=torch.int32,
    )
    # has_cache=True -> MLA present: slot-mapping + indexer cache written.
    expected_cache = kernel.CompileKey(
        slot_mapping_present=True,
        indexer_cache_present=True,
        **common,
    )
    # has_cache=False -> no MLA: both drop out.
    expected_no_cache = kernel.CompileKey(
        slot_mapping_present=False,
        indexer_cache_present=False,
        **common,
    )

    assert kernel.dispatch(**warmup_kwargs, has_cache=True) == expected_cache
    assert kernel.dispatch(**warmup_kwargs, has_cache=False) == expected_no_cache
    assert set(kernel.get_warmup_keys(**warmup_kwargs)) == {
        expected_cache,
        expected_no_cache,
    }


def test_fused_norm_rope_dispatch_ds_mla_tiles() -> None:
    """fp8_ds_mla splits the kv_lora dim into 128-wide tiles when the cache is
    present; tile geometry falls back to a single 512-wide tile without it."""
    kernel = K.FusedNormRopeKernel()
    warmup_kwargs = _norm_rope_kwargs(mla_kv_cache_dtype="fp8_ds_mla")

    key_cache = kernel.dispatch(**warmup_kwargs, has_cache=True)
    key_no_cache = kernel.dispatch(**warmup_kwargs, has_cache=False)

    assert key_cache.MLA_CACHE_DS_MLA is True
    assert key_cache.MLA_CACHE_FP8 is False  # ds_mla is not the per-tensor fp8 path
    assert (key_cache.MLA_NUM_TILES, key_cache.MLA_TILE_DIM) == (KV_LORA // 128, 128)
    assert key_cache.slot_mapping_present is True
    # No cache -> tiling collapses (num_tiles=1) but the tile dim still tracks
    # ds_mla (kv_lora // 1).
    assert (key_no_cache.MLA_NUM_TILES, key_no_cache.MLA_TILE_DIM) == (1, KV_LORA)
    assert key_no_cache.slot_mapping_present is False

    assert set(kernel.get_warmup_keys(**warmup_kwargs)) == {key_cache, key_no_cache}


def test_fused_norm_rope_dispatch_fp8_per_tensor() -> None:
    """Per-tensor fp8 MLA cache sets MLA_CACHE_FP8 without ds-MLA tiling."""
    kernel = K.FusedNormRopeKernel()
    key = kernel.dispatch(**_norm_rope_kwargs(mla_kv_cache_dtype="fp8"), has_cache=True)
    assert key.MLA_CACHE_FP8 is True
    assert key.MLA_CACHE_DS_MLA is False
    assert (key.MLA_NUM_TILES, key.MLA_TILE_DIM) == (1, 1)


def test_fused_norm_rope_dispatch_pcp_collapses_cache_axis() -> None:
    """PCP materializes KV out-of-cache, so MLA/slot pointers are absent
    regardless of has_cache and the two branches dedupe to a single key with the
    cross-rank output pointers present instead."""
    kernel = K.FusedNormRopeKernel()
    warmup_kwargs = _norm_rope_kwargs(use_pcp=True)

    key_cache = kernel.dispatch(**warmup_kwargs, has_cache=True)
    key_no_cache = kernel.dispatch(**warmup_kwargs, has_cache=False)

    assert key_cache == key_no_cache
    assert key_cache.slot_mapping_present is False
    assert key_cache.indexer_cache_present is False
    assert key_cache.kv_out_present is True
    assert key_cache.kpe_out_present is True
    assert key_cache.index_k_out_present is True  # has_indexer and use_pcp

    keys = kernel.get_warmup_keys(**warmup_kwargs)
    assert keys == [key_cache]  # deduped to one


def test_fused_norm_rope_dispatch_no_indexer_layer() -> None:
    """Shared (no-indexer) layer collapses the indexer dims/flags to 1/False."""
    kernel = K.FusedNormRopeKernel()
    key = kernel.dispatch(**_norm_rope_kwargs(has_indexer=False), has_cache=True)
    assert key.HAS_INDEXER is False
    assert key.INDEX_K_DIM == 1
    assert key.INDEX_K_BLOCK_SIZE == 1
    assert key.index_k_out_present is False
    assert key.indexer_cache_present is False  # has_indexer and mla_present


# ── FusedQTritonKernel ───────────────────────────────────────────────────────


def _fused_q_kwargs(**overrides):
    kwargs = dict(
        num_q_heads=NUM_HEADS,
        qk_rope_head_dim=ROPE_DIM,
        kv_lora_rank=KV_LORA,
        index_n_head=INDEX_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        has_indexer=True,
        index_rope_interleave=False,
        quantize_mqa=True,
        use_pdl=True,
        act_dtype=torch.bfloat16,
        cos_sin_dtype=torch.bfloat16,
    )
    kwargs.update(overrides)
    return kwargs


def test_fused_q_dispatch_matches_runtime_indexer_layer() -> None:
    kernel = K.FusedQTritonKernel()
    kwargs = _fused_q_kwargs()
    expected = kernel.CompileKey(
        NUM_Q_HEADS=NUM_HEADS,
        Q_PE_HALF_ROT_DIM=HALF_ROT,
        NUM_INDEX_Q_HEADS=INDEX_HEADS,
        INDEX_Q_HALF_ROT_DIM=HALF_ROT,
        INDEX_Q_HEAD_DIM=INDEX_HEAD_DIM,
        QL_NOPE_DIM=KV_LORA,
        QL_NOPE_BLOCK=KV_LORA,  # next_power_of_2(512) == 512
        HAS_INDEXER=True,
        INDEX_ROPE_INTERLEAVE=False,
        QUANTIZE_MQA=True,
        USE_PDL=True,
        act_dtype=torch.bfloat16,
        cos_sin_dtype=torch.bfloat16,
    )
    assert kernel.dispatch(**kwargs) == expected
    assert kernel.get_warmup_keys(**kwargs) == [expected]


def test_fused_q_dispatch_no_indexer_layer() -> None:
    """No-indexer layer collapses the indexer-Q dims/count to 1."""
    kernel = K.FusedQTritonKernel()
    key = kernel.dispatch(**_fused_q_kwargs(has_indexer=False))
    assert key.HAS_INDEXER is False
    assert key.NUM_INDEX_Q_HEADS == 1
    assert key.INDEX_Q_HEAD_DIM == 1
    # The MQA path (NUM_Q_HEADS / QL_NOPE) is unaffected by has_indexer.
    assert key.NUM_Q_HEADS == NUM_HEADS
    assert key.QL_NOPE_DIM == KV_LORA


def test_fused_q_dispatch_bf16_query_skips_mqa_quant() -> None:
    """A bf16 MQA query (quantize_mqa=False) flips only QUANTIZE_MQA."""
    kernel = K.FusedQTritonKernel()
    key = kernel.dispatch(**_fused_q_kwargs(quantize_mqa=False))
    assert key.QUANTIZE_MQA is False


# ── FusedEhNormKernel (MTP) ──────────────────────────────────────────────────


def test_fused_eh_norm_dispatch_and_warmup_key() -> None:
    kernel = K.FusedEhNormKernel()
    expected = kernel.CompileKey(H=HIDDEN, BLOCK=8192)  # next_power_of_2(6144)
    assert kernel.dispatch(h=HIDDEN) == expected

    # get_warmup_keys derives H from the (duck-typed) draft model config.
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(hidden_size=HIDDEN)
            )
        )
    )
    assert kernel.get_warmup_keys(vllm_config) == [expected]


# ── FusedQCuteDSLKernel (Blackwell fp8-query path) ───────────────────────────
# The CuTeDSL owner is the runtime fused_q backend on a supported SM100 build
# (fp8 MQA query); the Triton owner above is the ROCm / unsupported-CUDA path.
# These pin the same warmup-covers-runtime contract: dispatch(**runtime_kwargs)
# reproduces the compile key of an inline launch, and get_warmup_keys(register
# kwargs) covers exactly that key. No GPU work -- only the CPU-side compile-key
# space is exercised.


def _fused_q_cutedsl_kwargs(**overrides):
    # bf16 caches/weights -> all three cute dtype axes collapse to one cute type.
    kwargs = dict(
        num_q_heads=NUM_HEADS,
        qk_rope_head_dim=ROPE_DIM,
        kv_lora_rank=KV_LORA,
        index_n_head=INDEX_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        has_indexer=True,
        index_rope_interleave=False,
        rope_cache_dtype=torch.bfloat16,
        idx_rope_cache_dtype=torch.bfloat16,
        idx_weights_dtype=torch.bfloat16,
    )
    kwargs.update(overrides)
    return kwargs


@requires_cutedsl
def test_fused_q_cutedsl_dispatch_matches_runtime_indexer_layer() -> None:
    kernel = FQC.FusedQCuteDSLKernel()
    kwargs = _fused_q_cutedsl_kwargs()
    bf16 = FQC._TORCH_TO_CUTE_DTYPE[torch.bfloat16]
    expected = kernel.CompileKey(
        rope_dim=ROPE_DIM,
        nope_dim=KV_LORA,
        num_heads=NUM_HEADS,
        rope_type=bf16,
        idx_dim=INDEX_HEAD_DIM,
        num_idx_heads=INDEX_HEADS,
        idx_rope_type=bf16,
        idx_weights_type=bf16,
        index_rope_interleave=False,
    )
    # dispatch is pure forwarding of already-cute-typed, already-collapsed axes
    # (the same axes __call__/get_warmup_keys derive via _key_args).
    assert (
        kernel.dispatch(
            rope_dim=ROPE_DIM,
            nope_dim=KV_LORA,
            num_heads=NUM_HEADS,
            rope_type=bf16,
            idx_dim=INDEX_HEAD_DIM,
            num_idx_heads=INDEX_HEADS,
            idx_rope_type=bf16,
            idx_weights_type=bf16,
            index_rope_interleave=False,
        )
        == expected
    )
    # get_warmup_keys converts runtime kwargs (torch dtypes) and covers the key.
    assert kernel.get_warmup_keys(**kwargs) == [expected]


@requires_cutedsl
def test_fused_q_cutedsl_dispatch_no_indexer_layer() -> None:
    """No-indexer layer drops both indexer dims to 0 and both indexer dtype axes
    to None (the MQA-only compiled specialization); the MQA axes are untouched."""
    kernel = FQC.FusedQCuteDSLKernel()
    keys = kernel.get_warmup_keys(**_fused_q_cutedsl_kwargs(has_indexer=False))
    assert len(keys) == 1
    key = keys[0]
    assert key.idx_dim == 0
    assert key.num_idx_heads == 0
    assert key.idx_rope_type is None
    assert key.idx_weights_type is None
    # MQA path is unaffected by has_indexer.
    assert key.num_heads == NUM_HEADS
    assert key.nope_dim == KV_LORA
    assert key.rope_type is FQC._TORCH_TO_CUTE_DTYPE[torch.bfloat16]


@requires_cutedsl
def test_fused_q_cutedsl_geometry_gate(monkeypatch) -> None:
    """The tensor-free registration gate mirrors is_fused_q_cutedsl_supported:
    it admits the fp8-query indexer layer on SM100 and rejects every runtime
    fall-back-to-Triton condition."""
    # Pretend we are on Blackwell so the capability check is not the thing under
    # test (it would be False on any non-SM100 CI box).
    monkeypatch.setattr(
        FQC.current_platform, "has_device_capability", lambda *a, **k: True
    )

    def supported(**overrides):
        base = dict(
            num_q_heads=NUM_HEADS,
            qk_rope_head_dim=ROPE_DIM,
            kv_lora_rank=KV_LORA,
            index_n_head=INDEX_HEADS,
            index_head_dim=INDEX_HEAD_DIM,
            has_indexer=True,
            quantize_mqa=True,
            act_dtype=torch.bfloat16,
        )
        base.update(overrides)
        return FQC.is_fused_q_cutedsl_geometry_supported(**base)

    assert supported() is True
    # MQA-only (no-indexer) layer still uses the CuTeDSL path.
    assert supported(has_indexer=False) is True
    # Every runtime fall-back condition rejects.
    assert supported(quantize_mqa=False) is False  # bf16 MQA -> Triton
    assert supported(act_dtype=torch.float16) is False  # non-bf16 compute
    assert supported(num_q_heads=6) is False  # not a multiple of 4
    assert supported(kv_lora_rank=576) is False  # nope dim != 512
    assert supported(qk_rope_head_dim=128) is False  # rope dim != 64
    assert supported(index_head_dim=64) is False  # indexer head dim != 128
    assert supported(index_n_head=8) is False  # indexer heads not multiple of 16
    # ...but a bad indexer geometry is irrelevant on a no-indexer layer.
    assert supported(has_indexer=False, index_head_dim=64, index_n_head=8) is True
