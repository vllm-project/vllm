# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_FLASHMLA_SPARSE_DSV41 capability contract.

DeepSeek V4.1's attention layer picks its backend class itself
(``DeepseekV41ROCMAiterMLAAttention.backend_cls``), so the generic selector
never vets the model's own layers; the declarations still matter to the
selector when a user names the backend, to the KV-cache manager (block sizes)
and to the runner (CUDA graph support). Each one is checked both ways: what
is declared passes, what is not is refused with a reason.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.platforms.rocm import _get_invalid_reasons
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig

rocm = pytest.importorskip("vllm.models.deepseek_v41.amd.rocm")

BACKEND = rocm.DeepseekV41ROCMAiterMLASparseBackend
SWA_BACKEND = rocm.DeepseekV41ROCMAiterSparseSWABackend
MLA_BUILDER = rocm.DeepseekV41ROCMAiterMLASparseMetadataBuilder
SWA_BUILDER = rocm.DeepseekV41ROCMAiterSparseSWAMetadataBuilder
# MI300X/MI325X (gfx942) and MI350X/MI355X (gfx950).
GFX942, GFX950 = DeviceCapability(9, 4), DeviceCapability(9, 5)


def _reasons(capability=GFX950, **overrides) -> list[str]:
    config = AttentionSelectorConfig(
        **{
            "head_size": 512,
            "dtype": torch.bfloat16,
            "kv_cache_dtype": "fp8_ds_mla",
            "block_size": 128,
            "use_mla": True,
            "has_sink": True,
            "use_sparse": True,
            **overrides,
        }
    )
    return _get_invalid_reasons(BACKEND, capability, config, is_turboquant_run=False)


def test_declarations():
    """Pin the declared surface; changing it must update this suite."""
    assert BACKEND.get_name() == "ROCM_FLASHMLA_SPARSE_DSV41"
    assert AttentionBackendEnum.ROCM_FLASHMLA_SPARSE_DSV41.get_class() is BACKEND
    # The V4 enum resolves to the V4.0 classes, never to the V4.1 ones.
    assert (
        AttentionBackendEnum.ROCM_FLASHMLA_SPARSE_DSV4.get_class().__module__
        == "vllm.models.deepseek_v4.amd.rocm"
    )
    assert BACKEND.supported_dtypes == [torch.bfloat16]
    assert set(BACKEND.supported_kv_cache_dtypes) == {"auto", "fp8_ds_mla", "fp8"}
    assert BACKEND.get_supported_kernel_block_sizes() == [64, 128]
    assert BACKEND.get_supported_head_sizes() == [512]
    assert BACKEND.is_mla() and BACKEND.is_sparse() and BACKEND.supports_sink()
    assert not BACKEND.supports_sliding_window()
    assert not BACKEND.supports_non_causal()
    assert BACKEND.get_builder_cls() is MLA_BUILDER
    assert SWA_BACKEND.get_builder_cls() is SWA_BUILDER
    with pytest.raises(NotImplementedError):
        BACKEND.get_impl_cls()
    layer = rocm.DeepseekV41ROCMAiterMLAAttention
    assert layer.backend_cls is BACKEND and layer.swa_backend_cls is SWA_BACKEND
    # No padding of the query heads on ROCm: every TP shard's count runs.
    assert all(layer.get_padded_num_q_heads(h) == h for h in (8, 16, 32, 64))


@pytest.mark.parametrize("capability", [GFX942, GFX950], ids=["gfx942", "gfx950"])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8", "fp8_ds_mla"])
@pytest.mark.parametrize("block_size", [64, 128])
def test_selector_accepts_declared(capability, kv_cache_dtype, block_size):
    assert (
        _reasons(capability, kv_cache_dtype=kv_cache_dtype, block_size=block_size) == []
    )


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"dtype": torch.float16}, "dtype not supported"),
        ({"dtype": torch.float32}, "dtype not supported"),
        # DeepSeek V3/V3.2's MLA latent row (512 + 64): a generic sparse MLA.
        ({"head_size": 576}, "head_size not supported"),
        ({"head_size": 128}, "head_size not supported"),
        ({"kv_cache_dtype": "bfloat16"}, "kv_cache_dtype not supported"),
        ({"kv_cache_dtype": "fp8_e5m2"}, "kv_cache_dtype not supported"),
        # The NVFP4 compressed record only the SM100 mega-attention reads.
        ({"kv_cache_dtype": "nvfp4_ds_mla"}, "kv_cache_dtype not supported"),
        ({"block_size": 16}, "block_size not supported"),
        ({"block_size": 32}, "block_size not supported"),
        ({"use_mla": False}, "non-MLA not supported"),
        ({"use_sparse": False}, "non-sparse not supported"),
        ({"attn_type": AttentionType.ENCODER}, "attention type"),
        ({"attn_type": AttentionType.ENCODER_ONLY}, "attention type"),
        ({"attn_type": AttentionType.ENCODER_DECODER}, "attention type"),
        ({"use_non_causal": True}, "non-causal attention not supported"),
        ({"use_pcp": True}, "PCP not supported"),
        ({"use_dcp": True}, "DCP not supported"),
        ({"use_batch_invariant": True}, "batch invariance not supported"),
    ],
    ids=lambda v: str(v) if isinstance(v, str) else None,
)
def test_selector_rejects_undeclared(override, reason):
    reasons = _reasons(**override)
    assert any(reason in r for r in reasons), reasons


def test_compute_capability():
    """``supports_compute_capability`` is the CUDA base's ``major in (9, 10)``:
    on ROCm it admits every gfx9 part (gfx942 is 9.4, gfx950 9.5) and refuses
    RDNA (gfx11/gfx12), which has no fp8_ds_mla kernels."""
    assert BACKEND.supports_compute_capability(GFX942)
    assert BACKEND.supports_compute_capability(GFX950)
    assert not BACKEND.supports_compute_capability(DeviceCapability(11, 0))
    assert not BACKEND.supports_compute_capability(DeviceCapability(12, 0))


def _vllm_config(backend=None, pcp: int = 1, adaptive: bool | None = None):
    spec = (
        None
        if adaptive is None
        else SimpleNamespace(enable_adaptive_verification=adaptive)
    )
    return SimpleNamespace(
        attention_config=SimpleNamespace(backend=backend),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
        speculative_config=spec,
    )


@pytest.mark.parametrize(
    "backend",
    [
        None,
        AttentionBackendEnum.ROCM_FLASHMLA_SPARSE_DSV41,
        # Deprecated alias, still served by the V4.1 layer (with a warning).
        AttentionBackendEnum.ROCM_FLASHMLA_SPARSE_DSV4,
    ],
    ids=lambda b: "auto" if b is None else b.name,
)
def test_model_selects_the_rocm_layer(backend):
    model = pytest.importorskip("vllm.models.deepseek_v41.amd.model")
    attn_cls = model._select_dsv4_attn_cls(_vllm_config(backend))
    assert attn_cls is rocm.DeepseekV41ROCMAiterMLAAttention


@pytest.mark.parametrize(
    "backend",
    [
        AttentionBackendEnum.ROCM_AITER_MLA_SPARSE,
        AttentionBackendEnum.ROCM_AITER_MLA,
        AttentionBackendEnum.TRITON_MLA,
        AttentionBackendEnum.FLASHMLA_SPARSE_DSV41,
        AttentionBackendEnum.TRITON_ATTN,
    ],
    ids=lambda b: b.name,
)
def test_model_rejects_other_backends(backend):
    model = pytest.importorskip("vllm.models.deepseek_v41.amd.model")
    with pytest.raises(ValueError, match="not supported for DeepSeek V4.1 on ROCm"):
        model._select_dsv4_attn_cls(_vllm_config(backend))


def test_model_rejects_pcp():
    """PCP is CUDA-only for V4.1: refused before any layer is built."""
    model = pytest.importorskip("vllm.models.deepseek_v41.amd.model")
    with pytest.raises(NotImplementedError, match="PCP"):
        model._select_dsv4_attn_cls(_vllm_config(pcp=2))


@pytest.mark.parametrize("builder", [MLA_BUILDER, SWA_BUILDER], ids=["mla", "swa"])
@pytest.mark.parametrize(
    ("adaptive", "support"),
    [
        (None, AttentionCGSupport.UNIFORM_BATCH),
        (False, AttentionCGSupport.UNIFORM_BATCH),
        # Adaptive verification builds ownership from device query bounds into
        # persistent buffers, so any batch shape can be captured.
        (True, AttentionCGSupport.ALWAYS),
    ],
    ids=["no-spec", "spec", "adaptive"],
)
def test_cudagraph_support(builder, adaptive, support):
    assert builder.get_cudagraph_support(_vllm_config(adaptive=adaptive), None) == (
        support
    )


def test_swa_builder_keeps_draft_decode_update_off():
    """The ROCm ragged SWA buffers are not refreshed by
    ``update_draft_decode_metadata``, so the builder must not offer it."""
    assert SWA_BUILDER.supports_draft_decode_metadata_update is False


@pytest.mark.parametrize(
    ("env", "has_kv_transfer", "enabled"),
    [("0", False, False), ("1", False, True), ("1", True, False)],
    ids=["off", "on", "on-with-kv-connector"],
)
def test_aiter_sparse_mla_gate(monkeypatch, env, has_kv_transfer, enabled):
    """The aiter sparse MLA path is opt-in, gfx950-only, and never taken with
    a KV connector (its kernel does not scrub NaNs from foreign cache rows)."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.platforms.rocm import on_gfx950

    if enabled and not on_gfx950():
        pytest.skip("the aiter sparse MLA kernel is gfx950-only")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA", env)
    rocm_aiter_ops.refresh_env_variables()
    try:
        assert rocm._aiter_sparse_mla_enabled(has_kv_transfer) is enabled
    finally:
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()


def test_aiter_sparse_mla_needs_aiter(monkeypatch):
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA", "1")
    rocm_aiter_ops.refresh_env_variables()
    try:
        assert rocm._aiter_sparse_mla_enabled(False) is False
    finally:
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()


@pytest.mark.parametrize(
    ("kv_cache_dtype", "resolved"),
    [("auto", "fp8_ds_mla"), ("fp8", "fp8_ds_mla"), ("fp8_ds_mla", "fp8_ds_mla")],
)
def test_kv_cache_dtype_resolves_to_the_packed_record(kv_cache_dtype, resolved):
    attention = pytest.importorskip("vllm.models.deepseek_v41.attention")
    assert attention._resolve_dsv4_kv_cache_dtype(True, kv_cache_dtype, None) == (
        resolved,
        torch.uint8,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["bfloat16", "fp8_e4m3"])
def test_kv_cache_dtype_rejects_plain_rows(kv_cache_dtype):
    """The ROCm layer only has packed-record kernels: a plain bf16 or
    per-tensor fp8 cache is refused when the layer is built."""
    attention = pytest.importorskip("vllm.models.deepseek_v41.attention")
    with pytest.raises(ValueError, match="only support fp8 kv-cache"):
        attention._resolve_dsv4_kv_cache_dtype(True, kv_cache_dtype, None)


def test_rocm_writes_the_v4_record():
    """Off SM100 the SWA and compressed caches hold the V4 record (448 fp8
    NoPE + 64 bf16 RoPE, 584 B), which the suite's reference decodes."""
    attention = pytest.importorskip("vllm.models.deepseek_v41.attention")
    assert attention._use_v41_mxfp8_kv_record() is False


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_layer_types(ratio):
    sparse_mla = pytest.importorskip("vllm.models.deepseek_v41.sparse_mla")
    assert sparse_mla.deepseek_v41_layer_type(ratio)


@pytest.mark.parametrize("ratio", [3, 4, 128])
def test_layer_types_reject_v4_ratios(ratio):
    """V4.0's ratios 4 and 128 are not V4.1 layer types."""
    sparse_mla = pytest.importorskip("vllm.models.deepseek_v41.sparse_mla")
    with pytest.raises(ValueError, match="compress_ratio"):
        sparse_mla.deepseek_v41_layer_type(ratio)


def _bare_layer():
    # Only the class-level dispatch of _indexer_cls is exercised.
    return object.__new__(rocm.DeepseekV41ROCMAiterMLAAttention)


def test_indexer_class_follows_the_cache(monkeypatch):
    attention = pytest.importorskip("vllm.models.deepseek_v41.attention")
    layer = _bare_layer()
    monkeypatch.setattr(rocm, "_gfx942_monokernel", lambda: False)
    mxfp4 = SimpleNamespace(rocm_mxfp4=True)
    fp8 = SimpleNamespace(rocm_mxfp4=False)
    assert layer._indexer_cls(mxfp4) is rocm.DeepseekV41RocmMxfp4Indexer
    assert layer._indexer_cls(fp8) is attention.DeepseekV4Indexer
    assert layer._indexer_cls(None) is attention.DeepseekV4Indexer
    # gfx942 --monokernel takes its top-512 indexer, unless the cache is MXFP4.
    monkeypatch.setattr(rocm, "_gfx942_monokernel", lambda: True)
    assert layer._indexer_cls(fp8) is rocm.DeepseekV41Gfx942Indexer
    assert layer._indexer_cls(mxfp4) is rocm.DeepseekV41RocmMxfp4Indexer


# ---- indexer backends (compress_ratio > 1 and two-level candidate paths) ----


def test_v4_indexer_backend_declarations():
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.mla.indexer import DeepseekV4IndexerBackend
    from vllm.v1.kv_cache_interface import KVCacheLayout

    backend = DeepseekV4IndexerBackend
    assert backend.get_name() == "DEEPSEEK_V4_INDEXER"
    assert backend.supported_kv_cache_layouts() == (
        KVCacheLayout.BLHNC,
        KVCacheLayout.BLNHC,
    )
    assert backend.get_supported_kernel_block_sizes() == [256]
    # ROCm runs adaptive verification through the flattened indexer path.
    if current_platform.is_rocm():
        assert backend.supports_device_cpu_query_lens_mismatch()


def test_v41_indexer_backend_block_sizes():
    """``DeepseekV41IndexerBackend`` keys its block size on the CUDA "family
    90" check. On ROCm every gfx9 part is 9.x, so the check holds there and
    the FP8 indexer asks for 64-token blocks, which is what the gfx942 FP8
    indexer kernels need. The MXFP4 backend (CDNA4) declares its own sizes."""
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
    from vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer import (
        DeepseekV4RocmMxfp4IndexerBackend,
        DeepseekV41RocmMxfp4IndexerBackend,
        DeepseekV41RocmMxfp4IndexerMetadataBuilder,
        RocmMxfp4IndexerMetadataBuilder,
    )

    if current_platform.is_rocm():
        assert DeepseekV41IndexerBackend.get_supported_kernel_block_sizes() == [64]
    assert DeepseekV41RocmMxfp4IndexerBackend.get_supported_kernel_block_sizes() == [
        64,
        128,
    ]
    assert (
        DeepseekV41RocmMxfp4IndexerBackend.get_builder_cls()
        is DeepseekV41RocmMxfp4IndexerMetadataBuilder
    )
    assert (
        DeepseekV4RocmMxfp4IndexerBackend.get_builder_cls()
        is RocmMxfp4IndexerMetadataBuilder
    )
    assert DeepseekV4RocmMxfp4IndexerBackend.get_supported_kernel_block_sizes() == [256]
    assert (
        RocmMxfp4IndexerMetadataBuilder.get_cudagraph_support(None, None)
        == AttentionCGSupport.ALWAYS
    )


def test_indexer_cache_picks_the_rocm_mxfp4_backend():
    """On ROCm with ``indexer_kv_dtype="mxfp4"`` the index cache's backend is
    the aiter paged MXFP4 one, also when ``indexer_sparse_logits`` asks for
    the candidate consumers' sparse path (whose CUDA backend needs SM100)."""
    attention = pytest.importorskip("vllm.models.deepseek_v41.attention")
    from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
    from vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer import (
        DeepseekV41RocmMxfp4IndexerBackend,
    )
    from vllm.v1.attention.backends.mla.sparse_indexer import (
        DeepseekV41SparseIndexerBackend,
    )

    cache = object.__new__(attention.DeepseekV4IndexerCache)
    for rocm_mxfp4, sparse_logits, want in [
        (True, True, DeepseekV41RocmMxfp4IndexerBackend),
        (True, False, DeepseekV41RocmMxfp4IndexerBackend),
        (False, True, DeepseekV41SparseIndexerBackend),
        (False, False, DeepseekV41IndexerBackend),
    ]:
        cache.__dict__.update(rocm_mxfp4=rocm_mxfp4, sparse_logits=sparse_logits)
        assert cache.get_attn_backend() is want
