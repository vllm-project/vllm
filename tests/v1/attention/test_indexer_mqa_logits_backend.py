# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selection of the DSA indexer's paged MQA-logits (decode) kernel.

``kernel_config.sparse_indexer_mqa_logits_backend`` chooses between DeepGEMM
and FlashInfer's SM120 route. The schedule metadata is kernel-specific, so the
metadata builder resolves the choice once from the config and records it on the
decode metadata for the layer; these tests pin that resolution.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.kernel import KernelConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla import indexer
from vllm.v1.kv_cache_interface import (
    KVCacheLayout,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

# (index_n_heads, page_kv) pairs the FlashInfer package exports, every one for
# each exported depth. Stands in for FlashInfer's capability query.
SHIPPED_ROUTES = frozenset({(32, 64), (32, 128), (64, 64)})
# Q rows per request (next_n = 1 + num_speculative_tokens) the catalog exports
# (policy.next_n): the pinned package ships 1, 2 and 4; a build that adds 3,
# 5 and 6 widens the native set.
EXPORTED_NEXT_N = (1, 2, 4)
EXTENDED_NEXT_N = (1, 2, 3, 4, 5, 6)
# FlashInfer's scheduler request ceiling (catalog policy.max_batch).
MAX_BATCH = 4096
# FP8 indexer K row: 128 e4m3 values + the fp32 scale.
INDEXER_HEAD_BYTES = 132


def _set_arch(
    monkeypatch: pytest.MonkeyPatch,
    family: int,
    *,
    cuda: bool = True,
    deep_gemm: bool = True,
    flashinfer_sm120: bool = True,
    routes: frozenset[tuple[int, int]] = SHIPPED_ROUTES,
    next_n: tuple[int, ...] = EXPORTED_NEXT_N,
    route_next_n: tuple[int, ...] | None = None,
    max_batch: int | None = MAX_BATCH,
) -> None:
    """``next_n`` is the catalog's exported depth set; ``route_next_n`` the
    depths every shipped (heads, page) route has a kernel for, when a depth is
    exported for other routes only."""
    monkeypatch.setattr(current_platform, "is_cuda", lambda: cuda)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: capability // 10 == family,
    )
    monkeypatch.setattr(indexer, "has_deep_gemm", lambda: deep_gemm)
    monkeypatch.setattr(
        indexer, "has_flashinfer_sm120_paged_mqa_logits", lambda: flashinfer_sm120
    )
    monkeypatch.setattr(
        indexer, "flashinfer_sm120_paged_mqa_logits_max_batch", lambda: max_batch
    )
    monkeypatch.setattr(
        indexer, "flashinfer_sm120_paged_mqa_logits_next_n", lambda: next_n
    )
    routed_next_n = next_n if route_next_n is None else route_next_n

    def route_available(num_heads: int, page_kv: int, depth: int) -> bool:
        return (num_heads, page_kv) in routes and depth in routed_next_n

    monkeypatch.setattr(
        indexer,
        "flashinfer_sm120_paged_mqa_logits_route_available",
        route_available,
    )


def _config(
    backend: str = "auto",
    *,
    num_speculative_tokens: int = 0,
    index_n_heads: int = 32,
    index_kpool: int | None = None,
    decode_context_parallel_size: int = 1,
    max_num_seqs: int = 256,
    layout: KVCacheLayout | None = KVCacheLayout.LBHNC,
) -> SimpleNamespace:
    hf_text_config = SimpleNamespace(index_n_heads=index_n_heads)
    if index_kpool is not None:
        hf_text_config.index_kpool = index_kpool
    speculative_config = (
        SimpleNamespace(enable_adaptive_verification=False)
        if num_speculative_tokens
        else None
    )

    def get_resolved_kv_cache_layout() -> KVCacheLayout:
        if layout is None:
            raise ValueError("KV cache layout has not been resolved yet")
        return layout

    return SimpleNamespace(
        kernel_config=SimpleNamespace(sparse_indexer_mqa_logits_backend=backend),
        model_config=SimpleNamespace(
            architectures=["DeepseekV32ForCausalLM"], hf_text_config=hf_text_config
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=decode_context_parallel_size
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=get_resolved_kv_cache_layout
        ),
        num_speculative_tokens=num_speculative_tokens,
        speculative_config=speculative_config,
    )


def _indexer_spec(
    block_size: int = 64, *, tokens_per_state: int = 1, alignment: int | None = None
) -> MLAAttentionSpec:
    """The indexer layer's spec (DeepseekV32IndexerCache.get_kv_cache_spec)."""
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=INDEXER_HEAD_BYTES,
        dtype=torch.uint8,
        tokens_per_state=tokens_per_state,
        alignment=alignment,
    )


def _mla_spec(block_size: int = 64) -> MLAAttentionSpec:
    """The MLA latent cache spec that shares the indexer's KV cache group."""
    return MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
    )


def _resolve(config: SimpleNamespace, spec: MLAAttentionSpec) -> str:
    """Resolve as the metadata builder does: its spec is the kernel-block copy,
    so the spec's block is the page the kernel reads."""
    return indexer.resolve_sparse_indexer_mqa_logits_backend(
        config, spec, spec.block_size
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("FlashInfer-SM120", "flashinfer_sm120"),
        ("Deep-GEMM", "deep_gemm"),
        ("AUTO", "auto"),
    ],
)
def test_kernel_config_normalizes_backend_spelling(raw: str, expected: str) -> None:
    config = KernelConfig(sparse_indexer_mqa_logits_backend=raw)
    assert config.sparse_indexer_mqa_logits_backend == expected


@pytest.mark.cpu_test
def test_kernel_config_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        KernelConfig(sparse_indexer_mqa_logits_backend="cutlass")


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "family,flashinfer,num_spec,heads,page_kv,expected",
    [
        (12, True, 0, 32, 64, "flashinfer_sm120"),
        (12, True, 1, 32, 128, "flashinfer_sm120"),
        (12, True, 3, 64, 64, "flashinfer_sm120"),
        # next_n=3 (MTP=2) has no exported kernel.
        (12, True, 2, 32, 64, "deep_gemm"),
        # Unshipped (heads, page) combinations.
        (12, True, 0, 64, 128, "deep_gemm"),
        (12, True, 0, 16, 64, "deep_gemm"),
        # FlashInfer build without the route.
        (12, False, 0, 32, 64, "deep_gemm"),
        # Datacenter Blackwell and Hopper keep DeepGEMM.
        (10, True, 0, 32, 64, "deep_gemm"),
        (9, True, 0, 32, 64, "deep_gemm"),
    ],
)
def test_auto_selects_flashinfer_only_for_shipped_sm120_routes(
    monkeypatch: pytest.MonkeyPatch,
    family: int,
    flashinfer: bool,
    num_spec: int,
    heads: int,
    page_kv: int,
    expected: str,
) -> None:
    _set_arch(monkeypatch, family, flashinfer_sm120=flashinfer)
    config = _config(num_speculative_tokens=num_spec, index_n_heads=heads)
    assert _resolve(config, _indexer_spec(page_kv)) == expected


@pytest.mark.cpu_test
def test_auto_keeps_kpool_indexer_on_deep_gemm(monkeypatch: pytest.MonkeyPatch):
    """GLM's kpool indexer scores its pool pages with DeepGEMM directly, so the
    shared metadata builder must not hand it FlashInfer schedule metadata."""
    _set_arch(monkeypatch, 12)
    config = _config(index_n_heads=32, index_kpool=16)
    assert _resolve(config, _indexer_spec(64)) == "deep_gemm"


@pytest.mark.cpu_test
def test_explicit_deep_gemm_never_probes_flashinfer(monkeypatch: pytest.MonkeyPatch):
    _set_arch(monkeypatch, 12)

    def probed() -> bool:
        raise AssertionError("deep_gemm must not import FlashInfer")

    monkeypatch.setattr(indexer, "has_flashinfer_sm120_paged_mqa_logits", probed)
    assert _resolve(_config("deep_gemm"), _indexer_spec(64)) == "deep_gemm"


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "family,flashinfer,num_spec,heads,page_kv,reason",
    [
        (10, True, 0, 32, 64, "SM12x"),
        (12, False, 0, 32, 64, "not importable"),
        (12, True, 2, 32, 64, "next_n"),
        (12, True, 0, 64, 128, "no SM120 paged MQA-logits route"),
    ],
)
def test_explicit_flashinfer_sm120_names_the_unmet_constraint(
    monkeypatch: pytest.MonkeyPatch,
    family: int,
    flashinfer: bool,
    num_spec: int,
    heads: int,
    page_kv: int,
    reason: str,
) -> None:
    _set_arch(monkeypatch, family, flashinfer_sm120=flashinfer)
    config = _config(
        "flashinfer_sm120", num_speculative_tokens=num_spec, index_n_heads=heads
    )
    with pytest.raises(RuntimeError, match=reason):
        _resolve(config, _indexer_spec(page_kv))


@pytest.mark.cpu_test
def test_explicit_flashinfer_sm120_resolves_when_constraints_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_arch(monkeypatch, 12)
    config = _config("flashinfer_sm120", num_speculative_tokens=3)
    assert _resolve(config, _indexer_spec(128)) == "flashinfer_sm120"


@pytest.mark.cpu_test
def test_decode_context_parallelism_keeps_deep_gemm(monkeypatch: pytest.MonkeyPatch):
    """FlashInfer requires max_context_len <= block_table.shape[1] * page_kv,
    and a DCP-sharded indexer block table only spans max_model_len / dcp."""
    _set_arch(monkeypatch, 12)
    config = _config(decode_context_parallel_size=2)
    assert _resolve(config, _indexer_spec(64)) == "deep_gemm"
    with pytest.raises(RuntimeError, match="decode context parallelism"):
        _resolve(
            _config("flashinfer_sm120", decode_context_parallel_size=2),
            _indexer_spec(64),
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "num_spec,max_num_seqs,expected",
    [
        # Plain decode: one native row per request.
        (0, MAX_BATCH, "flashinfer_sm120"),
        (0, MAX_BATCH + 1, "deep_gemm"),
        # next_n=4: a 3-token step has no native kernel and flattens to one
        # row per token, so up to 3 rows per request reach the scheduler.
        (3, MAX_BATCH // 3, "flashinfer_sm120"),
        (3, MAX_BATCH // 3 + 1, "deep_gemm"),
        # next_n=2 ships every reachable depth natively.
        (1, MAX_BATCH, "flashinfer_sm120"),
    ],
)
def test_scheduler_request_ceiling_bounds_the_decode_rows(
    monkeypatch: pytest.MonkeyPatch, num_spec: int, max_num_seqs: int, expected: str
) -> None:
    _set_arch(monkeypatch, 12)
    config = _config(num_speculative_tokens=num_spec, max_num_seqs=max_num_seqs)
    assert _resolve(config, _indexer_spec(64)) == expected


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "exported,num_spec,expected",
    [
        # The pinned package exports 1, 2 and 4: next_n=3 has no kernel.
        (EXPORTED_NEXT_N, 2, "deep_gemm"),
        (EXPORTED_NEXT_N, 3, "flashinfer_sm120"),
        # A catalog that adds 3, 5 and 6 serves those depths natively.
        (EXTENDED_NEXT_N, 2, "flashinfer_sm120"),
        (EXTENDED_NEXT_N, 4, "flashinfer_sm120"),
        (EXTENDED_NEXT_N, 5, "flashinfer_sm120"),
        (EXTENDED_NEXT_N, 6, "deep_gemm"),
        # A build without the depth query exports nothing.
        ((), 0, "deep_gemm"),
    ],
)
def test_native_depths_follow_the_catalog(
    monkeypatch: pytest.MonkeyPatch,
    exported: tuple[int, ...],
    num_spec: int,
    expected: str,
) -> None:
    """The depth set is read from FlashInfer's catalog (``exported_next_n``),
    never assumed, so a package that ships more depths widens the route."""
    _set_arch(monkeypatch, 12, next_n=exported)
    config = _config(num_speculative_tokens=num_spec)
    assert _resolve(config, _indexer_spec(64)) == expected
    for depth in range(1, 9):
        assert indexer._supports_native_decode(depth, "flashinfer_sm120") == (
            depth in exported
        ), f"next_n={depth}"


@pytest.mark.cpu_test
def test_exported_depth_without_a_route_for_the_model_keeps_deep_gemm(monkeypatch):
    """Every exported depth up to next_n needs a kernel for the model's
    (heads, page): a step may hand the kernel fewer rows than next_n, and a
    depth the catalog exports for other routes only would abort there."""
    _set_arch(monkeypatch, 12, next_n=EXTENDED_NEXT_N, route_next_n=EXPORTED_NEXT_N)
    config = _config(num_speculative_tokens=3, index_n_heads=64)
    assert _resolve(config, _indexer_spec(64)) == "deep_gemm"
    with pytest.raises(RuntimeError, match="no SM120 paged MQA-logits route"):
        _resolve(
            _config("flashinfer_sm120", num_speculative_tokens=3, index_n_heads=64),
            _indexer_spec(64),
        )
    # Depths below the missing one are unaffected.
    config = _config(num_speculative_tokens=1, index_n_heads=64)
    assert _resolve(config, _indexer_spec(64)) == "flashinfer_sm120"


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "num_spec,max_num_seqs,expected",
    [
        # Every depth up to next_n=4 is native: one scheduler row per request.
        (3, MAX_BATCH, "flashinfer_sm120"),
        (3, MAX_BATCH + 1, "deep_gemm"),
        # Likewise at the deepest exported depth.
        (5, MAX_BATCH, "flashinfer_sm120"),
        (5, MAX_BATCH + 1, "deep_gemm"),
    ],
)
def test_full_depth_coverage_keeps_one_scheduler_row_per_request(
    monkeypatch: pytest.MonkeyPatch, num_spec: int, max_num_seqs: int, expected: str
) -> None:
    """With 1..6 all exported no step depth flattens, so max_num_seqs itself
    (not max_num_seqs x the deepest non-native depth) meets the ceiling."""
    _set_arch(monkeypatch, 12, next_n=EXTENDED_NEXT_N)
    config = _config(num_speculative_tokens=num_spec, max_num_seqs=max_num_seqs)
    assert _resolve(config, _indexer_spec(64)) == expected


@pytest.mark.cpu_test
def test_explicit_flashinfer_sm120_names_the_request_ceiling(monkeypatch):
    _set_arch(monkeypatch, 12)
    config = _config(
        "flashinfer_sm120", num_speculative_tokens=3, max_num_seqs=MAX_BATCH
    )
    with pytest.raises(RuntimeError, match="request ceiling"):
        _resolve(config, _indexer_spec(64))
    _set_arch(monkeypatch, 12, max_batch=None)
    with pytest.raises(RuntimeError, match="max_batch"):
        _resolve(_config("flashinfer_sm120"), _indexer_spec(64))


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "layout", [KVCacheLayout.BLHNC, KVCacheLayout.BLNHC, KVCacheLayout.LBHNC, None]
)
def test_block_outermost_layouts_take_the_flashinfer_route(
    monkeypatch: pytest.MonkeyPatch, layout: KVCacheLayout | None
) -> None:
    """DeepSeek-V4/V4.1 pack every layer's page into each block, so the
    per-layer indexer view is strided; FlashInfer's kernel reads it through a
    TMA descriptor that carries the block stride (like DeepGEMM), so the
    layout -- resolved or not -- never decides the route."""
    _set_arch(monkeypatch, 12)
    assert _resolve(_config(layout=layout), _indexer_spec(64)) == "flashinfer_sm120"
    assert (
        _resolve(_config("flashinfer_sm120", layout=layout), _indexer_spec(64))
        == "flashinfer_sm120"
    )


@pytest.mark.cpu_test
def test_padded_indexer_page_takes_the_flashinfer_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alignment padding widens the block stride, which the kernel carries."""
    spec = _indexer_spec(64, alignment=576)
    assert spec.page_size_padded is not None
    _set_arch(monkeypatch, 12)
    assert _resolve(_config(), spec) == "flashinfer_sm120"
    assert _resolve(_config("flashinfer_sm120"), spec) == "flashinfer_sm120"


@pytest.mark.cpu_test
def test_unresolved_spec_keeps_deep_gemm(monkeypatch: pytest.MonkeyPatch):
    _set_arch(monkeypatch, 12)
    # Without a spec the page is unknown: never select a route blind.
    assert indexer.resolve_sparse_indexer_mqa_logits_backend(_config()) == "deep_gemm"
    assert not indexer._use_flattening(_config())


@pytest.mark.cpu_test
def test_flashinfer_route_widens_native_decode_depths(monkeypatch):
    """FlashInfer ships the catalog's depths natively on SM120 (1, 2 and 4 at
    the pinned package); DeepGEMM on SM120 keeps its conservative {1, 2} gate."""
    _set_arch(monkeypatch, 12)
    for next_n in (1, 2, 3, 4, 5, 8):
        assert indexer._supports_native_decode(next_n, "flashinfer_sm120") == (
            next_n in EXPORTED_NEXT_N
        ), f"next_n={next_n}"
        assert indexer._supports_native_decode(next_n) == (next_n in (1, 2)), (
            f"next_n={next_n}"
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_spec", [0, 1, 3])
def test_flashinfer_route_runs_native_rows_under_uniform_batch_graphs(
    monkeypatch: pytest.MonkeyPatch, num_spec: int
) -> None:
    """FlashInfer takes the native (B, next_n) rows and rejects the SM100
    varlen row indices, so SM120 stays on UNIFORM_BATCH graphs (like the SM90
    native path) instead of flattening MTP batches."""
    _set_arch(monkeypatch, 12)
    config = _config(num_speculative_tokens=num_spec)
    assert _resolve(config, _indexer_spec(64)) == "flashinfer_sm120"
    assert not indexer._supports_varlen_paged_mqa_logits()
    assert not indexer._use_flattening(config, "flashinfer_sm120")
    assert (
        indexer.DeepseekV32IndexerMetadataBuilder.get_cudagraph_support(
            config, _indexer_spec(64)
        )
        == AttentionCGSupport.UNIFORM_BATCH
    )


@pytest.mark.cpu_test
def test_unshipped_next_n_keeps_the_deep_gemm_flattening_path(monkeypatch):
    """next_n=3 has no FlashInfer kernel: "auto" stays on DeepGEMM, which
    flattens MTP batches on SM120 and so supports graphs for any batch."""
    _set_arch(monkeypatch, 12)
    config = _config(num_speculative_tokens=2)
    assert _resolve(config, _indexer_spec(64)) == "deep_gemm"
    assert indexer._use_flattening(config, "deep_gemm")
    assert (
        indexer.DeepseekV32IndexerMetadataBuilder.get_cudagraph_support(
            config, _indexer_spec(64)
        )
        == AttentionCGSupport.ALWAYS
    )


@pytest.mark.cpu_test
def test_mtp4_flattens_on_deep_gemm_but_runs_native_on_flashinfer(monkeypatch):
    _set_arch(monkeypatch, 12)
    config = _config(num_speculative_tokens=3)
    assert indexer._use_flattening(config, "deep_gemm")
    assert not indexer._use_flattening(config, "flashinfer_sm120")


@pytest.mark.cpu_test
def test_kernel_pages_of_manager_and_kernel_specs() -> None:
    """The runner's spec carries the manager block; the builder's the kernel
    block. The kernel page is among the manager block's whole-state splits."""
    manager = _indexer_spec(128)
    kernel = manager.copy_with_new_block_size(64)
    assert indexer.indexer_mqa_logits_kv_pages(kernel, kernel.block_size) == [64]
    assert indexer.indexer_mqa_logits_kv_pages(manager, None) == [
        128,
        64,
        32,
        16,
        8,
        4,
        2,
        1,
    ]
    # DeepseekV4: 4 tokens per state, 256-token kernel block -> 64-state page.
    compressed = _indexer_spec(256, tokens_per_state=4)
    assert indexer.indexer_mqa_logits_kv_pages(compressed, 256) == [64]
    assert indexer.indexer_mqa_logits_kv_pages(compressed, None) == [
        64,
        32,
        16,
        8,
        4,
        2,
        1,
    ]


@pytest.mark.cpu_test
def test_cudagraph_support_agrees_with_the_builder_across_a_kernel_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--block-size 128 with the 64-token indexer kernel block: the runner's
    manager spec has page 128 ((64, 128) is unshipped) while the builder's
    kernel copy has page 64 ((64, 64) is shipped). Both must land on the
    FlashInfer route, else MTP-3 batches would be replayed through full graphs
    the native builder never captured."""
    _set_arch(monkeypatch, 12)
    config = _config(num_speculative_tokens=3, index_n_heads=64)
    manager = _indexer_spec(128)
    kernel = manager.copy_with_new_block_size(64)
    assert _resolve(config, kernel) == "flashinfer_sm120"
    assert (
        indexer.DeepseekV32IndexerMetadataBuilder.get_cudagraph_support(config, manager)
        == AttentionCGSupport.UNIFORM_BATCH
    )
    # An unshipped head count fails for every possible split at both sites.
    config = _config(num_speculative_tokens=3, index_n_heads=16)
    assert _resolve(config, kernel) == "deep_gemm"
    assert (
        indexer.DeepseekV32IndexerMetadataBuilder.get_cudagraph_support(config, manager)
        == AttentionCGSupport.ALWAYS
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "family,num_spec,expected",
    [
        (12, 0, AttentionCGSupport.UNIFORM_BATCH),
        (12, 3, AttentionCGSupport.UNIFORM_BATCH),
        # next_n=3: DeepGEMM flattens on SM120.
        (12, 2, AttentionCGSupport.ALWAYS),
        (9, 0, AttentionCGSupport.UNIFORM_BATCH),
    ],
)
def test_cudagraph_support_accepts_the_uniform_type_group_spec(
    monkeypatch: pytest.MonkeyPatch,
    family: int,
    num_spec: int,
    expected: AttentionCGSupport,
) -> None:
    """The V1 runner hands get_cudagraph_support the KV cache group's spec, a
    UniformTypeKVCacheSpecs over the MLA and indexer layers (same block size,
    different pages). Its num_states is not defined (tokens_per_state raises
    NotImplementedError), so the members must be unwrapped."""
    _set_arch(monkeypatch, family)
    group_spec = UniformTypeKVCacheSpecs(
        block_size=64,
        kv_cache_specs={
            "layers.0.self_attn.mla_attn": _mla_spec(64),
            "layers.0.self_attn.indexer.k_cache": _indexer_spec(64),
        },
    )
    config = _config(num_speculative_tokens=num_spec)
    assert (
        indexer.DeepseekV32IndexerMetadataBuilder.get_cudagraph_support(
            config, group_spec
        )
        == expected
    )
