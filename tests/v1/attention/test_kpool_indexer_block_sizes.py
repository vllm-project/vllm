# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the kpool indexer's declared kernel block sizes (no GPU).

On ROCm ``DeepseekV32IndexerBackend`` used to report ``[1, MultipleOf(16)]``
for every deployment, which lets ``select_common_block_size`` keep the whole
manager block (640-4352 tokens depending on TP) as the kernel block size. The
kpool writer and the index-cache gather, however, address pool pages of 32/64
compressed states (``page_size * index_kpool`` tokens), so a manager-granular
block table aliases cache pages: #54359 (gfx950), #56380 (gfx942), #55280
(TP4 GPU fault), #58858 (confirmation).

Unpooled indexer caches (DeepSeek-V3.2, ``tokens_per_state == 1``) report
``[1, MultipleOf(16)]``; pooled-state caches report exactly the paged-MQA
pool-page lattice. The invariant the rest of the stack
relies on: for any manager block that passes ``Glm5NextIndexerCache``'s
assertions, ``select_common_block_size(manager) == spec.storage_block_size``,
i.e. kernel block == storage block, so the block table is already storage-page
granular and the builder's conversion gate correctly no-ops.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.glm5next.common.attention import Glm5NextIndexerCache
from vllm.utils.deep_gemm import PAGED_MQA_PAGE_SIZES
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla import indexer as indexer_mod
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    KpoolTailBackend,
    _kv_pool_tokens_per_state,
)
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.worker import utils as worker_utils_mod
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.utils import prepare_kernel_block_sizes, select_common_block_size

# ``index_kpool`` of zai-org/GLM-5.3-Flash: one indexer state pools 4
# compressed states (hence storage_block_size = page * 4).
INDEX_KPOOL = 4

# Representative manager blocks: the hybrid KDA/mamba floors at TP8..TP1
# (640/1152/2176/4352), user-specified --block-size values (128, 256, 8192),
# and blocks whose pools do not tile 64-state pages evenly (384 -> 96 pools,
# 1280, 10240).
MANAGER_BLOCKS = [128, 256, 384, 640, 1152, 1280, 2176, 4352, 8192, 10240]


def _mock_rocm_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    # Platform predicates are mutually exclusive in production. Override both
    # so these ROCm policy tests do not inherit CUDA capabilities from the CI
    # host running them (same approach as test_deepseek_v4_rocm_adaptive).
    monkeypatch.setattr(indexer_mod.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(indexer_mod.current_platform, "is_rocm", lambda: True)


def _assert_no_pooling_declaration(sizes: list) -> None:
    """The no-pooling declaration ``[1, MultipleOf(16)]``, compared by value
    (MultipleOf defines no __eq__)."""
    assert len(sizes) == 2
    assert sizes[0] == 1
    assert isinstance(sizes[1], MultipleOf)
    assert sizes[1].base == 16


def _kpool_storage_spec(manager_block: int) -> MLAAttentionSpec:
    """Real Glm5NextIndexerCache spec for the given cache block size."""
    with set_current_vllm_config(VllmConfig()):
        cache = Glm5NextIndexerCache(
            head_dim=128,
            dtype=torch.bfloat16,
            prefix="model.layers.0.indexer.k_cache_probe",
            cache_config=SimpleNamespace(block_size=manager_block),
            index_kpool=INDEX_KPOOL,
        )
        spec = cache.get_kv_cache_spec(VllmConfig())
    assert isinstance(spec, MLAAttentionSpec)
    return spec


def test_kpool_spec_tokens_per_state_flows_to_the_declaration(
    monkeypatch: pytest.MonkeyPatch,
):
    """The spec's tokens_per_state drives the declaration."""
    _mock_rocm_platform(monkeypatch)
    spec = _kpool_storage_spec(640)
    assert spec.tokens_per_state == INDEX_KPOOL
    assert _kv_pool_tokens_per_state(spec) == INDEX_KPOOL
    assert DeepseekV32IndexerBackend.get_supported_kernel_block_sizes(spec) == [
        page * INDEX_KPOOL for page in PAGED_MQA_PAGE_SIZES
    ]


@pytest.mark.parametrize("manager_block", MANAGER_BLOCKS)
def test_kpool_kernel_block_equals_storage_block(
    monkeypatch: pytest.MonkeyPatch, manager_block: int
):
    """select_common_block_size(manager, [indexer]) == storage_block_size.

    Equal sizes mean the builder conversion gate does not fire and the
    hybrid-split block table is already storage-page granular, i.e. the
    granularity assumed by the kpool writer and the gather.
    """
    _mock_rocm_platform(monkeypatch)
    spec = _kpool_storage_spec(manager_block)

    # storage_block_size is 256 tokens iff 256 divides the manager block,
    # else 128 tokens (kpool = 4); the selection must reproduce it.
    expected_storage = 256 if manager_block % 256 == 0 else 128
    assert spec.storage_block_size == expected_storage

    selected = select_common_block_size(
        manager_block, [DeepseekV32IndexerBackend], spec
    )
    assert selected == spec.storage_block_size
    assert manager_block % selected == 0  # hybrid block-table split stays legal


@pytest.mark.parametrize("manager_block", MANAGER_BLOCKS)
def test_kpool_selection_survives_the_full_real_group(
    monkeypatch: pytest.MonkeyPatch, manager_block: int
):
    """Same answer with the sparse-MLA and tail backends in the vote."""
    _mock_rocm_platform(monkeypatch)
    spec = _kpool_storage_spec(manager_block)
    selected = select_common_block_size(
        manager_block,
        [ROCMAiterMLASparseBackend, DeepseekV32IndexerBackend, KpoolTailBackend],
        spec,
    )
    assert selected == spec.storage_block_size


def test_uncompressed_indexer_ignores_pooling_inputs(
    monkeypatch: pytest.MonkeyPatch,
):
    """DeepSeek-V3.2-style caches: pooling inputs do not alter the
    ``[1, MultipleOf(16)]`` declaration."""
    _mock_rocm_platform(monkeypatch)
    spec = MLAAttentionSpec(
        block_size=640,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    assert spec.tokens_per_state == 1
    assert _kv_pool_tokens_per_state(spec) is None
    _assert_no_pooling_declaration(
        DeepseekV32IndexerBackend.get_supported_kernel_block_sizes(spec)
    )
    # Neither spec nor config: the no-pooling declaration (zero-arg form).
    assert _kv_pool_tokens_per_state(None) is None
    _assert_no_pooling_declaration(
        DeepseekV32IndexerBackend.get_supported_kernel_block_sizes()
    )


def test_config_fallback_without_spec(monkeypatch: pytest.MonkeyPatch):
    """Callers without a spec (hybrid alignment, `supports_block_size`) read
    ``index_kpool`` from the current config."""
    _mock_rocm_platform(monkeypatch)

    def _cfg(index_kpool):
        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(index_kpool=index_kpool)
            )
        )

    monkeypatch.setattr(indexer_mod, "get_current_vllm_config_or_none", lambda: _cfg(4))
    assert _kv_pool_tokens_per_state(None) == INDEX_KPOOL
    assert DeepseekV32IndexerBackend.get_supported_kernel_block_sizes() == [
        page * INDEX_KPOOL for page in PAGED_MQA_PAGE_SIZES
    ]

    # Not a kpool deployment: no-pooling declaration.
    monkeypatch.setattr(
        indexer_mod, "get_current_vllm_config_or_none", lambda: _cfg(None)
    )
    assert _kv_pool_tokens_per_state(None) is None
    _assert_no_pooling_declaration(
        DeepseekV32IndexerBackend.get_supported_kernel_block_sizes()
    )


def test_non_rocm_declarations_unchanged():
    """Non-ROCm declares [64] for this backend whether or not the cache
    pools states: 64 divides every ``PAGED_MQA_PAGE_SIZES * index_kpool``
    storage block, so ``DeepseekV32IndexerMetadataBuilder.build()`` rewrites
    the table to storage-page ids before consumers read it."""
    spec = _kpool_storage_spec(640)
    # This runner is non-ROCm (CI); the ROCm path is covered by the
    # monkeypatched tests above, this exercises the native dispatch.
    if indexer_mod.current_platform.is_rocm():
        pytest.skip("native platform is ROCm; covered by the patched tests")
    assert _kv_pool_tokens_per_state(spec) == INDEX_KPOOL
    assert DeepseekV32IndexerBackend.get_supported_kernel_block_sizes(spec) == [64]
    assert DeepseekV32IndexerBackend.get_supported_kernel_block_sizes() == [64]


def test_fallback_resolves_under_set_current_vllm_config(
    monkeypatch: pytest.MonkeyPatch,
):
    """Ambient config resolves the same lattice as the spec-fed path.

    The zero-arg ``get_supported_kernel_block_sizes()`` call sites - nixl and
    mooncake ``_sync_block_size_with_kernel`` (built from
    ``gpu_worker.initialize_from_config``) and HiSparse block resolution - all
    run under ``with set_current_vllm_config(...)``, so the config fallback
    must return the page lattice there and stay consistent with
    ``prepare_kernel_block_sizes``, which passes the spec explicitly.
    """
    _mock_rocm_platform(monkeypatch)
    vllm_config = VllmConfig()
    # Stub only the fields the fallback reads; constructing a real
    # ModelConfig would require hub access.
    vllm_config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(index_kpool=INDEX_KPOOL)
    )
    with set_current_vllm_config(vllm_config):
        assert _kv_pool_tokens_per_state(None) == INDEX_KPOOL
        assert DeepseekV32IndexerBackend.get_supported_kernel_block_sizes() == [
            page * INDEX_KPOOL for page in PAGED_MQA_PAGE_SIZES
        ]
        # 128 divides 640 and 256 does not, so 128 is the common size
        assert select_common_block_size(640, [DeepseekV32IndexerBackend]) == 128


def test_fallback_without_config_returns_no_pooling_sizes(
    monkeypatch: pytest.MonkeyPatch,
):
    """Without ambient config the method reports no-pooling sizes."""
    _mock_rocm_platform(monkeypatch)
    assert _kv_pool_tokens_per_state(None) is None
    _assert_no_pooling_declaration(
        DeepseekV32IndexerBackend.get_supported_kernel_block_sizes()
    )


def test_selection_equals_storage_for_every_legal_manager_block(
    monkeypatch: pytest.MonkeyPatch,
):
    """Select matches storage for every block Glm5NextIndexerCache accepts.

    ``Glm5NextIndexerCache`` requires the manager block to be a multiple of
    ``index_kpool * 32`` at startup, so enumerate all such blocks up to 64K
    instead of sampling TP-derived values: any TP/DP/EP/PP configuration feeds
    one of them through the hybrid floor or ``--block-size``. Each block also
    checks that the zero-arg declaration matches the spec-fed declaration.
    """
    _mock_rocm_platform(monkeypatch)
    vllm_config = VllmConfig()
    # worker-shaped ambient (a serving VllmConfig always carries the model):
    vllm_config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(index_kpool=INDEX_KPOOL)
    )
    with set_current_vllm_config(vllm_config):
        for block_size in range(128, 65537, 128):
            cache_config = SimpleNamespace(
                block_size=block_size,
                gpu_memory_utilization=0.9,
                cache_dtype="auto",
            )
            cache = Glm5NextIndexerCache(
                head_dim=128,
                dtype=torch.bfloat16,
                prefix=f"model.layers.{block_size}.indexer.k_cache",
                cache_config=cache_config,
                index_kpool=INDEX_KPOOL,
            )
            spec = cache.get_kv_cache_spec(vllm_config)
            assert isinstance(spec, MLAAttentionSpec)
            sizes = [
                s.base if isinstance(s, MultipleOf) else s
                for s in DeepseekV32IndexerBackend.get_supported_kernel_block_sizes(
                    spec
                )
            ]
            chosen = [s for s in sizes if block_size % s == 0]
            assert chosen, (block_size, sizes)
            assert max(chosen) == spec.storage_block_size, (
                block_size,
                max(chosen),
                spec.storage_block_size,
            )
            # ambient-config fallback (the zero-arg connector path) agrees
            assert DeepseekV32IndexerBackend.get_supported_kernel_block_sizes() == (
                DeepseekV32IndexerBackend.get_supported_kernel_block_sizes(spec)
            )


def test_prepare_kernel_block_sizes_forwards_group_spec(
    monkeypatch: pytest.MonkeyPatch,
):
    """The worker's startup entry passes the group spec into select.

    ``prepare_kernel_block_sizes`` is the one production caller that opts into
    spec forwarding; if it regresses to the zero-argument call, the zero-arg
    declaration returns the no-pooling list (no ambient config here) and the
    selection silently collapses back to the manager block.
    """
    _mock_rocm_platform(monkeypatch)
    spec = _kpool_storage_spec(640)
    assert spec.has_layer_views
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]
    )
    attn_groups = [[SimpleNamespace(backend=DeepseekV32IndexerBackend)]]

    # Behavior: the group selects its storage block, not the 640 manager block.
    assert prepare_kernel_block_sizes(kv_cache_config, attn_groups) == [
        spec.storage_block_size
    ]

    # Wiring: the exact spec object reaches select_common_block_size.
    seen = {}
    real_select = worker_utils_mod.select_common_block_size

    def _spy(manager_block, backends, kv_cache_spec=None):
        seen["spec"] = kv_cache_spec
        return real_select(manager_block, backends, kv_cache_spec)

    monkeypatch.setattr(worker_utils_mod, "select_common_block_size", _spy)
    prepare_kernel_block_sizes(kv_cache_config, attn_groups)
    assert seen["spec"] is spec


def test_block_table_width_spans_every_storage_page(
    monkeypatch: pytest.MonkeyPatch,
):
    """Observable symptom of #58858: the table must hold a column for every
    pool page of the longest request.

    The writer addresses ``column = pos // storage_block_size``; with
    manager-granular kernel selection ``get_block_table_width`` returns one
    column per manager block (16 for 32K ctx at manager 2176) while 32K of
    storage pages needs 256 columns - the aliasing of the bug report.
    """
    _mock_rocm_platform(monkeypatch)
    manager_block, max_len = 2176, 32768
    spec = _kpool_storage_spec(manager_block)
    kernel = select_common_block_size(manager_block, [DeepseekV32IndexerBackend], spec)
    max_num_blocks = spec.max_num_blocks_per_req(VllmConfig(), max_len)
    width = get_block_table_width(max_num_blocks, manager_block, kernel)

    max_column_needed = (max_len - 1) // spec.storage_block_size
    assert max_column_needed < width, (
        f"table width {width} cannot host storage column {max_column_needed}"
    )
    # The hybrid split the BlockTable uses must stay integral.
    assert manager_block % kernel == 0
