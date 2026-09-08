# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Padded-page handling in create_kv_cache_views.

Guards that a page_size_padded spec strides the block dimension by the padded page
while keeping per-block content compact, so padding bytes at the end of each page are
never addressed by the logical view.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport, MultipleOf
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.hisparse.binding import allocate_hisparse_kv_caches
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MLAAttentionSpec,
    SparseCacheRole,
    compute_layout_strides,
)
from vllm.v1.worker.gpu import attn_utils
from vllm.v1.worker.gpu.attn_utils import (
    get_attn_cg_support,
    get_query_lens_mismatch_unsupported_backend,
)
from vllm.v1.worker.utils import (
    AttentionGroup,
    allocate_kv_cache,
    copy_kv_cache_blocks_inplace,
)


@pytest.mark.parametrize(
    ("enabled", "block_size", "main_sizes", "indexer_sizes", "expected"),
    [
        (True, 256, [64], [64], 64),
        (True, 64, [32, 64], [16, 32], 32),
        (True, 64, [MultipleOf(16)], [32], 32),
        (True, 64, [64], [32], None),
        (False, 256, [64], [64], 256),
    ],
)
def test_get_kv_cache_spec_resolves_hisparse_block_size(
    monkeypatch, enabled, block_size, main_sizes, indexer_sizes, expected
):
    """Resolve shared MLA geometry before planning; leave other specs alone."""
    specs = {
        "main": MLAAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
        ),
        "indexer": MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_role=SparseCacheRole.INDEXER,
        ),
        "dense": FullAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
        ),
    }
    layers = {}
    for name, sizes in zip(specs, [main_sizes, indexer_sizes, [block_size]]):
        backend = SimpleNamespace(
            customize_spec=AttentionBackend.customize_spec,
            get_supported_kernel_block_sizes=lambda sizes=sizes: sizes,
        )
        layers[name] = SimpleNamespace(
            get_kv_cache_spec=lambda _, spec=specs[name]: spec,
            get_attn_backend=lambda backend=backend: backend,
        )
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_: layers)
    config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=object() if enabled else None)
    )
    if expected is None:
        with pytest.raises(ValueError, match="supported by every sparse"):
            attn_utils.get_kv_cache_spec(config)
        return

    resolved = attn_utils.get_kv_cache_spec(config)
    assert resolved["main"].block_size == resolved["indexer"].block_size == expected
    assert resolved["dense"] is specs["dense"]
    assert all(spec.block_size == block_size for spec in specs.values())


class _FakeMetadataBuilder:
    def __init__(self, support: AttentionCGSupport):
        self.support = support

    def get_cudagraph_support(self, *_args):
        return self.support


class _TargetBackend:
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return True


class _DraftBackend:
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False


def test_attention_checks_preserve_global_and_target_scoped_support():
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    target_group = AttentionGroup(
        _TargetBackend,
        ["target"],
        spec,
        0,  # type: ignore[arg-type]
    )
    target_group.metadata_builders = [
        _FakeMetadataBuilder(AttentionCGSupport.ALWAYS)  # type: ignore[list-item]
    ]
    draft_group = AttentionGroup(
        _DraftBackend,
        ["draft"],
        spec,
        0,  # type: ignore[arg-type]
    )
    draft_group.metadata_builders = [
        _FakeMetadataBuilder(AttentionCGSupport.UNIFORM_BATCH)  # type: ignore[list-item]
    ]
    groups = [[target_group, draft_group]]

    # The runner-wide execution mode must still honor the drafter's limit.
    unfiltered = get_attn_cg_support(groups, None)  # type: ignore[arg-type]
    assert unfiltered.min_cg_support == AttentionCGSupport.UNIFORM_BATCH
    assert unfiltered.min_cg_attn_backend == "_DraftBackend"

    # Adaptive verification validates only the target's varlen graphs.
    target_only = get_attn_cg_support(
        groups,
        None,  # type: ignore[arg-type]
        checked_layer_names={"target"},
    )
    assert target_only.min_cg_support == AttentionCGSupport.ALWAYS
    assert target_only.min_cg_attn_backend is None
    assert (
        get_query_lens_mismatch_unsupported_backend(
            groups,
            checked_layer_names={"target"},
        )
        is None
    )

    # Shared target/draft groups still participate in target-scoped checks.
    draft_group.layer_names.append("target")
    target_with_shared_group = get_attn_cg_support(
        groups,
        None,  # type: ignore[arg-type]
        checked_layer_names={"target"},
    )
    assert target_with_shared_group.min_cg_support == AttentionCGSupport.UNIFORM_BATCH
    assert (
        get_query_lens_mismatch_unsupported_backend(
            groups,
            checked_layer_names={"target"},
        )
        == "_DraftBackend"
    )


def test_get_kv_sharing_fast_prefill_eligible_layers(monkeypatch: pytest.MonkeyPatch):
    """Fast prefill applies to the contiguous suffix of KV-sharing layers.

    Draft-model layers register after the target model's and may share KV, so
    they must not extend (or break) the target's eligible suffix.
    """

    def check(
        layer_names: list[str],
        shared: dict[str, str],
        draft_layer_names: set[str] | None = None,
    ) -> set[str]:
        monkeypatch.setattr(
            attn_utils,
            "get_layers_from_vllm_config",
            lambda *a, **k: {name: None for name in layer_names},
        )
        monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda *a: shared)
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(kv_sharing_fast_prefill=True)
        )
        return attn_utils.get_kv_sharing_fast_prefill_eligible_layers(
            vllm_config, draft_layer_names
        )

    # No KV sharing: nothing is eligible.
    assert check(["t0", "t1"], {}) == set()

    # Trailing run of sharing layers (YOCO-style second half).
    assert check(["t0", "t1", "t2", "t3"], {"t2": "t1", "t3": "t1"}) == {"t2", "t3"}

    # A non-sharing layer after a sharing one breaks the suffix.
    assert check(["t0", "t1", "t2", "t3"], {"t1": "t0", "t3": "t0"}) == {"t3"}

    # KV-sharing draft layers at the end are collected without an exclusion...
    assert check(
        ["t0", "t1", "t2", "t3", "d0", "d1"],
        {"t2": "t1", "t3": "t1", "d0": "t1", "d1": "t1"},
    ) == {"t2", "t3", "d0", "d1"}

    # ...so the runner excludes them: skipped, not collected, and they do not
    # break the target's trailing run.
    assert check(
        ["t0", "t1", "t2", "t3", "d0", "d1"],
        {"t2": "t1", "t3": "t1", "d0": "t1", "d1": "t1"},
        draft_layer_names={"d0", "d1"},
    ) == {"t2", "t3"}

    # Feature flag off: nothing is eligible even with sharing layers.
    monkeypatch.setattr(
        attn_utils, "get_layers_from_vllm_config", lambda *a, **k: {"t0": None}
    )
    monkeypatch.setattr(
        attn_utils, "get_shared_kv_cache_layers", lambda *a: {"t0": "t0"}
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False)
    )
    assert attn_utils.get_kv_sharing_fast_prefill_eligible_layers(vllm_config) == set()


def test_reshape_padded_kv_cache_strides_by_padded_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 256

    raw = torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    (kv_cache,) = dense_kv_cache_views(raw, spec, num_blocks, 1, KVCacheLayout.LBHNC)

    elem_size = 4  # float32
    # Content dim packs K and V: 2 * head_size.
    assert kv_cache.shape == (num_blocks, 1, 16, 2 * spec.head_size)
    assert kv_cache.dtype == spec.dtype
    assert kv_cache.stride(0) == spec.page_size_padded // elem_size
    assert kv_cache[1].storage_offset() == spec.page_size_padded // elem_size
    # Within one block the (unpadded) content stays compact.
    assert kv_cache[0].is_contiguous()


@pytest.mark.parametrize(
    (
        "kernel_block_sizes",
        "storage_block_size",
        "expected_num_blocks",
        "expected_num_states",
    ),
    [
        (None, None, 4, 64),
        ([256], None, 4, 64),
        ([64], None, 16, 16),
        ([64], 256, 4, 64),
    ],
)
def test_allocate_compressed_mla_cache(
    kernel_block_sizes: list[int] | None,
    storage_block_size: int | None,
    expected_num_blocks: int,
    expected_num_states: int,
):
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
        storage_block_size=storage_block_size,
    )
    num_pages = 4
    config = KVCacheConfig(
        num_blocks=num_pages,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_pages * spec.page_size_bytes,
                layers=["layer.0"],
                layer_stride=num_pages * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )

    caches = allocate_kv_cache(
        config, torch.device("cpu"), KVCacheLayout.LBHNC, kernel_block_sizes
    )

    assert caches["layer.0"].shape == (expected_num_blocks, 1, expected_num_states, 128)


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_copy_kv_cache_blocks_shared_storage(layout: KVCacheLayout):
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(num_blocks):
            cache[block_idx].fill_(10 * layer_idx + block_idx)

    expected = [[cache[i].clone() for i in range(num_blocks)] for cache in caches]
    copies = [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)]

    copy_kv_cache_blocks_inplace(caches, num_blocks, copies)

    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[2], expected[layer_idx][0])
        torch.testing.assert_close(cache[1], expected[layer_idx][1])


def test_fixed_block_stride_propagates_outward_in_lhbnc():
    num_blocks = 3
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
    )
    natural = compute_layout_strides(spec, num_blocks, num_layers, KVCacheLayout.LHBNC)
    block_stride = natural[1] + 8

    strides = compute_layout_strides(
        spec,
        num_blocks,
        num_layers,
        KVCacheLayout.LHBNC,
        fixed_strides=(None, block_stride, None, None, None),
    )

    assert strides[1] == block_stride
    assert strides[2] == block_stride * num_blocks
    assert strides[0] == strides[2] * spec.num_heads


def test_copy_kv_cache_blocks_separate_head_groups():
    # LHBNC stores each head group separately, so a block's bytes are scattered
    # across L*H regions.
    layout = KVCacheLayout.LHBNC
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
        num_head_slots=2,
        state_content_bytes=2 * 2 * 4,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(num_blocks):
            for head_idx in range(cache.shape[1]):
                cache[block_idx, head_idx].fill_(
                    100 * layer_idx + 10 * head_idx + block_idx
                )

    expected = [[cache[i].clone() for i in range(num_blocks)] for cache in caches]
    copy_kv_cache_blocks_inplace(
        caches,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[2], expected[layer_idx][0])
        torch.testing.assert_close(cache[1], expected[layer_idx][1])


@pytest.mark.parametrize(
    "layout,num_layers",
    [
        (KVCacheLayout.LBHNC, 2),
        # Splitting needs a manager block to be one dense page, which a
        # block-outermost layout only gives when the block holds one layer.
        (KVCacheLayout.BLHNC, 1),
    ],
)
def test_copy_kv_cache_blocks_with_virtual_block_splitting(
    layout: KVCacheLayout, num_layers: int
):
    num_blocks = 4
    physical_per_logical = 2
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(
        raw,
        spec,
        num_blocks,
        num_layers,
        layout,
        kernel_block_size=spec.block_size // physical_per_logical,
    )

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(cache.shape[0]):
            cache[block_idx].fill_(100 * layer_idx + block_idx)
    expected = [[cache[i].clone() for i in range(cache.shape[0])] for cache in caches]

    copy_kv_cache_blocks_inplace(
        caches,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    dst_start = 2 * physical_per_logical
    for layer_idx, cache in enumerate(caches):
        for physical_idx in range(physical_per_logical):
            torch.testing.assert_close(
                cache[dst_start + physical_idx], expected[layer_idx][physical_idx]
            )


def test_allocate_hisparse_kv_caches_host_pool_and_view_less_specs():
    """Host tensors get their own backing; view-less specs keep the raw one."""
    spec = FullAttentionSpec(
        block_size=2, num_kv_heads=1, head_size=4, dtype=torch.float32
    )
    page = spec.page_size_bytes
    resident_spec = HiSparseResidentSpec(block_size=2, page_size=page)
    device_size = 4 * page
    config = KVCacheConfig(
        num_blocks=4,
        hisparse_host_num_blocks=3,
        kv_cache_tensors=[
            KVCacheTensor(
                size=3 * page,
                layers=["source"],
                layer_stride=3 * page,
                block_stride=page,
                host_resident=True,
            ),
            KVCacheTensor(
                size=device_size,
                layers=["indexer"],
                layer_stride=device_size,
                block_stride=page,
            ),
            KVCacheTensor(
                size=device_size,
                layers=["resident"],
                layer_stride=device_size,
                block_stride=page,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["source"], spec, host_resident=True),
            KVCacheGroupSpec(["indexer"], spec),
            KVCacheGroupSpec(["resident"], resident_spec),
        ],
    )
    host_buffers: list[torch.Tensor] = []

    def host_allocator(size: int) -> torch.Tensor:
        host_buffers.append(torch.zeros(size, dtype=torch.int8))
        return host_buffers[-1]

    caches = allocate_hisparse_kv_caches(
        config,
        torch.device("cpu"),
        KVCacheLayout.LBHNC,
        [2, 2, 2],
        SimpleNamespace(allocate=host_allocator),
    )
    assert len(config.kv_cache_tensors) == 3

    assert [buf.numel() for buf in host_buffers] == [3 * page]
    assert caches["source"].shape[0] == 3
    assert (
        caches["source"].untyped_storage().data_ptr()
        == host_buffers[0].untyped_storage().data_ptr()
    )
    assert caches["indexer"].shape[0] == 4
    backing = caches["resident"]
    assert backing.dtype == torch.int8 and backing.numel() >= device_size
    assert (
        backing.untyped_storage().data_ptr()
        == caches["indexer"].untyped_storage().data_ptr()
    )
