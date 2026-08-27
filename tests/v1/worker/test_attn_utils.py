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

import vllm.v1.worker.gpu.attn_utils as attn_utils_module
from tests.v1.attention.utils import dense_kv_cache_views
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MLAAttentionSpec,
    compute_layout_strides,
)
from vllm.v1.worker.gpu.attn_utils import (
    get_attn_cg_support,
    get_query_lens_mismatch_unsupported_backend,
)
from vllm.v1.worker.utils import (
    AttentionGroup,
    allocate_kv_cache,
    copy_kv_cache_blocks_inplace,
)


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


class _FakeSharedHostRegion:
    def __init__(self) -> None:
        self.cleanup_calls = 0
        self.base_tensor = torch.empty(1, dtype=torch.int8)

    def cleanup(self) -> None:
        self.cleanup_calls += 1


def test_profiling_cleanup_releases_tp_shared_region_once(monkeypatch):
    """TP-shared profiling pools must use region-aware chunk cleanup."""
    region = _FakeSharedHostRegion()
    runtime = SimpleNamespace(
        _host_cache=object(),
        registered_host_pool=region.base_tensor,
        hot_backing=object(),
        shared_host_region=region,
    )
    forward_context = {
        "layer": SimpleNamespace(
            hisparse_cache=SimpleNamespace(runtime=runtime),
        )
    }
    released = []

    def release_pinned_state(runtimes, pinned_host_pools, shared_host_region):
        released.append((runtimes, pinned_host_pools, shared_host_region))

    monkeypatch.setattr(
        attn_utils_module,
        "release_pinned_state",
        release_pinned_state,
    )

    attn_utils_module.release_hisparse_profiling_cache(forward_context)

    assert released == [([runtime], [], region)]


def test_allocate_hisparse_kv_cache_rolls_back_on_device_failure(monkeypatch):
    region = _FakeSharedHostRegion()
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    page_size = spec.page_size_bytes
    host_group = KVCacheGroupSpec(
        ["host"],
        spec,
        block_pool_id=None,
        role=KVCacheGroupRole.HISPARSE_SOURCE,
    )
    device_group = KVCacheGroupSpec(["device"], spec)
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[
            KVCacheTensor(
                size=page_size,
                layers=["host"],
                layer_stride=page_size,
                block_stride=page_size,
                host_resident=True,
                block_pool_id=None,
            ),
            KVCacheTensor(
                size=page_size,
                layers=["device"],
                layer_stride=page_size,
                block_stride=page_size,
            ),
        ],
        kv_cache_groups=[host_group, device_group],
        hisparse_host_num_blocks=1,
        hisparse_host_block_stride=4096,
        hisparse_shared_host_pool=True,
    )
    host_tensor = torch.empty(page_size, dtype=torch.int8)
    monkeypatch.setattr(
        attn_utils_module,
        "allocate_hisparse_host_pools",
        lambda *args, **kwargs: ([host_tensor], [], region),
    )

    def fail_device_allocation(*args, **kwargs):
        raise RuntimeError("device allocation failed")

    monkeypatch.setattr(attn_utils_module.torch, "zeros", fail_device_allocation)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC
        )
    )

    with pytest.raises(RuntimeError, match="device allocation failed"):
        attn_utils_module._allocate_hisparse_kv_cache(
            kv_cache_config,
            torch.device("cpu"),
            [spec.block_size, spec.block_size],
            vllm_config,
        )

    assert region.cleanup_calls == 1


def test_init_kv_cache_rolls_back_shared_region_on_bind_failure(monkeypatch):
    region = _FakeSharedHostRegion()
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=SimpleNamespace()),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )
    monkeypatch.setattr(
        attn_utils_module,
        "_allocate_hisparse_kv_cache",
        lambda *args, **kwargs: ({}, {}, {}, region),
    )

    def fail_cache_binding(**kwargs):
        raise RuntimeError("cache binding failed")

    monkeypatch.setattr(
        attn_utils_module, "_bind_hisparse_kv_caches", fail_cache_binding
    )

    with pytest.raises(RuntimeError, match="cache binding failed"):
        attn_utils_module.init_kv_cache(
            runner_kv_caches=[],
            forward_context={},
            kv_cache_config=SimpleNamespace(),
            device=torch.device("cpu"),
            kernel_block_sizes=[],
            vllm_config=vllm_config,
            block_tables=SimpleNamespace(),
        )

    assert region.cleanup_calls == 1


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
