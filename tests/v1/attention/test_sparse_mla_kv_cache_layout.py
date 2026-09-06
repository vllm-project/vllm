# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V3.2 sparse MLA layouts and physical row addressing (issue #55431)."""

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.config import CacheConfig
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
    FlashInferMLASparseTRTLLMBackend,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import (
    FlashInferMLASparseSM90Backend,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    DeepseekV32IndexerBackend,
    KpoolTailBackend,
)
from vllm.v1.attention.backends.utils import (
    get_supported_kv_cache_layouts,
    resolve_kv_cache_layout,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

V32_SPARSE_BACKENDS = (
    FlashMLASparseBackend,
    FlashInferMLASparseTRTLLMBackend,
    FlashInferMLASparseSM120Backend,
    DeepseekV32IndexerBackend,
)

BLOCK_OUTERMOST = {
    KVCacheLayout.BLHNC,
    KVCacheLayout.BLNHC,
    KVCacheLayout.BHLNC,
}

pytestmark = pytest.mark.skip_global_cleanup


def _mixed_page_specs(cache_dtype="auto", backend=FlashInferMLASparseTRTLLMBackend):
    config = SimpleNamespace(
        cache_config=CacheConfig(block_size=64),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    # Exercise the production spec factories without constructing model weights.
    mla = SimpleNamespace(
        kv_cache_dtype=cache_dtype,
        head_size=576,
        sliding_window=None,
        attn_backend=backend,
        non_causal_multi_token_decode=False,
    )
    indexer = SimpleNamespace(
        cache_config=config.cache_config,
        head_dim=128 + 4,
        dtype=torch.uint8,
    )
    return [
        MLAAttention.get_kv_cache_spec(mla, config),
        DeepseekV32IndexerCache.get_kv_cache_spec(indexer, config),
    ]


def _resolve(backends, specs, monkeypatch, requested=None) -> KVCacheLayout:
    supported = [layout.name for layout in get_supported_kv_cache_layouts(backends)]
    monkeypatch.setattr(envs, "VLLM_KV_CACHE_LAYOUT", requested)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_layout=None),
        kv_transfer_config=None,
    )
    return resolve_kv_cache_layout(vllm_config, [supported], specs)


@pytest.mark.parametrize("backend", V32_SPARSE_BACKENDS)
def test_backends_declare_only_validated_packed_layout(backend):
    declared = backend.supported_kv_cache_layouts()
    assert declared is not None
    assert BLOCK_OUTERMOST.intersection(declared) == {KVCacheLayout.BLHNC}
    assert all(layout.is_layer_compact for layout in declared[:-1])


def test_combined_supported_set_preserves_default_preference():
    supported = get_supported_kv_cache_layouts(list(V32_SPARSE_BACKENDS))
    assert BLOCK_OUTERMOST.intersection(supported) == {KVCacheLayout.BLHNC}
    assert supported[0] is KVCacheLayout.LBNHC


@pytest.mark.parametrize("name", ["BLNHC", "BHLNC"])
def test_unadvertised_block_outermost_fails_fast(monkeypatch, name):
    with pytest.raises(ValueError, match=name):
        _resolve(
            [FlashInferMLASparseTRTLLMBackend, DeepseekV32IndexerBackend],
            _mixed_page_specs(),
            monkeypatch,
            requested=name,
        )


def test_default_resolution_unchanged(monkeypatch):
    layout = _resolve(
        [FlashInferMLASparseTRTLLMBackend, DeepseekV32IndexerBackend],
        _mixed_page_specs(),
        monkeypatch,
    )
    assert layout is KVCacheLayout.LBNHC


@pytest.mark.parametrize("name", ["LBNHC", "LBHNC", "BLHNC"])
def test_explicit_supported_layout_accepted(monkeypatch, name):
    layout = _resolve(
        [FlashInferMLASparseTRTLLMBackend, DeepseekV32IndexerBackend],
        _mixed_page_specs(),
        monkeypatch,
        requested=name,
    )
    assert layout.name == name


def test_sm90_pairing_still_resolves_lbhnc():
    supported = get_supported_kv_cache_layouts(
        [FlashInferMLASparseSM90Backend, DeepseekV32IndexerBackend]
    )
    assert supported == [KVCacheLayout.LBHNC]


def test_packing_layout_backends_unaffected():
    assert DeepseekV4IndexerBackend.supported_kv_cache_layouts() == (
        KVCacheLayout.BLHNC,
        KVCacheLayout.BLNHC,
    )
    assert KpoolTailBackend.supported_kv_cache_layouts() == (KVCacheLayout.LBHNC,)


@pytest.mark.parametrize(
    "cache_dtype,row_bytes",
    [
        ("fp8_ds_mla", 656),
        ("nvfp4_ds_mla", 352),
        ("auto", 1152),
        ("fp8", 576),
    ],
)
def test_sparse_spec_factories_require_physical_row_alignment(cache_dtype, row_bytes):
    mla, indexer = _mixed_page_specs(cache_dtype)
    assert mla.state_content_size_bytes == row_bytes
    assert mla.block_stride_alignment_bytes == row_bytes
    assert indexer.state_content_size_bytes == 132
    assert indexer.block_stride_alignment_bytes == 132
    for spec in (mla, indexer):
        assert spec.page_size_padded is None
        assert spec.page_size_bytes == 64 * spec.state_content_size_bytes


def test_dense_mla_does_not_request_packed_alignment():
    from vllm.v1.attention.backend import AttentionBackend

    mla, _ = _mixed_page_specs(backend=AttentionBackend)
    assert mla.block_stride_alignment_bytes is None


@pytest.mark.parametrize("cache_dtype", ["auto", "fp8", "fp8_ds_mla", "nvfp4_ds_mla"])
@pytest.mark.parametrize("warmup", [False, True])
def test_bind_packed_cache_warmup_uses_physical_rows(monkeypatch, cache_dtype, warmup):
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.triton_utils import triton
    from vllm.utils.math_utils import next_power_of_2
    from vllm.v1.attention.backends.mla.sparse_utils import (
        _CONVERT_REQ_INDEX_TO_GLOBAL_INDEX_KERNEL as kernel,
    )
    from vllm.v1.attention.backends.mla.sparse_utils import (
        ConvertReqIndexToGlobalIndexKernel,
        flat_kv_row_view,
    )
    from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs
    from vllm.v1.worker.utils import allocate_kv_cache

    mla, indexer = _mixed_page_specs(cache_dtype)
    config = SimpleNamespace(
        cache_config=CacheConfig(block_size=64),
        kernel_config=SimpleNamespace(enable_jit_warmup=warmup),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        model_config=SimpleNamespace(
            max_model_len=512, hf_config=SimpleNamespace(index_topk=128)
        ),
    )
    config.cache_config.kv_cache_layout = "BLHNC"
    group = KVCacheGroupSpec(
        ["mla", "indexer"],
        UniformTypeKVCacheSpecs(
            block_size=64, kv_cache_specs={"mla": mla, "indexer": indexer}
        ),
    )
    cache_config = get_kv_cache_config_from_groups(config, [group], 2**20)
    view = allocate_kv_cache(cache_config, torch.device("cpu"), KVCacheLayout.BLHNC)[
        "mla"
    ]
    layer = SimpleNamespace(_vllm_config=config, attn_backend=FlashMLASparseBackend)
    registry = JitWarmupRegistry(config)
    keys: list[ConvertReqIndexToGlobalIndexKernel.CompileKey] = []
    # Expand real registration and dispatch keys; CUDA compilation is tested on GPU.
    monkeypatch.setattr(kernel, "compile", keys.append)
    # CPU builds use a Triton placeholder; key enumeration only needs this helper.
    monkeypatch.setattr(triton, "next_power_of_2", next_power_of_2, raising=False)
    monkeypatch.setattr("vllm.distributed.is_global_first_rank", lambda: False)
    with registry.activate():
        MLAAttention.bind_kv_cache(layer, view)
    registry.warmup()
    rows, stride_rows = flat_kv_row_view(layer.kv_cache, 64)
    assert rows.data_ptr() == view.data_ptr()
    assert (
        stride_rows * mla.state_content_size_bytes
        == cache_config.kv_cache_tensors[0].block_stride
    )
    if warmup:
        assert keys
        assert {key.block_stride_rows for key in keys} == {stride_rows}
    else:
        assert not keys


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("cache_dtype", ["auto", "fp8", "fp8_ds_mla", "nvfp4_ds_mla"])
@pytest.mark.parametrize("warmup", [False, True])
def test_cuda_packed_sparse_binding_and_index_conversion(
    dist_init, cache_dtype, warmup
):
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.v1.attention.backends.mla.sparse_utils import (
        flat_kv_row_view,
        triton_convert_req_index_to_global_index,
    )
    from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs
    from vllm.v1.worker.utils import allocate_kv_cache

    mla, indexer = _mixed_page_specs(cache_dtype)
    config = SimpleNamespace(
        cache_config=CacheConfig(block_size=64),
        kernel_config=SimpleNamespace(enable_jit_warmup=warmup),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, cp_kv_cache_interleave_size=1
        ),
        model_config=SimpleNamespace(
            max_model_len=256, hf_config=SimpleNamespace(index_topk=128)
        ),
    )
    config.cache_config.kv_cache_layout = "BLHNC"
    config.cache_config.num_gpu_blocks_override = 11
    specs = {"mla": mla, "indexer": indexer}
    group = KVCacheGroupSpec(
        list(specs), UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=specs)
    )
    cache = get_kv_cache_config_from_groups(config, [group], 0)
    views = allocate_kv_cache(cache, torch.device("cuda"), KVCacheLayout.BLHNC)
    registry = JitWarmupRegistry(config)
    with registry.activate():
        for view in views.values():
            layer = SimpleNamespace(
                _vllm_config=config, attn_backend=FlashMLASparseBackend
            )
            MLAAttention.bind_kv_cache(layer, view)
    assert bool(len(registry)) == warmup
    registry.warmup()  # Real Triton compilation, with no compile/FFI mock.

    blocks = [9, 2, 7, 1]
    tokens = [0, 63, 64, 127, 191, 255] + [-1] * 122
    table = torch.tensor([blocks], dtype=torch.int32, device="cuda")
    indices = torch.tensor([tokens], dtype=torch.int32, device="cuda")
    req_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    for view in views.values():
        _, stride_rows = flat_kv_row_view(view.squeeze(1), 64)
        expected = torch.tensor(
            [
                [
                    blocks[t // 64] * stride_rows + t % 64 if t >= 0 else -1
                    for t in tokens
                ]
            ],
            dtype=torch.int32,
            device="cuda",
        )
        actual = triton_convert_req_index_to_global_index(
            req_ids, table, indices, BLOCK_STRIDE_ROWS=stride_rows, NUM_TOPK_TOKENS=128
        )
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_cuda_sparse_rows_and_cache_offsets_above_two_gib():
    from vllm import _custom_ops as ops
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_convert_req_index_to_global_index,
    )
    from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs
    from vllm.v1.worker.utils import allocate_kv_cache

    config = SimpleNamespace(cache_config=CacheConfig(block_size=64))
    config.cache_config.kv_cache_layout = "BLHNC"
    config.cache_config.num_gpu_blocks_override = 20307
    req_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    table = torch.tensor([[20306, 7, 20101, 3]], dtype=torch.int32, device="cuda")
    tokens = [0, 63, 64, 127, 191, 255] + [-1] * 122
    indices = torch.tensor([tokens], dtype=torch.int32, device="cuda")

    # Convert production-scale coordinates without allocating the full inventory.
    for cache_dtype in ("auto", "fp8", "fp8_ds_mla", "nvfp4_ds_mla"):
        mla, indexer = _mixed_page_specs(cache_dtype)
        specs = {f"mla.{i}": mla for i in range(100)} | {"indexer": indexer}
        group = KVCacheGroupSpec(
            list(specs), UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=specs)
        )
        cache = get_kv_cache_config_from_groups(config, [group], 0)
        stride = cache.kv_cache_tensors[0].block_stride
        for spec in (mla, indexer):
            stride_rows = stride // spec.state_content_size_bytes
            expected = [
                [20306, 7, 20101, 3][t // 64] * stride_rows + t % 64 if t >= 0 else -1
                for t in tokens
            ]
            assert max(expected) < torch.iinfo(torch.int32).max
            assert max(expected) * spec.state_content_size_bytes > 2**31
            actual = triton_convert_req_index_to_global_index(
                req_ids,
                table,
                indices,
                BLOCK_STRIDE_ROWS=stride_rows,
                NUM_TOPK_TOKENS=128,
            )
            torch.testing.assert_close(
                actual, torch.tensor([expected], dtype=torch.int32, device="cuda")
            )

    # Separately dereference >2 GiB offsets through native cache write AND gather
    # kernels, using a small (~2 GiB) real allocation rather than a 101-layer pool.
    mla, indexer = _mixed_page_specs("auto")
    specs = {"mla": mla, "indexer": indexer}
    group = KVCacheGroupSpec(
        list(specs), UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=specs)
    )
    cache = get_kv_cache_config_from_groups(config, [group], 0)
    stride = cache.kv_cache_tensors[0].block_stride
    config.cache_config.num_gpu_blocks_override = 2**31 // stride + 2
    cache = get_kv_cache_config_from_groups(config, [group], 0)
    # All layer views share this allocation; leave 10% for scratch/allocator overhead.
    required_free_bytes = cache.kv_cache_tensors[0].size * 11 // 10
    if torch.accelerator.get_memory_info()[0] < required_free_bytes:
        pytest.skip(
            f"Requires approximately {required_free_bytes / 2**30:.2f} GiB of free "
            "CUDA memory (including 10% headroom)"
        )
    views = allocate_kv_cache(cache, torch.device("cuda"), KVCacheLayout.BLHNC)
    view = views["mla"].squeeze(1)
    last = cache.num_blocks - 1
    assert last * stride > 2**31
    values = torch.arange(64 * 576, device="cuda", dtype=torch.float32)
    values = (values.remainder(97) / 32).to(torch.bfloat16).view(64, 576)
    slots = torch.arange(last * 64, (last + 1) * 64, device="cuda", dtype=torch.int64)
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    ops.concat_and_cache_mla(
        values[:, :512],
        values[:, 512:],
        view,
        slots,
        kv_cache_dtype="auto",
        scale=scale,
    )
    gathered = torch.empty_like(values)
    ops.gather_and_maybe_dequant_cache(
        view,
        gathered,
        torch.tensor([[last]], dtype=torch.int32, device="cuda"),
        torch.tensor([0, 64], dtype=torch.int32, device="cuda"),
        torch.zeros(64, dtype=torch.int32, device="cuda"),
        64,
        "auto",
        scale,
    )
    torch.testing.assert_close(gathered, values, rtol=0, atol=0)


def test_trtllm_native_32_row_packed_view():
    """Standalone TRTLLM supports 32 rows; the CUDA V3.2 indexer requires 64."""
    from dataclasses import replace

    from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
    from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec
    from vllm.v1.worker.utils import allocate_kv_cache

    assert 32 in FlashInferMLASparseTRTLLMBackend.get_supported_kernel_block_sizes()
    spec = replace(_mixed_page_specs()[0], block_size=32)
    config = SimpleNamespace(cache_config=CacheConfig(block_size=32))
    config.cache_config.kv_cache_layout = "BLHNC"
    config.cache_config.num_gpu_blocks_override = 3
    cache = get_kv_cache_config_from_groups(
        config, [KVCacheGroupSpec(["mla.0", "mla.1"], spec)], 0
    )
    views = allocate_kv_cache(cache, torch.device("cpu"), KVCacheLayout.BLHNC, [32])
    views["mla.0"].fill_(1)
    views["mla.1"].fill_(2)
    view = views["mla.0"].squeeze(1)
    rows, stride_rows = flat_kv_row_view(view, 32)
    assert stride_rows == 64
    assert rows.data_ptr() == views["mla.0"].data_ptr()
    assert rows[stride_rows].data_ptr() == views["mla.0"][1].data_ptr()
    torch.testing.assert_close(rows[stride_rows : stride_rows + 32], view[1])
    assert torch.all(views["mla.1"] == 2)
