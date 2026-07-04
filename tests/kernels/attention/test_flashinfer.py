# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from itertools import product
from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    nvfp4_kv_cache_full_dim,
    nvfp4_kv_cache_split_views,
    set_random_seed,
)

try:
    import flashinfer
except ImportError:
    if current_platform.is_rocm():
        pytest.skip(
            "flashinfer is not supported for vLLM on ROCm.", allow_module_level=True
        )

import torch

from vllm.platforms.interface import DeviceCapability

NUM_HEADS = [(32, 8), (6, 1)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16, 32]
DTYPES = [torch.bfloat16]
NUM_BLOCKS = 32768  # Large enough to test overflow in index calculation.
SOFT_CAPS = [None, 30.0]
SLIDING_WINDOWS = [None, 64]


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def _make_paged_kv_metadata(
    kv_lens: list[int],
    block_size: int,
    num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build paged-KV metadata tensors for fast_plan_decode tests.

    Returns:
        kv_indptr          – CPU int32, shape [num_seqs + 1]
        kv_indices         – CUDA int32, shape [total_blocks]
        kv_last_page_lens  – CPU int32, shape [num_seqs]
        block_tables       – CUDA int32, shape [num_seqs, max_blocks_per_seq]
    """
    num_seqs = len(kv_lens)
    max_blocks = (max(kv_lens) + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_blocks), dtype=torch.int32, device="cuda"
    )

    indptr_list = [0]
    indices_list: list[int] = []
    last_lens_list: list[int] = []
    for i, seq_len in enumerate(kv_lens):
        n = (seq_len + block_size - 1) // block_size
        indices_list.extend(block_tables[i, :n].cpu().tolist())
        indptr_list.append(indptr_list[-1] + n)
        last_lens_list.append(seq_len % block_size or block_size)

    return (
        torch.tensor(indptr_list, dtype=torch.int32, device="cpu"),
        torch.tensor(indices_list, dtype=torch.int32, device="cuda"),
        torch.tensor(last_lens_list, dtype=torch.int32, device="cpu"),
        block_tables,
    )


def _make_cg_decode_wrapper(
    num_seqs: int,
    kv_indices_buffer: torch.Tensor,
    workspace_buffer: torch.Tensor,
    use_tensor_cores: bool = True,
) -> "flashinfer.BatchDecodeWithPagedKVCacheWrapper":
    """Create a cudagraph-enabled BatchDecodeWithPagedKVCacheWrapper.

    *kv_indices_buffer* is shared with the caller so that fast_plan_decode
    can avoid the device-to-device index copy on subsequent (cudagraph) calls.
    """
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        "NHD",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=torch.zeros(
            num_seqs + 1, dtype=torch.int32, device="cuda"
        ),
        paged_kv_indices_buffer=kv_indices_buffer,
        paged_kv_last_page_len_buffer=torch.zeros(
            num_seqs, dtype=torch.int32, device="cuda"
        ),
        use_tensor_cores=use_tensor_cores,
    )


def test_flashinfer_backend_accepts_nvfp4_kv_cache() -> None:
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    invalid_reasons = FlashInferBackend.validate_configuration(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype="nvfp4",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(8, 6),
        attn_type="decoder",
    )

    assert invalid_reasons == []


def _make_flashinfer_q_dtype_builder(
    *,
    cache_dtype: str,
    model_dtype: torch.dtype = torch.bfloat16,
    disable_q_quantization: bool = False,
):
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder.cache_dtype = cache_dtype
    builder.is_kvcache_nvfp4 = cache_dtype == "nvfp4"
    builder.kv_cache_dtype = model_dtype
    builder.kv_cache_spec = SimpleNamespace(dtype=model_dtype)
    builder.model_config = SimpleNamespace(dtype=model_dtype)
    builder.vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(
            disable_flashinfer_q_quantization=disable_q_quantization
        )
    )
    return builder


@pytest.mark.parametrize("is_prefill", [True, False])
def test_flashinfer_nvfp4_native_q_dtype_uses_model_dtype(
    is_prefill: bool,
) -> None:
    builder = _make_flashinfer_q_dtype_builder(cache_dtype="nvfp4")

    q_dtype = builder.get_q_data_type(
        is_prefill=is_prefill,
        use_trtllm_gen=False,
    )

    assert q_dtype == torch.bfloat16


@pytest.mark.parametrize("is_prefill", [True, False])
def test_flashinfer_nvfp4_trtllm_gen_q_dtype_uses_fp8(is_prefill: bool) -> None:
    from vllm.v1.attention.backends.flashinfer import FP8_DTYPE

    builder = _make_flashinfer_q_dtype_builder(cache_dtype="nvfp4")

    q_dtype = builder.get_q_data_type(
        is_prefill=is_prefill,
        use_trtllm_gen=True,
    )

    assert q_dtype == FP8_DTYPE


def test_flashinfer_q_quantization_disable_overrides_nvfp4_trtllm_gen() -> None:
    builder = _make_flashinfer_q_dtype_builder(
        cache_dtype="nvfp4",
        disable_q_quantization=True,
    )

    q_dtype = builder.get_q_data_type(
        is_prefill=False,
        use_trtllm_gen=True,
    )

    assert q_dtype == torch.bfloat16


def test_flashinfer_nvfp4_mixed_head_shape_uses_packed_layout() -> None:
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    shape = FlashInferBackend.get_kv_cache_shape(
        num_blocks=3,
        block_size=16,
        num_kv_heads=2,
        head_size=256,
        cache_dtype_str="nvfp4",
        head_size_v=512,
    )

    assert shape == (
        3,
        16,
        2,
        nvfp4_kv_cache_full_dim(256) + nvfp4_kv_cache_full_dim(512),
    )


def test_flashinfer_nvfp4_same_head_shape_keeps_stacked_layout() -> None:
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    shape = FlashInferBackend.get_kv_cache_shape(
        num_blocks=3,
        block_size=16,
        num_kv_heads=2,
        head_size=256,
        cache_dtype_str="nvfp4",
        head_size_v=256,
    )

    assert shape == (3, 2, 16, 2, nvfp4_kv_cache_full_dim(256))


def _storage_offsets(tensor: torch.Tensor) -> set[int]:
    return {
        tensor.storage_offset()
        + sum(idx * stride for idx, stride in zip(indices, tensor.stride()))
        for indices in product(*(range(size) for size in tensor.shape))
    }


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((2, 4, 3), id="NHD"),
        pytest.param((2, 3, 4), id="HND"),
    ],
)
def test_nvfp4_kv_cache_split_views_mixed_packed_layout(
    shape: tuple[int, int, int],
) -> None:
    head_size = 64
    head_size_v = 128
    k_full_dim = nvfp4_kv_cache_full_dim(head_size)
    v_full_dim = nvfp4_kv_cache_full_dim(head_size_v)
    num_pages, dim_1, dim_2 = shape
    kv_cache = torch.empty(
        num_pages, dim_1, dim_2, k_full_dim + v_full_dim, dtype=torch.uint8
    )

    (k_data, v_data), (k_scales, v_scales) = nvfp4_kv_cache_split_views(
        kv_cache, head_size, head_size_v
    )

    assert k_data.shape == (num_pages, dim_1, dim_2, head_size // 2)
    assert k_scales.shape == (num_pages, dim_1, dim_2, head_size // 16)
    assert v_data.shape == (num_pages, dim_1, dim_2, head_size_v // 2)
    assert v_scales.shape == (num_pages, dim_1, dim_2, head_size_v // 16)

    page_items = dim_1 * dim_2
    assert k_data.storage_offset() == kv_cache.storage_offset()
    assert k_scales.storage_offset() == kv_cache.storage_offset() + page_items * (
        head_size // 2
    )
    assert v_data.storage_offset() == kv_cache.storage_offset() + page_items * (
        k_full_dim
    )
    assert v_scales.storage_offset() == kv_cache.storage_offset() + page_items * (
        k_full_dim + head_size_v // 2
    )

    offset_sets = [
        _storage_offsets(k_data),
        _storage_offsets(k_scales),
        _storage_offsets(v_data),
        _storage_offsets(v_scales),
    ]
    assert len(set().union(*offset_sets)) == sum(
        len(offsets) for offsets in offset_sets
    )


def test_nvfp4_kv_cache_split_views_rejects_incompatible_strides() -> None:
    head_size = 64
    full_dim = nvfp4_kv_cache_full_dim(head_size)
    storage = torch.empty(1000, dtype=torch.uint8)
    kv_side = torch.as_strided(
        storage,
        (2, 4, 3, full_dim),
        (500, 100, 37, 1),
    )

    with pytest.raises(ValueError, match="strides are not compatible"):
        nvfp4_kv_cache_split_views(kv_side, head_size)


def test_nvfp4_kv_cache_split_views_accepts_size_one_strides() -> None:
    head_size = 64
    full_dim = nvfp4_kv_cache_full_dim(head_size)
    storage = torch.empty(256, dtype=torch.uint8)
    kv_side = torch.as_strided(
        storage,
        (2, 1, 3, full_dim),
        (128, 7, full_dim, 1),
    )

    (data,), (scale,) = nvfp4_kv_cache_split_views(kv_side, head_size)

    assert data.shape == (2, 1, 3, head_size // 2)
    assert scale.shape == (2, 1, 3, head_size // 16)


def test_nvfp4_kv_cache_split_views_rejects_mixed_incompatible_strides() -> None:
    head_size = 64
    head_size_v = 128
    k_full_dim = nvfp4_kv_cache_full_dim(head_size)
    v_full_dim = nvfp4_kv_cache_full_dim(head_size_v)
    full_dim = k_full_dim + v_full_dim
    storage = torch.empty(1000, dtype=torch.uint8)
    kv_cache = torch.as_strided(
        storage,
        (2, 4, 3, full_dim),
        (500, 100, 37, 1),
    )

    with pytest.raises(ValueError, match="strides are not compatible"):
        nvfp4_kv_cache_split_views(kv_cache, head_size, head_size_v)


def test_nvfp4_kv_cache_split_views_rejects_mixed_side_slice() -> None:
    head_size = 64
    head_size_v = 128
    k_full_dim = nvfp4_kv_cache_full_dim(head_size)
    v_full_dim = nvfp4_kv_cache_full_dim(head_size_v)
    kv_cache = torch.empty((2, 4, 3, k_full_dim + v_full_dim), dtype=torch.uint8)

    with pytest.raises(ValueError, match="strides are not compatible"):
        nvfp4_kv_cache_split_views(kv_cache[..., :k_full_dim], head_size)


def test_nvfp4_kv_cache_split_views_rejects_non_inferable_dim() -> None:
    kv_side = torch.empty((2, 4, 3, 10), dtype=torch.uint8)

    with pytest.raises(ValueError, match="last dimension cannot be inferred"):
        nvfp4_kv_cache_split_views(kv_side)


@pytest.mark.parametrize(
    ("head_dim", "head_dim_v", "expected_calls"),
    [
        (128, 128, 1),
        (256, 512, 0),
        (512, 512, 0),
    ],
)
def test_flashinfer_nvfp4_fa2_prefill_reservation_requires_matching_head_dims(
    monkeypatch, head_dim: int, head_dim_v: int, expected_calls: int
) -> None:
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    FlashInferMetadataBuilder = flashinfer_backend.FlashInferMetadataBuilder
    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder.is_kvcache_nvfp4 = True
    builder.head_dim = head_dim
    builder.head_dim_v = head_dim_v
    builder.use_dcp = False
    builder.model_config = SimpleNamespace(max_model_len=1024, dtype=torch.float16)
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=True,
            max_num_batched_tokens=8,
        )
    )
    builder.num_kv_heads = 2

    calls = []

    class FakeWorkspaceManager:
        def get_simultaneous(self, *specs):
            calls.append(specs)

    monkeypatch.setattr(
        flashinfer_backend, "_is_flash_attn_varlen_func_available", lambda: True
    )
    monkeypatch.setattr(
        flashinfer_backend.flashinfer,
        "nvfp4_kv_dequantize_paged",
        object(),
        raising=False,
    )
    monkeypatch.setattr(
        flashinfer_backend, "is_workspace_manager_initialized", lambda: True
    )
    monkeypatch.setattr(
        flashinfer_backend, "current_workspace_manager", FakeWorkspaceManager
    )

    builder._reserve_nvfp4_fa2_prefill_workspace(can_use_trtllm=False)

    assert len(calls) == expected_calls
    if expected_calls:
        assert calls[0] == (
            ((1024, 2, head_dim), torch.float16),
            ((1024, 2, head_dim_v), torch.float16),
        )


def test_flashinfer_impl_caches_nvfp4_slot_mapping_writer(monkeypatch) -> None:
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    def fake_slot_writer(*args, **kwargs):
        pass

    monkeypatch.setattr(
        flashinfer_backend.flashinfer,
        "nvfp4_quantize_append_paged_kv_cache_with_slot_mapping",
        fake_slot_writer,
        raising=False,
    )
    monkeypatch.setattr(
        flashinfer_backend.current_platform,
        "is_device_capability_family",
        lambda family: False,
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "can_use_trtllm_attention",
        lambda num_heads, num_kv_heads, is_prefill=False: False,
    )

    impl = flashinfer_backend.FlashInferImpl(
        num_heads=1,
        head_size=128,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="nvfp4",
    )

    assert impl._nvfp4_slot_writer is fake_slot_writer


def test_flashinfer_impl_caches_nvfp4_kv_cache_views(monkeypatch) -> None:
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    def fake_slot_writer(*args, **kwargs):
        pass

    monkeypatch.setattr(
        flashinfer_backend.flashinfer,
        "nvfp4_quantize_append_paged_kv_cache_with_slot_mapping",
        fake_slot_writer,
        raising=False,
    )
    monkeypatch.setattr(
        flashinfer_backend.current_platform,
        "is_device_capability_family",
        lambda family: False,
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "can_use_trtllm_attention",
        lambda num_heads, num_kv_heads, is_prefill=False: False,
    )
    monkeypatch.setattr(flashinfer_backend, "get_kv_cache_layout", lambda: "NHD")

    head_size = 64
    head_size_v = 128
    impl = flashinfer_backend.FlashInferImpl(
        num_heads=1,
        head_size=head_size,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="nvfp4",
        head_size_v=head_size_v,
    )

    full_dim = nvfp4_kv_cache_full_dim(head_size) + nvfp4_kv_cache_full_dim(head_size_v)
    kv_cache = torch.empty((2, 4, 3, full_dim), dtype=torch.uint8)

    views = impl._get_nvfp4_kv_cache_views(kv_cache)
    cached_views = impl._get_nvfp4_kv_cache_views(kv_cache)

    assert cached_views is views
    assert cached_views.kv_cache is views.kv_cache
    assert cached_views.data[0] is views.data[0]
    assert cached_views.block_scales[0] is views.block_scales[0]

    rebound_kv_cache = torch.empty_like(kv_cache)
    rebound_views = impl._get_nvfp4_kv_cache_views(rebound_kv_cache)

    assert rebound_views is not views
    assert rebound_views.data[0].data_ptr() == rebound_kv_cache.data_ptr()


def test_flashinfer_impl_requires_nvfp4_slot_mapping_writer(monkeypatch) -> None:
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    monkeypatch.delattr(
        flashinfer_backend.flashinfer,
        "nvfp4_quantize_append_paged_kv_cache_with_slot_mapping",
        raising=False,
    )
    monkeypatch.setattr(
        flashinfer_backend.current_platform,
        "is_device_capability_family",
        lambda family: False,
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "can_use_trtllm_attention",
        lambda num_heads, num_kv_heads, is_prefill=False: False,
    )

    with pytest.raises(RuntimeError, match="NVFP4 slot-mapping KV cache update"):
        flashinfer_backend.FlashInferImpl(
            num_heads=1,
            head_size=128,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="nvfp4",
        )


def test_fast_decode_plan_importable() -> None:
    """fast_decode_plan must be importable from flashinfer.decode.

    This is a forward-compatibility smoke test: if FlashInfer reorganises its
    public API the import will fail before any other test does.
    """
    from flashinfer.decode import fast_decode_plan  # noqa: F401

    assert callable(fast_decode_plan)


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode
def test_fast_plan_decode_warmup_uses_full_plan(dtype: torch.dtype) -> None:
    """On the first call fast_plan_decode must route through self.plan() and
    flip vllm_first_call to False on the wrapper object."""
    from unittest.mock import patch

    from vllm.v1.attention.backends.flashinfer import fast_plan_decode

    torch.set_default_device("cuda")
    set_random_seed(0)

    kv_lens = [128, 64]
    block_size = 16
    num_seqs = len(kv_lens)
    num_query_heads, num_kv_heads = 8, 2
    head_size = 128

    kv_indptr, kv_indices, kv_last_page_lens, _ = _make_paged_kv_metadata(
        kv_lens, block_size, NUM_BLOCKS
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = _make_cg_decode_wrapper(num_seqs, kv_indices.clone(), workspace)

    assert getattr(wrapper, "vllm_first_call", True) is True

    with patch.object(wrapper, "plan", wraps=wrapper.plan) as mock_plan:
        fast_plan_decode(
            wrapper,
            indptr_cpu=kv_indptr,
            indices=kv_indices,
            last_page_len_cpu=kv_last_page_lens,
            num_qo_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_size,
            page_size=block_size,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        mock_plan.assert_called_once()

    assert wrapper.vllm_first_call is False, (
        "vllm_first_call should be False after the first fast_plan_decode call"
    )


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode
def test_fast_plan_decode_accepts_nvfp4_kv_plan_dtype(dtype: torch.dtype) -> None:
    from vllm.v1.attention.backends.flashinfer import fast_plan_decode

    torch.set_default_device("cuda")
    set_random_seed(0)

    kv_lens = [128, 64]
    block_size = 16
    num_seqs = len(kv_lens)
    num_query_heads, num_kv_heads = 8, 2
    head_size = 128

    kv_indptr, kv_indices, kv_last_page_lens, _ = _make_paged_kv_metadata(
        kv_lens, block_size, NUM_BLOCKS
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = _make_cg_decode_wrapper(num_seqs, kv_indices.clone(), workspace)

    fast_plan_decode(
        wrapper,
        indptr_cpu=kv_indptr,
        indices=kv_indices,
        last_page_len_cpu=kv_last_page_lens,
        num_qo_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        page_size=block_size,
        q_data_type=dtype,
        kv_data_type=torch.uint8,
        o_data_type=dtype,
    )

    assert wrapper.vllm_first_call is False


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode
def test_flashinfer_prefill_accepts_nvfp4_kv_plan_dtype(
    dtype: torch.dtype,
) -> None:
    torch.set_default_device("cuda")
    set_random_seed(0)

    batch_size = 2
    qo_len = 8
    kv_len = 16
    block_size = 16
    num_query_heads, num_kv_heads = 8, 2
    head_size = 128
    num_pages_per_seq = (kv_len + block_size - 1) // block_size
    total_num_pages = num_pages_per_seq * batch_size

    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), kv_len, dtype=torch.int32, device="cuda"
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")

    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        q_data_type=dtype,
        kv_data_type=torch.uint8,
        o_data_type=dtype,
    )

    assert wrapper._cached_kv_data_type == torch.uint8


@pytest.mark.parametrize("kv_lens", [[1328, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode
def test_fast_plan_decode_matches_full_plan(
    kv_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
) -> None:
    """fast_plan_decode's cudagraph path (delegating to FlashInfer's
    fast_decode_plan) must produce attention output numerically identical to
    a standard plan() call.

    Both the warmup call (self.plan) and the subsequent fast call
    (fast_decode_plan) are verified against the same reference.
    """
    from vllm.v1.attention.backends.flashinfer import fast_plan_decode

    torch.set_default_device("cuda")
    set_random_seed(0)
    num_seqs = len(kv_lens)
    num_query_heads, num_kv_heads = num_heads

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
    key_value_cache = torch.randn(
        NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype
    )

    kv_indptr, kv_indices, kv_last_page_lens, _ = _make_paged_kv_metadata(
        kv_lens, block_size, NUM_BLOCKS
    )

    # Reference output via the standard plan()
    workspace_ref = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    ref_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_ref, "NHD", use_tensor_cores=True
    )
    ref_wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        "NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    ref_output = ref_wrapper.run(query, key_value_cache)

    # CUDAGraph wrapper exercised through fast_plan_decode
    kv_indices_buf = kv_indices.clone()
    workspace_cg = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    cg_wrapper = _make_cg_decode_wrapper(num_seqs, kv_indices_buf, workspace_cg)

    plan_kwargs: dict = dict(
        indptr_cpu=kv_indptr,
        indices=kv_indices_buf,
        last_page_len_cpu=kv_last_page_lens,
        num_qo_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        page_size=block_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    # First call – warmup path (routes through self.plan)
    fast_plan_decode(cg_wrapper, **plan_kwargs)
    warmup_output = cg_wrapper.run(query, key_value_cache)
    torch.testing.assert_close(warmup_output, ref_output, atol=1e-2, rtol=1e-2)

    # Second call – fast path (routes through fast_decode_plan from FlashInfer)
    fast_plan_decode(cg_wrapper, **plan_kwargs)
    fast_output = cg_wrapper.run(query, key_value_cache)
    torch.testing.assert_close(fast_output, ref_output, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("kv_lens", [[1328, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@torch.inference_mode
def test_flashinfer_decode_with_paged_kv(
    kv_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    sliding_window: int | None,
) -> None:
    torch.set_default_device("cuda")
    set_random_seed(0)
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)

    key_value_cache = torch.randn(
        NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype
    )
    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)

    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_tensor_cores=True
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        "NONE",
        window_left=sliding_window - 1 if sliding_window is not None else -1,
        q_data_type=dtype,
        kv_data_type=dtype,
        logits_soft_cap=soft_cap,
    )

    output = wrapper.run(query, key_value_cache)

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * num_seqs,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        soft_cap=soft_cap,
        sliding_window=sliding_window,
    )
    (
        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@torch.inference_mode
def test_flashinfer_prefill_with_paged_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    sliding_window: int | None,
) -> None:
    torch.set_default_device("cuda")
    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_value_cache = torch.randn(
        NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype
    )
    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

    # Normalize the scale of the key and value caches to mitigate
    # numerical instability.
    key_cache /= head_size**0.5
    value_cache /= head_size**0.5

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    qo_indptr = [0]
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)
        qo_indptr.append(qo_indptr[-1] + query_lens[i])

    qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32)
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        window_left=sliding_window - 1 if sliding_window is not None else -1,
        q_data_type=dtype,
        kv_data_type=dtype,
        logits_soft_cap=soft_cap,
    )

    output = wrapper.run(
        query,
        key_value_cache,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        soft_cap=soft_cap,
        sliding_window=sliding_window,
    )
    (
        torch.testing.assert_close(output, ref_output, atol=5e-2, rtol=1e-2),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.parametrize("seq_lens", [[(1, 132), (5, 18)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
def test_flashinfer_prefill_with_paged_fp8_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
) -> None:
    pytest.skip("TODO: fix the accuracy issue")
    torch.set_default_device("cuda")
    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    kv_cache_dtype = torch.float8_e4m3fn

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    NUM_BLOCKS_FP8 = 2048
    key_value_cache = torch.randn(
        NUM_BLOCKS_FP8, 2, block_size, num_kv_heads, head_size, dtype=dtype
    )
    key_cache, value_cache = torch.chunk(key_value_cache, 2, dim=1)
    key_cache /= head_size**0.5
    value_cache /= head_size**0.5

    k_scale = key_cache.amax().item() / 448.0
    v_scale = value_cache.amax().item() / 448.0

    kv_cache_fp8 = torch.cat([key_cache / k_scale, value_cache / v_scale], dim=1).to(
        kv_cache_dtype
    )

    assert kv_cache_fp8.shape == key_value_cache.shape
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS_FP8, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    qo_indptr = [0]
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)
        qo_indptr.append(qo_indptr[-1] + query_lens[i])

    qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32)
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        q_data_type=dtype,
        kv_data_type=kv_cache_dtype,
        logits_soft_cap=soft_cap,
    )

    output = wrapper.run(query, kv_cache_fp8, k_scale=k_scale, v_scale=v_scale)

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache.squeeze(1),
        value_cache=value_cache.squeeze(1),
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        soft_cap=soft_cap,
    )
    del query
    del block_tables
    # verify prefill fp8
    (
        torch.testing.assert_close(output, ref_output, atol=5e-2, rtol=1e-2),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.parametrize("kv_lens", [[1328, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.skip(reason="TODO: fix the accuracy issue")
@torch.inference_mode
def test_flashinfer_decode_with_paged_fp8_kv(
    kv_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
) -> None:
    # test doesn't work for num_heads = (16,16)
    torch.set_default_device("cuda")
    set_random_seed(0)
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    use_tensor_cores = True
    kv_cache_dtype = torch.float8_e4m3fn

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
    NUM_BLOCKS_FP8 = 2048
    key_value_cache = torch.randn(
        NUM_BLOCKS_FP8, 2, block_size, num_kv_heads, head_size, dtype=dtype
    )
    key_cache, value_cache = torch.chunk(key_value_cache, 2, dim=1)
    key_cache /= head_size**0.5
    value_cache /= head_size**0.5

    k_scale = key_cache.amax().item() / 448.0
    v_scale = value_cache.amax().item() / 448.0

    key_cache_fp8 = (key_cache / k_scale).to(kv_cache_dtype)
    value_cache_fp8 = (value_cache / v_scale).to(kv_cache_dtype)
    assert key_cache_fp8.shape[1] == 1 and value_cache_fp8.shape[1] == 1
    kv_cache_fp8 = torch.cat([key_cache_fp8, value_cache_fp8], dim=1)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS_FP8, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)

    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_tensor_cores=use_tensor_cores
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        "NONE",
        q_data_type=dtype,
        kv_data_type=kv_cache_dtype,
        logits_soft_cap=soft_cap,
    )
    output = wrapper.run(query, kv_cache_fp8, k_scale=k_scale, v_scale=v_scale)
    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * num_seqs,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        soft_cap=soft_cap,
    )
    # Temporary fix: Increasing the tolerance. Seems like a flashinfer issue
    (
        torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2),
        f"{torch.max(torch.abs(output - ref_output))}",
    )
