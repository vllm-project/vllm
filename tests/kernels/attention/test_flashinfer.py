# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from itertools import product
from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform
from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim, set_random_seed

try:
    import flashinfer
except ImportError:
    if current_platform.is_rocm():
        pytest.skip(
            "flashinfer is not supported for vLLM on ROCm.", allow_module_level=True
        )

import torch
from flashinfer import get_seq_lens

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.flashinfer import _nvfp4_combined_page_views
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheLayout,
    KVQuantMode,
    compute_layer_kv_cache_shape_bytes,
)

NUM_HEADS = [(32, 8), (6, 1)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16, 32]
DTYPES = [torch.bfloat16]
NUM_BLOCKS = 32768  # Large enough to test overflow in index calculation.
SOFT_CAPS = [None, 30.0]
SLIDING_WINDOWS = [None, 64]


_TEST_KV_LAYOUTS = {"NHD": KVCacheLayout.LBNHC, "HND": KVCacheLayout.LBHNC}


def _patch_impl_kv_cache_layout(monkeypatch, flashinfer_backend, name: str):
    """The impl reads its layout from cache_config via a property now; the
    module-level get_kv_cache_layout() these tests used to patch is gone."""
    monkeypatch.setattr(
        flashinfer_backend.FlashInferImpl,
        "kv_cache_layout",
        property(lambda self: _TEST_KV_LAYOUTS[name]),
    )


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


def _validate_nvfp4_configuration(
    capability: tuple[int, int],
    kv_cache_dtype: str = "nvfp4",
    **kwargs,
) -> list[str]:
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    return FlashInferBackend.validate_configuration(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(*capability),
        attn_type="decoder",
        **kwargs,
    )


@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 9), (10, 0)])
def test_flashinfer_backend_accepts_nvfp4_kv_cache(
    capability: tuple[int, int],
) -> None:
    assert _validate_nvfp4_configuration(capability) == []


@pytest.mark.parametrize("capability", [(9, 0), (12, 0)])
def test_flashinfer_backend_rejects_nvfp4_on_xqa_decode_arch(
    capability: tuple[int, int],
) -> None:
    """SM90 and SM12x decode through XQA, which asserts against NVFP4 in
    forward; the configuration check has to refuse them first."""
    reasons = _validate_nvfp4_configuration(capability)

    assert any("SM8x or SM100" in reason for reason in reasons)


def test_flashinfer_backend_rejects_nvfp4_with_dcp() -> None:
    reasons = _validate_nvfp4_configuration((8, 0), use_dcp=True)

    assert any("decode context parallelism" in reason for reason in reasons)


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


def test_flashinfer_nvfp4_customize_spec_drives_view_shape():
    """NVFP4 packing is published through the spec, so the shared allocator
    reproduces the layout the FlashInfer NVFP4 path expects: K and V in
    separate head slots."""
    head_size = head_size_v = 128
    try:
        from vllm.v1.attention.backends.flashinfer import FlashInferBackend
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"FlashInfer backend unavailable: {exc}")

    num_blocks = 2
    block_size = 16
    num_kv_heads = 2
    spec = FlashInferBackend.customize_spec(
        FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            head_size_v=head_size_v,
            dtype=torch.uint8,
            kv_quant_mode=KVQuantMode.NVFP4,
        )
    )

    full_k = nvfp4_kv_cache_full_dim(head_size)
    expected_heads, expected_content = 2 * num_kv_heads, full_k
    assert spec.num_heads == expected_heads
    assert spec.state_content_size_bytes == expected_content

    raw = torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    (kv_cache,) = dense_kv_cache_views(raw, spec, num_blocks, 1, KVCacheLayout.LBHNC)

    assert kv_cache.shape == (
        num_blocks,
        expected_heads,
        block_size,
        expected_content,
    )
    assert kv_cache[0].is_contiguous()


@pytest.mark.parametrize(
    ("quant_mode_name", "head_size_v", "expected_shape"),
    [
        pytest.param("NONE", None, (1392, 4, 32, 2 * 256 * 2), id="auto"),
        pytest.param("FP8_PER_TENSOR", None, (1392, 4, 32, 2 * 256 * 2), id="fp8"),
        pytest.param(
            "NVFP4",
            256,
            (1392, 8, 32, nvfp4_kv_cache_full_dim(256) * 2),
            id="nvfp4-same-head",
        ),
    ],
)
def test_flashinfer_kv_cache_byte_shape(
    quant_mode_name: str,
    head_size_v: int | None,
    expected_shape: tuple[int, ...],
) -> None:
    """NVFP4 splits K and V across head slots. The trailing dimension counts
    bytes."""
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    spec = FlashInferBackend.customize_spec(
        FullAttentionSpec(
            block_size=32,
            num_kv_heads=4,
            head_size=256,
            head_size_v=head_size_v,
            dtype=torch.bfloat16,
            kv_quant_mode=KVQuantMode[quant_mode_name],
        )
    )

    assert compute_layer_kv_cache_shape_bytes(spec, 1392) == expected_shape


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
def test_nvfp4_combined_page_views_cover_packed_layout(
    shape: tuple[int, int, int],
) -> None:
    head_size = 128
    full_dim = nvfp4_kv_cache_full_dim(head_size)
    num_pages, dim_1, dim_2 = shape
    kv_cache = torch.empty(num_pages, dim_1, dim_2, 2 * full_dim, dtype=torch.uint8)

    (k_data, v_data), (k_scales, v_scales) = _nvfp4_combined_page_views(
        kv_cache, head_size
    )

    assert k_data.shape == (num_pages, dim_1, dim_2, head_size // 2)
    assert k_scales.shape == (num_pages, dim_1, dim_2, head_size // 16)
    assert v_data.shape == k_data.shape
    assert v_scales.shape == k_scales.shape

    page_items = dim_1 * dim_2
    base = kv_cache.storage_offset()
    assert k_data.storage_offset() == base
    assert k_scales.storage_offset() == base + page_items * (head_size // 2)
    assert v_data.storage_offset() == base + page_items * full_dim
    assert v_scales.storage_offset() == base + page_items * (full_dim + head_size // 2)

    offset_sets = [
        _storage_offsets(k_data),
        _storage_offsets(k_scales),
        _storage_offsets(v_data),
        _storage_offsets(v_scales),
    ]
    assert len(set().union(*offset_sets)) == sum(
        len(offsets) for offsets in offset_sets
    )


def test_nvfp4_combined_page_views_reject_incompatible_strides() -> None:
    head_size = 128
    full_dim = nvfp4_kv_cache_full_dim(head_size)
    storage = torch.empty(4000, dtype=torch.uint8)
    kv_cache = torch.as_strided(
        storage,
        (2, 4, 3, 2 * full_dim),
        (2000, 400, 37, 1),
    )

    with pytest.raises(ValueError, match="strides are not compatible"):
        _nvfp4_combined_page_views(kv_cache, head_size)


def test_nvfp4_combined_page_views_accept_size_one_strides() -> None:
    head_size = 128
    full_dim = nvfp4_kv_cache_full_dim(head_size)
    storage = torch.empty(4 * 3 * 2 * full_dim, dtype=torch.uint8)
    kv_cache = torch.as_strided(
        storage,
        (2, 1, 3, 2 * full_dim),
        (3 * 2 * full_dim, 7, 2 * full_dim, 1),
    )

    (k_data, _), (k_scale, _) = _nvfp4_combined_page_views(kv_cache, head_size)

    assert k_data.shape == (2, 1, 3, head_size // 2)
    assert k_scale.shape == (2, 1, 3, head_size // 16)


def test_nvfp4_combined_page_views_reject_side_slice() -> None:
    head_size = 128
    full_dim = nvfp4_kv_cache_full_dim(head_size)
    kv_cache = torch.empty((2, 4, 3, 2 * full_dim), dtype=torch.uint8)

    with pytest.raises(ValueError, match="last dimension does not match head size"):
        _nvfp4_combined_page_views(kv_cache[..., :full_dim], head_size)


def test_nvfp4_combined_page_views_reject_wrong_rank() -> None:
    head_size = 128
    full_dim = nvfp4_kv_cache_full_dim(head_size)
    kv_cache = torch.empty((2, 2, 4, 3, 2 * full_dim), dtype=torch.uint8)

    with pytest.raises(ValueError, match="must be 4D"):
        _nvfp4_combined_page_views(kv_cache, head_size)


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


@pytest.mark.parametrize(
    ("prefill_ok", "decode_ok", "expected_native"),
    [
        (True, True, True),
        (False, True, False),
        (True, False, False),
        (False, False, False),
    ],
)
def test_flashinfer_impl_gates_native_nvfp4_update_on_trtllm_availability(
    monkeypatch, prefill_ok: bool, decode_ok: bool, expected_native: bool
) -> None:
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    def fake_slot_writer(*args, **kwargs):
        pass

    def fake_can_use_trtllm_attention(
        num_heads: int, num_kv_heads: int, is_prefill: bool = False
    ) -> bool:
        return prefill_ok if is_prefill else decode_ok

    monkeypatch.setattr(
        flashinfer_backend.flashinfer,
        "nvfp4_quantize_append_paged_kv_cache_with_slot_mapping",
        fake_slot_writer,
        raising=False,
    )
    monkeypatch.setattr(
        flashinfer_backend.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "can_use_trtllm_attention",
        fake_can_use_trtllm_attention,
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

    assert impl.use_native_nvfp4_kv_cache_update is expected_native
    if expected_native:
        assert impl._nvfp4_slot_writer is None
    else:
        assert impl._nvfp4_slot_writer is fake_slot_writer


@pytest.mark.parametrize("cache_layout", ["NHD", "HND"])
def test_flashinfer_impl_same_head_nvfp4_views_cover_compact_pages(
    monkeypatch, cache_layout: str
) -> None:
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    _patch_impl_kv_cache_layout(monkeypatch, flashinfer_backend, cache_layout)

    num_blocks = 2
    num_kv_heads = 3
    block_size = 16
    head_size = 128
    full_dim = nvfp4_kv_cache_full_dim(head_size)
    logical_shape = (num_blocks, 2 * num_kv_heads, block_size, full_dim)
    stride_order = (0, 2, 1, 3) if cache_layout == "NHD" else (0, 1, 2, 3)
    physical_shape = tuple(logical_shape[i] for i in stride_order)
    physical_cache = torch.empty(physical_shape, dtype=torch.uint8)
    inverse_order = tuple(stride_order.index(i) for i in range(4))
    kv_cache = physical_cache.permute(*inverse_order)

    impl = flashinfer_backend.FlashInferImpl.__new__(flashinfer_backend.FlashInferImpl)
    impl.head_size = head_size
    impl.num_kv_heads = num_kv_heads
    impl.kv_cache_dtype = "nvfp4"
    impl.use_native_nvfp4_kv_cache_update = False
    impl._nvfp4_kv_cache_view_key = None
    impl._nvfp4_kv_cache_views = None

    views = impl._get_nvfp4_kv_cache_views(kv_cache)
    cached_views = impl._get_nvfp4_kv_cache_views(kv_cache)

    assert cached_views is views
    if cache_layout == "NHD":
        expected_prefix = (num_blocks, block_size, num_kv_heads)
    else:
        expected_prefix = (num_blocks, num_kv_heads, block_size)
    assert views.data[0].shape == (*expected_prefix, head_size // 2)
    assert views.data[1].shape == (*expected_prefix, head_size // 2)
    assert views.block_scales[0].shape == (*expected_prefix, head_size // 16)
    assert views.block_scales[1].shape == (*expected_prefix, head_size // 16)

    offset_sets = [
        _storage_offsets(views.data[0]),
        _storage_offsets(views.block_scales[0]),
        _storage_offsets(views.data[1]),
        _storage_offsets(views.block_scales[1]),
    ]
    assert len(set().union(*offset_sets)) == sum(
        len(offsets) for offsets in offset_sets
    )
    assert len(set().union(*offset_sets)) == physical_cache.numel()

    # Rebinding a different KV cache tensor must invalidate the cached views.
    rebound_physical = torch.empty_like(physical_cache)
    rebound_kv_cache = rebound_physical.permute(*inverse_order)
    rebound_views = impl._get_nvfp4_kv_cache_views(rebound_kv_cache)

    assert rebound_views is not views
    assert rebound_views.data[0].data_ptr() == rebound_kv_cache.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_nvfp4_slot_write_then_native_prefill_matches_dequantized_reference(
    monkeypatch,
) -> None:
    """Drive the SM8x NVFP4 contract end to end.

    The slot-mapping writer fills a production-shaped combined page, the
    cached views are rebuilt from that same allocation, and the native
    FlashInfer prefill reads it back.  The result is compared against
    attention over the KV the writer actually stored, so a layout or stride
    mistake in either direction shows up as a numerical mismatch.
    """
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    # The slot-mapping writer is the contract this path is built on; a
    # FlashInfer build without it must fail here, not quietly skip.
    assert hasattr(flashinfer, "nvfp4_quantize_append_paged_kv_cache_with_slot_mapping")

    torch.manual_seed(0)
    dtype = torch.bfloat16
    page_size = 16
    num_qo_heads = 4
    num_kv_heads = 1
    head_size = 128
    # One request whose context is longer than the current chunk, i.e. the
    # continuation-prefill shape the fallback has to serve.
    seq_lens = [96]
    query_lens = [32]
    num_reqs = len(seq_lens)
    pages_per_req = [(s + page_size - 1) // page_size for s in seq_lens]
    num_pages = sum(pages_per_req)

    _patch_impl_kv_cache_layout(monkeypatch, flashinfer_backend, "NHD")
    monkeypatch.setattr(
        flashinfer_backend,
        "can_use_trtllm_attention",
        lambda num_heads, num_kv_heads, is_prefill=False: False,
    )

    impl = flashinfer_backend.FlashInferImpl(
        num_heads=num_qo_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="nvfp4",
    )
    assert not impl.use_native_nvfp4_kv_cache_update

    # Production allocation: 2 * num_kv_heads head slots of packed fp4 + scales.
    full_dim = nvfp4_kv_cache_full_dim(head_size)
    physical = torch.zeros(
        (num_pages, page_size, 2 * num_kv_heads, full_dim),
        dtype=torch.uint8,
        device="cuda",
    )
    kv_cache = physical.permute(0, 2, 1, 3)

    block_tables = torch.arange(num_pages, dtype=torch.int32, device="cuda").reshape(
        num_reqs, max(pages_per_req)
    )

    raw_key = (
        torch.randn((seq_lens[0], num_kv_heads, head_size), dtype=dtype, device="cuda")
        * 0.2
    )
    raw_value = torch.randn_like(raw_key) * 0.2
    slot_mapping = torch.tensor(
        [
            int(block_tables[0, t // page_size].item()) * page_size + t % page_size
            for t in range(seq_lens[0])
        ],
        dtype=torch.int64,
        device="cuda",
    )

    one = torch.ones((), dtype=torch.float32, device="cuda")
    layer = SimpleNamespace(_k_scale=one, _v_scale=one)
    layer._q_scale_float = 1.0
    layer._k_scale_float = 1.0
    layer._v_scale_float = 1.0

    # Real write through the production entry point.
    impl.do_kv_cache_update(layer, raw_key, raw_value, kv_cache, slot_mapping)
    assert physical.any(), "slot-mapping writer stored nothing"

    views = impl._get_nvfp4_kv_cache_views(kv_cache)

    # Reference KV: what the writer actually stored, dequantized.
    deq_k = torch.empty(
        (num_reqs, seq_lens[0], num_kv_heads, head_size), dtype=dtype, device="cuda"
    )
    deq_v = torch.empty_like(deq_k)
    flashinfer.nvfp4_kv_dequantize_paged(
        views.data,
        views.block_scales,
        block_tables,
        torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        one,
        one,
        deq_k,
        deq_v,
        kv_layout="NHD",
    )

    query = (
        torch.randn(
            (query_lens[0], num_qo_heads, head_size), dtype=dtype, device="cuda"
        )
        * 0.2
    )
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        torch.tensor([0, query_lens[0]], dtype=torch.int32),
        torch.tensor([0, pages_per_req[0]], dtype=torch.int32),
        block_tables[0, : pages_per_req[0]].contiguous(),
        torch.tensor([seq_lens[0] % page_size or page_size], dtype=torch.int32),
        num_qo_heads,
        num_kv_heads,
        head_size,
        page_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=torch.uint8,
        o_data_type=dtype,
    )

    output = torch.empty_like(query)
    impl._run_native_nvfp4_prefill(
        layer, wrapper, query, output, views.data, views.block_scales
    )

    # Causal attention over exactly the KV the writer stored.
    key = deq_k[0].repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
    value = deq_v[0].repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
    logits = torch.einsum("qhd,khd->hqk", query.float(), key.float())
    logits *= head_size**-0.5
    q_pos = torch.arange(
        seq_lens[0] - query_lens[0], seq_lens[0], device="cuda"
    ).unsqueeze(1)
    k_pos = torch.arange(seq_lens[0], device="cuda").unsqueeze(0)
    logits.masked_fill_(k_pos > q_pos, float("-inf"))
    reference = torch.einsum("hqk,khd->qhd", logits.softmax(dim=-1), value.float())

    torch.testing.assert_close(output.float(), reference.float(), atol=2e-2, rtol=2e-2)


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


@pytest.mark.parametrize("trtllm_supported", [False, True])
def test_flashinfer_backend_gates_nvfp4_scale_search_on_trtllm(
    monkeypatch, trtllm_supported: bool
) -> None:
    """NVFP4 variants that only change the store-time scale search need the
    trtllm-gen native store path; plain nvfp4 stays available either way."""
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    monkeypatch.setattr(
        flashinfer_backend,
        "supports_trtllm_attention",
        lambda is_prefill: trtllm_supported,
    )

    backend = flashinfer_backend.FlashInferBackend
    assert backend.supports_kv_cache_dtype("nvfp4")
    assert backend.supports_kv_cache_dtype("nvfp4_4over6") is trtllm_supported


def test_flashinfer_impl_rejects_nvfp4_scale_search_without_native_update(
    monkeypatch,
) -> None:
    """The FlashInfer slot-mapping writer records plain max/6 scales, so an
    NVFP4 scale-search dtype must fail instead of silently degrading."""
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    monkeypatch.setattr(
        flashinfer_backend,
        "can_use_trtllm_attention",
        lambda num_heads, num_kv_heads, is_prefill=False: False,
    )

    with pytest.raises(ValueError, match="trtllm-gen native NVFP4 KV cache update"):
        flashinfer_backend.FlashInferImpl(
            num_heads=1,
            head_size=128,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="nvfp4_4over6",
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
            seq_lens_cpu=torch.tensor(kv_lens, dtype=torch.int32, device="cpu"),
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
        seq_lens_cpu=get_seq_lens(kv_indptr, kv_last_page_lens, block_size),
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
        seq_lens_cpu=torch.tensor(kv_lens, dtype=torch.int32, device="cpu"),
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
