# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the format-aware per-layer helpers in
:mod:`lmcache.v1.gpu_connector.utils`.
"""

# Third Party
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PageBufferShapeDesc and GPUKVFormat require the CUDA build",
)

# First Party
from lmcache.v1.gpu_connector.kv_format.contiguity import (  # noqa: E402
    attempt_permute_to_contiguous_view,
)
from lmcache.v1.gpu_connector.utils import (  # noqa: E402
    get_device,
    get_dtype,
    get_group_data_ptrs,
    get_head_size,
    get_num_heads,
    make_page_buffer_shape_desc,
)
import lmcache.c_ops as lmc_ops  # noqa: E402


def test_make_shape_desc_vllm_flash_attn_nhd():
    kv_caches = [torch.empty(2, 32, 16, 8, 64, dtype=torch.bfloat16) for _ in range(4)]
    sd = make_page_buffer_shape_desc(
        kv_caches,
        lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        layer_idx=0,
        num_layers_in_group=4,
        num_blocks=32,
        block_size=16,
    )
    assert sd.kv_size == 2
    assert sd.nl == 4
    assert sd.nb == 32
    assert sd.bs == 16
    assert sd.nh == 8
    assert sd.hs == 64
    assert sd.element_size == 2


def test_make_shape_desc_vllm_flash_infer_nhd():
    kv_caches = [torch.empty(32, 2, 16, 8, 64, dtype=torch.float16) for _ in range(2)]
    sd = make_page_buffer_shape_desc(
        kv_caches,
        lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,
        layer_idx=0,
        num_layers_in_group=2,
        num_blocks=32,
        block_size=16,
    )
    assert sd.nh == 8
    assert sd.hs == 64
    assert sd.kv_size == 2


def test_make_shape_desc_vllm_mla():
    kv_caches = [torch.empty(32, 16, 512, dtype=torch.bfloat16) for _ in range(3)]
    sd = make_page_buffer_shape_desc(
        kv_caches,
        lmc_ops.EngineKVFormat.NL_X_NB_BS_HS,
        layer_idx=0,
        num_layers_in_group=3,
        num_blocks=32,
        block_size=16,
    )
    assert sd.kv_size == 1
    assert sd.nh == 1
    assert sd.hs == 512


def test_make_shape_desc_sglang_mla():
    kv_caches = [torch.empty(512, 1, 128, dtype=torch.bfloat16) for _ in range(2)]
    sd = make_page_buffer_shape_desc(
        kv_caches,
        lmc_ops.EngineKVFormat.NL_X_NBBS_ONE_HS,
        layer_idx=0,
        num_layers_in_group=2,
        num_blocks=32,
        block_size=16,
    )
    assert sd.kv_size == 1
    assert sd.nh == 1
    assert sd.hs == 128


def test_make_shape_desc_sglang_mha():
    k = [torch.empty(512, 8, 64, dtype=torch.bfloat16) for _ in range(4)]
    v = [torch.empty(512, 8, 64, dtype=torch.bfloat16) for _ in range(4)]
    kv_caches = [k, v]
    sd = make_page_buffer_shape_desc(
        kv_caches,
        lmc_ops.EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS,
        layer_idx=0,
        num_layers_in_group=4,
        num_blocks=32,
        block_size=16,
    )
    assert sd.kv_size == 2
    assert sd.nh == 8
    assert sd.hs == 64


def test_per_layer_scalar_accessors_per_layer_list():
    """For per-layer list formats, each scalar accessor honours layer_idx."""
    kv_caches = [
        torch.randn(2, 32, 16, 8 + i, 64, dtype=torch.float16, device="cuda")
        for i in range(3)  # distinct num_heads per layer: 8, 9, 10
    ]
    fmt = lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS

    assert get_num_heads(kv_caches, fmt, layer_idx=0) == 8
    assert get_num_heads(kv_caches, fmt, layer_idx=2) == 10
    assert get_head_size(kv_caches, fmt, layer_idx=1) == 64
    assert get_dtype(kv_caches, fmt, layer_idx=0) == torch.float16


def test_per_layer_scalar_accessors_sglang_mha():
    """For SGLang MHA (nested list), accessors walk into [k_list|v_list]."""
    k = [torch.randn(512, 8, 64, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    v = [torch.randn(512, 8, 64, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    kv_caches = [k, v]
    fmt = lmc_ops.EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS

    assert get_num_heads(kv_caches, fmt, layer_idx=0) == 8
    assert get_dtype(kv_caches, fmt, layer_idx=1) == torch.bfloat16


def test_get_group_data_ptrs_per_layer_list_flattens_in_order():
    kv_caches = [
        torch.randn(2, 32, 16, 8, 64, dtype=torch.float16, device="cuda")
        for _ in range(4)
    ]
    fmt = lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS

    ptrs = get_group_data_ptrs(kv_caches, fmt, [0, 2, 3])
    assert ptrs == [
        kv_caches[0].data_ptr(),
        kv_caches[2].data_ptr(),
        kv_caches[3].data_ptr(),
    ]


def test_get_group_data_ptrs_sglang_mha_groups_k_before_v():
    """SGLang MHA kernel contract: [K0, K1, ..., KN, V0, V1, ..., VN] —
    not per-layer [K0, V0, K1, V1, ...]."""
    k = [torch.randn(512, 8, 64, dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    v = [torch.randn(512, 8, 64, dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    kv_caches = [k, v]
    fmt = lmc_ops.EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS

    ptrs = get_group_data_ptrs(kv_caches, fmt, [0, 1, 2])
    expected = [
        k[0].data_ptr(),
        k[1].data_ptr(),
        k[2].data_ptr(),
        v[0].data_ptr(),
        v[1].data_ptr(),
        v[2].data_ptr(),
    ]
    assert ptrs == expected


def test_get_group_data_ptrs_cross_layer_returns_single_base():
    """Cross-layer format packs every layer into one tensor; the kernel
    (csrc/mp_mem_kernels.cu) reads paged_buffer_ptrs[0] and computes
    per-layer offsets from shape_desc.nl internally. The group helper
    must return a single base pointer, not num_layers entries."""
    big = torch.empty(32, 80, 2, 16, 8, 64, dtype=torch.bfloat16, device="cuda")
    fmt = lmc_ops.EngineKVFormat.NB_NL_TWO_BS_NH_HS
    ptrs = get_group_data_ptrs(big, fmt, list(range(80)))
    assert ptrs == [big.data_ptr()]


def test_attempt_permute_preserves_bare_tensor():
    """A bare torch.Tensor input (cross-layer shape) must come back as a bare
    tensor — no list wrapping, no copy, shape intact — so the
    DiscoverableKVCache recursive union is respected end-to-end."""
    big = torch.empty(32, 80, 2, 16, 8, 64, dtype=torch.bfloat16, device="cuda")
    out = attempt_permute_to_contiguous_view(big)
    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == tuple(big.shape)
    assert out.data_ptr() == big.data_ptr()


def test_attempt_permute_recurses_all_shapes():
    """attempt_permute_to_contiguous_view must descend into every
    DiscoverableKVCache shape and permute non-contiguous tensor leaves."""
    # Build a flash-attention HND-layout tensor (non-contiguous after
    # logical→physical permute exposure — the vLLM HND case).
    nhd_view = (
        torch.empty(2, 32, 8, 16, 64, dtype=torch.bfloat16, device="cuda")
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .permute(0, 1, 3, 2, 4)  # NHD logical view over HND physical
    )
    assert not nhd_view.is_contiguous()

    # bare tensor
    out_tensor = attempt_permute_to_contiguous_view(nhd_view)
    assert out_tensor.is_contiguous()

    # flat list
    lst = [nhd_view.clone() for _ in range(2)]
    out_list = attempt_permute_to_contiguous_view(lst)
    assert isinstance(out_list, list)
    assert all(t.is_contiguous() for t in out_list)

    # nested list (SGLang-shaped)
    k = [nhd_view.clone() for _ in range(2)]
    v = [nhd_view.clone() for _ in range(2)]
    out_nested = attempt_permute_to_contiguous_view([k, v])
    assert isinstance(out_nested, list)
    assert all(t.is_contiguous() for sublist in out_nested for t in sublist)


def test_attempt_permute_reorders_misleading_size_one_dims():
    """A single-head HND-physical tensor exposed as an NHD logical view is
    torch-contiguous (size-1 dims are stride-exempt), so a plain contiguity
    check would pass it through with a shape that lies about the physical
    layout. The stride sort must still reorder it to the stride-truthful
    HND shape, zero-copy."""
    phys = torch.empty(2, 32, 1, 16, 64, dtype=torch.bfloat16, device="cuda")
    logical = phys.permute(0, 1, 3, 2, 4)  # [2, NB, BS, 1, HS]
    assert logical.is_contiguous()

    out = attempt_permute_to_contiguous_view(logical)
    assert tuple(out.shape) == (2, 32, 1, 16, 64)
    assert out.data_ptr() == phys.data_ptr()


def test_attempt_permute_reviews_collapsed_size_one_strides():
    """A contiguous view whose size-1 dims carry collapsed (stride-1) strides
    hides the true extent of its inner dim; the re-view must recover the
    tightest shape implied by the strides, zero-copy."""
    base = torch.empty(8 * 32, dtype=torch.bfloat16, device="cuda")
    t = base.as_strided((8, 1, 1, 32), (32, 1, 1, 1))
    assert t.is_contiguous()

    out = attempt_permute_to_contiguous_view(t)
    assert tuple(out.shape) == (8, 32, 1, 1)
    assert out.data_ptr() == base.data_ptr()


def test_get_device_handles_every_kvcaches_shape():
    """get_device must work for every DiscoverableKVCache shape without format hints."""
    t = torch.empty(8, dtype=torch.bfloat16, device="cuda")
    assert get_device(t) == t.device

    flat = [torch.empty(4, dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    assert get_device(flat) == flat[0].device

    k = [torch.empty(4, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    v = [torch.empty(4, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    assert get_device([k, v]) == k[0].device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
