# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx1x, on_gfx12x
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops import chunked_prefill_paged_decode as paged_decode_ops
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    _choose_fallback_block_size,
    _paged_attention_2d_splitkv_decode,
    kernel_paged_attention_2d,
    reserve_splitkv_workspace,
)
from vllm.v1.attention.ops.rdna4_splitkv import (
    can_use_rdna4_flydsl_splitkv_paged_attention,
)
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    lock_workspace,
    reset_workspace_manager,
)

DEVICE = current_platform.device_type


@dataclass(frozen=True)
class SplitKVCase:
    query_dtype: torch.dtype
    kv_dtype: torch.dtype
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    page_size: int
    seq_lens: tuple[int, ...]
    splits: int | None
    k_scale: float = 1.0
    v_scale: float = 1.0
    check_torch_reference: bool = False
    padded_stride: bool = False


CASES = [
    pytest.param(
        SplitKVCase(torch.bfloat16, torch.bfloat16, 4, 4, 128, 16, (257,), 4),
        id="native-bf16-mha",
    ),
    pytest.param(
        SplitKVCase(
            torch.float16,
            torch.float16,
            16,
            4,
            128,
            32,
            (257, 513, 1025),
            4,
        ),
        id="native-fp16-gqa4",
    ),
    pytest.param(
        SplitKVCase(
            torch.bfloat16,
            torch.bfloat16,
            12,
            2,
            256,
            1568,
            (1567, 1568, 1569),
            7,
            check_torch_reference=True,
            padded_stride=True,
        ),
        id="native-qwen-page-boundary",
    ),
    pytest.param(
        SplitKVCase(
            torch.bfloat16,
            torch.float8_e4m3fn,
            12,
            2,
            256,
            1568,
            (4014,),
            1,
            0.73,
            1.27,
            True,
            True,
        ),
        id="fp8-qwen-one-split",
    ),
    pytest.param(
        SplitKVCase(
            torch.bfloat16,
            torch.float8_e4m3fn,
            12,
            2,
            256,
            1568,
            (8192,),
            14,
            0.73,
            1.27,
            padded_stride=True,
        ),
        id="fp8-qwen-long",
    ),
    pytest.param(
        SplitKVCase(
            torch.bfloat16,
            torch.float8_e4m3fn,
            12,
            2,
            256,
            1568,
            (32768, 16384, 8192),
            14,
            0.73,
            1.27,
            padded_stride=True,
        ),
        id="fp8-qwen-ragged",
    ),
    pytest.param(
        SplitKVCase(
            torch.float16,
            torch.float8_e4m3fn,
            8,
            1,
            128,
            528,
            (527, 528, 529),
            7,
            0.5,
            1.5,
        ),
        id="fp8-fp16-mqa-odd-page",
    ),
    pytest.param(
        SplitKVCase(
            torch.bfloat16,
            torch.float8_e4m3fn,
            16,
            4,
            128,
            784,
            (783, 784, 785),
            16,
            0.75,
            1.25,
        ),
        id="fp8-empty-splits",
    ),
    pytest.param(
        SplitKVCase(
            torch.bfloat16,
            torch.float8_e4m3fn,
            5,
            1,
            256,
            1056,
            (1055, 1056, 1057),
            4,
            0.75,
            1.25,
        ),
        id="fp8-odd-head-group",
    ),
    pytest.param(
        SplitKVCase(
            torch.bfloat16,
            torch.float8_e4m3fn,
            4,
            4,
            128,
            32,
            (2049, 4097, 8193),
            None,
            0.75,
            1.25,
        ),
        id="fp8-production-heuristic",
    ),
]


def _pack_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    padded_stride: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks, page_size, num_kv_heads, head_size = key_cache.shape
    x = 16 // key_cache.element_size()
    packed_key = (
        key_cache.view(num_blocks, page_size, num_kv_heads, head_size // x, x)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    packed_value = value_cache.permute(0, 2, 3, 1).contiguous()
    if not padded_stride:
        return packed_key, packed_value

    page_elements = packed_key[0].numel()
    backing = torch.empty(
        num_blocks * 2 * page_elements,
        dtype=packed_key.dtype,
        device=packed_key.device,
    )
    key_cache = torch.as_strided(
        backing,
        size=packed_key.shape,
        stride=(2 * page_elements, *packed_key.stride()[1:]),
    )
    value_cache = torch.as_strided(
        backing,
        size=packed_value.shape,
        stride=(2 * page_elements, *packed_value.stride()[1:]),
        storage_offset=page_elements,
    )
    key_cache.copy_(packed_key)
    value_cache.copy_(packed_value)
    return key_cache, value_cache


def _make_inputs(case: SplitKVCase):
    batch_size = len(case.seq_lens)
    blocks_per_seq = [
        (seq_len + case.page_size - 1) // case.page_size for seq_len in case.seq_lens
    ]
    num_blocks = sum(blocks_per_seq)
    max_blocks = max(blocks_per_seq)

    query = torch.randn(
        batch_size,
        case.num_query_heads,
        case.head_size,
        dtype=case.query_dtype,
        device=DEVICE,
    )
    dense_key = torch.randn(
        num_blocks,
        case.page_size,
        case.num_kv_heads,
        case.head_size,
        dtype=torch.bfloat16,
        device=DEVICE,
    ).to(case.kv_dtype)
    dense_value = torch.randn(
        num_blocks,
        case.page_size,
        case.num_kv_heads,
        case.head_size,
        dtype=torch.bfloat16,
        device=DEVICE,
    ).to(case.kv_dtype)

    permutation = torch.randperm(num_blocks, device=DEVICE, dtype=torch.int64)
    block_tables = torch.zeros(batch_size, max_blocks, dtype=torch.int32, device=DEVICE)
    cursor = 0
    for seq_idx, (seq_len, num_seq_blocks) in enumerate(
        zip(case.seq_lens, blocks_per_seq)
    ):
        physical_blocks = permutation[cursor : cursor + num_seq_blocks]
        block_tables[seq_idx, :num_seq_blocks] = physical_blocks.to(torch.int32)
        cursor += num_seq_blocks
        tail = seq_len % case.page_size
        if tail:
            last_block = physical_blocks[-1]
            dense_key[last_block, tail:] = float("nan")
            dense_value[last_block, tail:] = float("nan")

    key_cache, value_cache = _pack_cache(
        dense_key, dense_value, padded_stride=case.padded_stride
    )
    seq_lens = torch.tensor(case.seq_lens, dtype=torch.int32, device=DEVICE)
    k_scale = torch.tensor(case.k_scale, dtype=torch.float32, device=DEVICE)
    v_scale = torch.tensor(case.v_scale, dtype=torch.float32, device=DEVICE)
    return (
        query,
        dense_key,
        dense_value,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    )


def _run_non_split(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    filter_by_query_len: bool = False,
) -> torch.Tensor:
    if output is None:
        output = torch.empty_like(query)
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    head_size = query.shape[2]
    page_size = key_cache.shape[3]
    block_size = _choose_fallback_block_size(page_size)
    num_queries_per_kv = num_query_heads // num_kv_heads

    kernel_paged_attention_2d[(seq_lens.shape[0], num_kv_heads)](
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        sink_ptr=None,
        block_tables_ptr=block_tables,
        seq_lens_ptr=seq_lens,
        alibi_slopes_ptr=None,
        scale=scale,
        k_scale=k_scale,
        v_scale=v_scale,
        out_scale_inv=1.0,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        num_queries_per_kv_padded=max(triton.next_power_of_2(num_queries_per_kv), 16),
        block_table_stride=block_tables.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        BLOCK_SIZE=block_size,
        PHYSICAL_BLOCK_SIZE=page_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=False,
        SLIDING_WINDOW=0,
        x=key_cache.shape[4],
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        filter_by_query_len=filter_by_query_len,
        query_start_len_ptr=query_start_loc,
        USE_SINKS=False,
        USE_FP8=False,
        num_warps=8 if key_cache.element_size() == 1 else 4,
    )
    return output


def _torch_reference(
    query: torch.Tensor,
    dense_key: torch.Tensor,
    dense_value: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    k_scale: float,
    v_scale: float,
) -> torch.Tensor:
    outputs = []
    page_size = dense_key.shape[1]
    num_kv_heads = dense_key.shape[2]
    repeat = query.shape[1] // num_kv_heads
    for seq_idx, seq_len_tensor in enumerate(seq_lens):
        seq_len = int(seq_len_tensor.item())
        num_blocks = (seq_len + page_size - 1) // page_size
        block_ids = block_tables[seq_idx, :num_blocks].long()
        key = dense_key[block_ids].reshape(-1, num_kv_heads, query.shape[2])
        value = dense_value[block_ids].reshape(-1, num_kv_heads, query.shape[2])
        key = key[:seq_len].float() * k_scale
        value = value[:seq_len].float() * v_scale
        key = torch.repeat_interleave(key, repeat, dim=1)
        value = torch.repeat_interleave(value, repeat, dim=1)
        scores = torch.einsum("hd,shd->hs", query[seq_idx].float(), key) * scale
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hs,shd->hd", probabilities, value))
    return torch.stack(outputs).to(query.dtype)


@pytest.mark.skipif(not on_gfx12x(), reason="FlyDSL SplitKV requires gfx12x")
@torch.inference_mode()
def test_rdna4_flydsl_gate_covers_generalized_batch_one() -> None:
    """The FlyDSL gate covers fallback shapes without widening the HIP gate."""
    case = SplitKVCase(
        torch.bfloat16,
        torch.float8_e4m3fn,
        7,
        1,
        256,
        1056,
        (1057,),
        4,
        0.73,
        1.27,
    )
    (
        query,
        _,
        _,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    output = torch.empty_like(query)
    gate_args = dict(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        output=output,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        k_scale=k_scale,
        v_scale=v_scale,
        scale=0.071,
        actual_max_splits=4,
        max_seq_len=1057,
        filter_by_query_len=True,
    )

    assert can_use_rdna4_flydsl_splitkv_paged_attention(**gate_args)
    assert not can_use_rdna4_flydsl_splitkv_paged_attention(
        **(gate_args | {"seq_lens": seq_lens.repeat(2)})
    )
    assert can_use_rdna4_flydsl_splitkv_paged_attention(
        **(gate_args | {"query": query[:, :5], "output": output[:, :5]})
    )


@pytest.mark.parametrize(
    "query_dtype,kv_dtype,batch_size,num_query_heads,num_kv_heads,head_size",
    [
        (torch.float16, torch.float8_e4m3fn, 1, 9, 1, 128),
    ],
)
def test_rdna4_flydsl_rejects_value_sensitive_experimental_routes(
    query_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    batch_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
) -> None:
    """Do not expose routes whose correctness depends on input magnitude."""
    from vllm.v1.attention.ops.flydsl_kernels.rdna4_splitkv import (
        select_kernel_config,
    )

    page_size = 128
    cache_pack = 16 // kv_dtype.itemsize
    query = torch.empty(
        batch_size, num_query_heads, head_size, dtype=query_dtype, device="meta"
    )
    key_cache = torch.empty(
        64,
        num_kv_heads,
        head_size // cache_pack,
        page_size,
        cache_pack,
        dtype=kv_dtype,
        device="meta",
    )
    seq_lens = torch.empty(batch_size, dtype=torch.int32, device="meta")

    assert select_kernel_config(query, key_cache, seq_lens, 8192) is None


@pytest.mark.skipif(not on_gfx12x(), reason="FlyDSL SplitKV requires gfx12x")
@torch.inference_mode()
def test_rdna4_flydsl_d128_gqa1_wave8_matches_triton(monkeypatch) -> None:
    """The faster fused Wave8 route supersedes direct-finalize for D128/GQA1."""
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops
    from vllm.v1.attention.ops.flydsl_kernels.rdna4_splitkv import (
        SplitKVRoute,
        select_kernel_config,
    )

    set_random_seed(0)
    case = SplitKVCase(
        torch.bfloat16,
        torch.float8_e4m3fn,
        2,
        2,
        128,
        544,
        (1024,),
        4,
        0.73,
        1.27,
    )
    (
        query,
        _,
        _,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    output = torch.empty_like(query)
    mid_out = torch.empty((1, 2, 4, 128), dtype=torch.float32, device=DEVICE)
    mid_lse = torch.empty((1, 2, 4), dtype=torch.float32, device=DEVICE)
    scale = case.head_size**-0.5

    config = select_kernel_config(query, key_cache, seq_lens, max(case.seq_lens))
    assert config is not None
    assert config.route == SplitKVRoute.WAVE8
    monkeypatch.setattr(rdna4_ops.envs, "VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL", True)
    _paged_attention_2d_splitkv_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
        output=output,
        actual_max_splits=case.splits,
        mid_out=mid_out,
        mid_lse=mid_lse,
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )
    reference = _run_non_split(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )

    torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not on_gfx12x(), reason="FlyDSL SplitKV requires gfx12x")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_rdna4_flydsl_native_d128_gqa16_matches_triton(
    monkeypatch, dtype: torch.dtype
) -> None:
    """Exercise the four-wave direct-register schedule for both native types."""
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops
    from vllm.v1.attention.ops.flydsl_kernels.rdna4_splitkv import (
        SplitKVRoute,
        select_kernel_config,
    )

    set_random_seed(0)
    case = SplitKVCase(
        dtype,
        dtype,
        32,
        2,
        128,
        16,
        (257, 241, 225, 209, 193, 177, 161, 145),
        4,
    )
    (
        query,
        dense_key,
        dense_value,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query_start_loc = torch.arange(9, dtype=torch.int32, device=DEVICE)
    output = torch.empty_like(query)
    mid_out = torch.empty((8, 32, 4, 128), dtype=torch.float32, device=DEVICE)
    mid_lse = torch.empty((8, 32, 4), dtype=torch.float32, device=DEVICE)
    scale = case.head_size**-0.5

    config = select_kernel_config(query, key_cache, seq_lens, 257)
    assert config is not None
    assert config.route == SplitKVRoute.NATIVE_D128_GQA16_DIRECT
    monkeypatch.setattr(rdna4_ops.envs, "VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL", True)
    _paged_attention_2d_splitkv_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
        output=output,
        actual_max_splits=case.splits,
        mid_out=mid_out,
        mid_lse=mid_lse,
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )
    reference = _torch_reference(
        query,
        dense_key,
        dense_value,
        block_tables,
        seq_lens,
        scale,
        1.0,
        1.0,
    )

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not on_gfx12x(), reason="FlyDSL SplitKV requires gfx12x")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_rdna4_flydsl_native_d128_gqa16_lds_matches_torch(
    monkeypatch, dtype: torch.dtype
) -> None:
    """Exercise the batch-16 cooperative LDS K/V schedule."""
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops
    from vllm.v1.attention.ops.flydsl_kernels.rdna4_splitkv import (
        SplitKVRoute,
        select_kernel_config,
    )

    set_random_seed(1)
    case = SplitKVCase(
        dtype,
        dtype,
        32,
        2,
        128,
        16,
        (
            257,
            249,
            241,
            233,
            225,
            217,
            209,
            201,
            193,
            185,
            177,
            169,
            161,
            153,
            145,
            137,
        ),
        4,
    )
    (
        query,
        dense_key,
        dense_value,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query_start_loc = torch.arange(17, dtype=torch.int32, device=DEVICE)
    output = torch.empty_like(query)
    mid_out = torch.empty((16, 32, 4, 128), dtype=torch.float32, device=DEVICE)
    mid_lse = torch.empty((16, 32, 4), dtype=torch.float32, device=DEVICE)
    scale = case.head_size**-0.5

    config = select_kernel_config(query, key_cache, seq_lens, 257)
    assert config is not None
    assert config.route == SplitKVRoute.NATIVE_D128_GQA16_LDS
    monkeypatch.setattr(rdna4_ops.envs, "VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL", True)
    _paged_attention_2d_splitkv_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
        output=output,
        actual_max_splits=case.splits,
        mid_out=mid_out,
        mid_lse=mid_lse,
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )
    reference = _torch_reference(
        query,
        dense_key,
        dense_value,
        block_tables,
        seq_lens,
        scale,
        1.0,
        1.0,
    )

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not on_gfx12x(), reason="FlyDSL SplitKV requires gfx12x")
@pytest.mark.parametrize(
    "query_dtype,kv_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.float8_e4m3fn),
        (torch.float16, torch.float8_e4m3fnuz),
    ],
)
@torch.inference_mode()
def test_rdna4_flydsl_d128_gqa16_tile32_matches_torch(
    monkeypatch, query_dtype: torch.dtype, kv_dtype: torch.dtype
) -> None:
    """Exercise the batch-one 16-wave register-PV schedule."""
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops
    from vllm.v1.attention.ops.flydsl_kernels.rdna4_splitkv import (
        SplitKVRoute,
        select_kernel_config,
    )

    set_random_seed(2)
    case = SplitKVCase(
        query_dtype,
        kv_dtype,
        32,
        2,
        128,
        32,
        (257,),
        4,
        0.73,
        1.27,
    )
    (
        query,
        dense_key,
        dense_value,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    output = torch.empty_like(query)
    mid_out = torch.empty((1, 32, 4, 128), dtype=torch.float32, device=DEVICE)
    mid_lse = torch.empty((1, 32, 4), dtype=torch.float32, device=DEVICE)
    scale = case.head_size**-0.5

    config = select_kernel_config(query, key_cache, seq_lens, 257)
    assert config is not None
    assert config.route == SplitKVRoute.D128_GQA16_TILE32
    monkeypatch.setattr(rdna4_ops.envs, "VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL", True)
    _paged_attention_2d_splitkv_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
        output=output,
        actual_max_splits=case.splits,
        mid_out=mid_out,
        mid_lse=mid_lse,
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )
    reference = _torch_reference(
        query,
        dense_key,
        dense_value,
        block_tables,
        seq_lens,
        scale,
        case.k_scale if kv_dtype.itemsize == 1 else 1.0,
        case.v_scale if kv_dtype.itemsize == 1 else 1.0,
    )

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not on_gfx12x(), reason="FlyDSL SplitKV requires gfx12x")
@pytest.mark.parametrize(
    "case",
    [
        SplitKVCase(
            torch.bfloat16,
            torch.bfloat16,
            16,
            2,
            256,
            128,
            (257, 249, 241),
            4,
        ),
        SplitKVCase(
            torch.float16,
            torch.float16,
            8,
            2,
            256,
            32,
            (257, 249, 241, 233, 225, 217, 209, 201),
            4,
        ),
        SplitKVCase(
            torch.bfloat16,
            torch.float8_e4m3fnuz,
            16,
            2,
            256,
            32,
            (257,),
            4,
            0.73,
            1.27,
        ),
    ],
    ids=("native-bf16-gqa8", "native-fp16-gqa4", "fnuz-bf16-gqa8"),
)
@torch.inference_mode()
def test_rdna4_flydsl_d256_gqa4_8_matches_torch(monkeypatch, case: SplitKVCase) -> None:
    """Exercise the promoted D256 eight-wave two-output-block schedule."""
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops
    from vllm.v1.attention.ops.flydsl_kernels.rdna4_splitkv import (
        SplitKVRoute,
        select_kernel_config,
    )

    set_random_seed(3)
    (
        query,
        dense_key,
        dense_value,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    batch_size = len(case.seq_lens)
    query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32, device=DEVICE)
    output = torch.empty_like(query)
    mid_out = torch.empty(
        (batch_size, case.num_query_heads, case.splits, 256),
        dtype=torch.float32,
        device=DEVICE,
    )
    mid_lse = torch.empty(
        (batch_size, case.num_query_heads, case.splits),
        dtype=torch.float32,
        device=DEVICE,
    )
    scale = case.head_size**-0.5

    config = select_kernel_config(query, key_cache, seq_lens, max(case.seq_lens))
    assert config is not None
    assert config.route == SplitKVRoute.D256_GQA4_8_TILE32
    monkeypatch.setattr(rdna4_ops.envs, "VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL", True)
    _paged_attention_2d_splitkv_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
        output=output,
        actual_max_splits=case.splits,
        mid_out=mid_out,
        mid_lse=mid_lse,
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )
    reference = _torch_reference(
        query,
        dense_key,
        dense_value,
        block_tables,
        seq_lens,
        scale,
        case.k_scale if case.kv_dtype.itemsize == 1 else 1.0,
        case.v_scale if case.kv_dtype.itemsize == 1 else 1.0,
    )

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not on_gfx12x(), reason="FlyDSL SplitKV requires gfx12x")
@pytest.mark.parametrize(
    "query_dtype,kv_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.float8_e4m3fn),
        (torch.float16, torch.float8_e4m3fnuz),
    ],
)
@torch.inference_mode()
def test_rdna4_flydsl_d256_gqa16_matches_torch(
    monkeypatch, query_dtype: torch.dtype, kv_dtype: torch.dtype
) -> None:
    """Exercise the promoted D256/GQA16 Tile32 register-PV schedule."""
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops
    from vllm.v1.attention.ops.flydsl_kernels.rdna4_splitkv import (
        SplitKVRoute,
        select_kernel_config,
    )

    set_random_seed(4)
    case = SplitKVCase(
        query_dtype,
        kv_dtype,
        32,
        2,
        256,
        32,
        (257,),
        4,
        0.73,
        1.27,
    )
    (
        query,
        dense_key,
        dense_value,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    output = torch.empty_like(query)
    mid_out = torch.empty((1, 32, 4, 256), dtype=torch.float32, device=DEVICE)
    mid_lse = torch.empty((1, 32, 4), dtype=torch.float32, device=DEVICE)
    scale = case.head_size**-0.5

    config = select_kernel_config(query, key_cache, seq_lens, 257)
    assert config is not None
    assert config.route == SplitKVRoute.D256_GQA16_TILE32
    monkeypatch.setattr(rdna4_ops.envs, "VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL", True)
    _paged_attention_2d_splitkv_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
        output=output,
        actual_max_splits=case.splits,
        mid_out=mid_out,
        mid_lse=mid_lse,
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )
    reference = _torch_reference(
        query,
        dense_key,
        dense_value,
        block_tables,
        seq_lens,
        scale,
        case.k_scale if kv_dtype.itemsize == 1 else 1.0,
        case.v_scale if kv_dtype.itemsize == 1 else 1.0,
    )

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not on_gfx12x(), reason="FlyDSL SplitKV requires gfx12x")
@pytest.mark.parametrize(
    "query_dtype,kv_dtype",
    [
        (torch.bfloat16, torch.float8_e4m3fn),
        (torch.float16, torch.float8_e4m3fnuz),
    ],
)
@torch.inference_mode()
def test_rdna4_flydsl_d128_gqa8_matches_torch(
    monkeypatch, query_dtype: torch.dtype, kv_dtype: torch.dtype
) -> None:
    """Exercise the promoted D128/GQA8 Tile32 register-PV schedule."""
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops
    from vllm.v1.attention.ops.flydsl_kernels.rdna4_splitkv import (
        SplitKVRoute,
        select_kernel_config,
    )

    set_random_seed(5)
    case = SplitKVCase(
        query_dtype,
        kv_dtype,
        16,
        2,
        128,
        32,
        (257, 249, 241),
        4,
        0.73,
        1.27,
    )
    (
        query,
        dense_key,
        dense_value,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query_start_loc = torch.arange(4, dtype=torch.int32, device=DEVICE)
    output = torch.empty_like(query)
    mid_out = torch.empty((3, 16, 4, 128), dtype=torch.float32, device=DEVICE)
    mid_lse = torch.empty((3, 16, 4), dtype=torch.float32, device=DEVICE)
    scale = case.head_size**-0.5

    config = select_kernel_config(query, key_cache, seq_lens, 257)
    assert config is not None
    assert config.route == SplitKVRoute.D128_GQA8_TILE32
    monkeypatch.setattr(rdna4_ops.envs, "VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL", True)
    _paged_attention_2d_splitkv_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
        output=output,
        actual_max_splits=case.splits,
        mid_out=mid_out,
        mid_lse=mid_lse,
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )
    reference = _torch_reference(
        query,
        dense_key,
        dense_value,
        block_tables,
        seq_lens,
        scale,
        case.k_scale,
        case.v_scale,
    )

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not on_gfx12x(), reason="FlyDSL SplitKV requires gfx12x")
@pytest.mark.parametrize(
    "case",
    [
        SplitKVCase(
            torch.bfloat16,
            torch.float8_e4m3fn,
            5,
            1,
            256,
            16,
            (257,),
            8,
            0.73,
            1.27,
        ),
        SplitKVCase(
            torch.bfloat16,
            torch.float8_e4m3fn,
            13,
            1,
            256,
            16,
            (257,),
            8,
            0.73,
            1.27,
        ),
        SplitKVCase(
            torch.bfloat16,
            torch.float8_e4m3fn,
            14,
            1,
            256,
            32,
            (257,),
            4,
            0.73,
            1.27,
        ),
        SplitKVCase(
            torch.bfloat16,
            torch.bfloat16,
            8,
            1,
            128,
            16,
            (257,),
            4,
        ),
    ],
    ids=("d256-gqa5", "d256-gqa13", "d256-gqa14", "d128-native-gqa8"),
)
@torch.inference_mode()
def test_rdna4_flydsl_generic_tile32_matches_torch(
    monkeypatch, case: SplitKVCase
) -> None:
    """Exercise accurate Tile32 coverage beyond the fixed named routes."""
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops
    from vllm.v1.attention.ops.flydsl_kernels.rdna4_splitkv import (
        SplitKVRoute,
        select_kernel_config,
    )

    set_random_seed(6)
    (
        query,
        dense_key,
        dense_value,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    output = torch.empty_like(query)
    mid_out = torch.empty(
        (1, case.num_query_heads, case.splits, case.head_size),
        dtype=torch.float32,
        device=DEVICE,
    )
    mid_lse = torch.empty(
        (1, case.num_query_heads, case.splits),
        dtype=torch.float32,
        device=DEVICE,
    )
    scale = case.head_size**-0.5

    config = select_kernel_config(query, key_cache, seq_lens, 257)
    assert config is not None
    assert config.route == SplitKVRoute.GENERIC_TILE32
    monkeypatch.setattr(rdna4_ops.envs, "VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL", True)
    _paged_attention_2d_splitkv_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
        output=output,
        actual_max_splits=case.splits,
        mid_out=mid_out,
        mid_lse=mid_lse,
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )
    reference = _torch_reference(
        query,
        dense_key,
        dense_value,
        block_tables,
        seq_lens,
        scale,
        case.k_scale if case.kv_dtype.itemsize == 1 else 1.0,
        case.v_scale if case.kv_dtype.itemsize == 1 else 1.0,
    )

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not on_gfx12x(), reason="FlyDSL SplitKV requires gfx12x")
@pytest.mark.parametrize(
    "case",
    [
        SplitKVCase(
            torch.bfloat16,
            torch.float8_e4m3fn,
            1,
            1,
            256,
            16,
            (257,),
            8,
            0.73,
            1.27,
        ),
        SplitKVCase(
            torch.bfloat16,
            torch.float8_e4m3fn,
            3,
            1,
            256,
            128,
            (257,),
            4,
            0.73,
            1.27,
        ),
        SplitKVCase(
            torch.float16,
            torch.float8_e4m3fnuz,
            4,
            1,
            128,
            32,
            (257,),
            4,
            0.73,
            1.27,
        ),
        SplitKVCase(
            torch.float16,
            torch.float16,
            1,
            1,
            128,
            16,
            (257,),
            4,
        ),
    ],
    ids=("d256-fp8-gqa1", "d256-fp8-gqa3", "d128-fnuz-gqa4", "d128-native-gqa1"),
)
@torch.inference_mode()
def test_rdna4_flydsl_wave8_matches_torch(monkeypatch, case: SplitKVCase) -> None:
    """Exercise the accurate scalar-FP32 low-GQA fallback."""
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops
    from vllm.v1.attention.ops.flydsl_kernels.rdna4_splitkv import (
        SplitKVRoute,
        select_kernel_config,
    )

    set_random_seed(7)
    (
        query,
        dense_key,
        dense_value,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    output = torch.empty_like(query)
    mid_out = torch.empty(
        (1, case.num_query_heads, case.splits, case.head_size),
        dtype=torch.float32,
        device=DEVICE,
    )
    mid_lse = torch.empty(
        (1, case.num_query_heads, case.splits),
        dtype=torch.float32,
        device=DEVICE,
    )
    scale = case.head_size**-0.5

    config = select_kernel_config(query, key_cache, seq_lens, 257)
    assert config is not None
    assert config.route == SplitKVRoute.WAVE8
    monkeypatch.setattr(rdna4_ops.envs, "VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL", True)
    _paged_attention_2d_splitkv_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
        output=output,
        actual_max_splits=case.splits,
        mid_out=mid_out,
        mid_lse=mid_lse,
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )
    reference = _torch_reference(
        query,
        dense_key,
        dense_value,
        block_tables,
        seq_lens,
        scale,
        case.k_scale if case.kv_dtype.itemsize == 1 else 1.0,
        case.v_scale if case.kv_dtype.itemsize == 1 else 1.0,
    )

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not on_gfx12x(), reason="FlyDSL SplitKV requires gfx12x")
@torch.inference_mode()
def test_rdna4_flydsl_qwen38_tp2_generic_matches_triton(monkeypatch) -> None:
    """Exercise the accurate D256/GQA6 path used by Qwen3.8 TP2."""
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops
    from vllm.v1.attention.ops.flydsl_kernels.rdna4_splitkv import (
        SplitKVRoute,
        select_kernel_config,
    )

    set_random_seed(0)
    case = SplitKVCase(
        torch.bfloat16,
        torch.float8_e4m3fn,
        12,
        2,
        256,
        1568,
        (4014,),
        4,
        0.73,
        1.27,
        padded_stride=True,
    )
    (
        query,
        _,
        _,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    output = torch.empty_like(query)
    mid_out = torch.empty((1, 12, case.splits, 256), dtype=torch.float32, device=DEVICE)
    mid_lse = torch.empty((1, 12, case.splits), dtype=torch.float32, device=DEVICE)
    scale = case.head_size**-0.5

    config = select_kernel_config(query, key_cache, seq_lens, 4014)
    assert config is not None
    assert config.route == SplitKVRoute.GENERIC_TILE32
    monkeypatch.setattr(rdna4_ops.envs, "VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL", True)
    _paged_attention_2d_splitkv_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
        output=output,
        actual_max_splits=case.splits,
        mid_out=mid_out,
        mid_lse=mid_lse,
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )
    reference = _run_non_split(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )

    torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not on_gfx12x(), reason="FlyDSL SplitKV requires gfx12x")
@torch.inference_mode()
def test_rdna4_flydsl_generic_preserves_prefill_rows(monkeypatch) -> None:
    """The generic reducer must not overwrite a non-decode query span."""
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops

    set_random_seed(0)
    case = SplitKVCase(
        torch.bfloat16,
        torch.float8_e4m3fn,
        12,
        2,
        256,
        1568,
        (4014,),
        4,
        0.73,
        1.27,
    )
    (
        _,
        _,
        _,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query = torch.randn(2, 12, 256, dtype=torch.bfloat16, device=DEVICE)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32, device=DEVICE)
    output = torch.full_like(query, 7.0)
    mid_out = torch.empty((1, 12, 4, 256), dtype=torch.float32, device=DEVICE)
    mid_lse = torch.empty((1, 12, 4), dtype=torch.float32, device=DEVICE)

    monkeypatch.setattr(rdna4_ops.envs, "VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL", True)
    _paged_attention_2d_splitkv_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        256**-0.5,
        k_scale,
        v_scale,
        output=output,
        actual_max_splits=case.splits,
        mid_out=mid_out,
        mid_lse=mid_lse,
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )

    torch.testing.assert_close(output, torch.full_like(output, 7.0))


@pytest.mark.skipif(not on_gfx1x(), reason="SplitKV decode requires gfx1x")
@pytest.mark.skipif(
    not torch.accelerator.is_available(), reason="SplitKV decode requires a GPU"
)
@pytest.mark.parametrize("case", CASES)
@torch.inference_mode()
def test_paged_attention_2d_splitkv_decode(case: SplitKVCase) -> None:
    if case.kv_dtype.itemsize == 1 and not on_gfx12x():
        pytest.skip("FP8 SplitKV decode requires gfx12x")
    set_random_seed(0)
    (
        query,
        dense_key,
        dense_value,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    scale = case.head_size**-0.5

    output = _paged_attention_2d_splitkv_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
        actual_max_splits=case.splits,
        max_seq_len=max(case.seq_lens),
    )
    reference = _run_non_split(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
    )

    assert output.dtype == query.dtype
    assert torch.isfinite(output).all()
    atol = get_default_atol(output)
    rtol = get_default_rtol(output)
    if case.kv_dtype.itemsize == 1:
        atol = max(atol, 0.01)
        rtol = max(rtol, 0.01)
    torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)

    if case.check_torch_reference:
        torch_reference = _torch_reference(
            query,
            dense_key,
            dense_value,
            block_tables,
            seq_lens,
            scale,
            case.k_scale if case.kv_dtype.itemsize == 1 else 1.0,
            case.v_scale if case.kv_dtype.itemsize == 1 else 1.0,
        )
        torch.testing.assert_close(output, torch_reference, atol=0.03, rtol=0.03)


@pytest.mark.skipif(not on_gfx12x(), reason="FP8 SplitKV decode requires gfx12x")
@pytest.mark.parametrize(
    "query_lens,seq_lens",
    [
        ((1, 3, 0, 1), (4014, 8192, 1568, 32768)),
        ((2, 4), (1568, 4014)),
    ],
)
@pytest.mark.parametrize("use_flydsl", [False, True])
@torch.inference_mode()
def test_fp8_splitkv_preserves_non_decode_rows(
    monkeypatch,
    query_lens: tuple[int, ...],
    seq_lens: tuple[int, ...],
    use_flydsl: bool,
) -> None:
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops

    monkeypatch.setattr(
        rdna4_ops.envs, "VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL", use_flydsl
    )
    set_random_seed(0)
    case = SplitKVCase(
        torch.bfloat16,
        torch.float8_e4m3fn,
        12,
        2,
        256,
        1568,
        seq_lens,
        16,
        0.73,
        1.27,
    )
    (
        _,
        _,
        _,
        key_cache,
        value_cache,
        block_tables,
        seq_lens_tensor,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query = torch.randn(sum(query_lens), 12, 256, dtype=torch.bfloat16, device=DEVICE)
    query_start_loc = torch.tensor(
        [0] + [sum(query_lens[: index + 1]) for index in range(len(query_lens))],
        dtype=torch.int32,
        device=DEVICE,
    )
    output = torch.full_like(query, 7.0)
    reference = output.clone()
    scale = 256**-0.5

    _paged_attention_2d_splitkv_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens_tensor,
        scale,
        k_scale,
        v_scale,
        output=output,
        actual_max_splits=16,
        max_seq_len=max(seq_lens),
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )
    _run_non_split(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens_tensor,
        scale,
        k_scale,
        v_scale,
        output=reference,
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )

    decode_rows = [
        int(query_start_loc[index].item())
        for index, query_len in enumerate(query_lens)
        if query_len == 1
    ]
    if decode_rows:
        torch.testing.assert_close(
            output[decode_rows], reference[decode_rows], atol=0.01, rtol=0.01
        )
    non_decode = torch.ones(sum(query_lens), dtype=torch.bool, device=DEVICE)
    non_decode[decode_rows] = False
    torch.testing.assert_close(
        output[non_decode], torch.full_like(output[non_decode], 7)
    )


@pytest.mark.skipif(not on_gfx12x(), reason="FP8 SplitKV decode requires gfx12x")
@torch.inference_mode()
def test_chunked_decode_routes_padded_fp8_cache_and_forwards_scales(
    monkeypatch,
) -> None:
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops

    set_random_seed(0)
    case = SplitKVCase(
        torch.bfloat16,
        torch.float8_e4m3fn,
        12,
        2,
        256,
        1568,
        (1374,),
        None,
        0.73,
        1.27,
        padded_stride=True,
    )
    (
        query,
        _,
        _,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    output = torch.empty_like(query)
    expected = _run_non_split(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        256**-0.5,
        k_scale,
        v_scale,
    )
    key_storage = key_cache.view(torch.uint8)
    value_storage = value_cache.view(torch.uint8)
    original = paged_decode_ops._paged_attention_2d_splitkv_decode
    forwarded = {}
    flydsl_routed = []

    def route_spy(*args, **kwargs):
        forwarded["k_scale"] = kwargs["k_scale"]
        forwarded["v_scale"] = kwargs["v_scale"]
        forwarded["cache_dtype"] = kwargs["key_cache"].dtype
        return original(*args, **kwargs)

    def flydsl_spy(*args, **kwargs):
        flydsl_routed.append(True)
        args[8].copy_(expected)

    monkeypatch.setattr(
        paged_decode_ops, "_paged_attention_2d_splitkv_decode", route_spy
    )
    monkeypatch.setattr(rdna4_ops.envs, "VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL", True)
    select_kernel_config, _ = rdna4_ops._load_flydsl_splitkv()
    monkeypatch.setattr(
        rdna4_ops,
        "_load_flydsl_splitkv",
        lambda: (select_kernel_config, flydsl_spy),
    )
    paged_decode_ops.chunked_prefill_paged_decode(
        query=query,
        key=None,
        value=None,
        output=output,
        kv_cache_dtype="fp8",
        key_cache=key_storage,
        value_cache=value_storage,
        block_table=block_tables,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=1374,
        max_query_len=1,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    assert forwarded == {
        "k_scale": k_scale,
        "v_scale": v_scale,
        "cache_dtype": current_platform.fp8_dtype(),
    }
    assert flydsl_routed == [True]
    torch.testing.assert_close(output, expected, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not on_gfx1x(), reason="BF16 SplitKV decode requires gfx1x")
@torch.inference_mode()
def test_chunked_decode_routes_padded_bf16_cache(monkeypatch) -> None:
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops

    set_random_seed(0)
    case = SplitKVCase(
        torch.bfloat16,
        torch.bfloat16,
        12,
        2,
        256,
        1568,
        (8192,),
        None,
        padded_stride=True,
    )
    (
        query,
        _,
        _,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    output = torch.empty_like(query)
    expected = _run_non_split(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        256**-0.5,
        k_scale,
        v_scale,
    )
    original = paged_decode_ops._paged_attention_2d_splitkv_decode
    routed = []
    flydsl_routed = []

    def route_spy(*args, **kwargs):
        routed.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        paged_decode_ops, "_paged_attention_2d_splitkv_decode", route_spy
    )
    monkeypatch.setattr(rdna4_ops.envs, "VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL", True)
    select_kernel_config, run_flydsl = rdna4_ops._load_flydsl_splitkv()

    def flydsl_spy(*args, **kwargs):
        flydsl_routed.append(True)
        return run_flydsl(*args, **kwargs)

    monkeypatch.setattr(
        rdna4_ops,
        "_load_flydsl_splitkv",
        lambda: (select_kernel_config, flydsl_spy),
    )
    paged_decode_ops.chunked_prefill_paged_decode(
        query=query,
        key=None,
        value=None,
        output=output,
        kv_cache_dtype="auto",
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_tables,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=8192,
        max_query_len=1,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    assert routed == [True]
    assert flydsl_routed == [True]
    torch.testing.assert_close(output, expected, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not on_gfx1x(), reason="Native ROCm decode requires gfx1x")
@torch.inference_mode()
def test_native_paged_attention_precedes_splitkv_for_standard_layout(
    monkeypatch,
) -> None:
    set_random_seed(0)
    case = SplitKVCase(
        torch.bfloat16,
        torch.bfloat16,
        16,
        4,
        128,
        32,
        (257,),
        None,
    )
    (
        query,
        _,
        _,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    output = torch.empty_like(query)
    native_called = []

    def native_spy(output, *args, **kwargs) -> None:
        native_called.append(True)
        output.fill_(3)

    def unexpected_splitkv(*args, **kwargs):
        raise AssertionError("SplitKV must not precede the native ROCm kernel")

    monkeypatch.setattr(
        "vllm.platforms.rocm.use_rocm_custom_paged_attention",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(paged_decode_ops.ops, "paged_attention_rocm", native_spy)
    monkeypatch.setattr(
        paged_decode_ops, "_paged_attention_2d_splitkv_decode", unexpected_splitkv
    )
    paged_decode_ops.chunked_prefill_paged_decode(
        query=query,
        key=None,
        value=None,
        output=output,
        kv_cache_dtype="auto",
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_tables,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=257,
        max_query_len=1,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    assert native_called == [True]
    torch.testing.assert_close(output, torch.full_like(output, 3))


@pytest.mark.skipif(not on_gfx12x(), reason="FP8 SplitKV decode requires gfx12x")
@torch.inference_mode()
def test_fp8_splitkv_locked_workspace_cudagraph_replays_dynamic_sequence() -> None:
    set_random_seed(0)
    case = SplitKVCase(
        torch.bfloat16,
        torch.float8_e4m3fn,
        12,
        2,
        256,
        1568,
        (8192,),
        16,
        0.73,
        1.27,
        padded_stride=True,
    )
    (
        query,
        _,
        _,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    output = torch.empty_like(query)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)

    reset_workspace_manager()
    init_workspace_manager(query.device)
    reserve_splitkv_workspace(
        max_batch_size=1,
        num_query_heads=12,
        num_kv_heads=2,
        head_size=256,
        physical_block_size=1568,
        max_seq_len=8192,
        allow_short_context=True,
    )
    lock_workspace()
    seq_lens.fill_(1)

    def run_splitkv() -> None:
        _paged_attention_2d_splitkv_decode(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            256**-0.5,
            k_scale,
            v_scale,
            output=output,
            actual_max_splits=16,
            max_seq_len=8192,
            query_start_loc=query_start_loc,
            filter_by_query_len=True,
        )

    graph = None
    try:
        run_splitkv()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_splitkv()

        query.copy_(torch.randn_like(query))
        seq_lens.fill_(8192)
        graph.replay()
        torch.accelerator.synchronize()
        expected = _run_non_split(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            256**-0.5,
            k_scale,
            v_scale,
        )
        torch.testing.assert_close(output, expected, atol=0.01, rtol=0.01)
    finally:
        del graph
        reset_workspace_manager()


def _broad_splitkv_cases():
    # Pairwise dispatch sweep: every GQA ratio and batch threshold, both heads.
    dtypes = (
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.float8_e4m3fn),
        (torch.float16, torch.float8_e4m3fn),
        (torch.bfloat16, torch.float8_e4m3fnuz),
        (torch.float16, torch.float8_e4m3fnuz),
    )
    index = 0
    for head in (128, 256):
        for gqa in range(1, 17):
            for batch in (1, 3, 7, 8, 9, 16, 33):
                dtype, kv_dtype = dtypes[index % len(dtypes)]
                page = (16, 32, 528, 784, 1568)[index % 5]
                lengths = tuple(
                    (1, page - 1, page, page + 1, 257)[i % 5] for i in range(batch)
                )
                kv_heads = (1, 2, 4)[index % 3]
                yield pytest.param(
                    SplitKVCase(
                        dtype,
                        kv_dtype,
                        gqa * kv_heads,
                        kv_heads,
                        head,
                        page,
                        lengths,
                        (2, 4, 8, 16)[index % 4],
                        0.73,
                        1.27,
                        padded_stride=True,
                    ),
                    index % 3,
                    id=f"dispatch-d{head}-g{gqa}-b{batch}",
                )
                index += 1
    # Cross each promoted specialization with all applicable dtype pairs.
    for head, gqa, batch in (
        (128, 16, 8),
        (128, 16, 16),
        (128, 16, 1),
        (128, 8, 3),
        (256, 4, 3),
        (256, 16, 3),
        (256, 6, 3),
        (128, 3, 3),
    ):
        for dtype, kv_dtype in dtypes:
            for profile in range(3):
                page = (16, 784, 32)[profile]
                lengths = tuple(
                    (31, 32, 33, 127, 128, 129, 513)[i % 7] for i in range(batch)
                )
                yield pytest.param(
                    SplitKVCase(
                        dtype,
                        kv_dtype,
                        gqa * 2,
                        2,
                        head,
                        page,
                        lengths,
                        (2, 8, 16)[profile],
                        0.73,
                        1.27,
                        padded_stride=True,
                    ),
                    profile,
                    id=f"route-d{head}-g{gqa}-b{batch}-{dtype}-{kv_dtype}-p{profile}",
                )

    for head, gqa in ((128, 16), (256, 6)):
        for batch in (64, 128, 256):
            for dtype, kv_dtype in dtypes:
                yield pytest.param(
                    SplitKVCase(
                        dtype,
                        kv_dtype,
                        gqa * 2,
                        2,
                        head,
                        32,
                        (33,) * batch,
                        4,
                        0.73,
                        1.27,
                        padded_stride=True,
                    ),
                    0,
                    id=f"high-batch-d{head}-g{gqa}-b{batch}-{dtype}-{kv_dtype}",
                )


def _qwen_tp_splitkv_cases():
    # Qwen's full attention has 24 Q / 4 KV heads of dimension 256.
    # Exercise the exact TP1/TP2/TP4 local shapes, including hybrid-cache pages.
    for tp in (1, 2, 4):
        for dtype, kv_dtype in (
            (torch.bfloat16, torch.bfloat16),
            (torch.float16, torch.float16),
            (torch.bfloat16, torch.float8_e4m3fn),
            (torch.bfloat16, torch.float8_e4m3fnuz),
        ):
            hybrid_page = 1568 if kv_dtype.itemsize == 1 else 784
            for layout, page, lengths in (
                ("standard", 32, (31, 32, 33)),
                (
                    "hybrid-boundary",
                    hybrid_page,
                    (hybrid_page - 1, hybrid_page, hybrid_page + 1),
                ),
                ("batch8", hybrid_page, tuple(4095 + i for i in range(8))),
                ("long", hybrid_page, (32767,)),
            ):
                for profile in range(3):
                    yield pytest.param(
                        SplitKVCase(
                            dtype,
                            kv_dtype,
                            24 // tp,
                            4 // tp,
                            256,
                            page,
                            lengths,
                            (2, 4, 16)[profile],
                            0.73,
                            1.27,
                            padded_stride=True,
                        ),
                        profile,
                        id=f"qwen-tp{tp}-{kv_dtype}-{layout}-p{profile}",
                    )


@pytest.mark.skipif(not on_gfx12x(), reason="FlyDSL SplitKV requires gfx12x")
@pytest.mark.parametrize(
    "case,profile", [*_broad_splitkv_cases(), *_qwen_tp_splitkv_cases()]
)
@pytest.mark.parametrize("use_flydsl", [False, True], ids=["triton", "flydsl"])
@torch.inference_mode()
def test_rdna4_splitkv_dispatch_boundaries_match_torch(
    monkeypatch, record_property, case: SplitKVCase, profile: int, use_flydsl: bool
) -> None:
    """Guard dispatch boundaries, masked NaN tails and nonuniform attention.

    Exercise the public SplitKV wrapper, including unsupported-shape fallback,
    against dense FP32 attention over the actual stored cache values. Profile 1
    stresses peaked softmax; profile 2 stresses nearly uniform attention.
    """
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops

    set_random_seed(100 + profile)
    q, dk, dv, k, v, tables, lens, ks, vs = _make_inputs(case)
    q.mul_((1.0, 8.0, 0.01)[profile])
    batch = len(case.seq_lens)
    # Noncontiguous query/output head and row strides are part of the contract.
    query_backing = torch.empty(
        (batch, case.num_query_heads * 2, case.head_size + 8),
        dtype=q.dtype,
        device=DEVICE,
    )
    query = query_backing[:, ::2, : case.head_size]
    query.copy_(q)
    output_backing = torch.full_like(query_backing, 123)
    output = output_backing[:, ::2, : case.head_size]
    starts = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE)
    mid = torch.empty(
        (batch, case.num_query_heads, case.splits, case.head_size),
        dtype=torch.float32,
        device=DEVICE,
    )
    lse = torch.empty(mid.shape[:-1], dtype=torch.float32, device=DEVICE)
    monkeypatch.setattr(
        rdna4_ops.envs, "VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL", use_flydsl
    )
    if not use_flydsl:

        def unexpected_flydsl_load():
            pytest.fail("Triton-only validation must not load FlyDSL kernels")

        monkeypatch.setattr(rdna4_ops, "_load_flydsl_splitkv", unexpected_flydsl_load)
    kwargs = dict(
        query=query,
        key_cache=k,
        value_cache=v,
        output=output,
        block_tables=tables,
        seq_lens=lens,
        query_start_loc=starts,
        k_scale=ks,
        v_scale=vs,
        scale=case.head_size**-0.5,
        actual_max_splits=case.splits,
        max_seq_len=max(case.seq_lens),
        filter_by_query_len=True,
    )
    config = rdna4_ops.get_rdna4_flydsl_splitkv_config(**kwargs) if use_flydsl else None
    used = rdna4_ops.try_rdna4_splitkv_paged_attention(
        **kwargs, mid_out=mid, mid_lse=lse
    )
    assert used == (config is not None)
    record_property("route", config.route.value if config else "triton_fallback")
    record_property("kv_dtype", str(case.kv_dtype))
    record_property("backend_requested", "flydsl" if use_flydsl else "triton")
    if not used:
        _paged_attention_2d_splitkv_decode(**kwargs, mid_out=mid, mid_lse=lse)
    reference = _torch_reference(
        query,
        dk,
        dv,
        tables,
        lens,
        case.head_size**-0.5,
        case.k_scale if case.kv_dtype.itemsize == 1 else 1.0,
        case.v_scale if case.kv_dtype.itemsize == 1 else 1.0,
    )
    assert torch.isfinite(output).all()
    record_property(
        "max_abs_error", (output.float() - reference.float()).abs().max().item()
    )
    torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)
    assert (output_backing[:, 1::2] == 123).all()
    assert (output_backing[:, ::2, case.head_size :] == 123).all()


@pytest.mark.skipif(not on_gfx12x(), reason="FlyDSL SplitKV requires gfx12x")
@pytest.mark.parametrize(
    "head,gqa,batch,kv_dtype",
    [
        (128, 16, 8, torch.bfloat16),
        (128, 16, 16, torch.bfloat16),
        (128, 16, 1, torch.bfloat16),
        (128, 8, 3, torch.float8_e4m3fn),
        (256, 4, 3, torch.bfloat16),
        (256, 16, 3, torch.float8_e4m3fn),
        (256, 6, 3, torch.float8_e4m3fn),
        (128, 3, 3, torch.float8_e4m3fn),
        (256, 3, 3, torch.float8_e4m3fnuz),
    ],
)
@torch.inference_mode()
def test_rdna4_flydsl_graph_replays_mixed_rows_and_lengths(
    monkeypatch, head: int, gqa: int, batch: int, kv_dtype: torch.dtype
) -> None:
    """Every specialization must tolerate changing decode/prefill membership.

    Reusing a captured graph also checks that Wave8 completion counters reset
    between launches, including launches containing no decode rows.
    """
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops

    set_random_seed(19)
    case = SplitKVCase(
        torch.bfloat16,
        kv_dtype,
        gqa * 2,
        2,
        head,
        32,
        (513,) * batch,
        16,
        0.73,
        1.27,
        padded_stride=True,
    )
    q, dk, dv, k, v, tables, lens, ks, vs = _make_inputs(case)
    query = torch.randn((batch * 3, gqa * 2, head), dtype=q.dtype, device=DEVICE)
    output = torch.full_like(query, 123)
    starts = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE)
    mid = torch.empty((batch, gqa * 2, 16, head), dtype=torch.float32, device=DEVICE)
    lse = torch.empty(mid.shape[:-1], dtype=torch.float32, device=DEVICE)
    monkeypatch.setattr(rdna4_ops.envs, "VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL", True)

    def run():
        assert rdna4_ops.try_rdna4_splitkv_paged_attention(
            query=query,
            key_cache=k,
            value_cache=v,
            output=output,
            block_tables=tables,
            seq_lens=lens,
            query_start_loc=starts,
            k_scale=ks,
            v_scale=vs,
            scale=head**-0.5,
            actual_max_splits=16,
            max_seq_len=513,
            filter_by_query_len=True,
            mid_out=mid,
            mid_lse=lse,
        )

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for step, length in enumerate((0, 1, 15, 16, 17, 31, 32, 33, 513, 0)):
        counts = [(i + step) % 3 for i in range(batch)]
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        starts.copy_(torch.tensor(offsets, dtype=torch.int32, device=DEVICE))
        lens.fill_(length)
        query.normal_()
        output.fill_(123)
        for _ in range(10):
            graph.replay()
        torch.accelerator.synchronize()
        ref = _torch_reference(
            query[starts[:-1].long()],
            dk,
            dv,
            tables,
            lens,
            head**-0.5,
            case.k_scale if kv_dtype.itemsize == 1 else 1.0,
            case.v_scale if kv_dtype.itemsize == 1 else 1.0,
        )
        untouched = torch.ones(query.shape[0], dtype=torch.bool, device=DEVICE)
        for seq, count in enumerate(counts):
            if count == 1:
                row = offsets[seq]
                torch.testing.assert_close(output[row], ref[seq], atol=0.01, rtol=0.01)
                untouched[row] = False
        assert (output[untouched] == 123).all()


@pytest.mark.skipif(not on_gfx12x(), reason="FlyDSL SplitKV requires gfx12x")
@pytest.mark.parametrize(
    "head,gqa,batch", [(128, 1, 1), (128, 16, 1), (256, 8, 1), (256, 6, 3)]
)
@pytest.mark.parametrize(
    "kv_dtype", [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.bfloat16]
)
@pytest.mark.parametrize("code_offset", [0, 128])
@torch.inference_mode()
def test_rdna4_flydsl_preserves_cache_value_range(
    monkeypatch, head, gqa, batch, kv_dtype, code_offset
) -> None:
    """Single-token attention must return every finite FP8 code or large BF16 V."""
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops

    case = SplitKVCase(
        torch.bfloat16,
        kv_dtype,
        gqa,
        1,
        head,
        32,
        (1,) * batch,
        4,
        0.73,
        1.27,
        padded_stride=True,
    )
    q, dk, dv, _, _, tables, lens, ks, vs = _make_inputs(case)
    dk.zero_()
    if kv_dtype.itemsize == 1:
        codes = torch.arange(256, dtype=torch.int32, device=DEVICE).to(torch.uint8)
        values = codes.view(kv_dtype).float()
        values.nan_to_num_(nan=0.0)
    else:
        values = torch.linspace(-1e5, 1e5, 256, device=DEVICE)
    dv[:, 0, 0, :] = values[(torch.arange(head, device=DEVICE) + code_offset) % 256].to(
        kv_dtype
    )
    k, v = _pack_cache(dk, dv, padded_stride=True)
    out = torch.empty_like(q)
    starts = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE)
    mid = torch.empty((batch, gqa, 4, head), dtype=torch.float32, device=DEVICE)
    lse = torch.empty(mid.shape[:-1], dtype=torch.float32, device=DEVICE)
    monkeypatch.setattr(rdna4_ops.envs, "VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL", True)
    assert rdna4_ops.try_rdna4_splitkv_paged_attention(
        query=q,
        key_cache=k,
        value_cache=v,
        output=out,
        block_tables=tables,
        seq_lens=lens,
        query_start_loc=starts,
        k_scale=ks,
        v_scale=vs,
        scale=head**-0.5,
        actual_max_splits=4,
        max_seq_len=1,
        filter_by_query_len=True,
        mid_out=mid,
        mid_lse=lse,
    )
    ref = _torch_reference(
        q,
        dk,
        dv,
        tables,
        lens,
        head**-0.5,
        case.k_scale if kv_dtype.itemsize == 1 else 1.0,
        case.v_scale if kv_dtype.itemsize == 1 else 1.0,
    )
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not on_gfx12x(), reason="FlyDSL SplitKV requires gfx12x")
@pytest.mark.parametrize("splits", [2, 4, 8, 16])
@pytest.mark.parametrize(
    "head,page,query_dtype,kv_dtype",
    [
        (128, 8, torch.bfloat16, torch.float8_e4m3fnuz),
        (256, 24, torch.float16, torch.float8_e4m3fn),
        (128, 24, torch.bfloat16, torch.bfloat16),
        (256, 8, torch.float16, torch.float16),
    ],
)
@torch.inference_mode()
def test_rdna4_wave8_concurrent_streams_do_not_share_counters(
    monkeypatch, splits, head, page, query_dtype, kv_dtype
) -> None:
    """Concurrent captured launches must use independent completion counters."""
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops

    set_random_seed(71)
    case = SplitKVCase(
        query_dtype,
        kv_dtype,
        6,
        2,
        head,
        page,
        (1, 33, 513),
        splits,
        0.73,
        1.27,
        padded_stride=True,
    )
    q, dk, dv, k, v, tables, lens, ks, vs = _make_inputs(case)
    starts = torch.arange(4, dtype=torch.int32, device=DEVICE)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = [torch.empty_like(q) for _ in streams]
    mids = [
        torch.empty((3, 6, splits, head), dtype=torch.float32, device=DEVICE)
        for _ in streams
    ]
    lses = [torch.empty(t.shape[:-1], dtype=torch.float32, device=DEVICE) for t in mids]
    graphs = []
    monkeypatch.setattr(rdna4_ops.envs, "VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL", True)

    def run(i):
        assert rdna4_ops.try_rdna4_splitkv_paged_attention(
            query=q,
            key_cache=k,
            value_cache=v,
            output=outputs[i],
            block_tables=tables,
            seq_lens=lens,
            query_start_loc=starts,
            k_scale=ks,
            v_scale=vs,
            scale=head**-0.5,
            actual_max_splits=splits,
            max_seq_len=513,
            filter_by_query_len=True,
            mid_out=mids[i],
            mid_lse=lses[i],
        )

    for i, stream in enumerate(streams):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(i)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run(i)
        graphs.append(graph)
    for _ in range(50):
        for stream, graph in zip(streams, graphs):
            with torch.cuda.stream(stream):
                graph.replay()
    torch.accelerator.synchronize()
    ref = _torch_reference(
        q,
        dk,
        dv,
        tables,
        lens,
        head**-0.5,
        case.k_scale if kv_dtype.itemsize == 1 else 1.0,
        case.v_scale if kv_dtype.itemsize == 1 else 1.0,
    )
    for out in outputs:
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, ref, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not on_gfx12x(), reason="FlyDSL SplitKV requires gfx12x")
@pytest.mark.parametrize(
    "gqa,batch,length,fp8,route",
    [
        (16, 16, 8191, False, "native_d128_gqa16_lds"),
        (16, 16, 8192, False, "d128_gqa16_tile32"),
        (3, 1, 2047, False, "wave8"),
        (3, 1, 2048, False, None),
        (2, 1, 2048, False, "wave8"),
        (2, 2, 2048, False, None),
        (4, 1, 2048, True, "wave8"),
        (4, 3, 2048, True, None),
        (8, 3, 8192, False, "generic_tile32"),
    ],
)
def test_rdna4_splitkv_keeps_faster_long_context_routes(gqa, batch, length, fp8, route):
    from vllm.v1.attention.ops.flydsl_kernels.rdna4_splitkv import select_kernel_config

    query = torch.empty((batch, gqa * 2, 128), dtype=torch.bfloat16, device="meta")
    cache = torch.empty(
        (1, 2, 8, 784, 16),
        dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
        device="meta",
    )
    lens = torch.empty(batch, dtype=torch.int32, device="meta")
    config = select_kernel_config(query, cache, lens, length)
    assert (config.route.value if config else None) == route


@pytest.mark.skipif(not on_gfx12x(), reason="FP8 SplitKV requires gfx12x")
@torch.inference_mode()
def test_fp8_unsplit_d256_padded_cache_compiles_and_matches_torch():
    """Eight warps avoid a gfx1201 register allocator abort for this layout."""
    set_random_seed(51)
    case = SplitKVCase(
        torch.bfloat16,
        torch.float8_e4m3fn,
        2,
        2,
        256,
        784,
        (8193,),
        1,
        0.73,
        1.27,
        padded_stride=True,
    )
    q, dk, dv, k, v, tables, lens, ks, vs = _make_inputs(case)
    q.mul_(8)
    out = _paged_attention_2d_splitkv_decode(
        q,
        k,
        v,
        tables,
        lens,
        256**-0.5,
        ks,
        vs,
        actual_max_splits=1,
        max_seq_len=8193,
    )
    ref = _torch_reference(q, dk, dv, tables, lens, 256**-0.5, 0.73, 1.27)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, atol=0.01, rtol=0.01)


@pytest.mark.parametrize(
    "namespace,missing_api",
    [
        ("llvm", "memory_fence"),
        ("llvm", "atomic_add"),
        ("llvm", "generic_store"),
        (None, "AtomicOrdering"),
        ("rocdl", "SyncScope"),
    ],
)
def test_rdna4_flydsl_missing_memory_api_falls_back(
    monkeypatch, namespace, missing_api
):
    """An older FlyDSL with scalar FP8 alone cannot safely launch Wave8."""
    fx = pytest.importorskip("flydsl.expr")
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops

    rdna4_ops._load_flydsl_splitkv.cache_clear()
    rdna4_ops.is_rdna4_flydsl_splitkv_available.cache_clear()
    try:
        with monkeypatch.context() as patch:
            module = fx if namespace is None else getattr(fx, namespace)
            patch.delattr(module, missing_api, raising=False)
            assert not rdna4_ops.is_rdna4_flydsl_splitkv_available()
    finally:
        rdna4_ops._load_flydsl_splitkv.cache_clear()
        rdna4_ops.is_rdna4_flydsl_splitkv_available.cache_clear()


def test_rdna4_flydsl_loads_without_legacy_rocdl_memory_apis(monkeypatch):
    fx = pytest.importorskip("flydsl.expr")
    from vllm.v1.attention.ops import rdna4_splitkv as rdna4_ops

    rdna4_ops._load_flydsl_splitkv.cache_clear()
    rdna4_ops.is_rdna4_flydsl_splitkv_available.cache_clear()
    try:
        with monkeypatch.context() as patch:
            for name in (
                "memory_fence",
                "atomic_fetch_add",
                "global_store",
                "MemoryOrder",
            ):
                patch.delattr(fx.rocdl, name, raising=False)
            assert rdna4_ops.is_rdna4_flydsl_splitkv_available()
    finally:
        rdna4_ops._load_flydsl_splitkv.cache_clear()
        rdna4_ops.is_rdna4_flydsl_splitkv_available.cache_clear()
