# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

import vllm.v1.attention.backends.triton_attn as triton_attn_backend
from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import (
    nvfp4_kv_cache_full_dim,
    nvfp4_kv_cache_split_views,
    set_random_seed,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionImpl,
    TritonAttentionMetadata,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.attention.ops.triton_unified_attention import (
    _get_nvfp4_launch_config,
    unified_attention,
)
from vllm.v1.kv_cache_interface import KVQuantMode

DEVICE_TYPE = current_platform.device_type

NUM_HEADS = [(4, 4), (8, 2), (5, 1)]
HEAD_SIZES = [128, 256]
NVFP4_BYTEWISE_HEAD_SIZES = [64, 128, 192, 256, 320, 384, 448, 512]
NVFP4_MIXED_KV_HEAD_SIZES = [64, 128, 192, 320, 512]
BLOCK_SIZES = [16]

DTYPES = [torch.bfloat16]
QDTYPES = [None, current_platform.fp8_dtype()]
FP8_DTYPE = current_platform.fp8_dtype()

# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]

# 0: use 2D kernel for decode
# 8: use 3D kernel for decode
SEQ_THRESHOLD_3D_VALUES = [0, 8]


def test_nvfp4_launch_config_large_full_decode_heads() -> None:
    assert _get_nvfp4_launch_config(
        16, 128, 128, is_3d=False, sliding_window_val=0
    ) == (
        16,
        8,
        3,
    )
    assert _get_nvfp4_launch_config(16, 128, 128, is_3d=True, sliding_window_val=0) == (
        16,
        8,
        2,
    )
    assert _get_nvfp4_launch_config(16, 256, 256, is_3d=True, sliding_window_val=0) == (
        16,
        8,
        1,
    )
    assert _get_nvfp4_launch_config(16, 320, 512, is_3d=True, sliding_window_val=0) == (
        16,
        8,
        1,
    )
    assert _get_nvfp4_launch_config(16, 384, 512, is_3d=True, sliding_window_val=0) == (
        16,
        8,
        1,
    )
    assert _get_nvfp4_launch_config(16, 448, 512, is_3d=True, sliding_window_val=0) == (
        16,
        8,
        1,
    )
    assert _get_nvfp4_launch_config(16, 512, 512, is_3d=True, sliding_window_val=0) == (
        16,
        8,
        1,
    )
    assert _get_nvfp4_launch_config(
        32, 512, 512, is_3d=True, sliding_window_val=1024
    ) == (
        32,
        8,
        1,
    )


def _make_triton_metadata(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    is_all_pure_prefill: bool,
) -> TritonAttentionMetadata:
    head_size_padded = next_power_of_2(head_size)
    return TritonAttentionMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=num_tokens,
        query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32),
        max_seq_len=num_tokens,
        seq_lens=torch.tensor([num_tokens], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.arange(num_tokens, dtype=torch.long),
        is_all_pure_prefill=is_all_pure_prefill,
        seq_threshold_3D=1,
        num_par_softmax_segments=16,
        softmax_segm_output=torch.empty(
            (1, num_heads, 16, head_size_padded), dtype=torch.float32
        ),
        softmax_segm_max=torch.empty((1, num_heads, 16), dtype=torch.float32),
        softmax_segm_expsum=torch.empty((1, num_heads, 16), dtype=torch.float32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="NVFP4 Triton path is CUDA")
@torch.inference_mode()
def test_triton_attn_nvfp4_pure_prefill_uses_raw_kv(monkeypatch) -> None:
    torch.set_default_device(DEVICE_TYPE)

    num_tokens = 4
    num_query_heads = 4
    num_kv_heads = 2
    head_size = 128
    dtype = torch.bfloat16
    metadata = _make_triton_metadata(
        num_tokens, num_query_heads, head_size, is_all_pure_prefill=True
    )
    query = torch.randn(num_tokens, num_query_heads, head_size, dtype=dtype)
    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    layer = SimpleNamespace(
        _q_scale=torch.tensor(1.0),
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
    )
    impl = TritonAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="nvfp4",
    )

    called = {}

    def fake_context_attention_fwd(**kwargs):
        called["q"] = kwargs["q"]
        called["k"] = kwargs["k"]
        called["v"] = kwargs["v"]
        kwargs["o"].zero_()

    def fail_unified_attention(**kwargs):
        raise AssertionError("pure prefill should not read the FP4 cache")

    monkeypatch.setattr(
        triton_attn_backend, "context_attention_fwd", fake_context_attention_fwd
    )
    monkeypatch.setattr(
        triton_attn_backend, "unified_attention", fail_unified_attention
    )

    result = impl.forward(
        layer,
        query,
        key,
        value,
        torch.empty(0, dtype=torch.uint8),
        metadata,
        output=output,
    )

    assert result is output
    assert called["q"].data_ptr() == query.data_ptr()
    assert called["k"].data_ptr() == key.data_ptr()
    assert called["v"].data_ptr() == value.data_ptr()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="NVFP4 Triton path is CUDA")
@torch.inference_mode()
def test_triton_attn_nvfp4_pure_prefill_softcap_uses_raw_kv(monkeypatch) -> None:
    torch.set_default_device(DEVICE_TYPE)

    num_tokens = 4
    num_query_heads = 4
    num_kv_heads = 2
    head_size = 128
    dtype = torch.bfloat16
    metadata = _make_triton_metadata(
        num_tokens, num_query_heads, head_size, is_all_pure_prefill=True
    )
    query = torch.randn(num_tokens, num_query_heads, head_size, dtype=dtype)
    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    layer = SimpleNamespace(
        _q_scale=torch.tensor(1.0),
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
    )
    impl = TritonAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="nvfp4",
        logits_soft_cap=50.0,
    )
    called = {}

    def fake_context_attention_fwd(**kwargs):
        called["softcap"] = kwargs["softcap"]
        called["q"] = kwargs["q"]
        called["k"] = kwargs["k"]
        called["v"] = kwargs["v"]
        kwargs["o"].zero_()

    def fail_unified_attention(**kwargs):
        raise AssertionError("softcap pure prefill should use context attention")

    monkeypatch.setattr(
        triton_attn_backend, "context_attention_fwd", fake_context_attention_fwd
    )
    monkeypatch.setattr(
        triton_attn_backend, "unified_attention", fail_unified_attention
    )

    result = impl.forward(
        layer,
        query,
        key,
        value,
        torch.empty(0, dtype=torch.uint8),
        metadata,
        output=output,
    )

    assert result is output
    assert called["softcap"] == 50.0
    assert called["q"].data_ptr() == query.data_ptr()
    assert called["k"].data_ptr() == key.data_ptr()
    assert called["v"].data_ptr() == value.data_ptr()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="NVFP4 Triton path is CUDA")
@torch.inference_mode()
def test_triton_attn_nvfp4_kv_sharing_uses_cache(monkeypatch) -> None:
    torch.set_default_device(DEVICE_TYPE)

    num_tokens = 4
    num_query_heads = 4
    num_kv_heads = 2
    head_size = 128
    block_size = 16
    dtype = torch.bfloat16
    metadata = _make_triton_metadata(
        num_tokens, num_query_heads, head_size, is_all_pure_prefill=True
    )
    query = torch.randn(num_tokens, num_query_heads, head_size, dtype=dtype)
    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    layer = SimpleNamespace(
        _q_scale=torch.tensor(1.0),
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
    )
    impl = TritonAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="nvfp4",
        kv_sharing_target_layer_name="target_layer",
    )
    kv_cache = torch.empty(
        1,
        2,
        block_size,
        num_kv_heads,
        nvfp4_kv_cache_full_dim(head_size),
        dtype=torch.uint8,
    )
    called = {}

    def fail_context_attention_fwd(**kwargs):
        raise AssertionError("KV-sharing layers must read the shared cache")

    def fake_unified_attention(**kwargs):
        called["kv_quant_mode"] = kwargs["kv_quant_mode"]
        called["raw_k"] = kwargs["raw_k"]
        called["raw_v"] = kwargs["raw_v"]
        kwargs["out"].zero_()

    monkeypatch.setattr(
        triton_attn_backend, "context_attention_fwd", fail_context_attention_fwd
    )
    monkeypatch.setattr(
        triton_attn_backend, "unified_attention", fake_unified_attention
    )

    result = impl.forward(layer, query, key, value, kv_cache, metadata, output=output)

    assert result is output
    assert called["kv_quant_mode"] == KVQuantMode.NVFP4
    assert called["raw_k"] is None
    assert called["raw_v"] is None


@pytest.mark.skipif(not current_platform.is_cuda(), reason="NVFP4 Triton path is CUDA")
@torch.inference_mode()
def test_triton_attn_nvfp4_mm_prefix_uses_raw_current_kv(monkeypatch) -> None:
    torch.set_default_device(DEVICE_TYPE)

    num_tokens = 4
    num_query_heads = 4
    num_kv_heads = 2
    head_size = 128
    block_size = 16
    dtype = torch.bfloat16
    metadata = _make_triton_metadata(
        num_tokens, num_query_heads, head_size, is_all_pure_prefill=True
    )
    metadata.mm_prefix_range_tensor = torch.tensor([[[1, 2]]], dtype=torch.int32)
    query = torch.randn(num_tokens, num_query_heads, head_size, dtype=dtype)
    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    layer = SimpleNamespace(
        _q_scale=torch.tensor(1.0),
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
    )
    impl = TritonAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=4,
        kv_cache_dtype="nvfp4",
    )
    kv_cache = torch.empty(
        1,
        2,
        block_size,
        num_kv_heads,
        nvfp4_kv_cache_full_dim(head_size),
        dtype=torch.uint8,
    )
    called = {}

    def fail_context_attention_fwd(**kwargs):
        raise AssertionError("MM-prefix attention must stay on the cache path")

    def fake_unified_attention(**kwargs):
        called["kv_quant_mode"] = kwargs["kv_quant_mode"]
        called["mm_prefix_range"] = kwargs["mm_prefix_range"]
        called["raw_k"] = kwargs["raw_k"]
        called["raw_v"] = kwargs["raw_v"]
        kwargs["out"].zero_()

    monkeypatch.setattr(
        triton_attn_backend, "context_attention_fwd", fail_context_attention_fwd
    )
    monkeypatch.setattr(
        triton_attn_backend, "unified_attention", fake_unified_attention
    )

    result = impl.forward(layer, query, key, value, kv_cache, metadata, output=output)

    assert result is output
    assert called["kv_quant_mode"] == KVQuantMode.NVFP4
    assert called["mm_prefix_range"] is metadata.mm_prefix_range_tensor
    assert called["raw_k"] is key
    assert called["raw_v"] is value


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
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def _make_sequential_block_tables(
    kv_lens: list[int],
    block_size: int,
) -> tuple[torch.Tensor, int]:
    max_num_blocks_per_seq = (max(kv_lens) + block_size - 1) // block_size
    block_tables = torch.zeros(
        (len(kv_lens), max_num_blocks_per_seq), dtype=torch.int32
    )
    next_block = 0
    for seq_idx, kv_len in enumerate(kv_lens):
        num_blocks = (kv_len + block_size - 1) // block_size
        block_tables[seq_idx, :num_blocks] = torch.arange(
            next_block, next_block + num_blocks, dtype=torch.int32
        )
        next_block += num_blocks
    return block_tables, next_block


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 64, 128, 256])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("q_dtype", QDTYPES)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    q_dtype: torch.dtype | None,
    seq_threshold_3D: int,
) -> None:
    torch.set_default_device(DEVICE_TYPE)

    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty_like(query)

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None
    k_descale = None
    v_descale = None
    kv_quant_mode = KVQuantMode.NONE
    if q_dtype is not None:
        # Use non-1 scales so FP8 Q/K/V descale handling is tested explicitly.
        q_scale = torch.tensor(0.75, dtype=torch.float32)
        k_scale = torch.tensor(0.5, dtype=torch.float32)
        v_scale = torch.tensor(0.25, dtype=torch.float32)
        q_descale = q_scale
        scale_shape = (num_seqs, num_kv_heads)
        k_descale = torch.full(scale_shape, k_scale.item(), dtype=torch.float32)
        v_descale = torch.full(scale_shape, v_scale.item(), dtype=torch.float32)
        maybe_quantized_query = (query / q_scale).to(q_dtype)
        maybe_quantized_key_cache = (key_cache / k_scale).to(q_dtype)
        maybe_quantized_value_cache = (value_cache / v_scale).to(q_dtype)
        kv_quant_mode = KVQuantMode.FP8_PER_TENSOR

    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    unified_attention(
        q=maybe_quantized_query,
        k=maybe_quantized_key_cache,
        v=maybe_quantized_value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        kv_quant_mode=kv_quant_mode,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    atol, rtol = 1.5e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    (
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn_bf16_query_fp8_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    num_blocks: int,
    seq_threshold_3D: int,
) -> None:
    """Test bf16 Q with FP8 per-tensor KV cache (dequant via _cast_kv_tile)."""
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (-1, -1)
    scale = head_size**-0.5

    dtype = torch.bfloat16
    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    k_scale = torch.tensor(0.5, dtype=torch.float32)
    v_scale = torch.tensor(0.25, dtype=torch.float32)
    fp8_key_cache = (key_cache / k_scale).to(FP8_DTYPE)
    fp8_value_cache = (value_cache / v_scale).to(FP8_DTYPE)

    scale_shape = (num_seqs, num_kv_heads)
    k_descale = torch.full(scale_shape, k_scale.item(), dtype=torch.float32)
    v_descale = torch.full(scale_shape, v_scale.item(), dtype=torch.float32)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty_like(query)

    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    unified_attention(
        q=query,
        k=fp8_key_cache,
        v=fp8_value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
    )

    atol, rtol = 1.5e-1, 1.5e-1
    (
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="NVFP4 Triton path is CUDA")
@pytest.mark.parametrize("head_size", NVFP4_BYTEWISE_HEAD_SIZES)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn_nvfp4_kv(
    seq_threshold_3D: int,
    head_size: int,
) -> None:
    """Test BF16 Q with packed NVFP4 KV cache and inline Triton dequant."""
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    seq_lens = [(1, 1328), (1, 37), (1, 2011)]
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_seqs = len(seq_lens)
    num_query_heads = 8
    num_kv_heads = 2
    block_size = 16
    num_blocks = 256
    dtype = torch.bfloat16
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache_ref = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache_ref = torch.randn_like(key_cache_ref)

    full_dim = nvfp4_kv_cache_full_dim(head_size)
    key_cache = torch.empty(
        num_blocks, block_size, num_kv_heads, full_dim, dtype=torch.uint8
    )
    value_cache = torch.empty_like(key_cache)

    slot_mapping = torch.arange(num_blocks * block_size, dtype=torch.long)
    k_scale = (key_cache_ref.abs().amax() / 448.0).to(torch.float32)
    v_scale = (value_cache_ref.abs().amax() / 448.0).to(torch.float32)
    triton_reshape_and_cache_flash(
        key_cache_ref.reshape(-1, num_kv_heads, head_size),
        value_cache_ref.reshape(-1, num_kv_heads, head_size),
        key_cache,
        value_cache,
        slot_mapping,
        "nvfp4",
        k_scale,
        v_scale,
    )

    (key_data_cache,), (key_scale_cache,) = nvfp4_kv_cache_split_views(key_cache)
    (value_data_cache,), (value_scale_cache,) = nvfp4_kv_cache_split_views(value_cache)

    from tests.kernels.quantization.nvfp4_utils import dequant_nvfp4_kv_cache

    key_cache_dequant = (
        dequant_nvfp4_kv_cache(
            key_data_cache.permute(0, 2, 1, 3),
            key_scale_cache.permute(0, 2, 1, 3),
            k_scale.item(),
            head_size,
            block_size,
            triton_scale_layout=True,
        )
        .permute(0, 2, 1, 3)
        .to(dtype)
        .contiguous()
    )
    value_cache_dequant = (
        dequant_nvfp4_kv_cache(
            value_data_cache.permute(0, 2, 1, 3),
            value_scale_cache.permute(0, 2, 1, 3),
            v_scale.item(),
            head_size,
            block_size,
            triton_scale_layout=True,
        )
        .permute(0, 2, 1, 3)
        .to(dtype)
        .contiguous()
    )

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    max_kv_len = max(kv_lens)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty_like(query)
    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    unified_attention(
        q=query,
        k=key_data_cache,
        v=value_data_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=k_scale,
        v_descale=v_scale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        kv_quant_mode=KVQuantMode.NVFP4,
        k_scale_cache=key_scale_cache.view(torch.uint8),
        v_scale_cache=value_scale_cache.view(torch.uint8),
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache_dequant,
        value_cache=value_cache_dequant,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
    )
    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="NVFP4 Triton path is CUDA")
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(8, 24), (20, 52)],  # chunked prefill / extend, includes boundary tiles
        [(1, 65), (1, 97), (1, 33)],  # decode, covers 2D and 3D paths below
    ],
)
@pytest.mark.parametrize("head_size", NVFP4_MIXED_KV_HEAD_SIZES)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn_nvfp4_mixed_raw_current_kv(
    seq_lens: list[tuple[int, int]],
    head_size: int,
    seq_threshold_3D: int,
) -> None:
    """Use raw current K/V while cached prefix remains packed NVFP4."""
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = 8
    num_kv_heads = 2
    block_size = 16
    dtype = torch.bfloat16
    scale = head_size**-0.5

    block_tables, used_blocks = _make_sequential_block_tables(kv_lens, block_size)
    num_blocks = used_blocks + 4
    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    raw_key = torch.randn(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    raw_value = torch.randn_like(raw_key)
    key_cache_ref = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache_ref = torch.randn_like(key_cache_ref)

    query_start = 0
    for seq_idx, (query_len, kv_len) in enumerate(seq_lens):
        context_len = kv_len - query_len
        for local_query_idx in range(query_len):
            kv_pos = context_len + local_query_idx
            block_idx = block_tables[seq_idx, kv_pos // block_size].item()
            slot_idx = kv_pos % block_size
            raw_idx = query_start + local_query_idx
            key_cache_ref[block_idx, slot_idx] = raw_key[raw_idx]
            value_cache_ref[block_idx, slot_idx] = raw_value[raw_idx]
        query_start += query_len

    full_dim = nvfp4_kv_cache_full_dim(head_size)
    key_cache = torch.empty(
        num_blocks, block_size, num_kv_heads, full_dim, dtype=torch.uint8
    )
    value_cache = torch.empty_like(key_cache)

    slot_mapping = torch.arange(num_blocks * block_size, dtype=torch.long)
    k_scale = (key_cache_ref.abs().amax() / 448.0).to(torch.float32)
    v_scale = (value_cache_ref.abs().amax() / 448.0).to(torch.float32)
    triton_reshape_and_cache_flash(
        key_cache_ref.reshape(-1, num_kv_heads, head_size),
        value_cache_ref.reshape(-1, num_kv_heads, head_size),
        key_cache,
        value_cache,
        slot_mapping,
        "nvfp4",
        k_scale,
        v_scale,
    )

    (key_data_cache,), (key_scale_cache,) = nvfp4_kv_cache_split_views(key_cache)
    (value_data_cache,), (value_scale_cache,) = nvfp4_kv_cache_split_views(value_cache)

    from tests.kernels.quantization.nvfp4_utils import dequant_nvfp4_kv_cache

    key_cache_mixed_ref = (
        dequant_nvfp4_kv_cache(
            key_data_cache.permute(0, 2, 1, 3),
            key_scale_cache.permute(0, 2, 1, 3),
            k_scale.item(),
            head_size,
            block_size,
            triton_scale_layout=True,
        )
        .permute(0, 2, 1, 3)
        .to(dtype)
        .contiguous()
    )
    value_cache_mixed_ref = (
        dequant_nvfp4_kv_cache(
            value_data_cache.permute(0, 2, 1, 3),
            value_scale_cache.permute(0, 2, 1, 3),
            v_scale.item(),
            head_size,
            block_size,
            triton_scale_layout=True,
        )
        .permute(0, 2, 1, 3)
        .to(dtype)
        .contiguous()
    )

    query_start = 0
    for seq_idx, (query_len, kv_len) in enumerate(seq_lens):
        context_len = kv_len - query_len
        for local_query_idx in range(query_len):
            kv_pos = context_len + local_query_idx
            block_idx = block_tables[seq_idx, kv_pos // block_size].item()
            slot_idx = kv_pos % block_size
            raw_idx = query_start + local_query_idx
            key_cache_mixed_ref[block_idx, slot_idx] = raw_key[raw_idx]
            value_cache_mixed_ref[block_idx, slot_idx] = raw_value[raw_idx]
        query_start += query_len

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    output = torch.empty_like(query)
    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    unified_attention(
        q=query,
        k=key_data_cache,
        v=value_data_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(kv_lens),
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=k_scale,
        v_descale=v_scale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        kv_quant_mode=KVQuantMode.NVFP4,
        k_scale_cache=key_scale_cache.view(torch.uint8),
        v_scale_cache=value_scale_cache.view(torch.uint8),
        raw_k=raw_key,
        raw_v=raw_value,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache_mixed_ref,
        value_cache=value_cache_mixed_ref,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
    )
    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="NVFP4 Triton path is CUDA")
@torch.inference_mode()
def test_triton_unified_attn_nvfp4_mm_prefix_raw_current_clamps_seq_len() -> None:
    """MM prefix must not extend raw-current tile loads past seq_len."""
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    seq_len = 276
    num_query_heads = 16
    num_kv_heads = 8
    head_size = 256
    block_size = 16
    sliding_window = 1024
    dtype = torch.bfloat16
    scale = head_size**-0.5

    block_tables, used_blocks = _make_sequential_block_tables([seq_len], block_size)
    num_blocks = used_blocks
    query = torch.randn(seq_len, num_query_heads, head_size, dtype=dtype)

    # Keep deterministic NaNs immediately after the logical raw K/V tensors.
    # A stale tile bound that includes q-block padding will load these rows.
    raw_key_storage = torch.randn(seq_len + 4, num_kv_heads, head_size, dtype=dtype)
    raw_value_storage = torch.randn_like(raw_key_storage)
    raw_key_storage[seq_len:].fill_(float("nan"))
    raw_value_storage[seq_len:].fill_(float("nan"))
    raw_key = raw_key_storage[:seq_len]
    raw_value = raw_value_storage[:seq_len]

    key_cache_ref = torch.zeros(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache_ref = torch.zeros_like(key_cache_ref)
    for token_idx in range(seq_len):
        block_idx = block_tables[0, token_idx // block_size].item()
        slot_idx = token_idx % block_size
        key_cache_ref[block_idx, slot_idx] = raw_key[token_idx]
        value_cache_ref[block_idx, slot_idx] = raw_value[token_idx]

    full_dim = nvfp4_kv_cache_full_dim(head_size)
    key_cache = torch.empty(
        num_blocks, block_size, num_kv_heads, full_dim, dtype=torch.uint8
    )
    value_cache = torch.empty_like(key_cache)
    slot_mapping = torch.arange(num_blocks * block_size, dtype=torch.long)
    k_scale = (key_cache_ref.abs().amax() / 448.0).to(torch.float32)
    v_scale = (value_cache_ref.abs().amax() / 448.0).to(torch.float32)
    triton_reshape_and_cache_flash(
        key_cache_ref.reshape(-1, num_kv_heads, head_size),
        value_cache_ref.reshape(-1, num_kv_heads, head_size),
        key_cache,
        value_cache,
        slot_mapping,
        "nvfp4",
        k_scale,
        v_scale,
    )
    (key_data_cache,), (key_scale_cache,) = nvfp4_kv_cache_split_views(key_cache)
    (value_data_cache,), (value_scale_cache,) = nvfp4_kv_cache_split_views(value_cache)

    output = torch.empty_like(query)
    cu_query_lens = torch.tensor([0, seq_len], dtype=torch.int32)
    kv_lens_t = torch.tensor([seq_len], dtype=torch.int32)
    mm_prefix_range = torch.tensor([[[5, 260]]], dtype=torch.int32)
    num_par_softmax_segments = 16
    unified_attention(
        q=query,
        k=key_data_cache,
        v=value_data_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        max_seqlen_q=seq_len,
        seqused_k=kv_lens_t,
        max_seqlen_k=seq_len,
        softmax_scale=scale,
        causal=True,
        window_size=(sliding_window - 1, 0),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=k_scale,
        v_descale=v_scale,
        seq_threshold_3D=0,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=torch.empty(
            (1, num_query_heads, num_par_softmax_segments, head_size),
            dtype=torch.float32,
        ),
        softmax_segm_max=torch.empty(
            (1, num_query_heads, num_par_softmax_segments), dtype=torch.float32
        ),
        softmax_segm_expsum=torch.empty(
            (1, num_query_heads, num_par_softmax_segments), dtype=torch.float32
        ),
        kv_quant_mode=KVQuantMode.NVFP4,
        k_scale_cache=key_scale_cache.view(torch.uint8),
        v_scale_cache=value_scale_cache.view(torch.uint8),
        raw_k=raw_key,
        raw_v=raw_value,
        mm_prefix_range=mm_prefix_range,
    )

    key_ref = torch.repeat_interleave(
        raw_key.float(), num_query_heads // num_kv_heads, dim=1
    )
    value_ref = torch.repeat_interleave(
        raw_value.float(), num_query_heads // num_kv_heads, dim=1
    )
    scores = torch.einsum("qhd,khd->hqk", query.float() * scale, key_ref)
    query_pos = torch.arange(seq_len)[:, None]
    key_pos = torch.arange(seq_len)[None, :]
    allowed = (key_pos <= query_pos) & ((query_pos - key_pos) < sliding_window)
    mm_range = (query_pos >= 5) & (query_pos <= 260) & (key_pos >= 5) & (key_pos <= 260)
    allowed |= mm_range
    scores.masked_fill_(~allowed[None, :, :], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    ref_output = torch.einsum("hqk,khd->qhd", attn, value_ref).to(dtype)

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 1328), (5, 18), (129, 463)],
        [(1, 523), (1, 37), (1, 2011)],
        [(1, 1)] * 533,
        [(533, 533)] * 533,
    ],
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 64, 128, 256])
@pytest.mark.parametrize("soft_cap", [None, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn_fp16_input_fp8_output(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    seq_threshold_3D: int,
) -> None:
    """Test with fp16 input and fp8 output using output_scale."""
    torch.set_default_device(DEVICE_TYPE)

    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    dtype = torch.float16
    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty(sum(query_lens), num_query_heads, head_size, dtype=FP8_DTYPE)

    output_scale = torch.tensor(0.5, dtype=torch.float32)

    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        output_scale=output_scale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )

    output_fp16 = output.to(torch.float32) * output_scale.item()
    output_fp16 = output_fp16.to(torch.float16)

    atol, rtol = 2e-1, 2e-1
    (
        torch.testing.assert_close(output_fp16, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output_fp16 - ref_output))}",
    )


# USE_TD path covers two head-size regimes:
# - pow2 (HEAD_SIZE == HEAD_SIZE_PADDED): full TD path including Q/O.
# - non-pow2 (96, HEAD_SIZE_PADDED=128): gates USE_TD_QO off — Q load
#   and output store fall back to pointer path, KV tile TD load remains.
# The non-pow2 case mirrors real models like Phi-3-mini (head_size=96).
HEAD_SIZES_USE_TD = [128, 256, 96]


def _run_use_td_case(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    sliding_window: int | None,
    soft_cap: float | None,
    seq_threshold_3D: int,
    dtype: torch.dtype = torch.bfloat16,
    num_blocks: int = 2048,
) -> None:
    """Shared driver for the USE_TD test cases.

    Runs ``unified_attention(..., use_td=True)`` and compares against the
    reference paged-attention implementation that the sibling non-TD
    tests use.
    """
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty_like(query)

    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        use_td=True,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    torch.testing.assert_close(output, ref_output, atol=1.5e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES_USE_TD)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 128])
@pytest.mark.parametrize("soft_cap", [None, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn_use_td(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    seq_threshold_3D: int,
) -> None:
    """Exercise the USE_TD (tensor-descriptor) Q/K/V load/store path.

    Covers both 2D and 3D kernels via ``seq_threshold_3D``. Two routes
    to the USE_TD_QO=False fallback (pointer path for Q/O with TD still
    active for KV tile loads):

    - non-pow2 ``num_queries_per_kv`` via ``NUM_HEADS`` entry ``(5, 1)``,
    - non-pow2 ``head_size`` via ``HEAD_SIZES_USE_TD`` entry ``96``.
    """
    _run_use_td_case(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        seq_threshold_3D=seq_threshold_3D,
        num_blocks=num_blocks,
    )


# Prefill-heavy shape: long query drives the prefill kernel path where
# ``_get_tile_size`` returns 32, which exceeds block_size=16 and must be
# clamped by the fix in 'clamp TILE_SIZE to block_size when USE_TD'.
# Only the prefill launch exercises the clamp, so parameterize only over
# the (num_heads, seq_threshold_3D=0) combinations needed to cover it.
@pytest.mark.parametrize("num_heads", [(4, 4), (5, 1)])
@torch.inference_mode()
def test_triton_unified_attn_use_td_tile_clamp(
    num_heads: tuple[int, int],
) -> None:
    """Regression guard: ``USE_TD`` needs ``BLOCK_SIZE % TILE_SIZE == 0``.

    With ``block_size=16`` and ``head_size=128`` (non-Gemma3),
    ``_get_tile_size`` returns 32 for prefill, which violates the
    ``USE_TD`` constraint unless clamped to ``block_size``.  Without
    the clamp the triton kernel ``static_assert`` fires at compile time.
    """
    _run_use_td_case(
        seq_lens=[(256, 256), (128, 128)],
        num_heads=num_heads,
        head_size=128,
        block_size=16,
        sliding_window=None,
        soft_cap=None,
        seq_threshold_3D=0,
    )
