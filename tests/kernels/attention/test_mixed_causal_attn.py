# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for per-request causal/non-causal attention (mixed batches).

Validates that both triton and flash-attention backends correctly handle
batches where some sequences use causal masking and others use non-causal
(bidirectional) masking — needed by DiffusionGemma.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import (
    nvfp4_kv_cache_full_dim,
    nvfp4_kv_cache_split_views,
    set_random_seed,
)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionImpl,
    TritonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import KVQuantMode

# Mixed causal/non-causal attention is only validated on a subset of GPUs:
# the Triton path on Hopper (SM90) and B200 (SM100); the FA4 path on Hopper
# (SM90) only.
_device_capability = current_platform.get_device_capability()
_major = _device_capability.major if _device_capability is not None else None

NUM_HEADS = [(4, 4), (8, 2)]
HEAD_SIZES = [128]
BLOCK_SIZES = [16]
DTYPES = [torch.bfloat16]


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    per_seq_causal: list[bool],
    sliding_window: int | None = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q = q * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_np[i, :num_kv_blocks]
        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()

        if per_seq_causal[i]:
            mask = torch.triu(
                torch.ones(query_len, kv_len, device=attn.device),
                diagonal=kv_len - query_len + 1,
            ).bool()
        else:
            mask = torch.zeros(query_len, kv_len, device=attn.device).bool()

        if sliding_window is not None:
            sw_mask = (
                torch.triu(
                    torch.ones(query_len, kv_len, device=attn.device),
                    diagonal=kv_len - (query_len + sliding_window) + 1,
                )
                .bool()
                .logical_not()
            )
            mask |= sw_mask

        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def _make_block_tables(
    kv_lens: list[int],
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, int]:
    max_num_blocks = (max(kv_lens) + block_size - 1) // block_size
    block_tables = torch.zeros(
        (len(kv_lens), max_num_blocks), dtype=torch.int32, device=device
    )
    next_block = 0
    for seq_idx, kv_len in enumerate(kv_lens):
        num_blocks = (kv_len + block_size - 1) // block_size
        block_tables[seq_idx, :num_blocks] = torch.arange(
            next_block,
            next_block + num_blocks,
            dtype=torch.int32,
            device=device,
        )
        next_block += num_blocks
    return block_tables, next_block


def _make_current_slot_mapping(
    seq_lens: list[tuple[int, int]],
    block_tables: torch.Tensor,
    block_size: int,
    device: str,
) -> torch.Tensor:
    slots: list[int] = []
    for seq_idx, (query_len, kv_len) in enumerate(seq_lens):
        context_len = kv_len - query_len
        for local_query_idx in range(query_len):
            kv_pos = context_len + local_query_idx
            block_idx = block_tables[seq_idx, kv_pos // block_size].item()
            slots.append(block_idx * block_size + kv_pos % block_size)
    return torch.tensor(slots, dtype=torch.long, device=device)


def _make_triton_metadata(
    seq_lens: list[tuple[int, int]],
    block_tables: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_query_heads: int,
    head_size: int,
    causal: bool | torch.Tensor,
    seq_threshold_3d: int,
) -> TritonAttentionMetadata:
    query_lens = [query_len for query_len, _ in seq_lens]
    kv_lens = [kv_len for _, kv_len in seq_lens]
    head_size_padded = next_power_of_2(head_size)
    return TritonAttentionMetadata(
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        query_start_loc=torch.tensor(
            [0] + query_lens, dtype=torch.int32, device=block_tables.device
        ).cumsum(dim=0, dtype=torch.int32),
        max_seq_len=max(kv_lens),
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device=block_tables.device),
        block_table=block_tables,
        slot_mapping=slot_mapping,
        is_all_pure_prefill=False,
        is_decode_only=all(query_len == 1 for query_len in query_lens),
        seq_threshold_3D=seq_threshold_3d,
        num_par_softmax_segments=16,
        softmax_segm_output=torch.empty(
            (len(seq_lens), num_query_heads, 16, head_size_padded),
            dtype=torch.float32,
            device=block_tables.device,
        ),
        softmax_segm_max=torch.empty(
            (len(seq_lens), num_query_heads, 16),
            dtype=torch.float32,
            device=block_tables.device,
        ),
        softmax_segm_expsum=torch.empty(
            (len(seq_lens), num_query_heads, 16),
            dtype=torch.float32,
            device=block_tables.device,
        ),
        causal=causal,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )


def _causal_value(
    causal_kind: str,
    num_seqs: int,
    device: str,
) -> tuple[bool | torch.Tensor, list[bool]]:
    if causal_kind == "true":
        return True, [True] * num_seqs
    if causal_kind == "false":
        return False, [False] * num_seqs
    assert causal_kind == "mixed"
    causal_list = [(idx % 2) == 0 for idx in range(num_seqs)]
    return torch.tensor(causal_list, dtype=torch.bool, device=device), causal_list


# ---- Triton backend test ----


@pytest.mark.skipif(
    _major not in (9, 10),
    reason="Triton mixed causal attention requires Hopper (SM90) or B200 (SM100).",
)
@pytest.mark.parametrize(
    "seq_lens",
    [[(1, 128), (5, 64), (1, 256)]],
)
@pytest.mark.parametrize(
    "per_seq_causal",
    [[True, False, True], [False, True, False], [True, True, False]],
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_triton_mixed_causal(
    seq_lens: list[tuple[int, int]],
    per_seq_causal: list[bool],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
):
    if not current_platform.is_cuda():
        pytest.skip("Triton attention requires CUDA")

    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    set_random_seed(42)
    device = "cuda"

    num_query_heads, num_kv_heads = num_heads
    assert len(seq_lens) == len(per_seq_causal)

    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    num_seqs = len(seq_lens)

    num_query_tokens = sum(query_lens)
    max_kv_len = max(kv_lens)
    max_num_blocks = (max_kv_len + block_size - 1) // block_size
    num_blocks = max_num_blocks * num_seqs + 10

    scale = head_size**-0.5
    query = torch.randn(
        num_query_tokens, num_query_heads, head_size, dtype=dtype, device=device
    )
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    value_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )

    block_tables_list = []
    for i in range(num_seqs):
        n_blocks = (kv_lens[i] + block_size - 1) // block_size
        blocks = list(range(i * max_num_blocks, i * max_num_blocks + n_blocks))
        blocks += [0] * (max_num_blocks - n_blocks)
        block_tables_list.append(blocks)
    block_tables = torch.tensor(block_tables_list, dtype=torch.int32, device=device)

    cu_seqlens_q = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    for i, ql in enumerate(query_lens):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + ql

    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    max_seqlen_q = max(query_lens)
    max_seqlen_k = max(kv_lens)

    causal_tensor = torch.tensor(per_seq_causal, dtype=torch.bool, device=device)

    output = torch.empty_like(query)
    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        causal=causal_tensor,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0.0,
        q_descale=None,
        k_descale=1.0,
        v_descale=1.0,
    )

    ref_output = ref_paged_attn(
        query,
        key_cache,
        value_cache,
        query_lens,
        kv_lens,
        block_tables,
        scale,
        per_seq_causal,
    )

    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="NVFP4 Triton attention requires CUDA.",
)
@pytest.mark.parametrize(
    ("seq_lens", "use_raw_current_kv", "per_seq_causal"),
    [
        ([(1, 128), (1, 64)], False, [True, False]),
        ([(1, 128), (1, 64)], False, [False, True]),
        ([(8, 32), (4, 64)], True, [True, False]),
        ([(8, 32), (4, 64)], True, [False, True]),
        ([(32, 126)], True, [False]),
        ([(256, 350)], True, [False]),
        ([(256, 350)], True, [True]),
        ([(256, 350)], True, False),
        ([(256, 256)], False, [False]),
        ([(256, 256)], True, [False]),
    ],
)
@pytest.mark.parametrize("head_size", [128, 256, 512])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
@torch.inference_mode()
def test_triton_nvfp4_mixed_causal(
    seq_lens: list[tuple[int, int]],
    use_raw_current_kv: bool,
    per_seq_causal: list[bool] | bool,
    head_size: int,
    num_kv_heads: int,
) -> None:
    """Validate per-request causal masks with packed NVFP4 KV cache."""
    from tests.kernels.quantization.nvfp4_utils import dequant_nvfp4_kv_cache
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
    )
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    set_random_seed(43)
    device = "cuda"
    dtype = torch.bfloat16
    num_query_heads = 8
    block_size = 16
    scale = head_size**-0.5

    if isinstance(per_seq_causal, bool):
        causal_arg = per_seq_causal
        ref_per_seq_causal = [per_seq_causal] * len(seq_lens)
    else:
        assert len(seq_lens) == len(per_seq_causal)
        causal_arg = torch.tensor(per_seq_causal, dtype=torch.bool, device=device)
        ref_per_seq_causal = per_seq_causal

    query_lens = [query_len for query_len, _ in seq_lens]
    kv_lens = [kv_len for _, kv_len in seq_lens]
    num_query_tokens = sum(query_lens)

    block_tables, used_blocks = _make_block_tables(kv_lens, block_size, device)
    num_blocks = used_blocks + 4
    query = torch.randn(
        num_query_tokens, num_query_heads, head_size, dtype=dtype, device=device
    )
    key_cache_ref = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    value_cache_ref = torch.randn_like(key_cache_ref)
    raw_key = torch.randn(
        num_query_tokens, num_kv_heads, head_size, dtype=dtype, device=device
    )
    raw_value = torch.randn_like(raw_key)

    if use_raw_current_kv:
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
        num_blocks, block_size, num_kv_heads, full_dim, dtype=torch.uint8, device=device
    )
    value_cache = torch.empty_like(key_cache)
    slot_mapping = torch.arange(
        num_blocks * block_size, dtype=torch.long, device=device
    )
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
    if use_raw_current_kv:
        query_start = 0
        for seq_idx, (query_len, kv_len) in enumerate(seq_lens):
            context_len = kv_len - query_len
            for local_query_idx in range(query_len):
                kv_pos = context_len + local_query_idx
                block_idx = block_tables[seq_idx, kv_pos // block_size].item()
                slot_idx = kv_pos % block_size
                raw_idx = query_start + local_query_idx
                key_cache_dequant[block_idx, slot_idx] = raw_key[raw_idx]
                value_cache_dequant[block_idx, slot_idx] = raw_value[raw_idx]
            query_start += query_len

    cu_seqlens_q = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device=device
    ).cumsum(dim=0, dtype=torch.int32)
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    output = torch.empty_like(query)

    unified_attention(
        q=query,
        k=key_data_cache,
        v=value_data_cache,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(query_lens),
        seqused_k=seqused_k,
        max_seqlen_k=max(kv_lens),
        softmax_scale=scale,
        causal=causal_arg,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0.0,
        q_descale=None,
        k_descale=k_scale,
        v_descale=v_scale,
        kv_quant_mode=KVQuantMode.NVFP4,
        k_scale_cache=key_scale_cache.view(torch.uint8),
        v_scale_cache=value_scale_cache.view(torch.uint8),
        raw_k=raw_key if use_raw_current_kv else None,
        raw_v=raw_value if use_raw_current_kv else None,
    )

    ref_output = ref_paged_attn(
        query,
        key_cache_dequant,
        value_cache_dequant,
        query_lens,
        kv_lens,
        block_tables,
        scale,
        ref_per_seq_causal,
    )
    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="NVFP4 Triton attention requires CUDA.",
)
@pytest.mark.parametrize(
    ("seq_lens", "causal_kind", "use_raw_current_kv", "uses_shared_kv_cache"),
    [
        ([(1, 128)], "true", False, False),
        ([(1, 128)], "false", False, False),
        ([(1, 128), (1, 64)], "mixed", False, False),
        ([(1, 128), (1, 64)], "mixed", False, True),
        ([(8, 32), (4, 64)], "mixed", True, False),
        ([(8, 32), (4, 64)], "mixed", False, True),
        ([(32, 126)], "false", True, False),
        ([(256, 350)], "false", True, False),
        ([(256, 256)], "false", True, False),
    ],
)
@pytest.mark.parametrize("head_size", [128, 256, 512])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
@torch.inference_mode()
def test_triton_impl_nvfp4_cache_update_forward_mixed_causal(
    seq_lens: list[tuple[int, int]],
    causal_kind: str,
    use_raw_current_kv: bool,
    uses_shared_kv_cache: bool,
    head_size: int,
    num_kv_heads: int,
) -> None:
    """Validate NVFP4 cache update + TritonAttentionImpl.forward together."""
    from tests.kernels.quantization.nvfp4_utils import dequant_nvfp4_kv_cache
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
    )

    set_random_seed(44)
    device = "cuda"
    dtype = torch.bfloat16
    num_query_heads = 16
    block_size = 16
    scale = head_size**-0.5

    assert not (uses_shared_kv_cache and use_raw_current_kv)
    query_lens = [query_len for query_len, _ in seq_lens]
    kv_lens = [kv_len for _, kv_len in seq_lens]
    num_query_tokens = sum(query_lens)
    seq_threshold_3d = 8 if all(query_len == 1 for query_len in query_lens) else 0

    block_tables, used_blocks = _make_block_tables(kv_lens, block_size, device)
    slot_mapping = _make_current_slot_mapping(
        seq_lens, block_tables, block_size, device
    )
    num_blocks = used_blocks + 4
    full_dim = nvfp4_kv_cache_full_dim(head_size)

    query = torch.randn(
        num_query_tokens,
        num_query_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    key_cache_ref = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    value_cache_ref = torch.randn_like(key_cache_ref)
    raw_key = torch.randn(
        num_query_tokens, num_kv_heads, head_size, dtype=dtype, device=device
    )
    raw_value = torch.randn_like(raw_key)

    k_scale = (
        torch.cat((key_cache_ref.reshape(-1), raw_key.reshape(-1))).abs().amax() / 448.0
    ).to(torch.float32)
    v_scale = (
        torch.cat((value_cache_ref.reshape(-1), raw_value.reshape(-1))).abs().amax()
        / 448.0
    ).to(torch.float32)
    layer = SimpleNamespace(
        _q_scale=torch.tensor(1.0, dtype=torch.float32, device=device),
        _k_scale=k_scale,
        _v_scale=v_scale,
        kv_sharing_target_layer_name="source_layer" if uses_shared_kv_cache else None,
    )
    impl = TritonAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="nvfp4",
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name="source_layer" if uses_shared_kv_cache else None,
    )
    assert impl.attn_type == AttentionType.DECODER

    kv_cache = torch.empty(
        num_blocks,
        2,
        block_size,
        num_kv_heads,
        full_dim,
        dtype=torch.uint8,
        device=device,
    )
    key_cache, value_cache = kv_cache.unbind(1)
    triton_reshape_and_cache_flash(
        key_cache_ref.reshape(-1, num_kv_heads, head_size),
        value_cache_ref.reshape(-1, num_kv_heads, head_size),
        key_cache,
        value_cache,
        torch.arange(num_blocks * block_size, dtype=torch.long, device=device),
        "nvfp4",
        k_scale,
        v_scale,
    )

    if not uses_shared_kv_cache:
        impl.do_kv_cache_update(layer, raw_key, raw_value, kv_cache, slot_mapping)

    causal, per_seq_causal = _causal_value(causal_kind, len(seq_lens), device)
    metadata = _make_triton_metadata(
        seq_lens,
        block_tables,
        slot_mapping,
        num_query_heads,
        head_size,
        causal,
        seq_threshold_3d,
    )
    output = torch.empty_like(query)
    impl.forward(
        layer,
        query,
        raw_key,
        raw_value,
        kv_cache,
        metadata,
        output=output,
    )

    key_cache, value_cache = kv_cache.unbind(1)
    (key_data_cache,), (key_scale_cache,) = nvfp4_kv_cache_split_views(key_cache)
    (value_data_cache,), (value_scale_cache,) = nvfp4_kv_cache_split_views(value_cache)
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
    if use_raw_current_kv:
        query_start = 0
        for seq_idx, (query_len, kv_len) in enumerate(seq_lens):
            context_len = kv_len - query_len
            for local_query_idx in range(query_len):
                kv_pos = context_len + local_query_idx
                block_idx = block_tables[seq_idx, kv_pos // block_size].item()
                slot_idx = kv_pos % block_size
                raw_idx = query_start + local_query_idx
                key_cache_dequant[block_idx, slot_idx] = raw_key[raw_idx]
                value_cache_dequant[block_idx, slot_idx] = raw_value[raw_idx]
            query_start += query_len

    ref_output = ref_paged_attn(
        query,
        key_cache_dequant,
        value_cache_dequant,
        query_lens,
        kv_lens,
        block_tables,
        scale,
        per_seq_causal,
    )
    torch.testing.assert_close(output, ref_output, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="NVFP4 Triton attention requires CUDA.",
)
@torch.inference_mode()
def test_attention_forward_nvfp4_opaque_cache_update_then_attention() -> None:
    """Validate the production opaque custom-op boundary for NVFP4 attention."""
    from tests.kernels.quantization.nvfp4_utils import dequant_nvfp4_kv_cache
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
    )

    set_random_seed(45)
    device = "cuda"
    dtype = torch.bfloat16
    seq_lens = [(94, 94)]
    per_seq_causal = [True]
    num_query_heads = 8
    num_kv_heads = 1
    head_size = 512
    block_size = 16
    scale = head_size**-0.5
    layer_name = "model.layers.0.self_attn.attn"

    query_lens = [query_len for query_len, _ in seq_lens]
    kv_lens = [kv_len for _, kv_len in seq_lens]
    num_query_tokens = sum(query_lens)
    block_tables, used_blocks = _make_block_tables(kv_lens, block_size, device)
    slot_mapping = _make_current_slot_mapping(
        seq_lens, block_tables, block_size, device
    )
    num_blocks = used_blocks + 4
    full_dim = nvfp4_kv_cache_full_dim(head_size)

    query = torch.randn(
        num_query_tokens,
        num_query_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    # attention_k_eq_v models pass the same current projection for K and V.
    raw_key = torch.randn(
        num_query_tokens,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    raw_value = raw_key
    key_cache_ref = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    value_cache_ref = key_cache_ref.clone()
    k_scale = (
        torch.cat((key_cache_ref.reshape(-1), raw_key.reshape(-1))).abs().amax() / 448.0
    ).to(torch.float32)
    v_scale = k_scale.clone()

    kv_cache = torch.empty(
        num_blocks,
        2,
        block_size,
        num_kv_heads,
        full_dim,
        dtype=torch.uint8,
        device=device,
    )
    key_cache, value_cache = kv_cache.unbind(1)
    triton_reshape_and_cache_flash(
        key_cache_ref.reshape(-1, num_kv_heads, head_size),
        value_cache_ref.reshape(-1, num_kv_heads, head_size),
        key_cache,
        value_cache,
        torch.arange(num_blocks * block_size, dtype=torch.long, device=device),
        "nvfp4",
        k_scale,
        v_scale,
    )

    vllm_config = VllmConfig(cache_config=CacheConfig(block_size=block_size))
    vllm_config.cache_config.cache_dtype = "nvfp4"
    with set_current_vllm_config(vllm_config):
        attn = Attention(
            num_heads=num_query_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            cache_config=vllm_config.cache_config,
            prefix=layer_name,
            attn_backend=AttentionBackendEnum.TRITON_ATTN.get_class(),
        )
    attn.kv_cache = kv_cache
    attn._k_scale = attn._k_scale.to(device)
    attn._v_scale = attn._v_scale.to(device)
    attn._k_scale.copy_(k_scale)
    attn._v_scale.copy_(v_scale)

    causal = torch.tensor(per_seq_causal, dtype=torch.bool, device=device)
    metadata = _make_triton_metadata(
        seq_lens,
        block_tables,
        slot_mapping,
        num_query_heads,
        head_size,
        causal,
        seq_threshold_3d=0,
    )
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(
            {layer_name: metadata},
            vllm_config,
            slot_mapping={layer_name: slot_mapping},
        ),
    ):
        output = attn(
            query.reshape(num_query_tokens, -1),
            raw_key.reshape(num_query_tokens, -1),
            raw_value.reshape(num_query_tokens, -1),
        )

    key_cache, value_cache = kv_cache.unbind(1)
    (key_data_cache,), (key_scale_cache,) = nvfp4_kv_cache_split_views(key_cache)
    (value_data_cache,), (value_scale_cache,) = nvfp4_kv_cache_split_views(value_cache)
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
    query_start = 0
    for seq_idx, (query_len, kv_len) in enumerate(seq_lens):
        context_len = kv_len - query_len
        for local_query_idx in range(query_len):
            kv_pos = context_len + local_query_idx
            block_idx = block_tables[seq_idx, kv_pos // block_size].item()
            slot_idx = kv_pos % block_size
            raw_idx = query_start + local_query_idx
            key_cache_dequant[block_idx, slot_idx] = raw_key[raw_idx]
            value_cache_dequant[block_idx, slot_idx] = raw_value[raw_idx]
        query_start += query_len

    ref_output = ref_paged_attn(
        query,
        key_cache_dequant,
        value_cache_dequant,
        query_lens,
        kv_lens,
        block_tables,
        scale,
        per_seq_causal,
    )
    torch.testing.assert_close(
        output.view_as(ref_output), ref_output, atol=3e-2, rtol=3e-2
    )


# ---- Flash Attention 4 backend test (native per_seq_causal) ----


@pytest.mark.skipif(
    _major != 9,
    reason="FA4 mixed causal attention requires Hopper (SM90).",
)
@pytest.mark.parametrize(
    "seq_lens",
    [[(1, 128), (5, 64), (1, 256)]],
)
@pytest.mark.parametrize(
    "per_seq_causal",
    [[True, False, True], [False, True, False]],
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_flash_attn4_mixed_causal(
    seq_lens: list[tuple[int, int]],
    per_seq_causal: list[bool],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
):
    if not current_platform.is_cuda():
        pytest.skip("Flash attention requires CUDA")

    try:
        from vllm.vllm_flash_attn import (
            fa_version_unsupported_reason,
            flash_attn_varlen_func,
            is_fa_version_supported,
        )
    except ImportError:
        pytest.skip("vllm_flash_attn not available")

    if not is_fa_version_supported(4):
        reason = fa_version_unsupported_reason(4)
        pytest.skip(f"FA4 not supported: {reason}")

    set_random_seed(42)
    device = "cuda"

    num_query_heads, num_kv_heads = num_heads
    assert len(seq_lens) == len(per_seq_causal)

    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    num_seqs = len(seq_lens)

    num_query_tokens = sum(query_lens)
    max_kv_len = max(kv_lens)
    max_num_blocks = (max_kv_len + block_size - 1) // block_size
    num_blocks = max_num_blocks * num_seqs + 10

    scale = head_size**-0.5
    query = torch.randn(
        num_query_tokens, num_query_heads, head_size, dtype=dtype, device=device
    )
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    value_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )

    block_tables_list = []
    for i in range(num_seqs):
        n_blocks = (kv_lens[i] + block_size - 1) // block_size
        blocks = list(range(i * max_num_blocks, i * max_num_blocks + n_blocks))
        blocks += [0] * (max_num_blocks - n_blocks)
        block_tables_list.append(blocks)
    block_tables = torch.tensor(block_tables_list, dtype=torch.int32, device=device)

    cu_seqlens_q = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    for i, ql in enumerate(query_lens):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + ql

    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    per_seq_causal_tensor = torch.tensor(
        per_seq_causal, dtype=torch.int32, device=device
    )

    ref_output = ref_paged_attn(
        query,
        key_cache,
        value_cache,
        query_lens,
        kv_lens,
        block_tables,
        scale,
        per_seq_causal,
    )

    output = torch.empty_like(query)
    flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(query_lens),
        seqused_k=seqused_k,
        max_seqlen_k=max(kv_lens),
        softmax_scale=scale,
        # The kernel must be compiled causal for `dynamic_causal` to take effect.
        causal=True,
        block_table=block_tables,
        softcap=0.0,
        dynamic_causal=per_seq_causal_tensor,
        fa_version=4,
    )

    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)
