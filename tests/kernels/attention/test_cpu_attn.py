# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import math

import pytest
import torch

from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.cpu_attn import _get_attn_isa

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

from vllm._custom_ops import (
    cpu_attention_with_kv_cache,
    cpu_attn_get_scheduler_metadata,
    cpu_attn_reshape_and_cache,
)

# Enable AMX tile data registers so isolated runs (e.g. -k fp8_amx) don't rely
# on ref_paged_attn's einsum to trigger oneDNN's _init_amx() first.
if torch.cpu._is_amx_tile_supported():
    torch.cpu._init_amx()


NUM_HEADS = [
    (4, 4),
    (8, 2),
    (9, 3),
]
HEAD_SIZES = [96, 128, 512]
HEAD_SIZES_VEC16 = [96, 80, 112, 128]
QTYPES = [torch.bfloat16, torch.half, torch.float32]
SLIDING_WINDOWS = [None, 256]
NUM_BLOCKS = [
    1024,
]
SEQ_LENS = [  # (q_len, kv_len)
    [(1, 213), (1, 1), (1, 312), (1, 7), (1, 7812)],  # decode batch
    [(2345, 2345), (5, 5), (3, 16), (134, 5131)],  # prefill batch
    [(992, 2456), (1, 1234), (98, 1145), (1, 4162), (2345, 2345)],  # mixed batch
]


def get_attn_isa(
    block_size: int | None = None,
    dtype: torch.dtype | None = None,
):
    # Delegate to _get_attn_isa so the fallback path applies the same arch
    # gating (e.g. RISC-V RVV is only chosen when the build's hardcoded
    # VLEN=128 kernel is actually present; on VLEN=256 / scalar hosts it
    # correctly falls through to vec/vec16).
    return _get_attn_isa(
        dtype if dtype is not None else torch.bfloat16,
        block_size if block_size else 32,
    )


# rand number generation takes too much time, cache rand tensors
@functools.lru_cache(maxsize=128, typed=False)
def tensor_cache(
    elem_num: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.randn(elem_num, dtype=dtype)

    return tensor


def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        num_remaining_heads = min(
            closest_power_of_2, total_num_heads - closest_power_of_2
        )
        extra_powers = torch.arange(
            start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes.float()


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
    alibi_slopes: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    dtype = query.dtype

    outputs: list[torch.Tensor] = []
    start_idx = 0

    if alibi_slopes is not None:
        alibi_slopes = alibi_slopes[:, None, None]

    if s_aux is not None:
        s_aux = s_aux.float()
        s_aux = s_aux[:, None, None]

    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len].float()
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len].float()
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len].float()

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

        if alibi_slopes is not None:
            q_start_pos = kv_len - query_len
            q_pos = q_start_pos + torch.arange(0, query_len)[None, :, None]
            kv_pos = torch.arange(0, kv_len)[None, None, :]
            dist = q_pos - kv_pos
            alibi_bias = -alibi_slopes * dist
            attn += alibi_bias

        attn.masked_fill_(mask, float("-inf"))

        if s_aux is not None:
            s_aux_ext = s_aux.repeat(1, query_len, 1)
            attn = torch.cat((s_aux_ext, attn), dim=-1)

        attn = torch.softmax(attn, dim=-1)

        if s_aux is not None:
            attn = attn[:, :, 1:]

        out = torch.einsum("hqk,khd->qhd", attn, v).to(dtype=dtype)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


_FP8_ATOL = {"fp8_e4m3": 0.2, "fp8_e5m2": 0.3}
_FP8_RTOL = 0.1


@torch.inference_mode()
def varlen_with_paged_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5
    token_num = sum(query_lens)

    # for n heads the set of slopes is the geometric sequence that starts
    # 2^(-8/n)
    alibi_slopes = _get_alibi_slopes(num_query_heads) if use_alibi else None

    s_aux = (
        15 * torch.rand((num_query_heads,), dtype=torch.bfloat16) if use_sink else None
    )

    is_fp8 = kv_cache_dtype != "auto"
    if is_fp8 and current_platform.get_cpu_architecture() != CpuArchEnum.X86:
        pytest.skip("FP8 KV cache only supported on x86")

    query = tensor_cache(
        elem_num=token_num * num_query_heads * head_size,
        dtype=dtype,
    )
    query = query.view(
        token_num,
        num_query_heads,
        head_size,
    )

    key_value = tensor_cache(
        elem_num=2 * num_blocks * num_kv_heads * block_size * head_size,
        dtype=dtype,
    )
    key_value = key_value.view(
        2,
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
    )
    if is_fp8:
        # Clamp KV to [-1, 1] so FP8 quantization error (<=12.5% for E4M3,
        # <=25% for E5M2) stays within the test tolerances regardless of
        # which tensor_cache values happen to be in use.
        key_value = key_value.clamp(-1, 1)
    key_cache, value_cache = key_value.unbind(0)

    # KV cache for CPU attention
    cache_dtype = torch.uint8 if is_fp8 else dtype
    packed_key_cache = torch.empty(
        num_blocks, num_kv_heads, block_size, head_size, dtype=cache_dtype
    )
    packed_value_cache = torch.empty_like(packed_key_cache)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    # use reshape_and_cache to pack key_cache and value_cache
    slot_mapping = torch.arange(0, num_blocks * block_size, dtype=torch.int64)
    fp8_kwargs: dict = (
        dict(k_scale=k_scale, v_scale=v_scale, kv_cache_dtype=kv_cache_dtype)
        if is_fp8
        else {}
    )
    cpu_attn_reshape_and_cache(
        key=key_cache.view(-1, num_kv_heads, head_size),
        value=value_cache.view(-1, num_kv_heads, head_size),
        key_cache=packed_key_cache,
        value_cache=packed_value_cache,
        slot_mapping=slot_mapping,
        isa=isa,
        **fp8_kwargs,
    )

    metadata = cpu_attn_get_scheduler_metadata(
        num_reqs=num_seqs,
        num_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        seq_lens=kv_lens_tensor,
        dtype=dtype,
        query_start_loc=cu_query_lens,
        causal=True,
        sliding_window_size=sliding_window if sliding_window is not None else -1,
        isa=isa,
        enable_kv_split=False,
    )

    out_without_split = torch.empty_like(query)
    cpu_attention_with_kv_cache(
        query=query,
        key_cache=packed_key_cache,
        value_cache=packed_value_cache,
        output=out_without_split,
        query_start_loc=cu_query_lens,
        seq_lens=kv_lens_tensor,
        scale=scale,
        causal=True,
        alibi_slopes=alibi_slopes,
        sliding_window=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        scheduler_metadata=metadata,
        s_aux=s_aux,
        **fp8_kwargs,
    )

    metadata = cpu_attn_get_scheduler_metadata(
        num_reqs=num_seqs,
        num_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        seq_lens=kv_lens_tensor,
        dtype=dtype,
        query_start_loc=cu_query_lens,
        causal=True,
        sliding_window_size=sliding_window if sliding_window is not None else -1,
        isa=isa,
        enable_kv_split=True,
    )

    out_with_split = torch.empty_like(query)
    cpu_attention_with_kv_cache(
        query=query,
        key_cache=packed_key_cache,
        value_cache=packed_value_cache,
        output=out_with_split,
        query_start_loc=cu_query_lens,
        seq_lens=kv_lens_tensor,
        scale=scale,
        causal=True,
        alibi_slopes=alibi_slopes,
        sliding_window=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        scheduler_metadata=metadata,
        s_aux=s_aux,
        **fp8_kwargs,
    )

    if is_fp8:
        # Build a float KV cache via the non-FP8 path and run float attention
        # to use as the reference.
        ref_key_cache = torch.empty(
            num_blocks, num_kv_heads, block_size, head_size, dtype=dtype
        )
        ref_value_cache = torch.empty_like(ref_key_cache)
        cpu_attn_reshape_and_cache(
            key=key_cache.view(-1, num_kv_heads, head_size),
            value=value_cache.view(-1, num_kv_heads, head_size),
            key_cache=ref_key_cache,
            value_cache=ref_value_cache,
            slot_mapping=slot_mapping,
            isa=isa,
        )
        ref_output = torch.empty_like(query)
        cpu_attention_with_kv_cache(
            query=query,
            key_cache=ref_key_cache,
            value_cache=ref_value_cache,
            output=ref_output,
            query_start_loc=cu_query_lens,
            seq_lens=kv_lens_tensor,
            scale=scale,
            causal=True,
            alibi_slopes=alibi_slopes,
            sliding_window=window_size,
            block_table=block_tables,
            softcap=soft_cap if soft_cap is not None else 0,
            scheduler_metadata=metadata,
            s_aux=s_aux,
        )
        atol = _FP8_ATOL[kv_cache_dtype]
        rtol = _FP8_RTOL
    else:
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
            alibi_slopes=alibi_slopes,
            s_aux=s_aux,
        )
        atol, rtol = 1.5e-2, 1e-2

    (
        torch.testing.assert_close(out_with_split, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(out_with_split - ref_output))}",
    )
    (
        torch.testing.assert_close(out_without_split, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(out_without_split - ref_output))}",
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", [96, 128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", QTYPES)
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", ["vec"])
def test_varlen_with_paged_kv_normal_vec(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
    kv_cache_dtype: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
        kv_cache_dtype=kv_cache_dtype,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", [96, 128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", ["amx"])
@pytest.mark.skipif(not torch.cpu._is_amx_tile_supported(), reason="no AMX support.")
def test_varlen_with_paged_kv_normal_amx(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
    kv_cache_dtype: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
        kv_cache_dtype=kv_cache_dtype,
    )


@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES_VEC16)
@pytest.mark.parametrize("block_size", [48])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", ["vec16"])
def test_varlen_with_paged_kv_normal_vec16(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
    )


@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", [96, 128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", QTYPES)
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", ["neon"])
@pytest.mark.skipif(
    current_platform.get_cpu_architecture() != CpuArchEnum.ARM,
    reason="Not an Arm CPU.",
)
def test_varlen_with_paged_kv_normal_neon(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3"])
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", [96, 128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", QTYPES)
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", ["rvv"])
@pytest.mark.skipif(
    current_platform.get_cpu_architecture() != CpuArchEnum.RISCV,
    reason="Not a RISC-V CPU.",
)
def test_varlen_with_paged_kv_normal_rvv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
    kv_cache_dtype: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
        kv_cache_dtype=kv_cache_dtype,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3"])
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", [96])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [50])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", [get_attn_isa()])
def test_varlen_with_paged_kv_softcap(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
    kv_cache_dtype: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
        kv_cache_dtype=kv_cache_dtype,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3"])
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", [96])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [True])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", [get_attn_isa()])
def test_varlen_with_paged_kv_alibi(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
    kv_cache_dtype: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
        kv_cache_dtype=kv_cache_dtype,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3"])
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", [96])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [True])
@pytest.mark.parametrize("isa", [get_attn_isa()])
def test_varlen_with_paged_kv_sink(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
    kv_cache_dtype: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
        kv_cache_dtype=kv_cache_dtype,
    )
