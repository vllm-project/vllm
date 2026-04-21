# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import ctypes.util
import functools
import math
import sys

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


@pytest.fixture(autouse=True, scope="session")
def _init_amx_tile_state():
    """Enable AMX tile state for this process via arch_prctl.

    On Linux, AMX tile instructions (_tile_loadconfig, _tile_dpbf16ps, etc.)
    cause SIGILL unless the process has opted in via
    arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA).  The call is
    idempotent (safe to call multiple times) and only needed once per process.
    torch.cpu._is_amx_tile_supported() checks hardware capability via CPUID
    but does NOT enable the tile state — this fixture fills that gap.
    """
    if sys.platform == "linux" and torch.cpu._is_amx_tile_supported():
        _ARCH_REQ_XCOMP_PERM = 0x1023
        _XFEATURE_XTILEDATA = 18
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        _libc.syscall(158, _ARCH_REQ_XCOMP_PERM, _XFEATURE_XTILEDATA)


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
    if block_size and dtype:
        return _get_attn_isa(dtype, block_size)
    else:
        if current_platform.get_cpu_architecture() == CpuArchEnum.ARM:
            return "neon"
        elif torch.cpu._is_amx_tile_supported():
            return "amx"
        else:
            return "vec"


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
    key_cache, value_cache = key_value.unbind(0)

    # KV cache for CPU attention
    packed_key_cache = torch.empty(
        num_blocks, num_kv_heads, block_size, head_size, dtype=dtype
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
    cpu_attn_reshape_and_cache(
        key=key_cache.view(-1, num_kv_heads, head_size),
        value=value_cache.view(-1, num_kv_heads, head_size),
        key_cache=packed_key_cache,
        value_cache=packed_value_cache,
        slot_mapping=slot_mapping,
        isa=isa,
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


# ---------------------------------------------------------------------------
# FP8 KV cache tests (Phase 1)
# ---------------------------------------------------------------------------


# Tolerances for FP8 vs float attention output comparisons.
# E5M2 (2 mantissa bits) introduces slightly more noise than E4M3 (3 bits).
_FP8_ATOL = {"fp8_e4m3": 0.2, "fp8_e5m2": 0.3}
_FP8_RTOL = 0.1


def _dequant_fp8_cache(
    cache: torch.Tensor, scale: float, kv_cache_dtype: str
) -> torch.Tensor:
    """Dequantize a uint8 FP8 cache tensor to float32.

    For E4M3: reinterpret as torch.float8_e4m3fn.
    For E5M2: FP8 byte b → FP16 bits = b << 8 (bias=15 matches FP16).
    """
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return cache.view(torch.float8_e4m3fn).float() * scale
    # fp8_e5m2: single left-shift maps byte to FP16 bit pattern
    fp16_bits = (cache.to(torch.int32) << 8).to(torch.int16).view(torch.float16)
    return fp16_bits.float() * scale


def test_fp8_backend_init():
    """Test E: CPUAttentionBackendImpl init with FP8 dtype raises no error."""
    from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackendImpl

    impl = CPUAttentionBackendImpl(
        num_heads=4,
        head_size=128,
        scale=128**-0.5,
        num_kv_heads=4,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8_e4m3",
    )
    assert impl.is_fp8_kv_cache


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp8_round_trip(dtype: torch.dtype):
    """Test A: quant→dequant round-trip relative error < 3%."""
    head_size = 128
    token_num = 64
    num_kv_heads = 4
    block_size = 32
    num_blocks = (token_num + block_size - 1) // block_size + 4
    scale = 1.0

    key = torch.randn(token_num, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(token_num, num_kv_heads, head_size, dtype=dtype)
    key_cache = torch.zeros(
        num_blocks, num_kv_heads, block_size, head_size, dtype=torch.uint8
    )
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.arange(token_num, dtype=torch.int64)

    cpu_attn_reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        "vec",
        k_scale=scale,
        v_scale=scale,
        kv_cache_dtype="fp8_e4m3",
    )

    # Dequantize value cache (row-major layout) and check round-trip error.
    # Retrieve the quantized values for token 0, head 0:
    block_idx = 0
    block_offset = 0
    v_fp8 = value_cache[block_idx, :, block_offset, :]  # [num_kv_heads, head_size]
    v_dq = v_fp8.view(torch.float8_e4m3fn).float() * scale  # dequantize
    v_ref = value[0].float()
    rel_err = (v_dq - v_ref).abs() / (v_ref.abs() + 1e-6)
    # E4M3 truncating encoder: max ~12.5% relative error, typical mean ~6%.
    assert rel_err.mean().item() < 0.08, (
        f"Round-trip relative error too high: {rel_err.mean().item():.4f}"
    )


@torch.inference_mode()
@pytest.mark.parametrize("kv_cache_dtype", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("k_scale,v_scale", [(1.0, 1.0), (0.5, 2.0)])
def test_fp8_reshape_and_cache_round_trip(
    dtype: torch.dtype, k_scale: float, v_scale: float, kv_cache_dtype: str
):
    """Tests the encode path of cpu_attn_reshape_and_cache with FP8 kv_cache_dtype.

    Encodes KV to FP8, manually dequantizes in Python, then runs standard float
    attention to verify the encode round-trip is accurate enough.  This test
    does NOT use on-the-fly dequant; the on-the-fly dequant kernel
    is tested separately in test_fp8_end_to_end_native.
    """
    set_random_seed(0)
    num_query_heads = 4
    num_kv_heads = 4
    head_size = 128
    block_size = 32
    num_blocks = 64
    seq_lens = [(1, 128), (1, 64)]
    isa = "vec"  # This test targets the VEC reshape path explicitly.

    scale = head_size**-0.5
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_seqs = len(seq_lens)
    token_num = sum(query_lens)
    max_kv_len = max(kv_lens)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size

    query = torch.randn(token_num, num_query_heads, head_size, dtype=dtype)
    # Ground-truth KV in float (original, before any quant)
    kv_flat_size = num_blocks * block_size * num_kv_heads * head_size
    key_flat = torch.randn(kv_flat_size, dtype=dtype)
    value_flat = torch.randn(kv_flat_size, dtype=dtype)
    key_orig = key_flat.view(num_blocks, block_size, num_kv_heads, head_size)
    value_orig = value_flat.view(num_blocks, block_size, num_kv_heads, head_size)

    slot_mapping = torch.arange(num_blocks * block_size, dtype=torch.int64)

    # --- FP8 path ---
    key_cache_fp8 = torch.zeros(
        num_blocks, num_kv_heads, block_size, head_size, dtype=torch.uint8
    )
    value_cache_fp8 = torch.zeros_like(key_cache_fp8)
    cpu_attn_reshape_and_cache(
        key_orig.view(-1, num_kv_heads, head_size),
        value_orig.view(-1, num_kv_heads, head_size),
        key_cache_fp8,
        value_cache_fp8,
        slot_mapping,
        isa,
        k_scale=k_scale,
        v_scale=v_scale,
        kv_cache_dtype=kv_cache_dtype,
    )
    # Dequantize for attention
    key_cache_f32 = _dequant_fp8_cache(key_cache_fp8, k_scale, kv_cache_dtype)
    value_cache_f32 = _dequant_fp8_cache(value_cache_fp8, v_scale, kv_cache_dtype)

    # --- Reference path: non-quantized float cache ---
    key_cache_ref = torch.zeros(
        num_blocks, num_kv_heads, block_size, head_size, dtype=dtype
    )
    value_cache_ref = torch.zeros_like(key_cache_ref)
    cpu_attn_reshape_and_cache(
        key_orig.view(-1, num_kv_heads, head_size),
        value_orig.view(-1, num_kv_heads, head_size),
        key_cache_ref,
        value_cache_ref,
        slot_mapping,
        isa,
    )

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
    block_tables = torch.arange(
        max_num_blocks_per_seq * num_seqs, dtype=torch.int32
    ).view(num_seqs, max_num_blocks_per_seq)

    scheduler_metadata = cpu_attn_get_scheduler_metadata(
        num_reqs=num_seqs,
        num_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        seq_lens=kv_lens_tensor,
        dtype=dtype,
        query_start_loc=cu_query_lens,
        causal=True,
        sliding_window_size=-1,
        isa=isa,
        enable_kv_split=True,
    )

    output_fp8 = torch.zeros(token_num, num_query_heads, head_size, dtype=dtype)
    cpu_attention_with_kv_cache(
        query=query,
        key_cache=key_cache_f32.to(dtype),
        value_cache=value_cache_f32.to(dtype),
        output=output_fp8,
        query_start_loc=cu_query_lens,
        seq_lens=kv_lens_tensor,
        scale=scale,
        causal=True,
        alibi_slopes=None,
        sliding_window=(-1, -1),
        block_table=block_tables,
        softcap=0.0,
        scheduler_metadata=scheduler_metadata,
        s_aux=None,
    )

    output_ref = torch.zeros(token_num, num_query_heads, head_size, dtype=dtype)
    cpu_attention_with_kv_cache(
        query=query,
        key_cache=key_cache_ref,
        value_cache=value_cache_ref,
        output=output_ref,
        query_start_loc=cu_query_lens,
        seq_lens=kv_lens_tensor,
        scale=scale,
        causal=True,
        alibi_slopes=None,
        sliding_window=(-1, -1),
        block_table=block_tables,
        softcap=0.0,
        scheduler_metadata=scheduler_metadata,
        s_aux=None,
    )

    # FP8 output should be close to full-float output (within quantization noise).
    # E4M3 (3-bit mantissa): atol=0.1; E5M2 (2-bit mantissa): slightly more noise.
    atol = 0.15 if kv_cache_dtype == "fp8_e5m2" else 0.1
    torch.testing.assert_close(output_fp8, output_ref, atol=atol, rtol=0.1)


# ---------------------------------------------------------------------------
# FP8 KV cache tests (Phase 2) — on-the-fly dequantization via
# cpu_attention_with_kv_cache with kv_cache_dtype (no Python-level eager dequant).
# ---------------------------------------------------------------------------


def _fp8_attn_e2e(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    k_scale: float,
    v_scale: float,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    isa: str,
) -> None:
    """Run FP8 on-the-fly dequant attention and compare output to float reference.

    Shared body for test_fp8_end_to_end_native and test_fp8_amx_end_to_end;
    the two tests differ only in ISA selection and parametrize coverage.
    """
    set_random_seed(0)
    block_size = 32
    num_query_heads, num_kv_heads = num_heads
    scale = head_size**-0.5

    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_seqs = len(seq_lens)
    token_num = sum(query_lens)
    max_kv_len = max(kv_lens)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    # num_blocks must cover all block table entries (num_seqs * max_num_blocks_per_seq).
    num_blocks = max_num_blocks_per_seq * num_seqs

    query = torch.randn(token_num, num_query_heads, head_size, dtype=dtype)

    kv_flat_size = num_blocks * block_size * num_kv_heads * head_size
    key_orig = torch.randn(kv_flat_size, dtype=dtype).view(
        num_blocks, block_size, num_kv_heads, head_size
    )
    value_orig = torch.randn(kv_flat_size, dtype=dtype).view(
        num_blocks, block_size, num_kv_heads, head_size
    )
    slot_mapping = torch.arange(num_blocks * block_size, dtype=torch.int64)

    key_cache_fp8 = torch.zeros(
        num_blocks, num_kv_heads, block_size, head_size, dtype=torch.uint8
    )
    value_cache_fp8 = torch.zeros_like(key_cache_fp8)
    cpu_attn_reshape_and_cache(
        key_orig.view(-1, num_kv_heads, head_size),
        value_orig.view(-1, num_kv_heads, head_size),
        key_cache_fp8,
        value_cache_fp8,
        slot_mapping,
        isa,
        k_scale=k_scale,
        v_scale=v_scale,
        kv_cache_dtype=kv_cache_dtype,
    )

    key_cache_ref = torch.zeros(
        num_blocks, num_kv_heads, block_size, head_size, dtype=dtype
    )
    value_cache_ref = torch.zeros_like(key_cache_ref)
    cpu_attn_reshape_and_cache(
        key_orig.view(-1, num_kv_heads, head_size),
        value_orig.view(-1, num_kv_heads, head_size),
        key_cache_ref,
        value_cache_ref,
        slot_mapping,
        isa,
    )

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
    block_tables = torch.arange(
        max_num_blocks_per_seq * num_seqs, dtype=torch.int32
    ).view(num_seqs, max_num_blocks_per_seq)

    scheduler_metadata = cpu_attn_get_scheduler_metadata(
        num_reqs=num_seqs,
        num_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        seq_lens=kv_lens_tensor,
        dtype=dtype,
        query_start_loc=cu_query_lens,
        causal=True,
        sliding_window_size=-1,
        isa=isa,
        enable_kv_split=True,
    )

    output_fp8 = torch.zeros(token_num, num_query_heads, head_size, dtype=dtype)
    cpu_attention_with_kv_cache(
        query=query,
        key_cache=key_cache_fp8,
        value_cache=value_cache_fp8,
        output=output_fp8,
        query_start_loc=cu_query_lens,
        seq_lens=kv_lens_tensor,
        scale=scale,
        causal=True,
        alibi_slopes=None,
        sliding_window=(-1, -1),
        block_table=block_tables,
        softcap=0.0,
        scheduler_metadata=scheduler_metadata,
        s_aux=None,
        k_scale=k_scale,
        v_scale=v_scale,
        kv_cache_dtype=kv_cache_dtype,
    )

    output_ref = torch.zeros(token_num, num_query_heads, head_size, dtype=dtype)
    cpu_attention_with_kv_cache(
        query=query,
        key_cache=key_cache_ref,
        value_cache=value_cache_ref,
        output=output_ref,
        query_start_loc=cu_query_lens,
        seq_lens=kv_lens_tensor,
        scale=scale,
        causal=True,
        alibi_slopes=None,
        sliding_window=(-1, -1),
        block_table=block_tables,
        softcap=0.0,
        scheduler_metadata=scheduler_metadata,
        s_aux=None,
    )

    torch.testing.assert_close(
        output_fp8, output_ref, atol=_FP8_ATOL[kv_cache_dtype], rtol=_FP8_RTOL
    )


@torch.inference_mode()
@pytest.mark.parametrize("kv_cache_dtype", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 213), (1, 1), (1, 312), (1, 7), (1, 7812)],  # decode
        [(992, 2456), (1, 1234), (98, 1145), (1, 4162), (2345, 2345)],  # mixed
    ],
)
@pytest.mark.parametrize("num_heads", [(4, 4), (8, 2)])
@pytest.mark.parametrize("head_size", [96, 128])
@pytest.mark.parametrize("k_scale,v_scale", [(1.0, 1.0), (0.5, 2.0)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fp8_end_to_end_native(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    k_scale: float,
    v_scale: float,
    dtype: torch.dtype,
    kv_cache_dtype: str,
) -> None:
    """Phase 2: cpu_attention_with_kv_cache performs on-the-fly dequant.

    The output must match running the same attention over a float KV cache
    built from the same original values (within FP8 quantisation noise).
    """
    isa = _get_attn_isa(dtype, block_size=32, head_size=head_size)
    if isa == "amx":
        isa = "vec"
    _fp8_attn_e2e(
        seq_lens, num_heads, head_size, k_scale, v_scale, dtype, kv_cache_dtype, isa
    )


# ---------------------------------------------------------------------------
# FP8 KV cache tests — AMX layout (on-the-fly dequantization via AMX tiles).
# ---------------------------------------------------------------------------


@torch.inference_mode()
@pytest.mark.skipif(not torch.cpu._is_amx_tile_supported(), reason="no AMX support")
@pytest.mark.parametrize("kv_cache_dtype", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 213), (1, 1), (1, 312), (1, 7), (1, 7812)],  # decode
        [(32, 256), (1, 128), (16, 512)],  # mixed
    ],
)
@pytest.mark.parametrize("num_heads", [(4, 4), (8, 2)])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("k_scale,v_scale", [(1.0, 1.0), (0.5, 2.0)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fp8_amx_end_to_end(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    k_scale: float,
    v_scale: float,
    dtype: torch.dtype,
    kv_cache_dtype: str,
) -> None:
    """FP8 AMX: reshape with AMX layout, run AMX attention, compare to float.

    Uses isa='amx' for both reshape and the scheduler metadata so that the
    full AMX tile path is exercised (TileGemm224/122<c10::Float8_e4m3fn/e5m2>).
    """
    _fp8_attn_e2e(
        seq_lens, num_heads, head_size, k_scale, v_scale, dtype, kv_cache_dtype, "amx"
    )
