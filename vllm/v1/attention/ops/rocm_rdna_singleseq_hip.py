# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lookup of the RDNA3.5 attention kernels built into ``_rocm_C``.

Each entry of ``csrc/rocm/rdna_singleseq_{decode,prefill}_configs.h`` is one
build, with its shape and launch knobs as compile-time constants.
"""

import contextlib
import functools
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KernelVariant:
    """One decode build: its shape and launch knobs."""

    head_size: int
    num_q_heads: int
    num_kv_heads: int
    max_query_len: int
    page_size: int
    layout: int  # 0 = NHD, 1 = HND
    # Most KV segments per (kv head, row group).
    max_segments: int = 1
    # Workgroups sharing a kv head's query rows.
    row_groups: int = 1
    # Fewest KV blocks per segment.
    min_segment_blocks: int = 1
    # Waves per workgroup.
    waves: int = 8
    # Waves splitting the head dim of a key tile; 0 is the kernel default.
    head_dim_split: int = 0
    # Prefetch a second tile per wave.
    prefetch: int = 0
    # Stage V in LDS beside K.
    v_in_lds: int = 0
    # Per-q-head dot products instead of WMMA.
    dot_product: int = 0
    dtype: torch.dtype = torch.float16
    # Sliding window in keys; 0 is full causal attention.
    window: int = 0

    def __post_init__(self) -> None:
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"kernel is built for fp16 or bf16, not {self.dtype}")

    @property
    def name(self) -> str:
        """The DecodeConfig fields joined by "_"."""
        return "rdna_singleseq_decode_" + "_".join(map(str, config_fields(self)))

    @property
    def rows_padded(self) -> int:
        """Query rows per workgroup, padded to 16."""
        gqa = self.num_q_heads // self.num_kv_heads
        rows = gqa * self.max_query_len // self.row_groups
        return -(-rows // 16) * 16

    def scratch_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        """Shapes of the split-KV partials (acc, m/l) and the counter count."""
        if self.dot_product:
            rows = self.num_q_heads * self.max_segments * self.max_query_len
            cnt = self.num_q_heads
        else:
            rows = (
                self.num_kv_heads
                * self.row_groups
                * self.max_segments
                * self.rows_padded
            )
            # Per (kv head, row group): arrivals, generation, two masks.
            cnt = 4 * self.num_kv_heads * self.row_groups
        return (rows, self.head_size), (rows,), cnt


# DecodeConfig's fields (csrc/rocm/rdna_singleseq_decode_configs.h) in order,
# as KernelVariant fields; the dtype is 0 for fp16 and 1 for bf16 (Dtype in
# rdna_singleseq_attn_types.h). Every build is HND.
DECODE_CONFIG_FIELDS = (
    "head_size",
    "num_q_heads",
    "num_kv_heads",
    "window",
    "max_query_len",
    "dtype",
    "page_size",
    "max_segments",
    "row_groups",
    "min_segment_blocks",
    "waves",
    "head_dim_split",
    "prefetch",
    "v_in_lds",
    "dot_product",
)
_HND = 1


def config_fields(variant: KernelVariant) -> list[int]:
    """The variant's DecodeConfig fields, in order."""
    return [
        int(variant.dtype == torch.bfloat16) if f == "dtype" else getattr(variant, f)
        for f in DECODE_CONFIG_FIELDS
    ]


def _lookup_key(v: KernelVariant) -> tuple:
    return (
        v.num_q_heads,
        v.num_kv_heads,
        v.head_size,
        v.max_query_len,
        v.page_size,
        v.window,
        v.dtype,
    )


def _built_configs(op_name: str, fields: tuple[str, ...]) -> list[dict[str, int]]:
    """The builds in _rocm_C as fields, by index, with the dtype as a torch
    dtype."""
    with contextlib.suppress(ImportError):
        import vllm._rocm_C  # noqa: F401
    try:
        flat = getattr(torch.ops._rocm_C, op_name)()
    except (AttributeError, RuntimeError):
        return []
    n = len(fields)
    configs = []
    for i in range(0, len(flat), n):
        f = dict(zip(fields, flat[i : i + n], strict=True))
        f["dtype"] = torch.bfloat16 if f["dtype"] else torch.float16
        configs.append(f)
    return configs


@functools.cache
def _builds() -> list[KernelVariant]:
    """The decode builds in _rocm_C, by index."""
    return [
        KernelVariant(**f, layout=_HND)
        for f in _built_configs("rdna_singleseq_decode_configs", DECODE_CONFIG_FIELDS)
    ]


@functools.cache
def _by_key() -> dict[tuple, KernelVariant]:
    return {_lookup_key(v): v for v in _builds()}


def variants() -> list[KernelVariant]:
    """Every decode build in _rocm_C."""
    return list(_builds())


def variant_for(
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    max_query_len: int,
    page_size: int,
    layout: int,
    window: int,
    dtype: torch.dtype,
) -> KernelVariant | None:
    """The build for one call, or None."""
    if layout != _HND:
        return None
    return _by_key().get(
        (
            num_q_heads,
            num_kv_heads,
            head_size,
            max_query_len,
            page_size,
            window,
            dtype,
        )
    )


class _Builtin:
    """A decode build in _rocm_C."""

    def __init__(self, index: int):
        self.index = index

    def decode_attn(self, q, kv_cache, block_table, out, *scratch_and_args):
        torch.ops._rocm_C.rdna_singleseq_decode_attn(
            self.index, q, kv_cache, block_table, out, *scratch_and_args
        )


_loaded: dict[KernelVariant, _Builtin | None] = {}


def load(variant: KernelVariant) -> _Builtin | None:
    """The variant in _rocm_C, or None if unavailable."""
    if variant not in _loaded:
        builds = _builds()
        _loaded[variant] = (
            _Builtin(builds.index(variant)) if variant in builds else None
        )
    return _loaded[variant]


def make_scratch(
    variant: KernelVariant, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate the split-KV scratch, outside CUDA-graph capture."""
    acc_shape, ml_shape, cnt = variant.scratch_shapes()
    opts = {"dtype": torch.float32, "device": device}
    return (
        torch.empty(acc_shape, **opts),
        torch.empty(ml_shape, **opts),
        torch.empty(ml_shape, **opts),
        # Zeroed once; every launch leaves them zeroed.
        torch.zeros(cnt, dtype=torch.int32, device=device),
    )


def expected_kv_cache_strides(variant: KernelVariant) -> tuple[int, int, int]:
    """Strides of (block, kv head, token) of the HND cache the kernel assumes:
    (NB, HKV, PAGE_SIZE, 2D)."""
    kv_row = 2 * variant.head_size
    page = variant.page_size * variant.num_kv_heads * kv_row
    return page, variant.page_size * kv_row, kv_row


# Most query tokens the decode kernel serves.
MAX_M = 8

# Head dims the decode kernel supports.
SUPPORTED_HEAD_SIZES = (64, 96, 128, 256, 512)


# ---------------------------------------------------------------- prefill
#
# A shape has one build per step of M; a build serves M from its min_query_len
# up to the next one's, and the first also every smaller M.

# Fewest query tokens the prefill kernel serves.
PREFILL_MIN_M = MAX_M + 1


@functools.cache
def prefill_target_workgroups(device: torch.device) -> int:
    """Workgroups below which the prefill kernel splits the KV (one per SIMD)."""
    return 4 * torch.cuda.get_device_properties(device).multi_processor_count


# PrefillConfig's fields (csrc/rocm/rdna_singleseq_prefill_configs.h) in order,
# as PrefillVariant fields, the dtype as for decode.
PREFILL_CONFIG_FIELDS = (
    "head_size",
    "num_q_heads",
    "num_kv_heads",
    "window",
    "dtype",
    "page_size",
    "min_query_len",
    "waves",
    "key_tile",
    "head_dim_split",
    "v_t",
    "double_buffer",
    "key_waves",
    "max_segments",
    "head_group",
)


@dataclass(frozen=True)
class PrefillVariant:
    """One prefill build: its shape, smallest M and launch knobs."""

    head_size: int
    num_q_heads: int
    num_kv_heads: int
    window: int
    page_size: int
    min_query_len: int
    waves: int
    key_tile: int
    head_dim_split: int
    v_t: int
    double_buffer: int
    key_waves: int
    max_segments: int
    head_group: int
    dtype: torch.dtype = torch.float16

    @property
    def name(self) -> str:
        """The PrefillConfig fields joined by "_"."""
        return "rdna_singleseq_prefill_" + "_".join(
            map(str, prefill_config_fields(self))
        )

    @property
    def block_rows(self) -> int:
        """Query rows per workgroup."""
        return self.waves // (self.head_dim_split * self.key_waves) * 16

    def scratch_rows(self, device: torch.device) -> int:
        """Most split-KV partial rows a launch may write."""
        return 2 * prefill_target_workgroups(device) * self.block_rows


def prefill_config_fields(variant: PrefillVariant) -> list[int]:
    """The variant's PrefillConfig fields, in order."""
    return [
        int(variant.dtype == torch.bfloat16) if f == "dtype" else getattr(variant, f)
        for f in PREFILL_CONFIG_FIELDS
    ]


@functools.cache
def _prefill_builds() -> list[PrefillVariant]:
    """The prefill builds in _rocm_C, by index."""
    return [
        PrefillVariant(**f)
        for f in _built_configs("rdna_singleseq_prefill_configs", PREFILL_CONFIG_FIELDS)
    ]


@functools.cache
def _prefill_table() -> dict[tuple, list[PrefillVariant]]:
    """The builds by M per (Hq, Hkv, D, page, window, dtype)."""
    table: dict[tuple, list[PrefillVariant]] = {}
    for v in _prefill_builds():
        key = (
            v.num_q_heads,
            v.num_kv_heads,
            v.head_size,
            v.page_size,
            v.window,
            v.dtype,
        )
        table.setdefault(key, []).append(v)
    for by_m in table.values():
        by_m.sort(key=lambda v: v.min_query_len)
    return table


def prefill_variants(
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    page_size: int,
    window: int,
    dtype: torch.dtype,
) -> list[PrefillVariant]:
    """Every row of one shape, sorted by M."""
    return _prefill_table().get(
        (num_q_heads, num_kv_heads, head_size, page_size, window, dtype), []
    )


def prefill_variant_for(
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    page_size: int,
    layout: int,
    window: int,
    dtype: torch.dtype,
    num_tokens: int,
) -> PrefillVariant | None:
    """The row for num_tokens query tokens, or None."""
    if layout != _HND or num_tokens < PREFILL_MIN_M:
        return None
    rows = prefill_variants(
        num_q_heads, num_kv_heads, head_size, page_size, window, dtype
    )
    served = [v for v in rows if v.min_query_len <= num_tokens]
    return served[-1] if served else (rows[0] if rows else None)


class _BuiltinPrefill:
    """A prefill build in _rocm_C."""

    def __init__(self, index: int):
        self.index = index

    def prefill_attn(self, q, kv_cache, block_table, out, *scratch_and_args):
        torch.ops._rocm_C.rdna_singleseq_prefill_attn(
            self.index, q, kv_cache, block_table, out, *scratch_and_args
        )


_prefill_loaded: dict[PrefillVariant, _BuiltinPrefill | None] = {}


def load_prefill(variant: PrefillVariant) -> _BuiltinPrefill | None:
    """The variant in _rocm_C, or None if unavailable."""
    if variant not in _prefill_loaded:
        builds = _prefill_builds()
        _prefill_loaded[variant] = (
            _BuiltinPrefill(builds.index(variant)) if variant in builds else None
        )
    return _prefill_loaded[variant]


def make_prefill_scratch(
    variants: list[PrefillVariant], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split-KV scratch (O, max/sum, counters) for any of `variants`."""
    rows = max(v.scratch_rows(device) for v in variants)
    head_size = max(v.head_size for v in variants)
    opts = {"dtype": torch.float32, "device": device}
    return (
        torch.empty(rows * head_size, **opts),
        torch.empty(rows * 2, **opts),
        torch.zeros(
            prefill_target_workgroups(device), dtype=torch.int32, device=device
        ),
    )
