# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed host configuration and selection for segmented attention."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Literal, ParamSpec, TypedDict, TypeGuard, TypeVar, cast

from vllm.triton_utils import triton

KVUpcastMode = Literal["normal", "native1", "native4"]
KV_UPCAST_MODES: tuple[KVUpcastMode, ...] = ("normal", "native1", "native4")
MMPrefixMode = Literal["none", "clamped", "unclamped"]
MM_PREFIX_SPAN_BUCKETS = (256, 576, 1152, 2048, 4096, 8192)
MAX_QUERY_LEN = 8192
MAX_SPLITS = 64
MAX_LONG_EXTEND_WORKSPACE_BYTES = 32 * 1024 * 1024
_P = ParamSpec("_P")
_R = TypeVar("_R")


def _cache_config_selector(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Keep selector argument types through functools' cache wrapper."""
    return cast(Callable[_P, _R], lru_cache(maxsize=512)(func))


class SegmentedLaunchOptions(TypedDict, total=False):
    waves_per_eu: int
    reduce_d: int
    reduce_warps: int
    pv_split: bool
    prefix_fast: bool
    qk_pipeline: int | None
    qk_unroll: int | None
    native_fp8_upcast: bool
    native_fp8_pack: Literal[1, 4]


class SegmentedLaunchConfig(SegmentedLaunchOptions):
    """Required launch fields, also accepted from legacy config callers."""

    bm: int
    bn: int
    bk: int
    splits: int
    warps: int
    stages: int


class SegmentedAttentionConfig(SegmentedLaunchConfig):
    """Canonical selector and autotuner output with an explicit conversion."""

    kv_upcast: KVUpcastMode


class SegmentedConfigOverrides(SegmentedLaunchOptions, total=False):
    """Partial changes to a complete configuration during startup tuning."""

    bm: int
    bn: int
    bk: int
    splits: int
    warps: int
    stages: int
    kv_upcast: KVUpcastMode


def is_segmented_kv_upcast_mode(value: object) -> TypeGuard[KVUpcastMode]:
    return isinstance(value, str) and value in KV_UPCAST_MODES


def segmented_query_capacity(max_query_len: int) -> int:
    """Round a supported query length to its compiled power-of-two capacity."""
    if not 1 <= max_query_len <= MAX_QUERY_LEN:
        raise ValueError(f"Query length must be in [1, {MAX_QUERY_LEN}]")
    return 1 << (max_query_len - 1).bit_length()


def segmented_mm_prefix_bucket(span: int, ranges: int) -> tuple[int, int]:
    """Bound image span and range-loop work without inspecting device tensors."""
    if span <= 0 or ranges <= 0:
        return 0, 0
    span_bucket = next((s for s in MM_PREFIX_SPAN_BUCKETS if s >= span), span)
    return span_bucket, 1 << (ranges - 1).bit_length()


def segmented_attention_span(
    max_seq_len: int,
    max_query_len: int,
    sliding_window: int,
    mm_prefix_mode: MMPrefixMode = "none",
    mm_prefix_max_span: int = 0,
) -> int:
    """Bound the visited KV span, including reopened past and future image keys."""
    if sliding_window < 0:
        return max_seq_len
    if mm_prefix_mode == "none":
        return min(max_seq_len, sliding_window + max_query_len)
    if mm_prefix_max_span <= 0:
        return max_seq_len
    future = mm_prefix_max_span - 1
    past = sliding_window
    if mm_prefix_mode == "unclamped":
        past = max(past, future)
    return min(max_seq_len, max_query_len + past + future)


def _query_capacity_buckets() -> tuple[int, ...]:
    return tuple(1 << exponent for exponent in range(MAX_QUERY_LEN.bit_length()))


def _long_extend_splits(
    base_splits: int,
    batch: int,
    query_len: int,
    seq_len: int,
    hq: int,
    hk: int,
    dim: int,
    bm: int,
    *,
    fp8: bool = False,
) -> int:
    """Add long-prefix parallelism while bounding the extra split workspace."""
    if seq_len < 32768:
        return base_splits
    scratch_per_split = batch * segmented_query_capacity(query_len) * hq * (dim + 1) * 4
    scratch_limit = MAX_LONG_EXTEND_WORKSPACE_BYTES * dim // 128
    scratch_splits = scratch_limit // scratch_per_split
    scratch_cap = 1 << (scratch_splits.bit_length() - 1) if scratch_splits else 1
    if dim == 512 and hq // hk == 8:
        cache_target = (16 if fp8 else 32) * 1024**2
        kv_bytes = 2 * batch * hk * seq_len * dim * (1 if fp8 else 2)
        cache_splits: int = triton.next_power_of_2(triton.cdiv(kv_bytes, cache_target))
        if fp8:
            cache_splits = max(4, cache_splits)
        elif kv_bytes <= 64 * 1024**2:
            cache_splits = 1
        return max(base_splits, min(32, scratch_cap, cache_splits))
    groups = batch * hk * triton.cdiv(query_len * (hq // hk), bm)
    split_cap = 16 if seq_len < 131072 else 32
    occupancy_splits: int = triton.next_power_of_2(triton.cdiv(512, groups))
    return max(base_splits, min(split_cap, scratch_cap, occupancy_splits))


def segmented_kv_upcast_mode(config: Mapping[str, object]) -> KVUpcastMode:
    """Read the conversion mode, including pre-mode experimental configs."""
    mode = config.get("kv_upcast")
    if mode is None:
        mode = (
            f"native{config.get('native_fp8_pack', 1)}"
            if config.get("native_fp8_upcast", False)
            else "normal"
        )
    if not is_segmented_kv_upcast_mode(mode):
        raise ValueError(f"Unsupported segmented KV upcast mode: {mode}")
    return mode


@_cache_config_selector
def select_segmented_config(
    batch: int,
    max_query_len: int,
    max_seq_len: int,
    hq: int,
    hk: int,
    dim: int,
    fp8: bool,
    *,
    sliding_window: int = -1,
    native_fp8: bool = True,
    mm_prefix_mode: MMPrefixMode = "none",
    mm_prefix_max_span: int = 0,
    mm_prefix_max_ranges: int = 0,
) -> SegmentedAttentionConfig:
    """Select launch geometry and KV conversion for the attention workload."""
    if mm_prefix_mode != "none":
        max_seq_len = segmented_attention_span(
            max_seq_len,
            max_query_len,
            sliding_window,
            mm_prefix_mode,
            mm_prefix_max_span,
        )
    cfg = _select_segmented_launch_config(
        batch, max_query_len, max_seq_len, hq, hk, dim, fp8
    )
    _refine_segmented_launch_config(
        cfg,
        batch,
        max_query_len,
        max_seq_len,
        hq,
        hk,
        dim,
        fp8,
        native_fp8,
        sliding_window,
    )
    mm_image_tile = (
        (mm_prefix_mode == "clamped" or (mm_prefix_mode == "unclamped" and fp8))
        and dim == 256
        and max_query_len >= 256
        and sliding_window >= 511
        and hq // hk <= 4
        and (not fp8 or native_fp8)
        and not cfg.get("pv_split", False)
    )
    short_groups = (
        batch * hk * triton.cdiv(max_query_len * (hq // hk), 32)
        if mm_prefix_mode != "none"
        else 0
    )
    mm_short_tile = (
        mm_prefix_mode != "none"
        and dim == 256
        and 32 < max_query_len <= 128
        and max_seq_len <= 128
        and (max_seq_len > 64 or (not fp8 and short_groups < 64))
        and mm_prefix_max_ranges <= 1
        and hq // hk <= 4
        and (not fp8 or native_fp8)
    )
    if mm_short_tile:
        narrow = not fp8 and short_groups < 64
        cfg.update(
            {
                "bm": 16 if narrow else 32,
                "bn": 64 if narrow else 128,
                "bk": 128 if narrow else 64,
                "warps": 4 if narrow else 8,
                "stages": 1,
                "splits": 1,
                "waves_per_eu": 2,
                "pv_split": False,
                "prefix_fast": False,
                "qk_pipeline": None,
                "qk_unroll": None,
            }
        )
    if mm_image_tile:
        cfg.update(
            {
                "bm": 64,
                "bn": 64,
                "bk": 64,
                "warps": 8,
                "stages": 1,
                "waves_per_eu": 6,
                "pv_split": False,
                "prefix_fast": False,
                "qk_pipeline": None,
                "qk_unroll": None,
            }
        )
        groups = batch * hk * triton.cdiv(max_query_len * (hq // hk), cfg["bm"])
        splits = min(32, triton.next_power_of_2(triton.cdiv(64, groups)))
        while splits > 1 and max_seq_len < splits * cfg["bn"] * 2:
            splits //= 2
        cfg["splits"] = splits
        if (
            max_query_len >= 1024
            and mm_prefix_max_span > 0
            and max_query_len > 4 * mm_prefix_max_span * max(1, mm_prefix_max_ranges)
        ):
            if fp8:
                cfg.update({"bn": 32, "warps": 4})
            else:
                # Sparse-image prompts retain the wide BF16 accumulator schedule.
                cfg.update({"stages": 3, "pv_split": True, "qk_pipeline": 3})
    mode: KVUpcastMode = (
        _select_segmented_kv_upcast(
            cfg, batch, max_query_len, max_seq_len, hq, hk, dim, sliding_window
        )
        if fp8 and native_fp8
        else "normal"
    )
    if (mm_image_tile or mm_short_tile) and fp8 and native_fp8:
        mode = "native4"
    return SegmentedAttentionConfig(**cfg, kv_upcast=mode)


def _refine_segmented_launch_config(
    cfg: SegmentedLaunchConfig,
    batch: int,
    query_len: int,
    span: int,
    hq: int,
    hk: int,
    dim: int,
    fp8: bool,
    native_fp8: bool,
    sliding_window: int,
) -> None:
    """Use native-friendly tiles without changing unsupported-format defaults."""
    if fp8 and not native_fp8:
        return
    if fp8 and dim == 128 and query_len >= 256 and 0 <= sliding_window < 255:
        cfg["splits"] = 1
        return
    gqa = hq // hk
    if (
        dim == 512
        and gqa == 8
        and sliding_window < 0
        and query_len >= 256
        and 8192 <= span < 32768
        and span >= query_len + 256
    ):
        query_groups = batch * triton.cdiv(query_len * gqa, cfg["bm"])
        if query_groups * cfg["splits"] < 96:
            scratch = batch * segmented_query_capacity(query_len) * hq * (dim + 1) * 4
            scratch_splits = MAX_LONG_EXTEND_WORKSPACE_BYTES * dim // 128 // scratch
            scratch_cap = 1 << (max(1, scratch_splits).bit_length() - 1)
            kv_bytes = 2 * batch * hk * span * dim * (1 if fp8 else 2)
            cache_target = (16 if fp8 else 32) * 1024**2
            cache_cap = triton.next_power_of_2(triton.cdiv(kv_bytes, cache_target))
            if fp8:
                cache_cap = max(4, cache_cap)
            elif kv_bytes <= 64 * 1024**2:
                cache_cap = 1
            cfg["splits"] = max(
                cfg["splits"],
                min(2 * cfg["splits"], scratch_cap, cache_cap),
            )
    if not fp8 or span < 1024 or 0 <= sliding_window < 255:
        return
    if dim == 64 and query_len >= 64:
        cfg.update(
            {
                "bm": 128 if span >= 8192 and gqa >= 2 else 64,
                "bn": 32 if span >= 8192 and gqa >= 2 else 64,
                "bk": 64,
                "warps": 4,
                "stages": 1,
                "waves_per_eu": 2,
            }
        )
    elif dim in (128, 256) and 8 < query_len <= 32:
        cfg.update(
            {"bm": 32, "bn": 64, "bk": 64, "warps": 4, "stages": 1, "waves_per_eu": 2}
        )
    elif (
        dim == 128
        and gqa <= 2
        and query_len >= 256
        and span >= max(8192, query_len + 256)
    ):
        cfg.update(
            {
                "bm": 64,
                "bn": 128,
                "bk": 64,
                "warps": 8,
                "stages": 3,
                "waves_per_eu": 6,
                "prefix_fast": True,
                "qk_pipeline": 3,
            }
        )
    elif dim == 512 and gqa != 8 and query_len > 8:
        if query_len > 32 and query_len * gqa >= 256:
            cfg.update(
                {
                    "bm": 64,
                    "bn": 64 if span >= 8192 else 32,
                    "bk": 64,
                    "warps": 8,
                    "stages": 1,
                    "waves_per_eu": 6,
                    "pv_split": True,
                    "prefix_fast": False,
                    "qk_pipeline": None if span >= 8192 else 3,
                    "qk_unroll": None,
                }
            )
            if gqa == 1 and span >= 8192:
                groups = batch * hk * triton.cdiv(query_len, cfg["bm"])
                cfg["splits"] = max(
                    cfg["splits"],
                    min(32, triton.next_power_of_2(triton.cdiv(192, groups))),
                )
        else:
            cfg.update(
                {
                    "bm": 32,
                    "bn": 64,
                    "bk": 64,
                    "warps": 8,
                    "stages": 2,
                    "waves_per_eu": 2,
                }
            )


def _select_segmented_kv_upcast(
    cfg: SegmentedLaunchConfig,
    batch: int,
    query_len: int,
    span: int,
    hq: int,
    hk: int,
    dim: int,
    sliding_window: int,
) -> KVUpcastMode:
    """Choose conversion for the selected layout and visible attention span."""
    gqa = hq // hk
    kv_groups = batch * hk
    kv_bytes = 2 * kv_groups * span * dim
    if dim == 64:
        return "native1" if cfg["bm"] >= 32 and kv_bytes <= 64 * 1024 else "native4"
    if dim == 128:
        if query_len <= 2:
            return "native4" if span < 32768 else "normal"
        if (
            8 < query_len <= 32
            and cfg["bn"] == 64
            and cfg["stages"] == 1
            or cfg.get("prefix_fast", False)
            or kv_bytes <= 64 * 1024
            or query_len >= 64
            and (gqa <= 8 or query_len >= 512)
        ):
            return "native4"
        return "normal"
    if dim == 256:
        if query_len <= 2:
            return "native4" if span > 1 else "normal"
        if (
            query_len > 8
            and cfg["bm"] == 32
            and cfg["bn"] == 64
            and cfg["stages"] == 1
            and (gqa >= 4 or query_len < 128 or query_len >= 512)
        ):
            return "native4"
        return "normal"
    if dim == 512:
        if gqa != 8:
            if cfg["bm"] == 16 or cfg["warps"] == 8:
                return "native4"
            return "native1" if query_len > 32 else "normal"
        if query_len <= 2:
            short = span <= 256 and kv_groups <= 4
            if query_len == 2 and kv_groups == 2:
                return "native4"
            if not (span <= 2048 or not short and (kv_groups == 1 or kv_groups >= 8)):
                return "native1" if span <= 16384 else "normal"
            packed = kv_groups >= 16 or span <= 2048 and (span <= 1024 or hk <= 2)
            return "native4" if packed else "native1"
        prefix = cfg.get("prefix_fast", False)
        if query_len > 32:
            if (
                prefix
                and sliding_window < 0
                and 256 <= query_len <= 512
                and (hk <= 2 or 8192 <= span < 32768 and cfg["splits"] >= 2)
                and span >= 4096
            ):
                return "native4"
            return "native1"
        return "native4" if prefix else "native1"
    return "normal"


def _select_segmented_launch_config(
    batch: int,
    max_query_len: int,
    max_seq_len: int,
    hq: int,
    hk: int,
    dim: int,
    fp8: bool,
) -> SegmentedLaunchConfig:
    """Choose launch geometry within the segmented workspace bounds."""
    cfg: SegmentedLaunchConfig
    qcap = min(max_query_len, MAX_QUERY_LEN)
    if batch < 1 or qcap < 1:
        raise ValueError("Batch and query capacity must be positive")
    gqa = hq // hk
    kv_groups = batch * hk
    wide_gqa = dim == 512 and gqa == 8
    decode = qcap <= 2 and (not wide_gqa or qcap == 1 or fp8 or kv_groups == 1)

    if decode:
        if dim == 64 and qcap == 1 and batch == 1 and max_seq_len >= 131072:
            return dict(
                bm=16,
                bn=32,
                bk=64,
                splits=256,
                warps=1,
                stages=1,
            )
        if dim == 64 and max_seq_len <= 256:
            return dict(
                bm=16,
                bn=32 if fp8 else 64,
                bk=64,
                splits=1,
                warps=4,
                stages=1,
            )
        if dim == 64 and max_seq_len >= 32768:
            groups = batch * hk * triton.cdiv(qcap * (hq // hk), 16)
            return dict(
                bm=16,
                bn=32,
                bk=64,
                splits=min(64, triton.next_power_of_2(triton.cdiv(128, groups))),
                warps=4,
                stages=1,
            )
        if wide_gqa:
            short = max_seq_len <= 256 and kv_groups <= 4
            bn = 32 if short or not fp8 else 64
            target = 128 if fp8 or kv_groups <= 4 else 64
            if qcap == 2 and not fp8:
                target = 64
            splits = min(128, triton.next_power_of_2(triton.cdiv(target, kv_groups)))
            if short:
                splits = min(4, splits)
            cfg = dict(
                bm=16,
                bn=bn,
                bk=64,
                splits=splits,
                warps=8 if short else 4,
                stages=1,
                reduce_d=64 if fp8 or batch * qcap * hq < 64 else dim,
            )
            if fp8:
                if qcap == 1 and kv_groups == 8:
                    cfg["reduce_warps"] = 2
            elif short and qcap == 1:
                cfg["qk_unroll"] = 4
        elif kv_groups == 1:
            cfg = dict(
                bm=16,
                bn=64,
                bk=min(128, dim),
                splits=32,
                warps=4,
                stages=1,
            )
        else:
            groups = kv_groups * triton.cdiv(qcap * gqa, 16)
            cfg = dict(
                bm=16,
                bn=32,
                bk=dim,
                splits=min(
                    MAX_SPLITS,
                    triton.next_power_of_2(triton.cdiv(64, groups)),
                ),
                warps=4,
                stages=1,
            )
        if wide_gqa:
            tiles = max(1, triton.cdiv(max_seq_len, cfg["bn"]))
            cfg["splits"] = min(cfg["splits"], 1 << (tiles.bit_length() - 1))
        else:
            while cfg["splits"] > 1 and max_seq_len < cfg["splits"] * cfg["bn"] * 2:
                cfg["splits"] //= 2
        if dim == 512:
            cfg.update({"bk": 64})
        return cfg

    if qcap <= 8:
        bm, bn = 16, 64
        bk, stages = 64, 1
    elif qcap <= 32:
        bm, bk = 32, 64
        bn, stages = (32, 2) if fp8 else (64, 1)
    else:
        bm, bn, bk = 32, 64, 64
        stages = 1 if fp8 else 2
    target = 96
    if fp8 and qcap > 32:
        target = 192
    cfg = dict(
        bm=bm,
        bn=bn,
        bk=bk,
        splits=1,
        warps=4,
        stages=stages,
    )
    if wide_gqa:
        prefix = max_seq_len >= qcap + 256
        if fp8:
            cfg.update(
                {
                    "bm": 32 if qcap <= 4 or (qcap <= 8 and hk == 1) else 64,
                    "bn": 64,
                    "warps": 8,
                    "stages": 1,
                    "waves_per_eu": 2,
                    "prefix_fast": prefix,
                }
            )
            if not prefix and hk > 4 and qcap >= 512:
                cfg.update({"pv_split": True, "waves_per_eu": 6})
            target = 128 if qcap > 32 and hk > 1 else 64
        else:
            wide = hk >= 2 and (16 <= qcap <= 32 or batch * qcap >= 1024)
            cfg.update(
                {
                    "bm": 64 if wide else 32,
                    "bn": 32 if wide else 64,
                    "warps": 8,
                    "stages": 1 if wide else 2,
                    "waves_per_eu": 6 if wide else 4,
                }
            )
            if wide:
                cfg.update({"pv_split": True, "qk_pipeline": 3})
            elif qcap >= 256 and prefix:
                cfg.update({"bk": 128, "prefix_fast": hk == 1})
            if max_seq_len < 32768:
                if qcap <= 8:
                    cfg["qk_unroll"] = 2 if hk == 2 else 4
                elif (
                    wide
                    and (qcap <= 32 or prefix)
                    or hk == 1
                    and qcap >= 256
                    and prefix
                ):
                    cfg["qk_unroll"] = 2
            target = 64
    groups = kv_groups * triton.cdiv(qcap * gqa, cfg["bm"])
    splits = min(32, triton.next_power_of_2(triton.cdiv(target, groups)))
    min_span = cfg["bn"] * 2 if wide_gqa else 128
    while splits > 1 and max_seq_len < splits * min_span:
        splits //= 2
    cfg["splits"] = splits
    if dim == 64 and qcap >= 64 and qcap < max_seq_len <= qcap + 256:
        cfg.update(
            {
                "bm": 64 if fp8 or qcap <= 256 else 128,
                "bn": 32,
                "bk": 64,
                "splits": 1,
                "warps": 4,
                "stages": 1,
            }
        )
    elif dim == 64 and fp8 and max_seq_len <= 256:
        cfg.update({"bn": 32, "splits": 1})
    elif dim == 64 and qcap == 8 and max_seq_len >= 32768:
        groups = batch * hk * triton.cdiv(qcap * (hq // hk), 32)
        cfg.update(
            {
                "bm": 32,
                "bn": 32 if fp8 else 64,
                "splits": min(64, triton.next_power_of_2(triton.cdiv(256, groups))),
            }
        )
    elif dim == 64 and fp8 and qcap == 8 and max_seq_len >= 4096:
        groups = batch * hk * triton.cdiv(qcap * (hq // hk), 32)
        cfg.update(
            {
                "bm": 32,
                "bn": 64,
                "splits": min(32, triton.next_power_of_2(triton.cdiv(128, groups))),
            }
        )
    elif dim == 64 and fp8 and qcap >= 64 and max_seq_len >= 4096:
        groups = batch * hk * triton.cdiv(qcap * (hq // hk), 64)
        cfg.update(
            {
                "bm": 64,
                "bn": 32,
                "splits": min(8, triton.next_power_of_2(triton.cdiv(1024, groups))),
                "stages": 1,
            }
        )
    elif dim == 128 and fp8 and qcap <= 256 and max_seq_len <= qcap:
        cfg.update({"splits": 1})
    elif (
        dim == 128
        and fp8
        and qcap >= 256
        and max_seq_len >= 8192
        and (
            hq // hk >= 4
            or (max_seq_len >= 32768 and (hq // hk >= 2 or qcap >= 512))
            or max_seq_len >= 131072
        )
    ):
        # Shorter prefixes need sufficient grouped-query reuse; at GQA1/Q256
        # the wider tile pays off only once the prefix reaches 131K tokens.
        cfg.update(
            {
                "bm": 64,
                "bn": 128,
                "bk": 64,
                "warps": 8,
                "stages": 3,
                "waves_per_eu": 6,
                "prefix_fast": True,
            }
        )
        if qcap >= 512 or max_seq_len >= 32768:
            cfg["qk_pipeline"] = 3
        cfg["splits"] = _long_extend_splits(
            cfg["splits"], batch, qcap, max_seq_len, hq, hk, dim, cfg["bm"]
        )
    elif dim == 256 and fp8 and qcap >= 256 and max_seq_len >= 8192:
        # Wider tiles amortize each FP8 KV load over twice as many queries and
        # keys. Six waves is the best gfx1201 pressure/occupancy tradeoff.
        cfg.update(
            {"bm": 64, "bn": 128, "bk": 64, "warps": 8, "stages": 1, "waves_per_eu": 6}
        )
        cfg["splits"] = _long_extend_splits(
            cfg["splits"], batch, qcap, max_seq_len, hq, hk, dim, cfg["bm"]
        )
    elif dim == 256 and not fp8 and qcap >= 1024 and max_seq_len >= 8192:
        # Two half-width PV accumulators avoid spilling this BF16 D256 tile.
        # Keep Q256 on the incumbent: the wider tile loses at that capacity.
        cfg.update(
            {
                "bm": 64,
                "bn": 64,
                "bk": 64,
                "warps": 8,
                "stages": 3,
                "waves_per_eu": 6,
                "pv_split": True,
                "qk_pipeline": 3,
            }
        )
    elif dim == 128 and not fp8 and qcap > 128:
        if max_seq_len >= 8192:
            cfg.update(
                {
                    "bm": 128,
                    "bn": 32,
                    "bk": 128,
                    "warps": 8,
                    "stages": 1,
                    "waves_per_eu": 6,
                }
            )
            groups = batch * hk * triton.cdiv(qcap * (hq // hk), cfg["bm"])
            cfg["splits"] = min(16, triton.next_power_of_2(triton.cdiv(192, groups)))
        elif qcap >= 4096:
            cfg.update(
                {
                    "bm": 128,
                    "bn": 32,
                    "bk": 128,
                    "splits": 1,
                    "warps": 8,
                    "stages": 1,
                    "waves_per_eu": 6,
                }
            )
        elif qcap > 256 or hq > 10:
            cfg.update(
                {
                    "bm": 64,
                    "bn": 32,
                    "bk": 128,
                    "splits": 1,
                    "warps": 4,
                    "stages": 1,
                    "waves_per_eu": 6,
                }
            )
        else:
            cfg.update(
                {"bm": 32, "bn": 64, "bk": 64, "splits": 1, "warps": 4, "stages": 2}
            )
    elif wide_gqa and max_seq_len >= 32768:
        if not fp8:
            cfg.update({"bm": 32, "bn": 64, "bk": 128, "stages": 2, "waves_per_eu": 4})
            cfg.pop("pv_split", None)
            cfg.pop("qk_pipeline", None)
        cfg["splits"] = _long_extend_splits(
            cfg["splits"], batch, qcap, max_seq_len, hq, hk, dim, cfg["bm"], fp8=fp8
        )
    if dim == 512 and not wide_gqa:
        cfg.update({"bn": 64 if fp8 and qcap > 32 else 32, "stages": 1})
    return cfg
