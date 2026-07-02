#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for minimal envs.
    tqdm = None


KERNEL_DIR = Path(__file__).resolve().parent
VLLM_REPO_ROOT = KERNEL_DIR.parents[2]
if str(VLLM_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_REPO_ROOT))
RUNNER = KERNEL_DIR / "benchmark_runner.mojo"
DEFAULT_MOJO = "/opt/python/bin/mojo"
DEFAULT_BUILD_CACHE_DIR = KERNEL_DIR / ".build_cache"
DEFAULT_M_BUCKETS = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
]
DEFAULT_SMALL_M_BUCKETS = [1, 2, 4, 8, 16, 32, 64]
DEFAULT_POLICY_CONFIG_DIR = KERNEL_DIR / "policies"
DEFAULT_POLICY_OP_NAME = "awq_gemm"
MOJO_CACHE_SOURCES = (
    KERNEL_DIR / "benchmark_runner.mojo",
    KERNEL_DIR / "mojo" / "__init__.mojo",
    KERNEL_DIR / "mojo" / "common.mojo",
    KERNEL_DIR / "mojo" / "config_defaults.mojo",
    KERNEL_DIR / "mojo" / "kernel_common.mojo",
    KERNEL_DIR / "mojo" / "ring_buffer.mojo",
    KERNEL_DIR / "mojo" / "splitk_reduce.mojo",
    KERNEL_DIR / "mojo" / "kernels" / "__init__.mojo",
    KERNEL_DIR / "mojo" / "kernels" / "b_staged_sync.mojo",
    KERNEL_DIR / "mojo" / "kernels" / "kpacked_dot2_splitk.mojo",
    KERNEL_DIR / "mojo" / "kernels" / "kpacked_wmma16_splitk.mojo",
    KERNEL_DIR / "mojo" / "kernels" / "ring_ab_staged.mojo",
    KERNEL_DIR / "mojo" / "kernels" / "ring_b_staged.mojo",
)


class FlowList(list):
    pass


class FlowDict(dict):
    pass


@dataclass(frozen=True)
class ShapeFamily:
    n: int
    k: int
    group_size: int
    op_name: str = DEFAULT_POLICY_OP_NAME


class PolicyDumper(yaml.SafeDumper):
    pass


def _flow_list_representer(dumper: yaml.SafeDumper, data: FlowList):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def _flow_dict_representer(dumper: yaml.SafeDumper, data: FlowDict):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)


PolicyDumper.add_representer(FlowList, _flow_list_representer)
PolicyDumper.add_representer(FlowDict, _flow_dict_representer)


def with_flow_lists(value: Any) -> Any:
    if isinstance(value, list):
        return FlowList(with_flow_lists(item) for item in value)
    if isinstance(value, dict):
        return {key: with_flow_lists(item) for key, item in value.items()}
    return value


def dump_policy_yaml(payload: dict[str, Any]) -> str:
    styled_payload = {
        key: (
            [FlowDict(with_flow_lists(item)) for item in value]
            if key == "configs"
            else with_flow_lists(value)
        )
        for key, value in payload.items()
    }
    return yaml.dump(
        styled_payload,
        Dumper=PolicyDumper,
        sort_keys=False,
        width=1000,
    )


VARIANTS = {
    "ring",
    "kpacked_dot2",
    "wmma16",
    "ring_bonly",
    "ring_bonly_sync",
}

VARIANT_DESCRIPTIONS = {
    "ring": "full ring-buffer A+B LDS staging",
    "kpacked_dot2": "K-packed split-K decode with fdot2 reduction",
    "wmma16": "split-K WMMA16 decode path",
    "ring_bonly": "B-only ring-buffer WMMA path",
    "ring_bonly_sync": "single-stage synchronized B-only WMMA path",
}


def variant_flags(variant: str) -> dict[str, bool]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown W4A16 kernel variant: {variant}")
    return {
        "use_decode_kernel": False,
        "use_splitk_decode_kernel": False,
        "use_kpacked_decode_kernel": variant == "kpacked_dot2",
        "use_kpacked_dot2": variant == "kpacked_dot2",
        "use_wmma16_kernel": variant == "wmma16",
        "use_ring_bonly_kernel": variant == "ring_bonly",
        "use_ring_bonly_sync_kernel": variant == "ring_bonly_sync",
    }


@dataclass(frozen=True)
class RunConfig:
    m: int
    n: int
    k: int
    bm: int
    bn: int
    bk: int
    group_size: int = 128
    warps_m: int = 1
    warps_n: int = 2
    ring_producer_warps: int = 1
    ring_stages: int = 2
    block_swizzle_scale: int = 0
    group_size_m: int = 0
    use_lds_swizzle: bool = False
    ring_startup_all_warps: bool = True
    load_b_by_qpack: bool = True
    qpack_k_vector_width: int = 2
    dequant_b_in_bf16: bool = False
    scale_after_group: bool = True
    assume_even_n: bool = False
    use_decode_kernel: bool = False
    use_splitk_decode_kernel: bool = False
    use_kpacked_decode_kernel: bool = False
    use_kpacked_dot2: bool = False
    use_wmma16_kernel: bool = False
    use_ring_bonly_kernel: bool = False
    use_ring_bonly_sync_kernel: bool = False
    decode_threads: int = 64
    decode_block_k: int = 256
    decode_m_rows: int = 1
    smem_pad: int = 8
    warmup_iters: int = 20
    timed_iters: int = 100
    verify: bool = False

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.m, self.n, self.k

    @property
    def variant(self) -> str:
        if self.use_kpacked_decode_kernel and self.use_kpacked_dot2:
            return "kpacked_dot2"
        if self.use_wmma16_kernel:
            return "wmma16"
        if self.use_ring_bonly_kernel:
            return "ring_bonly"
        if self.use_ring_bonly_sync_kernel:
            return "ring_bonly_sync"
        if not (
            self.use_decode_kernel
            or self.use_splitk_decode_kernel
            or self.use_kpacked_decode_kernel
            or self.use_kpacked_dot2
            or self.use_wmma16_kernel
            or self.use_ring_bonly_kernel
            or self.use_ring_bonly_sync_kernel
        ):
            return "ring"
        raise ValueError(f"no compact variant name for {format_config(self)}")

    @classmethod
    def from_policy_dict(
        cls,
        data: dict[str, Any],
        defaults: dict[str, Any],
        verify: bool | None = None,
    ) -> RunConfig:
        def get(name: str, default: Any = None) -> Any:
            return data.get(name, defaults.get(name, default))

        tile = get("tile")
        warps = get("warps")
        decode = get("decode", defaults.get("decode", [64, 256, 1]))
        variant = str(get("variant", "ring"))
        return cls(
            m=int(get("m")),
            n=int(get("n")),
            k=int(get("k")),
            bm=int(tile[0]),
            bn=int(tile[1]),
            bk=int(tile[2]),
            group_size=int(get("group", 128)),
            warps_m=int(warps[0]),
            warps_n=int(warps[1]),
            ring_producer_warps=int(get("prod", 1)),
            ring_stages=int(get("stages", 2)),
            block_swizzle_scale=int(get("swizzle", 0)),
            group_size_m=int(get("group_m", 0)),
            use_lds_swizzle=bool(get("lds_swizzle", False)),
            ring_startup_all_warps=bool(get("startup_all", True)),
            load_b_by_qpack=bool(get("qpack_load", True)),
            qpack_k_vector_width=int(get("qv", 2)),
            dequant_b_in_bf16=bool(get("bf16_deq", False)),
            scale_after_group=bool(get("scale_after_group", True)),
            assume_even_n=bool(get("even_n", False)),
            **variant_flags(variant),
            decode_threads=int(decode[0]),
            decode_block_k=int(decode[1]),
            decode_m_rows=int(decode[2]),
            smem_pad=int(get("pad", 8)),
            warmup_iters=int(get("warmup_iters", 20)),
            timed_iters=int(get("timed_iters", 100)),
            verify=bool(get("verify", False) if verify is None else verify),
        )

    def to_policy_dict(
        self, mean_ms: float, defaults: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = {
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "group": self.group_size,
            "variant": self.variant,
            "tile": [self.bm, self.bn, self.bk],
            "warps": [self.warps_m, self.warps_n],
            "prod": self.ring_producer_warps,
            "stages": self.ring_stages,
            "swizzle": self.block_swizzle_scale,
            "group_m": self.group_size_m,
            "lds_swizzle": self.use_lds_swizzle,
            "startup_all": self.ring_startup_all_warps,
            "qpack_load": self.load_b_by_qpack,
            "qv": self.qpack_k_vector_width,
            "bf16_deq": self.dequant_b_in_bf16,
            "scale_after_group": self.scale_after_group,
            "even_n": self.assume_even_n,
            "decode": [
                self.decode_threads,
                self.decode_block_k,
                self.decode_m_rows,
            ],
            "pad": self.smem_pad,
            "warmup_iters": self.warmup_iters,
            "timed_iters": self.timed_iters,
            "mean_ms": mean_ms,
        }
        defaults = defaults or {}
        return {
            key: value
            for key, value in data.items()
            if key in ("m", "variant", "tile", "warps", "mean_ms")
            or defaults.get(key) != value
        }

    def to_runtime_policy_dict(self, mean_ms: float) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "tile_M": self.bm,
            "tile_N": self.bn,
            "tile_K": self.bk,
            "warps_M": self.warps_m,
            "warps_N": self.warps_n,
            "producer_warps": self.ring_producer_warps,
            "ring_stages": self.ring_stages,
            "block_swizzle_scale": self.block_swizzle_scale,
            "group_size_M": self.group_size_m,
            "use_lds_swizzle": self.use_lds_swizzle,
            "ring_startup_all_warps": self.ring_startup_all_warps,
            "load_B_by_qpack": self.load_b_by_qpack,
            "qpack_K_vector_width": self.qpack_k_vector_width,
            "dequant_B_in_bf16": self.dequant_b_in_bf16,
            "scale_after_group": self.scale_after_group,
            "assume_even_N": self.assume_even_n,
            "splitk_threads": self.decode_threads,
            "splitk_block_K": self.decode_block_k,
            "splitk_rows_per_CTA": self.decode_m_rows,
            "smem_pad": self.smem_pad,
            "mean_ms": mean_ms,
        }


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sweep_values(value: Any) -> list[Any]:
    if value is None:
        return [None]
    return value if isinstance(value, list) else [value]


def parse_int_csv(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item) for item in value.replace("x", ",").split(",") if item]


def parse_shape_family(value: str) -> ShapeFamily:
    parts = parse_int_csv(value)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"shape family must be N,K,G or NxKxG, got {value!r}"
        )
    return ShapeFamily(n=parts[0], k=parts[1], group_size=parts[2])


def _text_config(config: Any) -> Any:
    if hasattr(config, "get_text_config"):
        return config.get_text_config()
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config
    return config


def _config_int(config: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return None


def _quantization_group_size(config: Any) -> int | None:
    quantization_config = getattr(config, "quantization_config", None)
    if not isinstance(quantization_config, dict):
        return None
    group_size = quantization_config.get("group_size")
    if group_size is not None:
        return int(group_size)
    config_groups = quantization_config.get("config_groups", {})
    if isinstance(config_groups, dict):
        for group_cfg in config_groups.values():
            if not isinstance(group_cfg, dict):
                continue
            weights = group_cfg.get("weights", {})
            if isinstance(weights, dict) and weights.get("group_size") is not None:
                return int(weights["group_size"])
    return None


def _add_family(
    families: set[ShapeFamily],
    *,
    n: int | None,
    k: int | None,
    group_size: int,
    tp_size: int,
    partition_n: bool,
    partition_k: bool,
) -> None:
    if n is None or k is None:
        return
    if partition_n:
        if n % tp_size:
            raise ValueError(f"N={n} is not divisible by tp_size={tp_size}")
        n //= tp_size
    if partition_k:
        if k % tp_size:
            raise ValueError(f"K={k} is not divisible by tp_size={tp_size}")
        k //= tp_size
    if n <= 0 or k <= 0:
        return
    if k % group_size:
        raise ValueError(
            f"derived shape N={n}, K={k} is not divisible by group_size={group_size}"
        )
    families.add(ShapeFamily(n=n, k=k, group_size=group_size))


def discover_model_shape_families(
    model: str,
    *,
    tp_size: int,
    trust_remote_code: bool,
    explicit_group_size: int | None,
) -> list[ShapeFamily]:
    from vllm.transformers_utils.config import get_config

    config = get_config(model=model, trust_remote_code=trust_remote_code)
    config = _text_config(config)
    hidden_size = _config_int(config, "hidden_size", "n_embd")
    if hidden_size is None:
        raise ValueError(f"could not determine hidden_size for model {model}")
    intermediate_size = _config_int(
        config,
        "intermediate_size",
        "ffn_hidden_size",
        "moe_intermediate_size",
    )
    group_size = explicit_group_size or _quantization_group_size(config) or 128
    if group_size <= 0:
        group_size = hidden_size

    families: set[ShapeFamily] = set()

    num_heads = _config_int(config, "num_attention_heads", "n_head")
    num_kv_heads = _config_int(config, "num_key_value_heads", "num_kv_heads")
    head_dim = _config_int(config, "head_dim", "kv_channels")
    if num_heads is not None:
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if head_dim is None:
            head_dim = hidden_size // num_heads
        qkv_out = (num_heads + 2 * num_kv_heads) * head_dim
        _add_family(
            families,
            n=qkv_out,
            k=hidden_size,
            group_size=group_size,
            tp_size=tp_size,
            partition_n=True,
            partition_k=False,
        )
        _add_family(
            families,
            n=hidden_size,
            k=num_heads * head_dim,
            group_size=group_size,
            tp_size=tp_size,
            partition_n=False,
            partition_k=True,
        )

    if intermediate_size is not None:
        _add_family(
            families,
            n=2 * intermediate_size,
            k=hidden_size,
            group_size=group_size,
            tp_size=tp_size,
            partition_n=True,
            partition_k=False,
        )
        _add_family(
            families,
            n=hidden_size,
            k=intermediate_size,
            group_size=group_size,
            tp_size=tp_size,
            partition_n=False,
            partition_k=True,
        )

    return sorted(families, key=lambda item: (item.n, item.k, item.group_size))


def format_config(cfg: RunConfig) -> str:
    return (
        f"shape={cfg.m}x{cfg.n}x{cfg.k} "
        f"config=({cfg.variant}, {cfg.bm}, {cfg.bn}, {cfg.bk}, G={cfg.group_size}, "
        f"warps={cfg.warps_m}x{cfg.warps_n}, prod={cfg.ring_producer_warps}, "
        f"stages={cfg.ring_stages}, lds_swizzle={int(cfg.use_lds_swizzle)}, "
        f"startup_all={int(cfg.ring_startup_all_warps)}, "
        f"qpack={int(cfg.load_b_by_qpack)}, kvec={cfg.qpack_k_vector_width}, "
        f"bf16_deq={int(cfg.dequant_b_in_bf16)}, "
        f"scale_group={int(cfg.scale_after_group)}, "
        f"even_n={int(cfg.assume_even_n)}, "
        f"decode={int(cfg.use_decode_kernel)}, "
        f"splitk_decode={int(cfg.use_splitk_decode_kernel)}, "
        f"kpacked_decode={int(cfg.use_kpacked_decode_kernel)}, "
        f"kpacked_dot2={int(cfg.use_kpacked_dot2)}, "
        f"wmma16={int(cfg.use_wmma16_kernel)}, "
        f"ring_bonly={int(cfg.use_ring_bonly_kernel)}, "
        f"ring_bonly_sync={int(cfg.use_ring_bonly_sync_kernel)}, "
        f"decode_threads={cfg.decode_threads}, "
        f"decode_block_k={cfg.decode_block_k}, "
        f"decode_m_rows={cfg.decode_m_rows}, pad={cfg.smem_pad})"
    )


def invalid_reason(cfg: RunConfig) -> str | None:
    if cfg.n % 8:
        return "N must be divisible by 8"
    if cfg.k % cfg.group_size:
        return "K must be divisible by GROUP_SIZE"
    if cfg.bk > cfg.group_size:
        return "BK must be <= GROUP_SIZE"
    if cfg.group_size % cfg.bk:
        return "GROUP_SIZE must be divisible by BK"
    if cfg.bk % 16:
        return "BK must be divisible by 16"
    if cfg.bm % (cfg.warps_m * 16):
        return "BM must be divisible by WARPS_M * 16"
    if cfg.bn % (cfg.warps_n * 16):
        return "BN must be divisible by WARPS_N * 16"
    if cfg.ring_producer_warps < 1:
        return "RING_PRODUCER_WARPS must be positive"
    if cfg.ring_stages < 1:
        return "NUM_STAGES must be positive"
    max_qpack_k_vector_width = 8 if cfg.use_lds_swizzle else 16
    if (
        cfg.qpack_k_vector_width < 1
        or cfg.qpack_k_vector_width > max_qpack_k_vector_width
    ):
        return f"QPACK_K_VECTOR_WIDTH must be between 1 and {max_qpack_k_vector_width}"
    if cfg.bk % cfg.qpack_k_vector_width:
        return "BK must be divisible by QPACK_K_VECTOR_WIDTH"
    if cfg.decode_threads < 1:
        return "DECODE_THREADS must be positive"
    if cfg.decode_block_k < 1:
        return "DECODE_BLOCK_K must be positive"
    if cfg.decode_block_k % 2:
        return "DECODE_BLOCK_K must be even"
    if cfg.decode_m_rows < 1:
        return "DECODE_M_ROWS must be positive"
    decode_modes = (
        int(cfg.use_decode_kernel)
        + int(cfg.use_splitk_decode_kernel)
        + int(cfg.use_kpacked_decode_kernel)
        + int(cfg.use_wmma16_kernel)
        + int(cfg.use_ring_bonly_kernel)
        + int(cfg.use_ring_bonly_sync_kernel)
    )
    if decode_modes > 1:
        return "choose only one decode kernel mode"
    if cfg.use_decode_kernel or cfg.use_splitk_decode_kernel:
        return "legacy decode variants are not available in the packaged runner"
    if cfg.use_kpacked_decode_kernel and not cfg.use_kpacked_dot2:
        return "only the kpacked_dot2 split-K decode variant is available"
    if cfg.use_kpacked_decode_kernel and cfg.k % 8:
        return "K must be divisible by 8 for k-packed decode"
    if cfg.use_kpacked_decode_kernel and cfg.decode_block_k % 8:
        return "DECODE_BLOCK_K must be divisible by 8 for k-packed decode"
    if cfg.use_kpacked_dot2 and not cfg.use_kpacked_decode_kernel:
        return "USE_KPACKED_DOT2 requires USE_KPACKED_DECODE_KERNEL"
    if cfg.use_wmma16_kernel and cfg.k % 16:
        return "K must be divisible by 16 for WMMA16"
    if cfg.use_wmma16_kernel and cfg.n % 16:
        return "N must be divisible by 16 for WMMA16"
    if cfg.use_wmma16_kernel and cfg.decode_block_k % 16:
        return "DECODE_BLOCK_K must be divisible by 16 for WMMA16"
    if cfg.m < 1 or cfg.n < 1 or cfg.k < 1:
        return "M, N, and K must be positive"
    return None


def _bool_define(value: bool) -> str:
    return "True" if value else "False"


def compile_defines(cfg: RunConfig) -> dict[str, str]:
    return {
        "MAX_M": str(cfg.m),
        "MAX_N": str(cfg.n),
        "MAX_K": str(cfg.k),
        "BM": str(cfg.bm),
        "BN": str(cfg.bn),
        "BK": str(cfg.bk),
        "GROUP_SIZE": str(cfg.group_size),
        "WARPS_M": str(cfg.warps_m),
        "WARPS_N": str(cfg.warps_n),
        "RING_PRODUCER_WARPS": str(cfg.ring_producer_warps),
        "NUM_STAGES": str(cfg.ring_stages),
        "BLOCK_SWIZZLE_SCALE": str(cfg.block_swizzle_scale),
        "GROUP_SIZE_M": str(cfg.group_size_m),
        "USE_LDS_SWIZZLE": _bool_define(cfg.use_lds_swizzle),
        "RING_STARTUP_ALL_WARPS": _bool_define(cfg.ring_startup_all_warps),
        "LOAD_B_BY_QPACK": _bool_define(cfg.load_b_by_qpack),
        "QPACK_K_VECTOR_WIDTH": str(cfg.qpack_k_vector_width),
        "DEQUANT_B_IN_BF16": _bool_define(cfg.dequant_b_in_bf16),
        "SCALE_AFTER_GROUP": _bool_define(cfg.scale_after_group),
        "USE_DECODE_KERNEL": _bool_define(cfg.use_decode_kernel),
        "USE_SPLITK_DECODE_KERNEL": _bool_define(cfg.use_splitk_decode_kernel),
        "USE_KPACKED_DECODE_KERNEL": _bool_define(cfg.use_kpacked_decode_kernel),
        "USE_KPACKED_DOT2": _bool_define(cfg.use_kpacked_dot2),
        "USE_WMMA16_KERNEL": _bool_define(cfg.use_wmma16_kernel),
        "USE_RING_BONLY_KERNEL": _bool_define(cfg.use_ring_bonly_kernel),
        "USE_RING_BONLY_SYNC_KERNEL": _bool_define(cfg.use_ring_bonly_sync_kernel),
        "DECODE_THREADS": str(cfg.decode_threads),
        "DECODE_BLOCK_K": str(cfg.decode_block_k),
        "DECODE_M_ROWS": str(cfg.decode_m_rows),
        "SPLITK_THREADS": str(cfg.decode_threads),
        "SPLITK_BLOCK_K": str(cfg.decode_block_k),
        "SPLITK_ROWS_PER_CTA": str(cfg.decode_m_rows),
        "USE_BENCH_PARTIAL": _bool_define(
            cfg.use_kpacked_decode_kernel or cfg.use_wmma16_kernel
        ),
        "SMEM_PAD": str(cfg.smem_pad),
        "ASSUME_EVEN_K": _bool_define(cfg.k % cfg.bk == 0),
        "ASSUME_EVEN_N": _bool_define(cfg.assume_even_n),
        "ASSUME_EVEN_MN": _bool_define(cfg.m % cfg.bm == 0 and cfg.n % cfg.bn == 0),
        "WARMUP_ITERS": str(cfg.warmup_iters),
        "BENCH_ITERS": str(cfg.timed_iters),
        "VERIFY": _bool_define(cfg.verify),
    }


def define_args(cfg: RunConfig) -> list[str]:
    args: list[str] = []
    for key, value in compile_defines(cfg).items():
        args.extend(["-D", f"{key}={value}"])
    return args


def build_command(
    cfg: RunConfig, mojo_bin: str, target_accelerator: str | None
) -> list[str]:
    cmd = [mojo_bin, "run"]
    if target_accelerator:
        cmd.extend(["--target-accelerator", target_accelerator])
    cmd.extend(define_args(cfg))
    cmd.append(str(RUNNER))
    return cmd


def build_binary_command(
    cfg: RunConfig,
    mojo_bin: str,
    target_accelerator: str | None,
    output_path: Path,
) -> list[str]:
    cmd = [mojo_bin, "build"]
    if target_accelerator:
        cmd.extend(["--target-accelerator", target_accelerator])
    cmd.extend(define_args(cfg))
    cmd.extend(["-o", str(output_path), str(RUNNER)])
    return cmd


def source_digest() -> str:
    digest = hashlib.sha256()
    for path in MOJO_CACHE_SOURCES:
        digest.update(str(path.relative_to(KERNEL_DIR)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def cache_key(
    cfg: RunConfig,
    mojo_bin: str,
    target_accelerator: str | None,
    src_digest: str | None = None,
) -> str:
    payload = {
        "source_digest": src_digest or source_digest(),
        "mojo_bin": str(Path(mojo_bin).resolve())
        if Path(mojo_bin).exists()
        else mojo_bin,
        "target_accelerator": target_accelerator,
        "defines": compile_defines(cfg),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def cached_binary_path(
    cfg: RunConfig,
    mojo_bin: str,
    target_accelerator: str | None,
    cache_dir: Path,
    src_digest: str | None = None,
) -> Path:
    key = cache_key(cfg, mojo_bin, target_accelerator, src_digest)
    return cache_dir / key[:2] / key / "benchmark_runner"


def build_cached_binary(
    cfg: RunConfig,
    mojo_bin: str,
    target_accelerator: str | None,
    cache_dir: Path,
    timeout_seconds: float,
    src_digest: str | None = None,
    force_rebuild: bool = False,
) -> Path:
    reason = invalid_reason(cfg)
    if reason is not None:
        raise ValueError(f"invalid config: {reason}: {format_config(cfg)}")

    binary = cached_binary_path(
        cfg, mojo_bin, target_accelerator, cache_dir, src_digest
    )
    if binary.exists() and not force_rebuild:
        return binary

    binary.parent.mkdir(parents=True, exist_ok=True)
    tmp_binary = binary.with_name(f"{binary.name}.tmp.{os.getpid()}")
    if tmp_binary.exists():
        tmp_binary.unlink()
    proc = subprocess.run(
        build_binary_command(cfg, mojo_bin, target_accelerator, tmp_binary),
        cwd=KERNEL_DIR,
        text=True,
        capture_output=True,
        timeout=timeout_seconds if timeout_seconds > 0 else None,
        check=False,
    )
    if proc.returncode != 0:
        if tmp_binary.exists():
            tmp_binary.unlink()
        raise RuntimeError(
            f"build failed for {format_config(cfg)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    os.replace(tmp_binary, binary)
    binary.chmod(0o755)
    return binary


def progress_bar(iterable, *, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def parse_mean_ms(output: str) -> float:
    for line in output.splitlines():
        if line.startswith("TUNE_RESULT="):
            return float(line.split("=", 1)[1].strip())
    raise RuntimeError(f"missing TUNE_RESULT in runner output:\n{output}")


def run_single_config(
    cfg: RunConfig,
    mojo_bin: str,
    target_accelerator: str | None,
    timeout_seconds: float,
    *,
    use_build_cache: bool = False,
    build_cache_dir: Path = DEFAULT_BUILD_CACHE_DIR,
    force_rebuild: bool = False,
    src_digest: str | None = None,
    require_cached_binary: bool = False,
) -> float:
    reason = invalid_reason(cfg)
    if reason is not None:
        raise ValueError(f"invalid config: {reason}: {format_config(cfg)}")
    if use_build_cache:
        binary = cached_binary_path(
            cfg, mojo_bin, target_accelerator, build_cache_dir, src_digest
        )
        if require_cached_binary and not binary.exists():
            raise RuntimeError(
                f"missing cached binary for {format_config(cfg)}: {binary}"
            )
        if not binary.exists() or force_rebuild:
            binary = build_cached_binary(
                cfg,
                mojo_bin,
                target_accelerator,
                build_cache_dir,
                timeout_seconds,
                src_digest=src_digest,
                force_rebuild=force_rebuild,
            )
        cmd = [str(binary)]
    else:
        cmd = build_command(cfg, mojo_bin, target_accelerator)
    proc = subprocess.run(
        cmd,
        cwd=KERNEL_DIR,
        text=True,
        capture_output=True,
        timeout=timeout_seconds if timeout_seconds > 0 else None,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"runner failed for {format_config(cfg)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    if cfg.verify:
        for line in proc.stdout.splitlines():
            if line and not line.startswith("TUNE_RESULT="):
                print(f"  {line}")
    return parse_mean_ms(proc.stdout)


def load_tuned_configs(path: Path, verify: bool) -> list[RunConfig]:
    payload = yaml.safe_load(path.read_text()) or {}
    defaults = payload.get("defaults", {})
    return [
        RunConfig.from_policy_dict(item, defaults, verify=verify)
        for item in payload.get("configs", [])
    ]


def load_seed_configs(
    path: Path,
) -> dict[tuple[int, int, int], tuple[RunConfig, float]]:
    payload = yaml.safe_load(path.read_text()) or {}
    defaults = payload.get("defaults", {})
    seeded: dict[tuple[int, int, int], tuple[RunConfig, float]] = {}
    for item in payload.get("configs", []):
        cfg = RunConfig.from_policy_dict(item, defaults, verify=False)
        seeded[cfg.shape] = (cfg, float(item.get("mean_ms", float("inf"))))
    return seeded


def select_nearest_tuned_config(
    configs: list[RunConfig],
    m: int,
    n: int,
    k: int,
    group_size: int | None = None,
) -> RunConfig | None:
    candidates = [
        cfg
        for cfg in configs
        if cfg.n == n
        and cfg.k == k
        and (group_size is None or cfg.group_size == group_size)
    ]
    if not candidates:
        return None
    selected = min(candidates, key=lambda cfg: (abs(cfg.m - m), cfg.m > m, cfg.m))
    return replace(selected, m=m, n=n, k=k)


def generic_safe_config(
    m: int,
    n: int,
    k: int,
    group_size: int,
    warmup_iters: int = 20,
    timed_iters: int = 100,
    verify: bool = False,
) -> RunConfig:
    """Conservative W4A16 baseline for untuned shapes.

    This is intended to be correct and broadly launchable, not optimal. It uses
    the B-only WMMA path because that path is the least shape-specialized of the
    current winner set and is also supported by the Python CustomOpLibrary
    experiment.
    """
    if group_size == -1:
        group_size = k
    if group_size < 16 or k % group_size:
        raise ValueError(f"unsupported generic W4A16 group_size={group_size} for K={k}")

    if group_size % 64 == 0:
        bk = 64
        qv = 8
    elif group_size % 32 == 0:
        bk = 32
        qv = 4
    elif group_size % 16 == 0:
        bk = 16
        qv = 2
    else:
        raise ValueError(
            f"generic W4A16 requires group_size divisible by 16; got {group_size}"
        )

    return RunConfig(
        m=m,
        n=n,
        k=k,
        bm=128,
        bn=64,
        bk=bk,
        group_size=group_size,
        warps_m=4,
        warps_n=1,
        ring_producer_warps=2,
        ring_stages=3,
        ring_startup_all_warps=True,
        qpack_k_vector_width=qv,
        scale_after_group=False,
        assume_even_n=n % 64 == 0,
        use_ring_bonly_kernel=True,
        decode_threads=64,
        decode_block_k=256,
        decode_m_rows=1,
        smem_pad=8,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        verify=verify,
    )


def qv_values_for_group(group_size: int) -> list[int]:
    if group_size >= 128:
        return [4, 8]
    if group_size >= 64:
        return [4]
    return [2, 4]


def bk_values_for_group(group_size: int) -> list[int]:
    if group_size >= 64:
        return [64]
    if group_size >= 32:
        return [32]
    return [16]


def make_tune_jobs_from_families(
    families: list[ShapeFamily],
    *,
    m_buckets: list[int],
    small_m_buckets: list[int],
    warmup_iters: int,
    timed_iters: int,
    verify: bool,
) -> list[RunConfig]:
    jobs: list[RunConfig] = []
    shapes_by_group: dict[int, list[tuple[int, int, int]]] = {}
    small_shapes_by_group: dict[int, list[tuple[int, int, int]]] = {}
    for family in families:
        shapes_by_group.setdefault(family.group_size, [])
        small_shapes_by_group.setdefault(family.group_size, [])
        shapes_by_group[family.group_size].extend(
            (m, family.n, family.k) for m in m_buckets
        )
        small_shapes_by_group[family.group_size].extend(
            (m, family.n, family.k) for m in small_m_buckets
        )

    def add_jobs(
        *,
        group_size: int,
        shapes: list[tuple[int, int, int]],
        variant: str,
        bm_values: list[int],
        bn_values: list[int],
        bk_values: list[int],
        warps_m_values: list[int],
        warps_n_values: list[int],
        producer_values: list[int],
        stage_values: list[int],
        qv_values: list[int],
        scale_after_group_values: list[bool],
        startup_all_values: list[bool],
        decode_threads_values: list[int],
        decode_block_k_values: list[int],
        decode_rows_values: list[int],
    ) -> None:
        flags = variant_flags(variant)
        for (
            (m, n, k),
            bm,
            bn,
            bk,
            warps_m,
            warps_n,
            prod,
            stages,
            qv,
            scale,
            startup,
            dthreads,
            dbk,
            drows,
        ) in itertools.product(
            shapes,
            bm_values,
            bn_values,
            bk_values,
            warps_m_values,
            warps_n_values,
            producer_values,
            stage_values,
            qv_values,
            scale_after_group_values,
            startup_all_values,
            decode_threads_values,
            decode_block_k_values,
            decode_rows_values,
        ):
            jobs.append(
                RunConfig(
                    m=m,
                    n=n,
                    k=k,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    group_size=group_size,
                    warps_m=warps_m,
                    warps_n=warps_n,
                    ring_producer_warps=prod,
                    ring_stages=stages,
                    ring_startup_all_warps=startup,
                    qpack_k_vector_width=qv,
                    scale_after_group=scale,
                    assume_even_n=n % bn == 0,
                    **flags,
                    decode_threads=dthreads,
                    decode_block_k=dbk,
                    decode_m_rows=drows,
                    warmup_iters=warmup_iters,
                    timed_iters=timed_iters,
                    verify=verify,
                )
            )

    for group_size, shapes in shapes_by_group.items():
        add_jobs(
            group_size=group_size,
            shapes=shapes,
            variant="ring_bonly",
            bm_values=[128, 256],
            bn_values=[64, 128],
            bk_values=bk_values_for_group(group_size),
            warps_m_values=[4, 8],
            warps_n_values=[1, 2],
            producer_values=[2, 4],
            stage_values=[2, 3],
            qv_values=qv_values_for_group(group_size),
            scale_after_group_values=[False],
            startup_all_values=[True, False],
            decode_threads_values=[64],
            decode_block_k_values=[256],
            decode_rows_values=[1],
        )

    for group_size, shapes in small_shapes_by_group.items():
        if group_size >= 32:
            add_jobs(
                group_size=group_size,
                shapes=shapes,
                variant="kpacked_dot2",
                bm_values=[16],
                bn_values=[32],
                bk_values=[32],
                warps_m_values=[1],
                warps_n_values=[1],
                producer_values=[2],
                stage_values=[2],
                qv_values=[2],
                scale_after_group_values=[True],
                startup_all_values=[True],
                decode_threads_values=[64, 128, 256],
                decode_block_k_values=[256, 512, 1024, 2048],
                decode_rows_values=[1, 2, 4, 8],
            )
        add_jobs(
            group_size=group_size,
            shapes=shapes,
            variant="ring_bonly_sync",
            bm_values=[32, 64, 128],
            bn_values=[32, 64, 128],
            bk_values=bk_values_for_group(group_size),
            warps_m_values=[1, 2, 4],
            warps_n_values=[1],
            producer_values=[2],
            stage_values=[2, 3],
            qv_values=qv_values_for_group(group_size),
            scale_after_group_values=[False],
            startup_all_values=[True, False],
            decode_threads_values=[64],
            decode_block_k_values=[256],
            decode_rows_values=[1],
        )
    return jobs


def load_tune_jobs(path: Path) -> list[RunConfig]:
    payload = yaml.safe_load(path.read_text()) or {}
    defaults = {
        "variant": "ring",
        "n": 5120,
        "k": 5120,
        "group": 128,
        "prod": 2,
        "stages": 2,
        "swizzle": 0,
        "group_m": 0,
        "lds_swizzle": False,
        "startup_all": True,
        "qpack_load": True,
        "qv": 2,
        "bf16_deq": False,
        "scale_after_group": True,
        "even_n": False,
        "decode_threads": 64,
        "decode_block_k": 256,
        "decode_rows": 1,
        "pad": 8,
        "warmup_iters": 20,
        "timed_iters": 100,
        "verify": False,
        **payload.get("defaults", {}),
    }
    keys = (
        "variant",
        "bm",
        "bn",
        "bk",
        "group",
        "warps_m",
        "warps_n",
        "prod",
        "stages",
        "swizzle",
        "group_m",
        "lds_swizzle",
        "startup_all",
        "qpack_load",
        "qv",
        "bf16_deq",
        "scale_after_group",
        "even_n",
        "decode_threads",
        "decode_block_k",
        "decode_rows",
        "pad",
    )
    jobs: list[RunConfig] = []
    for entry in payload.get("params", []):
        if "shapes" in entry:
            shapes = entry["shapes"]
        elif "shape" in entry:
            shapes = [entry["shape"]]
        else:
            m_values = sweep_values(entry.get("m", defaults.get("m")))
            n_values = sweep_values(entry.get("n", defaults.get("n")))
            k_values = sweep_values(entry.get("k", defaults.get("k")))
            if None in m_values or None in n_values or None in k_values:
                raise ValueError(f"entry must define m/n/k or shapes: {entry}")
            shapes = list(itertools.product(m_values, n_values, k_values))
        spaces = [sweep_values(entry.get(key, defaults.get(key))) for key in keys]
        for shape in shapes:
            for values in itertools.product(*spaces):
                data = dict(zip(keys, values))
                flags = variant_flags(str(data["variant"]))
                jobs.append(
                    RunConfig(
                        m=int(shape[0]),
                        n=int(shape[1]),
                        k=int(shape[2]),
                        bm=int(data["bm"]),
                        bn=int(data["bn"]),
                        bk=int(data["bk"]),
                        group_size=int(data["group"]),
                        warps_m=int(data["warps_m"]),
                        warps_n=int(data["warps_n"]),
                        ring_producer_warps=int(data["prod"]),
                        ring_stages=int(data["stages"]),
                        block_swizzle_scale=int(data["swizzle"]),
                        group_size_m=int(data["group_m"]),
                        use_lds_swizzle=bool(data["lds_swizzle"]),
                        ring_startup_all_warps=bool(data["startup_all"]),
                        load_b_by_qpack=bool(data["qpack_load"]),
                        qpack_k_vector_width=int(data["qv"]),
                        dequant_b_in_bf16=bool(data["bf16_deq"]),
                        scale_after_group=bool(data["scale_after_group"]),
                        assume_even_n=bool(data["even_n"]),
                        **flags,
                        decode_threads=int(data["decode_threads"]),
                        decode_block_k=int(data["decode_block_k"]),
                        decode_m_rows=int(data["decode_rows"]),
                        smem_pad=int(data["pad"]),
                        warmup_iters=int(
                            entry.get("warmup_iters", defaults.get("warmup_iters", 20))
                        ),
                        timed_iters=int(
                            entry.get("timed_iters", defaults.get("timed_iters", 100))
                        ),
                        verify=bool(entry.get("verify", defaults.get("verify", False))),
                    )
                )
    return jobs


def prebuild_configs(
    configs: list[RunConfig],
    *,
    mojo_bin: str,
    target_accelerator: str | None,
    build_cache_dir: Path,
    timeout_seconds: float,
    build_jobs: int,
    force_rebuild: bool,
) -> set[RunConfig]:
    src = source_digest()
    valid: list[RunConfig] = []
    failed: set[RunConfig] = set()
    seen: set[str] = set()
    skipped_invalid = 0
    for cfg in configs:
        reason = invalid_reason(cfg)
        if reason is not None:
            skipped_invalid += 1
            continue
        key = cache_key(cfg, mojo_bin, target_accelerator, src)
        if key in seen:
            continue
        seen.add(key)
        valid.append(cfg)

    if not valid:
        print(f"[{timestamp()}] build cache: no valid configs to build")
        return failed

    hits = 0
    misses: list[RunConfig] = []
    for cfg in valid:
        binary = cached_binary_path(
            cfg, mojo_bin, target_accelerator, build_cache_dir, src
        )
        if binary.exists() and not force_rebuild:
            hits += 1
        else:
            misses.append(cfg)

    print(
        f"[{timestamp()}] build cache: valid={len(valid)} hits={hits} "
        f"misses={len(misses)} invalid={skipped_invalid} jobs={build_jobs}"
    )
    if not misses:
        return failed

    started = time.time()
    max_workers = max(1, build_jobs)

    def build_one(cfg: RunConfig) -> tuple[RunConfig, Path | None, str | None]:
        try:
            path = build_cached_binary(
                cfg,
                mojo_bin,
                target_accelerator,
                build_cache_dir,
                timeout_seconds,
                src_digest=src,
                force_rebuild=force_rebuild,
            )
            return cfg, path, None
        except Exception as exc:  # noqa: BLE001
            return cfg, None, str(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(build_one, cfg) for cfg in misses]
        for fut in progress_bar(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="build",
        ):
            cfg, _, error = fut.result()
            if error is not None:
                failed.add(cfg)
                print(
                    f"[{timestamp()}] build failed {format_config(cfg)} reason={error}"
                )

    elapsed = time.time() - started
    print(
        f"[{timestamp()}] build cache: built={len(misses) - len(failed)} "
        f"failed={len(failed)} elapsed_s={elapsed:.1f}"
    )
    return failed


def compact_policy_defaults(configs: list[RunConfig]) -> dict[str, Any]:
    candidates = {
        "n": lambda cfg: cfg.n,
        "k": lambda cfg: cfg.k,
        "group": lambda cfg: cfg.group_size,
        "swizzle": lambda cfg: cfg.block_swizzle_scale,
        "group_m": lambda cfg: cfg.group_size_m,
        "lds_swizzle": lambda cfg: cfg.use_lds_swizzle,
        "qpack_load": lambda cfg: cfg.load_b_by_qpack,
        "bf16_deq": lambda cfg: cfg.dequant_b_in_bf16,
        "even_n": lambda cfg: cfg.assume_even_n,
        "pad": lambda cfg: cfg.smem_pad,
        "warmup_iters": lambda cfg: cfg.warmup_iters,
        "timed_iters": lambda cfg: cfg.timed_iters,
    }
    defaults: dict[str, Any] = {}
    for key, getter in candidates.items():
        values = [getter(cfg) for cfg in configs]
        if values and all(value == values[0] for value in values):
            defaults[key] = values[0]
    return defaults


def write_tuned(
    path: Path, source: Path, best: dict[tuple[int, int, int], tuple[RunConfig, float]]
) -> None:
    rows = [item for _, item in sorted(best.items())]
    defaults = compact_policy_defaults([cfg for cfg, _ in rows])
    payload = {
        "metadata": {
            "source": str(source.resolve()),
            "schema": "w4a16-policy-v2",
            "variants": VARIANT_DESCRIPTIONS,
        },
        "defaults": defaults,
        "configs": [cfg.to_policy_dict(ms, defaults) for cfg, ms in rows],
    }
    path.write_text(dump_policy_yaml(payload))


def maybe_write_tuned(
    path: Path | None,
    source: Path,
    best: dict[tuple[int, int, int], tuple[RunConfig, float]],
) -> None:
    if path is not None:
        write_tuned(path, source, best)


def current_policy_device_name() -> str:
    try:
        from vllm.platforms import current_platform

        return current_platform.get_device_name().replace(" ", "_")
    except Exception:  # noqa: BLE001
        return "unknown_device"


def runtime_policy_file_name(
    *,
    n: int,
    k: int,
    group_size: int,
    device_name: str,
    dtype: str,
    op_name: str,
) -> str:
    return (
        f"N={n},K={k},group={group_size},device_name={device_name},"
        f"dtype={dtype},op={op_name}.json"
    )


def write_runtime_policy_jsons(
    output_dir: Path,
    *,
    source: Path | None,
    model: str | None,
    best: dict[tuple[int, int, int], tuple[RunConfig, float]],
    dtype: str,
    device_name: str,
    op_name: str,
) -> list[Path]:
    grouped: dict[tuple[int, int, int], dict[int, tuple[RunConfig, float]]] = {}
    for (_, n, k), (cfg, mean_ms) in best.items():
        grouped.setdefault((n, k, cfg.group_size), {})[cfg.m] = (cfg, mean_ms)

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for (n, k, group_size), configs in sorted(grouped.items()):
        out_path = output_dir / runtime_policy_file_name(
            n=n,
            k=k,
            group_size=group_size,
            device_name=device_name,
            dtype=dtype,
            op_name=op_name,
        )
        payload = {
            "metadata": {
                "schema": "mojo-w4a16-policy-v3",
                "source": str(source.resolve()) if source is not None else None,
                "model": model,
                "device_name": device_name,
                "dtype": dtype,
                "op": op_name,
                "n": n,
                "k": k,
                "group": group_size,
                "generated_by": "vllm/csrc/rocm/mojo_gemm_w4a16/benchmark_w4a16.py",
                "variants": VARIANT_DESCRIPTIONS,
            },
            "defaults": {},
            "configs": {
                str(m): cfg.to_runtime_policy_dict(mean_ms)
                for m, (cfg, mean_ms) in sorted(configs.items())
            },
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
        written.append(out_path)
    return written


def print_shape_families(families: list[ShapeFamily]) -> None:
    print(f"[{timestamp()}] shape families: {len(families)}")
    for family in families:
        print(
            f"  N={family.n} K={family.k} group={family.group_size} op={family.op_name}"
        )


def tune(args: argparse.Namespace) -> int:
    source_path: Path | None = args.configs
    if args.model:
        families = discover_model_shape_families(
            args.model,
            tp_size=args.tp_size,
            trust_remote_code=args.trust_remote_code,
            explicit_group_size=args.model_group_size,
        )
        families = sorted(
            set(families + args.shape_family),
            key=lambda item: (item.n, item.k, item.group_size),
        )
        print_shape_families(families)
        jobs = make_tune_jobs_from_families(
            families,
            m_buckets=parse_int_csv(args.m_buckets),
            small_m_buckets=parse_int_csv(args.small_m_buckets),
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
            verify=args.verify,
        )
        source_path = None
    else:
        jobs = load_tune_jobs(args.configs)
        if args.shape_family:
            families = sorted(
                args.shape_family, key=lambda item: (item.n, item.k, item.group_size)
            )
            print_shape_families(families)
            jobs.extend(
                make_tune_jobs_from_families(
                    families,
                    m_buckets=parse_int_csv(args.m_buckets),
                    small_m_buckets=parse_int_csv(args.small_m_buckets),
                    warmup_iters=args.warmup_iters,
                    timed_iters=args.timed_iters,
                    verify=args.verify,
                )
            )
    if args.dry_run:
        valid = sum(1 for cfg in jobs if invalid_reason(cfg) is None)
        print(f"[{timestamp()}] dry-run jobs={len(jobs)} valid={valid}")
        return 0

    use_cache = True
    failed_builds: set[RunConfig] = set()
    src = source_digest()
    if args.run_only:
        print(
            f"[{timestamp()}] run-only: using cached binaries from "
            f"{args.build_cache_dir}"
        )
    else:
        print(f"[{timestamp()}] tune: checking/building cache before measurement")
        failed_builds = prebuild_configs(
            jobs,
            mojo_bin=args.mojo_bin,
            target_accelerator=args.target_accelerator,
            build_cache_dir=args.build_cache_dir,
            timeout_seconds=args.timeout_seconds,
            build_jobs=args.build_jobs,
            force_rebuild=args.force_rebuild,
        )
    if args.build_only:
        return 1 if failed_builds else 0

    best: dict[tuple[int, int, int], tuple[RunConfig, float]] = {}
    if args.seed_tuned_config is not None:
        best.update(load_seed_configs(args.seed_tuned_config))
        if best:
            maybe_write_tuned(args.output, args.configs, best)
            if args.save_dir is not None:
                write_runtime_policy_jsons(
                    args.save_dir,
                    source=source_path,
                    model=args.model,
                    best=best,
                    dtype=args.policy_dtype,
                    device_name=args.policy_device_name or current_policy_device_name(),
                    op_name=args.policy_op_name,
                )
            print(
                f"[{timestamp()}] seeded {len(best)} configs from "
                f"{args.seed_tuned_config}"
            )
    run_started = time.time()
    total_jobs = len(jobs)
    for cfg in progress_bar(jobs, total=total_jobs, desc="run"):
        reason = invalid_reason(cfg)
        if reason is not None:
            print(f"[{timestamp()}] skip invalid {format_config(cfg)} reason={reason}")
            continue
        if cfg in failed_builds:
            print(f"[{timestamp()}] skip build-failed {format_config(cfg)}")
            continue
        started = time.time()
        try:
            mean_ms = run_single_config(
                cfg,
                args.mojo_bin,
                args.target_accelerator,
                args.timeout_seconds,
                use_build_cache=use_cache,
                build_cache_dir=args.build_cache_dir,
                force_rebuild=False,
                src_digest=src,
                require_cached_binary=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{timestamp()}] failed {format_config(cfg)} reason={exc}")
            continue
        elapsed = time.time() - started
        print(
            f"[{timestamp()}] candidate {format_config(cfg)} "
            f"elapsed_s={elapsed:.1f} mean_ms={mean_ms}"
        )
        prev = best.get(cfg.shape)
        if prev is None or mean_ms < prev[1]:
            best[cfg.shape] = (cfg, mean_ms)
            maybe_write_tuned(args.output, args.configs, best)
            if args.save_dir is not None:
                written = write_runtime_policy_jsons(
                    args.save_dir,
                    source=source_path,
                    model=args.model,
                    best=best,
                    dtype=args.policy_dtype,
                    device_name=args.policy_device_name or current_policy_device_name(),
                    op_name=args.policy_op_name,
                )
                print(f"[{timestamp()}] wrote {len(written)} runtime JSON policy files")
    maybe_write_tuned(args.output, args.configs, best)
    if args.save_dir is not None:
        written = write_runtime_policy_jsons(
            args.save_dir,
            source=source_path,
            model=args.model,
            best=best,
            dtype=args.policy_dtype,
            device_name=args.policy_device_name or current_policy_device_name(),
            op_name=args.policy_op_name,
        )
        print(f"wrote {len(written)} runtime JSON policy files to {args.save_dir}")
    if args.output is not None:
        print(
            f"wrote {args.output} with {len(best)} configs "
            f"elapsed_s={time.time() - run_started:.1f}"
        )
    else:
        print(f"tuned {len(best)} configs elapsed_s={time.time() - run_started:.1f}")
    return 0 if best else 1


def replay(args: argparse.Namespace) -> int:
    configs = load_tuned_configs(args.tuned_config, verify=args.verify)
    use_cache = args.use_build_cache or args.build_only or args.run_only
    failed_builds: set[RunConfig] = set()
    src = source_digest() if use_cache else None
    if use_cache:
        if args.run_only:
            print(
                f"[{timestamp()}] run-only: using cached binaries from "
                f"{args.build_cache_dir}"
            )
        else:
            failed_builds = prebuild_configs(
                configs,
                mojo_bin=args.mojo_bin,
                target_accelerator=args.target_accelerator,
                build_cache_dir=args.build_cache_dir,
                timeout_seconds=args.timeout_seconds,
                build_jobs=args.build_jobs,
                force_rebuild=args.force_rebuild,
            )
        if args.build_only:
            return 1 if failed_builds else 0

    run_started = time.time()
    for cfg in progress_bar(configs, total=len(configs), desc="run"):
        if cfg in failed_builds:
            print(f"[{timestamp()}] skip build-failed {format_config(cfg)}")
            continue
        started = time.time()
        mean_ms = run_single_config(
            cfg,
            args.mojo_bin,
            args.target_accelerator,
            args.timeout_seconds,
            use_build_cache=use_cache,
            build_cache_dir=args.build_cache_dir,
            force_rebuild=False,
            src_digest=src,
            require_cached_binary=args.run_only,
        )
        elapsed = time.time() - started
        print(
            f"[{timestamp()}] tuned {format_config(cfg)} "
            f"elapsed_s={elapsed:.1f} mean_ms={mean_ms}"
        )
    print(f"[{timestamp()}] replay elapsed_s={time.time() - run_started:.1f}")
    return 0


def current(args: argparse.Namespace) -> int:
    cfg = RunConfig(
        m=args.m,
        n=args.n,
        k=args.k,
        bm=args.bm,
        bn=args.bn,
        bk=args.bk,
        group_size=args.group_size,
        warps_m=args.warps_m,
        warps_n=args.warps_n,
        ring_producer_warps=args.ring_producer_warps,
        ring_stages=args.ring_stages,
        block_swizzle_scale=args.block_swizzle_scale,
        group_size_m=args.group_size_m,
        use_lds_swizzle=bool(args.use_lds_swizzle),
        ring_startup_all_warps=bool(args.ring_startup_all_warps),
        load_b_by_qpack=bool(args.load_b_by_qpack),
        qpack_k_vector_width=args.qpack_k_vector_width,
        dequant_b_in_bf16=bool(args.dequant_b_in_bf16),
        scale_after_group=bool(args.scale_after_group),
        assume_even_n=bool(args.assume_even_n),
        use_decode_kernel=bool(args.use_decode_kernel),
        use_splitk_decode_kernel=bool(args.use_splitk_decode_kernel),
        use_kpacked_decode_kernel=bool(args.use_kpacked_decode_kernel),
        use_kpacked_dot2=bool(args.use_kpacked_dot2),
        use_wmma16_kernel=bool(args.use_wmma16_kernel),
        use_ring_bonly_kernel=bool(args.use_ring_bonly_kernel),
        use_ring_bonly_sync_kernel=bool(args.use_ring_bonly_sync_kernel),
        decode_threads=args.decode_threads,
        decode_block_k=args.decode_block_k,
        decode_m_rows=args.decode_m_rows,
        smem_pad=args.smem_pad,
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
        verify=args.verify,
    )
    use_cache = args.use_build_cache or args.build_only or args.run_only
    src = source_digest() if use_cache else None
    if use_cache and not args.run_only:
        failed = prebuild_configs(
            [cfg],
            mojo_bin=args.mojo_bin,
            target_accelerator=args.target_accelerator,
            build_cache_dir=args.build_cache_dir,
            timeout_seconds=args.timeout_seconds,
            build_jobs=args.build_jobs,
            force_rebuild=args.force_rebuild,
        )
        if failed:
            return 1
    if args.build_only:
        return 0
    mean_ms = run_single_config(
        cfg,
        args.mojo_bin,
        args.target_accelerator,
        args.timeout_seconds,
        use_build_cache=use_cache,
        build_cache_dir=args.build_cache_dir,
        force_rebuild=False,
        src_digest=src,
        require_cached_binary=args.run_only,
    )
    print(f"[{timestamp()}] current {format_config(cfg)} mean_ms={mean_ms}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune and benchmark Mojo W4A16 GEMM.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tune", action="store_true")
    mode.add_argument("--tuned-config", type=Path)
    parser.add_argument(
        "--configs",
        type=Path,
        default=KERNEL_DIR / "tuning_sweep_w4a16_overnight_65536.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional legacy YAML output. When omitted with --save-dir, tuning "
            "writes only serving-ready JSON policy files."
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help=(
            "Write serving-ready Mojo W4A16 JSON policy files to this directory. "
            f"Typical value: {DEFAULT_POLICY_CONFIG_DIR}"
        ),
    )
    parser.add_argument(
        "--model",
        help="Derive AWQ W4A16 shape families from this Hugging Face/vLLM model.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tp-size", "--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--model-group-size",
        type=int,
        help="Override quantization group size when deriving shapes from --model.",
    )
    parser.add_argument(
        "--shape-family",
        action="append",
        type=parse_shape_family,
        default=[],
        help="Explicit N,K,G or NxKxG family to tune. May be repeated.",
    )
    parser.add_argument(
        "--m-buckets",
        default=",".join(map(str, DEFAULT_M_BUCKETS)),
        help="Comma-separated M buckets for model/shape-family generated sweeps.",
    )
    parser.add_argument(
        "--small-m-buckets",
        default=",".join(map(str, DEFAULT_SMALL_M_BUCKETS)),
        help="Comma-separated M buckets for decode/small-M variants.",
    )
    parser.add_argument(
        "--policy-dtype",
        choices=("fp16_w4a16", "bf16_w4a16"),
        default="fp16_w4a16",
        help="Runtime policy dtype string for generated JSON files.",
    )
    parser.add_argument(
        "--policy-device-name",
        help="Override generated JSON device_name; defaults to current vLLM platform.",
    )
    parser.add_argument("--policy-op-name", default=DEFAULT_POLICY_OP_NAME)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered shape families and generated job count, then exit.",
    )
    parser.add_argument(
        "--seed-tuned-config",
        type=Path,
        help=(
            "Preload an existing tuned policy into the output before sweeping new jobs."
        ),
    )
    parser.add_argument("--mojo-bin", default=DEFAULT_MOJO)
    parser.add_argument("--target-accelerator", default="gfx1151")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--use-build-cache",
        action="store_true",
        help=(
            "Compile configs with mojo build and run cached binaries. "
            "This is always enabled for --tune."
        ),
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Compile cache misses and exit without benchmarking.",
    )
    parser.add_argument(
        "--run-only",
        action="store_true",
        help="Run only already cached binaries; fail if a binary is missing.",
    )
    parser.add_argument(
        "--build-cache-dir",
        type=Path,
        default=DEFAULT_BUILD_CACHE_DIR,
        help="Directory for cached benchmark_runner binaries.",
    )
    parser.add_argument(
        "--build-jobs",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1) // 2 or 1)),
        help="Parallel mojo build jobs for cache misses.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild cached binaries even when cache entries already exist.",
    )
    parser.add_argument(
        "--clear-build-cache",
        action="store_true",
        help="Delete the build cache directory before doing anything else.",
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--m", type=int, default=12)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--bm", type=int, default=16)
    parser.add_argument("--bn", type=int, default=32)
    parser.add_argument("--bk", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warps-m", type=int, default=1)
    parser.add_argument("--warps-n", type=int, default=1)
    parser.add_argument("--ring-producer-warps", type=int, default=2)
    parser.add_argument(
        "--num-stages", "--ring-stages", dest="ring_stages", type=int, default=2
    )
    parser.add_argument("--block-swizzle-scale", type=int, default=0)
    parser.add_argument("--group-size-m", type=int, default=0)
    parser.add_argument("--use-lds-swizzle", type=int, choices=(0, 1), default=0)
    parser.add_argument("--ring-startup-all-warps", type=int, choices=(0, 1), default=1)
    parser.add_argument("--load-b-by-qpack", type=int, choices=(0, 1), default=1)
    parser.add_argument("--qpack-k-vector-width", type=int, default=2)
    parser.add_argument("--dequant-b-in-bf16", type=int, choices=(0, 1), default=0)
    parser.add_argument("--scale-after-group", type=int, choices=(0, 1), default=1)
    parser.add_argument("--assume-even-n", type=int, choices=(0, 1), default=1)
    parser.add_argument("--use-decode-kernel", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--use-splitk-decode-kernel", type=int, choices=(0, 1), default=0
    )
    parser.add_argument(
        "--use-kpacked-decode-kernel", type=int, choices=(0, 1), default=0
    )
    parser.add_argument("--use-kpacked-dot2", type=int, choices=(0, 1), default=0)
    parser.add_argument("--use-wmma16-kernel", type=int, choices=(0, 1), default=0)
    parser.add_argument("--use-ring-bonly-kernel", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--use-ring-bonly-sync-kernel", type=int, choices=(0, 1), default=0
    )
    parser.add_argument("--decode-threads", type=int, default=64)
    parser.add_argument("--decode-block-k", type=int, default=256)
    parser.add_argument("--decode-m-rows", type=int, default=1)
    parser.add_argument("--smem-pad", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--timed-iters", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.build_only and args.run_only:
        raise SystemExit("--build-only and --run-only are mutually exclusive")
    if args.build_jobs < 1:
        raise SystemExit("--build-jobs must be >= 1")
    if args.output is None and args.save_dir is None:
        args.output = KERNEL_DIR / "tuned_configs_w4a16_overnight_65536.yaml"
    if args.clear_build_cache and args.build_cache_dir.exists():
        shutil.rmtree(args.build_cache_dir)
        print(f"[{timestamp()}] cleared build cache {args.build_cache_dir}")
    if args.tune:
        return tune(args)
    if args.tuned_config:
        return replay(args)
    return current(args)


if __name__ == "__main__":
    raise SystemExit(main())
