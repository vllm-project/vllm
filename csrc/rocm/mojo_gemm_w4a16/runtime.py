# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable Mojo W4A16 GEMM runtime helpers for ROCm.

This module owns policy selection, generated Mojo extension materialization,
runner caching, graph-safe scratch preparation, and native HIP-stream launches.
The vLLM linear integration imports it by path and keeps only torch custom-op
registration plus AWQ layer plumbing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

SOURCE_DIR = Path(__file__).resolve().parent
MOJO_SOURCE_DIR = SOURCE_DIR / "mojo"
KERNEL_SOURCE_DIR = MOJO_SOURCE_DIR / "kernels"
POLICY_CONFIG_DIR = SOURCE_DIR / "policies"
POLICY_OP_NAME = "awq_gemm"
TEMPLATE_PATH = SOURCE_DIR / "templates" / "direct_extension_template.mojo"
HIP_SHIM_PATH = SOURCE_DIR / "hip_launch_shim.cpp"
MOJO_BIN_CANDIDATES = ("mojo", "/opt/python/bin/mojo")
MAX_CLI_CANDIDATES = ("max", "/opt/python/bin/max")
GENERATOR_CACHE_VERSION = "ring-counter-init-barrier-v1"


@dataclass(frozen=True)
class VariantSpec:
    source_path: Path
    generated_module: str
    kernel_symbol: str
    active_kernel: str
    block_dim: str
    split_grid: str
    split_block: str
    qweight_layout: str
    needs_partial: bool = False
    uses_kpacked_qweight: bool = False


_COMMON_KERNEL_ARGS = """[
    ALayout,
    QWeightLayout,
    QZerosLayout,
    ScalesLayout,
    CLayout,
    ImmutExternalOrigin,
    ImmutExternalOrigin,
    ImmutExternalOrigin,
    ImmutExternalOrigin,
]"""
VARIANT_SPECS = {
    "ring": VariantSpec(
        source_path=KERNEL_SOURCE_DIR / "ring_ab_staged.mojo",
        generated_module="ring_ab_staged",
        kernel_symbol="gemm_w4a16_ring_ab_staged_kernel",
        active_kernel="""gemm_w4a16_ring_ab_staged_kernel[
    ALayout,
    QWeightLayout,
    QZerosLayout,
    ScalesLayout,
    CLayout,
    BM,
    BN,
    BK,
]""",
        block_dim="(PRODUCTION_TOTAL_THREADS, 1)",
        split_grid="(1, 1, 1)",
        split_block="(1, 1)",
        qweight_layout="awq_packed",
    ),
    "kpacked_dot2": VariantSpec(
        source_path=KERNEL_SOURCE_DIR / "kpacked_dot2_splitk.mojo",
        generated_module="kpacked_dot2_splitk",
        kernel_symbol="gemm_w4a16_kpacked_dot2_splitk_kernel",
        active_kernel="""gemm_w4a16_kpacked_dot2_splitk_kernel[
    ALayout,
    QWeightKPackedLayout,
    QZerosLayout,
    ScalesLayout,
    PartialLayout,
]""",
        block_dim="(SPLITK_THREADS, 1)",
        split_grid="""(
    ceildiv(ceildiv(n, 4), SPLITK_THREADS),
    ceildiv(m, SPLITK_ROWS_PER_CTA),
    ceildiv(k, SPLITK_BLOCK_K),
)""",
        split_block="(SPLITK_THREADS, 1)",
        qweight_layout="kpacked",
        needs_partial=True,
        uses_kpacked_qweight=True,
    ),
    "wmma16": VariantSpec(
        source_path=KERNEL_SOURCE_DIR / "kpacked_wmma16_splitk.mojo",
        generated_module="kpacked_wmma16_splitk",
        kernel_symbol="gemm_w4a16_kpacked_wmma16_splitk_kernel",
        active_kernel="""gemm_w4a16_kpacked_wmma16_splitk_kernel[
    ALayout,
    QWeightKPackedLayout,
    QZerosLayout,
    ScalesLayout,
    PartialLayout,
]""",
        block_dim="(32, 1)",
        split_grid="""(
    ceildiv(n, 16),
    ceildiv(m, 16),
    ceildiv(k, SPLITK_BLOCK_K),
)""",
        split_block="(32, 1)",
        qweight_layout="kpacked",
        needs_partial=True,
        uses_kpacked_qweight=True,
    ),
    "ring_bonly": VariantSpec(
        source_path=KERNEL_SOURCE_DIR / "ring_b_staged.mojo",
        generated_module="ring_b_staged",
        kernel_symbol="gemm_w4a16_ring_b_staged_kernel",
        active_kernel="gemm_w4a16_ring_b_staged_kernel" + _COMMON_KERNEL_ARGS,
        block_dim="(PRODUCTION_TOTAL_THREADS, 1)",
        split_grid="(1, 1, 1)",
        split_block="(1, 1)",
        qweight_layout="awq_packed",
    ),
    "ring_bonly_sync": VariantSpec(
        source_path=KERNEL_SOURCE_DIR / "b_staged_sync.mojo",
        generated_module="b_staged_sync",
        kernel_symbol="gemm_w4a16_b_staged_sync_kernel",
        active_kernel="gemm_w4a16_b_staged_sync_kernel" + _COMMON_KERNEL_ARGS,
        block_dim="(COMPUTE_WARPS * 32, 1)",
        split_grid="(1, 1, 1)",
        split_block="(1, 1)",
        qweight_layout="awq_packed",
    ),
}
SUPPORTED_VARIANTS = set(VARIANT_SPECS)
_RUNNERS: dict[tuple[str, int], object] = {}
_PARTIAL_SCRATCH_CACHE: dict[tuple[str, int], torch.Tensor] = {}
_SYMMETRIC_QZEROS_CACHE: set[tuple[int, tuple[int, ...], tuple[int, ...], int]] = set()
_QWEIGHT_KPACKED_CACHE: dict[
    tuple[int, tuple[int, ...], tuple[int, ...], int],
    tuple[torch.Tensor, torch.Tensor],
] = {}
_MISSING_POLICY_WARNED: set[tuple[str, str, int, int, int, int]] = set()
_POLICY_DEVICE_NAME_CACHE: str | None = None

logger = init_logger(__name__)


def _find_executable(candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        path = shutil.which(candidate)
        if path is not None:
            return path
        if Path(candidate).exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def mojo_bin() -> str:
    path = _find_executable(MOJO_BIN_CANDIDATES)
    if path is None:
        raise RuntimeError(
            "Mojo W4A16 requires the Mojo compiler. Install MAX/Mojo and make "
            "`mojo` available on PATH."
        )
    return path


@lru_cache(maxsize=1)
def dependency_error() -> str | None:
    if (
        importlib.util.find_spec("max") is None
        and _find_executable(MAX_CLI_CANDIDATES) is None
    ):
        return (
            "Mojo W4A16 requires MAX to be installed. Install MAX/Mojo before "
            "using --linear-backend mojo."
        )
    if _find_executable(MOJO_BIN_CANDIDATES) is None:
        return (
            "Mojo W4A16 requires the Mojo compiler. Install MAX/Mojo and make "
            "`mojo` available on PATH before using --linear-backend mojo."
        )
    return None


def check_dependencies() -> tuple[bool, str | None]:
    error = dependency_error()
    return error is None, error


def _debug_enabled() -> bool:
    return os.environ.get("VLLM_W4A16_DEBUG", "0") == "1"


def _mem_debug_enabled() -> bool:
    return os.environ.get("VLLM_W4A16_MEM_DEBUG", "0") == "1"


def _is_compiling() -> bool:
    return getattr(torch.compiler, "is_compiling", lambda: False)()


def _debug_log(message: str) -> None:
    if _debug_enabled() and not _is_compiling():
        print(f"[mojo_w4a16] {message}", file=sys.stderr, flush=True)


def _format_gib(value: int | float) -> str:
    return f"{float(value) / (1024**3):.3f}GiB"


def _mem_debug(label: str, tensor: torch.Tensor | None = None) -> None:
    if not _mem_debug_enabled():
        return
    try:
        device = (
            tensor.device
            if tensor is not None
            else torch.accelerator.current_device_index()
        )
        free, total = torch.accelerator.get_memory_info(device)
        allocated = torch.accelerator.memory_allocated(device)
        reserved = torch.accelerator.memory_reserved(device)
        print(
            f"mem {label}: free={_format_gib(free)} total={_format_gib(total)} "
            f"torch_alloc={_format_gib(allocated)} "
            f"torch_reserved={_format_gib(reserved)} runners={len(_RUNNERS)}",
            file=sys.stderr,
            flush=True,
        )
    except Exception as exc:
        print(
            f"mem {label}: unavailable: {exc}",
            file=sys.stderr,
            flush=True,
        )


def _policy_device_name() -> str:
    global _POLICY_DEVICE_NAME_CACHE
    if _POLICY_DEVICE_NAME_CACHE is None:
        _POLICY_DEVICE_NAME_CACHE = current_platform.get_device_name().replace(" ", "_")
    return _POLICY_DEVICE_NAME_CACHE


def _policy_dtype(a_dtype: torch.dtype) -> str:
    if a_dtype == torch.float16:
        return "fp16_w4a16"
    if a_dtype == torch.bfloat16:
        return "bf16_w4a16"
    raise TypeError(f"unsupported Mojo W4A16 activation dtype: {a_dtype}")


def _policy_file_name(
    *,
    n: int,
    k: int,
    group_size: int,
    dtype: str,
    op_name: str = POLICY_OP_NAME,
    device_name: str | None = None,
) -> str:
    if device_name is None:
        device_name = _policy_device_name()
    return (
        f"N={n},K={k},group={group_size},device_name={device_name},"
        f"dtype={dtype},op={op_name}.json"
    )


def _policy_candidate_paths(
    *,
    n: int,
    k: int,
    group_size: int,
    dtype: str,
    op_name: str = POLICY_OP_NAME,
) -> list[Path]:
    file_name = _policy_file_name(
        n=n,
        k=k,
        group_size=group_size,
        dtype=dtype,
        op_name=op_name,
    )
    paths: list[Path] = []
    if envs.VLLM_TUNED_CONFIG_FOLDER is not None:
        paths.append(Path(envs.VLLM_TUNED_CONFIG_FOLDER).expanduser() / file_name)
    paths.append(POLICY_CONFIG_DIR / file_name)
    return paths


def _warn_missing_policy_config(
    *,
    reason: str,
    m: int,
    n: int,
    k: int,
    group_size: int,
    dtype: str,
    device_name: str,
    op_name: str,
    fallback: MojoRunConfig,
    max_policy_m: int | None = None,
    searched_paths: list[Path] | None = None,
) -> None:
    if _is_compiling():
        return

    warn_key = (reason, dtype, m if reason == "row_bucket" else 0, n, k, group_size)
    if warn_key not in _MISSING_POLICY_WARNED:
        _MISSING_POLICY_WARNED.add(warn_key)
        if reason == "shape_family":
            logger.warning(
                "Using default Mojo W4A16 config. Performance might be "
                "sub-optimal! Matching config not found for shape "
                "(M=%d, N=%d, K=%d, group=%d), device_name=%s, dtype=%s, "
                "op=%s. Config file not found at %s. Falling back to %s.",
                m,
                n,
                k,
                group_size,
                device_name,
                dtype,
                op_name,
                ", ".join(str(path) for path in searched_paths or ()),
                fallback.key,
            )
        else:
            logger.warning(
                "Mojo W4A16 tuned config does not cover runtime M for shape "
                "(M=%d, N=%d, K=%d, group=%d), device_name=%s, dtype=%s, "
                "op=%s. Largest tuned M=%s. Falling back to %s. "
                "Add this M bucket to the tuning policy for better performance.",
                m,
                n,
                k,
                group_size,
                device_name,
                dtype,
                op_name,
                max_policy_m,
                fallback.key,
            )


def _current_stream_ptr(tensor: torch.Tensor) -> int:
    stream = torch.cuda.current_stream(tensor.device)
    stream_addr = getattr(stream, "cuda_stream", None)
    if stream_addr is None:
        stream_addr = getattr(stream, "hip_stream", None)
    if stream_addr is None:
        raise RuntimeError(f"could not read HIP stream pointer from {stream!r}")
    return int(stream_addr)


def _stream_debug(tensor: torch.Tensor) -> str:
    if _is_compiling():
        return (
            f"device={tensor.device} stream=<compile-trace> "
            "capturing=<compile-trace> is_compiling=True"
        )
    return (
        f"device={tensor.device} stream=0x{_current_stream_ptr(tensor):x} "
        f"capturing={torch.cuda.is_current_stream_capturing()} "
        f"is_compiling=False"
    )


def _pt2_tags() -> tuple[torch.Tag, ...]:
    return ()


@dataclass(frozen=True)
class MojoRunConfig:
    m: int
    n: int
    k: int
    bm: int
    bn: int
    bk: int
    group_size: int = 128
    variant: str = "ring_bonly"
    warps_m: int = 1
    warps_n: int = 1
    ring_producer_warps: int = 2
    ring_stages: int = 2
    block_swizzle_scale: int = 0
    group_size_m: int = 0
    use_lds_swizzle: bool = False
    ring_startup_all_warps: bool = True
    load_b_by_qpack: bool = True
    qpack_k_vector_width: int = 4
    dequant_b_in_bf16: bool = False
    scale_after_group: bool = True
    assume_even_n: bool = False
    splitk_threads: int = 64
    splitk_block_k: int = 256
    splitk_rows_per_cta: int = 1
    smem_pad: int = 8
    use_fp16: bool = False
    zp_bias: int = 8
    use_qzeros: bool = False
    zero_offset: int = 0

    @property
    def key(self) -> str:
        return (
            f"{self.variant}_m{self.m}_n{self.n}_k{self.k}"
            f"_bm{self.bm}_bn{self.bn}_bk{self.bk}"
            f"_wm{self.warps_m}_wn{self.warps_n}"
            f"_prod{self.ring_producer_warps}_st{self.ring_stages}"
            f"_qv{self.qpack_k_vector_width}_pad{self.smem_pad}"
            f"_fp16{int(self.use_fp16)}"
            f"_zp{self.zp_bias}_qz{int(self.use_qzeros)}_zo{self.zero_offset}"
        )

    @classmethod
    def from_policy_dict(
        cls,
        data: dict[str, Any],
        defaults: dict[str, Any],
    ) -> MojoRunConfig:
        def get(name: str, default: Any = None) -> Any:
            return data.get(name, defaults.get(name, default))

        return cls(
            m=int(get("m")),
            n=int(get("n")),
            k=int(get("k")),
            group_size=int(get("group", 128)),
            variant=str(get("variant", "ring")),
            bm=int(get("tile_M")),
            bn=int(get("tile_N")),
            bk=int(get("tile_K")),
            warps_m=int(get("warps_M")),
            warps_n=int(get("warps_N")),
            ring_producer_warps=int(get("producer_warps", 1)),
            ring_stages=int(get("ring_stages", 2)),
            block_swizzle_scale=int(get("block_swizzle_scale", 0)),
            group_size_m=int(get("group_size_M", 0)),
            use_lds_swizzle=bool(get("use_lds_swizzle", False)),
            ring_startup_all_warps=bool(get("ring_startup_all_warps", True)),
            load_b_by_qpack=bool(get("load_B_by_qpack", True)),
            qpack_k_vector_width=int(get("qpack_K_vector_width", 2)),
            dequant_b_in_bf16=bool(get("dequant_B_in_bf16", False)),
            scale_after_group=bool(get("scale_after_group", True)),
            assume_even_n=bool(get("assume_even_N", False)),
            splitk_threads=int(get("splitk_threads", 64)),
            splitk_block_k=int(get("splitk_block_K", 256)),
            splitk_rows_per_cta=int(get("splitk_rows_per_CTA", 1)),
            smem_pad=int(get("smem_pad", 8)),
        )


def _bool(value: bool) -> str:
    return "True" if value else "False"


def _compile_m_for_runtime_m(runtime_m: int, tile_m: int) -> int:
    if runtime_m <= 0:
        raise ValueError(f"Mojo W4A16 runtime M must be positive, got {runtime_m}")
    if tile_m <= 0:
        raise ValueError(f"Mojo W4A16 tile_M must be positive, got {tile_m}")
    compiled_m = ((runtime_m + tile_m - 1) // tile_m) * tile_m
    if compiled_m < runtime_m:
        raise AssertionError(
            f"Mojo W4A16 compiled M {compiled_m} does not cover runtime M {runtime_m}"
        )
    return compiled_m


def _packed_uniform_nibble(nibble: int) -> int:
    packed = 0
    for shift in range(0, 32, 4):
        packed |= (nibble & 0xF) << shift
    if packed >= 1 << 31:
        packed -= 1 << 32
    return packed


def _qzeros_are_symmetric(
    qzeros: torch.Tensor,
    *,
    zp_bias: int,
    use_v2_format: bool,
) -> bool:
    expected_nibble = zp_bias if use_v2_format else zp_bias - 1
    if expected_nibble < 0 or expected_nibble > 15:
        return False
    expected_word = _packed_uniform_nibble(expected_nibble)
    key = (
        qzeros.data_ptr(),
        tuple(qzeros.shape),
        tuple(qzeros.stride()),
        expected_word,
    )
    if key in _SYMMETRIC_QZEROS_CACHE:
        return True
    if bool(torch.all(qzeros == expected_word).item()):
        _SYMMETRIC_QZEROS_CACHE.add(key)
        return True
    return False


def _uses_kpacked_qweight(cfg: MojoRunConfig) -> bool:
    return VARIANT_SPECS[cfg.variant].uses_kpacked_qweight


def _qweight_kpacked_cache_key(
    qweight: torch.Tensor,
) -> tuple[int, tuple[int, ...], tuple[int, ...], int]:
    device_index = qweight.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    return (
        qweight.data_ptr(),
        tuple(qweight.shape),
        tuple(qweight.stride()),
        int(device_index),
    )


def _make_qweight_kpacked(qweight: torch.Tensor) -> torch.Tensor:
    k, n_packed = qweight.shape
    n = n_packed * 8
    qweight_kpacked = torch.empty(
        (k // 8, n),
        dtype=qweight.dtype,
        device=qweight.device,
    )
    for n_lane in range(8):
        packed_by_k = torch.zeros(
            (k // 8, n_packed),
            dtype=qweight.dtype,
            device=qweight.device,
        )
        for k_lane in range(8):
            nibble = (qweight[k_lane::8, :] >> (n_lane * 4)) & 0xF
            packed_by_k |= nibble << (k_lane * 4)
        qweight_kpacked[:, n_lane::8] = packed_by_k
    return qweight_kpacked


@torch.compiler.disable
def _get_qweight_kpacked(cfg: MojoRunConfig, qweight: torch.Tensor) -> torch.Tensor:
    if not _uses_kpacked_qweight(cfg):
        return qweight
    if qweight.shape[0] % 8:
        raise ValueError(
            f"{cfg.variant} requires K divisible by 8, got K={qweight.shape[0]}"
        )

    key = _qweight_kpacked_cache_key(qweight)
    cached = _QWEIGHT_KPACKED_CACHE.get(key)
    if cached is not None:
        return cached[1]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Mojo W4A16 qweight_kpacked was not prepared before graph capture"
        )

    _mem_debug(f"before qweight_kpacked cfg={cfg.key}", qweight)
    qweight_kpacked = _make_qweight_kpacked(qweight)
    _mem_debug(f"after qweight_kpacked cfg={cfg.key}", qweight)
    _QWEIGHT_KPACKED_CACHE[key] = (qweight, qweight_kpacked)
    _debug_log(
        "prepared qweight_kpacked "
        f"qweight={tuple(qweight.shape)} kpacked={tuple(qweight_kpacked.shape)} "
        f"variant={cfg.variant}"
    )
    return qweight_kpacked


def _cache_prepared_qweight_kpacked(
    qweight: torch.Tensor, qweight_kpacked: torch.Tensor
) -> None:
    _QWEIGHT_KPACKED_CACHE[_qweight_kpacked_cache_key(qweight)] = (
        qweight,
        qweight_kpacked,
    )


def _is_qweight_kpacked_arg(
    qweight: torch.Tensor,
    qweight_kpacked_or_dummy: torch.Tensor,
) -> bool:
    return (
        qweight_kpacked_or_dummy.dim() == 2
        and qweight.dim() == 2
        and qweight_kpacked_or_dummy.shape[0] == qweight.shape[0] // 8
        and qweight_kpacked_or_dummy.shape[1] == qweight.shape[1] * 8
    )


def _resolve_qweight_kpacked(
    cfg: MojoRunConfig,
    qweight: torch.Tensor,
    qweight_kpacked_or_dummy: torch.Tensor,
) -> torch.Tensor:
    if not _uses_kpacked_qweight(cfg):
        return qweight
    if _is_qweight_kpacked_arg(qweight, qweight_kpacked_or_dummy):
        _cache_prepared_qweight_kpacked(qweight, qweight_kpacked_or_dummy)
        return qweight_kpacked_or_dummy
    return _get_qweight_kpacked(cfg, qweight)


def _module_cache_root() -> Path:
    return Path.home() / ".cache" / "vllm" / "mojo_w4a16"


@lru_cache(maxsize=1)
def _load_hip_launch_shim() -> ModuleType:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Mojo W4A16 HIP launch shim was not built before graph capture"
        )

    from torch.utils.cpp_extension import load

    build_dir = _module_cache_root() / "hip_launch_shim"
    build_dir.mkdir(parents=True, exist_ok=True)
    _debug_log(f"building HIP launch shim source={HIP_SHIM_PATH} build_dir={build_dir}")
    return load(
        name="vllm_mojo_w4a16_hip_launch_shim",
        sources=[str(HIP_SHIM_PATH)],
        build_directory=str(build_dir),
        extra_cflags=["-O2"],
        extra_ldflags=["-ldl"],
        verbose=_debug_enabled(),
    )


def _validate_policy_config(path: Path, cfg: MojoRunConfig) -> None:
    for name, value in (
        ("M bucket", cfg.m),
        ("N", cfg.n),
        ("K", cfg.k),
        ("group", cfg.group_size),
        ("tile_M", cfg.bm),
        ("tile_N", cfg.bn),
        ("tile_K", cfg.bk),
        ("warps_M", cfg.warps_m),
        ("warps_N", cfg.warps_n),
    ):
        if value <= 0:
            raise ValueError(f"Mojo W4A16 policy {path} has invalid {name}: {value}")
    if cfg.group_size > cfg.k or cfg.k % cfg.group_size:
        raise ValueError(
            f"Mojo W4A16 policy {path} has invalid group={cfg.group_size} for K={cfg.k}"
        )
    if cfg.bk > cfg.group_size or cfg.group_size % cfg.bk:
        raise ValueError(
            f"Mojo W4A16 policy {path} has invalid tile_K={cfg.bk} "
            f"for group={cfg.group_size}"
        )
    if cfg.assume_even_n and cfg.n % cfg.bn:
        raise ValueError(
            f"Mojo W4A16 policy {path} sets assume_even_N=True but "
            f"N={cfg.n} is not divisible by tile_N={cfg.bn}"
        )


@lru_cache(maxsize=128)
def _load_policy_config(
    path: str,
    *,
    device_name: str,
    dtype: str,
    op_name: str,
    n: int,
    k: int,
    group_size: int,
) -> dict[int, MojoRunConfig]:
    path_obj = Path(path)
    payload = json.loads(path_obj.read_text())
    metadata = payload.get("metadata", {})
    expected = {
        "device_name": device_name,
        "dtype": dtype,
        "op": op_name,
        "n": n,
        "k": k,
        "group": group_size,
    }
    for key, expected_value in expected.items():
        if key in metadata and metadata[key] != expected_value:
            raise ValueError(
                f"Mojo W4A16 policy metadata mismatch in {path_obj}: "
                f"{key}={metadata[key]!r}, expected {expected_value!r}"
            )

    defaults = payload.get("defaults", {})
    configs_payload = payload.get("configs", {})
    configs: dict[int, MojoRunConfig] = {}
    for m_key, item in configs_payload.items():
        data = {
            "m": int(m_key),
            "n": n,
            "k": k,
            "group": group_size,
            **item,
        }
        cfg = MojoRunConfig.from_policy_dict(data, defaults)
        if cfg.variant not in VARIANT_SPECS:
            supported = ", ".join(sorted(VARIANT_SPECS))
            raise ValueError(
                f"unsupported Mojo W4A16 variant {cfg.variant!r} in {path}; "
                f"supported variants: {supported}"
            )
        _validate_policy_config(path_obj, cfg)
        configs[int(m_key)] = cfg
    return configs


def _generic_safe_config(m: int, n: int, k: int, group_size: int) -> MojoRunConfig:
    if m <= 8:
        bm = 32
    elif m <= 64:
        bm = 64
    else:
        bm = 128
    bn = 64 if n <= 12288 else 128
    bk = min(64, group_size)
    while bk > 16 and group_size % bk:
        bk //= 2
    return MojoRunConfig(
        m=_compile_m_for_runtime_m(m, bm),
        n=n,
        k=k,
        group_size=group_size,
        variant="ring_bonly",
        bm=bm,
        bn=bn,
        bk=bk,
        warps_m=4 if bm >= 64 else 2,
        warps_n=1,
        ring_producer_warps=2,
        ring_stages=2,
        qpack_k_vector_width=4,
        smem_pad=8,
    )


def _select_policy_bucket(
    m: int,
    n: int,
    k: int,
    group_size: int,
    dtype: str,
) -> MojoRunConfig:
    device_name = _policy_device_name()
    searched_paths = _policy_candidate_paths(
        n=n,
        k=k,
        group_size=group_size,
        dtype=dtype,
        op_name=POLICY_OP_NAME,
    )
    candidates: dict[int, MojoRunConfig] | None = None
    for path in searched_paths:
        if path.exists():
            logger.info_once(
                "Using configuration from %s for Mojo W4A16 kernel.",
                path,
                scope="global",
            )
            candidates = _load_policy_config(
                str(path),
                device_name=device_name,
                dtype=dtype,
                op_name=POLICY_OP_NAME,
                n=n,
                k=k,
                group_size=group_size,
            )
            break

    if not candidates:
        cfg = _generic_safe_config(m, n, k, group_size)
        _warn_missing_policy_config(
            reason="shape_family",
            m=m,
            n=n,
            k=k,
            group_size=group_size,
            dtype=dtype,
            device_name=device_name,
            op_name=POLICY_OP_NAME,
            fallback=cfg,
            searched_paths=searched_paths,
        )
        return cfg

    above = [bucket_m for bucket_m in candidates if bucket_m >= m]
    if above:
        bucket_m = min(above)
        bucket = candidates[bucket_m]
        compiled_m = _compile_m_for_runtime_m(m, bucket.bm)
        return replace(bucket, m=compiled_m)

    # If the policy does not cover this row count, compile a conservative bucket.
    max_policy_m = max(candidates)
    base = candidates[max_policy_m]
    compiled_m = _compile_m_for_runtime_m(m, base.bm)
    cfg = replace(base, m=compiled_m)
    _warn_missing_policy_config(
        reason="row_bucket",
        m=m,
        n=n,
        k=k,
        group_size=group_size,
        dtype=dtype,
        device_name=device_name,
        op_name=POLICY_OP_NAME,
        fallback=cfg,
        max_policy_m=max_policy_m,
        searched_paths=searched_paths,
    )
    return cfg


def _with_runtime_options(
    cfg: MojoRunConfig,
    *,
    use_fp16: bool,
    zp_bias: int,
    use_qzeros: bool,
    zero_offset: int,
) -> MojoRunConfig:
    return replace(
        cfg,
        use_fp16=use_fp16,
        zp_bias=zp_bias,
        use_qzeros=use_qzeros,
        zero_offset=zero_offset if use_qzeros else 0,
    )


def _validate_inputs(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    group_size: int,
    zp_bias: int,
) -> tuple[int, int, int]:
    if a.ndim != 2 or qweight.ndim != 2 or scales.ndim != 2:
        raise ValueError("expected a [M,K], qweight [K,N/8], scales [K/G,N]")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Mojo W4A16 requires fp16 or bf16 activations")
    if scales.dtype != a.dtype:
        raise TypeError("Mojo W4A16 scales must match activation dtype")
    if qweight.dtype != torch.int32:
        raise TypeError("Mojo W4A16 qweight must be int32")
    if not a.is_cuda or not qweight.is_cuda or not scales.is_cuda:
        raise ValueError("Mojo W4A16 tensors must be CUDA/ROCm tensors")
    if (
        not a.is_contiguous()
        or not qweight.is_contiguous()
        or not scales.is_contiguous()
    ):
        raise ValueError("Mojo W4A16 tensors must be contiguous")
    if zp_bias < 0 or zp_bias > 15:
        raise ValueError(
            f"Mojo W4A16 zp_bias must be in uint4 range [0, 15], got {zp_bias}"
        )
    if qzeros is None and zp_bias != 8:
        raise NotImplementedError(
            "Mojo W4A16 constant zero-point path currently supports zp_bias=8; "
            "non-8 zero points require explicit qzeros"
        )

    m, k = a.shape
    n = qweight.shape[1] * 8
    if qweight.shape != (k, n // 8):
        raise ValueError(f"qweight shape mismatch: {tuple(qweight.shape)}")
    if group_size <= 0:
        group_size = k
    if k % group_size:
        raise ValueError(f"K={k} must be divisible by group_size={group_size}")
    if scales.shape != (k // group_size, n):
        raise ValueError(f"scales shape mismatch: {tuple(scales.shape)}")
    if qzeros is not None:
        if not qzeros.is_cuda or not qzeros.is_contiguous():
            raise ValueError("Mojo W4A16 qzeros must be contiguous CUDA tensor")
        if qzeros.dtype != torch.int32:
            raise TypeError("Mojo W4A16 qzeros must be int32")
        if qzeros.shape != (k // group_size, n // 8):
            raise ValueError(f"qzeros shape mismatch: {tuple(qzeros.shape)}")
    return int(m), int(n), int(k)


def _config_defaults_source(cfg: MojoRunConfig) -> str:
    return f"""comptime DEFAULT_MAX_M = {cfg.m}
comptime DEFAULT_MAX_N = {cfg.n}
comptime DEFAULT_MAX_K = {cfg.k}
comptime DEFAULT_USE_FP16 = {_bool(cfg.use_fp16)}

comptime DEFAULT_BM = {cfg.bm}
comptime DEFAULT_BN = {cfg.bn}
comptime DEFAULT_BK = {cfg.bk}
comptime DEFAULT_GROUP_SIZE = {cfg.group_size}
comptime DEFAULT_ZP_BIAS = {cfg.zp_bias}
comptime DEFAULT_USE_QZEROS = {_bool(cfg.use_qzeros)}
comptime DEFAULT_ZERO_OFFSET = {cfg.zero_offset}
comptime DEFAULT_SMEM_PAD = {cfg.smem_pad}

comptime DEFAULT_RING_PRODUCER_WARPS = {cfg.ring_producer_warps}
comptime DEFAULT_NUM_STAGES = {cfg.ring_stages}
comptime DEFAULT_BLOCK_SWIZZLE_SCALE = {cfg.block_swizzle_scale}
comptime DEFAULT_GROUP_SIZE_M = {cfg.group_size_m}
comptime DEFAULT_USE_LDS_SWIZZLE = {_bool(cfg.use_lds_swizzle)}
comptime DEFAULT_RING_STARTUP_ALL_WARPS = {_bool(cfg.ring_startup_all_warps)}
comptime DEFAULT_LOAD_B_BY_QPACK = {_bool(cfg.load_b_by_qpack)}
comptime DEFAULT_QPACK_K_VECTOR_WIDTH = {cfg.qpack_k_vector_width}
comptime DEFAULT_DEQUANT_B_IN_BF16 = {_bool(cfg.dequant_b_in_bf16)}
comptime DEFAULT_SCALE_AFTER_GROUP = {_bool(cfg.scale_after_group)}
comptime DEFAULT_ASSUME_EVEN_K = {_bool(cfg.k % cfg.bk == 0)}
# vLLM reuses a compiled M bucket for smaller runtime M values during CUDA graph
# capture. Keep the row guard enabled so padded buckets never write past output.
comptime DEFAULT_ASSUME_EVEN_MN = False
comptime DEFAULT_ASSUME_EVEN_N = {_bool(cfg.assume_even_n)}
comptime DEFAULT_KERNEL_VARIANT = "{cfg.variant}"

comptime DEFAULT_SPLITK_THREADS = {cfg.splitk_threads}
comptime DEFAULT_SPLITK_BLOCK_K = {cfg.splitk_block_k}
comptime DEFAULT_SPLITK_ROWS_PER_CTA = {cfg.splitk_rows_per_cta}

comptime DEFAULT_WARPS_M = {cfg.warps_m}
comptime DEFAULT_WARPS_N = {cfg.warps_n}

comptime DEFAULT_WARMUP_ITERS = 0
comptime DEFAULT_BENCH_ITERS = 1
"""


def _variant_spec(variant: str) -> VariantSpec:
    try:
        return VARIANT_SPECS[variant]
    except KeyError as exc:
        supported = ", ".join(sorted(VARIANT_SPECS))
        raise ValueError(
            f"unsupported Mojo W4A16 variant {variant!r}; "
            f"supported variants: {supported}"
        ) from exc


def _common_mojo_files() -> dict[str, Path]:
    return {
        "common.mojo": MOJO_SOURCE_DIR / "common.mojo",
        "kernel_common.mojo": MOJO_SOURCE_DIR / "kernel_common.mojo",
        "ring_buffer.mojo": MOJO_SOURCE_DIR / "ring_buffer.mojo",
    }


def _mojo_file_manifest(spec: VariantSpec) -> dict[str, Path]:
    files = _common_mojo_files()
    if spec.needs_partial:
        files["splitk_reduce.mojo"] = MOJO_SOURCE_DIR / "splitk_reduce.mojo"
    files[f"{spec.generated_module}.mojo"] = spec.source_path
    return files


def _source_text_for_generated(path: Path) -> str:
    text = path.read_text()
    for module in (
        "config_defaults",
        "common",
        "kernel_common",
        "ring_buffer",
        "splitk_reduce",
    ):
        text = text.replace(f"from mojo.{module} import", f"from {module} import")
    return text


def _reduce_template_parts(spec: VariantSpec) -> dict[str, str]:
    if not spec.needs_partial:
        return {
            "reduce_import": "",
            "reduce_kernel_def": "",
            "reduce_field": "",
            "reduce_init": "",
        }
    return {
        "reduce_import": (
            "from splitk_reduce import gemm_w4a16_kpacked_splitk_reduce_kernel"
        ),
        "reduce_kernel_def": (
            "comptime reduce_kernel = "
            "gemm_w4a16_kpacked_splitk_reduce_kernel[\n"
            "    PartialLayout, CLayout\n"
            "]"
        ),
        "reduce_field": (
            "    var compiled_reduce_kernel: type_of(\n"
            "        DeviceContext().compile_function[reduce_kernel]()\n"
            "    )"
        ),
        "reduce_init": (
            "        self.compiled_reduce_kernel = "
            "self.ctx.compile_function[reduce_kernel]()"
        ),
    }


def _indent_mojo(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else "" for line in text.splitlines())


def _launch_body(spec: VariantSpec, *, spaces: int) -> str:
    if spec.needs_partial:
        body = f"""stream.enqueue_function(
    self_ptr[].compiled_kernel,
    partial_tt,
    a_tt,
    qweight_kpacked_tt,
    qzeros_tt,
    scales_tt,
    m,
    n,
    k,
    grid_dim={spec.split_grid},
    block_dim={spec.split_block},
)
stream.enqueue_function(
    self_ptr[].compiled_reduce_kernel,
    c_tt,
    partial_read_tt,
    m,
    n,
    k,
    grid_dim=(
        ceildiv(ceildiv(n, 4), SPLITK_THREADS),
        ceildiv(m, SPLITK_ROWS_PER_CTA),
    ),
    block_dim=(SPLITK_THREADS, 1),
)"""
    else:
        body = f"""stream.enqueue_function(
    self_ptr[].compiled_kernel,
    c_tt,
    a_tt,
    qweight_tt,
    qzeros_tt,
    scales_tt,
    m,
    n,
    k,
    grid_dim=(ceildiv(n, BN), ceildiv(m, BM)),
    block_dim={spec.block_dim},
)"""
    return _indent_mojo(body, spaces)


def _source_digest(cfg: MojoRunConfig) -> str:
    spec = _variant_spec(cfg.variant)
    digest = hashlib.sha256()
    payload = json.dumps(
        {
            "config": cfg.__dict__,
            "generator_cache_version": GENERATOR_CACHE_VERSION,
            "variant": {
                "module": spec.generated_module,
                "kernel": spec.kernel_symbol,
                "active_kernel": spec.active_kernel,
                "needs_partial": spec.needs_partial,
                "block_dim": spec.block_dim,
                "split_grid": spec.split_grid,
                "split_block": spec.split_block,
                "qweight_layout": spec.qweight_layout,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest.update(payload.encode())
    source_files = {
        "direct_extension_template.mojo": TEMPLATE_PATH,
        **_mojo_file_manifest(spec),
    }
    for name, path in source_files.items():
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _materialize_package(cfg: MojoRunConfig) -> tuple[Path, str]:
    spec = _variant_spec(cfg.variant)
    digest = _source_digest(cfg)
    module_name = f"vllm_mojo_gemm_w4a16_{digest[:16]}"
    package_dir = _module_cache_root() / cfg.key / digest[:16]
    package_dir.mkdir(parents=True, exist_ok=True)

    reduce_parts = _reduce_template_parts(spec)
    module_source = (
        TEMPLATE_PATH.read_text()
        .replace("__REDUCE_IMPORT__", reduce_parts["reduce_import"])
        .replace(
            "__KERNEL_IMPORT__",
            f"from {spec.generated_module} import {spec.kernel_symbol}",
        )
        .replace("__ACTIVE_KERNEL__", spec.active_kernel)
        .replace("__REDUCE_KERNEL_DEF__", reduce_parts["reduce_kernel_def"])
        .replace("__NEED_PARTIAL__", _bool(spec.needs_partial))
        .replace("__REDUCE_FIELD__", reduce_parts["reduce_field"])
        .replace("__REDUCE_INIT__", reduce_parts["reduce_init"])
        .replace("__BLOCK_DIM__", spec.block_dim)
        .replace("__SPLIT_GRID__", spec.split_grid)
        .replace("__SPLIT_BLOCK__", spec.split_block)
        .replace("__PY_LAUNCH_BODY__", _launch_body(spec, spaces=8))
        .replace("__NATIVE_LAUNCH_BODY__", _launch_body(spec, spaces=4))
        .replace("__MODULE_NAME__", module_name)
    )
    files = {
        f"{module_name}.mojo": module_source,
        "config_defaults.mojo": _config_defaults_source(cfg),
    }
    for name, path in _mojo_file_manifest(spec).items():
        files[name] = _source_text_for_generated(path)
    for name, text in files.items():
        path = package_dir / name
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
    return package_dir, module_name


def _build_extension(
    package_dir: Path, module_name: str, timeout: float = 300.0
) -> Path:
    so_path = package_dir / f"{module_name}.so"
    if so_path.exists():
        return so_path
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Mojo W4A16 extension was not built before graph capture")
    tmp_path = package_dir / f"{module_name}.tmp.{os.getpid()}.so"
    if tmp_path.exists():
        tmp_path.unlink()
    cmd = [
        mojo_bin(),
        "build",
        "--emit",
        "shared-lib",
        f"{module_name}.mojo",
        "-o",
        str(tmp_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=package_dir,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Mojo W4A16 build failed for {module_name}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    os.replace(tmp_path, so_path)
    return so_path


@lru_cache(maxsize=128)
def _load_extension(cfg: MojoRunConfig) -> ModuleType:
    _mem_debug(f"before load_extension cfg={cfg.key}")
    package_dir, module_name = _materialize_package(cfg)
    so_path = _build_extension(package_dir, module_name)
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Mojo W4A16 extension {so_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _mem_debug(f"after load_extension cfg={cfg.key}")
    return module


def _runner_key(module: ModuleType, tensor: torch.Tensor) -> tuple[str, int]:
    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    return module.__name__, int(device_index)


def _partial_scratch_numel(cfg: MojoRunConfig) -> int:
    spec = _variant_spec(cfg.variant)
    if not spec.needs_partial:
        return 1
    splitk_partitions = (cfg.k + cfg.splitk_block_k - 1) // cfg.splitk_block_k
    return splitk_partitions * cfg.m * cfg.n


def _get_partial_scratch(
    cfg: MojoRunConfig,
    module: ModuleType,
    tensor: torch.Tensor,
) -> torch.Tensor:
    key = _runner_key(module, tensor)
    required_numel = _partial_scratch_numel(cfg)
    scratch = _PARTIAL_SCRATCH_CACHE.get(key)
    if (
        scratch is not None
        and scratch.numel() >= required_numel
        and scratch.device == tensor.device
    ):
        return scratch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Mojo W4A16 partial scratch was not prepared before graph capture"
        )
    _mem_debug(
        f"before partial_scratch_alloc module={module.__name__} numel={required_numel}",
        tensor,
    )
    scratch = torch.empty(required_numel, device=tensor.device, dtype=torch.float32)
    _PARTIAL_SCRATCH_CACHE[key] = scratch
    _mem_debug(
        f"after partial_scratch_alloc module={module.__name__} numel={required_numel}",
        tensor,
    )
    return scratch


def _get_runner(module: ModuleType, tensor: torch.Tensor):
    key = _runner_key(module, tensor)
    runner = _RUNNERS.get(key)
    if runner is not None:
        _debug_log(
            f"runner hit module={module.__name__} device={key[1]} "
            f"{_stream_debug(tensor)}"
        )
        return runner
    if torch.cuda.is_current_stream_capturing():
        _debug_log(
            f"runner miss during capture module={module.__name__} "
            f"device={key[1]} prepared_devices={[k[1] for k in _RUNNERS]}"
        )
        raise RuntimeError("Mojo W4A16 runner was not prepared before graph capture")
    _debug_log(
        f"runner create module={module.__name__} device={key[1]} "
        f"{_stream_debug(tensor)}"
    )
    _mem_debug(f"before runner_create module={module.__name__}", tensor)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise RuntimeError(f"Mojo W4A16 module {module.__name__} has no file")
    runner = _load_hip_launch_shim().create_runner(
        str(module_file), _current_stream_ptr(tensor)
    )
    _RUNNERS[key] = runner
    _mem_debug(f"after runner_create module={module.__name__}", tensor)
    return runner


def _select_config_for_tensors(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    group_size: int,
    zp_bias: int,
    use_v2_format: bool,
) -> MojoRunConfig:
    m, n, k = _validate_inputs(a, qweight, scales, qzeros, group_size, zp_bias)
    effective_group_size = group_size if group_size > 0 else k
    cfg = _select_policy_bucket(
        m,
        n,
        k,
        effective_group_size,
        _policy_dtype(a.dtype),
    )
    symmetric_qzeros = qzeros is None or (
        zp_bias == 8
        and _qzeros_are_symmetric(qzeros, zp_bias=zp_bias, use_v2_format=use_v2_format)
    )
    use_qzeros = qzeros is not None and not symmetric_qzeros
    return _with_runtime_options(
        cfg,
        use_fp16=a.dtype == torch.float16,
        zp_bias=zp_bias,
        use_qzeros=use_qzeros,
        zero_offset=0 if use_v2_format else 1,
    )


def prepare_mojo_w4a16_gemm(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    group_size: int,
    zp_bias: int = 8,
    use_v2_format: bool = False,
) -> None:
    cfg = _select_config_for_tensors(
        a, qweight, scales, qzeros, group_size, zp_bias, use_v2_format
    )
    _debug_log(f"prepare cfg={cfg.key} {_stream_debug(a)}")
    _mem_debug(f"prepare start cfg={cfg.key}", a)
    _ = _get_qweight_kpacked(cfg, qweight)
    module = _load_extension(cfg)
    _ = _get_partial_scratch(cfg, module, a)
    _ = _get_runner(module, a)
    _mem_debug(f"prepare done cfg={cfg.key}", a)


def _mojo_w4a16_gemm_out_impl(
    out: torch.Tensor,
    a: torch.Tensor,
    qweight: torch.Tensor,
    qweight_kpacked_or_dummy: torch.Tensor,
    scales: torch.Tensor,
    qzeros_or_dummy: torch.Tensor,
    group_size: int,
    zp_bias: int,
    has_qzeros: bool,
    use_v2_format: bool,
) -> None:
    qzeros = qzeros_or_dummy if has_qzeros else None
    cfg = _select_config_for_tensors(
        a, qweight, scales, qzeros, group_size, zp_bias, use_v2_format
    )
    qweight_kpacked_or_dummy = _resolve_qweight_kpacked(
        cfg, qweight, qweight_kpacked_or_dummy
    )
    _debug_log(
        "launch "
        f"a={tuple(a.shape)} qweight={tuple(qweight.shape)} "
        f"qweight_kpacked={tuple(qweight_kpacked_or_dummy.shape)} "
        f"scales={tuple(scales.shape)} "
        f"qzeros={tuple(qzeros.shape) if qzeros is not None else None} "
        f"out={tuple(out.shape)} dtype={a.dtype} "
        f"group={group_size} zp_bias={zp_bias} "
        f"has_qzeros={has_qzeros} use_v2={use_v2_format} cfg={cfg.key} "
        f"{_stream_debug(a)}"
    )
    module = _load_extension(cfg)
    partial_scratch = _get_partial_scratch(cfg, module, a)
    runner = _get_runner(module, a)
    _mem_debug(f"before native launch cfg={cfg.key}", a)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise RuntimeError(f"Mojo W4A16 module {module.__name__} has no file")
    _debug_log(f"native launch module={module.__name__} runner=0x{int(runner):x}")
    _load_hip_launch_shim().launch(
        str(module_file),
        int(runner),
        _current_stream_ptr(a),
        out,
        a,
        qweight,
        qweight_kpacked_or_dummy,
        qzeros_or_dummy,
        scales,
        partial_scratch,
    )
    _mem_debug(f"after native launch cfg={cfg.key}", a)


def _mojo_w4a16_gemm_out_fake(
    out: torch.Tensor,
    a: torch.Tensor,
    qweight: torch.Tensor,
    qweight_kpacked_or_dummy: torch.Tensor,
    scales: torch.Tensor,
    qzeros_or_dummy: torch.Tensor,
    group_size: int,
    zp_bias: int,
    has_qzeros: bool,
    use_v2_format: bool,
) -> None:
    return None


def _mojo_w4a16_gemm_impl(
    a: torch.Tensor,
    qweight: torch.Tensor,
    qweight_kpacked_or_dummy: torch.Tensor,
    scales: torch.Tensor,
    qzeros_or_dummy: torch.Tensor,
    group_size: int,
    zp_bias: int,
    has_qzeros: bool,
    use_v2_format: bool,
) -> torch.Tensor:
    qzeros = qzeros_or_dummy if has_qzeros else None
    cfg = _select_config_for_tensors(
        a, qweight, scales, qzeros, group_size, zp_bias, use_v2_format
    )
    out_storage = torch.empty(
        (max(a.shape[0], cfg.m), qweight.shape[1] * 8),
        dtype=a.dtype,
        device=a.device,
    )
    _mojo_w4a16_gemm_out_impl(
        out_storage,
        a,
        qweight,
        qweight_kpacked_or_dummy,
        scales,
        qzeros_or_dummy,
        group_size,
        zp_bias,
        has_qzeros,
        use_v2_format,
    )
    return out_storage[: a.shape[0]]


def _mojo_w4a16_gemm_fake(
    a: torch.Tensor,
    qweight: torch.Tensor,
    qweight_kpacked_or_dummy: torch.Tensor,
    scales: torch.Tensor,
    qzeros_or_dummy: torch.Tensor,
    group_size: int,
    zp_bias: int,
    has_qzeros: bool,
    use_v2_format: bool,
) -> torch.Tensor:
    return torch.empty(
        (a.shape[0], qweight.shape[1] * 8),
        dtype=a.dtype,
        device=a.device,
    )


def pt2_tags() -> tuple[torch.Tag, ...]:
    return _pt2_tags()


def select_config_for_tensors(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    group_size: int,
    zp_bias: int,
    use_v2_format: bool,
) -> MojoRunConfig:
    return _select_config_for_tensors(
        a, qweight, scales, qzeros, group_size, zp_bias, use_v2_format
    )


def get_qweight_kpacked(cfg: MojoRunConfig, qweight: torch.Tensor) -> torch.Tensor:
    return _get_qweight_kpacked(cfg, qweight)


def make_qweight_kpacked(qweight: torch.Tensor) -> torch.Tensor:
    return _make_qweight_kpacked(qweight)


def cache_prepared_qweight_kpacked(
    qweight: torch.Tensor, qweight_kpacked: torch.Tensor
) -> None:
    _cache_prepared_qweight_kpacked(qweight, qweight_kpacked)


mojo_w4a16_gemm_out_impl = _mojo_w4a16_gemm_out_impl
mojo_w4a16_gemm_out_fake = _mojo_w4a16_gemm_out_fake
mojo_w4a16_gemm_impl = _mojo_w4a16_gemm_impl
mojo_w4a16_gemm_fake = _mojo_w4a16_gemm_fake

__all__ = [
    "MojoRunConfig",
    "cache_prepared_qweight_kpacked",
    "check_dependencies",
    "get_qweight_kpacked",
    "make_qweight_kpacked",
    "mojo_w4a16_gemm_fake",
    "mojo_w4a16_gemm_impl",
    "mojo_w4a16_gemm_out_fake",
    "mojo_w4a16_gemm_out_impl",
    "prepare_mojo_w4a16_gemm",
    "pt2_tags",
    "select_config_for_tensors",
]
