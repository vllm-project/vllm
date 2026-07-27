# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch checked-in Helion-generated Triton kernels."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from functools import cache
from typing import Any

import torch
from torch.library import Library

from vllm.kernels.helion_generated.manifests import (
    GENERATED_KERNEL_MANIFESTS,
    MANIFESTS,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

_SUPPORTED_PLATFORM_NAMES = {
    "nvidia_b200": "nvidia_b200",
    "nvidia_h100": "nvidia_h100",
    "nvidia_h100_80gb_hbm3": "nvidia_h100",
    "nvidia_h100_nvl": "nvidia_h100",
    "nvidia_h100_pcie": "nvidia_h100",
    "nvidia_h100_sxm5": "nvidia_h100",
}

vllm_helion_generated_lib = Library("vllm_helion_generated", "FRAGMENT")


def _canonical_device_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _runtime_platform() -> str | None:
    if not current_platform.is_cuda():
        return None
    name = _canonical_device_name(current_platform.get_device_name())
    return _SUPPORTED_PLATFORM_NAMES.get(name)


@cache
def _select_bucketed_module(
    kernel_name: str,
    platform: str | None,
    static_key: tuple[int, ...],
    num_tokens: int,
) -> str | None:
    if platform is None or num_tokens < 1:
        return None
    kernels = GENERATED_KERNEL_MANIFESTS.get(kernel_name, {}).get(platform)
    if kernels is None:
        return None
    buckets = sorted(case[-1] for case in kernels if case[:-1] == static_key)
    if not buckets:
        return None
    token_bucket = next((size for size in buckets if size >= num_tokens), buckets[-1])
    selected_case = (*static_key, token_bucket)
    return next(
        (module_path for case, module_path in kernels.items() if case == selected_case),
        None,
    )


@cache
def _select_module(
    platform: str | None,
    hidden_size: int,
    group_size: int,
    num_tokens: int,
) -> str | None:
    return _select_bucketed_module(
        "per_token_group_fp8_quant",
        platform,
        (hidden_size, group_size),
        num_tokens,
    )


@cache
def _load_launcher(module_path: str) -> Callable[..., None]:
    launcher = importlib.import_module(module_path).call
    return launcher


def _expected_scale_stride(
    num_tokens: int,
    groups_per_row: int,
    column_major: bool,
    tma_aligned: bool,
) -> tuple[int, int]:
    if not column_major:
        return (groups_per_row, 1)
    if tma_aligned:
        return (1, (num_tokens + 3) // 4 * 4)
    return (1, num_tokens)


def _eligible_module(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    column_major: bool,
    tma_aligned: bool,
) -> str | None:
    if (
        input.ndim != 2
        or input.dtype != torch.bfloat16
        or not input.is_cuda
        or not input.is_contiguous()
        or output_q.shape != input.shape
        or output_q.dtype != current_platform.fp8_dtype()
        or output_q.device != input.device
        or not output_q.is_contiguous()
        or output_s.ndim != 2
        or output_s.dtype != torch.float32
        or output_s.device != input.device
    ):
        return None

    num_tokens, hidden_size = input.shape
    if group_size < 1 or hidden_size % group_size != 0:
        return None
    groups_per_row = hidden_size // group_size
    if output_s.shape != (num_tokens, groups_per_row):
        return None
    if output_s.stride() != _expected_scale_stride(
        num_tokens, groups_per_row, column_major, tma_aligned
    ):
        return None
    return _select_module(_runtime_platform(), hidden_size, group_size, num_tokens)


def _native_fallback(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    column_major: bool,
    tma_aligned: bool,
) -> None:
    torch.ops._C.per_token_group_fp8_quant(
        input,
        output_q,
        output_s,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        column_major,
        tma_aligned,
    )


def per_token_group_fp8_quant(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    column_major: bool = False,
    tma_aligned: bool = False,
) -> None:
    module_path = _eligible_module(
        input,
        output_q,
        output_s,
        group_size,
        column_major,
        tma_aligned,
    )
    if module_path is None:
        _native_fallback(
            input,
            output_q,
            output_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0,
            column_major,
            tma_aligned,
        )
        return
    _load_launcher(module_path)(
        input,
        output_q,
        output_s,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        column_major,
        tma_aligned,
    )


def _fake_per_token_group_fp8_quant(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    column_major: bool = False,
    tma_aligned: bool = False,
) -> None:
    return None


def selected_token_buckets(token_counts: Iterable[int]) -> tuple[int, ...]:
    platform = _runtime_platform()
    kernels = MANIFESTS.get(platform or "", {})
    available = sorted({key[2] for key in kernels})
    if not available:
        return ()
    selected = {
        next((bucket for bucket in available if bucket >= count), available[-1])
        for count in token_counts
        if count > 0
    }
    return tuple(sorted(selected))


def warmup_per_token_group_fp8_quant(
    token_counts: Iterable[int],
    device: torch.device | str = "cuda",
) -> None:
    platform = _runtime_platform()
    kernels = MANIFESTS.get(platform or "", {})
    buckets = selected_token_buckets(token_counts)
    shapes = sorted({(hidden, group) for hidden, group, _ in kernels})
    if not buckets or not shapes:
        return

    fp8_dtype = current_platform.fp8_dtype()
    fp8_info = torch.finfo(fp8_dtype)
    for hidden_size, group_size in shapes:
        for num_tokens in buckets:
            input = torch.empty(
                (num_tokens, hidden_size), device=device, dtype=torch.bfloat16
            )
            output_q = torch.empty_like(input, dtype=fp8_dtype)
            output_s = torch.empty(
                (num_tokens, hidden_size // group_size),
                device=device,
                dtype=torch.float32,
            )
            for scale_ue8m0 in (False, True):
                per_token_group_fp8_quant(
                    input,
                    output_q,
                    output_s,
                    group_size,
                    1e-10,
                    fp8_info.min,
                    fp8_info.max,
                    scale_ue8m0,
                )


direct_register_custom_op(
    op_name="per_token_group_fp8_quant",
    op_func=per_token_group_fp8_quant,
    mutates_args=["output_q", "output_s"],
    fake_impl=_fake_per_token_group_fp8_quant,
    target_lib=vllm_helion_generated_lib,
)


_GENERATED_TO_NATIVE_OP = {
    "fused_qk_norm_rope": "fused_qk_norm_rope",
    "rms_norm_per_block_quant": "rms_norm_per_block_quant",
    "silu_and_mul_per_block_quant": "silu_and_mul_per_block_quant",
}


def _native_op(name: str) -> torch._ops.OpOverload:
    return getattr(torch.ops._C, name).default


def _eligible_rms_norm_per_block_quant(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    scale_ub: torch.Tensor | None,
    residual: torch.Tensor | None,
    group_size: int,
    is_scale_transposed: bool,
) -> str | None:
    if (
        input.ndim != 2
        or input.dtype != torch.bfloat16
        or not input.is_cuda
        or not input.is_contiguous()
        or result.shape != input.shape
        or result.dtype != current_platform.fp8_dtype()
        or result.device != input.device
        or not result.is_contiguous()
        or weight.shape != (input.shape[1],)
        or weight.dtype != input.dtype
        or weight.device != input.device
        or not weight.is_contiguous()
        or scale_ub is not None
        or residual is not None
        or not is_scale_transposed
        or scale.dtype != torch.float32
        or scale.device != input.device
        or group_size < 1
        or input.shape[1] % group_size != 0
        or scale.shape != (input.shape[0], input.shape[1] // group_size)
        or scale.stride() != (1, input.shape[0])
    ):
        return None
    return _select_bucketed_module(
        "rms_norm_per_block_quant",
        _runtime_platform(),
        (input.shape[1], group_size),
        input.shape[0],
    )


def rms_norm_per_block_quant(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: torch.Tensor | None,
    residual: torch.Tensor | None,
    group_size: int,
    is_scale_transposed: bool,
) -> None:
    module_path = _eligible_rms_norm_per_block_quant(
        result,
        input,
        weight,
        scale,
        scale_ub,
        residual,
        group_size,
        is_scale_transposed,
    )
    if module_path is None:
        _native_op("rms_norm_per_block_quant")(
            result,
            input,
            weight,
            scale,
            epsilon,
            scale_ub,
            residual,
            group_size,
            is_scale_transposed,
        )
        return
    _load_launcher(module_path)(
        result,
        input,
        weight,
        scale,
        epsilon,
        scale_ub,
        residual,
        group_size,
        is_scale_transposed,
    )


def _eligible_silu_and_mul_per_block_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    scale_ub: torch.Tensor | None,
    is_scale_transposed: bool,
) -> str | None:
    if (
        input.ndim != 2
        or input.shape[1] % 2 != 0
        or input.dtype != torch.bfloat16
        or not input.is_cuda
        or not input.is_contiguous()
        or out.shape != (input.shape[0], input.shape[1] // 2)
        or out.dtype != current_platform.fp8_dtype()
        or out.device != input.device
        or not out.is_contiguous()
        or scale_ub is not None
        or not is_scale_transposed
        or scales.dtype != torch.float32
        or scales.device != input.device
        or group_size < 1
        or out.shape[1] % group_size != 0
        or scales.shape != (input.shape[0], out.shape[1] // group_size)
        or scales.stride() != (1, input.shape[0])
    ):
        return None
    return _select_bucketed_module(
        "silu_and_mul_per_block_quant",
        _runtime_platform(),
        (out.shape[1], group_size),
        input.shape[0],
    )


def silu_and_mul_per_block_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    scale_ub: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
) -> None:
    module_path = _eligible_silu_and_mul_per_block_quant(
        out,
        input,
        scales,
        group_size,
        scale_ub,
        is_scale_transposed,
    )
    if module_path is None:
        _native_op("silu_and_mul_per_block_quant")(
            out,
            input,
            scales,
            group_size,
            scale_ub,
            is_scale_transposed,
        )
        return
    _load_launcher(module_path)(
        out,
        input,
        scales,
        group_size,
        scale_ub,
        is_scale_transposed,
    )


def _eligible_fused_qk_norm_rope(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor,
    forced_token_heads_per_warp: int,
) -> str | None:
    num_tokens = qkv.shape[0] if qkv.ndim == 2 else 0
    if (
        qkv.ndim != 2
        or qkv.dtype != torch.bfloat16
        or not qkv.is_cuda
        or not qkv.is_contiguous()
        or num_heads_v != num_heads_k
        or head_dim != 128
        or qkv.shape[1] != (num_heads_q + num_heads_k + num_heads_v) * head_dim
        or q_weight.shape != (head_dim,)
        or q_weight.dtype != qkv.dtype
        or q_weight.device != qkv.device
        or not q_weight.is_contiguous()
        or k_weight.shape != (head_dim,)
        or k_weight.dtype != qkv.dtype
        or k_weight.device != qkv.device
        or not k_weight.is_contiguous()
        or cos_sin_cache.shape != (40960, head_dim)
        or cos_sin_cache.dtype != qkv.dtype
        or cos_sin_cache.device != qkv.device
        or not cos_sin_cache.is_contiguous()
        or not is_neox
        or position_ids.shape != (num_tokens,)
        or position_ids.dtype != torch.int64
        or position_ids.device != qkv.device
        or not position_ids.is_contiguous()
        or forced_token_heads_per_warp != -1
    ):
        return None
    return _select_bucketed_module(
        "fused_qk_norm_rope",
        _runtime_platform(),
        (num_heads_q, num_heads_k),
        num_tokens,
    )


def fused_qk_norm_rope(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor,
    forced_token_heads_per_warp: int = -1,
) -> None:
    module_path = _eligible_fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        position_ids,
        forced_token_heads_per_warp,
    )
    if module_path is None:
        _native_op("fused_qk_norm_rope")(
            qkv,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_dim,
            eps,
            q_weight,
            k_weight,
            cos_sin_cache,
            is_neox,
            position_ids,
            forced_token_heads_per_warp,
        )
        return
    _load_launcher(module_path)(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        position_ids,
        forced_token_heads_per_warp,
    )


def _schema_tail(op: torch._ops.OpOverload) -> str:
    schema = str(op._schema)
    return schema[schema.index("(") :]


def _mutation_signature(
    op: torch._ops.OpOverload,
) -> tuple[tuple[str, bool], ...]:
    return tuple(
        (arg.name, bool(arg.alias_info and arg.alias_info.is_write))
        for arg in op._schema.arguments
    )


def _make_capture_routed_impl(
    native_op: torch._ops.OpOverload,
    generated_op: torch._ops.OpOverload,
) -> Callable[..., Any]:
    schema_args = list(generated_op._schema.arguments)
    names = [arg.name for arg in schema_args]
    defaults = {
        arg.name: arg.default_value for arg in schema_args if arg.has_default_value()
    }

    def impl(*args: object, **kwargs: object) -> Any:
        values = list(args)
        for name in names[len(args) :]:
            values.append(kwargs[name] if name in kwargs else defaults[name])
        if torch.cuda.is_current_stream_capturing():
            return generated_op(*values)
        return native_op(*values)

    return impl


def build_compiled_generated_op_map() -> dict[
    torch._ops.OpOverload, torch._ops.OpOverload
]:
    routed: dict[torch._ops.OpOverload, torch._ops.OpOverload] = {}
    platform = _runtime_platform()
    for generated_name, native_name in _GENERATED_TO_NATIVE_OP.items():
        if platform not in GENERATED_KERNEL_MANIFESTS.get(generated_name, {}):
            continue
        native_packet = getattr(torch.ops._C, native_name, None)
        generated_packet = getattr(
            torch.ops.vllm_helion_generated, generated_name, None
        )
        if native_packet is None or generated_packet is None:
            continue
        native_op = native_packet.default
        generated_op = generated_packet.default
        if _mutation_signature(native_op) != _mutation_signature(generated_op):
            raise RuntimeError(
                f"Generated op mutation mismatch for {generated_name}: "
                f"native={native_op._schema}, generated={generated_op._schema}"
            )

        routed_name = f"routed_{generated_name}"
        if not hasattr(torch.ops.vllm_helion_generated, routed_name):
            vllm_helion_generated_lib.define(routed_name + _schema_tail(generated_op))
            vllm_helion_generated_lib.impl(
                routed_name,
                _make_capture_routed_impl(native_op, generated_op),
                "CUDA",
            )
            vllm_helion_generated_lib._register_fake(
                routed_name, lambda *args, **kwargs: None
            )
        routed[native_op] = getattr(
            torch.ops.vllm_helion_generated, routed_name
        ).default
    return routed


def _fake(*args: object, **kwargs: object) -> None:
    return None


direct_register_custom_op(
    op_name="rms_norm_per_block_quant",
    op_func=rms_norm_per_block_quant,
    mutates_args=["result", "scale", "residual"],
    fake_impl=_fake,
    target_lib=vllm_helion_generated_lib,
)
direct_register_custom_op(
    op_name="silu_and_mul_per_block_quant",
    op_func=silu_and_mul_per_block_quant,
    mutates_args=["out", "scales"],
    fake_impl=_fake,
    target_lib=vllm_helion_generated_lib,
)
direct_register_custom_op(
    op_name="fused_qk_norm_rope",
    op_func=fused_qk_norm_rope,
    mutates_args=["qkv"],
    fake_impl=_fake,
    target_lib=vllm_helion_generated_lib,
)


def selected_fusion_token_buckets(
    kernel_name: str, token_counts: Iterable[int]
) -> tuple[int, ...]:
    platform = _runtime_platform()
    kernels = GENERATED_KERNEL_MANIFESTS.get(kernel_name, {}).get(platform or "", {})
    available = sorted({case[-1] for case in kernels})
    if not available:
        return ()
    return tuple(
        sorted(
            {
                next(
                    (bucket for bucket in available if bucket >= count),
                    available[-1],
                )
                for count in token_counts
                if count > 0
            }
        )
    )


def _selected_fusion_cases(
    kernel_name: str, token_counts: Iterable[int]
) -> tuple[tuple[int, ...], ...]:
    platform = _runtime_platform()
    kernels = GENERATED_KERNEL_MANIFESTS.get(kernel_name, {}).get(platform or "", {})
    counts = tuple(count for count in token_counts if count > 0)
    by_static_key: dict[tuple[int, ...], list[int]] = {}
    for case in kernels:
        by_static_key.setdefault(case[:-1], []).append(case[-1])

    selected: set[tuple[int, ...]] = set()
    for static_key, available in by_static_key.items():
        buckets = sorted(available)
        for count in counts:
            bucket = next(
                (candidate for candidate in buckets if candidate >= count),
                buckets[-1],
            )
            selected.add((*static_key, bucket))
    return tuple(sorted(selected))


def warmup_generated_fusion_kernels(
    token_counts: Iterable[int],
    device: torch.device | str = "cuda",
) -> None:
    token_counts = tuple(token_counts)
    fp8_dtype = current_platform.fp8_dtype()

    for hidden_size, group_size, num_tokens in _selected_fusion_cases(
        "rms_norm_per_block_quant", token_counts
    ):
        input = torch.empty(
            (num_tokens, hidden_size), device=device, dtype=torch.bfloat16
        )
        result = torch.empty_like(input, dtype=fp8_dtype)
        weight = torch.empty(hidden_size, device=device, dtype=input.dtype)
        groups_per_row = hidden_size // group_size
        scale = torch.empty(
            (groups_per_row, num_tokens), device=device, dtype=torch.float32
        ).t()
        rms_norm_per_block_quant(
            result,
            input,
            weight,
            scale,
            1e-6,
            None,
            None,
            group_size,
            True,
        )

    for intermediate_size, group_size, num_tokens in _selected_fusion_cases(
        "silu_and_mul_per_block_quant", token_counts
    ):
        input = torch.empty(
            (num_tokens, 2 * intermediate_size),
            device=device,
            dtype=torch.bfloat16,
        )
        out = torch.empty(
            (num_tokens, intermediate_size), device=device, dtype=fp8_dtype
        )
        groups_per_row = intermediate_size // group_size
        scales = torch.empty(
            (groups_per_row, num_tokens), device=device, dtype=torch.float32
        ).t()
        silu_and_mul_per_block_quant(out, input, scales, group_size, None, True)

    cos_sin_cache: torch.Tensor | None = None
    for q_heads, kv_heads, num_tokens in _selected_fusion_cases(
        "fused_qk_norm_rope", token_counts
    ):
        head_dim = 128
        qkv = torch.empty(
            (num_tokens, (q_heads + 2 * kv_heads) * head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        q_weight = torch.empty(head_dim, device=device, dtype=qkv.dtype)
        k_weight = torch.empty_like(q_weight)
        if cos_sin_cache is None:
            cos_sin_cache = torch.empty(
                (40960, head_dim), device=device, dtype=qkv.dtype
            )
        position_ids = torch.arange(num_tokens, device=device, dtype=torch.int64)
        fused_qk_norm_rope(
            qkv,
            q_heads,
            kv_heads,
            kv_heads,
            head_dim,
            1e-6,
            q_weight,
            k_weight,
            cos_sin_cache,
            True,
            position_ids,
        )
