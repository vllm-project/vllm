# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in adapter for the fused HIP DeepSeek-V4 compressors (gfx950 / CDNA4).

Bridges the model's ``compress_norm_rope_store_fn(...)`` call site
(``compressor.py``) to the AOT ``torch.ops._rocm_C.dsv4_*`` ops, dispatching by
``(head_dim, compress_ratio)``:

    (512, 4)   -> dsv4_csa_compress
    (512, 128) -> dsv4_hca_compress
    (128, 4)   -> dsv4_indexer_compress   (FP8 or MXFP4 via use_fp4_cache)

These kernels read a **bf16** state cache and add APE in-kernel, emit the legacy
paged UE8M0 / packed layout (``kv_cache`` dtype uint8), and are CDNA4-only — they
are built into ``_rocm_C`` only when gfx950 is among the target archs
(VLLM_ROCM_GFX950). ``hip_compressor_supported`` enforces every precondition.
The default enabled mode includes CSA/HCA; HIP indexer exists for explicit
opt-in and focused testing, but defaults to Triton.
"""

from typing import Any, Literal, cast

import torch

from vllm import envs

CompressorKind = Literal["csa", "hca", "indexer"]

HIP_COMPRESSOR_SHAPES_BY_KIND: dict[CompressorKind, tuple[int, int]] = {
    "csa": (512, 4),
    "hca": (512, 128),
    "indexer": (128, 4),
}
DEFAULT_HIP_COMPRESSOR_KINDS: frozenset[CompressorKind] = frozenset({"csa", "hca"})
DEFAULT_HIP_COMPRESSOR_SHAPES = frozenset({(512, 4), (512, 128)})
ALL_HIP_COMPRESSOR_SHAPES = DEFAULT_HIP_COMPRESSOR_SHAPES | frozenset({(128, 4)})
SUPPORTED_SHAPES = ALL_HIP_COMPRESSOR_SHAPES


def parse_hip_compressor_modes(value: str | None) -> frozenset[CompressorKind]:
    """Parse VLLM_ROCM_DSV4_HIP_COMPRESSOR as bool-compatible modes."""
    if value is None:
        return frozenset()

    normalized = value.strip().lower()
    if normalized in ("", "0", "false"):
        return frozenset()
    if normalized in ("1", "true"):
        return DEFAULT_HIP_COMPRESSOR_KINDS

    allowed = set(HIP_COMPRESSOR_SHAPES_BY_KIND)
    modes = {part.strip() for part in normalized.split(",")}
    if not modes or "" in modes or not modes <= allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(
            "VLLM_ROCM_DSV4_HIP_COMPRESSOR must be one of 0/false, "
            "1/true, or a comma-separated list containing: "
            f"{allowed_values}."
        )
    return frozenset(cast(CompressorKind, mode) for mode in modes)


def selected_hip_compressor_kinds() -> frozenset[CompressorKind]:
    return parse_hip_compressor_modes(envs.VLLM_ROCM_DSV4_HIP_COMPRESSOR)


def selected_hip_compressor_shapes() -> frozenset[tuple[int, int]]:
    return frozenset(
        HIP_COMPRESSOR_SHAPES_BY_KIND[kind] for kind in selected_hip_compressor_kinds()
    )


def classify_compressor(head_dim: int, compress_ratio: int) -> CompressorKind | None:
    for kind, shape in HIP_COMPRESSOR_SHAPES_BY_KIND.items():
        if shape == (head_dim, compress_ratio):
            return kind
    return None


def hip_compressor_selected(head_dim: int, compress_ratio: int) -> bool:
    kind = classify_compressor(head_dim, compress_ratio)
    return kind in selected_hip_compressor_kinds()


def hip_compressor_runtime_available() -> bool:
    """Return whether this ROCm process can use gfx950-only compressor kernels."""
    try:
        from vllm.platforms.rocm import on_gfx950

        return on_gfx950()
    except Exception:
        return False


def _aot_op(head_dim: int, compress_ratio: int):
    """The torch.ops._rocm_C op for this shape, or None if not registered.

    The op is registered only when _rocm_C was built with VLLM_ROCM_GFX950.
    """
    rocm_C = getattr(torch.ops, "_rocm_C", None)
    if rocm_C is None:
        return None
    if head_dim == 512 and compress_ratio == 4:
        return getattr(rocm_C, "dsv4_csa_compress", None)
    if head_dim == 512 and compress_ratio == 128:
        return getattr(rocm_C, "dsv4_hca_compress", None)
    if head_dim == 128 and compress_ratio == 4:
        return getattr(rocm_C, "dsv4_indexer_compress", None)
    return None


def hip_compressor_supported(
    head_dim: int,
    compress_ratio: int,
    kv_cache: torch.Tensor,
    allowed_shapes: frozenset[tuple[int, int]] | None = None,
) -> bool:
    """True iff the fused HIP compressor can serve this configuration.

    Checks (all required): the caller-selected HIP mode includes this shape,
    gfx950 (CDNA4), the bf16 state-cache path (so APE is added in-kernel), the
    legacy paged uint8 cache layout (the kernels do not emit FlashInfer
    full-cache rows), and that the matching AOT op is actually registered (i.e.
    _rocm_C was built with VLLM_ROCM_GFX950). The caller has already checked the
    opt-in flag + is_rocm().
    """
    if allowed_shapes is None:
        allowed_shapes = DEFAULT_HIP_COMPRESSOR_SHAPES
    if (head_dim, compress_ratio) not in allowed_shapes:
        return False
    if kv_cache.dtype != torch.uint8:
        return False
    if not hip_compressor_runtime_available():
        return False
    return _aot_op(head_dim, compress_ratio) is not None


def compress_norm_rope_store_hip(
    *,
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    pdl_kwargs: dict,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
    ape: torch.Tensor,
    use_bf16_state_cache: bool = True,
    hca_plan_scratch: torch.Tensor | None = None,
    hca_counter_scratch: torch.Tensor | None = None,
    **_ignored: Any,
) -> None:
    """Dispatch one fused HIP compressor launch (signature mirrors the model's
    ``compress_norm_rope_store_fn``). Assumes ``hip_compressor_supported`` already
    returned True for this configuration."""
    if num_actual == 0:
        return

    op = _aot_op(head_dim, compress_ratio)
    if op is None:
        raise RuntimeError(
            f"HIP compressor op unavailable for (head_dim={head_dim}, "
            f"compress_ratio={compress_ratio}); was _rocm_C built with "
            f"VLLM_ROCM_GFX950?"
        )

    # Positional args, matching the TORCH_LIBRARY schema in
    # csrc/rocm/torch_bindings.cpp. kv_cache is written in place.
    args = [
        state_cache,
        num_actual,
        ape,  # RAW APE; expanded / added on-device
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_size,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        kv_cache,
        k_cache_metadata.slot_mapping,
        kv_cache.shape[1],  # kv_cache_block_size
        scale_dim,
    ]
    if head_dim == 128 and compress_ratio == 4:
        args.append(use_fp4_cache)
    elif head_dim == 512 and compress_ratio == 128:
        if hca_plan_scratch is None or hca_counter_scratch is None:
            raise RuntimeError(
                "HCA HIP compressor requires reusable plan/counter scratch buffers."
            )
        args.extend([hca_plan_scratch, hca_counter_scratch])
    op(*args)
