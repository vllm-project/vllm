# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidates that consume canonical NVFP4 weights without preparing a copy."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch

_CANONICAL_LAYOUT = "nvfp4_k_major_sf128x4"


@dataclass(frozen=True)
class DynamicNvFp4Backend:
    activation_bits: Literal[4, 16]
    weight_layout: str
    is_supported: Callable[[int], bool]
    # Alpha includes the activation dequantization scale only for W4A4.
    run: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]


def _flashinfer_supported(cc: int, *, api: str, backend: str) -> bool:
    try:
        import flashinfer

        return getattr(flashinfer, api).is_backend_supported(backend, cc)
    except (ImportError, AttributeError, KeyError, ValueError):
        return False


def _flashinfer_w4a4(a, weight, a_scale, weight_scale, alpha, *, backend):
    import flashinfer

    return flashinfer.mm_fp4(
        a,
        weight.T,
        a_scale.view(torch.uint8),
        weight_scale.view(torch.uint8).T,
        alpha,
        out_dtype=torch.bfloat16,
        backend=backend,
        use_nvfp4=True,
        use_8x4_sf_layout=False,
        enable_pdl=True,
    )


def _flashinfer_w4a16(a, weight, a_scale, weight_scale, alpha):
    import flashinfer

    return flashinfer.mm_bf16_fp4(
        a,
        weight,
        weight_scale,
        alpha,
        backend="cute-dsl-native",
        enable_pdl=True,
    )


_BACKENDS = {
    name: DynamicNvFp4Backend(
        activation_bits=4,
        weight_layout=_CANONICAL_LAYOUT,
        is_supported=partial(_flashinfer_supported, api="mm_fp4", backend=backend),
        run=partial(_flashinfer_w4a4, backend=backend),
    )
    for name, backend in (
        ("flashinfer_cutlass", "cutlass"),
        ("flashinfer_b12x", "b12x"),
        ("flashinfer_cudnn", "cudnn"),
        ("flashinfer_cutedsl", "cute-dsl"),
    )
}
_BACKENDS["flashinfer_cutedsl_native"] = DynamicNvFp4Backend(
    activation_bits=16,
    weight_layout=_CANONICAL_LAYOUT,
    is_supported=partial(
        _flashinfer_supported, api="mm_bf16_fp4", backend="cute-dsl-native"
    ),
    run=_flashinfer_w4a16,
)


def get_dynamic_backend(name: str, activation_bits: int) -> DynamicNvFp4Backend:
    candidate = _BACKENDS.get(name)
    if candidate is None:
        raise ValueError(f"Unknown canonical NVFP4 candidate {name!r}")
    if candidate.weight_layout != _CANONICAL_LAYOUT:
        raise ValueError(f"NVFP4 candidate {name!r} requires a different weight layout")
    if candidate.activation_bits != activation_bits:
        raise ValueError(
            f"NVFP4 candidate {name!r} does not implement W4A{activation_bits}"
        )
    return candidate
