# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared zentorch stubs / fixtures for CPU quantization tests.

Each test module that needs the stub ops imports the fixture by name into
its own module namespace::

    from tests.quantization._zentorch_helpers import zentorch_ops_mock  # noqa: F401

Pytest then discovers ``zentorch_ops_mock`` as a fixture when a test in that
module declares it as a parameter.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from types import SimpleNamespace

import pytest
import torch

# ---------------------------------------------------------------------------
# Stub op implementations
# ---------------------------------------------------------------------------


def _stub_dynamic_qlinear(
    inp,
    weight,
    weight_scales,
    bias=None,
    zentorch_op_name: str = "zentorch::zentorch_dynamic_qlinear",
):
    """Shape-preserving stub for ``zentorch_dynamic_qlinear``."""
    out_features = weight.shape[0]
    out = torch.zeros(
        inp.shape[:-1] + (out_features,), dtype=inp.dtype, device=inp.device
    )
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


def _stub_woq_repack_weight(weight: torch.Tensor) -> torch.Tensor:
    """Stub for ``zentorch_woq_repack_weight``: identity (the production op
    re-tiles for AVX-512 layout; tests only need a tensor of consistent
    shape and dtype)."""
    return weight.clone().contiguous()


def _stub_woq_linear(inp, weight_packed, weight_scale, weight_zp, bias=None):
    """Shape-preserving stub for ``zentorch_woq_linear``."""
    # ``weight_packed`` here is the zentorch-packed transposed view; we infer
    # the output features from the scale tensor (transposed in production
    # code so the second dim is N).
    out_features = weight_scale.shape[1]
    out = torch.zeros(
        inp.shape[:-1] + (out_features,), dtype=inp.dtype, device=inp.device
    )
    if bias is not None:
        out = out + bias
    return out


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


# (op_name, schema_inside_parens, impl)
_OPS_TO_REGISTER: tuple[tuple[str, str, Callable], ...] = (
    (
        "zentorch_dynamic_qlinear",
        "Tensor input, Tensor weight, Tensor weight_scales, "
        "Tensor? bias=None, "
        'str zentorch_op_name="zentorch::zentorch_dynamic_qlinear"',
        _stub_dynamic_qlinear,
    ),
    (
        "zentorch_woq_repack_weight",
        "Tensor weight",
        _stub_woq_repack_weight,
    ),
    (
        "zentorch_woq_linear",
        "Tensor input, Tensor weight_packed, Tensor weight_scale, "
        "Tensor? weight_zero_point, Tensor? bias=None",
        _stub_woq_linear,
    ),
)


def _missing(ops: Iterable[tuple[str, str, Callable]]):
    return [
        (name, schema, impl)
        for name, schema, impl in ops
        if not (hasattr(torch.ops, "zentorch") and hasattr(torch.ops.zentorch, name))
    ]


@pytest.fixture
def zentorch_ops_mock():
    """Register stub ``torch.ops.zentorch.*`` ops for the duration of a test.

    On dev/CI machines that already have a real ``zentorch`` build, the
    fixture is a strict no-op for every op already present (no double
    registration), so production tests can also be run against a real build.
    """
    missing = _missing(_OPS_TO_REGISTER)
    lib_def = None
    lib_impl = None
    if missing:
        lib_def = torch.library.Library("zentorch", "FRAGMENT")
        lib_impl = torch.library.Library("zentorch", "IMPL", "CPU")
        for name, schema, impl in missing:
            lib_def.define(f"{name}({schema}) -> Tensor")
            lib_impl.impl(name, impl)

    yield SimpleNamespace(
        dynamic_qlinear=_stub_dynamic_qlinear,
        woq_repack_weight=_stub_woq_repack_weight,
        woq_linear=_stub_woq_linear,
    )

    if lib_impl is not None:
        lib_impl._destroy()
    if lib_def is not None:
        lib_def._destroy()
