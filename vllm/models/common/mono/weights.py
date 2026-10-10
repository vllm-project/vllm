# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shape checks for the tensors a MonoKernel reads by pointer."""

from __future__ import annotations


def check_tensors(owner, tag: str, want: dict) -> None:
    """Assert each named tensor of ``owner`` matches ``(shape, dtype)``.

    A MonoKernel reads its weights through raw pointers with unbounded buffer
    loads, so a layout mismatch reads garbage rather than faulting. These checks
    run once, at build time, and are the only place that layout is stated.

    Args:
        owner: Object holding the tensors as attributes.
        tag: Name used in the failure message.
        want: ``name -> (shape, dtype)`` the kernels read.

    Raises:
        AssertionError: On the first tensor whose shape, dtype, or contiguity
            differs.

    """
    for name, (shape, dtype) in want.items():
        t = getattr(owner, name)
        assert tuple(t.shape) == shape and t.dtype == dtype and t.is_contiguous(), (
            f"{tag} {name}: {tuple(t.shape)} {t.dtype}, want {shape} {dtype}"
        )
