# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whether NVFP4 MoE can pick W4A16 or W4A4 per forward.

At small M a weight-only (W4A16) expert GEMM beats a fully quantised (W4A4)
one, and at large M the FP4 tensor cores win. M varies per forward, so the
choice would ideally be made per forward rather than once per server.

It cannot be, today, and the reason is the expert weight layout rather than
the selection logic. ``convert_to_nvfp4_moe_kernel_format`` passes
``reorder_w13=use_a16`` to the B12X preparation, so the W4A16 and W4A4 paths
produce different w13 tensors from the same checkpoint. Serving both from one
layer would mean keeping both resident, and for expert weights that is the
dominant memory cost in the model.

This module states that constraint in code, so the dispatch question has one
answer that tests can pin rather than being rediscovered per backend. It is
keyed on backend *name* to stay free of an import cycle with the oracle.
"""

# Backends whose expert weight preparation is known to depend on use_a16, with
# the specific reason. Anything absent is treated as unaudited rather than
# compatible: wrongly claiming sharability produces silently wrong numerics.
_LAYOUT_DIVERGES: dict[str, str] = {
    "B12X": (
        "prepare_nvfp4_moe_layer_for_b12x is called with reorder_w13=use_a16, "
        "so the W4A16 and W4A4 paths produce different w13 tensors"
    ),
}


def nvfp4_moe_weight_layout_key(backend_name: str, use_a16: bool) -> tuple[str, str]:
    """Identify the on-device expert weight layout a backend prepares.

    Two configurations can share one set of expert weights only if their keys
    are equal.

    Args:
        backend_name: ``NvFp4MoeBackend`` member name.
        use_a16: Whether the weight-only path is in use.

    Returns:
        A tuple identifying the layout. Unaudited backends get a key that
        varies with ``use_a16``, so they are reported as not sharable.
    """
    if backend_name in _LAYOUT_DIVERGES:
        return (backend_name, "w13_reordered" if use_a16 else "w13_checkpoint")
    return (backend_name, f"unaudited_a16={use_a16}")


def can_dispatch_a16_per_forward(backend_name: str) -> tuple[bool, str | None]:
    """Whether one loaded layer can serve both W4A16 and W4A4 forwards.

    Args:
        backend_name: ``NvFp4MoeBackend`` member name.

    Returns:
        ``(False, reason)`` for every backend today. The reason names the
        preparation step responsible, so it can be acted on rather than
        merely reported.
    """
    if nvfp4_moe_weight_layout_key(backend_name, True) == nvfp4_moe_weight_layout_key(
        backend_name, False
    ):
        return True, None
    reason = _LAYOUT_DIVERGES.get(
        backend_name,
        "its expert weight preparation has not been audited for W4A16 and "
        "W4A4 layout equality",
    )
    return False, f"{backend_name} cannot dispatch per forward: {reason}"
