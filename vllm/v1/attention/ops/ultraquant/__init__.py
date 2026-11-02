# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UltraQuant 4-bit KV cache: FP4 E2M1 codes + UE8M0 group-of-32 scales.

Production decode uses the FlyDSL D=256 kernel on gfx950, with Triton
unified attention as the fallback. Slot size is ``slot_size(head_dim)``
(272 B at D=256). Format helpers live in ``format``; import them from
there, not this package root.
"""

__all__ = ["ultraquant_store"]


def __getattr__(name: str):
    if name == "ultraquant_store":
        from vllm.v1.attention.ops.ultraquant.triton_store import ultraquant_store

        return ultraquant_store
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
