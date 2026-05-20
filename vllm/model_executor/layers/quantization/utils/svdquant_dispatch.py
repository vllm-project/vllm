# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware gate for SVDQuant W4A4.

The only in-tree backend is the `nunchaku` pip package, covering
consumer NVIDIA GPUs (Turing SM_75 through consumer Blackwell SM_120).
Hopper SM_90 is intentionally unsupported: the kernel families
nunchaku targets are PTX-MMA on older arches, and SM_90's tensor unit
shape has no validated SVDQuant kernel.

Datacenter Blackwell SM_100/103 (B200/GB300) is out of scope here —
the planned datacenter path is to be hosted in FlashInfer so SGLang
and vLLM can share the same primitive.
"""

from typing import Literal

from vllm.platforms import current_platform
from vllm.utils.nunchaku import has_nunchaku_w4a4

SVDQuantPrecision = Literal["int4", "nvfp4"]


def assert_svdquant_supported(precision: SVDQuantPrecision) -> None:
    """Raise if the active platform cannot run SVDQuant at this precision."""
    if not current_platform.is_cuda():
        raise RuntimeError(
            f"SVDQuant has no available backend on platform "
            f"{current_platform.device_name!r}. CUDA + nunchaku required."
        )

    cap = current_platform.get_device_capability()
    sm = f"SM_{cap.to_int()}" if cap is not None else "<unknown>"

    if current_platform.is_device_capability_family(90):
        raise RuntimeError(
            "SVDQuant W4A4 is not supported on Hopper (SM_90). Use a "
            "consumer GPU (SM_75–SM_89, SM_120) with nunchaku, or wait "
            "for the datacenter Blackwell (SM_100/103) path planned in "
            "FlashInfer."
        )

    if current_platform.is_device_capability_family(100):
        raise RuntimeError(
            f"SVDQuant on {sm} (B200/GB300) is not supported in-tree; "
            "the datacenter path is planned in FlashInfer."
        )

    if not current_platform.has_device_capability((7, 5)):
        raise RuntimeError(
            f"Unsupported CUDA compute capability for SVDQuant: {sm}"
        )

    # nvfp4 needs SM_100+ tensor units; pre-Blackwell consumer cards
    # (Turing/Ampere/Ada) cannot run it.
    if precision == "nvfp4" and not current_platform.has_device_capability(100):
        raise ValueError(
            f"NVFP4 SVDQuant requires SM_100+ or SM_120; got {sm}. "
            f"Use precision='int4'."
        )

    if not has_nunchaku_w4a4():
        # The PyPI `nunchaku` is an unrelated Bayesian library; the
        # SVDQuant kernels ship as GitHub release wheels only.
        raise ImportError(
            f"SVDQuant on {sm} requires nunchaku-ai's W4A4 wheels from "
            "https://github.com/nunchaku-ai/nunchaku/releases "
            "(not `pip install nunchaku`, which is a different project)."
        )


__all__ = ["SVDQuantPrecision", "assert_svdquant_supported"]
