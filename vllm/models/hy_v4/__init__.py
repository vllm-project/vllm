# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY V4 (`hy_v4`) model — hardware-isolated entry point.

HY V4 combines three architectural pieces:

- **iHC** (independent Hyper-Connections): the single residual stream is
  replaced by `hc_mult` parallel residual channels, gated per sub-block.
- **MLA + lightning indexer**: multi-head latent attention with an optional
  DSA-style sparse top-k selection, plus an output gate and a learnable sink.
- **MoE**: sigmoid-routed experts with a clamped SwiGLU and shared experts.

The package is organized like `vllm.models.deepseek_v32`: this module is the
only public entry point and dispatches on the current platform, so registry
entries never reach into a platform subpackage.

NVIDIA and ROCm are supported through `nvidia/` and `amd/`, with
vendor-neutral code in `common/`. The port also drops
the reference implementation's HPC/TPCP fusion paths, which depend on
infrastructure that does not exist in this tree.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .common.vl_model import HYV4VLForConditionalGeneration
    from .nvidia.dfly import HYV4DFlyForCausalLM
    from .nvidia.dfly_speculator import HYV4DFlySpeculator
    from .nvidia.model import HYV4ForCausalLM
    from .nvidia.mtp import HYV4MTP

__all__ = [
    "HYV4ForCausalLM",
    "HYV4MTP",
    "HYV4DFlyForCausalLM",
    "HYV4DFlySpeculator",
    "HYV4VLForConditionalGeneration",
]


def __getattr__(name: str) -> Any:
    """Load model code only when requested, keeping config-only imports light."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        raise NotImplementedError("hy_v4 does not yet support XPU.")
    platform = "amd" if current_platform.is_rocm() else "nvidia"
    modules = {
        "HYV4ForCausalLM": f".{platform}.model",
        # The MTP head runs on ROCm through the NVIDIA layers it shares.
        "HYV4MTP": ".nvidia.mtp",
        "HYV4DFlyForCausalLM": f".{platform}.dfly",
        "HYV4DFlySpeculator": f".{platform}.dfly_speculator",
        "HYV4VLForConditionalGeneration": (
            ".common.vl_model" if current_platform.is_cuda_alike() else ".vl_stub"
        ),
    }
    model_cls = getattr(import_module(modules[name], __name__), name)
    globals()[name] = model_cls
    return model_cls
