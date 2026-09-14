# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY V4 (``hy_v4``) model — hardware-isolated entry point.

HY V4 combines three architectural pieces:

- **iHC** (independent Hyper-Connections): the single residual stream is
  replaced by ``hc_mult`` parallel residual channels, gated per sub-block.
- **MLA + lightning indexer**: multi-head latent attention with an optional
  DSA-style sparse top-k selection, plus an output gate and a learnable sink.
- **MoE**: sigmoid-routed experts with a clamped SwiGLU and shared experts.

The package is organized like `vllm.models.deepseek_v32`: this module is the
only public entry point and dispatches on the current platform, so registry
entries never reach into a platform subpackage.

Only NVIDIA is supported for now. The port also drops the reference
implementation's HPC/TPCP fusion paths, which depend on infrastructure that
does not exist in this tree.
"""

from vllm.platforms import current_platform

if current_platform.is_rocm():
    raise NotImplementedError("hy_v4 does not yet support ROCm.")
elif current_platform.is_xpu():
    raise NotImplementedError("hy_v4 does not yet support XPU.")
else:
    # Covers Blackwell (sm100) and all other CUDA devices.
    from .nvidia.model import HYV4ForCausalLM
    from .nvidia.mtp import HYV4MTP

__all__ = [
    "HYV4ForCausalLM",
    "HYV4MTP",
]
