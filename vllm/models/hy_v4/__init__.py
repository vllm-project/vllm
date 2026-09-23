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

NVIDIA and ROCm are supported, the latter through `amd/`. The port also drops
the reference implementation's HPC/TPCP fusion paths, which depend on
infrastructure that does not exist in this tree.
"""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if current_platform.is_xpu():
    raise NotImplementedError("hy_v4 does not yet support XPU.")

# The NVIDIA branch is the static default that type-checkers see; the ROCm
# branch overrides it at runtime (kept type-compatible via type: ignore).
if TYPE_CHECKING or not current_platform.is_rocm():
    from .nvidia.model import HYV4ForCausalLM
else:
    from .amd.model import HYV4ForCausalLM  # type: ignore[assignment]

# The draft head has no ROCm-specific implementation yet.
from .nvidia.mtp import HYV4MTP  # noqa: E402

__all__ = [
    "HYV4ForCausalLM",
    "HYV4MTP",
]
