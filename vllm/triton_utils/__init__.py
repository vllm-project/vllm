# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.triton_utils.importing import (
    HAS_TRITON,
    TritonLanguagePlaceholder,
    TritonPlaceholder,
)

if TYPE_CHECKING or HAS_TRITON:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.language.core import _aggregate as aggregate  # noqa: E501
else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()
    tldevice = TritonLanguagePlaceholder()
    gluon = TritonLanguagePlaceholder()
    gl = TritonLanguagePlaceholder()
    aggregate = TritonLanguagePlaceholder()

from vllm.triton_utils.tensor_descriptor import use_tensor_descriptor

# Replace Autotuner.run so it always picks the first valid configuration, to
# eliminate autotune variability. Installed here rather than in
# vllm.env_override because importing this package resolves current_platform,
# which must not happen during `import vllm`.
if envs.VLLM_TRITON_FORCE_FIRST_CONFIG:
    from vllm.triton_utils.force_first_config import (
        install as _install_force_first_config,
    )

    _install_force_first_config()

LOG2E = 1.4426950408889634
LOGE2 = 0.6931471805599453


def fp32_dot_precision(dtype: torch.dtype) -> str | None:
    """``tl.dot`` ``input_precision`` for ``dtype`` operands. Float32 dots get
    float32 accuracy unless TF32 is allowed by ``torch.set_float32_matmul_precision``
    (``VLLM_FLOAT32_MATMUL_PRECISION``): 3xTF32 on CUDA, which matches IEEE
    accuracy on tensor cores, and IEEE elsewhere. Other dtypes keep Triton's
    default."""
    if dtype != torch.float32 or torch.get_float32_matmul_precision() != "highest":
        return None
    return "tf32x3" if current_platform.is_cuda() else "ieee"


__all__ = [
    "HAS_TRITON",
    "triton",
    "tl",
    "tldevice",
    "LOG2E",
    "LOGE2",
    "gluon",
    "gl",
    "aggregate",
    "fp32_dot_precision",
    "use_tensor_descriptor",
]
