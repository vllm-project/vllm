# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings

from vllm.utils.flashinfer_utils import *

warnings.warn(
    "flashinfer is deprecated, use vllm.utils.flashinfer_utils instead",
    DeprecationWarning,
    stacklevel=2,
)
