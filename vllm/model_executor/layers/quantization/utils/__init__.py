# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .kernel_utils import is_compute_capability_supported
from .layer_utils import replace_parameter, update_tensor_inplace

__all__ = [
    "update_tensor_inplace",
    "replace_parameter",
    "is_compute_capability_supported",
]
