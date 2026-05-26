# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .layer_utils import replace_parameter, update_tensor_inplace
from .quant_fusion import QuantizedActivation

__all__ = ["update_tensor_inplace", "replace_parameter", "QuantizedActivation"]
