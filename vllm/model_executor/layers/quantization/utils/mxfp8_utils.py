# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger
logger = init_logger(__name__)


def mxfp8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

    try:
        from flashinfer import mxfp8_quantize
    except ImportError as err:
        raise ImportError("The package `flashinfer` is required to do "
                          "MX-FP8 quantization. Please install it with" \
                          "`pip install flashinfer`") from err

    return mxfp8_quantize(x, is_sf_swizzled_layout=False)
