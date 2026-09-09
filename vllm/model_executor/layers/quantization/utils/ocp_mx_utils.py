# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp6E2M3Dynamic,
    kMxfp6E2M3Static,
    kMxfp6E3M2Dynamic,
    kMxfp6E3M2Static,
)

OCP_MX_BLOCK_SIZE = 32

_WEIGHT_QUANT_KEY_MAP: dict[QuantKey, str] = {
    kMxfp4Static: "mxfp4",
    kMxfp6E3M2Static: "mxfp6_e3m2",
    kMxfp6E2M3Static: "mxfp6_e2m3",
}

_ACTIVATION_QUANT_KEY_MAP: dict[QuantKey, str] = {
    kFp8StaticTensorSym: "fp8",
    kMxfp4Dynamic: "mxfp4",
    kMxfp6E3M2Dynamic: "mxfp6_e3m2",
    kMxfp6E2M3Dynamic: "mxfp6_e2m3",
}

_WEIGHT_QUANT_DTYPE_MAP = {value: key for key, value in _WEIGHT_QUANT_KEY_MAP.items()}
_ACTIVATION_QUANT_DTYPE_MAP = {
    value: key for key, value in _ACTIVATION_QUANT_KEY_MAP.items()
}

OCP_MX_DTYPES = {
    "mxfp4",
    "mxfp6_e3m2",
    "mxfp6_e2m3",
    "mxfp8_e4m3",
    "mxfp8_e5m2",
    "mxint8",
}
