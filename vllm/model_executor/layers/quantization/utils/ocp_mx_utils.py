# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum
from typing import Any

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp6E2M3Dynamic,
    kMxfp6E2M3Static,
    kMxfp6E3M2Dynamic,
    kMxfp6E3M2Static,
    kMxfp8Dynamic,
    kMxfp8Static,
)

logger = init_logger(__name__)

OCP_MX_BLOCK_SIZE = 32

_WEIGHT_QUANT_KEY_MAP: dict[str, QuantKey] = {
    "mxfp4": kMxfp4Static,
    "mxfp6_e3m2": kMxfp6E3M2Static,
    "mxfp6_e2m3": kMxfp6E2M3Static,
    "mxfp8_e4m3": kMxfp8Static,
}

_ACTIVATION_QUANT_KEY_MAP: dict[str, QuantKey] = {
    "mxfp4": kMxfp4Dynamic,
    "mxfp6_e3m2": kMxfp6E3M2Dynamic,
    "mxfp6_e2m3": kMxfp6E2M3Dynamic,
    "mxfp8_e4m3": kMxfp8Dynamic,
}

# Unpacked OCP MX dtypes: one byte per element, no `packed_factor`.
OCP_MX_UNPACKED_DTYPES = {"mxfp8_e4m3"}

OCP_MX_DTYPES = {
    "mxfp4",
    "mxfp6_e3m2",
    "mxfp6_e2m3",
    "mxfp8_e4m3",
    "mxfp8_e5m2",
    "mxint8",
}


def ocp_mx_weight_dtype_and_rows(
    weight_quant: dict[str, Any] | None,
) -> tuple[str, int] | None:
    """Normalize a Quark weight quant config to ``(mx_dtype, scale_block_rows)``.

    Quark spells OCP MX weights two ways. The canonical one is 1-D per-group:
    ``qscheme="per_group"``, ``group_size=32``, ``scale_format="e8m0"``. MXFP8
    checkpoints may instead use a 2-D per-block spelling
    (``qscheme="per_block"``, ``block_size=[R, 32]``,
    ``scale_type="float8_e8m0fnu"``), which carries one scale per ``R`` weight
    rows rather than one per row. ``scale_block_rows`` is that ``R``; it is 1
    for the canonical spelling, where the two layouts coincide.

    Returns ``None`` when the config is not an OCP MX weight quantization.
    """
    if not isinstance(weight_quant, dict):
        return None

    dtype = weight_quant.get("dtype")
    if not isinstance(dtype, str):
        return None
    mx_dtype = dtype.replace("fp", "mxfp")
    if mx_dtype not in _WEIGHT_QUANT_KEY_MAP:
        return None

    qscheme = weight_quant.get("qscheme")
    if qscheme == "per_group":
        if weight_quant.get("group_size") != OCP_MX_BLOCK_SIZE:
            return None
        if weight_quant.get("scale_format") != "e8m0":
            return None
        return mx_dtype, 1

    if qscheme == "per_block":
        # Only the unpacked dtypes are accepted here: expanding a 2-D scale
        # over a sub-byte packed weight has no checkpoint to validate against.
        if mx_dtype not in OCP_MX_UNPACKED_DTYPES:
            return None
        block_size = list(weight_quant.get("block_size") or [])
        if len(block_size) != 2 or block_size[1] != OCP_MX_BLOCK_SIZE:
            return None
        if weight_quant.get("scale_type") != "float8_e8m0fnu":
            return None
        if weight_quant.get("symmetric") is not True or weight_quant.get("is_dynamic"):
            return None
        return mx_dtype, block_size[0]

    return None


class OCP_MX_Scheme(str, Enum):
    w_mxfp4 = "w_mxfp4"
    w_mxfp4_a_mxfp4 = "w_mxfp4_a_mxfp4"
    w_mxfp4_a_mxfp6_e3m2 = "w_mxfp4_a_mxfp6_e3m2"
    w_mxfp4_a_mxfp6_e2m3 = "w_mxfp4_a_mxfp6_e2m3"
    w_mxfp4_a_fp8 = "w_mxfp4_a_fp8"
    w_mxfp6_e3m2 = "w_mxfp6_e3m2"
    w_mxfp6_e3m2_a_mxfp6_e3m2 = "w_mxfp6_e3m2_a_mxfp6_e3m2"
    w_mxfp6_e3m2_a_fp8 = "w_mxfp6_e3m2_a_fp8"
    w_mxfp6_e2m3 = "w_mxfp6_e2m3"
    w_mxfp6_e2m3_a_mxfp6_e2m3 = "w_mxfp6_e2m3_a_mxfp6_e2m3"
    w_mxfp6_e2m3_a_fp8 = "w_mxfp6_e2m3_a_fp8"

    @classmethod
    def from_quant_dtype(cls, input_dtype: str | None, weight_dtype: str | None):
        if input_dtype not in OCP_MX_DTYPES and weight_dtype not in OCP_MX_DTYPES:
            return None
        elif input_dtype is None and weight_dtype == "mxfp4":
            return cls.w_mxfp4
        elif input_dtype is None and weight_dtype == "mxfp6_e3m2":
            return cls.w_mxfp6_e3m2
        elif input_dtype is None and weight_dtype == "mxfp6_e2m3":
            return cls.w_mxfp6_e2m3
        elif input_dtype == "mxfp4" and weight_dtype == "mxfp4":
            return cls.w_mxfp4_a_mxfp4
        elif input_dtype == "mxfp6_e3m2" and weight_dtype == "mxfp4":
            return cls.w_mxfp4_a_mxfp6_e3m2
        elif input_dtype == "mxfp6_e2m3" and weight_dtype == "mxfp4":
            return cls.w_mxfp4_a_mxfp6_e2m3
        elif input_dtype == "fp8" and weight_dtype == "mxfp4":
            return cls.w_mxfp4_a_fp8
        elif input_dtype == "mxfp6_e3m2" and weight_dtype == "mxfp6_e3m2":
            return cls.w_mxfp6_e3m2_a_mxfp6_e3m2
        elif input_dtype == "fp8" and weight_dtype == "mxfp6_e3m2":
            return cls.w_mxfp6_e3m2_a_fp8
        elif input_dtype == "mxfp6_e2m3" and weight_dtype == "mxfp6_e2m3":
            return cls.w_mxfp6_e2m3_a_mxfp6_e2m3
        elif input_dtype == "fp8" and weight_dtype == "mxfp6_e2m3":
            return cls.w_mxfp6_e2m3_a_fp8
        else:
            logger.warning(
                "input_dtype='%s' and"
                " weight_dtype='%s' is not supported "
                "in OCP_MX_Scheme at the moment.",
                input_dtype,
                weight_dtype,
            )
            return None
