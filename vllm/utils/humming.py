# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy facade for the optional ``humming`` package.

vLLM code should import humming symbols from here so that ``import humming``
(which has import-time side effects) is deferred until first use. Add new
symbols to both the ``TYPE_CHECKING`` imports and ``_EXPORTS``, using
``"module.path:attr"`` or ``"module.path"`` for a whole-module re-export.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from humming import dtypes as dtypes
    from humming.config import GemmType as GemmType
    from humming.config import InputQuantizationMode as InputQuantizationMode
    from humming.config import LayerConfig as LayerConfig
    from humming.config import MmaType as MmaType
    from humming.config import WeightScale2Type as WeightScale2Type
    from humming.config import WeightScaleType as WeightScaleType
    from humming.dtypes import DataType as DataType
    from humming.forward import humming_forward as humming_forward
    from humming.forward import may_process_input as may_process_input
    from humming.forward import may_quant_input as may_quant_input
    from humming.ops import process_input as process_input
    from humming.schema import AWQWeightSchema as AWQWeightSchema
    from humming.schema import BaseInputSchema as BaseInputSchema
    from humming.schema import BaseWeightSchema as BaseWeightSchema
    from humming.schema import BitnetWeightSchema as BitnetWeightSchema
    from humming.schema import (
        CompressedTensorsInputSchema as CompressedTensorsInputSchema,
    )
    from humming.schema import (
        CompressedTensorsWeightSchema as CompressedTensorsWeightSchema,
    )
    from humming.schema import Fp8InputSchema as Fp8InputSchema
    from humming.schema import GptOssMxfp4WeightSchema as GptOssMxfp4WeightSchema
    from humming.schema import GPTQWeightSchema as GPTQWeightSchema
    from humming.schema import HummingInputSchema as HummingInputSchema
    from humming.schema import HummingWeightSchema as HummingWeightSchema
    from humming.schema import Mxfp4WeightSchema as Mxfp4WeightSchema
    from humming.schema.fp8 import Fp8WeightSchema as Fp8WeightSchema
    from humming.schema.modelopt import (
        ModeloptMxfp8WeightSchema as ModeloptMxfp8WeightSchema,
    )
    from humming.schema.modelopt import (
        ModeloptNvfp4InputSchema as ModeloptNvfp4InputSchema,
    )
    from humming.schema.modelopt import (
        ModeloptNvfp4WeightSchema as ModeloptNvfp4WeightSchema,
    )
    from humming.transform import prepare_layer_config as prepare_layer_config
    from humming.transform import transform_humming_tensors as transform_humming_tensors
    from humming.tune import get_heuristics_config as get_heuristics_config
    from humming.utils.weight import quantize_weight as quantize_weight

_EXPORTS: dict[str, str] = {
    "dtypes": "humming.dtypes",
    "DataType": "humming.dtypes:DataType",
    "GemmType": "humming.config:GemmType",
    "LayerConfig": "humming.config:LayerConfig",
    "WeightScaleType": "humming.config:WeightScaleType",
    "WeightScale2Type": "humming.config:WeightScale2Type",
    "InputQuantizationMode": "humming.config:InputQuantizationMode",
    "MmaType": "humming.config:MmaType",
    "humming_forward": "humming.forward:humming_forward",
    "may_process_input": "humming.forward:may_process_input",
    "process_input": "humming.ops:process_input",
    "may_quant_input": "humming.forward:may_quant_input",
    "prepare_layer_config": "humming.transform:prepare_layer_config",
    "transform_humming_tensors": "humming.transform:transform_humming_tensors",
    "get_heuristics_config": "humming.tune:get_heuristics_config",
    "BaseInputSchema": "humming.schema:BaseInputSchema",
    "BaseWeightSchema": "humming.schema:BaseWeightSchema",
    "HummingInputSchema": "humming.schema:HummingInputSchema",
    "HummingWeightSchema": "humming.schema:HummingWeightSchema",
    "quantize_weight": "humming.utils.weight:quantize_weight",
    "AWQWeightSchema": "humming.schema:AWQWeightSchema",
    "BitnetWeightSchema": "humming.schema:BitnetWeightSchema",
    "ModeloptMxfp8WeightSchema": "humming.schema.modelopt:ModeloptMxfp8WeightSchema",
    "ModeloptNvfp4InputSchema": "humming.schema.modelopt:ModeloptNvfp4InputSchema",
    "ModeloptNvfp4WeightSchema": "humming.schema.modelopt:ModeloptNvfp4WeightSchema",
    "CompressedTensorsInputSchema": "humming.schema:CompressedTensorsInputSchema",
    "CompressedTensorsWeightSchema": "humming.schema:CompressedTensorsWeightSchema",
    "Fp8InputSchema": "humming.schema:Fp8InputSchema",
    "Fp8WeightSchema": "humming.schema.fp8:Fp8WeightSchema",
    "Mxfp4WeightSchema": "humming.schema:Mxfp4WeightSchema",
    "GptOssMxfp4WeightSchema": "humming.schema:GptOssMxfp4WeightSchema",
    "GPTQWeightSchema": "humming.schema:GPTQWeightSchema",
}


def __getattr__(name: str) -> Any:
    spec = _EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module 'vllm.utils.humming' has no attribute {name!r}")
    if ":" in spec:
        mod_path, attr = spec.split(":", 1)
        obj = getattr(importlib.import_module(mod_path), attr)
    else:
        obj = importlib.import_module(spec)
    globals()[name] = obj
    return obj


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
