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

import torch

from vllm.utils.platform_utils import num_compute_units

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
    from humming.transform import (
        prepare_and_transform_humming_tensors as prepare_and_transform_humming_tensors,
    )
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
    "prepare_and_transform_humming_tensors": (
        "humming.transform:prepare_and_transform_humming_tensors"
    ),
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


def create_humming_locks(device: torch.device | str) -> torch.Tensor:
    device = torch.device(device)
    device_index = device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    return torch.zeros(
        num_compute_units(device_index) * 8,
        dtype=torch.int32,
        device=device,
    )


_patched = False


def _patch_humming() -> None:
    """Make humming's per-call ``DeviceInfo(device)`` lookups traceable.

    The forward path (``LayerConfig.check_device``, PDL heuristics) builds a
    ``DeviceInfo`` on every call, which breaks vLLM's fullgraph compile:
    Dynamo recurses forever in ``DeviceInfo.__getattr__``
    (vllm-project/humming#147) and, with that fixed (#148), still cannot
    trace the extension's ``_DeviceInfo`` constructor. Hand out one cached
    instance per device instead; ``prepare_layer_config`` creates it eagerly
    during weight processing.

    TODO: remove after bumping humming to a release with
    vllm-project/humming#149.
    """
    global _patched
    if _patched:
        return
    _patched = True

    import humming.config.config
    import humming.transform
    import humming.tune
    import humming.tune.process_input
    import humming.utils.smem
    from humming.device import DeviceInfo, get_device_index

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        return getattr(self._get_value(), name)

    DeviceInfo.__getattr__ = __getattr__

    device_infos: dict[int, DeviceInfo] = {}

    def cached_device_info(index: int | torch.device | None = None) -> DeviceInfo:
        if isinstance(index, torch.device) and index.type != "cuda":
            return DeviceInfo(index)
        device_index = get_device_index(index)
        if device_index not in device_infos:
            device_infos[device_index] = DeviceInfo(device_index)
        return device_infos[device_index]

    for module in (
        humming.config.config,
        humming.transform,
        humming.tune,
        humming.tune.process_input,
        humming.utils.smem,
    ):
        if getattr(module, "DeviceInfo", None) is DeviceInfo:
            module.DeviceInfo = cached_device_info


def __getattr__(name: str) -> Any:
    spec = _EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module 'vllm.utils.humming' has no attribute {name!r}")
    _patch_humming()
    if ":" in spec:
        mod_path, attr = spec.split(":", 1)
        obj = getattr(importlib.import_module(mod_path), attr)
    else:
        obj = importlib.import_module(spec)
    globals()[name] = obj
    return obj


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
