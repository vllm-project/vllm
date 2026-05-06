# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# base.py is shadowed by this package directory. Load it explicitly so that
# consumers importing from `kernels.linear.base` continue to find the symbols
# that originally lived there.
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "vllm.model_executor.kernels.linear._base_legacy",
    pathlib.Path(__file__).parent.parent / "base.py",
)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

MMLinearLayerConfig = _mod.MMLinearLayerConfig
Params = _mod.Params
FP8Params = _mod.FP8Params
Int8Params = _mod.Int8Params
MMLinearKernel = _mod.MMLinearKernel
