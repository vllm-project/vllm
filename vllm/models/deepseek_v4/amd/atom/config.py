# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Single-node config shim for the ported DeepSeek V4 attention.

ATOM's real ``atom.config`` cascades into the plugin / distributed / quant-parser
subsystems (``atom.plugin``, ``atom.utils.distributed``, ``atom.quant_spec``
parsers). None of that is needed for a single-node attention port, so this shim
provides only the names the vendored ``model_ops`` files import:

  * ``QuantType``            — re-exported from ``aiter``.
  * ``LayerQuantConfig`` /
    ``should_skip_online_quant`` — re-exported from the vendored ``quant_spec``.
  * ``QuantizationConfig``   — model-wide quant container (no-quant / BF16
    default; the attention layers are typically constructed with
    ``quant_config=None`` in single-node BF16 mode).
  * ``Config`` / ``CompilationConfig`` / ``CompilationLevel`` — lightweight
    engine-config stands-in exposing ``static_forward_context`` (used by V4
    modules to register themselves for the custom-op dispatch) and
    ``torch_dtype``.
  * ``get_current_atom_config`` / ``set_current_atom_config`` — the global
    engine-config singleton accessor.
"""

from typing import Optional

import torch
from aiter import QuantType  # noqa: F401  (re-exported)

from vllm.models.deepseek_v4.amd.atom.quant_spec import (  # noqa: F401
    LayerQuantConfig,
    should_skip_online_quant,
)


class QuantizationConfig:
    """Model-wide quantization configuration (single-node shim).

    Mirrors the public surface the vendored ``linear`` / ``layernorm`` read
    (``quant_type`` / ``quant_dtype`` / ``is_dynamic`` / ``online_quant`` /
    ``get_layer_quant_config``). Constructed with ``config=None`` it is a
    no-quant BF16 default; a real ATOM engine would populate it from the HF
    quant config, which is out of scope for the attention-only single-node port.
    """

    def __init__(self, config=None, online_quant_config: Optional[dict] = None):
        self.torch_dtype = getattr(config, "torch_dtype", None) or torch.bfloat16
        # ``config`` may be an HF PretrainedConfig or a dict-like; read the
        # embedded quantization_config either way.
        hqc = getattr(config, "quantization_config", None)
        if hqc is None and isinstance(config, dict):
            hqc = config.get("quantization_config")
        self.hf_quant_config = hqc
        self.global_spec = LayerQuantConfig(
            quant_type=QuantType.No, quant_dtype=self.torch_dtype
        )
        self.layer_pattern_specs: list = []
        self.exclude_layers: list = []
        self.quant_method = ""
        if isinstance(self.hf_quant_config, dict):
            self.quant_method = str(self.hf_quant_config.get("quant_method", "") or "")
            if self.quant_method:
                from vllm.models.deepseek_v4.amd.atom.quant_spec import (
                    get_quant_parser,
                )

                parsed = get_quant_parser(self.quant_method).parse(self.hf_quant_config)
                self.global_spec = parsed.global_spec
                self.layer_pattern_specs = parsed.layer_pattern_specs
                self.exclude_layers = list(parsed.exclude_layers)
        self.online_quant = False
        self.online_quant_config_raw = online_quant_config
        self.online_global_spec = LayerQuantConfig()
        self.online_layer_pattern_specs: list = []
        self.online_exclude_layers: list = []

    @property
    def global_quant_config(self) -> LayerQuantConfig:
        return self.global_spec

    @property
    def quant_type(self):
        return self.global_spec.quant_type

    @property
    def quant_dtype(self):
        return self.global_spec.quant_dtype

    @property
    def is_dynamic(self) -> bool:
        return getattr(self.global_spec, "is_dynamic", False)

    def get_layer_quant_config(
        self, layer_name: str, use_online_quant: bool = False, **_: object
    ) -> LayerQuantConfig:
        import fnmatch

        for pat in self.exclude_layers:
            if fnmatch.fnmatch(layer_name, pat) or pat in layer_name:
                return LayerQuantConfig(
                    quant_type=QuantType.No, quant_dtype=self.torch_dtype
                )
        for pat, spec in self.layer_pattern_specs:
            if fnmatch.fnmatch(layer_name, pat) or pat in layer_name:
                return spec
        return self.global_spec


class CompilationLevel:
    NO_COMPILATION = 0
    DYNAMO_AS_IS = 1
    PIECEWISE = 3


class CompilationConfig:
    """Minimal compilation config exposing the static forward-context registry.

    V4 modules register themselves via
    ``get_current_atom_config().compilation_config.static_forward_context[prefix]
    = self`` so the ``torch.ops.aiter`` dispatchers can look them up by layer
    name. Only that dict is needed here.
    """

    def __init__(self):
        self.static_forward_context: dict = {}
        self.level = CompilationLevel.NO_COMPILATION


class ParallelConfig:
    """Single-node parallel-config stand-in (no data/pipeline parallel)."""

    def __init__(self):
        self.data_parallel_size = 1
        self.data_parallel_rank = 0
        self.tensor_parallel_size = 1
        self.pipeline_parallel_size = 1


class Config:
    """Engine-config stand-in (single-node, no distributed)."""

    def __init__(self):
        self.compilation_config = CompilationConfig()
        self.parallel_config = ParallelConfig()
        self.torch_dtype = torch.bfloat16


_current_atom_config: Optional[Config] = None


def set_current_atom_config(atom_config: Config) -> None:
    global _current_atom_config
    _current_atom_config = atom_config


def get_current_atom_config() -> Config:
    global _current_atom_config
    if _current_atom_config is None:
        _current_atom_config = Config()
    return _current_atom_config
