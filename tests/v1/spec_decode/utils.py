# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import Any

from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    SpeculatorConfig,
    get_speculator_config_class,
)


class _FakeSpeculativeConfig(SimpleNamespace):
    def __getattr__(self, name: str) -> Any:
        # Like SpeculativeConfig's shortcuts to speculator config properties.
        if name != "speculator_config" and "speculator_config" in vars(self):
            return getattr(self.speculator_config, name)
        raise AttributeError(name)


def fake_speculative_config(
    speculator_config_cls: type[SpeculatorConfig] | None = None, **fields: Any
) -> Any:
    """A `SimpleNamespace` standing in for a `SpeculativeConfig` with `fields`.

    Its `speculator_config` is of `speculator_config_cls`, or by default of the
    class a real config with the same `method` would use (ignoring the draft
    architecture). It reads `fields` lazily, as for a real config. Attributes
    that aren't fields are taken from the speculator config, like
    SpeculativeConfig's shortcuts to its properties.
    """
    spec = _FakeSpeculativeConfig(**{"custom_method": None, **fields})
    if speculator_config_cls is None:
        probe = SimpleNamespace(
            method=fields["method"],
            custom_method=None,
            draft_model_config=SimpleNamespace(architectures=[]),
            use_gemma4_mtp=lambda: False,
            use_multi_module_mtp=lambda: False,
        )
        speculator_config_cls = get_speculator_config_class(probe)  # type: ignore[arg-type]
    spec.speculator_config = speculator_config_cls(spec)
    return spec


# SpeculativeConfig's shortcuts to speculator config properties.
_SPECULATOR_CONFIG_SHORTCUTS = (
    "uses_target_hidden_states",
    "uses_aux_hidden_states",
    "uses_separate_workspace_lane",
)


def attach_speculator_config(
    spec: Any, speculator_config_cls: type[SpeculatorConfig]
) -> Any:
    """Give a mock standing in for a `SpeculativeConfig` a real
    `speculator_config` of `speculator_config_cls` (and the shortcuts to its
    properties), which reads the mock's attributes lazily."""
    spec.speculator_config = speculator_config_cls(spec)
    spec.method = speculator_config_cls.base_method
    spec.custom_method = None
    for name in _SPECULATOR_CONFIG_SHORTCUTS:
        setattr(spec, name, getattr(spec.speculator_config, name))
    return spec
