# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry for TieringAdmissionPolicy implementations.

Mirrors SecondaryTierFactory: built-in policies are pre-registered below;
out-of-tree policies can register a short name up front or be imported by
module path.
"""

import importlib
from collections.abc import Callable

from vllm.v1.kv_offload.tiering.admission.base import TieringAdmissionPolicy


class AdmissionPolicyFactory:
    """Registry for TieringAdmissionPolicy implementations, resolved by name."""

    _registry: dict[str, Callable[[], type[TieringAdmissionPolicy]]] = {}

    @classmethod
    def register_policy(cls, name: str, module_path: str, class_name: str) -> None:
        if name in cls._registry:
            raise ValueError(f"Admission policy '{name}' is already registered.")

        def loader() -> type[TieringAdmissionPolicy]:
            return cls.get_policy_class(class_name, module_path)

        cls._registry[name] = loader

    @classmethod
    def get_policy_class(
        cls, name: str, module_path: str | None = None
    ) -> type[TieringAdmissionPolicy]:
        """Get a registered admission policy class by name.

        Args:
            name: The registered policy name or out-of-tree class name.
            module_path: Optional module containing an out-of-tree policy.

        Raises:
            ValueError: If no policy is registered under ``name``.

        """
        if module_path is not None:
            module = importlib.import_module(module_path)
            policy_cls = getattr(module, name)
            if not issubclass(policy_cls, TieringAdmissionPolicy):
                raise TypeError(f"{name} is not a TieringAdmissionPolicy")
            return policy_cls
        if name not in cls._registry:
            raise ValueError(
                f"Unknown admission policy: {name!r}. "
                f"Supported policies: {list(cls._registry)}."
            )
        return cls._registry[name]()

    @classmethod
    def create_policy(cls, config: str | dict[str, object]) -> TieringAdmissionPolicy:
        """Create a policy from a name or an out-of-tree configuration."""
        if isinstance(config, str):
            return cls.get_policy_class(config)()
        if not isinstance(config, dict):
            raise ValueError("admission_policy must be a name or a configuration dict")
        name = config.get("type")
        module_path = config.get("module_path")
        if not isinstance(name, str):
            raise ValueError("admission_policy requires a string 'type'")
        if module_path is not None and not isinstance(module_path, str):
            raise ValueError("admission_policy 'module_path' must be a string")
        return cls.get_policy_class(name, module_path)()


AdmissionPolicyFactory.register_policy(
    "always",
    "vllm.v1.kv_offload.tiering.admission.always",
    "AlwaysAdmitPolicy",
)
