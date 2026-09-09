# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from collections.abc import Callable

from vllm.v1.kv_offload.tiering.admission.base import TieringAdmissionPolicy


class AdmissionPolicyFactory:
    """Registry for TieringAdmissionPolicy implementations, resolved by a
    config dict's "type" key. Mirrors SecondaryTierFactory."""

    _registry: dict[str, Callable[[], type[TieringAdmissionPolicy]]] = {}

    @classmethod
    def register_policy(
        cls, policy_type: str, module_path: str, class_name: str
    ) -> None:
        if policy_type in cls._registry:
            raise ValueError(f"Admission policy '{policy_type}' is already registered.")

        def loader() -> type[TieringAdmissionPolicy]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[policy_type] = loader

    @classmethod
    def get_policy_class(cls, policy_config: dict) -> type[TieringAdmissionPolicy]:
        policy_type = policy_config.get("type")
        if not policy_type:
            raise ValueError("Admission policy configuration must include 'type'")
        if policy_type not in cls._registry:
            raise ValueError(
                f"Unknown admission policy type: {policy_type!r}. "
                f"Supported types: {list(cls._registry)}"
            )
        return cls._registry[policy_type]()

    @classmethod
    def create_policy(cls, policy_config: dict) -> TieringAdmissionPolicy:
        policy_cls = cls.get_policy_class(policy_config)
        config = policy_config.copy()
        config.pop("type")
        return policy_cls(**config)


AdmissionPolicyFactory.register_policy(
    "always_admit", "vllm.v1.kv_offload.tiering.admission.always", "AlwaysAdmitPolicy"
)
