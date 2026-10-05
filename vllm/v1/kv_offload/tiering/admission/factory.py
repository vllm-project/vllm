# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry for TieringAdmissionPolicy implementations.

Mirrors SecondaryTierFactory: built-in policies are pre-registered below;
out-of-tree policies can register a short name up front or be imported by
module path. Policies that need constructed objects (e.g.
BackpressureAdmissionPolicy, which needs the secondary tier instances) are
intentionally not registered here -- construct those directly.
"""

import importlib
from collections.abc import Callable

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.admission.base import TieringAdmissionPolicy

logger = init_logger(__name__)


class AdmissionPolicyFactory:
    """Registry for TieringAdmissionPolicy implementations, resolved by name."""

    _registry: dict[str, Callable[[], type[TieringAdmissionPolicy]]] = {}

    @classmethod
    def register_policy(cls, name: str, module_path: str, class_name: str) -> None:
        if name in cls._registry:
            raise ValueError(f"Admission policy '{name}' is already registered.")

        def loader() -> type[TieringAdmissionPolicy]:
            module = importlib.import_module(module_path)
            policy_cls = getattr(module, class_name)
            assert issubclass(policy_cls, TieringAdmissionPolicy)
            return policy_cls

        cls._registry[name] = loader

    @classmethod
    def get_policy_class(cls, name: str) -> type[TieringAdmissionPolicy]:
        """Get a registered admission policy class by name.

        Args:
            name: The registered policy name (e.g. "always").

        Raises:
            ValueError: If no policy is registered under ``name``.

        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown admission policy: {name!r}. "
                f"Supported policies: {list(cls._registry)}."
            )
        return cls._registry[name]()


AdmissionPolicyFactory.register_policy(
    "always",
    "vllm.v1.kv_offload.tiering.admission.always",
    "AlwaysAdmitPolicy",
)
