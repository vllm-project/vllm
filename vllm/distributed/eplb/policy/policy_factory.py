# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import get_args

from vllm.config.parallel import EPLBPolicyOption

from .abstract import AbstractEplbPolicy
from .default import DefaultEplbPolicy
from .flashlb import FlashlbEplbPolicy
from .swift_balancer import SwiftBalancerPolicy

EPLB_POLICIES: dict[EPLBPolicyOption, type[AbstractEplbPolicy]] = {
    "default": DefaultEplbPolicy,
    "flashlb": FlashlbEplbPolicy,
    "swift_balancer": SwiftBalancerPolicy,
}

# Ensure that the EPLB_POLICIES keys match the EPLBPolicyOption values
assert set(EPLB_POLICIES.keys()) == set(get_args(EPLBPolicyOption))


class EplbPolicyFactory:
    """
    EPLB Policy Factory Class: Unified creation of different policy instances

    Features:
    1. Create instances based on string type (aligned with EPLBPolicyOption),
       rejecting numeric keys.
    2. Automatically adapt to thread-isolated instantiation of stateful policies.
    3. Provide policy registration mechanism to support extending new policies.
    """

    # Policy registry (aligned with EPLB_POLICIES, dynamically extensible)
    _POLICY_REGISTRY = EPLB_POLICIES

    @classmethod
    def register_policy(
        cls, policy_name: EPLBPolicyOption, policy_cls: type[AbstractEplbPolicy]
    ):
        """
        Register a new policy (for extension).

        Parameters:
            policy_name: Policy name (must comply with EPLBPolicyOption)
            policy_cls: Policy class (must inherit from AbstractEplbPolicy)
        """
        if not issubclass(policy_cls, AbstractEplbPolicy):
            raise TypeError(
                f"Policy class {policy_cls.__name__} must"
                f" inherit from AbstractEplbPolicy"
            )
        cls._POLICY_REGISTRY[policy_name] = policy_cls

    @classmethod
    def create_policy(
        cls, policy_type: EPLBPolicyOption = "default"
    ) -> AbstractEplbPolicy:
        """
        Create a policy instance (core method).

        Parameters:
            policy_type: Policy type (default/flashlb/swift_balancer)
        Returns:
            instance: Policy instance (subclass of AbstractEplbPolicy)
        """
        if policy_type not in cls._POLICY_REGISTRY:
            raise ValueError(
                f"Invalid policy type: {policy_type}. "
                f"Supported types: {list(cls._POLICY_REGISTRY.keys())}"
            )

        policy_cls = cls._POLICY_REGISTRY[policy_type]
        instance = policy_cls()
        return instance

    @classmethod
    def get_supported_policies(cls) -> list[EPLBPolicyOption]:
        """
        Get all supported policy types.
        """
        return list(cls._POLICY_REGISTRY.keys())
