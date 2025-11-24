# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config.parallel import EPLBPolicyOption

from .abstract import AbstractEplbPolicy
from .default import DefaultEplbPolicy


class EplbPolicyFactory:
    @staticmethod
    def generate_policy(policy_type: EPLBPolicyOption) -> type[AbstractEplbPolicy]:
        """
        DefaultEplbPolicy: The rearrangement algorithm
        is adapted from [DeepSeek EPLB]
        Dynamic EPLB:  expert replacement with
        constrained number of expert shuffle
        """

        policy: dict[EPLBPolicyOption, type[AbstractEplbPolicy]] = {
            "default": DefaultEplbPolicy,
        }
        if policy_type not in policy:
            raise ValueError(
                f"Invalid EPLB policy type: '{policy_type}'. "
                f"Available policies are: {list(policy.keys())}"
            )
        return policy[policy_type]
