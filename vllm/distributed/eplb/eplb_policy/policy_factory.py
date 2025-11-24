# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.parallel import EPLBPolicyOption

from .abstract_policy import AbstractEplbPolicy
from .default_eplb_policy import DefaultEplb


class PolicyFactory:
    @staticmethod
    def generate_policy(policy_type: EPLBPolicyOption) -> type[AbstractEplbPolicy]:
        """
        DefaultEplb: The rearrangement algorithm
        is adapted from [DeepSeek EPLB]
        Dynamic EPLB:  expert replacement with
        constrained number of expert shuffle
        """

        policy: dict[EPLBPolicyOption, type[AbstractEplbPolicy]] = {
            "default": DefaultEplb,
        }
        if policy_type not in policy:
            raise ValueError(
                f"Invalid EPLB policy type: '{policy_type}'. "
                f"Available policies are: {list(policy.keys())}"
            )
        selected_policy = policy.get(policy_type, DefaultEplb)
        return selected_policy
