# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.parallel import EPLBPolicy

from .abstract_policy import EplbPolicy
from .default_eplb_policy import DefaultEplb


class PolicyFactory:
    @staticmethod
    def generate_policy(policy_type: EPLBPolicy) -> EplbPolicy:
        """
        DefaultEplb: The rearrangement algorithm
        is adapted from [DeepSeek EPLB]
        Dynamic EPLB:  expert replacement with
        constrained number of expert shuffle
        """

        policy: dict[EPLBPolicy, type[EplbPolicy]] = {
            "default": DefaultEplb,
        }
        selected_policy = policy.get(policy_type, DefaultEplb)
        return selected_policy()
