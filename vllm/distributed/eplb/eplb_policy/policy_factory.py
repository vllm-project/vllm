# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .abstract_policy import EplbPolicy
from .default_eplb_policy import DefaultEplb
from vllm.config.parallel import EPLBPolicyType


class PolicyFactory:
    @staticmethod
    def generate_policy(policy_type: EPLBPolicyType) -> EplbPolicy:
        """
        DefaultEplb: The rearrangement algorithm
        is adapted from [DeepSeek EPLB]
        Dynamic EPLB:  expert replacement with
        constrained number of expert shuffle
        """
        
        policy: dict[EPLBPolicyType, type[EplbPolicy]] = {
            EPLBPolicyType.DEFAULT_EPLB: DefaultEplb,
        }
        selected_policy = policy.get(policy_type, DefaultEplb)
        return selected_policy()
