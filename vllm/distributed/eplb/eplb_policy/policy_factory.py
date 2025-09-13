# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .abstract_policy import EplbPolicy
from .default_eplb_policy import DefaultEplb

from .flashlb_policy import FlashLB
from .swift_balancer_policy import SwiftBalancer



class PolicyFactory:

    @staticmethod
    def generate_policy(policy_type: int) -> EplbPolicy:

        """
        DefaultEplb: The rearrangement algorithm
        is adapted from [DeepSeek EPLB]

        Dynamic EPLB:  expert replacement with
        constrained number of expert shuffle
        """
        policy = {
            0:
            DefaultEplb,
            1:
            SwiftBalancer,
            2:
            FlashLB,
        }

        return policy.get(policy_type, DefaultEplb)()



