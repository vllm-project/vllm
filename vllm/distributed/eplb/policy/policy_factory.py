from .policy_abstract import EplbPolicy
from .policy_swift_balancer import SwiftBalancer
from .policy_default_eplb import DefaultEplb


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
        }

        return policy.get(policy_type, DefaultEplb)()



