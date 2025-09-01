from .abstract_policy import EplbPolicy
from .default_eplb_policy import DefaultEplb
from .swift_eplb_policy import SwiftBalancer
from .dynamic_ep_policy import DynamicEplb
from .dynamic_ep_v2_policy import DynamicEplbV2
from .random_policy import RandomLoadBalance

from vllm.distributed.eplb.eplb_policy.abstract_policy import DynamicConfig


class PolicyFactory:

    @staticmethod
    def generate_policy(policy_type: int, config: DynamicConfig) -> EplbPolicy:

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
            RandomLoadBalance,  # RandomLoadBalance: shuffle last physical expert on NPU 1 and 3
            3:
            DynamicEplb,  # Dynamic EPLB policy: overall expert replacement based on current moe load
            4:
            DynamicEplbV2,  # Dynamic EPLB policy V2:  expert replacement with constrained number of expert shuffle

        }

        return policy.get(policy_type, DefaultEplb)(config)