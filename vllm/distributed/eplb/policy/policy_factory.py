# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from .policy_abstract import DynamicConfig, EplbPolicy
from .policy_dynamic_ep import DynamicEplb
from .policy_default_eplb import DefaultEplb


class PolicyFactory:

    @staticmethod
    def generate_policy(policy_type: int, config: DynamicConfig) -> EplbPolicy:
        policy = {
            0:
            DefaultEplb,  # RandomLoadBalance: shuffle last physical expert on NPU 1 and 3
            1:
            DynamicEplbV2,  # Dynamic EPLB policy:  expert replacement with constrained number of expert shuffle
        }

        return policy.get(policy_type, DefaultEplb)(config)

