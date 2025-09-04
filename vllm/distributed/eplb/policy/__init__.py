# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import get_args

from vllm.config.parallel import EPLBPolicyOption

from .abstract import AbstractEplbPolicy
from .default import DefaultEplbPolicy
from .flashlb import FlashlbEplbPolicy
from .swift_balancer import SwiftBalancerPolicy

EPLB_POLICIES = {"default": DefaultEplbPolicy,"flashlb": FlashlbEplbPolicy,"swift_balancer": SwiftBalancerPolicy}

# Ensure that the EPLB_POLICIES keys match the EPLBPolicyOption values
assert set(EPLB_POLICIES.keys()) == set(get_args(EPLBPolicyOption))

__all__ = [
    "AbstractEplbPolicy",
    "DefaultEplbPolicy",
    "FlashlbEplbPolicy",
    "SwiftBalancerPolicy",
    "EPLB_POLICIES",
]
