# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import get_args

from vllm.config.parallel import EPLBPolicyOption

from .abstract import AbstractEplbPolicy
from .batched import BatchedEplbPolicy
from .default import DefaultEplbPolicy

EPLB_POLICIES = {"batched": BatchedEplbPolicy, "default": DefaultEplbPolicy}

# Ensure that the EPLB_POLICIES keys match the EPLBPolicyOption values
assert set(EPLB_POLICIES.keys()) == set(get_args(EPLBPolicyOption))

__all__ = [
    "AbstractEplbPolicy",
    "BatchedEplbPolicy",
    "DefaultEplbPolicy",
    "EPLB_POLICIES",
]
