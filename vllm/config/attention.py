# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum

from pydantic.dataclasses import dataclass

from vllm.config.utils import config


class MultiCascadeAllocateMethod(enum.Enum):
    """Methods for grouping requests for multi-cascade attention."""

    LEAF_PASS: str = "leaf_pass"
    """Only group requests if their latest `KVCacheBlock` is the same."""
    FULL_PASS: str = "full_pass"
    """Group requests even if they share a partially common prefix.
    Takes longer than leaf_pass but results in larger groups."""


@config
@dataclass
class MultiCascadeConfig:
    """Contains configurations for deciding how to schedule multi-cascade
    attention."""

    absorption_threshold: float = 0.8
    """Threshold parameter that specifies whether to group more leaves with
    lower prefix length (lowabsorption threshold), or fewer leaves with higher
    common prefix length (higher absorption threshold)."""

    allocate_method: MultiCascadeAllocateMethod = \
        MultiCascadeAllocateMethod.LEAF_PASS
    """Method used to group requests with common prefixes. Logic behind
    grouping is in [vllm.v1.core.multi_cascade_manager][]."""
