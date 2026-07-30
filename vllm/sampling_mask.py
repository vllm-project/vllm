# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class SamplingMask:
    """CSR token support sets aligned with completion token IDs.

    Args:
        token_ids: Flattened token IDs from every support set.
        offsets: Start offsets for each support set, including the final end
            offset.
    """

    token_ids: Sequence[int]
    offsets: Sequence[int]
