# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass
class SamplingMask:
    """Per-token sampling support sets aligned with completion token IDs.

    Each inner list contains the vocabulary token IDs that survived
    top-k / top-p / min-p filtering for the corresponding generated token.
    """

    token_ids: list[list[int]]
