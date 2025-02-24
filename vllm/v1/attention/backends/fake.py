# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass


@dataclass
class FakeAttentionMetadata:

    # Used for DP.
    num_input_tokens: int  # Number of tokens including padding.
