# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass


@dataclass
class FakeAttentionMetadata:

    # For logging.
    num_input_tokens: int  # Number of tokens including padding.
