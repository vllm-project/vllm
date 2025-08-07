# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass
class AdapterMapping:
    # Per every token in input_ids:
    index_mapping: tuple[int, ...]
    # Per sampled token:
    prompt_mapping: tuple[int, ...]

    def __post_init__(self):
        self.index_mapping = tuple(self.index_mapping)
        self.prompt_mapping = tuple(self.prompt_mapping)