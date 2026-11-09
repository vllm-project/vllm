# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Hopper executor's fused all-reduce + mHC: `BoundaryAllReduce` with
BF16 layer input, whose finalize waits for the low-batch MoE kernel before it
reads the route map and weights."""

from transformers import PreTrainedConfig

from ..boundary.all_reduce import BoundaryAllReduce


class HopperBoundaryAllReduce(BoundaryAllReduce):
    """`BoundaryAllReduce` with BF16 layer input. The low-batch MoE kernel,
    the grid right before each finalize, writes the route map and weights the
    finalize reads."""

    def __init__(self, config: PreTrainedConfig, max_num_tokens: int) -> None:
        super().__init__(
            config,
            quantize_layer_input=False,
            max_num_tokens=max_num_tokens,
            finalize_metadata_before_pdl=False,
        )
