# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decoder-side SWA bounded replay: the replay batch of a step.

Past the last KV-source layer (the cut layer), the layers own nothing but
sliding-window KV, and the cut layer's own output feeds only them. So in eager
prefill steps the cut layer writes its KV for every row, and its query side,
its FFN and every layer after it run on each request's last ``window`` rows
only. ``DeepseekV41ModelState`` prepares those rows as a sub-batch with
attention metadata and forward contexts of its own, like a microbatch;
``DeepseekV4Model._run_replay`` gathers the layer inputs by its rows, runs the
layers under those contexts and scatters the outputs back to batch rows. Steps
that run in a CUDA graph keep the layers on the whole batch, inside the graph.
"""

import torch

from vllm.forward_context import ForwardContext


class DecoderReplayLayers:
    """The step's replay batch, set by the model state every step.

    ``rows`` are its rows of the batch, None when the step does not replay.
    ``forward_context`` serves the layers after the cut layer, whose window KV
    starts at each request's first replay row; ``cut_forward_context`` serves
    the cut layer's query side, whose window KV the cut layer wrote for every
    row, so it keeps the encoder-side window starts.
    """

    def __init__(self, window: int) -> None:
        self.window = window
        self.rows: torch.Tensor | None = None
        self.forward_context: ForwardContext | None = None
        self.cut_forward_context: ForwardContext | None = None
