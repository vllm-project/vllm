# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Per-request LoRA scaling inputs.

Two requests in the same batch may share a LoRA adapter (identical
`lora_int_id`, identical A/B weights) while asking for different strengths via
`LoRARequest.lora_scale`. Rather than expanding the scale to one float per
token, we keep it as per-request state and pair it with a token -> request
index map, mirroring how the sampler keeps `temperature` as `[max_num_reqs]`
state:

    request_scales = [0.3, 0.8, 1.2]        # one entry per request
    token_to_req   = [0, 0, 0, 1, 1, 2]     # one entry per scheduled token

    scale(token) = base_scale * request_scales[token_to_req[token]]

For a 256-request / 32k-token batch that is ~1KB instead of ~128KB, and it
keeps the scale orthogonal to `token_lora_mapping`, which continues to decide
*which* adapter a token uses rather than how strongly it applies.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LoRAScaleInputs:
    """Per-request LoRA scales and the index maps that reach them."""

    # scale multiplier per request in the batch
    request_scales: tuple[float, ...]
    # scheduled-token index -> request index
    token_to_req: tuple[int, ...]
    # sampled-token index -> request index (used by the logits LoRA path)
    sample_to_req: tuple[int, ...]


def make_lora_scale_inputs(
    request_scales: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    num_sampled_tokens: np.ndarray,
) -> LoRAScaleInputs | None:
    """
    Build the per-request scale inputs for one forward pass.

    Returns None when every request uses the default strength of 1.0, so that
    batches which don't use the feature take exactly the stock code path and
    pay nothing for it.

    Args:
        request_scales: float array of shape [num_reqs], the per-request
            multiplier as stored in the persistent batch.
        num_scheduled_tokens: int array of shape [num_reqs].
        num_sampled_tokens: int array of shape [num_reqs].
    """
    if request_scales.size == 0 or bool(np.all(request_scales == 1.0)):
        return None

    req_indices = np.arange(request_scales.size, dtype=np.int32)
    return LoRAScaleInputs(
        request_scales=tuple(request_scales.astype(np.float32).tolist()),
        token_to_req=tuple(req_indices.repeat(num_scheduled_tokens).tolist()),
        sample_to_req=tuple(req_indices.repeat(num_sampled_tokens).tolist()),
    )
