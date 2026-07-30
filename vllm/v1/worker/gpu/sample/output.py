# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import NamedTuple

import torch

from vllm.v1.outputs import LogprobsTensors


class SamplingMaskTensors(NamedTuple):
    """Compact device-side sampling mask data pending async D2H."""

    # [num_kept_tokens]
    token_ids: torch.Tensor
    # [num_requests]
    counts: torch.Tensor

    def to_cpu_nonblocking(self) -> "SamplingMaskTensors":
        if self.token_ids.device.type == "cpu":
            return self
        return SamplingMaskTensors(
            self.token_ids.to("cpu", non_blocking=True),
            self.counts.to("cpu", non_blocking=True),
        )


@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None
    num_nans: torch.Tensor | None
    num_sampled: torch.Tensor | None
    num_rejected: torch.Tensor | None = None
    sampling_mask_tensors: SamplingMaskTensors | None = None
