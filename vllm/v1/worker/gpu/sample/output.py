# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from vllm.v1.outputs import LogprobsTensors, SamplingMaskLists


class SamplingMaskTensors(NamedTuple):
    """Compact device-side sampling mask data pending async D2H."""

    # [num_kept_tokens]
    token_ids: torch.Tensor
    # [num_requests]
    counts: torch.Tensor

    @classmethod
    def from_logits(
        cls,
        logits: torch.Tensor,
        num_sampled_tokens: torch.Tensor,
    ) -> "SamplingMaskTensors":
        """Compact the finite-logit support for requests that sampled tokens."""
        keep = torch.isfinite(logits)[num_sampled_tokens.bool()]
        counts = keep.sum(dim=-1, dtype=torch.int32)
        token_ids = keep.flatten().nonzero(as_tuple=True)[0]
        token_ids = token_ids.remainder(keep.shape[1]).to(torch.int32)
        return cls(token_ids, counts)

    def to_cpu_nonblocking(self) -> "SamplingMaskTensors":
        if self.token_ids.device.type == "cpu":
            return self
        return SamplingMaskTensors(
            self.token_ids.to("cpu", non_blocking=True),
            self.counts.to("cpu", non_blocking=True),
        )

    def tolists(self, num_sampled_tokens: np.ndarray) -> SamplingMaskLists:
        """Convert compact mask tensors to the scheduler's CSR representation."""
        counts = self.counts.cpu().numpy()
        offsets = np.empty(len(counts) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, dtype=np.int64, out=offsets[1:])
        return SamplingMaskLists(
            token_ids=self.token_ids.cpu().numpy(),
            offsets=offsets,
            cu_num_generated_tokens=np.cumsum(
                np.concatenate(([0], num_sampled_tokens))
            ).tolist(),
        )


@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None
    num_nans: torch.Tensor | None
    num_sampled: torch.Tensor | None
    num_rejected: torch.Tensor | None = None
    sampling_mask_tensors: SamplingMaskTensors | None = None
