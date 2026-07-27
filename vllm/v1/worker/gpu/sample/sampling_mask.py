# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors


def compact_sampling_mask(
    processed_logits: torch.Tensor,
) -> SamplingMaskTensors:
    keep = torch.isfinite(processed_logits)
    return SamplingMaskTensors(
        keep=keep,
        counts=keep.sum(dim=-1, dtype=torch.int32),
    )
