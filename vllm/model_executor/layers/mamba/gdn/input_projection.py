# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache

import torch

from vllm.utils.multi_stream_utils import maybe_execute_in_parallel


@cache
def _resources(device: torch.device, parent_stream: int):
    from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE

    with torch.accelerator.device_index(device.index):
        stream = torch.cuda.Stream()
        events = (torch.cuda.Event(), torch.cuda.Event())
        workspaces = [
            torch.empty(DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device=device)
            for _ in range(2)
        ]
    return stream, events, workspaces


@torch.library.custom_op("vllm::gdn_input_gemms", mutates_args=(), device_types="cuda")
def gdn_input_gemms(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
    input_scale: torch.Tensor,
    qkvz_scale: torch.Tensor,
    ba_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concurrent scalar-FP8 projections with separate scratch and BF16 outputs."""
    # The public bmm_fp8 API shares one workspace across streams. This narrow
    # adapter uses its existing dispatcher with explicit workspace ownership.
    from flashinfer.gemm.gemm_base import fp8_gemm_sm100

    parent = torch.cuda.current_stream(x.device)
    side, events, workspaces = _resources(x.device, parent.cuda_stream)
    qkvz = torch.empty(
        (x.shape[0], qkvz_weight.shape[1]), dtype=torch.bfloat16, device=x.device
    )
    ba = torch.empty(
        (x.shape[0], ba_weight.shape[1]), dtype=torch.bfloat16, device=x.device
    )
    maybe_execute_in_parallel(
        lambda: fp8_gemm_sm100(
            x.unsqueeze(0),
            qkvz_weight.unsqueeze(0),
            input_scale,
            qkvz_scale,
            qkvz.unsqueeze(0),
            workspaces[0],
            ["cublas"],
        ),
        lambda: fp8_gemm_sm100(
            x.unsqueeze(0),
            ba_weight.unsqueeze(0),
            input_scale,
            ba_scale,
            ba.unsqueeze(0),
            workspaces[1],
            ["cublas"],
        ),
        events[0],
        events[1],
        side,
    )
    return qkvz, ba


@gdn_input_gemms.register_fake
def _gdn_input_gemms_fake(x, qkvz_weight, ba_weight, input_scale, qkvz_scale, ba_scale):
    return (
        x.new_empty((x.shape[0], qkvz_weight.shape[1]), dtype=torch.bfloat16),
        x.new_empty((x.shape[0], ba_weight.shape[1]), dtype=torch.bfloat16),
    )
