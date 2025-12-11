# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.layers.fused_moe.fused_moe_router import FusedMoERouter


class FusedMoEParams(torch.nn.Module):
    def __init__(self, router: FusedMoERouter):
        super().__init__()
        self.router = router
