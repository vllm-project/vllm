# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch


class FusedMoEParams(torch.nn.Module):
    def __init__(self):
        super().__init__()
