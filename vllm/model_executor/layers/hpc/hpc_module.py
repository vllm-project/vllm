# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn


class HpcModule(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def support(cls, *args, **kwargs):
        return True

    def forward(self, *args, **kwargs):
        pass
