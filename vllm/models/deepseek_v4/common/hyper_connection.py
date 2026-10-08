# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hyper-connection parameters of DeepSeek-V4, named as in Transformers."""

import torch
from torch import nn


def _fp32_param(*shape: int) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, dtype=torch.float32), requires_grad=False)


class DeepseekV4HyperConnection(nn.Module):
    """Mixing weights of the hyper-connection around one sub-layer."""

    def __init__(self, hc_mult: int, hidden_size: int):
        super().__init__()
        mix_hc = (2 + hc_mult) * hc_mult
        self.fn = _fp32_param(mix_hc, hc_mult * hidden_size)
        self.base = _fp32_param(mix_hc)
        self.scale = _fp32_param(3)


class DeepseekV4HyperConnectionHead(nn.Module):
    """Mixing weights that merge the hyper-connection copies before the head."""

    def __init__(self, hc_mult: int, hidden_size: int):
        super().__init__()
        self.hc_fn = _fp32_param(hc_mult, hc_mult * hidden_size)
        self.hc_base = _fp32_param(hc_mult)
        self.hc_scale = _fp32_param(1)
