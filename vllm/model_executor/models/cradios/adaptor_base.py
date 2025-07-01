# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from typing import NamedTuple

import torch
from torch import nn


class AdaptorInput(NamedTuple):
    images: torch.Tensor
    summary: torch.Tensor
    features: torch.Tensor
    feature_fmt: str
    patch_size: int


class RadioOutput(NamedTuple):
    summary: torch.Tensor
    features: torch.Tensor

    def to(self, *args, **kwargs):
        return RadioOutput(
            self.summary.to(*args, **kwargs)
            if self.summary is not None else None,
            self.features.to(*args, **kwargs)
            if self.features is not None else None,
        )


class AdaptorBase(nn.Module):

    def forward(self, input: AdaptorInput) -> RadioOutput:
        raise NotImplementedError("Subclasses must implement this!")
