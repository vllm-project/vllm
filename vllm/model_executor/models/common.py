# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from dataclasses import dataclass
from typing import Optional

from .radio_model import Resolution


@dataclass
class RadioResource:
    url: str
    patch_size: int
    max_resolution: int
    preferred_resolution: Resolution
    vitdet_num_windowed: Optional[int] = None
    vitdet_num_global: Optional[int] = None


RESOURCE_MAP = {
    # RADIOv2.5
    "radio_v2.5-b":
    RadioResource(
        "https://huggingface.co/nvidia/RADIO/resolve/main/radio-v2.5-b_half.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=(768, 768),
        vitdet_num_global=4,
    ),
    "radio_v2.5-l":
    RadioResource(
        "https://huggingface.co/nvidia/RADIO/resolve/main/radio-v2.5-l_half.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=(768, 768),
        vitdet_num_global=4,
    ),
    "radio_v2.5-h":
    RadioResource(
        "https://huggingface.co/nvidia/RADIO/resolve/main/radio_v2.5-h.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=(768, 768),
        vitdet_num_global=4,
    ),
    "radio_v2.5-h-norm":
    RadioResource(
        "https://huggingface.co/nvidia/RADIO/resolve/main/radio_v2.5-h-norm.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=(768, 768),
        vitdet_num_global=4,
    ),
    "radio_v2.5-g":
    RadioResource(
        "https://huggingface.co/nvidia/RADIO/resolve/main/radio_v2.5-g.pth.tar?download=true",
        patch_size=14,
        max_resolution=1792,
        preferred_resolution=(896, 896),
        vitdet_num_global=8,
    ),
    # RADIO
    "radio_v2.1":
    RadioResource(
        "https://huggingface.co/nvidia/RADIO/resolve/main/radio_v2.1_bf16.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=Resolution(432, 432),
        vitdet_num_windowed=5,
    ),
    "radio_v2":
    RadioResource(
        "https://huggingface.co/nvidia/RADIO/resolve/main/radio_v2.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=Resolution(432, 432),
        vitdet_num_windowed=5,
    ),
    "radio_v1":
    RadioResource(
        "https://huggingface.co/nvidia/RADIO/resolve/main/radio_v1.pth.tar?download=true",
        patch_size=14,
        max_resolution=1050,
        preferred_resolution=Resolution(378, 378),
    ),
    # E-RADIO
    "e-radio_v2":
    RadioResource(
        "https://huggingface.co/nvidia/RADIO/resolve/main/eradio_v2.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=Resolution(512, 512),
    ),
    # C-RADIO
    "c-radio_v2.5-g":
    RadioResource(
        "https://huggingface.co/nvidia/C-RADIOv2-g/resolve/main/c-radio_v2-g_half.pth.tar",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=(768, 768),
        vitdet_num_global=8,
    ),
    "c-radio_v3-l":
    RadioResource(
        # NOTE: Currently, this model cannot be loaded via TorchHub. Instead, use the transformers API at https://huggingface.co/nvidia/C-RADIOv3-L
        # and accept the license terms.
        "https://huggingface.co/nvidia/C-RADIOv3-L/resolve/main/c-radio-v3_l_half.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=Resolution(512, 512),
    ),
}

DEFAULT_VERSION = "radio_v2.5-h"
