# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://huggingface.co/nvidia/NVLM-D-72B/blob/main/configuration_nvlm_d.py
# --------------------------------------------------------
# NVLM-D
# Copyright (c) 2024 NVIDIA
# Licensed under Apache 2.0 License [see LICENSE for details]
# --------------------------------------------------------
from .internvl import InternVLChatConfig


class NVLM_D_Config(InternVLChatConfig):
    model_type = 'NVLM_D'
