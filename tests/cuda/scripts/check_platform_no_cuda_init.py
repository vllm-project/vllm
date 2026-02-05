#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that vllm.platforms import does not initialize CUDA."""

import os

for key in ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]:
    os.environ.pop(key, None)

import torch

assert not torch.cuda.is_initialized(), "CUDA initialized before import"

from vllm.platforms import current_platform

assert not torch.cuda.is_initialized(), (
    f"CUDA was initialized during vllm.platforms import on {current_platform}"
)
print("OK")
