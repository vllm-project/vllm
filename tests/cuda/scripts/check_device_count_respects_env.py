#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that device_count respects CUDA_VISIBLE_DEVICES after platform import."""

import os
import sys

for key in ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]:
    os.environ.pop(key, None)

import torch  # noqa: E402
from vllm.platforms import current_platform  # noqa: F401, E402

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
count = torch.cuda.device_count()

if count == 0:
    sys.exit(0)  # Skip: no GPUs available

assert count == 1, f"device_count()={count}, expected 1"
print("OK")
