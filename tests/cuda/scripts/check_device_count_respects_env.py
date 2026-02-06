#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that device_count respects CUDA_VISIBLE_DEVICES after platform import."""

import os
import sys

for key in ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]:
    os.environ.pop(key, None)

import torch  # noqa: E402

if torch.cuda.device_count() == 0:
    sys.exit(0)  # Skip: no GPUs

initial_count = torch.cuda.device_count()
if initial_count <= 1:
    sys.exit(0)  # Skip: need multiple GPUs

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
count = torch.cuda.device_count()

assert count == 1, f"device_count()={count}, expected 1 (initial was {initial_count})"
print("OK")
