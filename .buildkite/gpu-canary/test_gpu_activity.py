# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import time

import torch

buffers = []


def test_01_idle():
    assert torch.accelerator.is_available()
    time.sleep(8)


def test_02_all_gpus_busy():
    for device in range(torch.accelerator.device_count()):
        with torch.accelerator.device_index(device):
            buffers.append(
                (
                    torch.randn(4096, 4096, device=device, dtype=torch.float16),
                    torch.randn(4096, 4096, device=device, dtype=torch.float16),
                    torch.empty(4096, 4096, device=device, dtype=torch.float16),
                    torch.zeros(512 * 1024 * 1024, device=device, dtype=torch.uint8),
                )
            )
    stop = time.monotonic() + 40
    while time.monotonic() < stop:
        for _ in range(16):
            for a, b, c, retained in buffers:
                torch.mm(a, b, out=c)
        for device in range(torch.accelerator.device_count()):
            torch.accelerator.synchronize(device)


def test_03_idle_with_memory():
    assert buffers
    time.sleep(10)


def test_04_memory_released():
    buffers.clear()
    gc.collect()
    for device in range(torch.accelerator.device_count()):
        with torch.accelerator.device_index(device):
            torch.accelerator.empty_cache()
    time.sleep(8)
