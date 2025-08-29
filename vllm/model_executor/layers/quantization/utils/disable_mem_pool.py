# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adopted from
# https://github.com/vllm-project/vllm/issues/19855
"""Utility helpers for native fp8 support in sleep_mode"""
from contextlib import contextmanager
from torch._C import (_cuda_beginAllocateToPool,
                      _cuda_endAllocateCurrentStreamToPool)
from torch.cuda.memory import MemPoolContext
from vllm.device_allocator.cumem import CuMemAllocator

@contextmanager
def disable_mem_pool(disable=False):
    try:
        if disable \
                and "weights" in CuMemAllocator.get_instance().allocator_and_pools \
                and MemPoolContext.active_pool() == \
                    CuMemAllocator.get_instance().allocator_and_pools["weights"][0]:
            pool = MemPoolContext.active_pool()
            ctx = MemPoolContext(None)
            device_index = torch.cuda.current_device()
            try:
                _cuda_endAllocateCurrentStreamToPool(device_index, pool.id)
            finally:
                need_restart = True
        else:
            need_restart = False
        yield
    finally:
        if disable and need_restart:
            _cuda_beginAllocateToPool(device_index, pool.id)
            del ctx
