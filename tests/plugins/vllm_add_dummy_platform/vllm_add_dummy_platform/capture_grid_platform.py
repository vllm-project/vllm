# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

if torch.version.hip:
    from vllm.platforms.rocm import RocmPlatform as NativePlatform
elif torch.version.cuda:
    from vllm.platforms.cuda import CudaPlatform as NativePlatform
else:
    from vllm.platforms.xpu import XPUPlatform as NativePlatform


def register() -> str:
    return f"{__name__}.CaptureGridPlatform"


def _grid(limit: int, middle: int, dense_head: bool = False) -> list[int]:
    sizes = {1, middle, limit}
    if dense_head:
        sizes.update(range(1, min(limit, 32) + 1))
    return sorted(n for n in sizes if 1 <= n <= limit)


class CaptureGridPlatform(NativePlatform):
    """Exercise the platform extension contract on the real GPU backend."""

    def get_cudagraph_capture_defaults(self, vllm_config):
        from vllm.config.compilation import CUDAGraphCaptureDefaults

        if os.environ["VLLM_TEST_CAPTURE_DEFAULTS"] == "legacy":
            return CUDAGraphCaptureDefaults(legacy_max_tokens=24)
        return CUDAGraphCaptureDefaults(
            legacy_max_tokens=24,
            max_prefill_tokens=17,
            max_decode_requests=7,
            max_decode_tokens=21,
        )

    @classmethod
    def get_legacy_cudagraph_capture_grid(cls, max_size, dense_head=False):
        return _grid(max_size, 5, dense_head)

    @classmethod
    def get_prefill_cudagraph_capture_grid(cls, max_tokens, dense_head=False):
        return _grid(max_tokens, 5, dense_head)

    @classmethod
    def get_decode_cudagraph_capture_grid(cls, max_requests):
        if os.environ["VLLM_TEST_CAPTURE_DEFAULTS"] == "invalid":
            return [max_requests + 1]
        return _grid(max_requests, 3)

    @classmethod
    def get_varlen_decode_cudagraph_capture_grid(cls, max_tokens, dense_head=False):
        return _grid(max_tokens, 7, dense_head)
