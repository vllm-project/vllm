# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.platforms import resolve_current_platform_cls_qualname


def test_resolve_platform_honors_cpu_target_device(monkeypatch):
    # Simulate a GPU machine where CUDA would otherwise be auto-detected.
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "cpu")
    monkeypatch.setattr(
        "vllm.platforms.builtin_platform_plugins",
        {
            "cuda": lambda: "vllm.platforms.cuda.CudaPlatform",
            "cpu": lambda: "vllm.platforms.cpu.CpuPlatform",
        },
    )

    platform_cls_qualname = resolve_current_platform_cls_qualname()

    assert platform_cls_qualname == "vllm.platforms.cpu.CpuPlatform"
