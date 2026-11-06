# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm.collect_env as collect_env


def _make_system_env(**overrides) -> collect_env.SystemEnv:
    defaults = dict(
        torch_version="2.7.0",
        is_debug_build="False",
        cuda_compiled_version="N/A",
        gcc_version="gcc 13",
        clang_version="clang 18",
        cmake_version="3.30",
        os="Ubuntu 24.04",
        libc_version="glibc-2.39",
        python_version="3.12.9",
        python_platform="Linux-6.8",
        is_cuda_available="True",
        cuda_runtime_version="N/A",
        cuda_module_loading="N/A",
        nvidia_driver_version=None,
        nvidia_gpu_models="AMD Instinct MI300X",
        cudnn_version="N/A",
        xpu_available="False",
        xpu_runtime_version="N/A",
        intel_graphics_compiler_version="N/A",
        intel_gpu_models="N/A",
        oneapi_compiler_version="N/A",
        level_zero_loader_version="N/A",
        level_zero_driver_version="N/A",
        oneccl_version="N/A",
        libigdgmm_version="N/A",
        vllm_xpu_kernels_version="N/A",
        sycl_version="N/A",
        pip_version="pip3",
        pip_packages="vllm==0.19.0",
        conda_packages="",
        hip_compiled_version="7.1.0",
        hip_runtime_version="7.1.0",
        miopen_runtime_version="3.5.0",
        caching_allocator_config="",
        is_xnnpack_available="False",
        cpu_info="cpu info",
        rocm_version="7.1.0",
        vllm_version="0.19.0",
        vllm_build_flags="flags",
        gpu_topo="topo",
        env_vars="",
    )
    defaults.update(overrides)
    return collect_env.SystemEnv(**defaults)


def test_pretty_str_uses_rocm_specific_labels(monkeypatch):
    monkeypatch.setattr(collect_env, "TORCH_AVAILABLE", False)
    rendered = collect_env.pretty_str(_make_system_env())

    assert "ROCm / GPU Info" in rendered
    assert "Is torch.cuda available" in rendered
    assert "GPU models and configuration : AMD Instinct MI300X" in rendered
    assert "Nvidia driver version" not in rendered


def test_pretty_str_keeps_cuda_labels_for_cuda_env(monkeypatch):
    monkeypatch.setattr(collect_env, "TORCH_AVAILABLE", False)
    rendered = collect_env.pretty_str(
        _make_system_env(
            cuda_compiled_version="12.8",
            cuda_runtime_version="12.8",
            cuda_module_loading="LAZY",
            nvidia_driver_version="570.124",
            nvidia_gpu_models="NVIDIA H100",
            cudnn_version="9.8",
            hip_compiled_version="N/A",
            hip_runtime_version="N/A",
            miopen_runtime_version="N/A",
            rocm_version="N/A",
        )
    )

    assert "CUDA / GPU Info" in rendered
    assert "Is CUDA available" in rendered
    assert "Nvidia driver version" in rendered
    assert "NVIDIA H100" in rendered


def test_get_env_vars_includes_rocm_runtime_prefixes(monkeypatch):
    monkeypatch.setenv("ROCM_PATH", "/opt/rocm")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.5.1")
    monkeypatch.setenv("MIOPEN_DEBUG_CONV_DIRECT", "1")
    monkeypatch.setenv("RCCL_P2P_DISABLE", "1")

    rendered = collect_env.get_env_vars()

    assert "ROCM_PATH=/opt/rocm" in rendered
    assert "HIP_VISIBLE_DEVICES=0,1" in rendered
    assert "HSA_OVERRIDE_GFX_VERSION=11.5.1" in rendered
    assert "MIOPEN_DEBUG_CONV_DIRECT=1" in rendered
    assert "RCCL_P2P_DISABLE=1" in rendered
