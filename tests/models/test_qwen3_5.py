# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest


def test_qwen3_5_models_use_platform_specific_implementation():
    from vllm.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MTP
    from vllm.platforms import current_platform

    # The NVIDIA and ROCm CDNA4+ variants drop torch.compile; other platforms
    # keep the previous torch.compile implementation in
    # vllm/model_executor/models/qwen3_5{,_mtp}.py.
    if current_platform.is_cuda():
        model_module = "vllm.models.qwen3_5.nvidia.model"
        mtp_module = "vllm.models.qwen3_5.nvidia.mtp"
    elif current_platform.is_rocm() and (
        current_platform.is_device_capability(95)
        or current_platform.is_device_capability(125)
    ):
        model_module = "vllm.models.qwen3_5.amd.model"
        mtp_module = "vllm.models.qwen3_5.amd.mtp"
    else:
        model_module = "vllm.model_executor.models.qwen3_5"
        mtp_module = "vllm.model_executor.models.qwen3_5_mtp"
    assert Qwen3_5ForCausalLM.__module__ == model_module
    assert Qwen3_5MTP.__module__ == mtp_module


@pytest.mark.parametrize("platform", ["nvidia", "amd"])
def test_qwen3_5_platform_models_do_not_use_torch_compile_wrapper(platform: str):
    # Import the platform modules directly so the assertion is independent of
    # the platform this test runs on.
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    model_module = importlib.import_module(f"vllm.models.qwen3_5.{platform}.model")
    mtp_module = importlib.import_module(f"vllm.models.qwen3_5.{platform}.mtp")

    assert not issubclass(model_module.Qwen3_5Model, TorchCompileWithNoGuardsWrapper)
    assert not issubclass(
        mtp_module.Qwen3_5MultiTokenPredictor, TorchCompileWithNoGuardsWrapper
    )
    assert not issubclass(mtp_module.Qwen3_5MTP, TorchCompileWithNoGuardsWrapper)


def test_qwen3_5_legacy_models_keep_torch_compile_wrapper():
    # The legacy implementation in vllm/model_executor/models/ serves the
    # other platforms; it must keep the torch.compile wrapper.
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.model_executor.models import qwen3_5 as model_module
    from vllm.model_executor.models import qwen3_5_mtp as mtp_module

    assert issubclass(model_module.Qwen3_5Model, TorchCompileWithNoGuardsWrapper)
    assert issubclass(
        mtp_module.Qwen3_5MultiTokenPredictor, TorchCompileWithNoGuardsWrapper
    )
    assert issubclass(mtp_module.Qwen3_5MTP, TorchCompileWithNoGuardsWrapper)
