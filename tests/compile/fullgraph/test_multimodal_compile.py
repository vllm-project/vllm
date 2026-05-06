# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.platforms import current_platform


def test_compile():
    vllm_config = VllmConfig()
    # Default configuration does not compile mm encoder
    assert not vllm_config.compilation_config.compile_mm_encoder


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
def test_qwen2_5_vl_compilation(vllm_runner, monkeypatch):
    """Test that Qwen2.5-VL vision submodules are compiled.

    This test verifies that the 3 vision submodules (Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionBlock, and Qwen2_5_VisionPatchMerger) are properly tagged
    for compilation by checking that num_models_seen increases by at least 3.
    """
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    with (
        # NOTE: Qwen2.5-VL has 35 models in total - the LLM backend
        # Vision Patch Embed, Vision Patch Merger, and then 32 Vision Blocks
        # (one for each layer) - in the future, we should fix vLLM compilation
        # logic to handle this case and only compile the Vision submodules once
        # and reuse the compiled code for all layers
        # See https://github.com/vllm-project/vllm/issues/27590
        compilation_counter.expect(num_models_seen=35),
        vllm_runner(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            compilation_config={
                "mode": CompilationMode.VLLM_COMPILE,
                "compile_mm_encoder": True,
            },
        ) as _,
    ):
        pass


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
def test_qwen2_5_vl_no_vit_compilation(vllm_runner, monkeypatch):
    """Test that Qwen2.5-VL vision submodules are not compiled when the
    config is passed off
    """
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    with (
        compilation_counter.expect(num_models_seen=1),
        vllm_runner(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            compilation_config={
                "mode": CompilationMode.VLLM_COMPILE,
                "compile_mm_encoder": False,
            },
        ) as _,
    ):
        pass


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
# Requires Cuda and 8 gpus as well
@pytest.mark.forked
@pytest.mark.skip(reason="Skipping due to CI resource constraints")
def test_mllama4_vit_compilation(vllm_runner, monkeypatch):
    """Test that Mllama4 vision submodules are compiled.

    This test verifies that the 2 vision submodules (Llama4VisionEncoder,
    Llama4VisionPixelShuffleMLP) are properly tagged
    for compilation by checking that num_models_seen increases to 3.

    However since we are using TP=8, we compilation_counter will not
    work properly so we will just check the run succeeds rn
    """
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    with (
        monkeypatch.context(),
        # TODO: Since we require TP=8, this messes with the compilation
        # counter. We should fix this in the future, but leave for now
        # to make sure that compilation runs (no crash) with llama vision encoder
        compilation_counter.expect(num_models_seen=0),
        vllm_runner(
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            max_model_len=512,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=8,
            compilation_config={
                "mode": CompilationMode.VLLM_COMPILE,
                "compile_mm_encoder": True,
            },
        ),
    ):
        pass
