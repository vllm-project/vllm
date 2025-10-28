# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.compilation.counter import compilation_counter
from vllm.config.compilation import CompilationMode


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
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
            gpu_memory_utilization=0.7,
            compilation_config={"mode": CompilationMode.VLLM_COMPILE},
        ) as _,
    ):
        pass
