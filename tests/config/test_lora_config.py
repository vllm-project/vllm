# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.compilation import CompilationConfig
from vllm.config.lora import LoRAConfig
from vllm.v1.worker.gpu.lora_utils import get_lora_capture_cases


def test_specialize_active_lora_changes_compilation_hash():
    """`specialize_active_lora` selects how many cuda graphs are captured
    (see `get_lora_capture_cases`), so it must change the config hash that
    keys the compiled-artifact cache."""
    disabled = LoRAConfig()
    enabled = LoRAConfig(specialize_active_lora=True)

    assert disabled.compute_hash() != enabled.compute_hash()


def test_lora_capture_cases_follow_specialize_active_lora():
    compilation_config = CompilationConfig(cudagraph_specialize_lora=True)

    base = LoRAConfig(max_loras=4)
    specialized = LoRAConfig(max_loras=4, specialize_active_lora=True)

    assert get_lora_capture_cases(base, compilation_config) != get_lora_capture_cases(
        specialized, compilation_config
    )
