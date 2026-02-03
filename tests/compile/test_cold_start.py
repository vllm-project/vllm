# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from torch._dynamo.utils import counters

from vllm import LLM
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode


def test_moe_compilation_cold_start(monkeypatch, use_fresh_inductor_cache):
    # Run in same process so we can access PyTorch's internal counters
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # I'm not sure if this is going to affect the numbers
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "0")

    # Force cold compilation
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

    compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.NONE,  # make the model loading faster
    )

    counters.clear()

    _ = LLM(
        model="microsoft/Phi-tiny-MoE-instruct",
        max_model_len=256,
        load_format="dummy",  # make the model loading faster
        compilation_config=compilation_config,
        num_gpu_blocks_override=8,  # make the model loading faster
    )

    # vLLM-compile cold start is special. By default, we do
    # one full dynamo capture of the entire forward pass.
    # The forward pass consists of 32 transformer layers.
    # Then, we split on the attention operation. This results in
    # 33 subgraphs (not including the attention operation).
    # We then standalone_compile the unique subgraphs.
    #
    # There are actually only 3 unique subgraphs for this model
    # (all of its transformer layers are the same modulo weights);
    # this is true for most vLLM models.
    # So we test that during cold start, only 3 subgraphs are compiled
    # These 3 subgraphs should cache miss, and then there should be
    # no other compilation (so no cache hits).
    assert counters["aot_autograd"]["autograd_cache_miss"] == 3
    assert counters["aot_autograd"]["autograd_cache_hit"] == 0
