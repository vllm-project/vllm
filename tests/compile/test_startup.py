# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold start and warm start tests for vLLM-compile.

Cold start runs in a forked child (must fork before CUDA init) which
populates on-disk caches and asserts cold-start counters.  Warm start
then runs in the parent with clean in-memory state but populated caches.
"""

import multiprocessing as mp

import pytest
from torch._dynamo.utils import counters

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

from ..utils import fork_new_process_for_each_test

MODEL = "microsoft/Phi-tiny-MoE-instruct"


def _run_vllm(vllm_runner):
    with vllm_runner(
        MODEL,
        trust_remote_code=False,
        max_model_len=256,
        max_num_batched_tokens=1024,
        load_format="dummy",
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
        num_gpu_blocks_override=8,
    ):
        pass


def _cold_start(vllm_runner):
    counters.clear()
    with compilation_counter.expect(
        num_compiled_artifacts_saved=3,
        num_compiled_artifacts_loaded=0,
    ):
        _run_vllm(vllm_runner)
    assert counters["aot_autograd"]["total"] == 33
    assert counters["aot_autograd"]["autograd_cache_miss"] == 3
    assert counters["aot_autograd"]["autograd_cache_hit"] == 0


@fork_new_process_for_each_test
@pytest.mark.parametrize("mega_aot_artifact", ["0", "1"])
def test_moe_startup(monkeypatch, vllm_runner, fresh_vllm_cache, mega_aot_artifact):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", mega_aot_artifact)

    # Cold start in a forked child (must fork before CUDA init).
    # This model has 32 identical transformer layers which produce
    # 33 subgraphs after splitting on attention — only 3 are unique.
    ctx = mp.get_context("fork")
    p = ctx.Process(target=_cold_start, args=(vllm_runner,))
    p.start()
    p.join()
    assert p.exitcode == 0, "Cold-start child failed"

    # Warm start — compiled artifacts loaded from disk cache.
    counters.clear()
    with compilation_counter.expect(
        num_compiled_artifacts_loaded=3,
        num_compiled_artifacts_saved=0,
    ):
        _run_vllm(vllm_runner)
    if envs.VLLM_USE_MEGA_AOT_ARTIFACT:
        # MEGA_AOT_ARTIFACT is enabled, so we expect no aot_autograd running on
        # subgraphs.
        assert counters["aot_autograd"]["total"] == 0
    else:
        assert counters["aot_autograd"]["total"] == 30
    assert counters["aot_autograd"]["autograd_cache_miss"] == 0
    assert (
        counters["aot_autograd"]["autograd_cache_hit"] == 0
    )  # No miss at aot_autograd level causing disk I/O.
