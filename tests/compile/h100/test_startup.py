# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold start and warm start tests for vLLM-compile.

Cold start runs in a forked child (must fork before CUDA init) which
populates on-disk caches and asserts cold-start counters.  Warm start
then runs in the parent with clean in-memory state but populated caches.
"""

import multiprocessing as mp
from typing import NamedTuple

import pytest
from torch._dynamo.utils import counters

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode, PassConfig
from vllm.utils.torch_utils import is_torch_equal_or_newer

from ...utils import fork_new_process_for_each_test

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
    mega_aot_active = envs.VLLM_USE_MEGA_AOT_ARTIFACT and is_torch_equal_or_newer(
        "2.10.0"
    )
    if mega_aot_active:
        # MEGA_AOT_ARTIFACT is enabled, so we expect no aot_autograd running on
        # subgraphs.
        assert counters["aot_autograd"]["total"] == 0
    else:
        assert counters["aot_autograd"]["total"] == 30
    assert counters["aot_autograd"]["autograd_cache_miss"] == 0
    assert (
        counters["aot_autograd"]["autograd_cache_hit"] == 0
    )  # No miss at aot_autograd level causing disk I/O.


# ---------------------------------------------------------------------------
# Parametrized model startup tests
# ---------------------------------------------------------------------------


class ModelStartupSpec(NamedTuple):
    model: str
    hf_overrides: dict
    cold_artifacts_saved: int
    warm_artifacts_saved: int
    warm_artifacts_loaded: int


_SMALL_MOE_OVERRIDES = {
    "num_hidden_layers": 8,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "n_routed_experts": 8,
}

MODEL_SPECS = [
    pytest.param(
        ModelStartupSpec(
            model="openai/gpt-oss-120b",
            hf_overrides={
                "num_hidden_layers": 8,
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "num_local_experts": 8,
            },
            cold_artifacts_saved=3,
            warm_artifacts_saved=0,
            warm_artifacts_loaded=3,
        ),
        id="gpt_oss_120b",
    ),
    # NOTE: DeepSeek-V3.2 requires sparse MLA (index_topk) which needs
    # Hopper+ GPUs. This test must run on H100 (see pytorch.yaml).
    pytest.param(
        ModelStartupSpec(
            model="deepseek-ai/DeepSeek-V3.2",
            hf_overrides=_SMALL_MOE_OVERRIDES,
            cold_artifacts_saved=4,
            # https://github.com/vllm-project/vllm/issues/38051
            warm_artifacts_saved=0 if is_torch_equal_or_newer("2.12.0") else 4,
            warm_artifacts_loaded=4 if is_torch_equal_or_newer("2.12.0") else 0,
        ),
        id="deepseek_v3.2",
    ),
    pytest.param(
        ModelStartupSpec(
            model="moonshotai/Kimi-K2.5",
            hf_overrides={"text_config": _SMALL_MOE_OVERRIDES},
            cold_artifacts_saved=4,
            # https://github.com/vllm-project/vllm/issues/38051
            warm_artifacts_saved=0 if is_torch_equal_or_newer("2.12.0") else 4,
            warm_artifacts_loaded=4 if is_torch_equal_or_newer("2.12.0") else 0,
        ),
        id="kimi_k2.5",
    ),
    pytest.param(
        ModelStartupSpec(
            model="zai-org/GLM-4.5",
            hf_overrides=_SMALL_MOE_OVERRIDES,
            cold_artifacts_saved=4,
            warm_artifacts_saved=0,
            warm_artifacts_loaded=4,
        ),
        id="glm_4.5",
    ),
    pytest.param(
        ModelStartupSpec(
            model="MiniMaxAI/MiniMax-M2.5",
            hf_overrides=_SMALL_MOE_OVERRIDES,
            cold_artifacts_saved=3,
            warm_artifacts_saved=0,
            warm_artifacts_loaded=3,
        ),
        id="minimax_m2.5",
    ),
]


def _run_model(vllm_runner, spec: ModelStartupSpec):
    with vllm_runner(
        spec.model,
        trust_remote_code=True,
        max_model_len=256,
        max_num_batched_tokens=1024,
        block_size=64,
        load_format="dummy",
        hf_overrides=spec.hf_overrides,
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.NONE,
            pass_config=PassConfig(fuse_allreduce_rms=False),
        ),
        num_gpu_blocks_override=8,
    ):
        pass


def _check_model_run(vllm_runner, spec: ModelStartupSpec, is_cold_start: bool):
    """Runs a model and checks the number of compiled artifacts."""
    old = compilation_counter.clone()
    _run_model(vllm_runner, spec)
    saved = (
        compilation_counter.num_compiled_artifacts_saved
        - old.num_compiled_artifacts_saved
    )
    loaded = (
        compilation_counter.num_compiled_artifacts_loaded
        - old.num_compiled_artifacts_loaded
    )

    start_type = "COLD" if is_cold_start else "WARM"
    # Print actual values for debugging — intentional, helps diagnose
    # failures and calibrate expected counts when adding new models.
    print(f"\n=== {start_type} START for {spec.model} ===")
    print(f"  num_compiled_artifacts_saved={saved}")
    print(f"  num_compiled_artifacts_loaded={loaded}")

    if is_cold_start:
        expected_saved = spec.cold_artifacts_saved
        expected_loaded = 0
    else:
        expected_saved = spec.warm_artifacts_saved
        expected_loaded = spec.warm_artifacts_loaded

    assert saved == expected_saved, f"{start_type.lower()}_artifacts_saved: got {saved}"
    assert loaded == expected_loaded, (
        f"{start_type.lower()}_artifacts_loaded: got {loaded}"
    )


def _cold_start_model(vllm_runner, spec: ModelStartupSpec):
    _check_model_run(vllm_runner, spec, is_cold_start=True)


@pytest.mark.parametrize("spec", MODEL_SPECS)
@fork_new_process_for_each_test
def test_model_startup(monkeypatch, vllm_runner, fresh_vllm_cache, spec):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Cold start in a forked child (must fork before CUDA init).
    ctx = mp.get_context("fork")
    p = ctx.Process(target=_cold_start_model, args=(vllm_runner, spec))
    p.start()
    p.join()
    assert p.exitcode == 0, "Cold-start child failed"

    # Warm start — compiled artifacts loaded from disk cache.
    _check_model_run(vllm_runner, spec, is_cold_start=False)
