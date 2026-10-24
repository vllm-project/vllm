# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses

import pytest

from vllm.compilation.counter import compilation_counter
from vllm.config import CompilationConfig, CompilationMode
from vllm.platforms import current_platform

from ...models.utils import check_logprobs_close
from ...utils import compare_all_settings

ATTN_BACKEND = "FLASH_ATTN" if not current_platform.is_rocm() else "ROCM_ATTN"


@dataclasses.dataclass
class TestSetting:
    model: str
    model_args: list[str]
    pp_size: int
    tp_size: int
    attn_backend: str
    method: str


# we cannot afford testing the full Cartesian product
# of all models and all modes
@pytest.mark.parametrize(
    "test_setting",
    [
        # MoE model
        TestSetting(
            model="ibm-granite/granite-3.0-1b-a400m-instruct",
            model_args=["--max-model-len", "2048"],
            pp_size=1,
            tp_size=1,
            attn_backend=ATTN_BACKEND,
            method="generate",
        ),
        # embedding model
        TestSetting(
            model="BAAI/bge-multilingual-gemma2",
            model_args=[
                "--runner",
                "pooling",
                "--dtype",
                "bfloat16",
                "--max-model-len",
                "2048",
                "--gpu-memory-utilization",
                "0.98",
            ],
            pp_size=1,
            tp_size=1,
            attn_backend=ATTN_BACKEND,
            method="encode",
        ),
    ],
)
@pytest.mark.parametrize("use_v2_model_runner", [False, True])
def test_compile_correctness(
    test_setting: TestSetting,
    use_v2_model_runner: bool,
):
    model = test_setting.model
    model_args = test_setting.model_args
    pp_size = test_setting.pp_size
    tp_size = test_setting.tp_size
    attn_backend = test_setting.attn_backend
    method = test_setting.method
    final_args = [
        *model_args,
        "-pp",
        str(pp_size),
        "-tp",
        str(tp_size),
        "-cc.cudagraph_mode=none",
        f"--attention-backend={attn_backend}",
    ]

    # Pin every compared setting to the same model runner so that only the
    # compilation mode differs within a comparison.
    runner_env = {"VLLM_USE_V2_MODEL_RUNNER": str(int(use_v2_model_runner))}

    all_args: list[list[str]] = []
    all_envs: list[dict[str, str] | None] = []

    # Test all compilation modes with inductor backend
    for mode in [
        CompilationMode.NONE,
        CompilationMode.STOCK_TORCH_COMPILE,
        CompilationMode.DYNAMO_TRACE_ONCE,
        CompilationMode.VLLM_COMPILE,
    ]:
        all_args.append(final_args + [f"-cc.mode={mode.name}", "-cc.backend=inductor"])
        all_envs.append(dict(runner_env))
    # inductor will change the output, so we only compare if the output
    # is close, not exactly the same.
    compare_all_settings(
        model,
        all_args,
        all_envs,
        method=method if method != "generate" else "generate_close",
    )

    all_envs.clear()
    all_args.clear()

    # Test all compilation modes with eager backend
    for mode in [
        CompilationMode.NONE,
        CompilationMode.STOCK_TORCH_COMPILE,
        CompilationMode.DYNAMO_TRACE_ONCE,
        CompilationMode.VLLM_COMPILE,
    ]:
        all_args.append(final_args + [f"-cc.mode={mode.name}", "-cc.backend=eager"])
        all_envs.append(dict(runner_env))
    compare_all_settings(model, all_args, all_envs, method=method)


def test_moe_splitting_op_piecewise_cudagraph(
    vllm_runner, example_prompts, monkeypatch
):
    """PIECEWISE graphs split at the MoE op match graphs split at attention
    only."""
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    moe_split = [
        *CompilationConfig._attention_ops,
        "vllm::unified_kv_cache_update",
        "vllm::unified_mla_kv_cache_update",
        "vllm::moe_forward",
    ]
    outputs = []
    num_captured = []
    for splitting_ops in (None, moe_split):
        captured_before = compilation_counter.num_cudagraph_captured
        with vllm_runner(
            "ibm-granite/granite-3.0-1b-a400m-instruct",
            max_num_seqs=16,
            compilation_config={
                "cudagraph_mode": "PIECEWISE",
                "splitting_ops": splitting_ops,
            },
        ) as llm:
            outputs.append(
                llm.generate_greedy_logprobs(
                    example_prompts, max_tokens=32, num_logprobs=5
                )
            )
        num_captured.append(
            compilation_counter.num_cudagraph_captured - captured_before
        )
    # Splitting at the MoE op adds pieces to capture.
    assert num_captured[1] > num_captured[0]
    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="attention split",
        name_1="MoE split",
    )
