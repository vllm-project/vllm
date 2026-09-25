# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses

from vllm.config import CompilationMode
from vllm.platforms import current_platform

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


GRANITE_SETTING = TestSetting(
    model="ibm-granite/granite-3.0-1b-a400m-instruct",
    model_args=["--max-model-len", "2048"],
    pp_size=1,
    tp_size=1,
    attn_backend=ATTN_BACKEND,
    method="generate",
)

BGE_MULTILINGUAL_GEMMA2_SETTING = TestSetting(
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
)


def run_compile_correctness(
    test_setting: TestSetting,
    backend: str,
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

    all_args: list[list[str]] = []
    all_envs: list[dict[str, str] | None] = []

    # Test all compilation modes with the given backend
    for mode in [
        CompilationMode.NONE,
        CompilationMode.STOCK_TORCH_COMPILE,
        CompilationMode.DYNAMO_TRACE_ONCE,
        CompilationMode.VLLM_COMPILE,
    ]:
        all_args.append(
            final_args + [f"-cc.mode={mode.name}", f"-cc.backend={backend}"]
        )
        all_envs.append({})
    # inductor will change the output, so we only compare if the output
    # is close, not exactly the same.
    if backend == "inductor" and method == "generate":
        method = "generate_close"
    compare_all_settings(model, all_args, all_envs, method=method, force_v1_runner=True)
