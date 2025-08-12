import dataclasses
from typing import Dict, List, Optional

import pytest

from vllm.compilation.levels import CompilationLevel
from vllm.utils import cuda_device_count_stateless

from ..utils import compare_all_settings


@dataclasses.dataclass
class TestSetting:
    model: str
    model_args: List[str]
    pp_size: int
    tp_size: int
    attn_backend: str
    method: str
    fullgraph: bool


# representative settings for testing
test_settings = [
    # basic llama model
    TestSetting(
        model="meta-llama/Llama-3.2-1B",
        model_args=[],
        pp_size=2,
        tp_size=2,
        attn_backend="FLASHINFER",
        method="generate",
        fullgraph=True,
    ),
    # llama model with quantization
    TestSetting(
        model="TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ",
        model_args=["--quantization", "gptq"],
        pp_size=1,
        tp_size=1,
        attn_backend="FLASH_ATTN",
        method="generate",
        fullgraph=True,
    ),
    # MoE model
    TestSetting(
        model="ibm/PowerMoE-3b",
        model_args=[],
        pp_size=1,
        tp_size=2,
        attn_backend="FLASH_ATTN",
        method="generate",
        fullgraph=True,
    ),
    # embedding model
    TestSetting(
        model="BAAI/bge-multilingual-gemma2",
        model_args=["--task", "embedding"],
        pp_size=1,
        tp_size=1,
        attn_backend="FLASHINFER",
        method="encode",
        fullgraph=True,
    ),
    # vision language model
    TestSetting(
        model="microsoft/Phi-3.5-vision-instruct",
        model_args=["--trust-remote-code", "--max-model-len", "2048"],
        pp_size=2,
        tp_size=1,
        attn_backend="FLASH_ATTN",
        method="generate_with_image",
        fullgraph=False,
    ),
]


# we cannot afford testing the full Catesian product
# of all models and all levels
@pytest.mark.parametrize("test_setting", test_settings)
def test_compile_correctness(test_setting: TestSetting):
    # this test is run under multiple suits, with different GPUs.
    # make sure we only run the test with correct CUDA devices.
    # don't use "<", as it will duplicate the tests.
    model = test_setting.model
    model_args = test_setting.model_args
    pp_size = test_setting.pp_size
    tp_size = test_setting.tp_size
    attn_backend = test_setting.attn_backend
    method = test_setting.method
    fullgraph = test_setting.fullgraph
    if cuda_device_count_stateless() != pp_size * tp_size:
        pytest.skip("Not correct CUDA devices for the test.")
    import os
    os.environ["VLLM_ATTENTION_BACKEND"] = attn_backend
    final_args = ["--enforce-eager"] + model_args + ["-pp", str(pp_size)] + \
                ["-tp", str(tp_size)]

    all_envs: List[Optional[Dict[str, str]]] = []

    for level in [
            CompilationLevel.NO_COMPILATION,
            CompilationLevel.PIECEWISE,
    ]:
        all_envs.append({"VLLM_TORCH_COMPILE_LEVEL": str(level)})

    # inductor will change the output, so we only compare if the output
    # is close, not exactly the same.
    compare_all_settings(
        model, [final_args] * 2,
        all_envs,
        method=method if method != "generate" else "generate_close")
    all_envs.clear()

    for level in [
            CompilationLevel.NO_COMPILATION,
            CompilationLevel.DYNAMO_AS_IS,
            CompilationLevel.DYNAMO_ONCE,
    ]:
        all_envs.append({"VLLM_TORCH_COMPILE_LEVEL": str(level)})
        if level != CompilationLevel.DYNAMO_ONCE and not fullgraph:
            # "DYNAMO_ONCE" will always use fullgraph
            all_envs[-1][
                "VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE"] = "0"  # type: ignore

    compare_all_settings(model, [final_args] * 3, all_envs, method=method)
