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


test_settings = [
    TestSetting(
        model="meta-llama/Llama-3.2-1B",
        model_args=[],
        pp_size=2,
        tp_size=2,
        attn_backend="FLASHINFER",
        method="generate",
        fullgraph=True,
    ),
    TestSetting(
        model=
        "nm-testing/Meta-Llama-3-8B-Instruct-W8A8-Dyn-Per-Token-2048-Samples",
        model_args=["--quantization", "compressed-tensors"],
        pp_size=1,
        tp_size=1,
        attn_backend="FLASH_ATTN",
        method="generate",
        fullgraph=True,
    ),
    TestSetting(
        model="ibm/PowerMoE-3b",
        model_args=[],
        pp_size=1,
        tp_size=2,
        attn_backend="FLASH_ATTN",
        method="generate",
        fullgraph=True,
    ),
    TestSetting(
        model="BAAI/bge-multilingual-gemma2",
        model_args=["--task", "embedding"],
        pp_size=1,
        tp_size=1,
        attn_backend="FLASHINFER",
        method="encode",
        fullgraph=True,
    ),
    TestSetting(
        model="llava-hf/llava-1.5-7b-hf",
        model_args=[],
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
    all_args = [["--enforce-eager"] + model_args + ["-pp", str(pp_size)] +
                ["-tp", str(tp_size)]] * 3
    # don't test VLLM_TORCH_COMPILE_LEVEL == 3 case
    # inductor will change the output, so we cannot compare them.
    all_envs: List[Optional[Dict[str, str]]] = []
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

    compare_all_settings(model, all_args, all_envs, method=method)
