from typing import Dict, List, Optional

import pytest

from vllm.compilation.levels import CompilationLevel
from vllm.utils import cuda_device_count_stateless

from ..utils import compare_all_settings


# we cannot afford testing the full Catesian product
# of all models and all levels
@pytest.mark.parametrize(
    "model, model_args, pp_size, tp_size, attn_backend, method",
    [
        ("meta-llama/Meta-Llama-3-8B", [], 2, 2, "FLASH_ATTN", "generate"),
        ("nm-testing/Meta-Llama-3-8B-Instruct-W8A8-Dyn-Per-Token-2048-Samples",
         ["--quantization", "compressed-tensors"
          ], 1, 1, "FLASH_ATTN", "generate"),
        ("google/gemma-2-2b-it", [], 1, 2, "FLASHINFER", "generate"),
        # TODO: add multi-modality test for llava
        ("llava-hf/llava-1.5-7b-hf", [], 2, 1, "FLASHINFER", "generate")
    ])
def test_compile_correctness(model, model_args, pp_size, tp_size, attn_backend,
                             method):
    # this test is run under multiple suits, with different GPUs.
    # make sure we only run the test with correct CUDA devices.
    # don't use "<", as it will duplicate the tests.
    if cuda_device_count_stateless() != pp_size * tp_size:
        pytest.skip("Not correct CUDA devices for the test.")
    all_args = [["--enforce-eager"] + model_args + ["--max_model_len", "1024"]
                + ["-pp", str(pp_size)] + ["-tp", str(tp_size)]] * 3
    # don't test VLLM_TORCH_COMPILE_LEVEL == 3 case
    # inductor will change the output, so we cannot compare them.
    all_envs: List[Optional[Dict[str, str]]] = [{
        "VLLM_TORCH_COMPILE_LEVEL":
        str(level)
    } for level in [
        CompilationLevel.NO_COMPILATION,
        CompilationLevel.DYNAMO_AS_IS,
        CompilationLevel.DYNAMO_ONCE,
    ]]
    compare_all_settings(model, all_args, all_envs)
