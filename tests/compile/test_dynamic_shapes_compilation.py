# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import tempfile
from contextlib import contextmanager

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.config.compilation import (
    CompilationMode,
    DynamicShapesConfig,
    DynamicShapesType,
)
from vllm.forward_context import set_forward_context
from vllm.platforms import current_platform
from vllm.tokenizers import get_tokenizer
from vllm.utils.torch_utils import is_torch_equal_or_newer


def get_test_models():
    """Get list of models to test based on PyTorch version"""
    # TODO "Qwen/Qwen3-4B-Instruct-2507" fails Fix issue and support it.
    return ["gpt2", "Qwen/Qwen2-7B-Instruct", "meta-llama/Llama-3.1-8B"]


@pytest.mark.parametrize("model_name", get_test_models())
@pytest.mark.parametrize(
    "shapes_type",
    [
        DynamicShapesType.BACKED,
        DynamicShapesType.UNBACKED,
        DynamicShapesType.BACKED_SIZE_OBLIVIOUS,
    ],
)
@pytest.mark.parametrize("use_aot_compile", ["0", "1"])
@pytest.mark.parametrize("use_bytecode_hook", [True, False])
@pytest.mark.parametrize("evaluate_guards", [False, True])
@pytest.mark.skipif(not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10")
def test_dynamic_shapes_compilation(
    monkeypatch,
    model_name,
    shapes_type,
    use_aot_compile,
    use_bytecode_hook,
    evaluate_guards,
):
    """Test that all dynamic shapes types compile successfully"""
    if use_bytecode_hook and shapes_type == DynamicShapesType.UNBACKED:
        pytest.skip("UNBACKED dynamic shapes require VLLM_USE_BYTECODE_HOOK=0")

    if evaluate_guards and shapes_type == DynamicShapesType.UNBACKED:
        pytest.skip("unbacked dynamic shapes do not add guards")

    if evaluate_guards and use_aot_compile:
        pytest.skip("evaluate_guards requires use_aot_compile=0")

    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", use_aot_compile)
    monkeypatch.setenv("VLLM_USE_BYTECODE_HOOK", "1" if use_bytecode_hook else "0")

    prompt = "Hello, my name is"

    print(f"Testing {shapes_type.name} dynamic shapes...")

    # Initialize the model with specific dynamic shapes configuration
    model = LLM(
        model=model_name,
        compilation_config={
            "mode": CompilationMode.VLLM_COMPILE,
            "dynamic_shapes_config": {
                "type": shapes_type.value,
                "evaluate_guards": evaluate_guards,
            },
        },
        max_model_len=1024,
    )

    output = model.generate(prompt)
    result = output[0].outputs[0].text
    # Example of setting the sampling parameters
    tokenizer = get_tokenizer(model_name)
    yes_tokens = tokenizer.encode("yes", add_special_tokens=False)
    no_tokens = tokenizer.encode("no", add_special_tokens=False)
    allowed_ids = list(set(yes_tokens + no_tokens))
    sampling_params = SamplingParams(
        max_tokens=1, temperature=0, allowed_token_ids=allowed_ids
    )

    output = model.generate(
        "answer with yes or no is " + result + " rubbish for prompt " + prompt + "?",
        sampling_params=sampling_params,
    )
    result = output[0].outputs[0].text
    assert result == "yes"

    # Clean up GPU memory
    del model
    gc.collect()
    current_platform.empty_cache()
    current_platform.synchronize()
    print("GPU memory cleared")


@pytest.mark.parametrize("use_aot_compile", ["0", "1"])
@pytest.mark.parametrize(
    "dynamic_shapes_type",
    [
        DynamicShapesType.BACKED,
        DynamicShapesType.BACKED_SIZE_OBLIVIOUS,
    ],
)
@pytest.mark.parametrize("evaluate_guards", [False, True])
def test_model_specialization_with_evaluate_guards(
    monkeypatch, use_aot_compile, dynamic_shapes_type, evaluate_guards
):
    """Test that evaluate_guards correctly detects shape specialization
    violations.
    """

    if (
        use_aot_compile == "1"
        and dynamic_shapes_type == DynamicShapesType.BACKED
        and evaluate_guards
    ):
        pytest.skip("evaluate_guards for backed does not work with aot_compile=1")

    @support_torch_compile
    class ModelWithSizeCheck(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, x: torch.Tensor):
            # This will cause specialization - torch.compile will guard on
            # sx.shape[0]
            if x.shape[0] >= 10:
                return x * 10
            else:
                return x * 10

    @support_torch_compile
    class ModelWithOneSizeCheck(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, x: torch.Tensor):
            # This will cause 0/1 specializations.
            if x.shape[0] == 0:
                return x * 10
            if x.shape[0] == 1:
                return x * 10
            else:
                return x * 10

    @contextmanager
    def use_vllm_config(vllm_config: VllmConfig):
        with set_forward_context({}, vllm_config), set_current_vllm_config(vllm_config):
            yield

    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", use_aot_compile)
    monkeypatch.setenv("VLLM_USE_BYTECODE_HOOK", "0")

    # Create vllm config with the desired settings
    from vllm.config import CompilationMode

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            dynamic_shapes_config=DynamicShapesConfig(
                type=dynamic_shapes_type,
                evaluate_guards=evaluate_guards,
            ),
        )
    )

    def test(model_class, input1, input2, is_01_specialization=False):
        with (
            torch.no_grad(),
            use_vllm_config(vllm_config),
            tempfile.TemporaryDirectory() as tmpdirname,
        ):
            monkeypatch.setenv("VLLM_CACHE_ROOT", tmpdirname)

            model = model_class(vllm_config=vllm_config).to(
                current_platform.device_type
            )

            model(input1)

            if evaluate_guards and (
                not (
                    is_01_specialization
                    and dynamic_shapes_type == DynamicShapesType.BACKED
                )
            ):
                # This should fail because guards were added.
                with pytest.raises(RuntimeError) as excinfo:
                    model(input2)

                # Expected failure - guard was violated
                error_msg = str(excinfo.value)
                assert (
                    "GuardManager check failed" in error_msg
                    or "Detected recompile when torch.compile stance" in error_msg
                ), error_msg

            else:
                model(input2)

    test(
        ModelWithSizeCheck,
        torch.randn(20, 10).to(current_platform.device_type),
        torch.randn(5, 10).to(current_platform.device_type),
    )
    test(
        ModelWithSizeCheck,
        torch.randn(5, 10).to(current_platform.device_type),
        torch.randn(20, 10).to(current_platform.device_type),
    )
    test(
        ModelWithOneSizeCheck,
        torch.randn(20, 10).to(current_platform.device_type),
        torch.randn(1, 10).to(current_platform.device_type),
        is_01_specialization=True,
    )
