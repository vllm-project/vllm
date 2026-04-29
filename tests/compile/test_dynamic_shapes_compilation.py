# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import tempfile
from contextlib import contextmanager

import pytest
import torch

from tests.models.utils import check_logprobs_close
from vllm import LLM, SamplingParams
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.config.compilation import (
    CompilationMode,
    DynamicShapesConfig,
    DynamicShapesType,
)
from vllm.forward_context import set_forward_context
from vllm.utils.torch_utils import is_torch_equal_or_newer


def get_test_models():
    """Get list of models to test based on PyTorch version"""
    models = [
        "gpt2",
        "Qwen/Qwen2-7B-Instruct",
        "meta-llama/Llama-3.1-8B",
    ]
    if is_torch_equal_or_newer("2.12.0.dev"):
        models.append("Qwen/Qwen3-4B-Instruct-2507")
    return models


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

    sampling_params = SamplingParams(max_tokens=5, temperature=0, logprobs=10)
    test_prompts = [prompt, "The capital of France is"]

    compiled_outputs = []
    for p in test_prompts:
        output = model.generate(p, sampling_params)[0].outputs[0]
        assert len(output.text.strip()) > 0, "Compiled model produced empty output"
        compiled_outputs.append((output.token_ids, output.text, output.logprobs))

    del model
    gc.collect()
    torch.accelerator.empty_cache()
    torch.accelerator.synchronize()

    eager_model = LLM(model=model_name, enforce_eager=True, max_model_len=1024)
    eager_outputs = []
    for p in test_prompts:
        output = eager_model.generate(p, sampling_params)[0].outputs[0]
        assert len(output.text.strip()) > 0, "Eager model produced empty output"
        eager_outputs.append((output.token_ids, output.text, output.logprobs))
    del eager_model
    gc.collect()
    torch.accelerator.empty_cache()
    torch.accelerator.synchronize()

    check_logprobs_close(
        outputs_0_lst=eager_outputs,
        outputs_1_lst=compiled_outputs,
        name_0="eager",
        name_1="compiled",
    )


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

            model = model_class(vllm_config=vllm_config).cuda()

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

    test(ModelWithSizeCheck, torch.randn(20, 10).cuda(), torch.randn(5, 10).cuda())
    test(ModelWithSizeCheck, torch.randn(5, 10).cuda(), torch.randn(20, 10).cuda())
    test(
        ModelWithOneSizeCheck,
        torch.randn(20, 10).cuda(),
        torch.randn(1, 10).cuda(),
        is_01_specialization=True,
    )


@pytest.mark.skipif(not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10")
def test_piecewise_backend_empty_sym_shape_indices():
    """Test that PiecewiseBackend handles empty sym_shape_indices correctly.

    When all inputs have static shapes (no torch.SymInt), sym_shape_indices
    will be empty. The fix in PiecewiseBackend.__call__ handles this case
    by using the first compiled range_entry.
    """
    gc.collect()
    torch.accelerator.empty_cache()
    torch.accelerator.synchronize()

    # Use small max_model_len and max_num_batched_tokens to encourage
    # static shape compilation with empty sym_shape_indices
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        max_model_len=512,
        max_num_batched_tokens=1,
        compilation_config={
            "mode": CompilationMode.VLLM_COMPILE,
            "dynamic_shapes_config": {
                "type": DynamicShapesType.BACKED.value,
            },
        },
    )

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    # Generate with static shape inputs
    output = llm.generate("Hello, my name is", sampling_params=sampling_params)
    result = output[0].outputs[0].text
    assert len(result) > 0, "Should generate non-empty output"

    # Generate again to verify compilation works with empty sym_shape_indices
    output = llm.generate("The capital of France is", sampling_params=sampling_params)
    result = output[0].outputs[0].text
    assert len(result) > 0, "Should generate non-empty output on second run"

    del llm
    gc.collect()
    torch.accelerator.empty_cache()
    torch.accelerator.synchronize()
