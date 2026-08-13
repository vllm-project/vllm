# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch

from tests.models.utils import check_logprobs_close
from vllm import SamplingParams
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
        "Qwen/Qwen3-0.6B",
        "openai-community/gpt2",
    ]
    return models


@dataclass(frozen=True)
class DynamicShapesTestCase:
    model_name: str
    shapes_type: DynamicShapesType
    use_aot_compile: bool
    use_bytecode_hook: bool

    def __str__(self) -> str:
        model_id = self.model_name.rsplit("/", 1)[-1]
        return (
            f"{model_id}-{self.shapes_type.value}-"
            f"aot{int(self.use_aot_compile)}-hook{int(self.use_bytecode_hook)}"
        )


def get_dynamic_shapes_test_cases():
    """Pairwise-cover compilation options and smoke-test model classes."""
    reference_model = "Qwen/Qwen3-0.6B"
    cases = [
        DynamicShapesTestCase(reference_model, DynamicShapesType.BACKED, False, True),
        DynamicShapesTestCase(reference_model, DynamicShapesType.BACKED, True, False),
        DynamicShapesTestCase(
            reference_model, DynamicShapesType.UNBACKED, False, False
        ),
        DynamicShapesTestCase(reference_model, DynamicShapesType.UNBACKED, True, True),
        DynamicShapesTestCase(
            reference_model,
            DynamicShapesType.BACKED_SIZE_OBLIVIOUS,
            False,
            True,
        ),
        DynamicShapesTestCase(
            reference_model,
            DynamicShapesType.BACKED_SIZE_OBLIVIOUS,
            True,
            False,
        ),
    ]
    cases.extend(
        DynamicShapesTestCase(model_name, DynamicShapesType.BACKED, False, True)
        for model_name in get_test_models()
        if model_name != reference_model
    )
    return cases


TEST_PROMPTS = ["Hello, my name is", "The capital of France is"]


def generate_outputs(vllm_model):
    sampling_params = SamplingParams(max_tokens=5, temperature=0, logprobs=10)
    outputs = []
    for prompt in TEST_PROMPTS:
        output = vllm_model.llm.generate(prompt, sampling_params)[0].outputs[0]
        assert len(output.text.strip()) > 0, "Model produced empty output"
        outputs.append((output.token_ids, output.text, output.logprobs))
    return outputs


@pytest.fixture(scope="module")
def get_eager_outputs(vllm_runner):
    cache = {}

    def get(model_name):
        if model_name not in cache:
            with vllm_runner(
                model_name,
                enforce_eager=True,
                max_model_len=1024,
                enable_chunked_prefill=None,
            ) as vllm_model:
                cache[model_name] = generate_outputs(vllm_model)
        return cache[model_name]

    return get


@pytest.mark.parametrize("test_case", get_dynamic_shapes_test_cases(), ids=str)
@pytest.mark.skipif(not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10")
def test_dynamic_shapes_compilation(
    monkeypatch,
    vllm_runner,
    get_eager_outputs,
    test_case,
):
    """Test representative dynamic-shapes configurations end to end."""
    if (
        test_case.shapes_type == DynamicShapesType.UNBACKED
        and not is_torch_equal_or_newer("2.11.0")
    ):
        # NOTE[ROCm]: shape_id (used by Qwen2/Llama to relate input dims) only
        # landed in torch 2.11, but the ROCm CI still runs torch 2.10.x. On
        # older torch there's no way to express it, so unbacked shapes go
        # data-dependent and compilation blows up -- nothing to test.
        pytest.skip("unbacked dynamic shapes with shape_id require torch>=2.11")

    monkeypatch.setenv(
        "VLLM_USE_AOT_COMPILE", "1" if test_case.use_aot_compile else "0"
    )
    monkeypatch.setenv(
        "VLLM_USE_BYTECODE_HOOK", "1" if test_case.use_bytecode_hook else "0"
    )

    print(f"Testing {test_case.shapes_type.name} dynamic shapes...")

    # The compiled engine shuts down before an uncached eager baseline starts.
    with vllm_runner(
        test_case.model_name,
        compilation_config={
            "mode": CompilationMode.VLLM_COMPILE,
            "dynamic_shapes_config": {
                "type": test_case.shapes_type.value,
            },
        },
        max_model_len=1024,
        enable_chunked_prefill=None,
    ) as vllm_model:
        compiled_outputs = generate_outputs(vllm_model)

    check_logprobs_close(
        outputs_0_lst=get_eager_outputs(test_case.model_name),
        outputs_1_lst=compiled_outputs,
        name_0="eager",
        name_1="compiled",
    )


@pytest.mark.parametrize("use_aot_compile", [False, True])
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
        use_aot_compile
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
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "1" if use_aot_compile else "0")
    monkeypatch.setenv("VLLM_USE_BYTECODE_HOOK", "0")

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
def test_piecewise_backend_empty_sym_shape_indices(vllm_runner):
    """Test that PiecewiseBackend handles empty sym_shape_indices correctly.

    When all inputs have static shapes (no torch.SymInt), sym_shape_indices
    will be empty. The fix in PiecewiseBackend.__call__ handles this case
    by using the first compiled range_entry.
    """
    # Use small max_model_len and max_num_batched_tokens to encourage
    # static shape compilation with empty sym_shape_indices
    with vllm_runner(
        "Qwen/Qwen3-0.6B",
        max_model_len=512,
        max_num_batched_tokens=1,
        enable_chunked_prefill=None,
        compilation_config={
            "mode": CompilationMode.VLLM_COMPILE,
            "dynamic_shapes_config": {
                "type": DynamicShapesType.BACKED.value,
            },
        },
    ) as vllm_model:
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

        # Generate with static shape inputs
        output = vllm_model.llm.generate(
            "Hello, my name is", sampling_params=sampling_params
        )
        result = output[0].outputs[0].text
        assert len(result) > 0, "Should generate non-empty output"

        # Generate again to verify compilation works with empty sym_shape_indices
        output = vllm_model.llm.generate(
            "The capital of France is", sampling_params=sampling_params
        )
        result = output[0].outputs[0].text
        assert len(result) > 0, "Should generate non-empty output on second run"
