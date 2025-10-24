# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

import pytest
import torch
from torch.torch_version import TorchVersion

from vllm import LLM, SamplingParams
from vllm.config import set_current_vllm_config
from vllm.config.compilation import DynamicShapesType


def cleanup_gpu_memory():
    """Clean up GPU memory after each test"""
    gc.collect()  # Clear Python objects
    torch.cuda.empty_cache()  # Clear PyTorch GPU memory cache
    torch.cuda.synchronize()  # Wait for all GPU operations to complete


def get_test_models():
    """Get list of models to test based on PyTorch version"""
    # Parse PyTorch version
    result = ["microsoft/DialoGPT-small", "gpt2", "facebook/opt-125m"]
    # Handle alpha versions by removing pre-release suffixes
    version_parts = torch.__version__.split("+")[0].split("a")[0]
    clean_version = version_parts.split("b")[0].split("rc")[0]
    if TorchVersion(clean_version) >= TorchVersion("2.10"):
        # Requires some fixes only available in PyTorch 2.10+
        result.append("Qwen/Qwen2-1.5B-Instruct")
        result.append("Qwen/Qwen2-7B-Instruct")
        result.append("openlm-research/open_llama_13b")

    return result


@pytest.mark.parametrize("model_name", get_test_models())
@pytest.mark.parametrize("evaluate_guards", [False, True])
def test_dynamic_shapes_compilation(monkeypatch, model_name, evaluate_guards):
    """Test that all dynamic shapes types produce compiles"""
    print(f"\nTesting model: {model_name} with evaluate_guards={evaluate_guards}")

    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    # Note USE_AOT_COMPILE fails https://github.com/vllm-project/vllm/issues/27040.
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "0")

    prompt = "Hello, my name is"
    results = {}

    print("Testing EAGER (no compilation) baseline...")
    cleanup_gpu_memory()

    eager_model = LLM(
        model=model_name,
        compilation_config={
            "level": 0,  # NO_COMPILATION - eager mode
        },
        # gpu_memory_utilization=0.2,
    )

    # Generate text with deterministic sampling parameters
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0.0,  # Deterministic generation
        seed=42,  # Fixed seed for consistency
    )
    eager_output = eager_model.generate(prompt, sampling_params=sampling_params)
    results["EAGER"] = eager_output[0].outputs[0].text

    # Cleanup model
    del eager_model
    cleanup_gpu_memory()

    # Test all dynamic shapes types with compilation
    for shapes_type in [
        DynamicShapesType.BACKED,
        DynamicShapesType.UNBACKED,
        DynamicShapesType.BACKED_SIZE_OBLIVIOUS,
    ]:
        print(
            f"Testing {shapes_type.name} dynamic shapes with "
            f"evaluate_guards={evaluate_guards}..."
        )

        # Initialize the model with specific dynamic shapes configuration
        model = LLM(
            model=model_name,
            compilation_config={
                "level": 3,  # PIECEWISE compilation
                "dynamic_shapes_config": {
                    "dynamic_shapes_type": shapes_type.value,
                    "eval_dynamo_ds_guards": evaluate_guards,
                },
            },
            # gpu_memory_utilization=0.2,
        )

        output = model.generate(prompt, sampling_params=sampling_params)

        # Store results for comparison
        results[shapes_type.name] = output[0].outputs[0].text

        # Cleanup model
        del model
        cleanup_gpu_memory()

    # Verify all results are non-empty strings
    for shape_type, result in results.items():
        assert isinstance(result, str), f"{shape_type} should return a string"
        assert len(result.strip()) > 0, f"{shape_type} should generate non-empty text"

    # Print results
    for shape_type, result in results.items():
        print(f"{shape_type}: '{result}'")


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
    """Test that evaluate_guards correctly detects shape specialization violations."""
    from contextlib import contextmanager

    from vllm.compilation.decorators import support_torch_compile
    from vllm.config import CompilationConfig, VllmConfig
    from vllm.config.compilation import DynamicShapesConfig
    from vllm.forward_context import set_forward_context

    @support_torch_compile
    class ModelWithSizeCheck(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, x: torch.Tensor):
            x = self.linear(x)
            # This will cause specialization - torch.compile will guard on x.shape[0]
            if x.shape[0] >= 10:
                return x
            else:
                return x

    @contextmanager
    def use_vllm_config(vllm_config: VllmConfig):
        with set_forward_context({}, vllm_config), set_current_vllm_config(vllm_config):
            yield

    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", use_aot_compile)

    # Reset torch dynamo to clear any cached compilation state
    torch._dynamo.reset()

    config_desc = (
        f"AOT={use_aot_compile}, shapes={dynamic_shapes_type.name}, "
        f"eval_guards={evaluate_guards}"
    )
    print(f"\n{'=' * 60}")
    print(f"Testing: {config_desc}")
    print(f"{'=' * 60}")

    # Create vllm config with the desired settings
    from vllm.config import CompilationMode

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            dynamic_shapes_config=DynamicShapesConfig(
                dynamic_shapes_type=dynamic_shapes_type,
                evaluate_guards=evaluate_guards,
            ),
        )
    )

    assert (
        vllm_config.compilation_config.dynamic_shapes_config.evaluate_guards
        == evaluate_guards
    )
    with torch.no_grad(), use_vllm_config(vllm_config):
        model = ModelWithSizeCheck(vllm_config=vllm_config).cuda()

        # First call with size 20 - should always work
        input_10 = torch.randn(20, 10).cuda()
        model(input_10)

        # Second call with different size (5) - behavior depends on evaluate_guards
        input_5 = torch.randn(5, 10).cuda()

        # Allow recompiles for evaluate_guards=False case
        # Only when evaluate_guards=True do we want to detect guard violations
        if evaluate_guards:
            # With evaluate_guards=True, this should fail because
            # guards were added. The model specialized on size 10,
            # so size 5 violates the guard
            try:
                model(input_5)
                # If we get here, no guard violation occurred
                # This is a TEST FAILURE - evaluate_guards should have caused a failure
                pytest.fail(
                    f"{config_desc}: Expected guard violation did "
                    f"not occur! evaluate_guards=True should fail "
                    f"when shape changes from 10 to 5, but the "
                    f"model ran successfully without error."
                )
            except Exception as e:
                # Expected failure - guard was violated
                error_msg = str(e)
                if "guard" in error_msg.lower() or "recompile" in error_msg.lower():
                    print(f"✅ {config_desc}: Expected failure due to guard violation")
                    print(f"   Error (truncated): {error_msg[:150]}")
                else:
                    # Unexpected error type
                    print(f"❌ {config_desc}: Unexpected error type")
                    print(f"   Error: {e}")
                    raise
        else:
            # With evaluate_guards=False, guards are dropped, so this should work
            # However, recompilation may still occur, which is expected
            try:
                output_5 = model(input_5)
                assert output_5.shape == (
                    5,
                    10,
                ), "Output shape should match input"
                print(f"✅ {config_desc}: Passed without guard violations")
                print("   Second call (size 5): Success")
            except RuntimeError as e:
                # If it's a recompile error, that's expected when evaluate_guards=False
                # The model is allowed to recompile with different shapes
                if (
                    "recompile" in str(e).lower()
                    and "fail_on_recompile" in str(e).lower()
                ):
                    print(f"✅ {config_desc}: Recompile occurred (expected behavior)")
                    print("   Recompiles are allowed when evaluate_guards=False")
                else:
                    print(f"❌ {config_desc}: Unexpected failure")
                    print(f"   Error: {e}")
                    raise

    cleanup_gpu_memory()
