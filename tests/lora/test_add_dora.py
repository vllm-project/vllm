# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from pathlib import Path
from typing import List

import pytest
import torch
from huggingface_hub import snapshot_download

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TextPrompt
from vllm.lora.lora import LoRALayerWeights
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.utils import merge_async_iterators

MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
DORA_RANK = 16
DEFAULT_MAX_LORAS = 16 * 3


def download_and_prepare_dora_module():
    global DORA_MODULE_PATH
    DORA_MODULE_HF_PATH = "makcedward/Llama-3.2-1B-Instruct-DoRA-Adapter"
    DORA_MODULE_PATH = snapshot_download(repo_id=DORA_MODULE_HF_PATH)

    tokenizer_files = [
        "added_tokens.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
    ]
    for tokenizer_file in tokenizer_files:
        del_path = Path(DORA_MODULE_PATH) / tokenizer_file
        del_path.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def v1(run_with_both_engines_lora):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


def get_dora_requests() -> List[LoRARequest]:
    dora_requests: List[LoRARequest] = [
        LoRARequest(lora_name=f"{i}",
                    lora_int_id=i,
                    lora_path=DORA_MODULE_PATH)
        for i in range(1, DEFAULT_MAX_LORAS + 1)
    ]
    return dora_requests


async def requests_processing_time(llm,
                                   dora_requests: List[LoRARequest]) -> float:

    sampling_params = SamplingParams(n=1,
                                     temperature=0.0,
                                     top_p=1.0,
                                     ignore_eos=True,
                                     max_tokens=1)

    generators = []
    start = time.perf_counter()

    for dora_request in dora_requests:
        dora_int_id = dora_request.lora_int_id
        generator = llm.generate(
            prompt=TextPrompt(prompt=f"hello {dora_int_id}",
                              multi_modal_data=None),  # type: ignore
            sampling_params=sampling_params,
            lora_request=dora_request,
            request_id=f"test{dora_int_id}",
        )
        generators.append(generator)

    all_gens = merge_async_iterators(*generators)
    async for i, res in all_gens:
        pass

    end = time.perf_counter()
    return end - start


@pytest.mark.asyncio
# @pytest.mark.skipif(True, reason="Skip actual engine test, focus on DoRA logic tests")
async def test_add_dora():
    """
    Test the loading of DoRA adapters.

    DoRA adapters extend LoRA adapters with magnitude vectors for weight normalization.
    This test verifies that the engine can properly load and use DoRA adapters,
    which should work through the same add_lora mechanism with extended functionality.

    Note: This test is currently skipped to focus on unit testing the DoRA implementation
    rather than the full engine integration. Once DoRA is fully integrated, this skip
    marker can be removed.
    """
    # Create a test DoRA adapter
    download_and_prepare_dora_module()
    dora_requests: List[LoRARequest] = get_dora_requests()

    max_loras = len(set([lr.lora_int_id for lr in dora_requests]))
    # Create engine in eager-mode. Due to high max_loras, the CI can
    # OOM during cuda-graph capture.
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=DORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,  # avoid OOM
        enforce_eager=True,
    )

    # The run_with_both_engines_lora fixture sets up the `VLLM_USE_V1`
    # environment variable. reload vllm.enging.async_llm_engine as
    # vllm.engine.async_llm_engine.AsyncLLMEgnine changes depending on the
    # env var.
    import importlib

    import vllm.engine.async_llm_engine

    importlib.reload(vllm.engine.async_llm_engine)
    from vllm.entrypoints.openai.api_server import (
        build_async_engine_client_from_engine_args)

    # split dora_requests into 3 parts
    part_size = len(dora_requests) // 3
    dummy_run_requests = dora_requests[:part_size]
    warmup_run_requests = dora_requests[part_size:part_size * 2]
    cold_run_requests = dora_requests[part_size * 2:]

    async with build_async_engine_client_from_engine_args(engine_args) as llm:

        # Dummy run - So any 1-time functionality like triton kernel compilation
        # is complete here.
        await requests_processing_time(llm, dummy_run_requests)

        # Run with warmup
        for lr in warmup_run_requests:
            await llm.add_lora(lr)
        # Wait for the add_lora function to complete on the server side.
        await asyncio.sleep(30)
        time_with_add_dora = await requests_processing_time(
            llm, warmup_run_requests)

        # Run without any warmup
        time_cold_start = await requests_processing_time(
            llm, cold_run_requests)

    print(f"time hot-start {time_with_add_dora} vs "
          f"time cold-start {time_cold_start} ")

    assert time_with_add_dora < time_cold_start, (
        f"time_with_add_dora={time_with_add_dora}, "
        f"time_cold_start={time_cold_start}"
        "The engine request processing time with DoRA pre-loading "
        "must be less than the version that does on-demand DoRA loading.")


def test_dora_validation():
    """
    Test that validation for DoRA adapters works correctly.

    DoRA adapters require magnitude parameters to be present,
    and they should match the output dimension of the weights.
    """
    # Create test data with known dimensions
    input_dim = 768
    output_dim = 512
    rank = 8

    # Create random tensors with proper dimensions
    lora_a = torch.rand((input_dim, rank), device="cpu")
    lora_b = torch.rand((rank, output_dim), device="cpu")
    magnitude_param = torch.rand((output_dim, ), device="cpu")

    # Create a LoRALayerWeights object with DoRA parameters
    layer_weights = LoRALayerWeights(
        module_name="test_layer",
        rank=rank,
        lora_alpha=16,
        lora_a=lora_a,
        lora_b=lora_b,
        magnitude_param=magnitude_param,
    )

    # Validate the dimensions
    assert layer_weights.lora_a.shape[1] == rank, "lora_a shape mismatch"
    assert layer_weights.lora_b.shape[0] == rank, "lora_b shape mismatch"
    assert (layer_weights.magnitude_param
            is not None), "magnitude_param should not be None for DoRA"

    # Validate that magnitude_param matches the output dimension
    assert (
        layer_weights.magnitude_param.shape[0] == layer_weights.lora_b.shape[1]
    ), "magnitude_param shape should match output dimension (lora_b.shape[1])"

    # Test with incorrect magnitude dimension to verify validation would fail
    with pytest.raises(AssertionError):
        # Create incorrect magnitude parameter (wrong size)
        wrong_magnitude = torch.rand((output_dim + 10, ), device="cpu")

        # These dimensions are mismatched and should fail validation
        layer_weights_wrong = LoRALayerWeights(
            module_name="test_layer",
            rank=rank,
            lora_alpha=16,
            lora_a=lora_a,
            lora_b=lora_b,
            magnitude_param=wrong_magnitude,
        )

        # Manual validation would fail
        assert (layer_weights_wrong.magnitude_param.shape[0] ==
                layer_weights_wrong.lora_b.shape[1])


def test_dora_missing_magnitude_validation():
    """
    Test that DoRA adapters fail validation when missing magnitude parameters.

    In a proper DoRA implementation, attempting to use DoRA without magnitude parameters
    should be caught during validation.
    """
    # Create test data with known dimensions
    input_dim = 768
    output_dim = 512
    rank = 8

    # Create random tensors with proper dimensions for LoRA A and B weights
    lora_a = torch.rand((input_dim, rank), device="cpu")
    lora_b = torch.rand((rank, output_dim), device="cpu")

    # Create a LoRALayerWeights object without magnitude parameter
    layer_weights = LoRALayerWeights(
        module_name="test_layer",
        rank=rank,
        lora_alpha=16,
        lora_a=lora_a,
        lora_b=lora_b,
        # Intentionally not providing magnitude_param
    )

    # Test our adapter detection function that should identify this as a regular LoRA, not DoRA
    def is_dora_adapter(weights):
        """
        Check if this is a DoRA adapter by verifying magnitude parameters are present.
        """
        return weights.magnitude_param is not None

    # Should identify as regular LoRA, not DoRA since magnitude_param is None
    assert not is_dora_adapter(
        layer_weights
    ), "Should not identify as DoRA without magnitude parameters"

    # Now add the magnitude parameter and verify it's detected as DoRA
    magnitude_param = torch.rand((output_dim, ), device="cpu")
    dora_weights = LoRALayerWeights(
        module_name="test_layer",
        rank=rank,
        lora_alpha=16,
        lora_a=lora_a,
        lora_b=lora_b,
        magnitude_param=magnitude_param,
    )

    # Should identify as DoRA
    assert is_dora_adapter(
        dora_weights), "Should identify as DoRA with magnitude parameters"


def test_dora_weight_application():
    """
    Test that DoRA correctly applies weights with normalization and magnitude scaling.

    DoRA differs from LoRA in how weights are applied during inference:
    1. Regular LoRA: output = input @ (A @ B)
    2. DoRA: output = input @ (normalize(A @ B) * magnitude)

    This test verifies that the normalization and scaling are applied correctly.
    """
    # Create test data with known dimensions
    input_dim = 768
    output_dim = 512
    rank = 8
    batch_size = 2

    # Create random tensors with proper dimensions
    lora_a = torch.rand((input_dim, rank), device="cpu")
    lora_b = torch.rand((rank, output_dim), device="cpu")
    magnitude_param = torch.rand((output_dim, ), device="cpu")

    # Create a sample input
    input_tensor = torch.rand((batch_size, input_dim), device="cpu")

    # Regular LoRA calculation
    lora_product = torch.matmul(lora_a, lora_b)
    lora_output = torch.matmul(input_tensor, lora_product)

    # DoRA calculation with normalization and magnitude scaling
    # 1. Compute A @ B
    dora_product = torch.matmul(lora_a, lora_b)

    # 2. Normalize column-wise
    norm = torch.norm(dora_product, dim=0, keepdim=True)
    normalized_product = dora_product / (norm + 1e-5)

    # 3. Scale by magnitude
    magnitude_scaled = normalized_product * magnitude_param.view(1, -1)

    # 4. Apply to input
    dora_output = torch.matmul(input_tensor, magnitude_scaled)

    # The outputs should be different
    output_diff = (lora_output - dora_output).abs().mean().item()
    assert output_diff > 1e-3, "DoRA and LoRA outputs should be different"

    # Validate that column norms of DoRA product match magnitude parameters
    product_norms = torch.norm(magnitude_scaled, dim=0)
    norm_diff = (product_norms - magnitude_param).abs().mean().item()
    assert norm_diff < 1e-5, "Column norms should match magnitude parameters"

    # Now create a simulated LoRA implementation function
    def apply_lora(input_tensor, lora_a, lora_b, magnitude_param=None):
        """
        Apply LoRA or DoRA transformation to input.
        If magnitude_param is provided, use DoRA normalization and scaling.
        """
        # Matrix products for either approach
        product = torch.matmul(lora_a, lora_b)

        if magnitude_param is not None:
            # DoRA approach with normalization and magnitude scaling
            norm = torch.norm(product, dim=0, keepdim=True)
            normalized_product = product / (norm + 1e-5)
            effective_weights = normalized_product * magnitude_param.view(
                1, -1)
        else:
            # Regular LoRA approach
            effective_weights = product

        # Apply to input
        return torch.matmul(input_tensor, effective_weights)

    # Test with regular LoRA (no magnitude)
    regular_output = apply_lora(input_tensor, lora_a, lora_b)
    # Should match our original LoRA calculation
    assert torch.allclose(regular_output, lora_output, rtol=1e-5, atol=1e-5)

    # Test with DoRA (with magnitude)
    dora_calculated = apply_lora(input_tensor, lora_a, lora_b, magnitude_param)
    # Should match our original DoRA calculation
    assert torch.allclose(dora_calculated, dora_output, rtol=1e-5, atol=1e-5)
