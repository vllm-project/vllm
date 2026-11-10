# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# flake8: noqa
"""Tests Model Optimizer nvfp4 models against ground truth generation
Note: these tests will only pass on B200
"""

import os
from typing import List
from unittest.mock import Mock

import pytest
from transformers import AutoTokenizer

from tests.quantization.utils import is_quant_method_supported
from vllm import LLM, SamplingParams
from vllm.model_executor.kernels.linear import HummingHadamardNvFp4LinearKernel

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsW4A4Fp4,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (  # noqa: E501
    CompressedTensorsLinearTransformMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.schemes import (  # noqa: E501
    linear_humming_nvfp4,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.schemes.linear_humming_nvfp4 import (  # noqa: E501
    HummingNvFP4HadamardLinearMethod,
)
from vllm.platforms import current_platform

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_MODEL_LEN = 1024

MODELS = ["nvidia/Llama-3.3-70B-Instruct-FP4"]

EXPECTED_STRS_MAP = {
    "nvidia/Llama-3.3-70B-Instruct-FP4": [
        "vLLM (Vectorized Large Language Model) is indeed a high-throughput and memory-efficient inference",
        "Here are the major milestones in the development of artificial intelligence (AI) from 1950 to ",
        "Artificial intelligence (AI) and human intelligence (HI) are two distinct forms of intelligence that process",
        "A neural network is a type of machine learning model inspired by the structure and function of the human brain",
        "In the heart of a cutting-edge robotics lab, a team of engineers had been working tirelessly to push",
        "The COVID-19 pandemic has had a profound impact on global economic structures and future business models, leading",
        "The Mona Lisa, painted by Leonardo da Vinci in the early 16th century, is one of",
        "Here are the translations:\n\n* Japanese: (Sasuga no tori ga miwa o ts",
    ]
}


# This test compares against golden strings for exact match since
# there is no baseline implementation to compare against
# and is unstable w.r.t specifics of the fp4 implementation or
# the hardware being run on.
# Disabled to prevent it from breaking the build
@pytest.mark.skip(
    reason="Prevent unstable test based on golden strings from breaking the build "
    " and test input model being too large and hanging the system."
)
@pytest.mark.skipif(
    not is_quant_method_supported("modelopt_fp4"),
    reason="modelopt_fp4 is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_name", MODELS)
def test_models(example_prompts, model_name) -> None:
    llm = LLM(
        model=model_name,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        enforce_eager=True,
        quantization="modelopt_fp4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in example_prompts
    ]
    params = SamplingParams(max_tokens=20, temperature=0)
    generations: List[str] = []
    # Note: these need to be run 1 at a time due to numerical precision,
    # since the expected strs were generated this way.
    for prompt in formatted_prompts:
        outputs = llm.generate(prompt, params)
        generations.append(outputs[0].outputs[0].text)
    del llm

    print(model_name, generations)
    expected_strs = EXPECTED_STRS_MAP[model_name]
    for i in range(len(example_prompts)):
        generated_str = generations[i]
        expected_str = expected_strs[i]
        assert expected_str == generated_str, (
            f"Test{i}:\nExpected: {expected_str!r}\nvLLM: {generated_str!r}"
        )


EAGER = [True, False]

SM_100_NVFP4_BACKENDS = [
    "flashinfer_cutedsl",
    "flashinfer_cudnn",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "torch",
]


@pytest.mark.parametrize("model", ["nvidia/Llama-3.1-8B-Instruct-NVFP4"])
@pytest.mark.parametrize("eager", EAGER)
@pytest.mark.parametrize(
    "backend",
    [
        "emulation",
        "flashinfer_cutedsl",
        "flashinfer_cudnn",
        "flashinfer_trtllm",  # the small seq_len ensures trtllm_8x4_layout backend is used
        "flashinfer_cutlass",
        "torch",
    ],
)
def test_nvfp4(vllm_runner, model, eager, backend):
    if backend == "flashinfer_cutedsl" and not (
        current_platform.is_device_capability_family(100)
    ):
        pytest.skip("The flashinfer_cutedsl backend is only supported on SM10x")

    if (
        not current_platform.has_device_capability(100)
        and backend in SM_100_NVFP4_BACKENDS
    ):
        pytest.skip(
            f"The backend {backend} is not supported with current_platform.has_device_capability(100) == False"
        )

    with vllm_runner(model, enforce_eager=eager, linear_backend=backend) as llm:
        output = llm.generate_greedy(["1 2 3 4 5"], max_tokens=2)
    assert output[0][1] == "1 2 3 4 5 6"


@pytest.mark.parametrize(
    "model",
    [
        "nvidia/Qwen3-30B-A3B-NVFP4",
        "RedHatAI/Qwen3-30B-A3B-NVFP4",
    ],
)
@pytest.mark.parametrize("backend", ["emulation"])
@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="NVFP4 MOE emulation is only useful on AMD Instinct MI3xx",
)
def test_nvfp4_moe(vllm_runner, model, backend):
    with vllm_runner(
        model,
        moe_backend=backend,
        linear_backend=backend,
        load_format="dummy",
        hf_overrides={"num_hidden_layers": 2},
    ) as llm:
        _ = llm.generate_greedy(["1 2 3 4 5"], max_tokens=2)


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="FP4 activations with a Hadamard input transform need SM100+.",
)
def test_quip_nvfp4_runs_checkpoint_activation_through_humming(
    vllm_runner, monkeypatch
):
    """A checkpoint declaring FP4 activations keeps them, compiled, end to end.

    The Hadamard must also see the raw activation: a fused producer that hands
    this layer a pre-quantized input would fail in torch.compile.
    """

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    def check_model(model):
        down_proj = model.model.layers[0].mlp.down_proj
        assert isinstance(down_proj.scheme.kernel, HummingHadamardNvFp4LinearKernel)
        assert down_proj.requires_unquantized_input
        assert down_proj.weight_schema.hadamard_block_size == 16
        assert down_proj.scheme.kernel.layer_config.input_quant_mode.should_quantize

    with vllm_runner("nm-testing/Llama-3.2-1B-Instruct-quipv16-nvfp4") as llm:
        llm.apply_model(check_model)
        perplexity = llm.generate_prompt_perplexity(
            ["Flat is better than nested.\nSparse is better than dense."]
        )[0]
    assert perplexity <= 150.0


def _transform_tuple(head_dim=16, scheme_type="hadamard"):
    return Mock(scheme=Mock(head_dim=head_dim, type=scheme_type))


@pytest.mark.parametrize(
    "use_a16,input_tfm,output_tfms,linear_backend,expect_humming",
    [
        (False, _transform_tuple(), {}, "auto", True),
        (False, _transform_tuple(128), {}, "humming", True),
        (True, _transform_tuple(), {}, "auto", False),
        (False, _transform_tuple(scheme_type="random-matrix"), {}, "auto", False),
        (False, _transform_tuple(head_dim=8), {}, "auto", False),
        (False, _transform_tuple(), {0: _transform_tuple()}, "auto", False),
        (False, _transform_tuple(), {}, "flashinfer_cutedsl", False),
    ],
    ids=["auto", "humming", "a16", "dense", "small", "output", "other_backend"],
)
def test_hadamard_fp4_routing(
    monkeypatch, use_a16, input_tfm, output_tfms, linear_backend, expect_humming
):
    """Only the transforms Humming computes natively may skip the checkpoint matrix."""
    monkeypatch.setattr(
        HummingHadamardNvFp4LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )
    monkeypatch.setattr(
        linear_humming_nvfp4,
        "_get_linear_backend",
        lambda quantization: linear_backend,
    )
    scheme = Mock(spec=CompressedTensorsW4A4Fp4, use_a16=use_a16)
    method = CompressedTensorsLinearTransformMethod.from_schemes(
        Mock(), scheme, {0: input_tfm}, output_tfms
    )
    assert isinstance(method, HummingNvFP4HadamardLinearMethod) is expect_humming
