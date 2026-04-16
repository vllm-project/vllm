# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# flake8: noqa
"""Tests Model Optimizer nvfp4 models against ground truth generation
Note: these tests will only pass on B200
"""

import os
from typing import List

import pytest
import torch
from vllm.triton_utils import triton
from transformers import AutoTokenizer

from tests.quantization.utils import is_quant_method_supported
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.quantization.utils import (
    nvfp4_emulation_utils,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    ref_nvfp4_quant_dequant,
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
    "flashinfer-cudnn",
    "flashinfer-trtllm",
    "flashinfer-cutlass",
]


@pytest.mark.parametrize("model", ["nvidia/Llama-3.1-8B-Instruct-NVFP4"])
@pytest.mark.parametrize("eager", EAGER)
@pytest.mark.parametrize(
    "backend",
    [
        "emulation",
        "flashinfer-cudnn",
        "flashinfer-trtllm",  # the small seq_len ensures trtllm_8x4_layout backend is used
        "flashinfer-cutlass",
    ],
)
def test_nvfp4(vllm_runner, model, eager, backend, monkeypatch):
    if (
        not current_platform.has_device_capability(100)
        and backend in SM_100_NVFP4_BACKENDS
    ):
        pytest.skip(
            f"The backend {backend} is not supported with current_platform.has_device_capability(100) == False"
        )

    monkeypatch.setenv("VLLM_NVFP4_GEMM_BACKEND", backend)
    with vllm_runner(model, enforce_eager=eager) as llm:
        output = llm.generate_greedy(["1 2 3 4 5"], max_tokens=2)
    assert output[0][1] == "1 2 3 4 5 6"


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton NVFP4 kernel requires CUDA.",
)
def test_triton_dequantize_nvfp4(monkeypatch) -> None:
    """Test the Triton dequantization kernel against the CPU reference
    using real NVFP4 weights from a checkpoint.

    Tests both 2D (attention projection) and 3D (stacked MoE experts).
    """
    import huggingface_hub
    from safetensors import safe_open

    checkpoint_path = huggingface_hub.snapshot_download(
        "nvidia/Qwen3-30B-A3B-NVFP4",
        allow_patterns=["model-00001-of-00004.safetensors"],
    )
    shard_path = f"{checkpoint_path}/model-00001-of-00004.safetensors"
    block_size = 16

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())

        # 2D case: attention projection
        tensor_fp4_2d = f.get_tensor("model.layers.9.self_attn.k_proj.weight")
        tensor_sf_2d = f.get_tensor("model.layers.9.self_attn.k_proj.weight_scale")
        global_scale_2d = f.get_tensor("model.layers.9.self_attn.k_proj.weight_scale_2")

        # 3D case: stack ALL experts for layer 9 up_proj
        expert_prefix = "model.layers.9.mlp.experts."
        expert_indices = sorted(
            int(key.split(".")[5])
            for key in all_keys
            if key.startswith(expert_prefix) and key.endswith(".up_proj.weight")
        )
        assert len(expert_indices) > 0

        all_fp4 = []
        all_sf = []
        all_global_scale = []
        for index in expert_indices:
            name = f"{expert_prefix}{index}.up_proj"
            all_fp4.append(f.get_tensor(f"{name}.weight"))
            all_sf.append(f.get_tensor(f"{name}.weight_scale"))
            all_global_scale.append(f.get_tensor(f"{name}.weight_scale_2"))

    tensor_fp4_3d = torch.stack(all_fp4)
    tensor_sf_3d = torch.stack(all_sf)
    global_scale_3d = torch.stack(all_global_scale)

    test_cases = [
        ("2D base", tensor_fp4_2d, tensor_sf_2d, global_scale_2d),
        (
            "2D 2x rows",
            tensor_fp4_2d.repeat(2, 1),
            tensor_sf_2d.repeat(2, 1),
            global_scale_2d,
        ),
        (
            "2D 4x rows",
            tensor_fp4_2d.repeat(4, 1),
            tensor_sf_2d.repeat(4, 1),
            global_scale_2d,
        ),
        (
            "2D 2x cols",
            tensor_fp4_2d.repeat(1, 2),
            tensor_sf_2d.repeat(1, 2),
            global_scale_2d,
        ),
        ("3D base", tensor_fp4_3d, tensor_sf_3d, global_scale_3d),
        (
            "3D 2x experts",
            tensor_fp4_3d.repeat(2, 1, 1),
            tensor_sf_3d.repeat(2, 1, 1),
            global_scale_3d.repeat(2),
        ),
        (
            "3D 2x rows",
            tensor_fp4_3d.repeat(1, 2, 1),
            tensor_sf_3d.repeat(1, 2, 1),
            global_scale_3d,
        ),
        (
            "3D 2x cols",
            tensor_fp4_3d.repeat(1, 1, 2),
            tensor_sf_3d.repeat(1, 1, 2),
            global_scale_3d,
        ),
    ]

    quantiles = [0.5, 0.001, 0.999]

    for label, tensor_fp4, tensor_sf, global_scale in test_cases:
        fp4_cuda = tensor_fp4.cuda()
        sf_cuda = tensor_sf.cuda()
        gs_cuda = global_scale.cuda()

        # Triton path
        triton_result = dequantize_to_dtype(
            fp4_cuda,
            sf_cuda,
            gs_cuda,
            torch.bfloat16,
            block_size,
            swizzle=False,
        )

        # Reference path
        with monkeypatch.context() as m:
            m.setattr(
                nvfp4_emulation_utils.current_platform,
                "is_cuda_alike",
                lambda: False,
            )
            reference = dequantize_to_dtype(
                tensor_fp4,
                tensor_sf,
                global_scale,
                torch.bfloat16,
                block_size,
                swizzle=False,
            )

        torch.testing.assert_close(triton_result.cpu(), reference, atol=0, rtol=0)

        # Benchmark
        shape = list(tensor_fp4.shape)

        triton_ms, triton_min, triton_max = triton.testing.do_bench(
            lambda: dequantize_to_dtype(
                fp4_cuda,
                sf_cuda,
                gs_cuda,
                torch.bfloat16,
                block_size,
                swizzle=False,
            ),
            quantiles=quantiles,
        )

        original_lut = nvfp4_emulation_utils.kE2M1ToFloat_handle.val
        nvfp4_emulation_utils.kE2M1ToFloat_handle.val = original_lut.cuda()

        def _reference_bench():
            with monkeypatch.context() as m2:
                m2.setattr(
                    nvfp4_emulation_utils.current_platform,
                    "is_cuda_alike",
                    lambda: False,
                )
                dequantize_to_dtype(
                    fp4_cuda,
                    sf_cuda,
                    gs_cuda,
                    torch.bfloat16,
                    block_size,
                    swizzle=False,
                )

        ref_ms, ref_min, ref_max = triton.testing.do_bench(
            _reference_bench,
            quantiles=quantiles,
        )
        nvfp4_emulation_utils.kE2M1ToFloat_handle.val = original_lut

        speedup = ref_ms / triton_ms if triton_ms > 0 else float("inf")
        print(f"  dequantize {label} {shape}:")
        print(
            f"    triton:    median={triton_ms:.3f}ms, "
            f"min={triton_min:.3f}ms, max={triton_max:.3f}ms"
        )
        print(
            f"    reference: median={ref_ms:.3f}ms, "
            f"min={ref_min:.3f}ms, max={ref_max:.3f}ms"
        )
        print(f"    speedup:   {speedup:.2f}x")


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton NVFP4 kernel requires CUDA.",
)
@pytest.mark.parametrize(
    "m, k",
    [
        (1, 16),
        (1, 32),
        (2, 48),
        (7, 64),
        (16, 128),
        (33, 160),
        (128, 256),
        (256, 512),
        (1024, 1024),
        (5120, 2048),
        (2048, 4096),
        (4096, 7168),
        (8192, 8192),
    ],
)
@pytest.mark.parametrize("global_scale_value", [0.5, 1.0, 0.001])
def test_triton_nvfp4_quant_dequant(
    monkeypatch, m: int, k: int, global_scale_value: float
) -> None:
    """Test the Triton quant-dequant kernel against the CPU reference."""
    block_size = 16
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    global_scale = torch.tensor(global_scale_value, dtype=torch.float32, device="cuda")

    # Triton path
    triton_result, _ = ref_nvfp4_quant_dequant(x, global_scale, block_size)

    # CPU reference path
    with monkeypatch.context() as mp:
        mp.setattr(
            nvfp4_emulation_utils.current_platform,
            "is_cuda_alike",
            lambda: False,
        )
        reference, _ = ref_nvfp4_quant_dequant(x.cpu(), global_scale.cpu(), block_size)

    torch.testing.assert_close(triton_result.cpu(), reference, atol=0, rtol=0)

    # Benchmark (both paths on CUDA tensors for fair comparison)
    quantiles = [0.5, 0.001, 0.999]

    triton_ms, triton_min, triton_max = triton.testing.do_bench(
        lambda: ref_nvfp4_quant_dequant(x, global_scale, block_size),
        quantiles=quantiles,
    )

    def _reference_bench():
        with monkeypatch.context() as mp2:
            mp2.setattr(
                nvfp4_emulation_utils.current_platform,
                "is_cuda_alike",
                lambda: False,
            )
            ref_nvfp4_quant_dequant(x, global_scale, block_size)

    ref_ms, ref_min, ref_max = triton.testing.do_bench(
        _reference_bench,
        quantiles=quantiles,
    )

    speedup = ref_ms / triton_ms if triton_ms > 0 else float("inf")
    print(f"  quant_dequant [{m}x{k}] gs={global_scale_value}:")
    print(
        f"    triton:    median={triton_ms:.3f}ms, "
        f"min={triton_min:.3f}ms, max={triton_max:.3f}ms"
    )
    print(
        f"    reference: median={ref_ms:.3f}ms, "
        f"min={ref_min:.3f}ms, max={ref_max:.3f}ms"
    )
    print(f"    speedup:   {speedup:.2f}x")
