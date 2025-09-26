# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import itertools
import logging
import tempfile
from collections.abc import Iterable
from typing import Any, Optional, Union

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm import LLM, SamplingParams
from vllm.attention.backends.registry import _Backend
from vllm.attention.selector import global_force_attn_backend_context_manager
from vllm.config import CompilationConfig, CompilationLevel, CUDAGraphMode, PassConfig
from vllm.platforms import current_platform
from vllm.utils import is_torch_equal_or_newer

from ..utils import create_new_process_for_each_test, flat_product, multi_gpu_test


def models_list(*, all: bool = True, keywords: list[str] | None = None):
    TEST_MODELS: list[tuple[str, dict[str, Any]]] = [
        ("facebook/opt-125m", {}),
        (
            "neuralmagic/Llama-3.2-1B-Instruct-FP8-dynamic",
            {
                "dtype": torch.float16,
            },
        ),
        ("meta-llama/Llama-3.2-1B-Instruct", {}),
    ]

    if all:
        if not current_platform.has_device_capability((10, 0)):
            # int8 removed on Blackwell
            TEST_MODELS.extend(
                [
                    ("neuralmagic/Llama-3.2-1B-Instruct-quantized.w8a8", {}),
                    (
                        "nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change",
                        {
                            "dtype": torch.float16,
                        },
                    ),
                ]
            )

        # TODO: figure out why this fails.
        if False and is_quant_method_supported("gguf"):  # noqa: SIM223
            TEST_MODELS.append(
                ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", {"quantization": "gguf"})
            )

        if is_quant_method_supported("gptq"):
            TEST_MODELS.append(
                ("TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ", {"quantization": "gptq"})
            )

        if is_quant_method_supported("gptq_marlin"):
            TEST_MODELS.append(
                (
                    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ",
                    {"quantization": "gptq_marlin"},
                )
            )

        if is_quant_method_supported("gptq_marlin_24"):
            TEST_MODELS.append(
                (
                    "alexm-nm/tinyllama-24-marlin24-4bit-g128",
                    {"quantization": "gptq_marlin_24"},
                )
            )

        if not current_platform.is_rocm() and is_quant_method_supported("awq"):
            TEST_MODELS.append(
                ("TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ", {"quantization": "AWQ"})
            )

    if keywords is None:
        return TEST_MODELS

    # filter by keywords
    pred = lambda model: any(keyword in model[0] for keyword in keywords)
    return list(filter(pred, TEST_MODELS))


@pytest.mark.parametrize(
    "optimization_level",
    [CompilationLevel.DYNAMO_ONCE, CompilationLevel.PIECEWISE],
)
@pytest.mark.parametrize("model, model_kwargs", models_list(all=True))
@create_new_process_for_each_test()
def test_full_graph(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    model_kwargs: dict[str, Any],
    optimization_level: int,
):
    with monkeypatch.context():
        print(f"MODEL={model}")

        run_model(optimization_level, model, **model_kwargs)


# TODO(luka) add other supported compilation config scenarios here
@pytest.mark.parametrize(
    "compilation_config, model_info",
    [
        # additional compile sizes, only some of the models
        (
            CompilationConfig(level=CompilationLevel.PIECEWISE, compile_sizes=[1, 2]),
            model,
        )
        for model in models_list(all=False)
    ]
    + [
        # RMSNorm + quant fusion, only 8-bit quant models
        (
            CompilationConfig(
                level=CompilationLevel.PIECEWISE,
                custom_ops=["+rms_norm"],
                pass_config=PassConfig(enable_fusion=True, enable_noop=True),
            ),
            model,
        )
        for model in models_list(keywords=["FP8-dynamic", "quantized.w8a8"])
    ]
    + [
        # Test depyf integration works
        (
            CompilationConfig(
                level=CompilationLevel.PIECEWISE, debug_dump_path=tempfile.gettempdir()
            ),
            ("facebook/opt-125m", {}),
        ),
    ]
    + [
        # graph inductor partition
        (
            CompilationConfig(
                level=CompilationLevel.PIECEWISE,
                # inductor graph partition uses
                # torch._C.Tag.cudagraph_unsafe to specify splitting ops
                use_inductor_graph_partition=True,
                cudagraph_mode=CUDAGraphMode.PIECEWISE,
                compile_sizes=[1, 2],
            ),
            model,
        )
        for model in models_list(all=False)
        if is_torch_equal_or_newer("2.9.0.dev")
    ],
)
# only test some of the models
@create_new_process_for_each_test()
def test_custom_compile_config(
    compilation_config: CompilationConfig,
    model_info: tuple[str, dict[str, Any]],
):
    if compilation_config.use_inductor_graph_partition and not is_torch_equal_or_newer(
        "2.9.0.dev"
    ):
        pytest.skip("inductor graph partition is only available in PyTorch 2.9+")

    model, model_kwargs = model_info
    print(f"MODEL={model}")
    run_model(compilation_config, model, **model_kwargs)


MODELS_FP8: list[tuple[str, dict[str, Any], _Backend]] = []
MODELS_FP4: list[tuple[str, dict[str, Any], _Backend]] = []
MODELS: list[tuple[str, dict[str, Any], _Backend]] = []  # tp-only

if current_platform.is_cuda():
    MODELS_FP8 += [
        (
            "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
            {"max_model_len": 1024},
            _Backend.TRITON_ATTN,
        )
    ]

    if current_platform.is_device_capability((10, 0)):
        MODELS_FP8 += [
            (
                "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
                {"kv_cache_dtype": "fp8", "max_model_len": 1024},
                _Backend.FLASHINFER,
            )
        ]

        MODELS_FP4 += [
            (
                "nvidia/Llama-4-Scout-17B-16E-Instruct-FP4",
                {"kv_cache_dtype": "fp8", "max_model_len": 1024},
                _Backend.FLASHINFER,
            )
        ]

        MODELS += [
            (
                "meta-llama/Llama-3.1-8B-Instruct",
                {"max_model_len": 1024},
                _Backend.FLASHINFER,
            )
        ]

elif current_platform.is_rocm():
    MODELS_FP8 += [("amd/Llama-3.1-8B-Instruct-FP8-KV", {}, _Backend.TRITON_ATTN)]


@pytest.mark.parametrize(
    "optimization_level",
    [CompilationLevel.NO_COMPILATION, CompilationLevel.PIECEWISE],
)
def test_fp8_kv_scale_compile(optimization_level: int):
    model = "Qwen/Qwen2-0.5B"
    model_kwargs = {
        "quantization": "fp8",
        "kv_cache_dtype": "fp8_e4m3",
        "calculate_kv_scales": True,
        "max_model_len": 512,
    }
    run_model(optimization_level, model, **model_kwargs)


INDUCTOR_GRAPH_PARTITION = (
    [False, True] if (is_torch_equal_or_newer("2.9.0.dev")) else [False]
)

# TODO(luka) test both in nightly
CUSTOM_OPS_FP8 = ["-quant_fp8"]  # , "+quant_fp8"]


@pytest.mark.parametrize(
    "model_name, model_kwargs, backend, custom_ops",
    # Test attention+quant_fp8 fusion with custom and torch impls of QuantFP8
    list(flat_product(MODELS_FP8, CUSTOM_OPS_FP8))
    # quant_fp4 only has the custom impl
    + list(flat_product(MODELS_FP4, [""])),
)
@pytest.mark.parametrize("inductor_graph_partition", INDUCTOR_GRAPH_PARTITION)
def test_e2e_fusion_attn_quant(
    model_name: str,
    model_kwargs: dict[str, Any],
    backend: _Backend,
    custom_ops: str,
    inductor_graph_partition: bool,
    caplog_vllm,
    monkeypatch,
):
    custom_ops_list = custom_ops.split(",") if custom_ops else []

    if inductor_graph_partition:
        mode = CUDAGraphMode.FULL_AND_PIECEWISE
        splitting_ops: Optional[list[str]] = None
    else:
        mode = CUDAGraphMode.FULL_DECODE_ONLY
        splitting_ops = []

    # Disable, compile cache to make sure custom passes run.
    # Otherwise, we can't verify fusion happened through the logs.
    # Log capture also doesn't work with multiprocessing yet.
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    compilation_config = CompilationConfig(
        # Testing properties
        custom_ops=custom_ops_list,
        use_inductor_graph_partition=inductor_graph_partition,
        cudagraph_mode=mode,
        splitting_ops=splitting_ops,
        # Common
        level=CompilationLevel.PIECEWISE,
        pass_config=PassConfig(enable_attn_fusion=True, enable_noop=True),
        # Inductor caches custom passes by default as well via uuid
        inductor_compile_config={"force_disable_caches": True},
    )

    with (
        caplog_vllm.at_level(logging.DEBUG),
        global_force_attn_backend_context_manager(backend),
    ):
        run_model(compilation_config, model_name, **model_kwargs)

    assert "Fused quant onto 48 attention nodes" in caplog_vllm.text, caplog_vllm.text


# TODO(luka) test both in nightly
CUSTOM_OPS_RMS_NORM = ["-rms_norm"]  # , "+rms_norm"]


def custom_ops_product(*custom_ops_lists: list[str]) -> Iterable[str]:
    for op_list in itertools.product(*custom_ops_lists):
        yield ",".join(op_list)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "model_name, model_kwargs, backend, custom_ops",
    # Toggle RMSNorm and QuantFP8 for FP8 models
    list(
        flat_product(
            MODELS_FP8, custom_ops_product(CUSTOM_OPS_FP8, CUSTOM_OPS_RMS_NORM)
        )
    )
    # Toggle RMSNorm for FP4 models and unquant models
    + list(flat_product(MODELS_FP4 + MODELS, CUSTOM_OPS_RMS_NORM)),
)
@pytest.mark.parametrize("inductor_graph_partition", INDUCTOR_GRAPH_PARTITION)
@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.has_device_capability((10, 0)),
    reason="allreduce+rmsnorm fusion only supported on blackwell",
)
def test_e2e_fusion_tp2_attn_quant_allreduce_rmsnorm(
    model_name,
    model_kwargs,
    backend,
    custom_ops: str,
    inductor_graph_partition: bool,
    caplog_vllm,
    monkeypatch,
):
    custom_ops_list = custom_ops.split(",") if custom_ops else []

    if inductor_graph_partition:
        mode = CUDAGraphMode.FULL_AND_PIECEWISE
        splitting_ops: Optional[list[str]] = None
    else:
        mode = CUDAGraphMode.FULL_DECODE_ONLY
        splitting_ops = []

    # Disable, compile cache to make sure custom passes run.
    # Otherwise, we can't verify fusion happened through the logs.
    # Log capture also doesn't work with multiprocessing yet.
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    compilation_config = CompilationConfig(
        # Testing properties
        use_inductor_graph_partition=inductor_graph_partition,
        cudagraph_mode=mode,
        custom_ops=custom_ops_list,
        splitting_ops=splitting_ops,
        # Common
        level=CompilationLevel.PIECEWISE,
        pass_config=PassConfig(
            enable_attn_fusion=True,
            enable_noop=True,
            enable_fi_allreduce_fusion=True,
        ),
        # Inductor caches custom passes by default as well via uuid
        inductor_compile_config={"force_disable_caches": True},
    )

    with (
        caplog_vllm.at_level(logging.DEBUG),
        global_force_attn_backend_context_manager(backend),
    ):
        run_model(
            compilation_config, model_name, tensor_parallel_size=2, **model_kwargs
        )

    assert "Fused quant onto 48 attention nodes" in caplog_vllm.text, caplog_vllm.text

    # TODO fill in correct number
    assert "Replaced 96 patterns" in caplog_vllm.text, caplog_vllm.text


def run_model(
    compile_config: Union[int, CompilationConfig], model: str, **model_kwargs
):
    compilation_config = (
        compile_config
        if isinstance(compile_config, CompilationConfig)
        else CompilationConfig(level=compile_config)
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0)
    # Allow override from model_kwargs
    model_kwargs = {"tensor_parallel_size": 1, **model_kwargs}
    model_kwargs = {"disable_custom_all_reduce": True, **model_kwargs}

    # No cudagraphs by default
    if compilation_config.cudagraph_mode is None:
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    llm = LLM(
        model=model,
        compilation_config=compilation_config,
        **model_kwargs,
    )
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
