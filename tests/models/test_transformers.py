# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test the functionality of the Transformers modeling backend."""

from typing import Any

import pytest

from vllm.platforms import current_platform

from ..conftest import HfRunner, VllmRunner
from ..utils import multi_gpu_test, prep_prompts
from .registry import HF_EXAMPLE_MODELS
from .utils import check_embeddings_close, check_logprobs_close


def get_model(arch: str) -> str:
    model_info = HF_EXAMPLE_MODELS.get_hf_info(arch)
    model_info.check_transformers_version(on_fail="skip")
    return model_info.default


def check_implementation(
    runner_ref: type[HfRunner | VllmRunner],
    runner_test: type[VllmRunner],
    example_prompts: list[str],
    model: str,
    kwargs_ref: dict[str, Any] | None = None,
    kwargs_test: dict[str, Any] | None = None,
    **kwargs,
):
    if kwargs_ref is None:
        kwargs_ref = {}
    if kwargs_test is None:
        kwargs_test = {}

    max_tokens = 32
    num_logprobs = 5

    args = (example_prompts, max_tokens, num_logprobs)

    with runner_test(model, **kwargs_test, **kwargs) as model_test:
        model_config = model_test.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()

        outputs_test = model_test.generate_greedy_logprobs(*args)

    with runner_ref(model, **kwargs_ref) as model_ref:
        if isinstance(model_ref, VllmRunner):
            outputs_ref = model_ref.generate_greedy_logprobs(*args)
        else:
            outputs_ref = model_ref.generate_greedy_logprobs_limit(*args)

    check_logprobs_close(
        outputs_0_lst=outputs_ref,
        outputs_1_lst=outputs_test,
        name_0="ref",
        name_1="test",
    )


@pytest.mark.parametrize(
    "model,model_impl",
    [
        ("meta-llama/Llama-3.2-1B-Instruct", "transformers"),
        ("hmellor/Ilama-3.2-1B", "auto"),  # CUSTOM CODE
        ("allenai/OLMoE-1B-7B-0924", "transformers"),  # MoE
    ],
)  # trust_remote_code=True by default
def test_models(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    example_prompts: list[str],
    model: str,
    model_impl: str,
) -> None:
    import transformers
    from packaging.version import Version

    installed = Version(transformers.__version__)
    required = Version("5.0.0.dev")
    if model == "allenai/OLMoE-1B-7B-0924" and installed < required:
        pytest.skip(
            "MoE models with the Transformers modeling backend require "
            f"transformers>={required}, but got {installed}"
        )

    check_implementation(
        hf_runner, vllm_runner, example_prompts, model, model_impl=model_impl
    )


def test_hybrid_attention(vllm_runner: type[VllmRunner]) -> None:
    prompts, _, _ = prep_prompts(4, (800, 801))
    kwargs_ref = {"max_model_len": 8192, "enforce_eager": True}
    kwargs_test = {"model_impl": "transformers", **kwargs_ref}
    check_implementation(
        vllm_runner,
        vllm_runner,
        prompts,
        model="hmellor/tiny-random-Gemma2ForCausalLM",
        kwargs_ref=kwargs_ref,
        kwargs_test=kwargs_test,
    )


@multi_gpu_test(num_gpus=2)
def test_distributed(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    example_prompts,
):
    kwargs = {"model_impl": "transformers", "tensor_parallel_size": 2}
    check_implementation(
        hf_runner,
        vllm_runner,
        example_prompts,
        "meta-llama/Llama-3.2-1B-Instruct",
        kwargs_test=kwargs,
    )


@pytest.mark.parametrize(
    "model, quantization_kwargs",
    [
        ("TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ", {}),
        ("TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ", {}),
        (
            "meta-llama/Llama-3.2-1B-Instruct",
            {
                "quantization": "bitsandbytes",
            },
        ),
    ],
)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_quantization(
    vllm_runner: type[VllmRunner],
    example_prompts: list[str],
    model: str,
    quantization_kwargs: dict[str, str],
    max_tokens: int,
    num_logprobs: int,
) -> None:
    if (
        current_platform.is_rocm()
        and quantization_kwargs.get("quantization", "") == "bitsandbytes"
    ):
        pytest.skip("bitsandbytes quantization is currently not supported in rocm.")

    with vllm_runner(
        model,
        model_impl="auto",
        enforce_eager=True,
        **quantization_kwargs,  # type: ignore[arg-type]
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens=max_tokens, num_logprobs=num_logprobs
        )

    with vllm_runner(
        model,
        model_impl="transformers",
        enforce_eager=True,
        **quantization_kwargs,  # type: ignore[arg-type]
    ) as vllm_model:
        model_config = vllm_model.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()

        transformers_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens=max_tokens, num_logprobs=num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=transformers_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="transformers",
        name_1="vllm",
    )


@pytest.mark.parametrize(
    "model",
    [
        # Layers live in `layers`
        "Qwen/Qwen3-Embedding-0.6B",
        # Layers live in `model.layers`
        "meta-llama/Llama-3.2-1B-Instruct",
    ],
)
def test_embed_loading(vllm_runner, model):
    with vllm_runner(
        model,
        max_model_len=1024,
        enforce_eager=True,
        runner="pooling",
        model_impl="transformers",
    ) as model_test:
        model_config = model_test.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()


@pytest.mark.parametrize(
    "arch", ["TransformersEmbeddingModel", "TransformersForSequenceClassification"]
)
def test_pooling(hf_runner, vllm_runner, example_prompts, arch):
    model = get_model(arch)

    vllm_kwargs = dict(max_model_len=None, model_impl="transformers")

    hf_kwargs = dict()
    if arch == "TransformersEmbeddingModel":
        hf_kwargs["is_sentence_transformer"] = True
    elif arch == "TransformersForSequenceClassification":
        from transformers import AutoModelForSequenceClassification

        hf_kwargs["auto_cls"] = AutoModelForSequenceClassification

    # The example_prompts has ending "\n", for example:
    # "Write a short story about a robot that dreams for the first time.\n"
    # sentence_transformers will strip the input texts, see:
    # https://github.com/UKPLab/sentence-transformers/blob/v3.1.1/sentence_transformers/models/Transformer.py#L159
    # This makes the input_ids different between hf_model and vllm_model.
    # So we need to strip the input texts to avoid test failing.
    example_prompts = [str(s).strip() for s in example_prompts]

    with (
        vllm_runner(model, **vllm_kwargs) as vllm_model,
        hf_runner(model, **hf_kwargs) as hf_model,
    ):
        model_config = vllm_model.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()

        if arch == "TransformersEmbeddingModel":
            vllm_outputs = vllm_model.embed(example_prompts)
            hf_outputs = hf_model.encode(example_prompts)
        elif arch == "TransformersForSequenceClassification":
            vllm_outputs = vllm_model.classify(example_prompts)
            hf_outputs = hf_model.classify(example_prompts)

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
