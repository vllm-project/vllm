# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/basic_correctness/test_basic_correctness.py`.
"""

import os
import weakref
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from packaging.version import Version
from transformers import __version__ as TRANSFORMERS_VERSION

import vllm.envs as envs
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
from vllm.v1.engine.llm_engine import LLMEngine

from ..conftest import VllmRunner
from ..utils import multi_gpu_test

ATTN_BACKEND = ["ROCM_ATTN"] if current_platform.is_rocm() else ["FLASH_ATTN"]

MODELS = [
    "hmellor/tiny-random-Gemma2ForCausalLM",
    "meta-llama/Llama-3.2-1B-Instruct",
]

TARGET_TEST_SUITE_ENV = "VLLM_TARGET_TEST_SUITE"
LEGACY_TARGET_TEST_SUITE_ENV = "TARGET_TEST_SUITE"

GENERIC_DISTRIBUTED_TEST_SUITES = ("L4", "MI250", "MI300", "MI325", "MI355")
ALL_DISTRIBUTED_TEST_SUITES = (*GENERIC_DISTRIBUTED_TEST_SUITES, "A100")


def _assert_input_logprobs_close(
    reference_logprobs: list[torch.Tensor],
    reference_prompt_token_ids: list[list[int]],
    vllm_outputs: list[RequestOutput],
) -> None:
    assert len(vllm_outputs) == len(reference_prompt_token_ids)
    for request_idx, (reference_ids, reference_scores, output) in enumerate(
        zip(reference_prompt_token_ids, reference_logprobs, vllm_outputs)
    ):
        assert output.prompt_logprobs is not None
        assert len(output.prompt_logprobs) == len(reference_ids)

        target_ids = reference_ids[1:]
        vllm_scores = []
        for prompt_position, (token_id, token_logprobs) in enumerate(
            zip(target_ids, output.prompt_logprobs[1:]), start=1
        ):
            assert token_logprobs is not None, (
                f"Missing vLLM input logprob for request {request_idx}, "
                f"prompt position {prompt_position}, target token {token_id}"
            )
            assert len(token_logprobs) == 1
            assert token_id in token_logprobs
            vllm_scores.append(token_logprobs[token_id].logprob)

        target_ids_tensor = torch.tensor(target_ids, device=reference_scores.device)
        expected_scores = reference_scores[
            torch.arange(len(target_ids), device=reference_scores.device),
            target_ids_tensor,
        ]
        torch.testing.assert_close(
            torch.tensor(vllm_scores, device=reference_scores.device),
            expected_scores,
            atol=2e-2,
            rtol=2e-2,
        )


def _default_target_test_suite() -> str:
    if not current_platform.is_rocm():
        return "L4"

    try:
        device_name = current_platform.get_device_name().upper()
    except Exception:
        device_name = ""

    if "MI355" in device_name:
        return "MI355"
    if "MI300" in device_name:
        return "MI300"
    if "MI325" in device_name:
        return "MI325"
    if "MI250" in device_name:
        return "MI250"

    try:
        from vllm.platforms import rocm as rocm_platform

        if rocm_platform.on_gfx950():
            return "MI355"
        if rocm_platform.on_gfx942():
            return "MI300"
    except Exception:
        pass

    return "MI250"


def _resolve_target_test_suite() -> str:
    for env_name in (TARGET_TEST_SUITE_ENV, LEGACY_TARGET_TEST_SUITE_ENV):
        value = os.environ.get(env_name, "").strip().upper()
        if value:
            return value
    return _default_target_test_suite()


TARGET_TEST_SUITE = _resolve_target_test_suite()


# ROCm can occasionally retain the object until fixture teardown. Retry only
# that assertion after cleanup; collecting cyclic garbage here would mask the
# reference cycles this test is intended to catch.
@pytest.mark.flaky(
    reruns=2,
    reruns_delay=5,
    only_rerun="AssertionError",
    condition=current_platform.is_rocm(),
)
def test_vllm_gc_ed():
    """Verify vllm instance is GC'ed when it is deleted"""
    llm = LLM("hmellor/tiny-random-LlamaForCausalLM")
    weak_llm = weakref.ref(llm)
    del llm
    # If there's any circular reference to vllm, this fails
    # because llm instance is not GC'ed.
    assert weak_llm() is None


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", ATTN_BACKEND)
@pytest.mark.parametrize("enforce_eager", [False])
@pytest.mark.parametrize("async_scheduling", [True, False])
@pytest.mark.parametrize("model_executor", ["uni", "mp"])
@pytest.mark.parametrize("enable_prompt_embeds", [True, False])
def test_models(
    hf_runner,
    model: str,
    backend: str,
    enforce_eager: bool,
    async_scheduling: bool,
    model_executor: str,
    enable_prompt_embeds: bool,
) -> None:
    # 5042 tokens for gemma2
    # gemma2 has alternating sliding window size of 4096
    # we need a prompt with more than 4096 tokens to test the sliding window
    prompt = (
        "The following numbers of the sequence "
        + ", ".join(str(i) for i in range(1024))
        + " are:"
    )
    example_prompts = [prompt]

    with hf_runner(model) as hf_model:
        hf_inputs = hf_model.get_inputs(example_prompts)
        reference_prompt_token_ids = [
            inputs["input_ids"][0].tolist() for inputs in hf_inputs
        ]
        vllm_prompts: list[list[int]] | list[dict[str, Any]]
        if enable_prompt_embeds:
            with torch.no_grad():
                prompt_embeds = hf_model.get_prompt_embeddings_from_inputs(hf_inputs)
            if model == "hmellor/tiny-random-Gemma2ForCausalLM" and (
                Version(TRANSFORMERS_VERSION) < Version("5.3.0.dev0")
            ):
                # For Gemma 1/2 models with Transformers 5.4.0+, the prompt embeddings
                # are normalised in `get_prompt_embeddings`, like Gemma 3.
                # For older versions, we need to manually normalise.
                embed_scale = hf_model.config.hidden_size**0.5
                normalizer = torch.tensor(embed_scale, dtype=prompt_embeds[0].dtype)
                prompt_embeds = [p_e * normalizer for p_e in prompt_embeds]
            reference_logprobs = hf_model.get_prompt_logprobs(
                prompt_embeds=prompt_embeds
            )
            vllm_prompts = [
                {
                    "prompt_embeds": prompt_embed,
                    "prompt_token_ids": prompt_token_ids,
                }
                for prompt_embed, prompt_token_ids in zip(
                    prompt_embeds, reference_prompt_token_ids
                )
            ]
        else:
            reference_logprobs = hf_model.get_prompt_logprobs(hf_inputs)
            vllm_prompts = reference_prompt_token_ids

    with VllmRunner(
        model,
        max_model_len=8192,
        enforce_eager=enforce_eager,
        enable_prompt_embeds=enable_prompt_embeds,
        gpu_memory_utilization=0.7,
        async_scheduling=async_scheduling,
        distributed_executor_backend=model_executor,
        attention_config={"backend": backend},
    ) as vllm_model:
        vllm_outputs = vllm_model.llm.generate(
            vllm_prompts,
            sampling_params=SamplingParams(
                temperature=0.0, max_tokens=1, prompt_logprobs=0
            ),
        )

    _assert_input_logprobs_close(
        reference_logprobs, reference_prompt_token_ids, vllm_outputs
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    (
        "model, distributed_executor_backend, attention_backend, "
        "target_test_suites, extra_env"
    ),
    [
        ("facebook/opt-125m", "ray", "", ALL_DISTRIBUTED_TEST_SUITES, {}),
        ("facebook/opt-125m", "mp", "", ALL_DISTRIBUTED_TEST_SUITES, {}),
        (
            "meta-llama/Llama-3.2-1B-Instruct",
            "ray",
            "",
            GENERIC_DISTRIBUTED_TEST_SUITES,
            {},
        ),
        (
            "meta-llama/Llama-3.2-1B-Instruct",
            "mp",
            "",
            GENERIC_DISTRIBUTED_TEST_SUITES,
            {},
        ),
    ],
)
@pytest.mark.parametrize("enable_prompt_embeds", [True, False])
def test_models_distributed(
    monkeypatch: pytest.MonkeyPatch,
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    distributed_executor_backend: str,
    attention_backend: str,
    target_test_suites: tuple[str, ...],
    extra_env: dict[str, str],
    enable_prompt_embeds: bool,
) -> None:
    if TARGET_TEST_SUITE and TARGET_TEST_SUITE not in target_test_suites:
        pytest.skip(f"Skip test for {TARGET_TEST_SUITE}")

    with monkeypatch.context() as monkeypatch_context:
        if (
            model == "meta-llama/Llama-3.2-1B-Instruct"
            and distributed_executor_backend == "ray"
            and attention_backend == ""
            and TARGET_TEST_SUITE == "L4"
            and enable_prompt_embeds
        ):  # noqa
            pytest.skip("enable_prompt_embeds does not work with ray compiled dag.")

        for k, v in extra_env.items():
            monkeypatch_context.setenv(k, v)

        dtype = "half"

        # NOTE: take care of the order. run vLLM first, and then run HF.
        # vLLM needs a fresh new process without cuda initialization.
        # if we run HF first, the cuda initialization will be done and it
        # will hurt multiprocessing backend with fork method
        # (the default method).
        attention_config = {"backend": attention_backend} if attention_backend else None
        with vllm_runner(
            model,
            dtype=dtype,
            tensor_parallel_size=2,
            distributed_executor_backend=distributed_executor_backend,
            enable_prompt_embeds=enable_prompt_embeds,
            gpu_memory_utilization=0.7,
            attention_config=attention_config,
        ) as vllm_model:
            if enable_prompt_embeds:
                with hf_runner(model, dtype=dtype) as hf_model:
                    hf_inputs = hf_model.get_inputs(example_prompts)
                    reference_prompt_token_ids = [
                        inputs["input_ids"][0].tolist() for inputs in hf_inputs
                    ]
                    with torch.no_grad():
                        prompt_embeds = hf_model.get_prompt_embeddings_from_inputs(
                            hf_inputs
                        )
                    reference_logprobs = hf_model.get_prompt_logprobs(
                        prompt_embeds=prompt_embeds
                    )
                    vllm_prompts = [
                        {
                            "prompt_embeds": prompt_embed,
                            "prompt_token_ids": prompt_token_ids,
                        }
                        for prompt_embed, prompt_token_ids in zip(
                            prompt_embeds, reference_prompt_token_ids
                        )
                    ]
                    vllm_outputs = vllm_model.llm.generate(
                        vllm_prompts,
                        sampling_params=SamplingParams(
                            temperature=0.0, max_tokens=1, prompt_logprobs=0
                        ),
                    )
            else:
                vllm_outputs = vllm_model.llm.generate(
                    example_prompts,
                    sampling_params=SamplingParams(
                        temperature=0.0, max_tokens=1, prompt_logprobs=0
                    ),
                )
                with hf_runner(model, dtype=dtype) as hf_model:
                    reference_prompt_token_ids = [
                        output.prompt_token_ids for output in vllm_outputs
                    ]
                    hf_inputs = [
                        {
                            "input_ids": torch.tensor(
                                token_ids, dtype=torch.long
                            ).unsqueeze(0)
                        }
                        for token_ids in reference_prompt_token_ids
                    ]
                    reference_logprobs = hf_model.get_prompt_logprobs(hf_inputs)

    _assert_input_logprobs_close(
        reference_logprobs, reference_prompt_token_ids, vllm_outputs
    )


def test_failed_model_execution(vllm_runner, monkeypatch) -> None:
    # Needed to mock an error in the same process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    with vllm_runner("facebook/opt-125m", enforce_eager=True) as vllm_model:
        if isinstance(vllm_model.llm.llm_engine, LLMEngine):
            v1_test_failed_model_execution(vllm_model)


@pytest.mark.parametrize("use_v2_model_runner", [False, True], ids=["v1", "v2"])
def test_raise_on_logit_nans(
    vllm_runner, monkeypatch, use_v2_model_runner: bool
) -> None:
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_RAISE_ON_LOGIT_NANS", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2_model_runner)))
    envs.disable_envs_cache()

    try:
        with vllm_runner("facebook/opt-125m", enforce_eager=True) as vllm_model:
            engine_core = vllm_model.llm.llm_engine.engine_core.engine_core
            model_runner = engine_core.model_executor.driver_worker.worker.model_runner
            assert model_runner.vllm_config.use_v2_model_runner is use_v2_model_runner
            model_cls = type(model_runner.model)
            original_compute_logits = model_cls.compute_logits

            def compute_nan_logits(self, *args, **kwargs):
                logits = original_compute_logits(self, *args, **kwargs)
                # `logits[0, 0] = ...` copies the scalar H2D and blocks.
                logits[0, 0].fill_(float("nan"))
                return logits

            monkeypatch.setattr(model_cls, "compute_logits", compute_nan_logits)

            with pytest.raises(RuntimeError, match="NaNs detected in logits"):
                vllm_model.generate_greedy(["Hello, my name is"], 1, use_tqdm=False)
    finally:
        envs.disable_envs_cache()


def v1_test_failed_model_execution(vllm_model):
    engine = vllm_model.llm.llm_engine
    mocked_execute_model = Mock(side_effect=RuntimeError("Mocked Critical Error"))
    engine.engine_core.engine_core.model_executor.execute_model = mocked_execute_model

    with pytest.raises(RuntimeError) as exc_info:
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        vllm_model.generate_greedy(prompts, 200, use_tqdm=False)
    assert isinstance(exc_info.value, RuntimeError)
    assert "Mocked Critical Error" in str(exc_info.value)
