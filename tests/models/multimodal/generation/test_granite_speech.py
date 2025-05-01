# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Optional

import pytest
from transformers import AutoModelForSpeechSeq2Seq

from vllm.lora.request import LoRARequest
from vllm.sequence import SampleLogprobs

from ....conftest import HfRunner, PromptAudioInput, VllmRunner, _AudioAssets
from ...registry import HF_EXAMPLE_MODELS
from ...utils import check_logprobs_close

HF_AUDIO_PROMPT = "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant<|end_of_text|>\n<|start_of_role|>user<|end_of_role|><|audio|>can you transcribe the speech into a written format?<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"  # noqa: E501


def vllm_to_hf_output(
    vllm_output: tuple[list[int], str, Optional[SampleLogprobs]],
) -> tuple[list[int], str, Optional[SampleLogprobs]]:
    """Sanitize hf output to be comparable with vllm output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "<|end_of_text|>"

    return output_ids, hf_output_str, out_logprobs


MODEL_NAME = "ibm-granite/granite-speech-3.3-8b"
# Audio lora co-exists directly in the model directory, but
# currently still needs to be passed directly to vLLM.
audio_lora_path = MODEL_NAME
models = [MODEL_NAME]


def run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    inputs: Sequence[tuple[list[str], PromptAudioInput]],
    model: str,
    *,
    max_model_len: int,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    """Inference result should be the same between hf and vllm.

    All the audio fixtures for the test are from AUDIO_ASSETS.
    For huggingface runner, we provide the audio as input.
    For vllm runner, we provide MultiModalDataDict objects
    and corresponding MultiModalConfig as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """
    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).
    # max_model_len should be greater than image_feature_size
    with vllm_runner(
            model,
            task="generate",
            max_model_len=max_model_len,
            max_num_seqs=1,
            dtype=dtype,
            limit_mm_per_prompt={"audio": 1},
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            enable_lora=True,
            max_lora_rank=64,
            enforce_eager=True,
    ) as vllm_model:
        lora_request = LoRARequest("audio", 1, audio_lora_path)
        vllm_outputs_per_case = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                audios=audios,
                                                lora_request=lora_request)
            for prompts, audios in inputs
        ]

    with hf_runner(model, dtype=dtype,
                   auto_cls=AutoModelForSpeechSeq2Seq) as hf_model:

        hf_processor = hf_model.processor
        eos_token_id = hf_processor.tokenizer.eos_token_id

        hf_outputs_per_case = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    audios=[audios],
                                                    eos_token_id=eos_token_id)
            for prompts, audios in inputs
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_case,
                                        vllm_outputs_per_case):
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=[
                vllm_to_hf_output(output) for output in vllm_outputs
            ],
            name_0="hf",
            name_1="vllm",
        )


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_model_len", [2048])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [10])
def test_models(hf_runner, vllm_runner, model: str, audio_assets: _AudioAssets,
                dtype: str, max_model_len: int, max_tokens: int,
                num_logprobs: int) -> None:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    audio, sr = audio_assets[0].audio_and_sample_rate
    # This model expects 16k sample rate, which our test audio
    # already is; if this changes, it may break this test,
    # so we check it directly
    assert sr == 16000
    run_test(
        hf_runner,
        vllm_runner,
        [
            ([HF_AUDIO_PROMPT], [audio]),
        ],
        model,
        dtype=dtype,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )
