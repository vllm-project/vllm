from typing import List, Optional, Tuple, Type

import librosa
import numpy as np
import pytest
from transformers import AutoModel, AutoTokenizer, BatchEncoding

from vllm.assets.audio import AudioAsset
from vllm.sequence import SampleLogprobs
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

from ..conftest import HfRunner, VllmRunner
from .utils import check_logprobs_close

pytestmark = pytest.mark.vlm

MODEL_NAME = "fixie-ai/ultravox-v0_3"

AudioTuple = Tuple[np.ndarray, int]

AUDIO_ASSETS = [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]


@pytest.fixture(params=[1, 2])
def prompts_and_audios(request):
    audio_count = request.param
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    vllm_placeholder = "<|reserved_special_token_0|>\n" * audio_count
    hf_placeholder = "<|audio|>\n" * audio_count

    question = ("What's in the audio?" if audio_count == 1 else
                "What sport and what nursery rhyme are referenced?")
    vllm_prompt = tokenizer.apply_chat_template(
        [{
            'role': 'user',
            'content': f"{vllm_placeholder}{question}"
        }],
        tokenize=False,
        add_generation_prompt=True)
    hf_prompt = tokenizer.apply_chat_template(
        [{
            'role': 'user',
            'content': f"{hf_placeholder}{question}"
        }],
        tokenize=False,
        add_generation_prompt=True)

    return [
        (vllm_prompt, hf_prompt,
         [asset.audio_and_sample_rate for asset in AUDIO_ASSETS[:audio_count]])
    ]


def vllm_to_hf_output(vllm_output: Tuple[List[int], str,
                                         Optional[SampleLogprobs]],
                      model: str):
    """Sanitize vllm output to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = output_ids[:]
    hf_output_str = output_str
    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    prompts_and_audios: List[Tuple[str, str, List[AudioTuple]]],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    """Inference result should be the same between hf and vllm."""
    torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    with vllm_runner(model,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True,
                     limit_mm_per_prompt={"audio":
                                          len(AUDIO_ASSETS)}) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            [vllm_prompt for vllm_prompt, *_ in prompts_and_audios],
            max_tokens,
            num_logprobs=num_logprobs,
            audios=[audios for *_, audios in prompts_and_audios])

    def process(hf_inputs: BatchEncoding):
        hf_inputs["audio_values"] = hf_inputs["audio_values"] \
            .to(torch_dtype)  # type: ignore
        return hf_inputs

    def resample(audio: AudioTuple) -> AudioTuple:
        return (librosa.resample(audio[0], orig_sr=audio[1],
                                 target_sr=16000), 16000)

    # The HuggingFace model doesn't support multiple audios yet.
    if all(len(audios) <= 1 for *_, audios in prompts_and_audios):
        with hf_runner(model,
                       dtype=dtype,
                       postprocess_inputs=process,
                       auto_cls=AutoModel) as hf_model:

            hf_outputs = hf_model.generate_greedy_logprobs_limit(
                [hf_prompt for _, hf_prompt, *_ in prompts_and_audios],
                max_tokens,
                num_logprobs=num_logprobs,
                audios=[
                    resample(audios[0]) if audios else None
                    for *_, audios in prompts_and_audios
                ])

        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=[
                vllm_to_hf_output(vllm_output, model)
                for vllm_output in vllm_outputs
            ],
            name_0="hf",
            name_1="vllm",
        )
    else:
        # We don't have anything to compare it to, but we can assert that
        # some tokens were generated.
        assert all(tokens for tokens, *_ in vllm_outputs)


@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(hf_runner, vllm_runner, prompts_and_audios, dtype: str,
                max_tokens: int, num_logprobs: int) -> None:
    run_test(
        hf_runner,
        vllm_runner,
        prompts_and_audios,
        MODEL_NAME,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )
