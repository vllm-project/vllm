# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference
with the correct prompt format on audio language models.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""

import os
from dataclasses import asdict
from typing import Any, NamedTuple

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.lora.request import LoRARequest
from vllm.utils.argparse_utils import FlexibleArgumentParser

audio_assets = [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]
question_per_audio_count = {
    0: "What is 1+1?",
    1: "What is recited in the audio?",
    2: "What sport and what nursery rhyme are referenced?",
}


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str | None = None
    prompt_token_ids: dict[str, list[int]] | None = None
    multi_modal_data: dict[str, Any] | None = None
    stop_token_ids: list[int] | None = None
    lora_requests: list[LoRARequest] | None = None


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.


# AudioFlamingo3
def run_audioflamingo3(question: str, audio_count: int) -> ModelRequestData:
    model_name = "nvidia/audio-flamingo-3-hf"
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"audio": audio_count},
        enforce_eager=True,
    )

    # AudioFlamingo3 uses <sound> token for audio
    audio_placeholder = "<sound>" * audio_count

    prompt = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{audio_placeholder}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
    )


# Gemma3N
def run_gemma3n(question: str, audio_count: int) -> ModelRequestData:
    model_name = "google/gemma-3n-E2B-it"
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        max_num_seqs=2,
        limit_mm_per_prompt={"audio": audio_count},
        enforce_eager=True,
    )
    prompt = f"<start_of_turn>user\n<audio_soft_token>{question}"
    "<end_of_turn>\n<start_of_turn>model\n"
    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
    )


# GLM-ASR
def run_glmasr(question: str, audio_count: int) -> ModelRequestData:
    model_name = "zai-org/GLM-ASR-Nano-2512"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # GLM-ASR uses <|pad|> token for audio
    audio_placeholder = "<|pad|>" * audio_count

    messages = [{"role": "user", "content": f"{audio_placeholder}{question}"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"audio": audio_count},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
    )


# Granite Speech
def run_granite_speech(question: str, audio_count: int) -> ModelRequestData:
    # NOTE - the setting in this example are somewhat different from what is
    # optimal for granite speech, and it is generally recommended to use beam
    # search. Check the model README for suggested settings.
    # https://huggingface.co/ibm-granite/granite-speech-3.3-8b
    model_name = "ibm-granite/granite-speech-3.3-8b"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=2048,
        max_num_seqs=2,
        enable_lora=True,
        max_lora_rank=64,
        limit_mm_per_prompt={"audio": audio_count},
    )

    # The model has an audio-specific lora directly in its model dir;
    # it should be enabled whenever you pass audio inputs to the model.
    speech_lora_path = model_name
    audio_placeholder = "<|audio|>" * audio_count
    prompts = f"<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>{audio_placeholder}{question}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"  # noqa: E501

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompts,
        lora_requests=[LoRARequest("speech", 1, speech_lora_path)],
    )


# MiDashengLM
def run_midashenglm(question: str, audio_count: int):
    model_name = "mispeech/midashenglm-7b"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"audio": audio_count},
    )

    audio_in_prompt = "".join(
        ["<|audio_bos|><|AUDIO|><|audio_eos|>" for idx in range(audio_count)]
    )

    default_system = "You are a helpful language and speech assistant."

    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{audio_in_prompt}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
    )


# MiniCPM-O
def run_minicpmo(question: str, audio_count: int) -> ModelRequestData:
    model_name = "openbmb/MiniCPM-o-2_6"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"audio": audio_count},
    )

    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    audio_placeholder = "(<audio>./</audio>)" * audio_count
    audio_chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>' }}{% endif %}"  # noqa: E501
    messages = [{"role": "user", "content": f"{audio_placeholder}\n{question}"}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=audio_chat_template,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
    )


# Phi-4-multimodal-instruct
def run_phi4mm(question: str, audio_count: int) -> ModelRequestData:
    """
    Phi-4-multimodal-instruct supports both image and audio inputs. Here, we
    show how to process audio inputs.
    """
    model_path = snapshot_download("microsoft/Phi-4-multimodal-instruct")
    # Since the vision-lora and speech-lora co-exist with the base model,
    # we have to manually specify the path of the lora weights.
    speech_lora_path = os.path.join(model_path, "speech-lora")
    placeholders = "".join([f"<|audio_{i + 1}|>" for i in range(audio_count)])

    prompts = f"<|user|>{placeholders}{question}<|end|><|assistant|>"

    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=12800,
        max_num_seqs=2,
        enable_lora=True,
        max_lora_rank=320,
        limit_mm_per_prompt={"audio": audio_count},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompts,
        lora_requests=[LoRARequest("speech", 1, speech_lora_path)],
    )


# Qwen2-Audio
def run_qwen2_audio(question: str, audio_count: int) -> ModelRequestData:
    model_name = "Qwen/Qwen2-Audio-7B-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"audio": audio_count},
    )

    audio_in_prompt = "".join(
        [
            f"Audio {idx + 1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
            for idx in range(audio_count)
        ]
    )

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{audio_in_prompt}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
    )


# Qwen2.5-Omni
def run_qwen2_5_omni(question: str, audio_count: int):
    model_name = "Qwen/Qwen2.5-Omni-7B"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"audio": audio_count},
    )

    audio_in_prompt = "".join(
        ["<|audio_bos|><|AUDIO|><|audio_eos|>\n" for idx in range(audio_count)]
    )

    default_system = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
        "Group, capable of perceiving auditory and visual inputs, as well as "
        "generating text and speech."
    )

    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{audio_in_prompt}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
    )


# Ultravox 0.5-1B
def run_ultravox(question: str, audio_count: int) -> ModelRequestData:
    model_name = "fixie-ai/ultravox-v0_5-llama-3_2-1b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [{"role": "user", "content": "<|audio|>\n" * audio_count + question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        trust_remote_code=True,
        limit_mm_per_prompt={"audio": audio_count},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
    )


# Voxtral
# Make sure to install mistral-common[audio].
def run_voxtral(question: str, audio_count: int) -> ModelRequestData:
    from mistral_common.audio import Audio
    from mistral_common.protocol.instruct.chunk import (
        AudioChunk,
        RawAudio,
        TextChunk,
    )
    from mistral_common.protocol.instruct.messages import (
        UserMessage,
    )
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    model_name = "mistralai/Voxtral-Mini-3B-2507"
    tokenizer = MistralTokenizer.from_hf_hub(model_name)

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"audio": audio_count},
        config_format="mistral",
        load_format="mistral",
        tokenizer_mode="mistral",
        enforce_eager=True,
        enable_chunked_prefill=False,
    )

    text_chunk = TextChunk(text=question)
    audios = [
        Audio.from_file(str(audio_assets[i].get_local_path()), strict=False)
        for i in range(audio_count)
    ]
    audio_chunks = [
        AudioChunk(input_audio=RawAudio.from_audio(audio)) for audio in audios
    ]

    messages = [UserMessage(content=[*audio_chunks, text_chunk])]

    req = ChatCompletionRequest(messages=messages, model=model_name)

    tokens = tokenizer.encode_chat_completion(req)
    prompt_ids, audios = tokens.tokens, tokens.audios

    audios_and_sr = [(au.audio_array, au.sampling_rate) for au in audios]

    multi_modal_data = {"audio": audios_and_sr}

    return ModelRequestData(
        engine_args=engine_args,
        prompt_token_ids=prompt_ids,
        multi_modal_data=multi_modal_data,
    )


# Whisper
def run_whisper(question: str, audio_count: int) -> ModelRequestData:
    assert audio_count == 1, "Whisper only support single audio input per prompt"
    model_name = "openai/whisper-large-v3-turbo"

    prompt = "<|startoftranscript|>"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=448,
        max_num_seqs=5,
        limit_mm_per_prompt={"audio": audio_count},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
    )


model_example_map = {
    "audioflamingo3": run_audioflamingo3,
    "gemma3n": run_gemma3n,
    "glmasr": run_glmasr,
    "granite_speech": run_granite_speech,
    "midashenglm": run_midashenglm,
    "minicpmo": run_minicpmo,
    "phi4_mm": run_phi4mm,
    "qwen2_audio": run_qwen2_audio,
    "qwen2_5_omni": run_qwen2_5_omni,
    "ultravox": run_ultravox,
    "voxtral": run_voxtral,
    "whisper": run_whisper,
}


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "audio language models"
    )
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        default="ultravox",
        choices=model_example_map.keys(),
        help='Huggingface "model_type".',
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1, help="Number of prompts to run."
    )
    parser.add_argument(
        "--num-audios",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Number of audio items per prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed when initializing `vllm.LLM`.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=None,
        help="Tensor parallel size to override the model's default setting. ",
    )

    return parser.parse_args()


def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    if args.tensor_parallel_size is not None and args.tensor_parallel_size < 1:
        raise ValueError(
            f"tensor_parallel_size must be a positive integer, "
            f"got {args.tensor_parallel_size}"
        )

    audio_count = args.num_audios
    req_data = model_example_map[model](
        question_per_audio_count[audio_count], audio_count
    )

    # Disable other modalities to save memory
    default_limits = {"image": 0, "video": 0, "audio": 0}
    req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
        req_data.engine_args.limit_mm_per_prompt or {}
    )

    engine_args = asdict(req_data.engine_args) | {"seed": args.seed}
    if args.tensor_parallel_size is not None:
        engine_args["tensor_parallel_size"] = args.tensor_parallel_size
    llm = LLM(**engine_args)

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(
        temperature=0.2, max_tokens=64, stop_token_ids=req_data.stop_token_ids
    )

    def get_input(start, end):
        mm_data = req_data.multi_modal_data
        if not mm_data:
            mm_data = {}
            if end - start > 0:
                mm_data = {
                    "audio": [
                        asset.audio_and_sample_rate for asset in audio_assets[start:end]
                    ]
                }

        inputs = {"multi_modal_data": mm_data}

        if req_data.prompt:
            inputs["prompt"] = req_data.prompt
        else:
            inputs["prompt_token_ids"] = req_data.prompt_token_ids

        return inputs

    # Batch inference
    assert args.num_prompts > 0
    if audio_count != 1:
        inputs = get_input(0, audio_count)
        inputs = [inputs] * args.num_prompts
    else:
        # For single audio input, we need to vary the audio input
        # to avoid deduplication in vLLM engine.
        inputs = []
        for i in range(args.num_prompts):
            start = i % len(audio_assets)
            inp = get_input(start, start + 1)
            inputs.append(inp)

    # Add LoRA request if applicable
    lora_request = (
        req_data.lora_requests * args.num_prompts if req_data.lora_requests else None
    )

    outputs = llm.generate(
        inputs,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    args = parse_args()
    main(args)
