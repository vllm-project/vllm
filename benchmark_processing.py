# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import vllm.envs as envs
from tests.models.registry import HF_EXAMPLE_MODELS
from tests.multimodal.utils import random_audio, random_image, random_video
from vllm import AsyncEngineArgs, SamplingParams
from vllm.config import ModelConfig, ParallelProcessorBackend
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.inputs import InputProcessingContext
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config
from vllm.usage.usage_lib import UsageContext


def get_engine(
    engine_args: AsyncEngineArgs,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
):
    # Create the engine configs.
    vllm_config = engine_args.create_engine_config(usage_context)

    engine_cls = AsyncLLMEngine
    if envs.VLLM_USE_V1:
        from vllm.v1.engine.async_llm import AsyncLLM
        engine_cls = AsyncLLM

    return engine_cls.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=usage_context,
        disable_log_stats=engine_args.disable_log_stats,
    )


async def benchmark_audio_processing(
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
    *,
    num_audios: int = 200,
) -> float:
    rng = np.random.RandomState(0)
    audios = [random_audio(rng, 1024, 4096, 16000) for _ in range(num_audios)]

    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    args = AsyncEngineArgs(
        model=model_id,
        tokenizer=model_info.tokenizer or model_id,
        tokenizer_mode=model_info.tokenizer_mode,
        trust_remote_code=model_info.trust_remote_code,
        parallel_processor_backend=parallel_backend,
        max_model_len=4096,
        limit_mm_per_prompt={
            "audio": 1,
            "image": 0,
            "video": 0
        },
        enforce_eager=True,
        disable_log_requests=True,
    )
    engine = get_engine(args)

    model_config = await engine.get_model_config()
    tokenizer = cached_tokenizer_from_config(model_config)
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "audio"
                    },
                    {
                        "type": "text",
                        "text": "Describe this audio"
                    },
                ]
            },
        ],
        tokenize=False,
    )

    start_s = time.monotonic()
    progbar = tqdm(
        [None] * len(audios),
        desc=f"Audio processing for {model_id=}, {parallel_backend=}",
    )

    async def _chat(idx: int):
        generator = engine.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "audio": [audios[idx]]
                }
            },
            sampling_params=SamplingParams(max_tokens=1),
            request_id=str(idx),
        )

        async for output in generator:
            progbar.update()

    await asyncio.gather(*(_chat(idx) for idx in range(len(audios))))

    total_s = (time.monotonic() - start_s)

    return total_s / num_audios


async def benchmark_image_processing(
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
    *,
    num_images: int = 400,
) -> float:
    rng = np.random.RandomState(0)
    images = [random_image(rng, 256, 1024) for _ in range(num_images)]

    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    args = AsyncEngineArgs(
        model=model_id,
        tokenizer=model_info.tokenizer or model_id,
        tokenizer_mode=model_info.tokenizer_mode,
        trust_remote_code=model_info.trust_remote_code,
        parallel_processor_backend=parallel_backend,
        max_model_len=4096,
        limit_mm_per_prompt={
            "audio": 0,
            "image": 1,
            "video": 0
        },
        enforce_eager=True,
        disable_log_requests=True,
    )
    engine = get_engine(args)

    model_config = await engine.get_model_config()
    tokenizer = cached_tokenizer_from_config(model_config)
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "image"
                    },
                    {
                        "type": "text",
                        "text": "Describe this image"
                    },
                ]
            },
        ],
        tokenize=False,
    )

    start_s = time.monotonic()
    progbar = tqdm(
        [None] * len(images),
        desc=f"Image processing for {model_id=}, {parallel_backend=}",
    )

    async def _chat(idx: int):
        generator = engine.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": [images[idx]]
                }
            },
            sampling_params=SamplingParams(max_tokens=1),
            request_id=str(idx),
        )

        async for output in generator:
            progbar.update()

    await asyncio.gather(*(_chat(idx) for idx in range(len(images))))

    total_s = (time.monotonic() - start_s)

    return total_s / num_images


async def benchmark_video_processing(
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
    *,
    num_videos: int = 100,
) -> float:
    rng = np.random.RandomState(0)
    videos = [random_video(rng, 4, 16, 256, 1024) for _ in range(num_videos)]

    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    args = AsyncEngineArgs(
        model=model_id,
        tokenizer=model_info.tokenizer or model_id,
        tokenizer_mode=model_info.tokenizer_mode,
        trust_remote_code=model_info.trust_remote_code,
        parallel_processor_backend=parallel_backend,
        max_model_len=4096,
        limit_mm_per_prompt={
            "audio": 0,
            "image": 0,
            "video": 1
        },
        enforce_eager=True,
        disable_log_requests=True,
    )
    engine = get_engine(args)

    model_config = await engine.get_model_config()
    tokenizer = cached_tokenizer_from_config(model_config)
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "video"
                    },
                    {
                        "type": "text",
                        "text": "Describe this video"
                    },
                ]
            },
        ],
        tokenize=False,
    )

    start_s = time.monotonic()
    progbar = tqdm(
        [None] * len(videos),
        desc=f"Video processing for {model_id=}, {parallel_backend=}",
    )

    async def _chat(idx: int):
        generator = engine.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "video": [videos[idx]]
                }
            },
            sampling_params=SamplingParams(max_tokens=1),
            request_id=str(idx),
        )

        async for output in generator:
            progbar.update()

    await asyncio.gather(*(_chat(idx) for idx in range(len(videos))))

    total_s = (time.monotonic() - start_s)

    return total_s / num_videos


BENCHMARK_FNS = {
    "audio": benchmark_audio_processing,
    "image": benchmark_image_processing,
    "video": benchmark_video_processing,
}


async def benchmark_processing(
    modality: str,
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
) -> float:
    return await BENCHMARK_FNS[modality](model_id, parallel_backend)


async def benchmark_one(
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
) -> dict[str, float]:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_config = ModelConfig(
        model_id,
        task="auto",
        tokenizer=model_info.tokenizer or model_id,
        tokenizer_mode=model_info.tokenizer_mode,
        trust_remote_code=model_info.trust_remote_code,
        seed=0,
        dtype="float16",
        revision=None,
        hf_overrides=model_info.hf_overrides,
    )

    model_cls = MULTIMODAL_REGISTRY._get_model_cls(model_config)
    factories = MULTIMODAL_REGISTRY._processor_factories[model_cls]
    ctx = InputProcessingContext(
        model_config,
        tokenizer=cached_tokenizer_from_config(model_config),
    )

    processing_info = factories.info(ctx)
    supported_mm_limits = processing_info.get_supported_mm_limits()

    return {
        modality: await benchmark_processing(modality, model_id,
                                             parallel_backend)
        for modality in supported_mm_limits
    }


# Some models OOM or don't have chat template
MODELS = [
    # "rhymes-ai/Aria",
    "CohereForAI/aya-vision-8b",
    # "Salesforce/blip2-opt-2.7b",
    "facebook/chameleon-7b",
    # "deepseek-ai/deepseek-vl2-tiny",
    # "microsoft/Florence-2-base",
    "adept/fuyu-8b",
    "google/gemma-3-4b-it",
    "THUDM/glm-4v-9b",
    "ibm-granite/granite-speech-3.3-8b",
    "h2oai/h2ovl-mississippi-800m",
    "OpenGVLab/InternVL2-1B",
    "HuggingFaceM4/Idefics3-8B-Llama3",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    # "moonshotai/Kimi-VL-A3B-Instruct",
    # "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "TIGER-Lab/Mantis-8B-siglip-llama3",
    # "openbmb/MiniCPM-Llama3-V-2_5",
    # "openbmb/MiniCPM-o-2_6",
    # "openbmb/MiniCPM-V-2_6",
    # "MiniMaxAI/MiniMax-VL-01",
    "allenai/Molmo-7B-D-0924",
    "allenai/Molmo-7B-O-0924",
    # "nvidia/NVLM-D-72B",
    "AIDC-AI/Ovis2-1B",
    # "google/paligemma-3b-mix-224",
    # "google/paligemma2-3b-ft-docci-448",
    "microsoft/Phi-3.5-vision-instruct",
    "microsoft/Phi-4-multimodal-instruct",
    # "mistralai/Pixtral-12B-2409",
    "mistral-community/pixtral-12b",
    "Qwen/Qwen-VL-Chat",
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2-Audio-7B-Instruct",
    # "Qwen/Qwen2.5-Omni-7B",
    # "Skywork/Skywork-R1V-38B",
    "fixie-ai/ultravox-v0_5-llama-3_2-1b",
    "openai/whisper-large-v3",
]


async def main(
    model_id: Union[str, Literal["all"]],
    parallel_backend: Literal[ParallelProcessorBackend, "all"],
    output_dir: str,
) -> None:
    models = MODELS if model_id == "all" else [model_id]
    parallel_backends = (("uni", "mp", "mt") if parallel_backend == "all" else
                         (parallel_backend, ))

    output_path = Path(output_dir)
    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError("Output path must be a directory")
    else:
        output_path.mkdir()

    timestamp = int(datetime.now().timestamp())
    csv_filepath = output_path / f"processing_{timestamp}.csv"

    with csv_filepath.open("w") as f:
        csv_writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_id",
                "parallel_backend",
                "ms_audio",
                "ms_image",
                "ms_video",
            ],
        )
        csv_writer.writeheader()

        for model in models:
            for backend in parallel_backends:
                res_one = await benchmark_one(model_id, backend)

                csv_writer.writerow({
                    "model_id": model,
                    "parallel_backend": backend,
                    **{
                        f"ms_{k}": v * 1000
                        for k, v in res_one.items()
                    },
                })

    print(f"Saved results to: {csv_filepath}")

    json_filepath = output_path / f"processing_{timestamp}.json"

    pd.read_csv(csv_filepath).to_json(json_filepath, indent=4)

    print(f"Saved results to: {json_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark parallel multi-modal processing")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="Name of the model, or 'all' to select all MODELS")
    parser.add_argument("-p",
                        "--parallel-backend",
                        type=str,
                        choices=["uni", "mp", "mt", "all"],
                        default="all",
                        help="Parallel backend to use, or 'all' to select all")
    parser.add_argument("-o",
                        "--output-dir",
                        type=str,
                        required=True,
                        help="Directory to save the results")

    args = parser.parse_args()
    coro = main(args.model, args.parallel_backend, args.output_dir)

    asyncio.run(coro)
