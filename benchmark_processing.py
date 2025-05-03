# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import csv
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Literal, Union, get_args

import numpy as np
import pandas as pd
from tqdm import tqdm

import vllm.envs as envs
from tests.multimodal.utils import random_audio, random_image, random_video
from vllm import AsyncEngineArgs, EngineArgs, SamplingParams
from vllm.config import ModelConfig, ParallelProcessorBackend
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.protocol import EngineClient
from vllm.inputs import InputProcessingContext
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config
from vllm.usage.usage_lib import UsageContext

ModalityStr = Literal["audio", "image", "video"]


def get_supported_modalities(model_id: str) -> list[ModalityStr]:
    model_config = ModelConfig(
        model_id,
        task="auto",
        trust_remote_code=True,
        seed=0,
        dtype="float16",
        revision=None,
    )

    model_cls = MULTIMODAL_REGISTRY._get_model_cls(model_config)
    factories = MULTIMODAL_REGISTRY._processor_factories[model_cls]
    ctx = InputProcessingContext(
        model_config,
        tokenizer=cached_tokenizer_from_config(model_config),
    )

    processing_info = factories.info(ctx)
    supported_mm_limits = processing_info.get_supported_mm_limits()

    return list(supported_mm_limits.keys())


def engine_from_args(engine_args: EngineArgs) -> LLMEngine:
    return LLMEngine.from_engine_args(
        engine_args,
        usage_context=UsageContext.ENGINE_CONTEXT,
    )


def async_engine_from_args(engine_args: AsyncEngineArgs) -> EngineClient:
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.ENGINE_CONTEXT)

    engine_cls = AsyncLLMEngine
    if envs.VLLM_USE_V1:
        from vllm.v1.engine.async_llm import AsyncLLM
        engine_cls = AsyncLLM

    return engine_cls.from_vllm_config(
        vllm_config=vllm_config,
        disable_log_requests=engine_args.disable_log_requests,
        disable_log_stats=engine_args.disable_log_stats,
    )


def get_engine(
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
    modalities: list[ModalityStr],
) -> LLMEngine:
    args = EngineArgs(
        model=model_id,
        trust_remote_code=True,
        parallel_processor_backend=parallel_backend,
        max_model_len=4096,
        limit_mm_per_prompt={
            m: m in modalities
            for m in get_args(ModalityStr)
        },
        enforce_eager=True,
    )

    return engine_from_args(args)


def get_async_engine(
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
    modalities: list[ModalityStr],
) -> EngineClient:
    args = AsyncEngineArgs(
        model=model_id,
        trust_remote_code=True,
        parallel_processor_backend=parallel_backend,
        max_model_len=4096,
        limit_mm_per_prompt={
            m: m in modalities
            for m in get_args(ModalityStr)
        },
        enforce_eager=True,
        disable_log_requests=True,
    )

    return async_engine_from_args(args)


def get_prompt(model_config: ModelConfig, modality: ModalityStr) -> str:
    tokenizer = cached_tokenizer_from_config(model_config)

    return tokenizer.apply_chat_template(
        [
            {
                "role":
                "user",
                "content": [
                    {
                        "type": modality
                    },
                    {
                        "type": "text",
                        "text": f"Describe this {modality}"
                    },
                ]
            },
        ],
        tokenize=False,
    )


def get_benchmark_data(modality: ModalityStr):
    rng = np.random.RandomState(0)

    if modality == "audio":
        return [random_audio(rng, 1024, 4096, 16000) for _ in range(100)]
    if modality == "image":
        return [random_image(rng, 256, 1024) for _ in range(200)]
    if modality == "video":
        return [random_video(rng, 4, 16, 256, 1024) for _ in range(50)]


def get_benchmark_mm_data(
        modalities: list[ModalityStr]) -> dict[ModalityStr, list]:
    return {m: get_benchmark_data(m) for m in modalities}


def benchmark(
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
    engine: LLMEngine,
    modality: ModalityStr,
    data: list,
) -> float:
    model_config = engine.get_model_config()
    prompt = get_prompt(model_config, modality)

    start_s = time.monotonic()
    progbar = tqdm(
        [None] * len(data),
        desc=f"Benchmarking {model_id=}, {parallel_backend=}, {modality=}",
    )

    def _chat(idx: int, item):
        engine.add_request(
            prompt={
                "prompt": prompt,
                "multi_modal_data": {
                    modality: item
                }
            },
            params=SamplingParams(max_tokens=1),
            request_id=str(idx),
        )

        engine.step()
        progbar.update()

    for args in enumerate(data):
        _chat(*args)

    total_s = (time.monotonic() - start_s)

    return total_s / len(data)


async def benchmark_async(
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
    engine: EngineClient,
    modality: ModalityStr,
    data: list,
) -> float:
    model_config = await engine.get_model_config()
    prompt = get_prompt(model_config, modality)

    start_s = time.monotonic()
    progbar = tqdm(
        [None] * len(data),
        desc=f"Benchmarking {model_id=}, {parallel_backend=}, {modality=}",
    )

    async def _chat(idx: int, item):
        generator = engine.generate(
            prompt={
                "prompt": prompt,
                "multi_modal_data": {
                    modality: item
                }
            },
            sampling_params=SamplingParams(max_tokens=1),
            request_id=str(idx),
        )

        async for output in generator:
            progbar.update()

    await asyncio.gather(*(_chat(*args) for args in enumerate(data)))

    total_s = (time.monotonic() - start_s)

    return total_s / len(data)


def benchmark_mm(
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
) -> dict[str, float]:
    modalities = get_supported_modalities(model_id)
    engine = get_engine(model_id, parallel_backend, modalities)
    mm_data = get_benchmark_mm_data(modalities)

    return {
        m: benchmark(model_id, parallel_backend, engine, m, data)
        for m, data in mm_data.items()
    }


async def benchmark_mm_async(
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
) -> dict[str, float]:
    modalities = get_supported_modalities(model_id)
    engine = get_async_engine(model_id, parallel_backend, modalities)
    mm_data = get_benchmark_mm_data(modalities)

    return {
        m: await benchmark_async(model_id, parallel_backend, engine, m, data)
        for m, data in mm_data.items()
    }


def benchmark_one(
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
) -> dict[str, float]:
    return benchmark_mm(
        model_id,
        parallel_backend,
    )


async def benchmark_one_async(
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
) -> dict[str, float]:
    return await benchmark_mm_async(
        model_id,
        parallel_backend,
    )


# Some models OOM or don't have chat template
MODELS = [
    # "rhymes-ai/Aria",
    "CohereForAI/aya-vision-8b",
    # "Salesforce/blip2-opt-2.7b",
    # "facebook/chameleon-7b",
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
    # "TIGER-Lab/Mantis-8B-siglip-llama3",
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


def _resolve_args(
    model_id: Union[str, Literal["all"]],
    parallel_backend: Literal[ParallelProcessorBackend, "all"],
    output_dir: str,
):
    models = MODELS if model_id == "all" else [model_id]
    parallel_backends = (("uni", "mp", "mt") if parallel_backend == "all" else
                         (parallel_backend, ))

    output_path = Path(output_dir)
    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError("Output path must be a directory")
    else:
        output_path.mkdir()

    print("[Models]")
    print("\n".join("- " + model for model in models))

    print("[Parallel Backends]")
    print("\n".join("- " + backend for backend in parallel_backends))

    return models, parallel_backends, output_path


def main(
    model_id: Union[str, Literal["all"]],
    parallel_backend: Literal[ParallelProcessorBackend, "all"],
    output_dir: str,
) -> None:
    models, parallel_backends, output_path = _resolve_args(
        model_id,
        parallel_backend,
        output_dir,
    )

    timestamp = int(datetime.now().timestamp())
    csv_filepath = output_path / f"processing_{timestamp}.csv"

    try:
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
                    try:
                        res_one = benchmark_one(model, backend)
                    except Exception as e:
                        print(f"Failed to benchmark {model=}, {backend=}")
                        traceback.print_exception(e)
                        continue

                    csv_writer.writerow({
                        "model_id": model,
                        "parallel_backend": backend,
                        **{
                            f"ms_{k}": v * 1000
                            for k, v in res_one.items()
                        },
                    })
    finally:
        print(f"Saved results to: {csv_filepath}")

        json_filepath = output_path / f"processing_{timestamp}.json"

        pd.read_csv(csv_filepath).to_json(json_filepath,
                                          orient="records",
                                          indent=4)

        print(f"Saved results to: {json_filepath}")


async def main_async(
    model_id: Union[str, Literal["all"]],
    parallel_backend: Literal[ParallelProcessorBackend, "all"],
    output_dir: str,
) -> None:
    models, parallel_backends, output_path = _resolve_args(
        model_id,
        parallel_backend,
        output_dir,
    )

    timestamp = int(datetime.now().timestamp())
    csv_filepath = output_path / f"processing_{timestamp}.csv"

    try:
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
                    try:
                        res_one = await benchmark_one_async(model, backend)
                    except Exception as e:
                        print(f"Failed to benchmark {model=}, {backend=}")
                        traceback.print_exception(e)
                        continue

                    csv_writer.writerow({
                        "model_id": model,
                        "parallel_backend": backend,
                        **{
                            f"ms_{k}": v * 1000
                            for k, v in res_one.items()
                        },
                    })
    finally:
        print(f"Saved results to: {csv_filepath}")

        json_filepath = output_path / f"processing_{timestamp}.json"

        pd.read_csv(csv_filepath).to_json(json_filepath,
                                          orient="records",
                                          indent=4)

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
    parser.add_argument("--sync",
                        action="store_true",
                        help="Test the sync engine instead of async engine")

    args = parser.parse_args()

    if args.sync:
        main(args.model, args.parallel_backend, args.output_dir)
    else:
        coro = main_async(args.model, args.parallel_backend, args.output_dir)
        asyncio.run(coro)
