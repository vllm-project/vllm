# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import csv
import time
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
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (apply_hf_chat_template,
                                         parse_chat_messages,
                                         resolve_chat_template_content_format,
                                         resolve_hf_chat_template)
from vllm.inputs import InputProcessingContext
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.utils import (encode_audio_base64, encode_image_base64,
                                   encode_video_base64)
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


_engine = None


def _reset_engine():
    global _engine

    _engine = None

    cleanup_dist_env_and_memory()

    time.sleep(10)


def get_engine(
    model_id: str,
    modalities: list[ModalityStr],
    parallel_backend: ParallelProcessorBackend,
) -> LLMEngine:
    global _engine

    if _engine is None or parallel_backend == "mt":
        args = EngineArgs(
            model=model_id,
            trust_remote_code=True,
            parallel_processor_backend=parallel_backend,
            limit_mm_per_prompt={
                m: m in modalities
                for m in get_args(ModalityStr)
            },
            disable_mm_preprocessor_cache=True,
            enforce_eager=True,
        )

        if _engine is not None:
            _reset_engine()

        engine = engine_from_args(args)
        _engine = engine
    else:
        model_config = _engine.get_model_config()
        model_config.parallel_processor_backend = parallel_backend
        model_config.multimodal_config.parallel_processor_backend = \
            parallel_backend

    return _engine


async def get_async_engine(
    model_id: str,
    modalities: list[ModalityStr],
    parallel_backend: ParallelProcessorBackend,
) -> EngineClient:
    global _engine

    if _engine is None or parallel_backend == "mt":
        args = AsyncEngineArgs(
            model=model_id,
            trust_remote_code=True,
            parallel_processor_backend=parallel_backend,
            limit_mm_per_prompt={
                m: m in modalities
                for m in get_args(ModalityStr)
            },
            disable_mm_preprocessor_cache=True,
            enforce_eager=True,
            disable_log_requests=True,
        )

        if _engine is not None:
            _reset_engine()

        engine = async_engine_from_args(args)
        _engine = engine
    else:
        model_config = await _engine.get_model_config()
        model_config.parallel_processor_backend = parallel_backend
        model_config.multimodal_config.parallel_processor_backend = \
            parallel_backend

    return _engine


def get_prompt(model_config: ModelConfig, modality: ModalityStr) -> str:
    tokenizer = cached_tokenizer_from_config(model_config)

    chat_template = None
    tools = None

    chat_template = resolve_hf_chat_template(
        tokenizer,
        chat_template,
        tools,
        trust_remote_code=True,
    )

    content_format = resolve_chat_template_content_format(
        chat_template,
        tools,
        "auto",
        tokenizer,
        trust_remote_code=True,
    )

    rng = np.random.RandomState(0)
    if modality == "audio":
        audio_base64 = encode_audio_base64(
            *random_audio(rng, 1024, 4096, 16000))
        mm_content = {
            "type": "audio_url",
            "audio_url": {
                "url": f"data:audio/wav;base64,{audio_base64}"
            }
        }
    elif modality == "image":
        image_base64 = encode_image_base64(random_image(rng, 256, 1024))
        mm_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        }
    elif modality == "video":
        video_base64 = encode_video_base64(random_video(rng, 4, 16, 256, 1024))
        mm_content = {
            "type": "video_url",
            "video_url": {
                "url": f"data:video/jpeg;base64,{video_base64}"
            }
        }

    conversation, mm_data = parse_chat_messages(
        [
            {
                "role":
                "user",
                "content": [
                    mm_content,
                    {
                        "type": "text",
                        "text": f"Describe this {modality}"
                    },
                ]
            },
        ],
        model_config,
        tokenizer,
        content_format,
    )

    for message in conversation:
        contents = message.get("content")
        if isinstance(contents, list):
            for content in contents:
                if content["type"] in get_args(ModalityStr):
                    content[content["type"]] = None

    return apply_hf_chat_template(
        tokenizer,
        conversation,
        chat_template,
        tools,
        trust_remote_code=True,
    )


def get_benchmark_data(modality: ModalityStr):
    rng = np.random.RandomState(0)

    if modality == "audio":
        return [random_audio(rng, 1024, 4096, 16000) for _ in range(200)]
    if modality == "image":
        return [random_image(rng, 256, 1024) for _ in range(200)]
    if modality == "video":
        return [random_video(rng, 4, 16, 256, 1024) for _ in range(50)]


def get_benchmark_mm_data(
        modalities: list[ModalityStr]) -> dict[ModalityStr, list]:
    return {m: get_benchmark_data(m) for m in modalities}


def benchmark_run(
    engine: LLMEngine,
    model_config: ModelConfig,
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
    modality: ModalityStr,
    data: list,
):
    prompt = get_prompt(model_config, modality)

    start_s = time.monotonic()
    progbar = tqdm(
        [None] * len(data),
        desc=f"Benchmarking {model_id=}, {parallel_backend=}, "
        f"{modality=}",
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


async def benchmark_run_async(
    engine: EngineClient,
    model_config: ModelConfig,
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
    modality: ModalityStr,
    data: list,
):
    prompt = get_prompt(model_config, modality)

    start_s = time.monotonic()
    progbar = tqdm(
        [None] * len(data),
        desc=f"Benchmarking {model_id=}, {parallel_backend=}, "
        f"{modality=}",
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


def benchmark(
    model_id: str,
    parallel_backends: list[ParallelProcessorBackend],
) -> dict[ParallelProcessorBackend, dict[ModalityStr, float]]:
    supported_modalities = get_supported_modalities(model_id)
    mm_data = get_benchmark_mm_data(supported_modalities)

    all_results = dict[ParallelProcessorBackend, dict[ModalityStr, float]]()
    for parallel_backend in parallel_backends:
        engine = get_engine(model_id, supported_modalities, parallel_backend)
        model_config = engine.get_model_config()

        parallel_results = dict[ModalityStr, float]()
        for modality, data in mm_data.items():
            parallel_results[modality] = benchmark_run(
                engine,
                model_config,
                model_id,
                parallel_backend,
                modality,
                data,
            )

        all_results[parallel_backend] = parallel_results
        del engine

    return all_results


async def benchmark_async(
    model_id: str,
    parallel_backends: list[ParallelProcessorBackend],
) -> dict[ParallelProcessorBackend, dict[ModalityStr, float]]:
    supported_modalities = get_supported_modalities(model_id)
    mm_data = get_benchmark_mm_data(supported_modalities)

    all_results = dict[ParallelProcessorBackend, dict[ModalityStr, float]]()
    for parallel_backend in parallel_backends:
        engine = await get_async_engine(model_id, supported_modalities,
                                        parallel_backend)
        model_config = await engine.get_model_config()

        parallel_results = dict[ModalityStr, float]()
        for modality, data in mm_data.items():
            parallel_results[modality] = await benchmark_run_async(
                engine,
                model_config,
                model_id,
                parallel_backend,
                modality,
                data,
            )

        all_results[parallel_backend] = parallel_results
        del engine

    return all_results


def resolve_output_path(output_dir: str):
    output_path = Path(output_dir)
    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError("Output path must be a directory")
    else:
        output_path.mkdir()

    return output_path


def save_results(
    all_results: dict[ParallelProcessorBackend, dict[ModalityStr, float]],
    model_id: str,
    output_path: Path,
    append: bool,
):
    timestamp = int(datetime.now().timestamp())

    if append:
        csv_filepaths = sorted(output_path.glob("processing_*.csv"))
        if csv_filepaths:
            prev_filepath = csv_filepaths[-1]
            with prev_filepath.open("r") as f:
                csv_reader = csv.DictReader(f)
                prev_rows = list(csv_reader)
        else:
            prev_rows = []
    else:
        prev_rows = []

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

        for row in prev_rows:
            csv_writer.writerow(row)

        for backend, parallel_results in all_results.items():
            csv_writer.writerow({
                "model_id": model_id,
                "parallel_backend": backend,
                **{
                    f"ms_{modality}": avg_s * 1000
                    for modality, avg_s in parallel_results.items()
                },
            })

    print(f"Saved results to: {csv_filepath}")

    if append:
        json_filepaths = sorted(output_path.glob("processing_*.json"))
        if json_filepaths:
            json_filepath = json_filepaths[-1]
        else:
            json_filepath = output_path / f"processing_{timestamp}.json"
    else:
        json_filepath = output_path / f"processing_{timestamp}.json"

    pd.read_csv(csv_filepath).to_json(json_filepath,
                                      orient="records",
                                      indent=4)

    print(f"Saved results to: {json_filepath}")


def main(
    model_id: Union[str, Literal["all"]],
    parallel_backends: list[ParallelProcessorBackend],
    output_dir: str,
    append: bool,
) -> None:
    output_path = resolve_output_path(output_dir)

    all_results = benchmark(model_id, parallel_backends)

    save_results(all_results, model_id, output_path, append=append)


async def main_async(
    model_id: Union[str, Literal["all"]],
    parallel_backends: list[ParallelProcessorBackend],
    output_dir: str,
    append: bool,
) -> None:
    output_path = resolve_output_path(output_dir)

    all_results = await benchmark_async(model_id, parallel_backends)

    save_results(all_results, model_id, output_path, append=append)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark parallel multi-modal processing")
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        default="HuggingFaceTB/SmolVLM-256M-Instruct",
                        help="Name of the model")
    parser.add_argument("-p",
                        "--parallel-backends",
                        type=str,
                        nargs="+",
                        choices=["uni", "mp", "mt"],
                        default=["uni", "mp", "mt"],
                        help="Parallel backends to test")
    parser.add_argument("-o",
                        "--output-dir",
                        type=str,
                        required=True,
                        help="Directory to save the results")
    parser.add_argument("--sync",
                        action="store_true",
                        help="Test the sync engine instead of async engine")
    parser.add_argument("--append",
                        action="store_true",
                        help="Append to results file if it exists")

    args = parser.parse_args()

    if args.sync:
        main(args.model,
             args.parallel_backends,
             args.output_dir,
             append=args.append)
    else:
        coro = main_async(args.model,
                          args.parallel_backends,
                          args.output_dir,
                          append=args.append)
        asyncio.run(coro)
