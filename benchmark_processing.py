# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from tests.models.registry import HF_EXAMPLE_MODELS
from tests.multimodal.utils import random_audio, random_image, random_video
from vllm.config import ModelConfig, ParallelProcessorBackend
from vllm.entrypoints.chat_utils import (apply_hf_chat_template,
                                         parse_chat_messages,
                                         resolve_chat_template_content_format,
                                         resolve_hf_chat_template)
from vllm.inputs import InputProcessingContext
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.multimodal.utils import (encode_audio_base64, encode_image_base64,
                                   encode_video_base64)
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

ROOT_DIR = Path(__file__).parent
EXAMPLES_DIR = ROOT_DIR / "examples"
assert EXAMPLES_DIR.exists()

ModalityStr = Literal["audio", "image", "video"]


def get_hf_tokenizer(model_id: str):
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    except Exception:
        return model_id
    else:
        return model_info.tokenizer or model_id


def get_hf_overrides(model_id: str):
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    except Exception:
        return {}
    else:
        return model_info.hf_overrides


def get_model_config(model_id: str) -> ModelConfig:
    return ModelConfig(
        model_id,
        tokenizer=get_hf_tokenizer(model_id),
        hf_overrides=get_hf_overrides(model_id),
        trust_remote_code=True,
        seed=0,
    )


def get_supported_modalities(model_config: ModelConfig) -> list[ModalityStr]:
    model_cls = MULTIMODAL_REGISTRY._get_model_cls(model_config)
    factories = MULTIMODAL_REGISTRY._processor_factories[model_cls]
    ctx = InputProcessingContext(
        model_config,
        tokenizer=cached_tokenizer_from_config(model_config),
    )

    processing_info = factories.info(ctx)
    supported_mm_limits = processing_info.get_supported_mm_limits()

    return list(supported_mm_limits.keys())


def get_processor(
    model_config: ModelConfig,
    modalities: list[ModalityStr],
    parallel_backend: ParallelProcessorBackend,
) -> BaseMultiModalProcessor:
    parallel_processor_backend = parallel_backend
    limit_mm_per_prompt = {m: m in modalities for m in get_args(ModalityStr)}

    model_config.parallel_processor_backend = parallel_processor_backend
    model_config.limit_mm_per_prompt = limit_mm_per_prompt

    mm_config = model_config.get_multimodal_config()
    mm_config.parallel_processor_backend = parallel_backend
    mm_config.limit_per_prompt = limit_mm_per_prompt

    return MULTIMODAL_REGISTRY.create_processor(model_config,
                                                disable_cache=True)


def get_prompt(model_config: ModelConfig, modality: ModalityStr) -> str:
    tokenizer = cached_tokenizer_from_config(model_config)

    chat_template = None
    tools = None

    chat_template = resolve_hf_chat_template(
        tokenizer,
        chat_template,
        tools,
        model_config=model_config,
    )

    content_format = resolve_chat_template_content_format(
        chat_template,
        tools,
        "auto",
        tokenizer,
        model_config=model_config,
    )

    rng = np.random.RandomState(0)
    if modality == "audio":
        audio_base64 = encode_audio_base64(
            *random_audio(rng, 8192, 32768, 16000))
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

    conversation, _ = parse_chat_messages(
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
        model_config=model_config,
    )


def get_benchmark_data(modality: ModalityStr):
    rng = np.random.RandomState(0)

    if modality == "audio":
        return [random_audio(rng, 32768, 131072, 16000) for _ in range(200)]
    if modality == "image":
        return [random_image(rng, 256, 1024) for _ in range(200)]
    if modality == "video":
        return [random_video(rng, 4, 16, 256, 1024) for _ in range(50)]


def get_benchmark_mm_data(
        modalities: list[ModalityStr]) -> dict[ModalityStr, list]:
    return {m: get_benchmark_data(m) for m in modalities}


def benchmark_run(
    processor: BaseMultiModalProcessor,
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
    modality: ModalityStr,
    data: list,
):
    prompt = get_prompt(processor.info.ctx.model_config, modality)

    start_s = time.monotonic()
    progbar = tqdm(
        [None] * len(data),
        desc=f"Benchmarking {model_id=}, {parallel_backend=}, "
        f"{modality=}",
    )

    def _chat(item):
        processor.apply(
            prompt=prompt,
            mm_data={modality: item},
            hf_processor_mm_kwargs={},
        )

        progbar.update()

    for item in data:
        _chat(item)

    total_s = (time.monotonic() - start_s)
    return total_s / len(data)


async def benchmark_run_async(
    processor: BaseMultiModalProcessor,
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
    modality: ModalityStr,
    data: list,
):
    prompt = get_prompt(processor.info.ctx.model_config, modality)

    start_s = time.monotonic()
    progbar = tqdm(
        [None] * len(data),
        desc=f"Benchmarking {model_id=}, {parallel_backend=}, "
        f"{modality=}",
    )

    async def _chat(item):
        await processor.apply_async(
            prompt=prompt,
            mm_data={modality: item},
            hf_processor_mm_kwargs={},
        )

        progbar.update()

    await asyncio.gather(*(_chat(item) for item in data))

    total_s = (time.monotonic() - start_s)
    return total_s / len(data)


def benchmark(
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
) -> dict[ParallelProcessorBackend, dict[ModalityStr, float]]:
    model_config = get_model_config(model_id)

    supported_modalities = get_supported_modalities(model_config)
    mm_data = get_benchmark_mm_data(supported_modalities)

    processor = get_processor(model_config, supported_modalities,
                              parallel_backend)

    results = dict[ModalityStr, float]()
    for modality, data in mm_data.items():
        try:
            results[modality] = benchmark_run(
                processor,
                model_id,
                parallel_backend,
                modality,
                data,
            )
        except Exception as e:
            print(f"Failed to benchmark {model_id=}, {parallel_backend=}, "
                  f"{modality=}:\n{e}")

    return {parallel_backend: results}


async def benchmark_async(
    model_id: str,
    parallel_backend: ParallelProcessorBackend,
) -> dict[ParallelProcessorBackend, dict[ModalityStr, float]]:
    model_config = get_model_config(model_id)

    supported_modalities = get_supported_modalities(model_config)
    mm_data = get_benchmark_mm_data(supported_modalities)

    processor = get_processor(model_config, supported_modalities,
                              parallel_backend)

    results = dict[ModalityStr, float]()
    for modality, data in mm_data.items():
        try:
            results[modality] = await benchmark_run_async(
                processor,
                model_id,
                parallel_backend,
                modality,
                data,
            )
        except Exception as e:
            print(f"Failed to benchmark {model_id=}, {parallel_backend=}, "
                  f"{modality=}:\n{e}")

    return {parallel_backend: results}


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
            csv_filepath = csv_filepaths[-1]
            with csv_filepath.open("r") as f:
                csv_reader = csv.DictReader(f)
                prev_rows = list(csv_reader)
        else:
            prev_rows = []
            csv_filepath = output_path / f"processing_{timestamp}.csv"
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
    parallel_backend: ParallelProcessorBackend,
    output_dir: str,
    append: bool,
) -> None:
    output_path = resolve_output_path(output_dir)

    all_results = benchmark(model_id, parallel_backend)

    save_results(all_results, model_id, output_path, append=append)


async def main_async(
    model_id: Union[str, Literal["all"]],
    parallel_backend: ParallelProcessorBackend,
    output_dir: str,
    append: bool,
) -> None:
    output_path = resolve_output_path(output_dir)

    all_results = await benchmark_async(model_id, parallel_backend)

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
                        "--parallel-backend",
                        type=str,
                        choices=["uni", "mp", "mt"],
                        default="uni",
                        help="Parallel backend to test")
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
             args.parallel_backend,
             args.output_dir,
             append=args.append)
    else:
        coro = main_async(args.model,
                          args.parallel_backend,
                          args.output_dir,
                          append=args.append)
        asyncio.run(coro)
