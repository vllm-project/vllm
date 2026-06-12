# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Evaluate Transcription API correctness by computing Word Error Rate (WER)
on a given ASR dataset. When provided, it will also compare the WER against
a baseline.
This simulates real work usage of the API and makes sure that the frontend and
AsyncLLMEngine are working correctly.
"""

import asyncio
import io
import time
from statistics import mean, median

import pytest
import soundfile
import torch
from datasets import Audio, load_dataset
from evaluate import load
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from vllm.benchmarks.datasets.datasets import ASRDataset
from vllm.multimodal.audio import get_audio_duration
from vllm.tokenizers import get_tokenizer

from ....models.registry import HF_EXAMPLE_MODELS
from ....utils import RemoteOpenAIServer

# Tuned to prevent OOM on 18GB GPUs in transcription correctness tests.
MAX_SEQS_FOR_TRANSCRIPTION_TEST = 8
GPU_UTIL_FOR_TRANSCRIPTION_TEST = 0.5


def to_bytes(y, sr):
    buffer = io.BytesIO()
    soundfile.write(buffer, y, sr, format="WAV")
    buffer.seek(0)
    return buffer


def load_audio_sample(audio):
    # Avoid torchcodec in CI by decoding dataset audio with soundfile.
    if "array" in audio and "sampling_rate" in audio:
        return audio["array"], audio["sampling_rate"]

    if audio.get("path"):
        return soundfile.read(audio["path"], dtype="float32")

    if audio.get("bytes") is not None:
        return soundfile.read(io.BytesIO(audio["bytes"]), dtype="float32")

    raise ValueError("Audio sample did not contain array, path, or bytes data")


# not all models have a normalizer so use the one from whisper as a standard option
normalizer_model_info = HF_EXAMPLE_MODELS.find_hf_info("openai/whisper-large-v3")
normalizer_tokenizer = get_tokenizer(
    "openai/whisper-large-v3",
    tokenizer_mode=normalizer_model_info.tokenizer_mode,
    trust_remote_code=normalizer_model_info.trust_remote_code,
)
normalizer = EnglishTextNormalizer(normalizer_tokenizer.english_spelling_normalizer)


async def transcribe_audio(client, tokenizer, y, sr, extra_body=None):
    # Send loaded audio directly instead of loading from disk,
    # don't account for that time though
    with to_bytes(y, sr) as f:
        start_time = time.perf_counter()
        transcription = await client.audio.transcriptions.create(
            file=f,
            model=tokenizer.name_or_path,
            language="en",
            temperature=0.0,
            extra_body=extra_body,
        )
        end_time = time.perf_counter()
        # NOTE there's no streaming in transcriptions, can't measure ttft
    latency = end_time - start_time
    num_output_tokens = len(
        tokenizer(transcription.text, add_special_tokens=False).input_ids
    )
    return latency, num_output_tokens, transcription.text


async def bound_transcribe(
    sem, client, tokenizer, audio, sr, reference, extra_body=None
):
    # Use semaphore to limit concurrent requests.
    async with sem:
        result = await transcribe_audio(
            client, tokenizer, audio, sr, extra_body=extra_body
        )
        # Normalize *english* output/reference for evaluation.
        out = normalizer(result[2])
        ref = normalizer(reference)
        return result[:2] + (out, ref)


async def process_dataset(model, client, data, concurrent_request, extra_body=None):
    sem = asyncio.Semaphore(concurrent_request)

    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    tokenizer = get_tokenizer(
        model,
        tokenizer_mode=model_info.tokenizer_mode,
        trust_remote_code=model_info.trust_remote_code,
    )

    # Warmup call as the first `load_audio` server-side is quite slow.
    audio, sr = load_audio_sample(data[0]["audio"])
    _ = await bound_transcribe(sem, client, tokenizer, audio, sr, "", extra_body)

    tasks: list[asyncio.Task] = []
    for sample in data:
        audio, sr = load_audio_sample(sample["audio"])
        task = asyncio.create_task(
            bound_transcribe(
                sem, client, tokenizer, audio, sr, sample["text"], extra_body
            )
        )
        tasks.append(task)
    return await asyncio.gather(*tasks)


def print_performance_metrics(results, total_time):
    latencies = [res[0] for res in results]
    total_tokens = sum([res[1] for res in results])

    total = len(results)
    print(f"Total Requests: {total}")
    print(f"Successful Requests: {len(latencies)}")
    print(f"Average Latency: {mean(latencies):.4f} seconds")
    print(f"Median Latency: {median(latencies):.4f} seconds")
    perc = sorted(latencies)[int(len(latencies) * 0.95) - 1]
    print(f"95th Percentile Latency: {perc:.4f} seconds")
    # Throughput
    req_throughput = len(latencies) / total_time
    print(f"Estimated req_Throughput: {req_throughput:.2f} requests/s")
    throughput = total_tokens / total_time
    print(f"Estimated Throughput: {throughput:.2f} tok/s")


def add_duration(sample):
    y, sr = load_audio_sample(sample["audio"])
    sample["duration_ms"] = get_audio_duration(y=y, sr=sr) * 1000
    return sample


def load_asr_dataset_rows(dataset_repo: str, split="validation", **hf_kwargs):
    if dataset_repo in ASRDataset.SUPPORTED_DATASET_PATHS:
        asr_dataset_kwargs = {
            "dataset_path": dataset_repo,
            "dataset_split": split,
            "disable_shuffle": True,
            "no_stream": True,
        }
        for key in ("dataset_subset", "hf_name", "trust_remote_code"):
            if key in hf_kwargs:
                asr_dataset_kwargs[key] = hf_kwargs[key]
        return ASRDataset(**asr_dataset_kwargs).data

    return load_dataset(dataset_repo, split=split, **hf_kwargs)


def load_shortform_eval_dataset(dataset_repo: str, split="validation", **hf_kwargs):
    ## Load and filter the dataset.
    dataset = load_asr_dataset_rows(dataset_repo, split=split, **hf_kwargs)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    if "duration_ms" not in dataset.column_names:
        # Compute duration to filter.
        dataset = dataset.map(add_duration)

    # Whisper max supported duration.
    dataset = dataset.filter(lambda example: example["duration_ms"] < 30000)
    return dataset


def run_evaluation(
    model: str,
    client,
    dataset,
    max_concurrent_reqs: int,
    n_examples: int = -1,
    print_metrics: bool = True,
    extra_body=None,
):
    if n_examples > 0:
        dataset = dataset.select(range(n_examples))
    start = time.perf_counter()
    results = asyncio.run(
        process_dataset(
            model, client, dataset, max_concurrent_reqs, extra_body=extra_body
        )
    )
    end = time.perf_counter()
    total_time = end - start
    print(f"Total Test Time: {total_time:.4f} seconds")
    if print_metrics:
        print_performance_metrics(results, total_time)
    # Compute WER
    predictions = [res[2] for res in results]
    references = [res[3] for res in results]
    wer = load("wer")
    wer_score = 100 * wer.compute(references=references, predictions=predictions)
    print("WER:", wer_score)
    return wer_score


LONGFORM_DATASET_REPO = ASRDataset.EARNINGS22_CLEANED_DATASET
LONGFORM_DATASET_SPLIT = "test"
LONGFORM_NUM_SAMPLES = 6


def load_longform_dataset():
    dataset = load_asr_dataset_rows(
        LONGFORM_DATASET_REPO,
        split=LONGFORM_DATASET_SPLIT,
    )
    assert len(dataset) >= LONGFORM_NUM_SAMPLES
    return dataset.select(range(LONGFORM_NUM_SAMPLES))


async def transcribe_audio_path(client, tokenizer, audio_path: str, extra_body=None):
    with open(audio_path, "rb") as f:
        start_time = time.perf_counter()
        transcription = await client.audio.transcriptions.create(
            file=f,
            model=tokenizer.name_or_path,
            language="en",
            temperature=0.0,
            extra_body=extra_body,
        )
        end_time = time.perf_counter()

    latency = end_time - start_time
    num_output_tokens = len(
        tokenizer(transcription.text, add_special_tokens=False).input_ids
    )
    return latency, num_output_tokens, transcription.text


async def bound_transcribe_path(
    sem, client, tokenizer, audio_path, reference, extra_body=None
):
    async with sem:
        result = await transcribe_audio_path(
            client, tokenizer, audio_path, extra_body=extra_body
        )
        out = normalizer(result[2])
        ref = normalizer(reference)
        return result[:2] + (out, ref)


async def process_longform_dataset(
    model, client, data, concurrent_request, extra_body=None
):
    sem = asyncio.Semaphore(concurrent_request)

    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    tokenizer = get_tokenizer(
        model,
        tokenizer_mode=model_info.tokenizer_mode,
        trust_remote_code=model_info.trust_remote_code,
    )

    warmup_path = data[0]["audio"]["path"]
    _ = await bound_transcribe_path(sem, client, tokenizer, warmup_path, "", extra_body)

    tasks: list[asyncio.Task] = []
    for sample in data:
        audio_path = sample["audio"]["path"]
        task = asyncio.create_task(
            bound_transcribe_path(
                sem, client, tokenizer, audio_path, sample["text"], extra_body
            )
        )
        tasks.append(task)
    return await asyncio.gather(*tasks)


def run_longform_evaluation(
    model: str,
    client,
    dataset,
    max_concurrent_reqs: int,
    print_metrics: bool = True,
    extra_body=None,
):
    start = time.perf_counter()
    results = asyncio.run(
        process_longform_dataset(
            model, client, dataset, max_concurrent_reqs, extra_body=extra_body
        )
    )
    end = time.perf_counter()
    total_time = end - start
    print(f"Total Test Time: {total_time:.4f} seconds")
    if print_metrics:
        print_performance_metrics(results, total_time)

    predictions = [res[2] for res in results]
    references = [res[3] for res in results]
    wer = load("wer")
    wer_score = 100 * wer.compute(references=references, predictions=predictions)
    print("WER:", wer_score)
    return wer_score


# alternatives "openai/whisper-large-v2", "openai/whisper-large-v3-turbo"..
# NOTE: Expected WER measured with equivalent hf.transformers args:
# whisper-large-v3 + esb-datasets-earnings22-validation-tiny-filtered.
@pytest.mark.parametrize(
    "model_config",
    [
        ("openai/whisper-large-v3", 12.744980),
        # CohereASR is used to test the variable encoder length code paths
        ("CohereLabs/cohere-transcribe-03-2026", 11.92),
    ],
)
# Original dataset is 20GB+ in size, hence we use a pre-filtered slice.
@pytest.mark.parametrize(
    "dataset_repo", ["D4nt3/esb-datasets-earnings22-validation-tiny-filtered"]
)
def test_wer_correctness(
    model_config, dataset_repo, n_examples=-1, max_concurrent_request=None
):
    model_name, expected_wer = model_config
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_name)
    server_args = [
        "--enforce-eager",
        f"--tokenizer_mode={model_info.tokenizer_mode}",
        f"--max_num_seqs={MAX_SEQS_FOR_TRANSCRIPTION_TEST}",
        f"--gpu_memory_utilization={GPU_UTIL_FOR_TRANSCRIPTION_TEST}",
    ]
    if model_info.trust_remote_code:
        server_args.append("--trust-remote-code")
    with RemoteOpenAIServer(
        model_name,
        server_args,
    ) as remote_server:
        dataset = load_shortform_eval_dataset(dataset_repo)

        if not max_concurrent_request:
            # No max concurrency
            max_concurrent_request = n_examples if n_examples > 0 else len(dataset)

        client = remote_server.get_async_client()
        wer = run_evaluation(
            model_name,
            client,
            dataset,
            max_concurrent_request,
            n_examples,
        )

        print(f"Expected WER: {expected_wer}, Actual WER: {wer}")

        if expected_wer:
            torch.testing.assert_close(wer, expected_wer, atol=1e-1, rtol=1e-2)


# 14-22mins of 6 audio samples of total ~115 mins and just 37MB.
# checks for long audio transcription correctness and RMS split.
@pytest.mark.parametrize(
    "model_config",
    [("openai/whisper-large-v3", 9.5)],
)
def test_long_audio_wer_correctness(model_config):
    model_name, expected_wer = model_config
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_name)
    server_args = [
        f"--tokenizer_mode={model_info.tokenizer_mode}",
    ]

    if model_info.trust_remote_code:
        server_args.append("--trust-remote-code")

    # 1800 seconds is 30 minutes
    env_dict = {
        "VLLM_MAX_AUDIO_DECODE_DURATION_S": "1800",
    }

    with RemoteOpenAIServer(
        model_name,
        server_args,
        env_dict=env_dict,
    ) as remote_server:
        dataset = load_longform_dataset()
        client = remote_server.get_async_client()
        wer = run_longform_evaluation(
            model=model_name,
            client=client,
            dataset=dataset,
            max_concurrent_reqs=LONGFORM_NUM_SAMPLES,
        )

    print(f"Expected WER: {expected_wer}, Actual WER: {wer}")
    torch.testing.assert_close(wer, expected_wer, atol=1e-1, rtol=1e-2)
