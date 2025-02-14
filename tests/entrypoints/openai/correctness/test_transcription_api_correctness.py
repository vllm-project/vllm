# SPDX-License-Identifier: Apache-2.0
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
from typing import List

import librosa
import pytest
import soundfile
import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer

from ....utils import RemoteOpenAIServer


def to_bytes(y, sr):
    buffer = io.BytesIO()
    soundfile.write(buffer, y, sr, format="WAV")
    buffer.seek(0)
    return buffer


async def transcribe_audio(client, tokenizer, y, sr):
    # Send loaded audio directly instead of loading from disk,
    # dont account for that time though
    with to_bytes(y, sr) as f:
        start_time = time.perf_counter()
        transcription = await client.audio.transcriptions.create(
            file=f,
            model=tokenizer.name_or_path,
            language="en",
            temperature=0.0,
        )
        end_time = time.perf_counter()
        # NOTE there's no streaming in transcriptions, can't measure ttft
    latency = end_time - start_time
    num_output_tokens = len(
        tokenizer(transcription.text, add_special_tokens=False).input_ids)
    return latency, num_output_tokens, transcription.text


async def bound_transcribe(model_name, sem, client, audio, reference):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Use semaphore to limit concurrent requests.
    async with sem:
        result = await transcribe_audio(client, tokenizer, *audio)
        # Normalize *english* output/reference for evaluation.
        out = tokenizer.normalize(result[2])
        ref = tokenizer.normalize(reference)
        return result[:2] + (out, ref)


async def process_dataset(model, client, data, concurrent_request):
    sem = asyncio.Semaphore(concurrent_request)

    # Warmup call as the first `librosa.load` server-side is quite slow.
    audio, sr = data[0]["audio"]["array"], data[0]["audio"]["sampling_rate"]
    _ = await bound_transcribe(model, sem, client, (audio, sr), "")

    tasks: List[asyncio.Task] = []
    for sample in data:
        audio, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]
        task = asyncio.create_task(
            bound_transcribe(model, sem, client, (audio, sr), sample["text"]))
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
    y, sr = sample['audio']["array"], sample['audio']["sampling_rate"]
    sample['duration_ms'] = librosa.get_duration(y=y, sr=sr) * 1000
    return sample


def load_hf_dataset(dataset_repo: str, split='validation', **hf_kwargs):
    ## Load and filter the dataset
    dataset = load_dataset(dataset_repo, split=split, **hf_kwargs)
    if 'duration_ms' not in dataset[0]:
        # compute duration to filter
        dataset = dataset.map(add_duration)

    # Whisper max supported duration
    dataset = dataset.filter(lambda example: example['duration_ms'] < 30000)
    return dataset


def run_evaluation(model: str,
                   client,
                   dataset,
                   max_concurrent_reqs: int,
                   n_examples: int = -1,
                   print_metrics: bool = True):
    if n_examples > 0:
        dataset = dataset.select(range(n_examples))
    start = time.perf_counter()
    results = asyncio.run(
        process_dataset(model, client, dataset, max_concurrent_reqs))
    end = time.perf_counter()
    total_time = end - start
    print(f"Total Test Time: {total_time:.4f} seconds")
    if print_metrics:
        print_performance_metrics(results, total_time)
    # Compute WER
    predictions = [res[2] for res in results]
    references = [res[3] for res in results]
    wer = load("wer")
    wer_score = 100 * wer.compute(references=references,
                                  predictions=predictions)
    print("WER:", wer_score)
    return wer_score


# alternatives "openai/whisper-large-v2", "openai/whisper-large-v3-turbo"..
@pytest.mark.parametrize("model_name", ["openai/whisper-large-v3"])
# Original dataset is 20GB+ in size, hence we use a pre-filtered slice.
@pytest.mark.parametrize(
    "dataset_repo", ["D4nt3/esb-datasets-earnings22-validation-tiny-filtered"])
# NOTE: Expected WER measured with equivalent hf.transformers args:
# whisper-large-v3 + esb-datasets-earnings22-validation-tiny-filtered.
@pytest.mark.parametrize("expected_wer", [12.744980])
def test_wer_correctness(model_name,
                         dataset_repo,
                         expected_wer,
                         n_examples=-1,
                         max_concurrent_request=None):
    with RemoteOpenAIServer(model_name, ['--enforce-eager']) as remote_server:
        dataset = load_hf_dataset(dataset_repo)

        if not max_concurrent_request:
            # No max concurrency
            max_concurrent_request = n_examples if n_examples > 0\
                else len(dataset)

        client = remote_server.get_async_client()
        wer = run_evaluation(model_name, client, dataset,
                             max_concurrent_request, n_examples)
        if expected_wer:
            torch.testing.assert_close(wer, expected_wer, atol=1e-1, rtol=1e-2)
