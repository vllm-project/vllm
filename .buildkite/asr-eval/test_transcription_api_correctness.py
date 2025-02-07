# SPDX-License-Identifier: Apache-2.0
"""
Evaluate Transcription API correctness by computing Word Error Rate (WER)
on a given ASR dataset. When provided, it will also compare the WER against
a baseline.
"""
import asyncio
import io
import time
from argparse import ArgumentParser
from statistics import mean, median
from typing import List, Optional

import librosa
import soundfile
import torch
from datasets import load_dataset
from evaluate import load
from openai import AsyncOpenAI
from transformers import AutoTokenizer

openai_api_base = "http://localhost:8000/v1"
client = AsyncOpenAI(api_key="EMPTY", base_url=openai_api_base)


def to_bytes(y, sr):
    buffer = io.BytesIO()
    soundfile.write(buffer, y, sr, format="WAV")
    buffer.seek(0)
    return buffer


async def transcribe_audio(client, tokenizer, y, sr):
    status = 200
    try:
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
    except Exception as e:
        print(f"Error: {e}")
        status = 500
    # Hard check on server working properly
    assert status == 200
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


async def process_dataset(model, data, concurrent_request):
    sem = asyncio.Semaphore(concurrent_request)
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


def load_hf_dataset(dataset_repo: str,
                    dataset_name: str,
                    split='validation',
                    **hf_kwargs):
    ## Load and filter the dataset
    dataset = load_dataset(dataset_repo,
                           dataset_name,
                           split=split,
                           **hf_kwargs)
    if 'duration_ms' not in dataset[0]:
        # compute duration to filter
        dataset = dataset.map(add_duration)

    # Whisper max supported duration
    dataset = dataset.filter(lambda example: example['duration_ms'] < 30000)
    return dataset


def run_evaluation(model: str,
                   dataset,
                   n_examples: int = -1,
                   max_concurrent_reqs: Optional[int] = None,
                   print_metrics: bool = True):
    if n_examples > 0:
        dataset = dataset.select(range(n_examples))

    # Warmup
    _ = asyncio.run(
        process_dataset(model, dataset.select(range(1)), max_concurrent_reqs))

    start = time.perf_counter()
    results = asyncio.run(process_dataset(model, dataset, max_concurrent_reqs))
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


if __name__ == "__main__":
    args = ArgumentParser()
    # alternatives "openai/whisper-large-v2", "openai/whisper-large-v3-turbo"..
    args.add_argument("-m",
                      "--model-name",
                      type=str,
                      help="Name of the ASR model to evaluate.",
                      default="openai/whisper-large-v3")
    args.add_argument("-dr",
                      "--dataset-repo",
                      type=str,
                      help="Path/repo of the hf asr dataset to test on.")
    args.add_argument("-dn",
                      "--dataset-name",
                      type=str,
                      help="Name of the hf asr dataset to test on.")
    args.add_argument("--n-examples",
                      type=int,
                      help="Limit the number of examples to evaluate on.",
                      default=-1)
    args.add_argument(
        "--max-concurrent-request",
        type=int,
        help="Limit the number of requests sent to the server at the same time"
    )
    args.add_argument("--expected-wer",
                      type=float,
                      help="Expected WER to compare against.")
    args.add_argument(
        "--extra",
        nargs="*",
        help="Extra keyword arguments (key=value pairs) to be passed "
        "to hf `load_dataset`")
    args = args.parse_args()

    extra_kwargs = {}
    if args.extra:
        for item in args.extra:
            key, value = item.split("=", 1)
            extra_kwargs[key] = value

    print("Running evaluation with args", vars(args))
    dataset = load_hf_dataset(args.dataset_repo, args.dataset_name,
                              **extra_kwargs)

    if not args.max_concurrent_request:
        # No max concurrency
        args.max_concurrent_request = args.n_examples if args.n_examples > 0\
              else len(dataset)

    wer = run_evaluation(args.model_name, dataset, args.n_examples,
                         args.max_concurrent_request)
    if args.expected_wer:
        torch.testing.assert_close(wer,
                                   args.expected_wer,
                                   atol=1e-1,
                                   rtol=1e-2)
