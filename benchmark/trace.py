import pickle
import random
from typing import List, Tuple

import numpy as np

from cacheflow.sampling_params import SamplingParams


def generate_text_completion_requests(
    dataset: str,
    request_rate: float,
    duration: int,
    seed: int,
    n1: float = 0.0,
    n2: float = 0.0,
    n3: float = 0.0,
    n4: float = 0.0,
    n6: float = 0.0,
    n2_beam: float = 0.0,
    n4_beam: float = 0.0,
    n6_beam: float = 0.0,
    n8_beam: float = 0.0,
    max_seq_len: int = 2048,
    time_quantum: int = 10,
) -> List[Tuple[float, List[int], SamplingParams]]:
    random.seed(seed)
    np.random.seed(seed)

    # Generate timestamps for requests using Poisson distribution.
    lam = request_rate * (time_quantum / 1000)
    quantums_per_sec = 1000 / time_quantum
    arrival_times = np.random.poisson(
        lam=lam, size=int(duration * quantums_per_sec))
    timestamps = []
    for i, n in enumerate(arrival_times):
        timestamps += [i * (time_quantum / 1000)] * n

    # Load and shuffle the dataset.
    num_requests = len(timestamps)
    with open(dataset, 'rb') as f:
        data = pickle.load(f)

    filtered = []
    for pair in data:
        input_tokens, output_tokens = pair
        input_len = len(input_tokens)
        output_len = len(output_tokens)
        # Filter out too long sequences.
        if input_len + output_len < max_seq_len:
            # Output tokens are not needed for the benchmark.
            filtered.append((input_tokens, output_len))

    data = []
    while len(data) < num_requests:
        data += filtered
    data = data[:num_requests]
    # Shuffle the data.
    assert len(data) == len(timestamps)
    random.shuffle(data)

    random_sampling_params_dict = {
        'temperature': 1.0,
        'top_p': 1.0,
        'use_beam_search': False,
        'stop_token_ids': set(),
        'num_logprobs': 0,
        'context_window_size': None,
    }
    beam_search_params_dict = {
        'temperature': 0.0,
        'top_p': 1.0,
        'use_beam_search': True,
        'stop_token_ids': set(),
        'num_logprobs': 0,
        'context_window_size': None,
    }

    # Generate requests based on the sampling parameter ratio.
    requests = []
    assert n1 + n2 + n3 + n4 + n6 + n2_beam + n4_beam + n6_beam + n8_beam == 1.0
    cum_sum = 0
    for timestamp, pair in zip(timestamps, data):
        input_tokens, output_len = pair
        if cum_sum < n1 * num_requests:
            sampling_params = SamplingParams(
                n=1, max_num_steps=output_len, **random_sampling_params_dict)
        elif cum_sum < (n1 + n2) * num_requests:
            sampling_params = SamplingParams(
                n=2, max_num_steps=output_len, **random_sampling_params_dict)
        elif cum_sum < (n1 + n2 + n3) * num_requests:
            sampling_params = SamplingParams(
                n=3, max_num_steps=output_len, **random_sampling_params_dict)
        elif cum_sum < (n1 + n2 + n3 + n4) * num_requests:
            sampling_params = SamplingParams(
                n=4, max_num_steps=output_len, **random_sampling_params_dict)
        elif cum_sum < (n1 + n2 + n3 + n4 + n6) * num_requests:
            sampling_params = SamplingParams(
                n=6, max_num_steps=output_len, **random_sampling_params_dict)
        elif cum_sum < (n1 + n2 + n3 + n4 + n6 + n2_beam) * num_requests:
            sampling_params = SamplingParams(
                n=2, max_num_steps=output_len, **beam_search_params_dict)
        elif cum_sum < (n1 + n2 + n3 + n4 + n6 + n2_beam + n4_beam) * num_requests:
            sampling_params = SamplingParams(
                n=4, max_num_steps=output_len, **beam_search_params_dict)
        elif cum_sum < (n1 + n2 + n3 + n4 + n6 + n2_beam + n4_beam + n6_beam) * num_requests:
            sampling_params = SamplingParams(
                n=6, max_num_steps=output_len, **beam_search_params_dict)
        elif cum_sum < (n1 + n2 + n3 + n4 + n6 + n2_beam + n4_beam + n6_beam + n8_beam) * num_requests:
            sampling_params = SamplingParams(
                n=8, max_num_steps=output_len, **beam_search_params_dict)
        else:
            raise ValueError('Invalid request ratio.')
        cum_sum += 1
        requests.append((timestamp, input_tokens, sampling_params))
    return requests


def generate_chatbot_requests(
    dataset: str,
    request_rate: float,
    duration: int,
    seed: int,
    n1: float = 0.0,
    n2: float = 0.0,
    n3: float = 0.0,
    n4: float = 0.0,
    n6: float = 0.0,
    n2_beam: float = 0.0,
    n4_beam: float = 0.0,
    n6_beam: float = 0.0,
    n8_beam: float = 0.0,
    max_seq_len: int = 2048,
    time_quantum: int = 10,
) -> List[Tuple[float, List[int], SamplingParams]]:
    random.seed(seed)
    np.random.seed(seed)

    # Generate timestamps for requests using Poisson distribution.
    lam = request_rate * (time_quantum / 1000)
    quantums_per_sec = 1000 / time_quantum
    arrival_times = np.random.poisson(
        lam=lam, size=int(duration * quantums_per_sec))
    timestamps = []
    for i, n in enumerate(arrival_times):
        timestamps += [i * (time_quantum / 1000)] * n

    # Load and shuffle the dataset.
    num_requests = len(timestamps)
    with open(dataset, 'rb') as f:
        data = pickle.load(f)

    conversations = []
    max_input_len = max_seq_len // 2
    ct = 0
    for i in range(len(data)):
        seqs = [x[1] for x in data[i]]
        seqs = seqs[:len(seqs) // 2 * 2]  # Assume human, gpt, human, gpt, ...
        if len(seqs) <= 0:
            continue
        input_tokens = []
        conv = []
        for j in range(0, len(seqs), 2):
            input_tokens += seqs[j]
            output_len = min(len(seqs[j+1]), max_seq_len - max_input_len - 8)
            conv.append((input_tokens[-max_input_len:], output_len))
            input_tokens += seqs[j+1]
            input_tokens = input_tokens[-max_seq_len:]

            ct += 1
            if ct >= num_requests:
                break
        conversations.append(conv)
        if ct >= num_requests:
            break
    assert ct == num_requests

    random.shuffle(conversations)

    random_sampling_params_dict = {
        'temperature': 1.0,
        'top_p': 1.0,
        'use_beam_search': False,
        'stop_token_ids': set(),
        'num_logprobs': 0,
        'context_window_size': None,
    }
    beam_search_params_dict = {
        'temperature': 0.0,
        'top_p': 1.0,
        'use_beam_search': True,
        'stop_token_ids': set(),
        'num_logprobs': 0,
        'context_window_size': None,
    }

    # Generate requests based on the sampling parameter ratio.
    assert n1 + n2 + n3 + n4 + n6 + n2_beam + n4_beam + n6_beam + n8_beam == 1.0
    cum_sum = 0
    conv_requests = []
    for i in range(len(conversations)):
        conv_request = []
        for input_tokens, output_len in conversations[i]:
            if cum_sum < n1 * num_requests:
                sampling_params = SamplingParams(
                    n=1, max_num_steps=output_len, **random_sampling_params_dict)
            elif cum_sum < (n1 + n2) * num_requests:
                sampling_params = SamplingParams(
                    n=2, max_num_steps=output_len, **random_sampling_params_dict)
            elif cum_sum < (n1 + n2 + n3) * num_requests:
                sampling_params = SamplingParams(
                    n=3, max_num_steps=output_len, **random_sampling_params_dict)
            elif cum_sum < (n1 + n2 + n3 + n4) * num_requests:
                sampling_params = SamplingParams(
                    n=4, max_num_steps=output_len, **random_sampling_params_dict)
            elif cum_sum < (n1 + n2 + n3 + n4 + n6) * num_requests:
                sampling_params = SamplingParams(
                    n=6, max_num_steps=output_len, **random_sampling_params_dict)
            elif cum_sum < (n1 + n2 + n3 + n4 + n6 + n2_beam) * num_requests:
                sampling_params = SamplingParams(
                    n=2, max_num_steps=output_len, **beam_search_params_dict)
            elif cum_sum < (n1 + n2 + n3 + n4 + n6 + n2_beam + n4_beam) * num_requests:
                sampling_params = SamplingParams(
                    n=4, max_num_steps=output_len, **beam_search_params_dict)
            elif cum_sum < (n1 + n2 + n3 + n4 + n6 + n2_beam + n4_beam + n6_beam) * num_requests:
                sampling_params = SamplingParams(
                    n=6, max_num_steps=output_len, **beam_search_params_dict)
            elif cum_sum < (n1 + n2 + n3 + n4 + n6 + n2_beam + n4_beam + n6_beam + n8_beam) * num_requests:
                sampling_params = SamplingParams(
                    n=8, max_num_steps=output_len, **beam_search_params_dict)
            else:
                raise ValueError('Invalid request ratio.')
            cum_sum += 1
            conv_request.append((input_tokens, sampling_params))
        conv_requests.append(conv_request)
    assert cum_sum == num_requests
    return conv_requests, timestamps
