import pickle
import random
from typing import List, Tuple

from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer

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


def generate_translation_requests(
    model: str,
    dataset: str,
    num_examples: int,
    request_rate: float,
    duration: int,
    seed: int,
    block_size: int,
    max_seq_len: int = 2048,
    time_quantum: int = 10,
) -> Tuple[List[int], List[Tuple[float, List[int], SamplingParams]]]:
    tokenizer = AutoTokenizer.from_pretrained(model)

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

    # Load the training dataset and sample examples.
    train_set = load_dataset('wmt16', 'de-en', split='train')
    train_size = train_set.num_rows
    if num_examples > train_size:
        raise ValueError(
            f'Number of examples ({num_examples}) is greater than the '
            f'number of training examples ({train_size}).')

    # Add instruction first.
    prefix = 'Please translate these following English sentence(s) to German sentence(s):\n'

    # Randomly sample examples from the training dataset and add them to the
    # prefix.
    indices = np.random.choice(train_size, num_examples, replace=False).tolist()
    for i in indices:
        pair = train_set[i]['translation']
        en = pair['en']
        de = pair['de']
        example = f'{en} => {de}\n'
        prefix += example
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=True)

    # If the prefix length is not a multiple of the block size, truncate it.
    prefix_len = len(prefix_tokens)
    remainder_tokens = []
    if prefix_len % block_size != 0:
        remainder_tokens = prefix_tokens[-(prefix_len % block_size):]
        prefix_tokens = prefix_tokens[:-(prefix_len % block_size)]
        prefix_len = len(prefix_tokens)
    
    # Tokenize the test set.
    test_set = load_dataset(dataset, 'de-en', split='test')
    tokenized = []
    for data in test_set:
        en = data['translation']['en'] + ' =>'
        # We skip the <start> token because the tokens will be appended to a prefix.
        en_tokens = tokenizer.encode(en, add_special_tokens=False)
        # NOTE: with byte-pair encoding, encode(a) + encode(b) != encode(a + b) 
        # input_tokens = remainder_tokens + en_tokens
        input_tokens = tokenizer.encode(prefix + en)[prefix_len:]

        de = data['translation']['de']
        output_tokens = tokenizer.encode(de, add_special_tokens=False)

        # Filter out too long sequences.
        if prefix_len + len(input_tokens) + len(output_tokens) > max_seq_len:
            continue
        tokenized.append((input_tokens, len(output_tokens)))

    # Generate requests.
    num_requests = len(timestamps)
    while len(tokenized) < num_requests:
        tokenized += tokenized
    tokenized = tokenized[:num_requests]
    # Shuffle the requests.
    random.shuffle(tokenized)
    random_sampling_params_dict = {
        'temperature': 0.0,
        'top_p': 1.0,
        'use_beam_search': False,
        'stop_token_ids': set(),
        'num_logprobs': 0,
        'context_window_size': None,
        'prefix_id': 0, # FIXME
    }
    requests = []
    for timestamp, pair in zip(timestamps, tokenized):
        input_tokens, output_len = pair
        sampling_params = SamplingParams(
            n=1, max_num_steps=output_len, **random_sampling_params_dict)
        requests.append((timestamp, input_tokens, sampling_params))
    return prefix_tokens, requests
