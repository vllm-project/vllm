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


def generate_translation_requests_orca(
    model: str,
    dataset: str,
    num_examples: int,
    request_rate: float,
    duration: int,
    seed: int,
    max_seq_len: int = 2048,
    time_quantum: int = 10,
) -> List[Tuple[float, List[int], SamplingParams]]:
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

    # 8 fixed common examples of about the same length
    # that was generated independently from the WMT16 dataset.
    examples = [
        ('The new shopping center has a wide variety of stores, including clothing, electronics, and a supermarket.'
         ' => Das neue Einkaufszentrum bietet eine große Vielfalt an Geschäften, einschließlich Bekleidung, Elektronik und einem Supermarkt.'),
        ('For a healthier lifestyle, try incorporating regular exercise, a balanced diet, and stress-reducing activities into your routine.'
         ' => Für einen gesünderen Lebensstil versuchen Sie, regelmäßige Bewegung, eine ausgewogene Ernährung und stressreduzierende Aktivitäten in Ihre Routine einzubauen.'),
        ('The library will be hosting a series of workshops on various topics, such as creative writing.'
         ' => Die Bibliothek veranstaltet eine Reihe von Workshops zu verschiedenen Themen wie kreativem Schreiben.'),
        ('The museum offers guided tours every day at 11:00 am and 4:00 pm, and admission is free on Sundays.'
         ' => Das Museum bietet jeden Tag um 11:00 Uhr und 16:00 Uhr Führungen an, und der Eintritt ist sonntags kostenlos.'),
        ('If you experience any technical difficulties during the conference, please don\'t hesitate to contact the support team.'
         ' => Wenn Sie während der Konferenz technische Schwierigkeiten haben, zögern Sie bitte nicht, das Support-Team zu kontaktieren.'),
        ('The local farmer\'s market offers fresh fruits, vegetables, and other produce directly from the farms every Saturday morning.'
         ' => Der örtliche Bauernmarkt bietet jeden Samstagmorgen frische Früchte, Gemüse und andere landwirtschaftliche Produkte direkt von den Höfen an.'),
        ('Remember to set your clocks one hour forward for daylight saving time this weekend to enjoy longer days and more sunlight.'
         ' => Denken Sie daran, Ihre Uhren am Wochenende für die Sommerzeit eine Stunde vorzustellen, um längere Tage und mehr Sonnenlicht zu genießen.'),
        ('The restaurant offers a diverse menu featuring international cuisine, including Italian, French, and Japanese dishes.'
         ' => Das Restaurant bietet eine vielfältige Speisekarte mit internationaler Küche, einschließlich italienischer, französischer und japanischer Gerichte'),
    ]
    assert num_examples <= len(examples)
    prefix += '\n'.join(examples[:num_examples]) + '\n'

    # Tokenize the test set.
    test_set = load_dataset(dataset, 'de-en', split='test')
    tokenized = []
    for data in test_set:
        en = data['translation']['en'] + ' =>'
        input_tokens = tokenizer.encode(prefix + en)

        de = data['translation']['de']
        output_tokens = tokenizer.encode(de, add_special_tokens=False)

        # Filter out too long sequences.
        if len(input_tokens) + len(output_tokens) > max_seq_len:
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
    }
    requests = []
    for timestamp, pair in zip(timestamps, tokenized):
        input_tokens, output_len = pair
        sampling_params = SamplingParams(
            n=1, max_num_steps=output_len, **random_sampling_params_dict)
        requests.append((timestamp, input_tokens, sampling_params))
    return requests
