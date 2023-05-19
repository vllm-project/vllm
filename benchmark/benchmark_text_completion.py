import argparse
import logging
import os
import pickle
import time
from typing import List

from tqdm import tqdm
from transformers import AutoConfig

from benchmark.trace import generate_text_completion_requests
from cacheflow.master.server import (
    add_server_arguments, process_server_arguments,
    init_local_server_and_frontend_with_arguments)
from cacheflow.sampling_params import SamplingParams


logger = logging.getLogger(__name__)


def main(args: argparse.Namespace):
    server, frontend = init_local_server_and_frontend_with_arguments(args)

    # Generate requests.
    requests = generate_text_completion_requests(
        args.dataset,
        args.request_rate,
        args.duration,
        args.seed,
        args.n1,
        args.n2,
        args.n3,
        args.n4,
        args.n6,
        args.n2_beam,
        args.n4_beam,
        args.n6_beam,
        args.n8_beam,
    )

    # Warm up.
    logger.info('Warming up.')
    num_warmup_requests = 8
    warmup_input_len = 8
    warmup_output_len = 32
    warmup_sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=0.99,
        max_num_steps=warmup_output_len,
        use_beam_search=False,
        stop_token_ids=set(),
        num_logprobs=0,
        context_window_size=None,
    )
    for _ in range(num_warmup_requests):
        frontend._add_query([0] * warmup_input_len, warmup_sampling_params)
    server.add_sequence_groups(frontend.get_inputs())
    while True:
        server.step()
        if not server.has_unfinished_requests():
            break

    # Start benchmarking.
    logger.info('Start benchmarking.')
    # Initialize tqdm.
    pbar = tqdm(total=len(requests), desc='Finished requests')

    finished = []
    server.scheduler.reset_stats()
    start_time = time.time()
    while True:
        now = time.time()
        if args.timeout is not None and now - start_time > args.timeout:
            logger.info('Timeout. Stop benchmarking.')
            break

        while requests:
            if requests[0][0] <= now - start_time:
                request_time, input_tokens, sampling_params = requests.pop(0)
                frontend._add_query(
                    input_tokens, sampling_params, arrival_time=start_time + request_time)
            else:
                break
        server.add_sequence_groups(frontend.get_inputs())
        updated_seq_groups = server.step()

        now = time.time()
        for seq_group in updated_seq_groups:
            if not seq_group.is_finished():
                continue
            arrival_time = seq_group.arrival_time
            finish_time = now
            for seq in seq_group.get_seqs():
                seq_len = seq.get_len()
                output_len = seq_len - seq.prompt_len
                finished.append({
                    'group_id': seq_group.group_id,
                    'seq_id': seq.seq_id,
                    'arrival_time': arrival_time,
                    'finish_time': finish_time,
                    'prompt_len': seq.prompt_len,
                    'output_len': output_len,
                })
            pbar.update(1)

        if not (requests or server.has_unfinished_requests()):
            break
    pbar.close()
    logger.info('Finish benchmarking. Saving stats.')
    server.scheduler.save_stats(args.output_dir)
    with open(os.path.join(args.output_dir, 'sequences.pkl'), 'wb') as f:
        pickle.dump(finished, f)
    logger.info('Done.')


def get_model_name(model: str) -> str:
    OPT_MODELS = [
        'opt-125m',
        'opt-350m',
        'opt-1.3b',
        'opt-2.7b',
        'opt-6.7b',
        'opt-13b',
        'opt-30b',
        'opt-66b',
        'opt-175b',
    ]
    for opt_model in OPT_MODELS:
        if opt_model in model:
            return opt_model

    config = AutoConfig.from_pretrained(model)
    assert config.model_type == 'llama'
    hidden_size = config.hidden_size
    if hidden_size == 4096:
        return 'llama-7b'
    elif hidden_size == 5120:
        return 'llama-13b'
    elif hidden_size == 6656:
        return 'llama-30b'
    elif hidden_size == 8192:
        return 'llama-65b'
    else:
        raise ValueError(f'Unknown model: {model}')


def get_dataset_name(dataset: str) -> str:
    if 'sharegpt' in dataset.lower():
        return 'sharegpt'
    elif 'alpaca' in dataset.lower():
        return 'alpaca'
    else:
        raise ValueError(f'Unknown dataset: {dataset}')


def get_sampling_dir_name(
    n1: float,
    n2: float,
    n3: float,
    n4: float,
    n6: float,
    n2_beam: float,
    n4_beam: float,
    n6_beam: float,
    n8_beam: float,
) -> str:
    method = ''
    if n1 > 0.0:
        method = 'n1' if n1 == 1.0 else method + f'n1-{n1}-'
    if n2 > 0.0:
        method = 'n2' if n2 == 1.0 else method + f'n2-{n2}-'
    if n3 > 0.0:
        method = 'n3' if n3 == 1.0 else method + f'n3-{n3}-'
    if n4 > 0.0:
        method = 'n4' if n4 == 1.0 else method + f'n4-{n4}-'
    if n6 > 0.0:
        method = 'n6' if n6 == 1.0 else method + f'n6-{n6}-'
    if n2_beam > 0.0:
        method = 'n2-beam' if n2_beam == 1.0 else method + f'n2-beam-{n2_beam}-'
    if n4_beam > 0.0:
        method = 'n4-beam' if n4_beam == 1.0 else method + f'n4-beam-{n4_beam}-'
    if n6_beam > 0.0:
        method = 'n6-beam' if n6_beam == 1.0 else method + f'n6-beam-{n6_beam}-'
    if n8_beam > 0.0:
        method = 'n8-beam' if n8_beam == 1.0 else method + f'n8-beam-{n8_beam}-'
    return method[:-1] if method.endswith('-') else method


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the performance on a series of requests.')
    parser = add_server_arguments(parser)
    parser.add_argument('--output-dir', type=str, help='path to output directory', default=None)

    parser.add_argument('--dataset', type=str, help='path to dataset', required=True)
    parser.add_argument('--request-rate', type=float, help='reqs/sec', required=True)
    parser.add_argument('--duration', type=int, help='duration in seconds', required=True)
    parser.add_argument('--do-memory-analysis', action='store_true',
        help='do memory analysis (This will lower the throughput. Use this only for analysis.)')
    parser.add_argument('--timeout', type=int, help='time out in seconds', default=None)

    parser.add_argument('--n1', type=float, help='ratio of requests with n=1', default=0.0)
    parser.add_argument('--n2', type=float, help='ratio of requests with n=2', default=0.0)
    parser.add_argument('--n3', type=float, help='ratio of requests with n=3', default=0.0)
    parser.add_argument('--n4', type=float, help='ratio of requests with n=4', default=0.0)
    parser.add_argument('--n6', type=float, help='ratio of requests with n=6', default=0.0)
    parser.add_argument('--n2-beam', type=float, help='ratio of requests with n=2 & beam search', default=0.0)
    parser.add_argument('--n4-beam', type=float, help='ratio of requests with n=4 & beam search', default=0.0)
    parser.add_argument('--n6-beam', type=float, help='ratio of requests with n=6 & beam search', default=0.0)
    parser.add_argument('--n8-beam', type=float, help='ratio of requests with n=8 & beam search', default=0.0)
    args = parser.parse_args()
    args = process_server_arguments(args)
    if args.n1 + args.n2 + args.n3 + args.n4 + args.n6 + args.n2_beam + args.n4_beam + args.n6_beam + args.n8_beam != 1.0:
        raise ValueError('The ratios of requests must sum to 1.')

    model_name = get_model_name(args.model)
    dataset_name = get_dataset_name(args.dataset)
    if 'opt' in model_name:
        if 'opt' not in args.dataset.lower():
            raise ValueError(f'OPT models can only be used with OPT datasets.')
    elif 'llama' in model_name:
        if 'llama' not in args.dataset.lower():
            raise ValueError(f'Llama models can only be used with Llama datasets.')

    dataset_name = 'sharegpt' if 'sharegpt' in args.dataset else 'alpaca'
    sample_dir = get_sampling_dir_name(
        args.n1, args.n2, args.n3, args.n4, args.n6, args.n2_beam, args.n4_beam, args.n6_beam, args.n8_beam)
    if args.output_dir is None:
        args.output_dir = os.path.join(
            '../exp',
            dataset_name,
            f'{model_name}-tp{args.tensor_parallel_size}',
            sample_dir,
            'cacheflow',
            f'block{args.block_size}',
            f'req-rate-{args.request_rate}',
            f'seed{args.seed}',
            f'duration-{args.duration}',
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, 'log.txt')),
        ],
    )
    logger.info(args)

    main(args)
