import argparse
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


SYSTEMS = [
    'orca-constant',
    'orca-power2',
    'orca-oracle',
    'cacheflow',
]

SYSTEM_TO_LABEL = {
    'orca-constant': 'Orca (Max)',
    'orca-power2': 'Orca (Next power of 2)',
    'orca-oracle': 'Orca (Oracle)',
    'cacheflow': 'CacheFlow',
}

SYSTEM_TO_COLOR = {
    'orca-constant': 'red',
    'orca-power2': 'orange',
    'orca-oracle': 'green',
    'cacheflow': 'blue',
}

SYSTEM_TO_MARKER = {
    'orca-constant': 'x',
    'orca-power2': '^',
    'orca-oracle': 's',
    'cacheflow': 'o',
}


def get_results(save_dir: str) -> List[Dict[str, Any]]:
    with open(os.path.join(save_dir, 'sequences.pkl'), 'rb') as f:
        results = pickle.load(f)
    return results


def get_request_rate(save_dir: str) -> float:
    """Get request rate from save_dir name."""
    # Directory name format:
    # .../req-rate-{req_rate}/seed-{seed}/duration-{duration}
    save_dir = os.path.abspath(save_dir)
    dir_names = save_dir.split('/')

    request_rate = None
    for dir_name in dir_names:
        if dir_name.startswith('req-rate-'):
            if request_rate is not None:
                raise ValueError(f'Found multiple request rates in {save_dir}')
            request_rate = float(dir_name.split('-')[-1])
    if request_rate is None:
        raise ValueError(f'Cannot find request rate in {save_dir}')
    return request_rate


def get_model(save_dir: str) -> Tuple[str, int]:
    save_dir = os.path.abspath(save_dir)
    dir_names = save_dir.split('/')

    model = None
    for dir_name in dir_names:
        if '-tp' in dir_name:
            if model is not None:
                raise ValueError(f'Found multiple models in {save_dir}')
            model = dir_name.split('-tp')[0]
            tp = int(dir_name.split('-tp')[-1])
    if model is None:
        raise ValueError(f'Cannot find model in {save_dir}')
    return model, tp


def get_system(save_dir: str) -> str:
    save_dir = os.path.abspath(save_dir)
    dir_names = save_dir.split('/')

    for dir_name in dir_names:
        if dir_name.startswith('orca-'):
            return dir_name
        if dir_name == 'cacheflow':
            return dir_name
    raise ValueError(f'Cannot find system in {save_dir}')


def get_sampling(save_dir: str) -> str:
    save_dir = os.path.abspath(save_dir)
    dir_names = save_dir.split('/')

    for dir_name in dir_names:
        if dir_name.startswith('n'):
            if dir_name.endswith('-beam'):
                return dir_name
            if dir_name[1:].isdigit():
                return dir_name
    raise ValueError(f'Cannot find sampling method in {save_dir}')


def plot_normalized_latency(
    exp_dir: str,
    duration: int,
    seed: int,
    warmup: int,
    xlim: Optional[float],
    ylim: Optional[float],
    log_scale: bool,
    format: str,
) -> None:
    # Get leaf directories.
    save_dirs = []
    for root, dirs, files in os.walk(exp_dir):
        if dirs:
            continue
        if 'sequences.pkl' not in files:
            continue
        if f'seed{seed}' not in root:
            continue
        if f'duration-{duration}' not in root:
            continue
        save_dirs.append(root)

    # Plot normalized latency.
    perf_per_system: Dict[str, Tuple[List[float], List[float]]] = {}
    for save_dir in save_dirs:
        per_seq_norm_latencies = []
        results = get_results(save_dir)
        for seq in results:
            arrival_time = seq['arrival_time']
            finish_time = seq['finish_time']
            output_len = seq['output_len']
            if arrival_time < warmup:
                continue
            latency = finish_time - arrival_time
            norm_latency = latency / output_len
            per_seq_norm_latencies.append(norm_latency)

        request_rate = get_request_rate(save_dir)
        normalized_latency = np.mean(per_seq_norm_latencies)
        system_name = get_system(save_dir)
        if system_name not in perf_per_system:
            perf_per_system[system_name] = ([], [])
        perf_per_system[system_name][0].append(request_rate)
        perf_per_system[system_name][1].append(normalized_latency)

        print('#seqs', len(per_seq_norm_latencies))
        print(f'{save_dir}: {normalized_latency:.3f} s')


    # Plot normalized latency.
    plt.figure(figsize=(6, 4))
    for system_name in reversed(SYSTEMS):
        if system_name not in perf_per_system:
            continue
        # Sort by request rate.
        request_rates, normalized_latencies = perf_per_system[system_name]
        request_rates, normalized_latencies = zip(*sorted(zip(request_rates, normalized_latencies)))
        label = SYSTEM_TO_LABEL[system_name]
        color = SYSTEM_TO_COLOR[system_name]
        marker = SYSTEM_TO_MARKER[system_name]
        plt.plot(request_rates, normalized_latencies, label=label, color=color, marker=marker)

    # plt.legend()
    plt.xlabel('Request rate (req/s)', fontsize=12)
    plt.ylabel('Normalized latency (s/token)', fontsize=12)

    if log_scale:
        plt.yscale('log')
    if xlim is not None:
        plt.xlim(left=0, right=xlim)
    if ylim is not None:
        if log_scale:
            plt.ylim(top=ylim)
        else:
            plt.ylim(bottom=0, top=ylim)

    # Save figure.
    model, tp = get_model(exp_dir)
    sampling = get_sampling(exp_dir)
    figname = f'{model}-tp{tp}-{sampling}.{format}'
    os.makedirs('./figures', exist_ok=True)
    plt.savefig(os.path.join('figures', figname), bbox_inches='tight')
    print(f'Saved figure to ./figures/{figname}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str)
    parser.add_argument('--duration', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=60)
    parser.add_argument('--xlim', type=float, required=False, default=None)
    parser.add_argument('--ylim', type=float, required=False, default=None)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--format', choices=['png', 'pdf'], default='png')
    args = parser.parse_args()

    plot_normalized_latency(
        args.exp_dir, args.duration, args.seed, args.warmup, args.xlim, args.ylim, args.log, args.format)
