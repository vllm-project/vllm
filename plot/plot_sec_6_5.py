import argparse
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


SYSTEMS = [
    'FasterTransformer',
    'orca-constant',
    'orca-power2',
    'orca-oracle',
    'cacheflow',
]

SYSTEM_TO_LABEL = {
    'FasterTransformer': 'FasterTransformer',
    'orca-constant': 'Orca (Max)',
    'orca-power2': 'Orca (Pow2)',
    'orca-oracle': 'Orca (Oracle)',
    'cacheflow': 'Astra',
}

SYSTEM_TO_COLOR = {
    'FasterTransformer': 'gray',
    'orca-constant': 'red',
    'orca-power2': 'orange',
    'orca-oracle': 'green',
    'cacheflow': 'blue',
}

SYSTEM_TO_MARKER = {
    'FasterTransformer': '.',
    'orca-constant': 'x',
    'orca-power2': '^',
    'orca-oracle': 's',
    'cacheflow': 'o',
}

MODEL_SHOW_NAME = {
    'opt-13b': 'OPT-13B, 1 GPU',
    'opt-66b': 'OPT-66B, 4 GPUs',
    'opt-175b': 'OPT-175B, 8 GPUs',
}

DATASET_SHOW_NAME = {
    'sharegpt': 'ShareGPT',
    'alpaca': 'Alpaca',
}

MODEL_RANK = {
    'opt-13b': 0,
    'opt-66b': 1,
    'opt-175b': 2,
}


def get_alpha_enum(i: int):
    return '(' + chr(ord('a') + i) + ')'


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
    dir_names = save_dir.split('/')[6:]

    for dir_name in dir_names:
        if dir_name.startswith('orca-'):
            return dir_name
        if dir_name == 'cacheflow':
            return dir_name
        if dir_name == 'ft':
            return 'FasterTransformer'
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

def get_dataset(save_dir: str) -> str:
    save_dir = os.path.abspath(save_dir)
    dir_names = save_dir.split('/')

    for dir_name in dir_names:
        if dir_name == 'alpaca':
            return 'alpaca'
        if dir_name == 'sharegpt':
            return 'sharegpt'
    raise ValueError(f'Cannot find dataset in {save_dir}')


def get_num_shot(save_dir: str) -> str:
    save_dir = os.path.abspath(save_dir)
    dir_names = save_dir.split('/')

    for dir_name in dir_names:
        if dir_name.startswith('wmt'):
            return dir_name.split('-')[1][:-4]
    raise ValueError(f'Cannot find shot number in {save_dir}')


def in_subset(save_dir: str, subset: str):
    if subset == 'n1-alpaca':
        return get_sampling(save_dir) == 'n1' and get_dataset(save_dir) == "alpaca"
    elif subset == 'n1-sharegpt':
        return get_sampling(save_dir) == 'n1' and get_dataset(save_dir) == "sharegpt"
    elif subset == 'parallel':
        if get_dataset(save_dir) != "alpaca":
            return False
        sampling = get_sampling(save_dir)
        return sampling == 'n2' or sampling == 'n4' or sampling == 'n6'
    elif subset == 'beam':
        if get_dataset(save_dir) != "alpaca":
            return False
        sampling = get_sampling(save_dir)
        return sampling == 'n2-beam' or sampling == 'n4-beam' or sampling == 'n6-beam'
    elif subset == 'prefix':
        return 'wmt' in save_dir
    elif subset == 'chat-sharegpt':
        return 'sharegpt_chat' in save_dir


def plot_normalized_latency(
    exp_dir: str,
    subset: str,
    duration: int,
    seed: int,
    warmup: int,
    xlim: Optional[float],
    ylim: Optional[float],
    label_offset: int,
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
        if f'duration-{duration}' not in root and 'duration-900' not in root:
            continue
        if 'unused' in root:
            continue
        if not in_subset(root, subset):
            continue
        save_dirs.append(root)
    # print(save_dirs)

    # Collect data points
    plot_names = []
    # model_name -> x_cut
    x_top: Dict[str, float] = {}
    # model_name -> system -> (request_rate, normalized_latency)
    perf: Dict[str, Dict[str, Tuple[List[float], List[float]]]] = {}
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

        model_name = get_model(save_dir)
        if model_name not in perf:
            perf[model_name] = {}
            plot_names.append(model_name)
        system_name = get_system(save_dir)
        if system_name not in perf[model_name]:
            perf[model_name][system_name] = ([], [])
        perf[model_name][system_name][0].append(request_rate)
        perf[model_name][system_name][1].append(normalized_latency)

        if model_name not in x_top:
            x_top[model_name] = 0
        if normalized_latency < 1.1:
            x_top[model_name] = max(x_top[model_name], request_rate)

        # print('#seqs', len(per_seq_norm_latencies))
        # print(f'{save_dir}: {normalized_latency:.3f} s')


    # Plot normalized latency.
    plot_names = sorted(plot_names)
    fig, axs = plt.subplots(1, 1)
    # for i, (model_name, ax) in enumerate(zip(plot_names, axs)):
    model_name = plot_names[0]
    ax = axs
    i = 0

    curves = []
    legends = []
    for system_name in SYSTEMS:
        if system_name not in perf[model_name]:
            continue
        # Sort by request rate.
        request_rates, normalized_latencies = perf[model_name][system_name]
        request_rates, normalized_latencies = zip(*sorted(zip(request_rates, normalized_latencies)))
        label = SYSTEM_TO_LABEL[system_name]
        color = SYSTEM_TO_COLOR[system_name]
        marker = SYSTEM_TO_MARKER[system_name]
        curve = ax.plot(request_rates, normalized_latencies, label=label, color=color, marker=marker, markersize=6)
        curves.append(curve[0])
        legends.append(label)

    enum = get_alpha_enum(i + label_offset)
    ax.set_xlabel(f'Request rate (req/s)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    if log_scale:
        ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(left=0, right=xlim)
    else:
        ax.set_xlim(left=0, right=x_top[model_name] * 1.1)
    if ylim is not None:
        if log_scale:
            ax.set_ylim(top=ylim)
        else:
            ax.set_ylim(bottom=0, top=ylim)
    ax.grid(linestyle='--')

        # handles, labels = plt.gca().get_legend_handles_labels()
        # handles = reversed(handles)
        # labels = reversed(labels)

        # plt.legend(
        #     handles, labels,
        #     ncol=5, fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 1.15),
        #     columnspacing=0.5, handletextpad=0.5, handlelength=1.5, frameon=False, borderpad=0)

    fig.text(-0.05, 0.45, 'Normalized latency\n       (s/token)', va='center', rotation='vertical', fontsize=14)
    if subset == 'chat-sharegpt':
        fig.legend(curves, legends, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.2), fontsize=14,
                   columnspacing=0.5, frameon=False)
    # fig.subplots_adjust(hspace=0.6)

    # Save figure.
    fig.set_size_inches((6, 2))
    figname = f'{subset}.{format}'
    os.makedirs('./figures', exist_ok=True)
    plt.savefig(os.path.join('figures', figname), bbox_inches='tight')
    print(f'Saved figure to ./figures/{figname}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str)
    parser.add_argument('--subset', choices=['chat-sharegpt'], default='chat-sharegpt')
    parser.add_argument('--duration', type=int, default=3600)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=60)
    parser.add_argument('--xlim', type=float, required=False, default=None)
    parser.add_argument('--ylim', type=float, required=False, default=1)
    parser.add_argument('--label-offset', type=int, default=0)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--format', choices=['png', 'pdf'], default='pdf')
    args = parser.parse_args()

    plot_normalized_latency(
        args.exp_dir, args.subset, args.duration, args.seed, args.warmup,
        args.xlim, args.ylim, args.label_offset, args.log, args.format)
