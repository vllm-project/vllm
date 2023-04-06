import os
import pickle

import matplotlib.pyplot as plt

STATS = [
    'input_lens',
    'num_running',
    'num_waiting',
    'num_preemption',
    'gpu_cache_usage',
    'cpu_cache_usage',
    'num_swapped',
    'swap_in_lens',
    'swap_out_lens',
]


def plot_stats(output_dir: str):
    # Get timestamps.
    with open(os.path.join(output_dir, 'timestamps.pkl'), 'rb') as f:
        timestamps = pickle.load(f)

    # Draw one figure for each stat.
    num_stats = len(STATS)
    COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    fig, axs = plt.subplots(num_stats, 1, figsize=(10, 2 * num_stats))
    for i, stat in enumerate(STATS):
        with open(os.path.join(output_dir, f'{stat}.pkl'), 'rb') as f:
            data = pickle.load(f)
        axs[i].plot(timestamps, data, color=COLORS[i % len(COLORS)])
        axs[i].set_ylabel(stat.replace('_', ' '), fontdict={'fontsize': 12})
        axs[i].set_ylim(bottom=0)

    plt.xlabel('Time (s)')
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'stats.png')
    plt.savefig(fig_path)
    print(f'Saved stats to {fig_path}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str, help='Output directory.')
    args = parser.parse_args()

    plot_stats(args.output_dir)
