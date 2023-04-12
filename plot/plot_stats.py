import os
import pickle

import matplotlib.pyplot as plt

STAT_NAMES = [
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
    # Get stats.
    with open(os.path.join(output_dir, 'stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    timestamps = stats['timestamps']

    # Draw one figure for each stat.
    num_stats = len(STAT_NAMES)
    COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'brown', 'gray']
    fig, axs = plt.subplots(num_stats, 1, figsize=(10, 2 * num_stats))
    for i, stat in enumerate(STAT_NAMES):
        data = stats[stat]
        if stat in ['gpu_cache_usage', 'cpu_cache_usage']:
            data = [x * 100 for x in data]
            stat = stat + ' (%)'
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
