import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

save_dir = sys.argv[-1]

RESULTS = [
    'input_lens.pkl',
    'num_pendings.pkl',
    'next_seq_lens.pkl',
    'swap_out_lens.pkl',
    'swap_in_lens.pkl',
    'gpu_cache_usage.pkl',
    'cpu_cache_usage.pkl',
    'requests_received.pkl',
]

for result in RESULTS:
    with open(save_dir + result, 'rb') as f:
        data = pickle.load(f)

    if result == 'input_lens.pkl':
        p = []
        g = []
        s = []
        for step in data:
            p.append(step[0])
            g.append(step[1])
            s.append(step[0] + step[1])
    elif result == 'num_pendings.pkl':
        n = []
        for step in data:
            n.append(step)
    elif result == 'next_seq_lens.pkl':
        i = []
        o = []
        for step in data:
            i.append(step[0])
            o.append(step[1])
    elif result == 'swap_out_lens.pkl':
        so = []
        for step in data:
            so.append(step)
    elif result == 'swap_in_lens.pkl':
        si = []
        for step in data:
            si.append(step)
    elif result == 'gpu_cache_usage.pkl':
        gpu = []
        for step in data:
            gpu.append(step * 100.0)
    elif result == 'cpu_cache_usage.pkl':
        cpu = []
        for step in data:
            cpu.append(step * 100.0)
    elif result == 'requests_received.pkl':
        r = []
        for step in data:
            r.append(step)

# Get the last timestep that r is not 0.
last_step = len(r) - 1
while r[last_step] == 0 and last_step >= 0:
    last_step -= 1

print(f'Median input len: {np.median(s)}')

# Draw four figures in one row.
# In each figure, draw a vertical line to indicate the last idx that p is not 0.

fig, axs = plt.subplots(5, 1, figsize=(10, 10))

axs[0].plot(r, label='requests_recieved', color='magenta')
axs[0].set_ylabel('# request arrivals')

axs[1].plot(s, label='batch_size', color='red')
axs[1].set_ylabel('# input tokens')

axs[2].plot(n, label='num_pendings', color='blue')
axs[2].set_ylabel('# pending seqs')

axs[3].plot(gpu, label='gpu_cache_usage', color='green')
axs[3].set_ylabel('GPU cache util (%)')

axs[4].plot(cpu, label='cpu_cache_usage', color='orange')
axs[4].set_ylabel('CPU cache util (%)')

for ax in axs:
    ax.axvline(x=last_step, color='black', linestyle='--')

axs[-1].set_xlabel('timestep')
plt.savefig(save_dir + 'results.png')
