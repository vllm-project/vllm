import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

model = "llama3.1-8B"
dataset = "instructcode"
method1 = "ngram"
method2 = "eagle3"

def get_datapath(method):
    datapath = f"/data/lily/batch-sd/data/{model}/{method}_{dataset}_acceptance_stats.jsonl"
    return datapath

def cleanup(data):
    # Remove the prefill phase
    data = data[1:]
    # Cap the maximum value to 10
    data = [min(x, 10) for x in data]
    return data

def load_data(datapath):
    acceptance_stats = {}
    with open(datapath, "r") as f:
        lines = f.readlines()
        for line in lines:  
            data = json.loads(line)
            key = hash(tuple(data['prompt_token_ids']))
            acceptance_stats[key] = cleanup(data['acc'])
    # Pad the acceptance stats to the same length
    max_length = max(len(stats) for k, stats in acceptance_stats.items())
    
    for key in acceptance_stats:
        acceptance_stats[key] += [-2] * (max_length - len(acceptance_stats[key]))
        
    print(f"Load {len(acceptance_stats)} with max length {max_length} from {datapath}")
    return acceptance_stats

def diff(acceptance_stats1, acceptance_stats2):
    diff = {}
    for key in acceptance_stats1:
        if key in acceptance_stats2:
            diff[key] = [a - b for a, b in zip(acceptance_stats1[key], acceptance_stats2[key])]
    return diff

datapath_1 = get_datapath(method1)
datapath_2 = get_datapath(method2)
acceptance_stats_1 = load_data(datapath_1)
acceptance_stats_2 = load_data(datapath_2)
acceptance_stats_diff = diff(acceptance_stats_1, acceptance_stats_2)

acceptance_stats = list(acceptance_stats_diff.values())


fig, ax = plt.subplots()
colors = ["red", "white", "blue"]
custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)
sns.heatmap(acceptance_stats, cmap=custom_cmap, center=0)
plt.xlabel("Position")
plt.ylabel("Request ID")
# Add Y-axis labels on the right
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())              # Match y-axis range
ax2.set_yticks([])                       # Remove right tick marks if undesired
ax2.set_ylabel("# of Accepted Tokens", labelpad=10)         # Set right y-axis label
plt.title(f"Diff between {method2} - {method1} acceptance stats for {dataset}")

plt.tight_layout()
plt.savefig(f"figures/{model}/diff_{method2}_{method1}_{dataset}_acceptance_stats.pdf")
