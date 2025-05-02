import json
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


model = "r1-distill-llama-8B"
MODEL_TO_NAMES = {
    "r1-distill-llama-8B" : "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
}
method = "ngram"
dataset = "aime"
datapath = f"/data/lily/batch-sd/data/{model}/{method}_{dataset}_acceptance_stats.jsonl"
tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_NAMES[model], use_fast=False)

def cleanup(data):
    # Remove the prefill phase
    data = data[1:]
    # Cap the maximum value to 10
    data = [min(x, 10) for x in data]
    return data

def load_data(datapath):
    acceptance_stats = []
    with open(datapath, "r") as f:
        lines = f.readlines()
        for line in lines:  
            data = json.loads(line)
            acceptance_stats.append(cleanup(data['acc']))
            print("Input:", tokenizer.decode(data['prompt_token_ids']))
            print("Output:", tokenizer.decode(data['generated_token_ids']))
            print("=============================================")
            
    # Pad the acceptance stats to the same length
    max_length = max(len(stats) for stats in acceptance_stats)
    for i in range(len(acceptance_stats)):
        acceptance_stats[i] += [-2] * (max_length - len(acceptance_stats[i]))
        
    print(f"Load {len(acceptance_stats)} with max length {max_length}")
    return acceptance_stats

acceptance_stats = load_data(datapath)


fig, ax = plt.subplots()
sns.heatmap(acceptance_stats, cmap="YlGnBu")
plt.xlabel("Position")
plt.ylabel("Request ID")
# Add Y-axis labels on the right
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())              # Match y-axis range
ax2.set_yticks([])                       # Remove right tick marks if undesired
ax2.set_ylabel("# of Accepted Tokens", labelpad=10)         # Set right y-axis label


plt.tight_layout()
plt.savefig(f"figures/{model}/{method}_{dataset}_acceptance_stats.png")
