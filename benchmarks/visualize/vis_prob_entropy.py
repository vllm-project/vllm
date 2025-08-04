from transformers import AutoTokenizer
from common import MODEL_TO_NAMES, load_data
import matplotlib.pyplot as plt


def plot_prob_entropy(acceptance_stats, 
                    output_path):
    
    acc_probs = []
    rej_probs = []
    for stat in acceptance_stats:
        for i, acc_len in enumerate(stat.lens):
            acc_probs.extend(stat.probs[i][:acc_len-1])
            rej_probs.extend(stat.probs[i][acc_len-1:])

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.hist(acc_probs, bins=100, alpha=0.5, 
             label='Accepted Probabilities', color='green')
    plt.hist(rej_probs, bins=100, alpha=0.5, 
             label='Rejected Probabilities', color='red')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Accepted and Rejected Probabilities')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    datapath = "/data/lily/sd-benchmark-paper/batch-sd/acceptance_stats.jsonl"
    model = "llama3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_NAMES[model], 
                                              use_fast=False)
    acceptance_stats = load_data(datapath, tokenizer)
    plot_prob_entropy(acceptance_stats, output_path="prob_entropy_figures")



