import json
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from .common import MODEL_TO_NAMES, load_data
import requests
import os
from pathlib import Path

class AcceptanceStatsClient:
    """Client for fetching and processing acceptance statistics data."""
    
    def __init__(self, model_name, method, dataset, data_path=None):
        """Initialize the client with model and dataset info."""
        self.model_name = model_name
        self.method = method
        self.dataset = dataset
        
        if data_path is None:
            self.data_path = f"/data/lily/batch-sd/data/{model_name}/{method}_{dataset}_acceptance_stats.jsonl"
        else:
            self.data_path = data_path
            
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_NAMES[model_name], use_fast=False)
        self.acceptance_stats = None
        
    def load_data(self):
        """Load the acceptance statistics from file."""
        self.acceptance_stats = load_data(self.data_path, self.tokenizer)
        return self.acceptance_stats
    
    def plot_heatmap(self, output_dir="figures"):
        """Plot the acceptance statistics as a heatmap."""
        if self.acceptance_stats is None:
            self.load_data()
            
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(self.acceptance_stats, cmap="YlGnBu")
        plt.xlabel("Position")
        plt.ylabel("Request ID")
        
        # Add Y-axis labels on the right
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks([])
        ax2.set_ylabel("# of Accepted Tokens", labelpad=10)
        
        plt.title(f"Acceptance Statistics: {self.model_name} - {self.method} - {self.dataset}")
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir) / self.model_name
        os.makedirs(output_path, exist_ok=True)
        
        output_file = output_path / f"{self.method}_{self.dataset}_acceptance_stats.pdf"
        plt.savefig(output_file)
        print(f"Saved heatmap to {output_file}")
        return fig
    
    def get_summary_stats(self):
        """Get summary statistics about the acceptance data."""
        if self.acceptance_stats is None:
            self.load_data()
            
        # Calculate average acceptance rate for each position
        avg_by_position = [sum(col)/len(col) for col in zip(*self.acceptance_stats) if sum(1 for v in col if v >= 0) > 0]
        
        # Calculate average acceptance rate for each request
        avg_by_request = [sum(row)/len(row) for row in self.acceptance_stats]
        
        return {
            "total_requests": len(self.acceptance_stats),
            "max_position": len(avg_by_position),
            "avg_acceptance_rate": sum(avg_by_request)/len(avg_by_request),
            "avg_by_position": avg_by_position,
            "avg_by_request": avg_by_request
        }

# Example model configuration
model = "llama3.1-8B"
# model = "r1-distill-llama-8B"
method = "eagle3"
dataset = "mtbench"
# dataset = "aime"
# method = "ngram"
# dataset = "cnndailymail"
# datapath = f"/data/lily/batch-sd/data/{model}/{method}_{dataset}_acceptance_stats.jsonl"
datapath = "acceptance_stats.jsonl"
tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_NAMES[model], use_fast=False)


if __name__ == "__main__":
    # Use the client instead of directly loading data
    client = AcceptanceStatsClient(model, method, dataset, datapath)
    acceptance_stats = client.load_data()
    
    # Get summary statistics
    summary = client.get_summary_stats()
    print("Summary Statistics:")
    print(f"Total Requests: {summary['total_requests']}")
    print(f"Max Position: {summary['max_position']}")
    print(f"Average Acceptance Rate: {summary['avg_acceptance_rate']:.2f}")

    # Create heatmap visualization
    plot_heatmap = False
    if plot_heatmap:
        client.plot_heatmap()

