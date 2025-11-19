# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import matplotlib.pyplot as plt
import pandas as pd

input_path = "triton_topk_topp_test_wo_fallback.csv"
output_name = "speedup_analysis_wo_fallback.png"


def load_and_parse_data(csv_file):
    """Load CSV data and parse it into a structured format."""
    df = pd.read_csv(csv_file, dtype={"p": str, "k": str})
    print(df.head())
    print(df.columns)
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    print(df.duplicated().sum())
    print(df.shape)
    print(df.head())
    return df


def get_filtered_data(df, vocab_size, p_val, k_val):
    """Filter data for specific vocab_size, p, and k values."""
    # Handle None values properly
    if p_val is None:
        p_condition = df["p"] == "NONE"
    else:
        p_condition = df["p"] == str(p_val)

    if k_val is None:
        k_condition = df["k"] == "NONE"
    else:
        k_condition = df["k"] == str(k_val)

    filtered_df = df[
        (df["vocab_size"] == vocab_size) & p_condition & k_condition
    ].copy()

    return filtered_df.sort_values("batch_size")


def create_speedup_plots(column_configs, vocab_sizes):
    """Create 4x4 grid of speedup vs batch size plots."""
    # Load data
    csv_file = input_path
    df = load_and_parse_data(csv_file)

    # We'll calculate y-axis limits per subplot now

    # Create figure with subplots
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle("Speedup vs Batch Size", fontsize=20, fontweight="bold")

    # Plot each combination
    for row, vocab_size in enumerate(vocab_sizes):
        for col, config in enumerate(column_configs):
            ax = axes[row, col]

            # Get filtered data
            data = get_filtered_data(df, vocab_size, config["p"], config["k"])

            if not data.empty:
                # Calculate y-axis limit for this specific subplot
                local_max_speedup = data["triton_speedup"].max()
                local_y_max = local_max_speedup * 1.1 if local_max_speedup > 0 else 10.0

                # Plot speedup vs batch size
                ax.plot(
                    data["batch_size"],
                    data["triton_speedup"],
                    "bo-",
                    linewidth=2,
                    markersize=6,
                )
                ax.set_xscale("log", base=2)
                ax.set_ylim(0.0, local_y_max)  # Set y-axis range from 0 to local max
                ax.grid(True, alpha=0.3)

                # Add horizontal line at speedup=1
                ax.axhline(
                    y=1,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                    label="Speedup=1",
                )

                # Set labels and title
                if row == 3:  # Bottom row
                    ax.set_xlabel("Batch Size", fontsize=12)
                if col == 0:  # Left column
                    ax.set_ylabel("Speedup", fontsize=12)

                # Set title for top row
                if row == 0:
                    ax.set_title(config["title"], fontsize=14, fontweight="bold")

                # Add vocab size label on the left
                if col == 0:
                    vocab_size_str = f"Vocab Size {vocab_size}"

                    ax.text(
                        -0.2,
                        0.5,
                        vocab_size_str,
                        transform=ax.transAxes,
                        fontsize=14,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        rotation=90,
                    )

                # Set reasonable axis limits
                if len(data) > 0:
                    batch_sizes = data["batch_size"].values

                    ax.set_xlim(batch_sizes.min() * 0.8, batch_sizes.max() * 1.2)
                    # Y-axis is already set to 0-10 above

                # Format x-axis ticks
                ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
                ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, ""])

                # Add legend only to the first subplot
                if row == 0 and col == 0:
                    ax.legend(loc="upper left")

            else:
                # No data available - use default y-axis range
                default_y_max = 10.0
                ax.text(
                    0.5,
                    0.5,
                    "No Data\nAvailable",
                    transform=ax.transAxes,
                    fontsize=12,
                    ha="center",
                    va="center",
                    color="red",
                )
                ax.set_xlim(1, 2048)
                ax.set_ylim(0.0, default_y_max)  # Set y-axis range from 0 to default
                ax.set_xscale("log", base=2)
                ax.grid(True, alpha=0.3)

                # Add horizontal line at speedup=1
                ax.axhline(y=1, color="red", linestyle="--", linewidth=2, alpha=0.7)

                if row == 3:  # Bottom row
                    ax.set_xlabel("Batch Size", fontsize=12)
                if col == 0:  # Left column
                    ax.set_ylabel("triton_speedup", fontsize=12)
                if row == 0:
                    ax.set_title(config["title"], fontsize=12, fontweight="bold")
                if col == 0:
                    ax.text(
                        -0.2,
                        0.5,
                        f"Vocab Size {vocab_size}",
                        transform=ax.transAxes,
                        fontsize=14,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        rotation=90,
                    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, left=0.08)

    # Save the plot
    output_file = f"./{output_name}"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Speedup analysis plot saved to: {output_file}")

    # Show the plot
    plt.show()

    return fig


def print_data_summary(column_configs, vocab_sizes):
    """Print a summary of the available data."""
    csv_file = input_path
    df = load_and_parse_data(csv_file)

    print("Data Summary:")
    print(f"Total rows: {len(df)}")
    print(f"Unique batch sizes: {sorted(df['batch_size'].unique())}")
    print(f"Unique vocab sizes: {sorted(df['vocab_size'].unique())}")
    print(f"Unique p values: {sorted([p for p in df['p'].unique() if p != 'nan'])}")
    print(f"Unique k values: {sorted([k for k in df['k'].unique() if k != 'nan'])}")
    print()

    print("Data availability matrix:")
    print("Rows: Vocab sizes, Columns: Parameter combinations")
    print("Values: Number of data points available")
    print()

    header = f"{'Vocab Size':<12}"
    for config in column_configs:
        header += f"{config['title']:<15}"
    print(header)
    print("-" * len(header))

    for vocab_size in vocab_sizes:
        row = f"{vocab_size:<12}"
        for config in column_configs:
            data = get_filtered_data(df, vocab_size, config["p"], config["k"])
            row += f"{len(data):<15}"
        print(row)


if __name__ == "__main__":
    column_configs = [
        {"p": None, "k": 50, "title": "P=None, K=50"},
        {"p": 0.9, "k": None, "title": "P=0.9, K=None"},
        {"p": 0.9, "k": 50, "title": "P=0.9, K=50"},
        {"p": "RAND", "k": 3000, "title": "P=RAND, K=3000"},
    ]
    vocab_sizes = [16384, 65536, 102400, 128256]
    # Print data summary first
    print_data_summary(column_configs, vocab_sizes)
    print("\n" + "=" * 80 + "\n")

    # Create the plots
    create_speedup_plots(column_configs, vocab_sizes)
