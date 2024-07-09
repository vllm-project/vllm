
import matplotlib.pyplot as plt
import yaml
import pandas as pd
from tabulate import tabulate


def stringify(x):
    return [str(i) for i in x]


if __name__ == "__main__":
    
    with open("results/chunk_vs_disagg.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    df = pd.DataFrame.from_dict(data)

    print_df = df.copy()
    print_df.drop(columns=[
        "ttft_ratio",
        "itl_ratio",
        "prefill_decode_ratio",
    ], inplace=True)
    print_df.to_csv('results/chunk_vs_disagg.csv', index=False)

    df["chunk_e2e"] = df["chunk_ttft"] + df["chunk_itl"] * df["output_len"]
    df["disagg_e2e"] = df["disagg_ttft"] + df["disagg_itl"] * df["output_len"]
    df["e2e_ratio"] = df["chunk_e2e"] / df["disagg_e2e"]
    
    plt.rcParams['font.size'] = 20
    
    
    # qps vs performance
    qps_df = df[df["output_len"] == 150].copy()
    qps_df.drop(columns=[
        "chunk_itl",
        "chunk_ttft",
        "disagg_itl",
        "disagg_ttft",
        "output_len",
        "prefill_decode_ratio", 
    ], inplace=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    qps_df.plot(
        ax=ax,
        kind="bar",
        x="qps",
        y=["ttft_ratio", "itl_ratio", "e2e_ratio"],
        ylabel="$T_{chunked}~/~T_{disagg}$",
        rot=0,
    )
    ax.hlines(1, -1, 5, color='black')
    fig.savefig('results/qps.png')
    plt.close(fig)

    
    # prefill decode ratio vs performance
    tokens_df = df[df["output_len"] != 12]
    fig, ax = plt.subplots(figsize=(10, 7))
    tokens_df.plot(
        ax=ax,
        kind="bar",
        x="output_len",
        xlabel="# of output tokens",
        y=["ttft_ratio", "itl_ratio", "e2e_ratio", "prefill_decode_ratio"],
        ylabel="$T_{chunked}~/~T_{disagg}$",
        rot=0,
    )
    ax.hlines(1, -1, 5, color='black')
    fig.savefig('results/tokens.png')
    plt.close(fig)
    
    