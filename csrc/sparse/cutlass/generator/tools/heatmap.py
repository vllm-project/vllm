import pickle as pkl
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from select_kernels import select_kernels
from utils import Data, make_heatmap_data, measurement_to_data


def plot_heatmap(data: np.array,
                 y_labels: List[str],
                 x_labels: List[str],
                 save_filename='heatmap.png'):
    # min because of some matplotlib render restrictions.
    fig_size_x = min(len(x_labels), 320)
    fig_size_y = len(y_labels) + 25
    fig, ax = plt.subplots(figsize=(fig_size_x, fig_size_y))
    im = ax.imshow(data, cmap="Reds", vmin=0.0, vmax=1.0, interpolation=None)

    cbar = ax.figure.colorbar(im, ax=ax, cmap="Reds")
    cbar.ax.set_ylabel("Hot == Closer to peak perf.", rotation=90, va="top")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90)

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            ax.text(j,
                    i,
                    data[i, j],
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=6.0)

    #ax.colorbar()

    ax.set_title("GEMM shape vs Best cutlass op")
    #ax.set_aspect('equal')
    fig.tight_layout()

    #fig.set_dpi(300)
    #plt.show()
    print(f"Save location : {save_filename}")
    fig.savefig(save_filename, dpi=100)
    #fig.savefig(save_filename, dpi=10)


def select_top_k_kernels(gemm_ops: np.array,
                         gemm_problems: List[str],
                         ops: List[str],
                         k: int = 100) -> List[str]:
    """
    Simple top_k kernel selection. 
    Gather the top-k best performing kernels for each gemm problem and
    return the union.
    """
    n_rows = len(gemm_problems)

    max_kernels_per_gemm_shape = 100  # k-value
    gemm_efficiency_threshold = 0.90

    selected_ops = []
    for r in range(n_rows):
        gemm_ops_list = np.copy(gemm_ops[r])
        sorted_indices = list(reversed(np.argsort(gemm_ops_list).tolist()))

        selected_shape_ops = []
        for x in sorted_indices:
            if 'autogen' not in ops[x]:
                # select only autogen kernels/ops
                continue
            if len(selected_shape_ops) >= max_kernels_per_gemm_shape:
                break
            # we have reached the min requirement. Decide to break based on
            # the gemm_efficiency threshold.
            if gemm_ops_list[x] < gemm_efficiency_threshold:
                break
            else:
                selected_shape_ops.append(ops[x])

        selected_ops.append(selected_shape_ops)

        op_scores = []
        for idx in range(len(selected_shape_ops)):
            if 'autogen' not in ops[sorted_indices[idx]]:
                continue
            op_scores.append(gemm_ops_list[sorted_indices[idx]])
        print(f"Gemm problem {gemm_problems[r]} "
              f"- #kernels {len(selected_shape_ops)} "
              f"- selected kernel range [ {min(op_scores)} , "
              f"{max(op_scores)} ] ")

    # Merge all ops to create a final list
    selected_ops = [set(x) for x in selected_ops]
    selected_ops_set = set()
    for x in selected_ops:
        selected_ops_set = selected_ops_set.union(x)

    print(f"#Selected ops set {len(selected_ops_set)}")
    for x in selected_ops_set:
        print(x)
    return list(selected_ops_set)


def remove_less_performant_kernels(gemm_ops: np.array, ops: List[str]):
    """
    Removes kernel that are relatively less performant from gemm_ops.
    """
    n_ops = gemm_ops.shape[1]
    assert n_ops == len(ops)

    gemm_ops_predicated = gemm_ops < 0.75
    ops_predicated = np.all(gemm_ops_predicated, axis=0)

    bad_cols = list(range(n_ops))
    bad_cols = list(filter(lambda x: ops_predicated[x], bad_cols))
    bad_cols = sorted(list(set(bad_cols)), reverse=True)
    for bc in bad_cols:
        ops.pop(bc)
        gemm_ops = np.delete(gemm_ops, bc, 1)

    return gemm_ops, ops


def plot(gemm_ops: np.array,
         gemm_problems: List[str],
         ops: List[str],
         save_filename: str,
         prune_ops: bool = False):
    if prune_ops:
        gemm_ops, ops = remove_less_performant_kernels(gemm_ops, ops)
        print(f"Pruned gemm_ops {gemm_ops.shape}")

    plot_heatmap(gemm_ops, gemm_problems, ops, save_filename)


def select_kernels_and_plot(gemm_problems: List[str], ops: List[str],
                            data: List[str], save_filename: str):

    autogen_ops = list(filter(lambda x: x.startswith('autogen'), ops))
    cutlass_ops = list(filter(lambda x: x.startswith('cutlass'), ops))
    pytorch_ops = list(filter(lambda x: x.startswith('pytorch'), ops))
    assert len(autogen_ops) + len(cutlass_ops) + len(pytorch_ops) == len(ops)

    print("Selecting the autogen kernels ..")
    # select the best autogen kernels
    gemm_autogenops = make_heatmap_data(gemm_problems, autogen_ops, data)
    selected_autogen_ops = select_kernels(gemm_autogenops,
                                          gemm_problems,
                                          autogen_ops,
                                          min_gemm_efficiency=0.99)

    # prepare plot data
    selected_ops = selected_autogen_ops + cutlass_ops + pytorch_ops
    gemm_ops = make_heatmap_data(gemm_problems, selected_ops, data)
    print("Plotting autogen kernels ...")
    plot(gemm_ops, gemm_problems, selected_ops, save_filename)


def from_measurements(args):
    pkl_files: List[str] = args.input_pkl
    save_file: Optional[str] = args.save_file
    data: List[Data] = []

    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            pkl_data = pkl.load(f)
            data.extend(list(map(lambda x: measurement_to_data(x), pkl_data)))

    ops: List[str] = list(map(lambda x: x.description, data))
    ops = sorted(list(set(ops)))

    gemm_problems: List[str] = list(map(lambda x: (x.m, x.n, x.k), data))
    gemm_problems = sorted(list(set(gemm_problems)))

    print(f"#gemm_problems {len(gemm_problems)}")
    print(f"#gemm_ops {len(ops)}")

    # plot all data as heat map
    if args.plot_all_ops:
        gemm_ops: np.array = make_heatmap_data(gemm_problems, ops, data)
        out_file: str = pkl_file.replace(
            '.pkl', '_heatmap.png') if save_file is None else save_file
        plot(gemm_ops, gemm_problems, ops, save_filename=out_file)

    if args.select_kernels:
        out_file = None
        if save_file:
            out_file = Path(save_file).with_suffix("_selected.png")
        else:
            out_file = pkl_file.replace('.pkl', 'selected_heatmap.png')
        select_kernels_and_plot(gemm_problems, ops, data, out_file)


def main(args):
    from_measurements(args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='''
        Plot bench measurements pkl.
        Example invocation: 
        Plot all the ops in model bench pickle file:
            python3 heatmap.py \
              --input-pkl ./model_bench-torch.float8_e4m3fn-1730295961.pkl \
              --plot-all-ops
        Run select kernel on the input-pkl and plot the selected ops.
            python3 heatmap.py \
               --input-pkl ./model_bench-torch.float8_e4m3fn-1730295961.pkl \
               --select-kernels
        ''')

    parser.add_argument("--input-pkl",
                        "-i",
                        nargs="+",
                        required=True,
                        type=str,
                        help=("This is typically the pickle file output by "
                              "w8a8_benchmarks.py 's model_bench command"))
    parser.add_argument("--save-file", "-o", required=False, type=str)
    parser.add_argument("--select-kernels",
                        action='store_true',
                        help="Run kernel selection and plot the heatmap "
                        "for the selected kernels")
    parser.add_argument("--plot-all-ops",
                        action='store_true',
                        help="plot heatmap for all ops")
    args = parser.parse_args()

    if not args.plot_all_ops and not args.select_kernels:
        print("Argument error : Please provide at least one argument among"
              "[--plot-all-ops, --select-kernels]")

    main(args)
