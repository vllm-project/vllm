# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from datetime import datetime
from itertools import product

import regex as re
import torch

from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p,
    apply_top_k_top_p_triton,
)


def g_str(s):
    return "\033[32m" + s + "\033[0m"


def r_str(s):
    return "\033[31m" + s + "\033[0m"


def y_str(s):
    return "\033[33m" + s + "\033[0m"


def b_str(s):
    return "\033[34m" + s + "\033[0m"


def print_to_log(s, log_file):
    print(s)
    # Remove the color codes
    s = re.sub(r"\033\[[0-9;]*m", "", s)
    with open(log_file, "a") as f:
        f.write(s + "\n")


def test_accuracy(logits, k, p, func_list, log_file):
    input_logit_list = [logits.clone().detach() for i in range(len(func_list))]
    original_logits = func_list[0](input_logit_list[0], k, p)
    output_correct_list = []
    for i in range(1, len(func_list)):
        output_logits = func_list[i](input_logit_list[i], k, p)

        torch.cuda.synchronize()
        is_correct = True
        # original_logits_bin = original_logits.view(torch.int32)
        # output_logits_bin = output_logits.view(torch.int32)
        # is_correct = torch.all(original_logits_bin == output_logits_bin)
        # is_correct = is_correct and torch.allclose(
        #     output_logits, original_logits
        # )
        output_logits_sorted = torch.sort(output_logits, descending=True).values
        original_logits_sorted = torch.sort(original_logits, descending=True).values
        is_correct = is_correct and torch.allclose(
            output_logits_sorted, original_logits_sorted
        )
        output_correct_list.append(is_correct)
        func_name = func_list[i].__name__

        if not is_correct:
            print_to_log(
                r_str("Error: logits are not close on " + f"{func_name}"),
                log_file,
            )

            # Check for NaN values first
            output_has_nan = torch.isnan(output_logits).any().item()
            original_has_nan = torch.isnan(original_logits).any().item()
            output_nan_count = torch.isnan(output_logits).sum().item()
            original_nan_count = torch.isnan(original_logits).sum().item()

            print_to_log(
                "NaN check:\n"
                + f"  output_logits has NaN: {output_has_nan} (count: {output_nan_count})\n"
                + f"  original_logits has NaN: {original_has_nan} (count: {original_nan_count})\n"
                + "  Note: torch.allclose returns False if either tensor contains NaN (unless equal_nan=True)",
                log_file,
            )

            if output_has_nan or original_has_nan:
                # Show where NaN values are
                if output_has_nan:
                    output_nan_positions = torch.where(torch.isnan(output_logits))
                    print_to_log(
                        f"  output_logits NaN positions (first 10): "
                        f"{list(zip(output_nan_positions[0][:10].tolist(), output_nan_positions[1][:10].tolist()))}",
                        log_file,
                    )
                if original_has_nan:
                    original_nan_positions = torch.where(torch.isnan(original_logits))
                    print_to_log(
                        f"  original_logits NaN positions (first 10): "
                        f"{list(zip(original_nan_positions[0][:10].tolist(), original_nan_positions[1][:10].tolist()))}",
                        log_file,
                    )

            error = torch.abs(output_logits - original_logits)
            # Handle NaN in error computation
            error_has_nan = torch.isnan(error).any().item()
            if error_has_nan:
                error_nan_count = torch.isnan(error).sum().item()
                print_to_log(
                    f"  error tensor has NaN: True (count: {error_nan_count})",
                    log_file,
                )
                # Use masked operations for NaN handling (compatible with all PyTorch versions)
                valid_error = error[~torch.isnan(error)]
                if valid_error.numel() > 0:
                    max_error = torch.max(valid_error).item()
                    mean_error = torch.mean(valid_error).item()
                else:
                    max_error = float("nan")
                    mean_error = float("nan")
            else:
                max_error = torch.max(error).item()
                mean_error = torch.mean(error).item()

            # Use the same tolerance as torch.allclose (rtol=1e-05, atol=1e-08)
            atol = 1e-08
            rtol = 1e-05
            # torch.allclose checks: |input - other| <= atol + rtol * |other|
            # Exclude NaN from tolerance check
            valid_mask = ~torch.isnan(original_logits) & ~torch.isnan(output_logits)
            tolerance = atol + rtol * torch.abs(original_logits)
            error_mask = (error > tolerance) & valid_mask

            print_to_log(
                f"Max absolute error: {max_error:.2e}\n"
                + f"Mean absolute error: {mean_error:.2e}\n"
                + f"torch.allclose tolerance: rtol={rtol}, atol={atol}",
                log_file,
            )

            error_rows = torch.where(error_mask)[0]
            error_rows = torch.unique(error_rows)
            num_error_rows = error_rows.shape[0]
            error_cols = torch.where(error_mask)[1]
            error_cols = torch.unique(error_cols)
            num_error_cols = error_cols.shape[0]
            print_to_log(
                f"num_error_rows: {num_error_rows} - {error_rows}\n"
                + f"num_error_cols: {num_error_cols} - {error_cols}",
                log_file,
            )

            if num_error_rows > 0:
                row_to_show = 5 if num_error_rows > 5 else num_error_rows
                logits_to_show = torch.sort(
                    output_logits[error_rows], descending=True
                ).values

                logits_to_show = logits_to_show[:row_to_show, :50]
                print_to_log(f"logits: {logits_to_show}", log_file)
                original_logits_to_show = torch.sort(
                    original_logits[error_rows], descending=True
                ).values
                original_logits_to_show = original_logits_to_show[:row_to_show, :50]
                print_to_log(f"original_logits: {original_logits_to_show}", log_file)
                error_to_show = error[error_rows][:row_to_show, :50]
                print_to_log(f"error (abs diff): {error_to_show}", log_file)
            else:
                # If no errors found with the mask, show the largest errors anyway
                print_to_log(
                    "No errors found with tolerance mask, showing top errors:", log_file
                )
            # Handle NaN in topk - replace NaN with -inf so they're not selected
            error_for_topk = error.clone()
            error_for_topk[torch.isnan(error_for_topk)] = float("-inf")
            top_errors, top_indices = torch.topk(
                error_for_topk.flatten(), min(20, error.numel())
            )
            print_to_log(f"Top 20 absolute errors: {top_errors}", log_file)
            for idx, err_val in zip(top_indices, top_errors):
                row_idx = idx.item() // error.shape[1]
                col_idx = idx.item() % error.shape[1]
                output_val = output_logits[row_idx, col_idx].item()
                original_val = original_logits[row_idx, col_idx].item()
                err_val_item = err_val.item()
                # Check if values are NaN
                output_str = (
                    f"{output_val:.10f}"
                    if not torch.isnan(output_logits[row_idx, col_idx])
                    else "NaN"
                )
                original_str = (
                    f"{original_val:.10f}"
                    if not torch.isnan(original_logits[row_idx, col_idx])
                    else "NaN"
                )
                error_str = (
                    f"{err_val_item:.2e}"
                    if not torch.isnan(error[row_idx, col_idx])
                    else "NaN"
                )
                print_to_log(
                    f"  Position [{row_idx}, {col_idx}]: "
                    f"output={output_str}, "
                    f"original={original_str}, "
                    f"error={error_str}",
                    log_file,
                )
            # raise ValueError("Logits are not close")
    return output_correct_list


def test_time(logits, k, p, test_func, num_runs=30, num_warmup=5):
    # We must clone the logits for each run to avoid modifying the original
    warmup_tensor = [logits.clone().detach() for _ in range(num_warmup)]
    for _ in range(num_warmup):
        test_func(warmup_tensor[_], k, p)
    torch.cuda.synchronize()

    input_logits = [logits.clone().detach() for _ in range(num_runs)]

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(num_runs):
        input_logits[_] = test_func(input_logits[_], k, p)
    end.record()
    torch.cuda.synchronize()
    time_taken = start.elapsed_time(end) / num_runs

    return time_taken


if __name__ == "__main__":
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    vocab_size_list = [32768, 65536, 102400, 128256]
    p_list = [None, "RAND", 0.1, 0.4, 0.7, 0.9, 0.95, 0.99]
    k_list = [None, "RAND", 5, 20, 50, 200, 500, 3000]
    func_list = [apply_top_k_top_p, apply_top_k_top_p_triton]

    log_file = f"triton_topk_topp_test_{date_str}.log"
    csv_file = f"triton_topk_topp_test_{date_str}.csv"

    print_to_log(y_str("Testing TopKTopPSampler with Triton"), log_file)
    print_to_log(y_str("batch_size_list:") + f"{batch_size_list}", log_file)
    print_to_log(y_str("vocab_size_list:") + f"{vocab_size_list}", log_file)
    print_to_log(y_str("p_list:") + f"{p_list}", log_file)
    print_to_log(y_str("k_list:") + f"{k_list}", log_file)

    print_to_log(y_str("log_file:") + f"{log_file}", log_file)
    print_to_log(y_str("csv_file:") + f"{csv_file}", log_file)

    with open(csv_file, "w") as f:
        f.write(
            "dist_generator,batch_size,vocab_size,p,k,triton_correct,"
            "torch_time_taken,triton_time_taken,triton_speedup\n"
        )

    for batch_size, vocab_size, p, k in product(
        batch_size_list, vocab_size_list, p_list, k_list
    ):
        if p is None and k is None:
            continue

        logits_randn = torch.randn(batch_size, vocab_size, device="cuda") * 10
        top_5_logits = torch.topk(logits_randn, 5, dim=-1).values

        logits_list = [("RANDN", logits_randn)]

        if p == "RAND":
            p_tensor = torch.rand((batch_size,), device="cuda") * 0.98 + 0.01
        elif p is not None:
            p_tensor = torch.full((batch_size,), p, device="cuda")
        else:
            p_tensor = None

        if k == "RAND":
            k_tensor = torch.randint(
                1, int(vocab_size / 4) - 1, (batch_size,), device="cuda"
            )
        elif k is not None:
            k_tensor = torch.full((batch_size,), k, device="cuda")
        else:
            k_tensor = None

        for dist_generator, logits in logits_list:
            print_to_log(y_str("--------------------------------"), log_file)
            print_to_log(
                g_str("Testing ")
                + f"{dist_generator}"
                + y_str(" with batch_size: ")
                + f"{batch_size}"
                + y_str(" vocab_size: ")
                + f"{vocab_size}"
                + y_str(" p: ")
                + f"{p}"
                + y_str(" k: ")
                + f"{k}",
                log_file,
            )
            correct_list = test_accuracy(
                logits, k_tensor, p_tensor, func_list, log_file
            )
            time_list = []
            for func in func_list:
                time_taken = test_time(logits, k_tensor, p_tensor, test_func=func)
                time_list.append(time_taken)
            print_to_log(b_str("torch_time_taken: ") + f"{time_list[0]}", log_file)
            print_to_log(b_str("triton_time_taken: ") + f"{time_list[1]}", log_file)
            print_to_log(
                g_str("test Speedup over Torch: ")
                + f"{time_list[0] / time_list[1]:.8f}x",
                log_file,
            )
            with open(csv_file, "a") as f:
                p_str = "NONE" if p is None else str(p)
                k_str = "NONE" if k is None else str(k)
                f.write(
                    f"{dist_generator},{batch_size},{vocab_size},{p_str},{k_str},"
                    f"{correct_list[0]},{time_list[0]},{time_list[1]},"
                    f"{time_list[0] / time_list[1]:.8f}\n"
                )
            print_to_log(y_str("--------------------------------\n"), log_file)
