# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from datetime import datetime
from itertools import product

import regex as re
import torch

from vllm.v1.sample.ops.topk_topp_sampler import (apply_top_k_top_p,
                                                  apply_top_k_top_p_triton)


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


def test_accuracy(logits, k, p):
    input_logits = logits.clone()
    original_logits = apply_top_k_top_p(input_logits, k, p)
    logits = apply_top_k_top_p_triton(logits, k, p)

    torch.cuda.synchronize()
    is_correct = torch.allclose(logits, original_logits)

    if not is_correct:
        print_to_log(r_str("Error: logits are not close"), log_file)
        error_rows = torch.where(logits != original_logits)[0]
        error_rows = torch.unique(error_rows)
        num_error_rows = error_rows.shape[0]
        print_to_log(f"num_error_rows: {num_error_rows} - {error_rows}",
                     log_file)
        row_to_show = 12 if num_error_rows > 12 else num_error_rows
        logits_to_show = torch.sort(logits[error_rows], descending=True).values
        logits_to_show = logits_to_show[:row_to_show, :50]
        print_to_log(f"logits: {logits_to_show}", log_file)
        original_logits_to_show = \
            torch.sort(original_logits[error_rows], descending=True).values
        original_logits_to_show = original_logits_to_show[:row_to_show, :50]
        print_to_log(f"original_logits: {original_logits_to_show}", log_file)

    return is_correct


def test_time(logits, k, p, num_runs=256):
    # We must clone the logits for each run to avoid modifying the original
    input_logits_torch = [logits.clone() for _ in range(num_runs)]
    input_logits_triton = [logits.clone() for _ in range(num_runs)]

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        input_logits_torch[_] = apply_top_k_top_p(input_logits_torch[_], k, p)
    torch.cuda.synchronize()
    torch_time_taken = (time.time() - start_time) / num_runs

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        input_logits_triton[_] = apply_top_k_top_p_triton(
            input_logits_triton[_], k, p)
    torch.cuda.synchronize()
    triton_time_taken = (time.time() - start_time) / num_runs

    return torch_time_taken, triton_time_taken


if __name__ == "__main__":
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_size_list = [2**i for i in range(0, 11)]  # 1 to 1024
    vocab_size_list = [2**i for i in range(8, 19)]  # 256 to 262144
    p_list = [None, "RAND"] + [0.1 * i for i in range(1, 10)]
    k_list = [None, "RAND"] + [i for i in range(1, 10)
                               ] + [i for i in range(20, 210, 30)]
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
        f.write("dist_generator,batch_size,vocab_size,p,k,is_correct,"
                "torch_time_taken,triton_time_taken,speedup\n")

    for batch_size, vocab_size, p, k in product(batch_size_list,
                                                vocab_size_list, p_list,
                                                k_list):
        if p == "RAND" and k == "RAND":
            continue

        logits_rand = torch.rand(batch_size, vocab_size, device="cuda")
        logits_randn = torch.randn(batch_size, vocab_size, device="cuda")
        logits_list = [("RAND", logits_rand), ("RANDN", logits_randn)]

        if p == "RAND":
            p_tensor = torch.rand((batch_size, ), device="cuda") * 0.95 + 0.05
        elif p is not None:
            p_tensor = torch.full((batch_size, ), p, device="cuda")
        else:
            p_tensor = None

        if k == "RAND":
            k_tensor = torch.randint(1,
                                     vocab_size, (batch_size, ),
                                     device="cuda")
        elif k is not None:
            k_tensor = torch.full((batch_size, ), k, device="cuda")
        else:
            k_tensor = None

        for dist_generator, logits in logits_list:
            print_to_log(y_str("--------------------------------"), log_file)
            print_to_log(
                g_str("Testing ") + f"{dist_generator}" +
                y_str(" with batch_size: ") + f"{batch_size}" +
                y_str(" vocab_size: ") + f"{vocab_size}" + y_str(" p: ") +
                f"{p}" + y_str(" k: ") + f"{k}", log_file)
            is_correct = test_accuracy(logits, k_tensor, p_tensor)
            if not is_correct:
                print_to_log(
                    f"Error: logits are not close for batch_size: {batch_size},"
                    f" vocab_size: {vocab_size}, dist_generator: "
                    f"{dist_generator}, p: {p}, k: {k}", log_file)
            torch_time_taken, triton_time_taken = test_time(
                logits, k_tensor, p_tensor)
            print_to_log(
                b_str("torch_time_taken: ") + f"{torch_time_taken}", log_file)
            print_to_log(
                b_str("triton_time_taken: ") + f"{triton_time_taken}",
                log_file)
            print_to_log(
                g_str("Triton Speedup over Torch: ") +
                f"{torch_time_taken / triton_time_taken:.8f}x", log_file)
            with open(csv_file, "a") as f:
                f.write(f"{dist_generator},{batch_size},{vocab_size},{p},{k},"
                        f"{is_correct},{torch_time_taken},{triton_time_taken},"
                        f"{torch_time_taken / triton_time_taken:.8f}\n")
            print_to_log(y_str("--------------------------------\n"), log_file)
