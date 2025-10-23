# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from datetime import datetime
from itertools import product

import regex as re
import torch

from vllm.v1.sample.ops.topk_topp_sampler import (apply_top_k_top_p,
                                                  apply_top_k_top_p_triton,
                                                  apply_top_k_top_p_test,
                                            
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


def test_accuracy(logits, k, p, func_list):
    input_logit_list = [logits.clone().detach() for i in range(len(func_list))]
    original_logits = func_list[0](input_logit_list[0], k, p)
    output_correct_list = []
    for i in range(1, len(func_list)):
        output_logits = func_list[i](input_logit_list[i], k, p)

        torch.cuda.synchronize()
        is_correct = torch.allclose(original_logits, output_logits)
        output_correct_list.append(is_correct)
        func_name = func_list[i].__name__

        if not is_correct:
            print_to_log(r_str(f"Error: logits are not close on {i} - " + f"{func_name}"), log_file)
            error_mask = torch.abs(output_logits - original_logits) > 1e-5
            error_rows = torch.where(error_mask)[0]
            error_rows = torch.unique(error_rows)
            num_error_rows = error_rows.shape[0]
            error_cols = torch.where(error_mask)[1]
            error_cols = torch.unique(error_cols)
            num_error_cols = error_cols.shape[0]
            print_to_log(f"num_error_rows: {num_error_rows} - {error_rows}",
                        log_file)
            print_to_log(f"num_error_cols: {num_error_cols}", log_file)
            row_to_show = 5 if num_error_rows > 5 else num_error_rows
            logits_to_show = torch.sort(output_logits[error_rows],
                                        descending=True).values
            logits_to_show = logits_to_show[:row_to_show, :20]
            print_to_log(f"logits: {logits_to_show}", log_file)
            original_logits_to_show = \
                torch.sort(original_logits[error_rows], descending=True).values
            original_logits_to_show = original_logits_to_show[:row_to_show, :20]
            print_to_log(f"original_logits: {original_logits_to_show}", log_file)

    return output_correct_list


def test_time(logits, k, p, test_func, num_runs=30, num_warmup=5):
    # We must clone the logits for each run to avoid modifying the original
    warmup_tensor = logits.clone().detach()
    for _ in range(num_warmup):
        test_func(warmup_tensor, k, p)
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

    batch_size_list = [64, 128, 1024]
    vocab_size_list = [4096, 16384]
    p_list = [None, "RAND", 0.4, 0.7, 0.9, 0.95, 0.99]
    k_list = [None, "RAND", 5, 10, 50, 100, 200, 300, 3000]
    func_list = [apply_top_k_top_p, apply_top_k_top_p_triton, apply_top_k_top_p_triton]

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
        f.write("dist_generator,batch_size,vocab_size,p,k,triton_correct,test_correct"
                "torch_time_taken,triton_time_taken,test_time_taken,triton_speedup,test_speedup\n")

    for batch_size, vocab_size, p, k in product(batch_size_list,
                                                vocab_size_list, p_list,
                                                k_list):
        if p is None and k is None:
            continue

        logits_randn = torch.randn(batch_size, vocab_size, device="cuda") * 10
        logits_list = [("RANDN", logits_randn)]

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
            correct_list = test_accuracy(logits, k_tensor, p_tensor, func_list)
            for i in range(len(func_list) - 1):
                is_correct = correct_list[i]
                if not is_correct:
                    print_to_log(
                        f"Error: logits are not close for function {func_list[i + 1].__name__},"
                        f" batch_size: {batch_size},"
                        f" vocab_size: {vocab_size}, dist_generator: "
                        f"{dist_generator}, p: {p}, k: {k}", log_file)
            time_list = []
            for func in func_list:
                time_taken = test_time(logits, k_tensor, p_tensor, test_func=func)
                time_list.append(time_taken)
            print_to_log(
                b_str("torch_time_taken: ") + f"{time_list[0]}", log_file)
            print_to_log(
                b_str("triton_time_taken: ") + f"{time_list[1]}",
                log_file)
            print_to_log(
                b_str("test_time_taken: ") + f"{time_list[2]}", log_file)
            print_to_log(
                g_str("Triton Speedup over Torch: ") +
                f"{time_list[0] / time_list[1]:.8f}x", log_file)
            print_to_log(
                y_str("Test Speedup over Torch: ") +
                f"{time_list[0] / time_list[2]:.8f}x", log_file)
            with open(csv_file, "a") as f:
                f.write(f"{dist_generator},{batch_size},{vocab_size},{p},{k},"
                        f"{correct_list[0]},{correct_list[1]},{time_list[0]},{time_list[1]},{time_list[2]},"
                        f"{time_list[0] / time_list[1]:.8f}, {time_list[0] / time_list[2]:.8f}\n")
            print_to_log(y_str("--------------------------------\n"), log_file)

"""# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datetime import datetime
from itertools import product
import regex as re
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Default device:", torch.cuda.current_device())

# --- MODIFIED IMPORTS ---
# We need all the component kernels to time them individually
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p, 
    apply_top_k_with_pivot_filter,  # Used for accuracy check
    apply_top_k_only,               # This is the baseline AND our Kernel 2
    top_k_pivot_and_sort,           # Kernel 1
    scatter_topk_kernel             # Kernel 3
)
print("All kernels imported successfully")

x = torch.randn(2, 5, device="cuda")
y = apply_top_k_only(x, k=torch.tensor([2,2], device="cuda"))
print("apply_top_k_only ran successfully, output:", y)


def g_str(s): return "\033[32m" + s + "\033[0m"
def r_str(s): return "\033[31m" + s + "\033[0m"
def y_str(s): return "\033[33m" + s + "\033[0m"
def b_str(s): return "\033[34m" + s + "\033[0m"

def print_to_log(s, log_file):
    print(s)
    s = re.sub(r"\033[[0-9;]*m", "", s)
    with open(log_file, "a") as f:
        f.write(s + "\n")

# --- UNCHANGED ---
# test_accuracy still runs the *full* pipeline to check for correctness
def test_accuracy(logits, k, log_file):
    input_logits_torch = logits.clone().detach()
    input_logits_triton = logits.clone().detach()

    original_logits = apply_top_k_only(input_logits_torch, k)
    triton_pivot_logits = apply_top_k_with_pivot_filter(input_logits_triton, k)

    torch.cuda.synchronize()
    is_correct = torch.allclose(original_logits, triton_pivot_logits)

    if not is_correct:
        print_to_log(r_str("Error: logits are not close"), log_file)

    return is_correct

# --- REWRITTEN test_time FUNCTION ---
def test_time(logits, k, num_runs=30, num_warmup=5):
    
    batch_size, vocab_size = logits.shape
    
    # --- Warmup ---
    for _ in range(num_warmup):
        warmup_tensor_torch = logits.clone().detach()
        apply_top_k_only(warmup_tensor_torch, k)
        
        warmup_tensor_triton = logits.clone().detach()
        apply_top_k_with_pivot_filter(warmup_tensor_triton, k)
    torch.cuda.synchronize()

    # --- 1. Baseline `apply_top_k_only` timing ---
    start_torch = torch.cuda.Event(enable_timing=True)
    end_torch = torch.cuda.Event(enable_timing=True)
    
    start_torch.record()
    for i in range(num_runs):
        input_tensor = logits.clone().detach()
        apply_top_k_only(input_tensor, k)
    end_torch.record()
    torch.cuda.synchronize()
    apply_top_k_time = start_torch.elapsed_time(end_torch) / num_runs

    # --- 2. Triton Kernel 2 (Sort) Timing ---
    
    # Events for Kernel 2
    start_k2 = torch.cuda.Event(enable_timing=True)
    end_k2 = torch.cuda.Event(enable_timing=True)
    
    # Kernel 2 time accumulator
    triton_k2_time_acc = 0.0
    
    for i in range(num_runs):
                if (k == vocab_size).all():
            continue

        # 1. Setup
        input_tensor = logits.clone().detach()
        probs = torch.full_like(input_tensor, -float('inf'))
        l =  torch.empty((batch_size,), device=input_tensor.device, dtype=torch.int32)
        idx_tensor = torch.full_like(input_tensor, -1, dtype=torch.int)

        BLOCK_SIZE = 1024
        SIGMA = 2.0
        grid_pivot = (batch_size,)

        # 2. Run Kernel 1 (Pivot) - *No timer*
        top_k_pivot_and_sort[grid_pivot](
            input_tensor, probs, l, idx_tensor, k, batch_size,
            SIGMA=SIGMA, VOCAB_SIZE=vocab_size, BLOCK_SIZE=BLOCK_SIZE,
        )

        torch.cuda.synchronize() 
        max_l = torch.max(l).item()
        outliers = probs[:, :max_l] 
        outliers_idx = idx_tensor[:, :max_l]
        k_pinned = torch.minimum(k, l)

        # 4. Time Kernel 2 (Sort)  
        start_k2.record()
        apply_top_k_only(outliers, k_pinned) 
        end_k2.record()

        torch.cuda.synchronize()
        triton_k2_time_acc += start_k2.elapsed_time(end_k2)

    triton_sort_only_time = triton_k2_time_acc / num_runs
    
    return apply_top_k_time, triton_sort_only_time


def main():
    print("Starting compare.py...")
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    #batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]  # Up to 512
    #vocab_size_list = [4096, 16384, 65536, 262144, 102400]
    #k_list = [None, "RAND", 5, 10, 50, 100, 200, 300, 3000] 

    batch_size_list = [1, 2, 4, 8]
    vocab_size_list = [4096, 16384]
    k_list = [None, "RAND", 5, 10, 50, 100, 200, 300, 3000]


    log_file = f"triton_topk_topp_test_{date_str}.log"
    csv_file = f"triton_topk_topp_test_{date_str}.csv"

    print_to_log(y_str("Testing TopKTopPSampler with Triton"), log_file)
    print_to_log(y_str("batch_size_list:") + f"{batch_size_list}", log_file)
    print_to_log(y_str("vocab_size_list:") + f"{vocab_size_list}", log_file)
    print_to_log(y_str("k_list:") + f"{k_list}", log_file)
    print_to_log(y_str("log_file:") + f"{log_file}", log_file)
    print_to_log(y_str("csv_file:") + f"{csv_file}", log_file)

    # --- MODIFIED CSV HEADER ---
    with open(csv_file, "w") as f:
        f.write("dist_generator,batch_size,vocab_size,k,is_correct,"
                "apply_top_k_time,triton_sort_only_time,speedup_vs_baseline\n")

    for batch_size, vocab_size, k in product(batch_size_list,
                                             vocab_size_list,
                                             k_list):

        logits_randn = torch.randn(batch_size, vocab_size, device="cuda") * 10
        logits_list = [("RANDN", logits_randn)]

        if k == "RAND":
            k_tensor = torch.randint(1,
                                     vocab_size, (batch_size,),
                                     device="cuda")
        elif k is not None:
            k_val = min(k, vocab_size) # Ensure k is not > vocab_size
            k_tensor = torch.full((batch_size,), k_val, device="cuda")
        else:
            k_tensor = torch.full((batch_size,), vocab_size, device="cuda")

        for dist_generator, logits in logits_list:
            print_to_log(y_str("--------------------------------"), log_file)
            print_to_log(
                g_str("Testing ") + f"{dist_generator}" +
                y_str(" with batch_size: ") + f"{batch_size}" +
                y_str(" vocab_size: ") + f"{vocab_size}" +
                y_str(" k: ") + f"{k}", log_file)
            
            is_correct = test_accuracy(logits, k_tensor, log_file)
            if not is_correct:
                print_to_log(
                    r_str(f"Error: logits are not close for batch_size: {batch_size}, "
                    f"vocab_size: {vocab_size}, dist_generator: {dist_generator}, k: {k}"),
                    log_file)
            
            # --- MODIFIED TIMING CALL ---
            apply_top_k_time, triton_sort_only_time = test_time(logits, k_tensor)
            
            print_to_log(
                b_str("apply_top_k_time (Baseline): ") + f"{apply_top_k_time}", log_file)
            print_to_log(
                b_str("triton_sort_only_time (Kernel 2): ") + f"{triton_sort_only_time}",
                log_file)
            
            # --- THIS IS THE FIX ---
            # Handle the k: None case where triton_sort_only_time is 0.0
            if triton_sort_only_time > 0:
                speedup = apply_top_k_time / triton_sort_only_time
                speedup_str = f"{speedup:.8f}x"
            else:
                # 'k: None' case, speedup is not applicable (N/A)
                speedup = 0.0 
                speedup_str = "N/A (passthrough)"
            # --- END FIX ---

            print_to_log(
                g_str("Triton Sort Speedup vs. Full Baseline: ") +
                speedup_str, log_file)

            # Write to CSV
            with open(csv_file, "a") as f:
                f.write(f"{dist_generator},{batch_size},{vocab_size},{k},"
                        f"{is_correct},{apply_top_k_time},{triton_sort_only_time},"
                        f"{speedup:.8f}\n") # Still write the float for CSV
            print_to_log(y_str("--------------------------------\n"), log_file)

if __name__ == "__main__":
    main()"""