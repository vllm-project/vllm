import subprocess
import itertools


# batch_sizes = [128, 256]
batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
seq_lens = [8192, 16384, 32768]
# seq_lens = [32768]
in_features = [8192]
num_heads_kv = 8
nproc = [1, 2, 4, 8]


# Attention experiments
log_file = "HAMP/H100_tensorparallel_attention.csv"
# for batch, seq, inf, nproc in itertools.product(batch_sizes, seq_lens, in_features, nproc):
#     print(f"Running: batch_size={batch} seq_len={seq} in_features={inf} nproc={nproc}")
#     cmd = [
#         "torchrun",
#         f"--nproc_per_node={nproc}",
#         "-m", "HAMP.tensorparallel_attention",
#         "--batch_size", str(batch),
#         "--seq_len", str(seq),
#         "--in_features", str(inf),
#         "--log_file", log_file
#     ]
#     try:
#         subprocess.run(cmd, check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Experiment failed for batch_size={batch}, seq_len={seq}, in_features={inf}, nproc={nproc}")
#         # Optionally, log the error to a file:
#         with open("HAMP/error_experiments.log", "a") as f:
#             f.write(f"Failed: batch_size={batch}, seq_len={seq}, in_features={inf}, nproc={nproc}\n")


# MLP experiments
nproc_list = [1, 2, 4, 8]  # Rename your list
num_iterations = 100
log_file = "HAMP/H100_tensorparallel_mlp.csv"
for batch, inf, nproc in itertools.product(batch_sizes, in_features, nproc_list):
    print(f"Running: batch_size={batch} in_features={inf} nproc={nproc}")
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "-m", "HAMP.tensorparallel_mlp",
        "--batch_size", str(batch),
        "--in_features", str(inf),
        "--num_iterations", str(num_iterations),
        "--log_file", log_file
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed for batch_size={batch}, in_features={inf}, nproc={nproc}")
        with open("HAMP/error_mlp_experiments.log", "a") as f:
            f.write(f"Failed: batch_size={batch}, in_features={inf}, nproc={nproc}\n")