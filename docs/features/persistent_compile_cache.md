# Persistent compilation cache

The persistent compilation cache is disabled by default. Enable it with
`--compile-cache`, or explicitly disable it with `--no-compile-cache`. The two
options are mutually exclusive.

When enabled, vLLM restores and publishes a per-rank bundle under
`s3://eric-alcaide-dev/vllm_cache/`. The bundle contains:

- vLLM's `torch_compile_cache` artifacts;
- FlashInfer JIT artifacts under `~/.cache/flashinfer`, including architecture-
  specific generated operators such as `fused_moe_trtllm_sm100`;
- FlashInfer autotuner results, including TensorRT-LLM block-scaled FP4 MoE and
  dense `fp4_gemm` profiles.

The bundle is published once after torch compilation, again after kernel warmup,
autotuning, and CUDA graph capture, and on normal process exit. This timing is
important: runtime evidence from GLM-5.2 NVFP4 on eight B200 GPUs with vLLM
0.25.1 showed about 10 minutes for the one-time
`fused_moe_trtllm_sm100` nvcc/cicc/ptxas build and up to 20 minutes for 12 dense
FP4 GEMM profiles.

The SHA-256 key includes model identity and resolved revision, TP/PP/DP/EP/world
topology, model and KV-cache precision, GPU architecture, CUDA/PyTorch/vLLM/
FlashInfer/Triton/cuDNN versions, compilation and CUDA-graph configuration,
attention and MoE backends, and scheduler dimensions. In particular,
`max_num_seqs` and `max_num_batched_tokens` are explicit key fields because they
change profiling, autotuning, and graph capture.

CUDA graph executable objects are not serialized. They contain process-local
device pointers and CUDA runtime state. vLLM restores the compiled kernels and
autotuner decisions needed by capture, keys the exact capture configuration,
then safely captures fresh graph executables in the new process.

AWS credentials are obtained from the standard credential provider chain and
are never stored in cache objects, source code, or logs. Downloads are unpacked
only after rejecting path traversal, links, and special files.
