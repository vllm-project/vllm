# NVFP4 MoE with FlashInfer Prims-TS

`--moe-backend flashinfer_prims_ts` selects FlashInfer's Prims-TS (CuTe DSL)
expert GEMMs for NVFP4 Mixture-of-Experts layers. Routing and finalization use
TRTLLM-Gen, as does the `flashinfer_trtllm` backend. This option changes the MoE
kernels, not the attention backend or KV-cache dtype.

The backend is opt-in: `--moe-backend auto` does not select it. To enable it on
a supported NVFP4 checkpoint:

```bash
vllm serve <nvfp4-model> \
    --tensor-parallel-size 4 \
    --moe-backend flashinfer_prims_ts
```

Choose a parallel size that fits the checkpoint on the available GPUs. Expert
parallelism uses the usual `--enable-expert-parallel` and `--all2all-backend`
settings; see [Expert Parallel Deployment](../../serving/expert_parallel_deployment.md).
Unlike the `flashinfer_moe_ep_*` megakernels, Prims-TS is a modular expert
backend: vLLM still performs dispatch and combine through the selected
communication backend.

## Requirements and limitations

- CUDA GPUs with compute capability SM100, SM103 or SM107 (Blackwell or Vera
  Rubin). Other architectures are rejected when this backend is explicitly
  requested.
- NVFP4 expert weights and NVFP4 activations; BF16, FP8 and MXFP4 expert weights
  are not supported by this backend. The NVFP4 kernel supports both
  checkpoint-calibrated activation scaling and per-token activation scaling
  when the model's quantization configuration requests it.
- A FlashInfer installation providing
  `flashinfer.fused_moe.prims_ts_fp4_block_scale_routed_moe`, available in the
  pinned FlashInfer release. It uses the same shuffled expert-weight layout,
  activation and modular-dispatch constraints as the TRTLLM-Gen NVFP4 backend.
  vLLM pads the expert intermediate dimension to a multiple of 128.
- Kernel selection still checks the model's activation and deployment
  configuration. An explicit backend request fails at startup if no compatible
  implementation exists; it does not silently select a different MoE backend.

## Startup compilation and autotuning

Prims-TS chooses expert-GEMM configurations using the token count. Without
precompilation, a token count not visited during startup can trigger JIT
compilation during inference.

With `--kernel-config.enable_jit_warmup true` (the default), vLLM precompiles
the untuned FC1/FC2 configurations for each distinct layer layout before
serving. The log reports `Precompiled ... Prims-TS MoE expert configurations`.
If FlashInfer exposes the MoE operation but lacks its precompile helper, vLLM
warns and leaves compilation to inference.

JIT warmup is separate from FlashInfer autotuning. For example, this keeps the
untuned-kernel precompilation while disabling the autotuning sweep:

```bash
vllm serve <nvfp4-model> \
    --tensor-parallel-size 4 \
    --moe-backend flashinfer_prims_ts \
    --kernel-config.enable_jit_warmup true \
    --kernel-config.enable_flashinfer_autotune false
```

Enabling `--kernel-config.enable_flashinfer_autotune true` benchmarks candidate
implementations during warmup and can increase startup time. Precompilation of
the untuned configurations does not replace autotuning or imply that every
tuned tactic has been compiled.

## Comparison with SM107 MegaMoE

`flashinfer_prims_ts` is distinct from `flashinfer_moe_ep_cutedsl`. The latter
dispatches, computes and combines experts inside FlashInfer's megakernel and
requires expert parallelism. For NVFP4 experts on SM107, that megakernel uses
`knobs="cache"`: an offline knob-cache winner when present, otherwise
FlashInfer's built-in heuristic. MXFP4 experts retain their default knob
selection. These knobs do not configure the Prims-TS modular backend.

For the megakernel's NVSHMEM dependencies, architecture setting and tuning-cache
instructions, see [FlashInfer MegaMoE on SM107](../../serving/expert_parallel_deployment.md#flashinfer-megamoe-on-sm107).
