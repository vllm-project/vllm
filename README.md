# vLLM MoE CPU Offload

This branch adds Case 1 MoE CPU offload support to vLLM with one public flag:

```bash
--moe-cpu-offload
```

The original upstream vLLM README is preserved at [README.old](README.old).

![MoE CPU offload overview](moe-cpu-offload.png)

## Feature Summary

- Dense models ignore `--moe-cpu-offload` and run normally.
- MoE models keep expert weights in CPU memory as the source of truth.
- Router path and KV cache stay on GPU.
- Active expert weights are passively copied from CPU to GPU only after router
  computation identifies the routed expert set.
- Active expert GPU copies are retired/freed after expert computation.
- Large MoE LLMs can run on smaller GPUs because inactive expert weights do not
  occupy GPU memory.
- Multiple large MoE LLM endpoints can share the same GPU more easily because
  each instance only stages active experts for compute.
- If the active expert set cannot fit as one group, execution can be split into
  smaller passive waves.
- Before GPU expert transfer, free GPU memory is checked. If memory is
  insufficient, transfer waits 5 seconds and retries up to 10 times.

## Compute Model

The compute path is passive expert model loading:

1. Keep full expert weights in CPU memory.
2. Run router computation on GPU.
3. Identify active experts from router output.
4. Copy only needed active expert weights to GPU.
5. Compute the active expert bucket.
6. Free the GPU expert copy after computation.

This trades performance for memory capacity. Inference can be slower than full
GPU expert residency because active expert weights are copied from CPU memory to
GPU memory during execution.

## Example Validation

The local harness validated:

- one `gemma-4-26B-A4B-it` MoE model on one 40GB GPU,
- two `gemma-4-26B-A4B-it` MoE model instances sharing one 40GB GPU from
  separate vLLM endpoints,
- valid completion responses from all tested endpoints.

## Useful Files

- [Original README](README.old)
- [Feature work area](dev/moe)
- [Feature changelog](dev/moe/CHANGELOG.md)
- [Design notes](dev/moe/DESIGN.md)
- [Test harness](dev/moe/Makefile)
