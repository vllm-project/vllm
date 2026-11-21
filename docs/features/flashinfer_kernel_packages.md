# FlashInfer Kernel Packages

FlashInfer 0.7.5+ serves kernels from the
[sealed packages](https://github.com/flashinfer-ai/flashinfer/blob/main/docs/trace_apply.rst#sealed-kernel-packages)
named by `FLASHINFER_KERNEL_PACKAGES`. Off by default.

```bash
export FLASHINFER_KERNEL_PACKAGES=<digest>[,<digest>...]
export FLASHINFER_KERNEL_PACKAGE_ROOT=/opt/flashinfer/kernel_packages  # default
vllm serve <model>
```

A package whose Definition is tagged `fi_api:<module>.<function>` can also serve a vLLM
function marked `@kernel_site`, such as
`vllm.model_executor.layers.mamba.ops.ssd_chunk_scan._chunk_scan_fwd`. Calls whose
arguments match the Definition run the package's Solution with the Definition inputs as
keyword arguments; other calls run the function itself. The log and
`flashinfer.trace_apply.stats()` report hits and fallbacks.
`FLASHINFER_KERNEL_PLUGINS_DISABLE=1` turns packages off.

A package with a setup step keeps state derived from the weights, so vLLM refuses it
alongside sleep mode, EPLB, weight transfer, CPU offload or `reload_weights`.
