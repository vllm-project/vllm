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
function marked `@kernel_site`:
`vllm.model_executor.layers.mamba.ops.ssd_chunk_scan._chunk_scan_fwd`, and the native
`rms_norm` and `fused_add_rms_norm` in `vllm.ir.ops.layernorm`. This needs a FlashInfer
that provides `installed_packages()`. A call matches when every Definition input is
passed or left at its default (one marked `"optional": true` may be `None`), shapes and
dtypes fit, and no other argument is set. It then runs the package's Solution with the
Definition inputs as keyword arguments; other calls run the function itself. The log
and `flashinfer.trace_apply.stats()` report hits and fallbacks.
`FLASHINFER_KERNEL_PLUGINS_DISABLE=1` turns packages off.

Inside a compiled graph the choice is made once, when the graph is compiled. The log
reports it as `compiled_hit` or `compiled_fallback_<reason>`, `stats()` does not count
it, and a Definition that pins a size the graph leaves dynamic does not match. The
norms are compiled this way by default, so their Solution must be PyTorch code that can
be traced. Without compilation, vLLM on CUDA serves its `vllm_c` norm kernels, which
packages do not replace.

A package with a setup step keeps state derived from the weights, so vLLM refuses it
alongside sleep mode, EPLB, weight transfer, CPU offload or `reload_weights`.
