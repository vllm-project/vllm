# FlashTopK (vendored)

[FlashTopK](https://github.com/Tencent-Hunyuan/FlashTopK) at commit
`090902abad86e5bb3324519124a680122fff91a1` (`main`), Apache-2.0 (`LICENSE`, `NOTICE`). It backs
`VLLM_FUSED_MQA_TOPK_BACKEND=flashtopk`
(`vllm/model_executor/kernels/attention/dsa/fused_mqa_topk.py`).

The `flashtopk/` package is copied unchanged except that its custom op
is registered as `torch.ops.vllm_flashtopk.select` (`ops.py`, `__init__.py`),
so this copy can be imported alongside a separately installed `flashtopk`.

Runtime dependencies are vLLM's own pins: torch, triton,
nvidia-cutlass-dsl, apache-tvm-ffi. The kernels are CuTe DSL and Triton,
compiled at runtime (vLLM compiles them in `kernel_warmup`).

To update: copy `flashtopk/`, `LICENSE` and `NOTICE` from the new commit,
reapply the rename above, and update the commit here.
