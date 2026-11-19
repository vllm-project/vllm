# FlashTopK (vendored)

[FlashTopK](https://github.com/Tencent-Hunyuan/FlashTopK) at commit
`090902abad86e5bb3324519124a680122fff91a1` (`main`), Apache-2.0 (`LICENSE`, `NOTICE`). It backs
`VLLM_FUSED_MQA_TOPK_BACKEND=flashtopk`
(`vllm/model_executor/kernels/attention/dsa/flashtopk.py`).

The `flashtopk/` package is copied unchanged except for two changes:

- Its custom op is registered as `torch.ops.vllm_flashtopk.select` (`ops.py`, `__init__.py`),
  so this copy can be imported alongside a separately installed `flashtopk`.
- A call's scratch can be the caller's (`topk.py`, `plan.py`, `recovery.py`,
  `kernels/launch.py`): `Selector.scratch_bytes(num_q, seq_len_kv)` sizes it,
  `Selector.plan(..., scratch=False)` leaves it out of the plan, and
  `Selector.run(..., scratch=buffer)` carves the candidates, a slice's scores,
  recovery's staged slices, the pruning plan and its mask out of `buffer`
  instead of the plan's own (formerly the plan held the candidates and scores,
  and each pruning call allocated its plan and mask). vLLM passes a
  workspace-manager buffer. A selector planned without scratch needs a buffer
  on every call: a `run()` that outgrows the plan does not allocate one, it
  raises `ValueError` naming `scratch_bytes()`.

Runtime dependencies are vLLM's own pins: torch, triton,
nvidia-cutlass-dsl, apache-tvm-ffi. The kernels are CuTe DSL and Triton,
compiled at runtime (vLLM compiles them in `kernel_warmup`).

To update: copy `flashtopk/`, `LICENSE` and `NOTICE` from the new commit,
reapply the changes above, and update the commit here.
