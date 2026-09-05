# Kimi-K2.5 Fused Kernels

This package hosts custom-tuned CuTeDSL kernels that can be used for
Kimi-K2.5 as well as other similar models such as Deepseek V3. Specific
requirements include:

* Q lora rank is 3 times KV lora rank
* Input hidden states is bfloat16; paged KV cache is e4m3 fp8.

The kernels are meant to be used for the low-latency decode path. They
save latency by fusing the attention kernels, both horizontally
and vertically, in a maximal way without pulling GeMM kernels into the fusion.

## Compilation & caching convention

Each kernel module under [`ops/`](./ops) defines a kernel class with a
`@cute.kernel` device function, a `@cute.jit` launcher, and a cached
`compile(...)` method. Compilation follows the convention in
[`vllm/models/deepseek_v4`](vllm/models/deepseek_v4):

- A `functools.cache`-decorated `compile(...)` method builds **fake tensors**
  inline with `cute.runtime.make_fake_tensor`, symbolic shapes/strides
  (`cute.sym_int` / `cute.sym_int64`), and a fake stream, then calls
  `cute.compile(..., options="--enable-tvm-ffi")`. The cache key is the set of
  compile-time (constexpr) parameters only, so a kernel compiles once per
  configuration and is reused across token counts.
- The compiled executor is invoked **directly with torch tensors** and sources
  its launch stream from the TVM-FFI environment, so the public kernel
  wrappers do not build CuTe tensors or pass a stream at call time.

## Testing

The kernels are validated against PyTorch / vLLM-op reference implementations in
[`tests/`](./tests):

```bash
pytest vllm/models/kimi_k2_5/nvidia/tests
```
