"""
vllm_xpu_kernels stub package for JGS simulator (v2).

Instead of removing is_xpu() from vLLM source, this package registers
pure-Python fallback implementations for all torch.ops._C.* and
torch.ops._C_cache_ops.* ops via torch.library.

When vllm/platforms/xpu.py does:
    import vllm_xpu_kernels._C
    import vllm_xpu_kernels._moe_C
    import vllm_xpu_kernels._xpu_C

...this package registers XPU implementations so the is_xpu() code path
works without any compiled C++/SYCL kernels.
"""
