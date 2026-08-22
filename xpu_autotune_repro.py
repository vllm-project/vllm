"""
Minimal reproduction of: torch.compile aborts during Inductor's compile-time
autotuning on Intel XPU, with an opaque "Failed to run autotuning code
block:" error, mimicking vLLM's `profile_run`/`_dummy_run` AOT compile path.

Root cause: with `triton.autotune_at_compile_time` enabled (the default
whenever `V.aot_compilation` is True, e.g. under `torch.compile(...).aot_
compile(...)`, which is exactly what vLLM's compilation wrapper calls during
profile_run), Inductor benchmarks each generated Triton kernel once, at
*compile time*, using synthetic example tensors instead of real runtime data
(torch._dynamo.testing.rand_strided). For plain integer/bool graph inputs,
rand_strided zero-fills the tensor -- so a kernel that only indexes directly
with a graph-input index tensor will NOT trip this bug. But any kernel that
computes an index from *floating-point* intermediate data (as RoPE/MLA
position/offset arithmetic commonly does) gets a genuinely randomized
(torch.randn-based) float buffer, and any downstream integer index derived
from it can be wildly out of range. Inductor still emits a
`tl.device_assert` bounds check for that kernel (since it cannot statically
prove the real runtime range), and this random out-of-range access is fed to
that kernel purely for the sake of finding good launch configs.

On CUDA a bad device access here typically raises a catchable Python
exception. On XPU it hits the Level-Zero driver as a fatal error (observed
here as `UR_RESULT_ERROR_OUT_OF_RESOURCES`, and as a full process abort/core
dump when `TRITON_DEBUG=1` enables the assert message itself), which
`generate_and_run_autotune_block()` cannot cleanly recover from.

Usage:
    python repro.py            # reproduces the crash
                                # (triton.autotune_at_compile_time enabled)
    python repro.py --fixed    # applies the proposed mitigation:
                                # triton.autotune_at_compile_time = False
                                # (defers autotuning to the first real call,
                                # where the index is always in-range)

    # Optionally set TRITON_DEBUG=1 to make the underlying device_assert
    # message itself visible (this tends to fully abort/core-dump the
    # process instead of raising a catchable Python exception):
    TRITON_DEBUG=1 python repro.py
"""

import argparse

import torch
import torch._inductor.config as inductor_config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--fixed",
    action="store_true",
    help="apply the proposed fix (defer compile-time autotuning to runtime)",
)
args = parser.parse_args()

device = "xpu"
VOCAB, DIM, N = 128, 256, 64
table = torch.randn(VOCAB, DIM, device=device)


def gather_fn(pos: torch.Tensor) -> torch.Tensor:
    # Index computed from a float tensor, mimicking RoPE/MLA position/offset
    # arithmetic. At real runtime `pos` is guaranteed in [0, 1), so `idx` is
    # always in [0, VOCAB) -- but Inductor cannot prove that statically, so
    # it still emits a tl.device_assert bounds check in the generated kernel.
    idx = (pos * float(VOCAB)).floor().long()
    gathered = table.index_select(0, idx)
    return gathered * 2.0 + 1.0


inductor_config.triton.autotune_at_compile_time = not args.fixed
print(f"[repro] triton.autotune_at_compile_time = "
      f"{inductor_config.triton.autotune_at_compile_time}")

pos = torch.rand(N, device=device)  # always in [0, 1) -> idx always valid at runtime

with torch._dynamo.config.patch(enable_aot_compile=True):
    compiled = torch.compile(gather_fn, fullgraph=True)

with torch.no_grad():
    out = compiled(pos)
    torch.xpu.synchronize()
    print("[repro] regular compiled call OK")

    # Mirrors vllm/compilation/wrapper.py's aot_compile() path used during
    # profile_run: self._compiled_callable.aot_compile((args, kwargs))
    aot_fn = compiled.aot_compile(((pos,), {}))
    torch.xpu.synchronize()
    print("[repro] aot_compile() returned OK")

    result = aot_fn(pos)
    torch.xpu.synchronize()

print("[repro] SUCCESS: no crash.")
