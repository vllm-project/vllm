# csrc

- **A `.cu` is one unit of build parallelism, and it is compiled once per GPU
  architecture** (`-gencode` on CUDA, `--offload-arch` on ROCm), so a file
  instantiating every variant of an op can be the build's whole critical path.
  Before adding instantiations, find the slowest unit of a cold build from
  `.ninja_log` (end minus start per object, `MAX_JOBS` fixed).
- **Split on the instantiation axis** (head size, output width, dtype, block
  size), never on line count: kernels in an include-only header, one thin `.cu`
  per variant, and an entry-point unit that only declares and dispatches, so
  every entry it declares is defined in a variant unit. Keep `<torch/all.h>` out
  of kernel headers. Copy an existing split in `csrc/` and its CMake wiring,
  including the arch-gated source list of the file it came from.
- **Never vary a shared function body with a per-unit `#define`.** Identical
  symbol names with different bodies are an ODR violation, and the linker keeps
  one copy, so the other variants silently run the wrong code or fail: give each
  variant its own symbol instead. Derive the variant list, the dispatch arms and
  the CMake sources from one place, generating them where you can so no unit
  compiles without a dispatch arm.
- **Call every variant from one test, and measure cold.** A build-time claim
  needs ccache off, `MAX_JOBS` fixed, before and after, naming the unit that was
  the critical path: a split only pays off once its unit is, or is about to
  become, that path, and a compiler flag is no substitute without a measurement.
