<!--
SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
SPDX-FileComment: Adapted from NVIDIA TensorRT-LLM at commit 395985c025c8d1cf5aa842bc752b337ba88721b6.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Triton Troubleshooting

Consult the current
[Triton debugging guide](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)
before relying on environment variables or interpreter limitations, which can
change between Triton versions.

## Triage order

1. Reproduce with the smallest failing shape and a deterministic input.
2. Determine whether the failure occurs during Python wrapping, Triton
   compilation, launch, memory access, or numerical comparison.
3. Compare pointer offsets, strides, block shapes, masks, dtypes, and precision
   modes with the reference contract.
4. Test boundary tiles and one full tile separately.
5. Add the smallest regression test that reproduces the failure before
   broadening the shape matrix.

## Built-in tools

- `tl.static_print` and `tl.static_assert` inspect or validate compile-time
  values.
- `tl.device_print` inspects runtime values. Restrict the printed programs and
  lanes to keep output usable.
- `tl.device_assert` can check runtime invariants when enabled as documented by
  the installed Triton version.
- `TRITON_INTERPRET=1` can help with supported operations, but interpreter
  behavior is not a substitute for running on the target GPU. Check current
  documented limitations before drawing conclusions from it.

For memory faults on NVIDIA GPUs, run a focused reproducer under
`compute-sanitizer --tool memcheck`. Use backend-appropriate tooling on other
platforms. Sanitizer success does not establish numerical correctness or the
absence of logical races.

## Symptom checklist

| Symptom | Inspect |
| --- | --- |
| Boundary-only differences | Load/store masks, neutral masked-load values, and final partial tiles |
| Shape or broadcast error | Block shapes and explicit singleton dimensions |
| NaN or infinity | Input domain, masked values, division guards, intermediate dtype, and overflow |
| Nondeterministic differences | Aliasing, atomics, cross-program ownership, and RNG state |
| Matmul-like precision mismatch | Input precision, accumulator dtype, reference backend settings, and reduction order |
| Resource exhaustion | Tile sizes, live values, warp count, pipeline stages, and compiler diagnostics |
| Unexpected recompilation | Specialization keys, meta-parameters, shapes, strides, and cache configuration |
| Unexpected stale result | Confirm the loaded source and cache path before moving the exact cache directory aside |

Do not prescribe a universal numerical tolerance. Derive it from the operation,
dtype, reduction depth, and supported precision contract.

Performance diagnosis belongs to `$kernel-microbenchmark`, which covers timing,
generated-code inspection, multi-GPU comparisons, and speed-of-light checks.
