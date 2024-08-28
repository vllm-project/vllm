# Machete (Mixed Precision Cutlass-Based GEMM)

Machete is a spiritual successor to the Marlin kernel but optimized for Hopper architectures and based on Cutlass. Being based on Cutlass, new type pairs and epilogues are easier to add compared to Marlin.

## Overview

Machete effectively performs

```
scale_type = w_s.dtype
compute_type = a.dtype
out = (w_q.to(scale_type) * w_s - w_z.to(scale_type)) @ a
```

Where `w_q` is a quantized weight matrix, `w_s` is the quantization scales, and 
`w_z` is the quantization zeropoints.

> **_NOTE:_**  `w_z` is added after the scales so we can 
use FMA operations, but this means they must have the scales pre-applied if the
supplied zeropoints assume that they will be subtracted before the scales are 
applied.

## API

The main optimization within Machete is prepacking the weight matrix to more closely match the tensor core layouts, allowing for wider shared memory loads when loading the weight matrix. This means that the weight matrix must be prepacked before calling `machete_gemm`. The flow looks something like:

```
from vllm import _custom_ops as ops

...
W_q_packed = ops.machete_prepack_B(w_q, wtype)
output = ops.machete_gemm(
    a,
    b_q=W_q_packed,
    b_type=wtype,
    b_scales=w_s,
    b_group_size=group_size
)
```

## Code Generation

Since Machete is based on Cutlass, we can generate multiple type pairs and different tile shapes using the same kernel template. We generate multiple instantiations of this template using `generate.py`. 

New type pairs (`TypeConfig`s) can be appended to `impl_configs` (in `generate()`), and these will get automatically generated (assuming they can be supported without issues). For each `TypeConfig`, you must also provide an `ImplConfig`, which bundles a `TypeConfig` with a list of `ScheduleConfig`s, `Specialization`s, and a default heuristic. The `ScheduleConfig`s (which contain info on tile shapes, tile scheduler, etc.) can perform differently for different problem shapes, and there is almost never one `ScheduleConfig` that works well for all problem shapes, so it is generally beneficial to generate different `ScheduleConfig`s for different potential problem shapes. This is where the heuristic comes in. For each `TypeConfig`, a default heuristic should be provided. This maps different problem shapes to different `ScheduleConfig`s and is used when the user does not provide the `schedule` parameter to `machete_gemm`. The `Specialization`s define what feature combinations to generate, i.e., `with_zeropoints`, `with_scales`, etc. We can reduce compile times and the final binary size by limiting the set of feature combinations we generate.