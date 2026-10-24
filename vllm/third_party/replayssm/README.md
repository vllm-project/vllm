# ReplaySSM Mamba2 kernels

Source: https://github.com/Johnny-Liou/ReplaySSM

Revision: `73a9592e9d8a2268620e8883a82f5cec7442ed36`

Original paths: `low_precision/mamba2/` and the required files in
`low_precision/common/`. The original Apache-2.0 license is in `LICENSE`.
Original file copyright notices are preserved.

This package contains the FP8 and NVFP4 separate-checkpoint kernels.
The recurrence checkpoint uses FP16 stochastic rounding. Read copies use
nearest-even rounding. Flush arithmetic uses BF16 tensor-core operands and FP32
accumulators. The NVFP4 nonflush dot uses FP16 partial sums before FP32 scaling.

The supported configuration is SM103, one token per request, 16 replay entries,
64 head dimensions, 128 state dimensions, and 8 or 16 heads per group.

Local changes:

- Imports use the local package.
- Tensor-memory load and store calls use the pinned CuTe 4.8 API.
- Fake tensor descriptors and compile keys include physical slot strides.
- Raw cache addresses use those strides.
- The dt and decay loop bounds also support eight heads per group.
- Kernels skip the reserved padding slot zero.
- The unused global CuTe runtime patch was removed.

The serving adapter is
`vllm/model_executor/layers/mamba/ops/replayssm_cutedsl.py`.
An eager decode call compiles both native kernels before CUDA graph capture.
Initialization resets reused slots and encodes both checkpoints from FP32
prefill results. NVFP4 retains the original delayed global-scale metadata.

The original SM103 kernels support standard decode only.

The local `mamba2/pending.py` implements the SM121 path. It does not use TMEM.
Accepted-prefix refresh uses BF16 `mma.sync` operands with FP32 accumulators.
Both FP8 and NVFP4 candidate readouts accumulate in FP32.
Checkpoint readout uses BF16 `mma.sync` operands. FP8 values convert exactly.
NVFP4 values times their block scales also convert exactly to BF16.
FP32 row and global scales apply after the matrix product.
Software stochastic rounding converts the recurrence checkpoint to FP16.
This differs from the original NVFP4 nonflush kernel's FP16 partial sums.

The SM121 cache has 16 committed-history entries and up to eight pending entries.
Each decode call commits the previous accepted prefix before evaluating candidates.
Only an accepted crossing of a 16-token boundary refreshes the checkpoints.
Rejected candidates cannot update either checkpoint.
All current candidates use the same quantized read checkpoint.
Per-slot counters store the accepted token count and pending candidate count.
Metadata preparation and ring copies use Triton; state arithmetic uses CuTe DSL.
The original SM103 single-token path retains its layout and immediate commit.
Neither path supports prefix materialization.
