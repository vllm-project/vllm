# Kimi-K3 mono MoE (gfx950, FlyDSL)

One persistent launch for a Kimi-K3 MoE layer of a small decode batch on
MI355X / MI350X: the router's biased sigmoid top-k, the expert sort, the a4w4
routed experts (gemm1, gemm2) and the bf16 shared expert. It replaces the
multi-kernel path of the same step:

    aiter.biased_grouped_topk                    top-k
    aiter.fused_moe (a4w4 SiTUv2, FlyDSL)        sort + quant, gemm1, gemm2
    KimiMLP (on vLLM's aux stream)               gate_up GEMM, situ_and_mul, down GEMM

At M <= 16 each of those kernels runs for a few microseconds, so launch and
drain are a large share of the step; one launch keeps the GPU busy across
the stage boundaries, and the shared expert fills the CUs that wait for
routing.

## Scope

- Opt-in: `VLLM_ROCM_MONO_DECODE=1`, gfx950, AITER's fused MoE with the a4w4
  SiTUv2 activation (`VLLM_ROCM_USE_AITER_MOE_SITUV2=1`). Otherwise, or when
  FlyDSL / AITER's FlyDSL kernels are missing, the multi-kernel path runs.
- Layers: every routed MoE layer the ROCm latent MoE runner owns whose router
  is biased sigmoid top-k, renormalized, unscaled (one group), with no EP /
  EPLB / DP / sequence parallel, no expert bias, no padding and no swiglu
  limit, and whose shared expert is KimiMLP as vLLM builds it (unquantized
  bf16, SiTU, unreduced down projection)
  (`ROCmLatentMoERunner._mono_layer_ok`).
- Steps: M <= 16 tokens (`runner.M_MAX`, one m-block of the shared expert).
  Larger steps take the multi-kernel path.
- Contract: the same tensors as the path it replaces: in `x` [M, H] bf16,
  router logits [M, E] f32 and the correction bias; out the routed output
  [M, H] bf16 (latent H, before the latent tail) and the shared output
  [M, H'] bf16 (full hidden H').
  Graph-safe: every argument is a fixed device pointer and the launch resets
  its own control words.

## Layout

    runner.py         launcher cache, workspace, mono_moe() (the host entry)
    layer.py          the launch: ticket loop over the stages below
    stages/route.py   top-k of one token, the expert sort
    stages/shared.py  shared expert gate/up K splits, down tiles
    stages/gemm1.py   AITER's a4w4 stage-1 tile with a write-through epilogue
    common/plan.py    geometry, control-word and workspace layout
    common/sync.py    tickets, counters, flags
    common/ops.py     device primitives, trace marks

gemm2 is AITER's a4w4 stage-2 tile, emitted into this launch through its
dispatcher's composition hook (`compile_gemm2_a4w4_port(_composition=...)`).
`stages/gemm1.py` is adapted from AITER v0.1.24.post1 (Apache-2.0); the only
change is `wt_out`. The rest of AITER the launch uses (`buffer_ops`,
`communication_ops_utils`, `mxfp4_gemm_common`, the gemm1 tile sizing) is
imported unchanged.

## The launch

Grid = min(work items, CUs), 256 threads, one workgroup a CU. Workgroups loop
on a global ticket counter; the ticket picks the item:

    0 .. M-1            top-k of token t (whole workgroup), zero its output row
    M .. M+GU-1         shared gate/up: 16 gate + 16 up columns x one K quarter
    .. +G1              gemm1 tile (m-block x BN 128; 256 once >= 96 m-blocks)
    .. +DN              shared down: 64 output columns, full K
    ..                  gemm2 tile (m-block x BN 512), m-block major

- Routing: the workgroup that finishes the last top-k sorts the routes
  (experts ascending, ceil(routes / 16) m-blocks each, padded with token M)
  and raises the routed flag.
- Waits: gemm1 waits for the routed flag; gemm2 waits for every gemm1 n-block
  of its m-block; shared down waits for every gate/up pair. Each wait names an
  item of a smaller ticket and a ticket's holder works on it until done, so
  the grid cannot deadlock however many workgroups are resident.
- Hand-offs: routing outputs, the gemm1 intermediate and the shared partials
  are written at device scope (through the XCD's L2), so a release only
  drains stores and an acquire only drops the CU's L1. Each control word has
  its own 128-byte line and is updated by one lane.
- Reset: the routed flag holds a launch epoch (epoch + 1 per launch, mod
  2^30) and is never cleared; the workgroup that takes the last ticket clears
  the ticket, top-k and m-block counters for the next launch.

## Numerics (the path each stage reproduces)

- Top-k: AITER's `biased_grouped_topk` (one group): score = sigmoid(logit),
  select on score + bias, ties to the lower expert id, weights the selected
  scores renormalized; NaN scores select as 0.
- Routed experts: AITER's a4w4 `fused_moe` (MXFP4 x, inline quant, SiTUv2
  with beta / linear_beta, MXFP4 intermediate, bf16 atomic adds in gemm2), the
  same tile code. Like the multi-kernel path the adds are unordered, so
  outputs vary in the last bits run to run.
- Shared expert: vLLM's KimiMLP: bf16 gate_up GEMM rounded to bf16, SiTU (no
  hard clamp; up soft-clipped by linear_beta when > 0), bf16, down GEMM.

## Validation

- `tests/models/test_kimi_k3_mono_moe.py` (gfx950): against
  `biased_grouped_topk` + `fused_moe` and the KimiMLP math, M = 1..16, pooled
  and uniform routing, eager and two graph replays.
- `tests/models/test_kimi_k3_mono_gate.py` (CPU): which calls take the launch.
- Replays of recorded serving calls (real Kimi-K3 weights and decode inputs,
  per layer and M; the recorder and the replay benchmark are a separate change):
  outputs within the stock path's own run-to-run spread, device time under
  graph replay against the stock path in the same process.
