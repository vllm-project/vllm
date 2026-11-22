# amd_kernels routes on ROCm (DeepSeek-V4.1-Flash)

[amd-kernels](https://github.com/Inferact/amd-kernels) is a library of FlyDSL
kernels for AMD Instinct gfx950 GPUs (MI350X / MI355X), each tuned for one
model's shapes. vLLM calls it through *routes*: one route replaces one stock
kernel or chain of kernels, at the token counts where it measured faster, and
the stock kernel runs everywhere else. The routes are configured with
`--kernel-config '{"amd_kernels": {...}}'` (`KernelConfig.amd_kernels`).

This page covers DeepSeek-V4.1-Flash served with data and expert parallelism on
8 GPUs (DEP8: `--data-parallel-size 8 --tensor-parallel-size 1
--enable-expert-parallel --moe-backend aiter_mega_moe`). MiniMax-M3.1 (TP4) has
its own routes in the same config; each model reads only its own.

## Requirements and defaults

- ROCm on gfx950, `VLLM_ROCM_USE_AITER=1`, and the `amd-kernels` package
  importable. `docker/Dockerfile.rocm` installs the revision this vLLM version
  was validated with (`AMD_KERNELS_TAG`).
- AITER 0.1.29.post1 or newer with FlyDSL 0.5.0.
- With those in place every route is **on by default** (`enabled: true`).
  Elsewhere (other GPUs, other models, `amd-kernels` not installed) nothing
  changes. The server log lists the active routes at startup
  (`amd_kernels routes: ...`) and warns once for a route that is taken out
  (`amd_kernels route X disabled: <reason>`).

```bash
# all routes (the default once amd-kernels is installed)
--kernel-config '{"amd_kernels": {"enabled": true}}'
# all routes except the fused EP8 MoE
--kernel-config '{"amd_kernels": {"disabled_routes": ["fused_ep8"]}}'
# stock kernels only
--kernel-config '{"amd_kernels": {"enabled": false}}'
```

## Routes

A route serves a call when it is on and the call's token count `m` (the static,
padded row count, so a captured CUDA graph takes one branch) is inside its row
range; `fused_ep8` decides on the maximum `m` over the DP group, so all EP
ranks take the same branch. Where the amd-kernels entry does not cover a call
(shape, cache layout, feature such as prefill context parallelism), the stock
path runs and the reason is logged once.

| Route | Replaces | Rows `m` | Needs |
| --- | --- | --- | --- |
| `rope_insert` | q RoPE + fp8 cast, KV RoPE + fp8 sliding-window insert | 1-1024 | |
| `norm_dual` | q and kv RMSNorm, q as MXFP8 for both `wq_b` linears | 1-1024 | |
| `compressor` | compressor pool, norm, RoPE, cache insert, indexer key store | 1-1024 | |
| `kv_score` | compressor `kv_score` + indexer `weights_proj` in one GEMM | 1-1024 | |
| `dense_gemm` | fused `wq_a` + `wkv`, indexer `wq_b` (MXFP8) | 1-768 | |
| `wq_b` | attention `wq_b` on `norm_dual`'s MXFP8 rows | 769-1024 | `norm_dual` |
| `indexer` | decode rows of the MXFP4 top-512 indexer | 1-768 | |
| `sparse_mla` | decode rows of the AITER sparse MLA, MXFP8 epilogue | 1-768 | |
| `wo_a` | grouped `wo_a` GEMM with an MXFP8 epilogue | 1-1024 | |
| `wo_b` | `wo_b` on `wo_a`'s bf16 output | 769-1024 | `wo_a` |
| `router` | backbone router (384 experts, top 6) | 1-1024 | |
| `shared_expert` | shared expert `gate_up` + clamped SiLU-mul for `mhc_ffn` | 1-192 | |
| `fused_ep8` | MegaMoE's routed experts: dispatch, both GEMMs, combine in one launch | 1-1024 per rank | |
| `mhc_seam` | AITER mHC seam (post-mix, pre-mix, Sinkhorn, RMSNorm), MXFP8 output | 1-288, 513-720 | |
| `mhc_tail` | last post-mix, collapse and final RMSNorm | 1-1024 | |
| `mhc_wo_b` | `wo_b` with the attention post-mix as epilogue | 1-64 | `mhc_seam`, `wo_a` |
| `mhc_ffn` | shared-expert down + routed add with the FFN post-mix as epilogue | 1-192 | `mhc_seam`, `shared_expert` |
| `lm_head` | bf16 `lm_head` GEMM | 113-128, 496-512 | |
| `main_proj` | DSpark draft input projection (MXFP8) | 640-1024 | |
| `draft_router` | DSpark draft router (128 experts, top 3) | 1-1024 | |
| `markov` | DSpark's sequential Markov draft steps in one launch | 1-160 requests | |

`fused_ep8` needs EP size 8 and the unfused shared expert; above 1024 rows per
rank MegaMoE runs. While the MoE-block routes are on, MegaMoE layers run the
shared expert after the routed experts on one stream.

## Configuration keys

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Use the routes where `amd-kernels` is installed and the shapes fit. |
| `disabled_routes` | `[]` | Routes that keep the stock kernels. A route whose prerequisite is disabled is off too. |
| `indexer_max_ctx` | `0` | Longest indexer row (compressed entries) the `indexer` kernel is sized for; 0 derives it from `--max-model-len` and each layer's compress ratio. A layer whose bound exceeds the kernel's limit stays on the stock indexer. |
| `prefill_warmup` | `true` | Before CUDA graph capture, run every linear at each AITER GEMM bucket above the captured sizes and a few prefill-sized dummy steps, so FlyDSL does not trace kernel variants while serving (adds about 90 s to startup). |
| `serving_transient_gib` | `20` | With routes on, the KV cache sizing keeps at least this much memory growth (GiB) free for serving transients; 0 keeps the measured value. |
| `expert_perm` | `""` | Static expert placement with replicas for the MegaMoE experts, see below. Independent of `enabled`. |

## Static expert placement (`expert_perm`)

With the default placement, EP rank `r` holds experts `48 r` to `48 r + 47` of
each layer, and the busiest rank of a step sets the pace of the MoE
collectives. On agentic traffic the routed load is skewed: the per-rank load
max/mean was 1.64 averaged over layers (2.09 on the worst layer) with the
default placement. A placement plan gives every rank `S >= 48` expert slots per
layer, replicates the busiest experts and places the replicas so that each
rank gets about the same load; per step the max/mean is then about 1.02-1.03,
the noise floor of uniform routing.

```bash
--kernel-config '{"amd_kernels": {"expert_perm": "expert_plan_S49.npz"}}'
```

- A bare file name is a plan shipped in
  `vllm/models/deepseek_v41/amd/configs/`; anything else is a path.
  `expert_plan_S49.npz` (S = 49 backbone slots per rank, 17 for the DSpark
  draft layers) was fitted on one agentic workload (InferenceX AgentX at
  concurrency 1024), which is why it is not the default.
- The routers keep choosing logical experts; right before dispatch each id is
  remapped to one of its replicas by a hash of the route (deterministic, CUDA
  graph safe, uniform over the replicas). Outputs are unchanged.
- Memory: each extra slot costs one MXFP4 expert (18.8 MB) per MoE layer, so
  S = 49 adds about 0.8 GB per rank (backbone and draft), taken from the KV
  cache.
- AITER keys its tuned MegaMoE profile for this model on 48 experts per rank;
  for S = 49 vLLM registers the same profile at startup. With `fused_ep8` off,
  S != 48 also loses AITER's fixed-slot MegaMoE path for small batches.
- The plan is validated at startup (EP size, expert counts, every backbone
  layer present) and logged with its sha256
  (`amd_kernels expert placement <path> sha256=...`).

### Generating a plan for your workload

1. Serve with `VLLM_AKR_EXPERT_COUNTS=<dir>` and the default placement, and
   run representative traffic. Each DP rank counts the routed (token, expert)
   pairs per layer, separately for decode-only steps and steps that carry
   prefill tokens, and writes `<dir>/expert_counts_dp<rank>.npz` every 60 s
   and at exit. The counting is one small kernel per MoE layer and is CUDA
   graph safe; unset, nothing runs.
2. Run the planner (CPU only):

   ```bash
   python -m vllm.models.deepseek_v41.amd.akr_eplb_plan \
       --counts <dir> --slots 49 --out my_plan.npz --sim-steps 100
   ```

   It replicates the busiest experts into the `8 * S - 384` extra slots, places
   the replicas so that the larger of each rank's decode and prefill load is
   smallest (`--mode 2d`, default), validates the file as the server will, and
   prints the rank load max/mean of the default placement and of the plan.
   `--slots` must stay at or below 63 for `fused_ep8`.
3. Serve with `--kernel-config '{"amd_kernels": {"expert_perm": "/abs/path/my_plan.npz"}}'`.

The shipped plan is reproducible:
`--counts tests/models/deepseek_v41/amd/data/expert_counts_agentx_c1024.npz
--slots 49` writes the same arrays.

### Plan file format

An `.npz` written with `numpy.savez` (no pickled objects):

| Array | Shape | Meaning |
| --- | --- | --- |
| `phys2log` | `[L, 8 * S]` int | Logical expert held by each physical slot of a backbone layer; slot `p` lives on EP rank `p // S`. Every expert appears at least once. |
| `layer_ids` | `[L]` int | Model layer of each row; every backbone MoE layer must be listed. |
| `draft_phys2log`, `draft_layer_ids` | `[D, 8 * S_d]`, `[D]` | Optional, the DSpark draft layers (128 experts); absent: default placement. |
| `world_size` | scalar | EP size the plan was made for (default 8); must match the server. |
