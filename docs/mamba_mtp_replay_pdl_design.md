# Mamba MTP Replay and PDL Design Notes

This note explains the replay-based Mamba-2 MTP state update path added in this
branch, why PDL is useful for it, which vLLM files changed, and what accuracy
issues we hit while validating it on NVIDIA Nemotron 3 Super 120B A12B NVFP4.

## Upstream Reference

The implementation is based on the TensorRT-LLM replay work in
[NVIDIA/TensorRT-LLM PR #13453][trtllm-pr], titled
`[None][feat] Use a replay method for state rollback in Mamba-2 speculative
decoding`.

That PR explains the core motivation:

- Attention speculative decode can roll back by invalidating KV cache entries.
- Mamba has a destructive recurrent SSM state update, so rollback is harder.
- The old approach stores candidate intermediate states for each draft token and
  copies the winning state after acceptance.
- Replay instead stores a compact trace of the previous draft block and later
  replays only the accepted prefix.

The local vLLM kernel is adapted from TensorRT-LLM's
`tensorrt_llm/_torch/modules/mamba/replay_selective_state_update.py` and the
Mamba selective state update Triton kernels.

## Mental Model

For one Mamba head, the recurrent update is conceptually:

```text
state[t + 1] = decay(dt[t], A) * state[t] + B[t] * x[t]
output[t]    = C[t] * state[t + 1] + D * x[t]
```

With MTP speculative decoding, a step may draft several tokens. For example,
with `num_speculative_tokens = 5`, one decode row contains:

```text
[verifier token, draft_1, draft_2, draft_3, draft_4, draft_5]
```

The scheduler later tells us how many draft transitions were accepted. If only
two were accepted, the real Mamba state should advance through only the first
two accepted transitions. The rejected draft transitions must not become part of
the committed SSM state.

The replay path turns that into a two-step protocol:

```text
step N:
  compute the draft block
  save compact trace: x, B, dt, dA_cumsum
  do not commit every draft token as the real SSM state

acceptance:
  scheduler records accepted count k

step N + 1:
  replay only the first k transitions from step N
  update the real SSM state once
  save the compact trace for step N + 1
```

This is different from storing a full candidate SSM state for every draft token.
Replay stores inputs and transition coefficients, then reconstructs only the
accepted prefix when it is actually needed.

## Concrete Example

Assume the real Mamba state at the beginning of step `N` is `S0`, and the model
drafts five transitions:

```text
draft transitions: t1, t2, t3, t4, t5
accepted count:    k = 2
```

The older candidate-state style is:

```text
step N:
  S1 = apply(t1, S0)
  S2 = apply(t2, S1)
  S3 = apply(t3, S2)
  S4 = apply(t4, S3)
  S5 = apply(t5, S4)
  store candidate states [S1, S2, S3, S4, S5]

acceptance:
  k = 2

step N + 1:
  copy S2 into the real cache
```

That is correct, but expensive: every draft token needs a full candidate SSM
state.

The replay path is:

```text
step N:
  run the draft computation
  save compact trace [x, B, dt, dA_cumsum] for t1..t5
  leave the committed real cache at S0

acceptance:
  k = 2

step N + 1:
  replay only t1 and t2 from the compact trace
  update the real cache to S2 once
  compute and save the compact trace for the new draft block
```

So the new kernel trades full candidate-state storage for a smaller replay trace
plus a short accepted-prefix replay on the next step.

## Decode Flow And Shapes

For a small example, assume:

```text
num_decodes = 2
num_speculative_tokens = 5
T = 1 + num_speculative_tokens = 6
nheads = 4
head_dim = 8
ngroups = 2
dstate = 16
```

The decode path first runs `causal_conv1d_update()` on the packed Mamba
projection. After the conv output is split, the replay path sees:

```text
x   / hidden_states : [2, 6, 4, 8]
dt                  : [2, 6, 4, 8]
B                   : [2, 6, 2, 16]
C                   : [2, 6, 2, 16]
ssm_state cache      : [cache_size, 4, 8, 16]
```

The control flow is:

```text
packed Mamba projection
  -> causal_conv1d_update()
  -> split into x, B, C
  -> replay_selective_state_update()
       -> _replay_precompute_kernel()
       -> _replay_state_update_kernel()
       -> _commit_replay_cache_kernel()
```

`_replay_precompute_kernel()` prepares data for the current draft block. For
each row/head it computes and stores:

```text
dA_cumsum[t] = cumulative sum of A * dt through token t
decay_vec[t] = exp(dA_cumsum[t])
raw_CB[t, j] = C[t] dot B[j]
CB_scaled[t, j] = raw_CB[t, j] *
                  exp(dA_cumsum[t] - dA_cumsum[j]) *
                  dt[j]
```

Shape-wise:

```text
CB_scaled : [num_decodes, nheads, T, T]
decay_vec : [num_decodes, nheads, T]
old_B     : [cache_size, 2, T, ngroups, dstate]
old_dt    : [cache_size, 2, nheads, T]
old_dA    : [cache_size, 2, nheads, T]
```

`CB_scaled[t, j]` is the precomputed contribution weight from candidate token
`j` to output token `t`. This lets the output side use a small matrix multiply
instead of recomputing every `C[t] @ B[j]` pair inside the state update kernel.

`_replay_state_update_kernel()` first consumes the previous step's compact trace
using the scheduler's accepted count. If the previous step accepted `k = 3`
tokens, it advances the real SSM cache through only positions `0, 1, 2` from the
previous trace. Rejected positions are masked out. It then computes the current
draft block outputs using the freshly precomputed `CB_scaled` and `decay_vec`.

## What PDL Means Here

PDL means CUDA Programmatic Dependent Launch. Normally, when kernel B depends on
kernel A, B effectively waits for A to complete. With PDL, A can signal from
inside the GPU kernel that dependent kernels may start once the data they need is
ready, even if A has remaining independent work.

In Triton this appears as:

```python
tl.extra.cuda.gdc_launch_dependents()
tl.extra.cuda.gdc_wait()
```

and on the Python launch side as:

```python
kernel[grid](..., launch_pdl=True)
```

In this branch there are two PDL boundaries:

- External PDL: `causal_conv1d_update -> replay precompute`.
- Internal PDL: `replay precompute -> replay state update`.

The external PDL path allows the replay precompute kernel to begin work that does
not require the convolution outputs, then wait before reading `x/B/C` produced by
the conv path. The internal PDL path lets the state update kernel overlap with
the tail of the replay precompute once the dependent data is ready.

The controlling env vars are:

```bash
VLLM_MAMBA_MTP_REPLAY=1
VLLM_MAMBA_MTP_REPLAY_PDL=1
VLLM_MAMBA_MTP_REPLAY_EXTERNAL_PDL=1
VLLM_MAMBA_MTP_REPLAY_INTERNAL_PDL=1
```

`VLLM_MAMBA_MTP_REPLAY_PDL` is the parent switch. The external/internal vars let
us debug the two PDL boundaries independently.

## Code Map

### `vllm/envs.py`

Adds the experimental replay and PDL switches:

```python
"VLLM_MAMBA_MTP_REPLAY": lambda: _getenv_bool(
    "VLLM_MAMBA_MTP_REPLAY", False),
"VLLM_MAMBA_MTP_REPLAY_PDL": lambda: _getenv_bool(
    "VLLM_MAMBA_MTP_REPLAY_PDL", False),
"VLLM_MAMBA_MTP_REPLAY_EXTERNAL_PDL": lambda: _getenv_bool(
    "VLLM_MAMBA_MTP_REPLAY_EXTERNAL_PDL",
    _getenv_bool("VLLM_MAMBA_MTP_REPLAY_PDL", False),
),
"VLLM_MAMBA_MTP_REPLAY_INTERNAL_PDL": lambda: _getenv_bool(
    "VLLM_MAMBA_MTP_REPLAY_INTERNAL_PDL",
    _getenv_bool("VLLM_MAMBA_MTP_REPLAY_PDL", False),
),
```

### `vllm/model_executor/layers/mamba/mamba_mixer2.py`

The mixer enables replay only for the supported MTP path:

```python
self._use_mtp_replay = (
    self.num_spec > 0
    and cache_config is not None
    and cache_config.mamba_cache_mode == "none"
    and self.mamba_config.backend == MambaBackendEnum.TRITON
    and current_platform.is_cuda()
    and envs.VLLM_MAMBA_MTP_REPLAY
)
```

It allocates persistent compact replay buffers:

```python
self._mtp_replay_old_x
self._mtp_replay_old_B
self._mtp_replay_old_dt
self._mtp_replay_old_dA_cumsum
self._mtp_replay_cache_buf_idx
self._mtp_replay_valid
self._mtp_replay_cb_scaled
self._mtp_replay_decay_vec
```

The key runtime detail is that the base SSM state indices are made contiguous
before launching Triton:

```python
replay_state_indices = state_indices_tensor_d[:num_decodes, 0].contiguous()
```

That copy is intentionally outside the external PDL chain. The TRTLLM replay path
also keeps the conv1d-to-precompute gap to view/no-op work only.

The decode path then launches conv1d with optional external PDL:

```python
hidden_states_B_C_d = causal_conv1d_update(
    ...,
    launch_dependent_kernels=use_mtp_replay_external_pdl,
)
```

After splitting `x/B/C`, it calls:

```python
replay_selective_state_update(
    ssm_state,
    self._mtp_replay_old_x,
    self._mtp_replay_old_B,
    self._mtp_replay_old_dt,
    self._mtp_replay_old_dA_cumsum,
    self._mtp_replay_cache_buf_idx,
    self._mtp_replay_valid,
    replay_hidden_states,
    replay_dt,
    A_d,
    replay_B,
    replay_C,
    replay_out,
    replay_kernel_num_accepted_tokens,
    D=D_d,
    dt_bias=dt_bias,
    dt_softplus=True,
    state_batch_indices=replay_state_indices,
    enable_stochastic_rounding=False,
    cb_scaled=self._mtp_replay_cb_scaled,
    decay_vec=self._mtp_replay_decay_vec,
    launch_with_pdl=use_mtp_replay_external_pdl,
    use_internal_pdl=use_mtp_replay_internal_pdl,
)
```

For non-full decode rows, it pads the missing speculative positions with zeros so
the next step never replays stale draft-slot data.

The mixer also has `preserve_mtp_replay_accepted_state()`. This handles the
no-cache Mamba block-table case where an accepted draft physical slot can become
the next step's base slot. Replay stores compact traces rather than full draft
SSM checkpoints, so the compact trace and double-buffer metadata must be copied
from the old base slot to the accepted promoted slot.

### `vllm/model_executor/layers/mamba/ops/replay_selective_state_update.py`

This file contains the new Triton replay kernels.

`_replay_precompute_kernel`:

- Computes `dt`, applies bias and optional softplus.
- Computes `dA_cumsum = cumsum(A * dt)`.
- Stores the compact previous-step trace into double-buffered caches.
- Computes a small lower-triangular `CB_scaled` matrix with Tensor Core
  `tl.dot(C, B^T)`.
- Optionally uses external and internal PDL.

The central precompute trick is:

```python
raw_CB = tl.dot(
    C_all,
    tl.trans(B_all),
    input_precision="ieee",
)
decay_matrix = tl.exp(dA_cumsum[:, None] - dA_cumsum[None, :])
CB_scaled = tl.where(valid_mask, raw_CB * decay_matrix * dt[None, :], 0.0)
```

Keeping `C_all` and `B_all` in their loaded dtype matters for performance. The
TensorRT-LLM replay PR calls out the fast path as BF16 inputs with FP32
accumulation. Casting both operands to FP32 before `tl.dot` can prevent the dot
from lowering to the intended Tensor Core path.

`_replay_state_update_kernel`:

- Reads `num_accepted_tokens`.
- Masks out rejected draft positions.
- Advances the real SSM state only through the accepted prefix.
- Produces current-step outputs from the replay trace and current compact inputs.

The accepted-prefix mask is:

```python
accepted_mask = t_mask & (offs_t < prev_num_accepted_tokens)
coeff = tl.where(accepted_mask, coeff, 0.0)
```

`_commit_replay_cache_kernel` toggles the per-cache-row double buffer:

```python
buf_read = tl.load(cache_buf_idx_ptr + cache_batch_idx).to(tl.int32)
tl.store(cache_buf_idx_ptr + cache_batch_idx, 1 - buf_read)
tl.store(replay_valid_ptr + cache_batch_idx, 1)
```

This lets step `N + 1` read the trace produced at step `N`, while step `N + 1`
writes into the other buffer slot.

All replay kernels defensively ignore invalid cache rows:

```python
if (
    cache_batch_idx == null_block_id
    or cache_batch_idx < 0
    or cache_batch_idx >= cache_size
):
    return
```

This mirrors the Python-side preservation guard. It is important under
high-concurrency TP runs because the scheduler/block table can transiently carry
inactive or stale rows. A positive out-of-range row must not index replay
buffers, otherwise CUDA may report the illegal memory access asynchronously at a
later NCCL/watchdog or tensor-copy site.

### `vllm/v1/worker/gpu_model_runner.py`

After speculative acceptance is known, the runner calls the Mamba layers so they
can preserve compact replay state for accepted/promoted draft slots:

```python
preserve_replay_state = getattr(
    layer, "preserve_mtp_replay_accepted_state", None)
if preserve_replay_state is not None:
    preserve_replay_state(state_indices, num_accepted_tokens)
```

Without this, a later step can use a promoted draft physical slot whose replay
trace was never copied there.

The same runner also collects newly allocated Mamba block IDs from scheduler
updates and invalidates replay metadata for those physical rows. This matters
because recycled Mamba rows are not covered by attention-cache zeroing:

```python
if new_block_ids is not None:
    self._collect_mamba_mtp_replay_block_ids(
        new_block_ids,
        mtp_replay_mamba_group_ids,
        mtp_replay_new_block_ids,
    )
...
if mtp_replay_new_block_ids:
    self._invalidate_mamba_mtp_replay_cache(mtp_replay_new_block_ids)
```

### `tests/kernels/mamba/test_replay_selective_state_update.py`

The test covers:

- Two-step replay correctness.
- Multiple groups.
- Non-contiguous `state_batch_indices`.
- Invalid replay state/cache indices.
- Accepted-slot preservation filtering and replay-valid movement.
- Padded replay `dt` retaining tied head-dim stride.

The non-contiguous index case mattered because serving uses a 2D Mamba block
table:

```text
state_indices_tensor_d shape = [num_decodes, 1 + num_spec_tokens]
```

Column 0 is the base verifier slot. `state_indices_tensor_d[:, 0]` is a
non-contiguous column view, so Triton pointer arithmetic would otherwise read the
wrong physical entries.

## Accuracy Bugs Found During Validation

### Non-contiguous state indices

The replay kernel originally received:

```python
state_indices_tensor_d[:num_decodes, 0]
```

That is logically `[base0, base1, ...]`, but physically has a stride equal to the
row width. Triton loaded it as contiguous:

```python
tl.load(state_batch_indices_ptr + pid_b)
```

So request 1 could read `draft0_1` instead of `base1`. This corrupted the SSM
state rows and collapsed GSM8K accuracy.

Fix:

```python
state_indices_tensor_d[:num_decodes, 0].contiguous()
```

The public replay wrapper also defensively contiguizes `state_batch_indices`.

### Accepted draft-slot preservation

The no-cache Mamba block table can promote an accepted draft slot into the next
step's base slot. Because replay keeps compact traces, not full per-draft SSM
states, the accepted slot must inherit the base slot's replay trace and
double-buffer state. This is handled by
`preserve_mtp_replay_accepted_state()`.

The replay-valid bit must be moved, not merely copied. After a row is promoted
from `src` to `dst`, the old source block may later be reused by another
request. If `replay_valid[src]` stays set, that future request can accidentally
consume a stale compact trace. The preservation path now snapshots
`replay_valid[src]`, clears the source rows, and writes the snapshot to the
destination rows:

```python
valid_values = replay_valid.index_select(0, src_indices)
replay_valid.index_fill_(0, src_indices, 0)
replay_valid[dst_indices] = valid_values
```

This also handles a row that is a source for one request and a destination for
another: clearing happens before the destination validity is restored.

### Newly allocated Mamba block invalidation

The TP=4 high-concurrency replay failures tended to surface near the tail of a
benchmark, when many requests had finished and physical cache blocks were being
reused. `new_block_ids_to_zero` only tracks attention-cache blocks; it does not
cover Mamba cache groups. That means a newly allocated Mamba physical row could
still have `replay_valid[row] = 1` from the previous request that owned the row.

The fix is to invalidate replay metadata for newly allocated Mamba block IDs in
the GPU model runner before the next forward:

```python
if mtp_replay_new_block_ids:
    self._invalidate_mamba_mtp_replay_cache(mtp_replay_new_block_ids)
```

The invalidation is grouped by KV-cache group and calls into each Mamba layer:

```python
state_indices = torch.as_tensor(block_ids, dtype=torch.long, device=self.device)
invalidate_replay_cache = getattr(layer, "invalidate_mtp_replay_cache", None)
if invalidate_replay_cache is not None:
    invalidate_replay_cache(state_indices)
```

The layer-side filter rejects null, negative, and out-of-range rows:

```python
valid_mask = (
    (state_indices != NULL_BLOCK_ID)
    & (state_indices >= 0)
    & (state_indices < cache_size)
)
```

This keeps valid replay traces for continuing requests, but prevents a recycled
physical row from inheriting another request's compact trace.

### PDL launch ordering

With CUDA graphs and PDL, replay inputs must be stable before the conv1d and
replay kernel chain is launched. We moved replay cache/workspace setup and
`replay_state_indices` creation before `causal_conv1d_update()`, and made the
precompute launch use `launch_pdl=launch_with_pdl`, matching the TRTLLM-style
external PDL dependency.

The internal precompute-to-state-update PDL signal also has to mean "the
current precompute outputs are ready." An earlier version called
`gdc_launch_dependents()` at the beginning of `_replay_precompute_kernel()`.
That allowed `_replay_state_update_kernel()` to pass its `gdc_wait()` before
`cb_scaled` and `decay_vec` were produced. Under TP=4/spec=5/high concurrency,
the client benchmark could finish but the server later reported an asynchronous
CUDA illegal-memory-access error. The current version calls
`gdc_launch_dependents()` at the end of precompute, after the replay precompute
stores have completed.

The external conv1d-to-precompute PDL signal has the same rule. The dependent
replay precompute reads `hidden_states_B_C_d` after conv1d has transformed the
packed projection, so `causal_conv1d_update` must not launch dependents until
the conv output has been stored. A later TP=4 PDL debug pass found that the
conv update kernel was doing this too early:

```python
if LAUNCH_DEPENDENT_KERNELS:
    tl.extra.cuda.gdc_launch_dependents()
...
tl.store(o_ptrs, acc, mask=mask_1d)
```

That creates a read-after-write race: replay precompute can start and pass its
`gdc_wait()` before conv has written the `x/B/C` data it consumes. The fix moves
the conv PDL signal after the output-store loop:

```python
tl.store(o_ptrs, acc, mask=mask_1d)
...
if LAUNCH_DEPENDENT_KERNELS:
    tl.extra.cuda.gdc_launch_dependents()
```

This makes external PDL mean "conv output is ready" instead of merely "conv has
started." Clean post-patch TP=4 validation jobs are tracking this fix:

```text
job 391650: replay with external PDL enabled, internal PDL enabled
job 391651: replay with external PDL enabled, internal PDL disabled
job 391652: replay with external PDL disabled, internal PDL enabled
```

Unit coverage now includes `use_internal_pdl=True` for the replay two-step
kernel test, plus a regression case for `NULL_BLOCK_ID` and positive
out-of-range replay state indices:

```text
.venv/bin/python -m pytest tests/kernels/mamba/test_replay_selective_state_update.py -q
11 passed, 16 warnings in 10.66s
```

The conv parent kernel's PDL signal path is also covered by a targeted CUDA
unit test:

```text
tests/kernels/mamba/test_causal_conv1d.py::test_causal_conv1d_update_with_pdl_signal
1 passed, 16 warnings in 4.25s
```

### TP4 padded decode rows and tied `dt`

The replay Triton kernel intentionally assumes the Mamba2 `A`, `dt`, and
`dt_bias` inputs are tied across `head_dim`. The precompute kernel loads one
scalar `dt` and one scalar `A` per head, so these tensors must have
`stride(-1) == 0` in the head-dim direction.

The normal full-block decode path preserved this because `dt_d` is built with:

```python
dt_d = dt_d[:, :, None].expand(-1, -1, self.head_dim)
```

But the variable-length padded replay branch allocated dense storage:

```python
replay_dt = dt_d.new_zeros(num_decodes, num_steps, num_heads, self.head_dim)
```

That materialized `dt` across `head_dim`, making `dt.stride(-1) == 1`. In TP=4,
DP=1, high-concurrency serving, this hit:

```text
AssertionError: replay_selective_state_update requires A, dt, and dt_bias to be tied across head_dim
```

The fix is to store only one scalar `dt` lane and expand it back over
`head_dim`:

```python
dt_storage = src.new_zeros(num_decodes, num_steps, num_heads, 1)
replay_dt = dt_storage.expand(-1, -1, -1, head_dim)
replay_dt_storage[decode_idx, :seq_len] = dt_d[start:end, :, :1]
```

This keeps the replay kernel's optimized tied-head-dim contract while still
padding shorter decode rows safely.

## Serving Flags Used For Correctness

Keep:

```bash
--mamba-ssm-cache-dtype float16
```

Do not pass:

```bash
--mamba-cache-dtype float16
```

For correctness eval, disable FlashInfer autotune:

```bash
--no-enable-flashinfer-autotune
```

For performance benchmarking, FlashInfer autotune can be enabled after the
correctness path is stable.

## Manual Commands

Inside a one-node AWS-DFW vLLM container, use the source install venv:

```bash
export MY_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_nvfm_llm/users/dafrimi
export VENV=/my_home/venvs/vllm_mamba_mtp_replay
export MODEL_PATH=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
export SERVED_MODEL=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
export PORT=8080

cd /my_home/code/vllm
source "$VENV/bin/activate"

export VLLM_MAMBA_MTP_REPLAY=1
export VLLM_MAMBA_MTP_REPLAY_PDL=1
export VLLM_MAMBA_MTP_REPLAY_EXTERNAL_PDL=1
export VLLM_MAMBA_MTP_REPLAY_INTERNAL_PDL=1

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.80 \
  --no-enable-prefix-caching \
  --reasoning-parser nemotron_v3 \
  --tool-call-parser qwen3_coder \
  --quantization modelopt_fp4 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --mamba-ssm-cache-dtype float16 \
  --mamba-backend triton \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --no-enable-flashinfer-autotune \
  --speculative-config '{"method":"nemotron_h_mtp","num_speculative_tokens":5,"moe_backend":"triton"}' \
  > serve.log 2>&1 &
```

Smoke test:

```bash
curl -s http://127.0.0.1:${PORT}/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4","prompt":"Question: What is 2+2?\nAnswer:","max_tokens":16,"temperature":0.0}'
```

GSM8K eval from the eval venv:

```bash
source /my_home/venvs/eval_venv/bin/activate

lm_eval \
  --model local-completions \
  --model_args model="$SERVED_MODEL",base_url=http://localhost:${PORT}/v1/completions,api_key=EMPTY,num_concurrent=50,timeout=45000 \
  --tasks gsm8k \
  --num_fewshot 5 \
  --gen_kwargs temperature=0.0,top_p=0.95,do_sample=true,seed=1 \
  --output_path ./results/gsm8k \
  --log_samples
```

## Validation Snapshot

Full DP=4/TP=1 GSM8K replay + PDL + CUDA graph passed:

```text
flexible-extract exact_match = 0.9287 +/- 0.0071
strict-match    exact_match = 0.9212 +/- 0.0074
```

The user's baseline full GSM8K reference was:

```text
flexible-extract exact_match = 0.9340 +/- 0.0068
strict-match    exact_match = 0.9242 +/- 0.0073
```

That means the replay + PDL path preserved accuracy within expected run-to-run
noise.

## Benchmark Observations

Early benchmark runs show that the end-to-end result is very sensitive to the
speculative acceptance rate. When acceptance is low, the replay kernel has less
accepted-prefix SSM work to save, while the server still pays the draft-model
and rejection-sampling overhead.

Representative no-autotune DP=4/TP=1, spec=5, 10k input / 16k output results:

```text
baseline: 1809 output tok/s, acceptance 29.79%, acceptance length 2.49
replay:   1434 output tok/s, acceptance 26.98%, acceptance length 2.35
```

With spec=3 on the same long synthetic workload, acceptance was higher and the
replay result was much closer to baseline:

```text
baseline: 2326 output tok/s, acceptance 36.64%, acceptance length 2.10
replay:   2309 output tok/s, acceptance 34.80%, acceptance length 2.04
```

The short 1k input / 10k output DP=4/TP=1 baseline run reached higher raw
throughput, but acceptance dropped to only 14.80%. That workload is therefore a
stress test for speculative decoding overhead more than a clean replay-kernel
win condition.

The clean no-autotune short DP=4/TP=1, spec=5, 1k input / 10k output pair after
the Tensor Core dtype patch was:

```text
baseline: 3978 output tok/s, mean TPOT 11.28 ms, acceptance 14.80%,
          acceptance length 1.74, completed 128 / 128
replay:   2213 output tok/s, mean TPOT 18.32 ms, acceptance 15.64%,
          acceptance length 1.78, completed 128 / 128
```

This result is intentionally fair in the sense that both sides used the same
parallelism, prompt/output lengths, concurrency, and no FlashInfer autotune.
It is also a poor win condition for replay because acceptance is only about
15%, so most draft work is rejected and the accepted-prefix replay does not
remove enough SSM work to pay for the extra kernels.

The clean no-autotune short DP=4/TP=1, spec=3, 1k input / 10k output paired
run after the Tensor Core dtype patch and cache-index guards was:

```text
baseline: 7003 output tok/s, mean TPOT 13.55 ms, median TPOT 12.61 ms,
          p99 TPOT 22.55 ms, acceptance 20.04%, acceptance length 1.60,
          completed 256 / 256
replay:   3961 output tok/s, mean TPOT 22.26 ms, median TPOT 21.19 ms,
          p99 TPOT 37.21 ms, acceptance 20.54%, acceptance length 1.62,
          completed 256 / 256
```

This is a stronger negative performance signal than the spec=5 short run:
acceptance is essentially the same between baseline and replay, but replay is
still much slower. The replay server log did not show CUDA illegal-memory, OOM,
or connection-reset errors during the benchmark. It only printed `EngineDeadError`
after the benchmark client completed, during the runner's forced server
shutdown. Therefore this row is clean enough for performance comparison, and it
suggests the missing win is in replay-path overhead, launch ordering,
parallelism, or kernel efficiency rather than acceptance rate alone.

A higher-concurrency no-autotune DP=4/TP=1, spec=3, 1k input / 10k output
paired run used `max_concurrency=256` and `num_prompts=512`:

```text
baseline: 10307 output tok/s, mean TPOT 19.25 ms, median TPOT 18.38 ms,
          p99 TPOT 32.60 ms, acceptance 19.00%, acceptance length 1.57,
          completed 512 / 512
replay:   6730 output tok/s, mean TPOT 28.10 ms, median TPOT 27.19 ms,
          p99 TPOT 47.28 ms, acceptance 19.50%, acceptance length 1.59,
          completed 512 / 512
```

This higher-concurrency run improved absolute replay throughput over the
`max_concurrency=128` replay run, but the baseline also improved. Replay stayed
about 35% lower than baseline with essentially identical acceptance. The server
again printed `EngineDeadError` only after the benchmark completed, during the
runner's forced `SIGTERM` / abort-mode shutdown; no CUDA illegal-memory, OOM, or
NCCL watchdog error appeared in the replay log.

The fresh no-autotune DP=4/TP=1, spec=3, 10k input / 16k output replay-only
rerun with the dtype patch completed cleanly:

```text
replay: 2564 output tok/s, mean TPOT 42.64 ms, acceptance 35.18%,
        acceptance length 2.06, completed 256 / 256
```

That run is not a full paired comparison because the matching baseline came
from the earlier no-autotune spec=3 run, but it supports the same pattern:
spec=3 has much better acceptance than spec=5 and replay is much closer to
baseline on the long synthetic workload.

For TP=4/DP=1 at global concurrency 512, `gpu-memory-utilization=0.80` was too
tight for at least one replay run. The worker died in the rejection sampler while
cloning `target_logits`:

```text
torch.OutOfMemoryError: Tried to allocate 690.00 MiB.
GPU 3 had 483.44 MiB free.
```

That failed benchmark completed only 275 / 1024 requests and should not be used
as a replay throughput result. The fair retry is to run both baseline and replay
with the same lower KV-cache reservation, for example
`--gpu-memory-utilization 0.75`, so the sampler has enough non-KV headroom.

For the shorter TP=4/DP=1, global concurrency 512, 1k input / 10k output
workload, the no-autotune baselines completed cleanly:

```text
spec=3 baseline: 4366 output tok/s, mean TPOT 99.52 ms,
                 acceptance 18.95%, acceptance length 1.57,
                 completed 1024 / 1024
spec=5 baseline: 3943 output tok/s, mean TPOT 101.51 ms,
                 acceptance 13.39%, acceptance length 1.67,
                 completed 1024 / 1024
```

The matching spec=3 replay run completed cleanly:

```text
spec=3 replay:   3302 output tok/s, mean TPOT 341.58 ms,
                 median TPOT 114.31 ms, p99 TPOT 4708.38 ms,
                 acceptance 19.07%, acceptance length 1.57,
                 completed 1024 / 1024
```

That means spec=3 replay avoided the TP=4 OOM/failure mode, and its TTFT was
better than baseline in this run, but the output throughput was still about
24% lower than baseline. The very large replay TPOT tail is a sign that the
remaining issue is not just acceptance rate; there may still be scheduling,
memory pressure, or replay-path synchronization overhead at high concurrency.

The matching spec=5 replay run at `gpu-memory-utilization=0.80` failed again in
the rejection sampler:

```text
torch.OutOfMemoryError: Tried to allocate 718.00 MiB.
GPU 1 had 701.44 MiB free.
```

That failed replay benchmark completed only 287 / 1024 requests and generated
only 44 output tokens, so its throughput is invalid. A lower-memory retry was
already running. Its no-autotune baseline completed cleanly:

```text
spec=5 baseline, gpu-memory-utilization=0.75:
  4187 output tok/s, mean TPOT 97.45 ms,
  acceptance 13.05%, acceptance length 1.65,
  completed 1024 / 1024
```

The matching GMU=0.75 replay benchmark produced a valid JSON result:

```text
spec=5 replay, gpu-memory-utilization=0.75:
  3462 output tok/s, mean TPOT 249.30 ms,
  median TPOT 105.34 ms, p99 TPOT 3378.63 ms,
  acceptance 13.27%, acceptance length 1.66,
  completed 1024 / 1024
```

So lowering the KV-cache reservation fixed the benchmark-level OOM/failure
mode: all requests completed. However, the server log still reported a CUDA
illegal memory access right after the benchmark completed, with the Python
stack surfacing at `preserve_mtp_replay_accepted_state` while copying replay
trace tensors:

```text
self._mtp_replay_old_B[dst_indices] = self._mtp_replay_old_B[src_indices]
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

Because CUDA errors are asynchronous, that stack is not proof that the tensor
copy itself caused the illegal access. It does show that the TP=4/spec=5 replay
path still has a stability issue under high concurrency, even when the client
benchmark reports 1024 successful requests. The performance result is useful,
but this run should not be considered clean enough for a final correctness
claim.

A later TP=4/DP=1, spec=3 replay-only retry used the preserve-index filtering
patch, `gpu-memory-utilization=0.80`, global concurrency 512, 1024 prompts,
1k input / 10k output, CUDA graph enabled, PDL enabled, and FlashInfer autotune
disabled. The client benchmark completed and wrote a valid JSON result:

```text
spec=3 replay, patched preserve filtering:
  3255 output tok/s, mean TPOT 331.91 ms,
  median TPOT 111.89 ms, p99 TPOT 4693.94 ms,
  acceptance 19.26%, acceptance length 1.58,
  completed 1024 / 1024
```

This is slightly slower than the earlier spec=3 replay result on the same
short TP=4 workload and still about 25% below the clean baseline
(`4366 output tok/s`). More importantly, it was not a clean stability pass. The
server reported a real CUDA illegal-address through the NCCL watchdog while the
tail of the benchmark was still active:

```text
Process group watchdog thread terminated with exception:
CUDA error: an illegal memory access was encountered
```

At that point the server metrics still showed 128 running requests, and the
engine later dumped state with `scheduled_spec_decode_tokens` containing
`[-1, -1, -1]` entries for the remaining requests before raising
`EngineDeadError`. Unlike the earlier failure, the visible stack no longer
points at `preserve_mtp_replay_accepted_state`; it surfaces asynchronously in
NCCL/CUDA event handling. So the row-copy bounds filtering fixed a plausible
host-side invalid-index hazard, but it did not eliminate the TP=4 replay
illegal-memory issue.

A later fair TP=4/DP=1, spec=3 pair used global concurrency 512, 1024 prompts,
1k input / 10k output, `gpu-memory-utilization=0.80`, CUDA graph enabled, PDL
enabled, no FlashInfer autotune, and no warmups. The baseline job completed
cleanly:

```text
spec=3 baseline, job 390832:
  4079.75 output tok/s, mean TPOT 103.10 ms,
  median TPOT 99.53 ms, p99 TPOT 161.96 ms,
  acceptance 18.74%, acceptance length 1.56,
  completed 1024 / 1024
```

The matching replay job completed from the benchmark client's point of view, but
the server later hit a CUDA illegal-memory error and `EngineDeadError`, so this
is not a clean performance result:

```text
spec=3 replay, job 391310:
  4786.53 output tok/s, mean TPOT 1196.12 ms,
  median TPOT 82.65 ms, p99 TPOT 3884.91 ms,
  acceptance 38.24%, acceptance length 2.15,
  completed 1024 / 1024 client-side, server failed
```

This unstable run is still informative: median TPOT and output throughput moved
in the right direction, but the enormous mean/p99 TPOT and server failure make
it unusable as a final benchmark claim. The current debug follow-up is job
`391443`, the same TP=4/DP=1 replay workload with `CUDA_LAUNCH_BLOCKING=1`, to
force the CUDA illegal access to surface at the real failing operation.

A shorter TP=4/DP=1, spec=3 replay + PDL tail probe used global concurrency
512, 1024 prompts, 1k input / 2k output, `gpu-memory-utilization=0.80`, CUDA
graph enabled, no FlashInfer autotune, and no warmups. This job completed
cleanly:

```text
spec=3 replay, PDL enabled, job 391481:
  1558.99 output tok/s, mean TPOT 152.84 ms,
  median TPOT 158.38 ms, p99 TPOT 208.34 ms,
  acceptance 8.15%, acceptance length 1.24,
  completed 1024 / 1024, server shutdown clean
```

This is primarily a stability data point, not a final performance comparison:
the synthetic workload had very low acceptance and much shorter outputs than
the 1k/10k benchmark. Still, it is useful because the run reached the
end-of-benchmark drain where `Waiting` went to 0 and `Running` counted down from
253 to 0 without a CUDA illegal memory access, NCCL watchdog failure, or
`EngineDeadError`. That weakens the theory that TP=4 replay + PDL always fails
at tail drain, and leaves the longer 1k/10k PDL-on and PDL-off follow-ups as
the deciding cases.

The matching long TP=4/DP=1 replay run with PDL disabled used global
concurrency 512, 1024 prompts, 1k input / 10k output,
`gpu-memory-utilization=0.80`, CUDA graph enabled, no FlashInfer autotune, and
no warmups. This job completed cleanly:

```text
spec=3 replay, PDL disabled, job 391460:
  2731.86 output tok/s, mean TPOT 85.04 ms,
  median TPOT 77.53 ms, p99 TPOT 181.94 ms,
  acceptance 37.70%, acceptance length 2.13,
  completed 1024 / 1024, server shutdown clean
```

This cleanly separates two issues. The core replay path survived the long
high-concurrency TP=4 tail when PDL was disabled, so the remaining long-run
illegal-memory failure is likely in the PDL chain rather than the accepted-state
row preservation or physical-cache invalidation logic. The PDL-off result is
not a good throughput win, though: it is much slower than the clean baseline job
390832 (`4079.75` output tok/s). Compared with the unstable PDL-on job 391310,
PDL-off has much healthier TPOT tails but lower aggregate throughput.

Two additional long TP=4/DP=1 isolation jobs were started to split the PDL
chain before the conv PDL race was found:

```text
job 391608: replay with external PDL disabled, internal PDL enabled
job 391609: replay with external PDL enabled, internal PDL disabled
```

They were canceled after the source patch because they were no longer useful as
post-fix evidence. The clean replacement jobs all completed successfully:

```text
job 391650: both PDL edges enabled, completed 1024 / 1024
job 391651: external PDL only, completed 1024 / 1024
job 391652: internal PDL only, completed 1024 / 1024
```

These used the same spec=3, global concurrency 512, 1024 prompt, 1k input /
10k output, `gpu-memory-utilization=0.80`, CUDA graph enabled, no FlashInfer
autotune, and no warmup setup.

```text
baseline, job 390832:
  4079.75 output tok/s, mean TPOT 103.10 ms,
  median TPOT 99.53 ms, p99 TPOT 161.96 ms,
  acceptance 18.74%, acceptance length 1.56,
  completed 1024 / 1024, server shutdown clean

replay PDL disabled, job 391460:
  2731.86 output tok/s, mean TPOT 85.04 ms,
  median TPOT 77.53 ms, p99 TPOT 181.94 ms,
  acceptance 37.70%, acceptance length 2.13,
  completed 1024 / 1024, server shutdown clean

replay full PDL, job 391650:
  2736.46 output tok/s, mean TPOT 84.83 ms,
  median TPOT 78.28 ms, p99 TPOT 188.99 ms,
  acceptance 38.22%, acceptance length 2.15,
  completed 1024 / 1024, server shutdown clean

replay external PDL only, job 391651:
  2493.68 output tok/s, mean TPOT 92.19 ms,
  median TPOT 84.63 ms, p99 TPOT 197.68 ms,
  acceptance 37.07%, acceptance length 2.11,
  completed 1024 / 1024, server shutdown clean

replay internal PDL only, job 391652:
  2779.06 output tok/s, mean TPOT 83.41 ms,
  median TPOT 76.84 ms, p99 TPOT 183.60 ms,
  acceptance 37.13%, acceptance length 2.11,
  completed 1024 / 1024, server shutdown clean
```

The post-fix result validates the PDL launch-ordering fix for the workload that
previously failed with CUDA illegal memory access. It does not show an
end-to-end throughput win yet. Full PDL is essentially tied with PDL disabled
on this workload (`2736` vs `2732` output tok/s), while both remain well below
the clean baseline (`4079` output tok/s). The earlier `4786` output tok/s
PDL-on replay row should be treated as invalid because that server failed after
the benchmark client completed.

## Commit Sequence In This Branch

```text
e1424e635 Add Mamba MTP state replay kernel
7e5395a04 Add PDL to Mamba MTP replay path
e0622edbc accuracy works no pdl
70771da5a cuda graph and pdl work
24c426c51 Fix Mamba MTP replay PDL launch ordering
```

[trtllm-pr]: https://github.com/NVIDIA/TensorRT-LLM/pull/13453
