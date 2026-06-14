# Mamba MTP Replay Accuracy Debug Notes

These notes summarize the accuracy issue found while validating replay-based
Mamba-2 MTP SSM state updates on NVIDIA Nemotron Super 120B NVFP4.

## Serving Flags Used For Correctness

For correctness evals, keep the SSM cache dtype override and do not force the
generic Mamba cache dtype:

```bash
--mamba-ssm-cache-dtype float16
```

Do not pass:

```bash
--mamba-cache-dtype float16
```

FlashInfer autotune was disabled for correctness-only runs to avoid startup
tuning cost:

```bash
--no-enable-flashinfer-autotune
```

PDL is also kept off by default during correctness debugging:

```bash
VLLM_MAMBA_MTP_REPLAY_PDL=0
```

## Symptom

Baseline GSM8K accuracy was healthy, but replay accuracy collapsed. A same-node
DP=1/TP=1 eager A/B run showed:

| Mode | GSM8K strict | GSM8K flexible | Limit |
| --- | ---: | ---: | ---: |
| Baseline | 0.9375 | 0.9375 | 64 |
| Replay, before fix | 0.21875 | 0.21875 | 64 |

Earlier full replay runs were even worse, around 0.05-0.07 exact match, while
baseline full GSM8K was around 0.92-0.93.

## Root Cause

The replay kernel receives one base SSM cache slot per request via
`state_batch_indices`.

In the vLLM MTP decode path, the full Mamba state table is a 2D tensor:

```text
state_indices_tensor_d shape = [num_decodes, 1 + num_spec_tokens]
```

Column 0 is the base verifier slot. Columns 1..N are speculative/draft slots.
The replay path passed the base column like this:

```python
replay_state_indices = state_indices_tensor_d[:num_decodes, 0]
```

That tensor is a non-contiguous column view. For example, with six columns:

```text
row0: [base0, draft0_1, draft0_2, draft0_3, draft0_4, draft0_5]
row1: [base1, draft1_1, draft1_2, draft1_3, draft1_4, draft1_5]
```

The view `state_indices_tensor_d[:, 0]` logically contains:

```text
[base0, base1, ...]
```

but its memory stride is the row width, not 1.

The Triton replay kernels loaded it as if it were contiguous:

```python
cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b)
```

So request 1 read `state_batch_indices_ptr + 1`, which pointed at `draft0_1`
in physical memory, not `base1`. Replay then used the wrong SSM cache slot for
later requests. That corrupted verifier hidden states and destroyed accuracy.

## Fix

Make the base-state index vector contiguous before launching Triton:

```python
replay_state_indices = state_indices_tensor_d[:num_decodes, 0].contiguous()
```

Also defensively contiguize inside the public replay wrapper:

```python
if state_batch_indices is not None:
    assert state_batch_indices.shape == (batch,)
    state_batch_indices = state_batch_indices.contiguous()
```

This protects future callers that pass non-contiguous 1D views.

## Regression Coverage

The existing replay kernel test used a fresh contiguous 1D index tensor, so it
did not catch this serving-layout bug. The test was extended with a
non-contiguous column-view index tensor derived from a 2D block table.

Command run:

```bash
/my_home/venvs/vllm_mamba_mtp_replay/bin/python -m pytest \
  tests/kernels/mamba/test_replay_selective_state_update.py -q
```

Result:

```text
4 passed
```

## Validation After Fix

A small synthetic kernel check originally showed replay output mismatches around
24 on the first block. After the contiguity fix, first-block replay and baseline
outputs matched exactly or to tiny rounding error.

Same-node DP=1/TP=1 eager A/B, GSM8K limit 64:

| Mode | GSM8K strict | GSM8K flexible | Limit |
| --- | ---: | ---: | ---: |
| Baseline | 0.90625 | 0.921875 | 64 |
| Replay, after fix | 0.921875 | 0.921875 | 64 |

This indicates the replay path is no longer causing the large accuracy collapse.
Full GSM8K replay validation should still be used before moving to performance
benchmarking.

## Other Correctness Notes

The accepted-state preservation hook is separate from the non-contiguous index
bug. It handles the no-cache Mamba block-table case where an accepted draft slot
can become the next step's base slot. Replay stores compact traces instead of a
full SSM checkpoint for every draft token, so when the scheduler promotes an
accepted physical slot, replay also has to preserve the compact trace and base
state under that promoted slot.

PDL is a performance feature, not required for correctness. It should stay
behind `VLLM_MAMBA_MTP_REPLAY_PDL` until accuracy is fully validated.
