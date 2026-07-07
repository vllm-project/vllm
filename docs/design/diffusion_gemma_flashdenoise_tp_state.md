# DiffusionGemma FlashDenoise TP State

This note defines the exact tensor-parallel sampler state for
DiffusionGemma FlashDenoise. The goal is to compute the same per-token
sampler outputs as a dense vocabulary pass without materializing the full
probability vector on every TP rank.

For one sampler row, let `z_i` be the final scaled logit for global token id
`i`, `E_i` the matching LM-head / embedding row, `g_i` the Gumbel noise used
for sampling, and `normalizer` the DiffusionGemma self-conditioning scalar.
Dense probabilities are:

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

The dense outputs are:

```text
entropy = -sum_i p_i log p_i
soft_embed = sum_i p_i E_i * normalizer
clean_argmax = argmax_i z_i
gumbel_argmax = argmax_i (z_i + g_i)
```

Argmax ties are resolved by the smallest global token id, matching the first
token a dense left-to-right argmax would see.

## TP-Local State

For rank `r`, let `I_r` be the real global-vocabulary ids owned by that rank.
Padded local rows, if any, are not part of `I_r` and must not contribute to
the state. The local state fields are exactly:

```text
m = max_{i in I_r} z_i
s = sum_{i in I_r} exp(z_i - m)
w = sum_{i in I_r} exp(z_i - m) * z_i
e = sum_{i in I_r} exp(z_i - m) * E_i
clean_val = max_{i in I_r} z_i
clean_idx = global token id for clean_val
sample_val = max_{i in I_r} (z_i + g_i)
sample_idx = global token id for sample_val
normalizer = DiffusionGemma self-conditioning scalar
```

`e` is unnormalized on purpose:

```text
e = sum_i exp(z_i - m_r) * E_i
```

It does not include division by `s`, all-reduced normalization, or
multiplication by `normalizer`.

## Merge

Given one local state per TP rank:

```text
m = max_r m_r
s = sum_r s_r * exp(m_r - m)
w = sum_r w_r * exp(m_r - m)
e = sum_r e_r * exp(m_r - m)
entropy = log(s) + m - w / s
soft_embed = e / s * normalizer
```

The merged `clean_argmax` is the best `(clean_val, clean_idx)` pair across
ranks using value descending and global id ascending. The merged
`gumbel_argmax` is the same reduction over `(sample_val, sample_idx)`. The
returned ids are global token ids, not local shard offsets.

## Difference From PR-Style TP Fixes

PR #46212 computes `sum_r(P_r @ W_r)` after full probabilities exist. This
work computes `entropy`, argmax, sampling, and `soft_embed` from TP-local
state before full probability materialization. That distinction matters for
FlashDenoise because the sampler needs entropy and selected token ids at the
same boundary where the soft embedding is produced.

The CPU reference tests in
`tests/models/test_diffusion_gemma_flashdenoise_state.py` compare this merged
TP state against dense softmax, dense entropy, dense soft embedding, clean
argmax, and Gumbel argmax for TP=4, non-divisible shard ranges with padding,
and deterministic global-id tie breaking.
