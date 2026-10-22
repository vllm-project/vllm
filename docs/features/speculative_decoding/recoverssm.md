# Kimi-K3 KDA RecoverSSM

RecoverSSM is a speculative-decoding mode for the KDA (linear attention) layers of Kimi-K3 and Kimi-Linear models.

Without it, each verify step stores the recurrent state after every draft token, in `1 + num_speculative_tokens` state slots per request. With RecoverSSM:

- the verify step reads one state slot per request and writes small per-token records (the delta-rule corrections, keys and gates);
- after sampling, a commit folds the accepted tokens into the state.

This saves state memory, and on SM10x GPUs it shortens the decode step.

## Enabling it

Pass `--use-replayssm` with speculative decoding (any method with `num_speculative_tokens` >= 1) on a Kimi-K3 model. RecoverSSM requires:

- Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`);
- `--mamba-backend triton` (the default);
- pipeline parallel size 1.

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve <kimi-k3-checkpoint> \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 4}' \
  --use-replayssm --mamba-ssm-cache-dtype bfloat16
```

## Kernels

On SM10x (Blackwell) GPUs, the verify and the commit run CuTe DSL tensor-core kernels; the verify also applies the short convolution. They need:

- a bfloat16 SSM state (`--mamba-ssm-cache-dtype bfloat16`);
- bfloat16 activations;
- the model's bounded KDA gate;
- at most 7 speculative tokens.

Otherwise, and on other GPUs, Triton kernels run instead. The server logs which kernels a deployment uses; to force the Triton kernels, use `--additional-config '{"kda_recoverssm_backend": "triton"}'`.

## Memory

A request's state between steps is its conv and SSM state, stored in the KV cache in the same layout as without RecoverSSM. The verify records are scratch space for one decode step (each verify writes them and the commit after it reads them), so they live in a separate buffer per KDA layer, sized by `--max-num-seqs`, rather than in the KV cache.

## P/D disaggregation and KV offloading

Because the KV cache layout is the same as without RecoverSSM, KV connectors (P/D disaggregation, KV cache offloading) transfer a request's state as usual. Run P/D instances with the same `--use-replayssm`, speculative and Mamba cache dtype configuration.
