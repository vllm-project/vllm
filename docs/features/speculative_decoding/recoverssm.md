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
- the model's bounded KDA gate.

Otherwise, and on other GPUs, Triton kernels run instead. To force the Triton kernels, use `--additional-config '{"kda_recoverssm_backend": "triton"}'`.

## P/D disaggregation

RecoverSSM runs with KV connectors. A request's state between steps is its conv and SSM state, in the same layout as without RecoverSSM. It has been tested with the NixlConnector for P/D transfers, together with the MooncakeStoreConnector for prefix caching between prefill instances.

The verify records extend each KDA Mamba page and thereby the attention block size (for Kimi-K3, 896 to 1024 tokens). A P/D prefill instance never verifies, so it can drop the records with `--kda-recoverssm-prefill-only`. This flag:

- requires `kv_role` `kv_producer` or `kv_both`;
- requires the NixlConnector, alone or in a MultiConnector. NIXL pairs the prefill and decode pages of different sizes;
- rejects requests whose `max_tokens` is not 1 (P/D proxies send prefill requests with `max_tokens=1`).

Run both instances with the same `--use-replayssm` and speculative configuration; NIXL rejects peers whose conv or SSM state sizes differ. With RecoverSSM, the MooncakeStoreConnector keys its objects by the instance's page layout, so prefill-only and decode instances sharing a store never load each other's pages.
