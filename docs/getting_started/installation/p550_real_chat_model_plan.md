# P550 Real Chat Model Bring-up Plan

This plan records the work needed to move from the local random smoke-test
model to a smallest practical trained chat model on the P550 CPU backend.

## Goal

Run a trained instruction/chat model through vLLM on the P550 CPU backend,
produce a semantically meaningful response, expose it through the minimal HTTP
chat validation script, and keep the manual validation workflow reproducible.

## Candidate Model

Use `HuggingFaceTB/SmolLM2-135M-Instruct` as the first target:

- It is a trained instruction model rather than a random checkpoint.
- It is small enough for the P550 memory budget.
- It uses a Llama-style text generation architecture that is a better fit for
  the existing CPU attention path than ad-hoc tiny random configs.
- It can be tested with short prompts and low `max_tokens` to keep CPU latency
  acceptable.

Fallbacks if this model cannot be loaded by the current P550 dependency stack:

1. Try the base `HuggingFaceTB/SmolLM2-135M` only as a language-model sanity
   check, not as the final chat acceptance target.
2. If model download is the blocker, download with `huggingface-cli` or
   `snapshot_download` into a local model directory and run vLLM from that path.
3. If vLLM architecture support is the blocker, document the exact unsupported
   operation and stop before adding broad model code changes.

## Work Plan

1. Preserve the current random-model smoke path for regression checks.
2. Add a trained-model validation script that:
   - defaults to `HuggingFaceTB/SmolLM2-135M-Instruct`,
   - downloads to a local cache or accepts `VLLM_P550_REAL_MODEL`,
   - runs a short offline vLLM generation,
   - prints prompt, generated text, runtime settings, and model path.
3. Extend the minimal HTTP service script so users can point it at the trained
   model with `VLLM_P550_MODEL=<path-or-model-id>` and use the existing
   `tools/p550_chat_test.sh` client.
4. On the P550 board:
   - install only missing lightweight runtime dependencies if required,
   - download or resolve the trained model,
   - run offline generation first,
   - run HTTP chat only after offline generation succeeds.
5. Record any new dependency or compatibility issue in this plan before adding
   further code.

## Acceptance Criteria

- Offline vLLM generation with the trained model exits with status 0.
- The response is semantically connected to a simple prompt such as
  "What is 1+2? Answer briefly."
- The minimal HTTP service can load the same trained model.
- `tools/p550_chat_test.sh` returns a non-random chat response from the service.
- No board IPs, credentials, or local secrets are added to the repository.

## Verified Result

The trained model path was validated with a local snapshot of
`HuggingFaceTB/SmolLM2-135M-Instruct`; model weights are not stored in this
repository.

Offline validation:

```bash
VLLM_P550_REAL_MODEL=$PWD/.p550_models/smollm2-135m-instruct \
    bash tools/p550_real_chat_smoke_test.sh
```

Result:

```text
generated: 1+2 = 3
```

HTTP validation:

```bash
VLLM_P550_MODEL=$PWD/.p550_models/smollm2-135m-instruct \
VLLM_P550_SERVED_MODEL_NAME=smollm2-135m-instruct \
VLLM_P550_KV_CACHE_BYTES=536870912 \
    tools/p550_start_vllm_service.sh
```

Then, from another shell:

```bash
MODEL=smollm2-135m-instruct \
    tools/p550_chat_test.sh "What is 1+2? Answer with only the number."
```

Result:

```text
assistant: 1+2 = 3
```

## Qwen 0.5B Validation

`Qwen/Qwen2.5-0.5B-Instruct` was validated as the next trained chat model. The
model was downloaded outside the repository and copied to a local model
directory on the P550 board; model weights are not stored in this repository.

Start the minimal HTTP service with:

```bash
VLLM_P550_MODEL=$PWD/.p550_models/qwen2.5-0.5b-instruct \
VLLM_P550_SERVED_MODEL_NAME=qwen2.5-0.5b-instruct \
VLLM_P550_MAX_MODEL_LEN=128 \
VLLM_P550_KV_CACHE_BYTES=536870912 \
    tools/p550_start_vllm_service.sh
```

The Qwen service was then tested through `/v1/chat/completions` with five short
chat sessions:

```text
What is 1+2? Answer with only the number. -> 3
What is 2*3? Answer with only the number. -> 6
Translate to Chinese: hello. Answer with only the translation. -> 你好。
What is the capital of France? Answer with one word. -> Paris
Answer yes or no: Is water wet? -> Yes.
```

Observed latency was about 19 to 20 seconds per short request on the scalar
RISC-V CPU path.

## Known Limits

- This target validates correctness of the model/runtime path, not performance.
- The P550 scalar CPU path may take tens of seconds or longer for short prompts.
- The generated answer quality is bounded by the selected small model.
