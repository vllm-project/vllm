# Benchmarking TRT-LLM with Structured Outputs (OpenAI-style)

This page describes how to use the benchmark scripts in `benchmarks/` to run structured-output (guided decoding) tests against a TensorRT-LLM server that exposes OpenAI-compatible APIs.

## When to use this

Use this path when your TRT-LLM server implements endpoints like `/v1/chat/completions` or `/v1/completions` and supports guided decoding via the OpenAI-compatible `response_format` field (e.g., JSON schema, regex, or EBNF grammar).

To support guided decoding, ensure that the endpoint was started with this addtional parameter to extra_llm_api_options.yaml as specified below

`guided_decoding_backend: xgrammar`

## Script

`benchmarks/benchmark_serving_structured_output.py` generates prompts and sends them to a configurable backend. For `tensorrt-llm`, it will:

- Send requests to `/v1/chat/completions` or `/v1/completions`.
- Provide structured decoding configuration via `response_format`.
- Stream responses when available; if a server streams only role updates (no token deltas) with `response_format`, it automatically retries once with `stream=false` to fetch the final response.

## CLI flags

Key flags added or modified for this workflow:

- `--backend tensorrt-llm`
- `--endpoint /v1/chat/completions` (or `/v1/completions`)
- `--api-key <token>` (optional) – adds `Authorization: Bearer <token>` if provided
- `--debug` (optional) – prints request/response diagnostics and chunk previews
- `--validate-schema` (optional) – validates JSON outputs against the schema (requires `jsonschema`)

Other useful existing flags: `--structured-output-ratio`, `--num-prompts`, `--request-rate`.

## Examples

### JSON schema (chat)

```bash
python benchmarks/benchmark_serving_structured_output.py \
  --backend tensorrt-llm \
  --endpoint /v1/chat/completions \
  --model MODEL-NAME \
  --dataset json \
  --structured-output-ratio 1.0 \
  --num-prompts 100 \
  --base-url https://YOUR-SERVER \
  --validate-schema \
  --debug
```

### JSON schema (completions)

```bash
python benchmarks/benchmark_serving_structured_output.py \
  --backend tensorrt-llm \
  --endpoint /v1/completions \
  --model MODEL-NAME \
  --dataset json \
  --structured-output-ratio 1.0 \
  --num-prompts 100 \
  --base-url https://YOUR-SERVER
```

### Regex / EBNF

- `--dataset regex` → `response_format: {"type": "regex", "regex": ...}`
- `--dataset grammar` → `response_format: {"type": "ebnf", "ebnf": ...}`

### Choice

- `--dataset choice` – converts a list of allowed labels into a JSON Schema enum (string).

## Notes

- For non-`tensorrt-llm` backends, the script uses vLLM-native `structured_outputs` for compatibility.
- With `--debug`, the script prints a truncated preview of the aggregated response; for detailed inspection, reduce `--num-prompts`.
- If your server requires authentication, pass `--api-key`.

## Validating outputs

- `--validate-schema` enforces that JSON outputs conform to the provided schema. This requires `jsonschema`:

```bash
pip install jsonschema
```

If a response fails validation, it is counted as incorrect in the final summary.
