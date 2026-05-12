# Pre-Serve Warmup

vLLM's engine initializes with basic CUDA graph capture, but many kernels compile
lazily on the first real request. This can cause a throughput drop of several
minutes before the system stabilizes.

The **pre-serve warmup** feature runs actual generation (or embedding) requests
before the server starts accepting traffic, so that the first served request
does not pay the lazy compilation cost.

## Quick start

Pass a JSON configuration file to `vllm serve`:

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct \
  --warmup-config warmup.json
```

## Configuration format

`--warmup-config` accepts either a file path or a raw JSON string with the
following schema:

|Field|Type|Default|Description|
|-----|----|-------|-----------|
|`prompts`|`list[object]`|**required**|List of warmup requests (see below).|
|<code>task</code>|<code>string</code>|<code>"generate"</code>|Engine task to exercise: `"generate"` or `"embed"`.|
|`concurrency`|`int \| list[int]`|`1`|Concurrency level(s) to sweep.|
|`request_params`|`object`|`{}`|Extra sampling or pooling params merged into every request.|

Each item in `prompts` is an object with one of the following input fields:

|Field|Type|Default|Used for endpoint|
|-----|----|-------|-----------------|
|`prompt`|`string`|—|`/v1/completions`|
|`messages`|`list[dict]`|—|`/v1/chat/completions`|
|`input`|`string \| list[string]`|—|`/v1/embeddings`|
|`max_tokens`|`int`|`256`|Max tokens for this request.|

Only one of `prompt`, `messages`, or `input` should be provided per item.

## Examples

### Completions

```json
{
  "prompts": [
    {"prompt": "Tell me about AI", "max_tokens": 256},
    {"prompt": "Explain quantum computing", "max_tokens": 128}
  ],
  "concurrency": [1, 4],
  "request_params": {"temperature": 0.0}
}
```

### Chat completions

```json
{
  "prompts": [
    {
      "messages": [{"role": "user", "content": "Hello!"}],
      "max_tokens": 100
    },
    {
      "messages": [{"role": "user", "content": "What is the capital of France?"}],
      "max_tokens": 50
    }
  ],
  "concurrency": [1, 4]
}
```

### Embeddings

```json
{
  "task": "embed",
  "prompts": [
    {"input": "Hello world"},
    {"input": "Machine learning is fascinating"}
  ]
}
```

## How it works

```text
vllm serve
├── build_async_engine_client()    # Load model + basic warmup
├── warmup_engine()                # Run real prompts
├── build_app()                    # FastAPI setup
└── serve_http()                   # Start accepting traffic
```

When `--warmup-config` is supplied, `warmup_engine()` runs after engine
creation and before the HTTP server starts. It:

1. Loads the JSON configuration.
2. For each `concurrency` level, iterates through the `prompts`.
3. Calls the appropriate engine API (`generate()` or `encode()`).
4. Drains the output stream so the full forward pass executes.

## Multi-API-server deployments

In data-parallel or multi-API-server setups the warmup configuration is propagated
to each API server worker via the existing `args` object. Every worker runs the
warmup independently against the shared engine processes.

## Notes

- During warmup, the server is not yet bound, so `/health` is unreachable until
  warmup completes.
- The requests use internal IDs prefixed with `warmup_` and do not appear in
  external metrics or logs beyond standard engine logging.
- If you specify `messages`, the renderer converts them to engine prompts using
  the model's chat template, exactly as `/v1/chat/completions` would.
- For embedding models, set `task: "embed"` so that `encode()` is called instead
  of `generate()`.

## When to use

- **Production deployments** where the first few minutes of traffic must be at
  full throughput.
- **Benchmarking** where you need stable numbers from the very first request.
