# Local Runtime Follow-ups

These items are intentionally tracked so the new local vLLM UX can ship in stages without losing scope.

## Remaining feature goals

- Expand built-in model aliases beyond the current curated catalog.
- Add richer `vllm ls` output for alias catalog browsing, not just pulled models.
- Add shell UX niceties for `vllm run`, such as slash commands, history persistence, and multiline editing.
- Add better cache eviction for `vllm rm --purge-cache` so it can clean Hugging Face cache metadata and shared blobs safely.
- Add deeper service management commands such as restart, rename, and structured status output.
- Add optional Ollama-compatible local API endpoints if parity beyond CLI ergonomics becomes a priority.
- Add model creation and packaging flows comparable to Ollama's `create` / modelfile workflow.
- Add model copy and publish flows comparable to Ollama's `cp` / `push`.
- Add a dedicated `TensorRT-LLM export` or staging workflow once the inspect/eligibility path is proven useful.

## Platform follow-ups

- Validate and document the local installer flow on macOS beyond the current source-build path.
- Validate and document the local installer flow on Windows.
- Add platform-specific launcher and process-management behavior where Linux assumptions are currently baked in.
- Add richer Apple Silicon GPU capability reporting via MLX / Metal plugins.
- Add more backend-specific preflight heuristics once Apple GPU plugins expose more standardized capability metadata.

## Documentation follow-ups

- Expand command-specific docs for `vllm pull`, `vllm run`, `vllm ls`, `vllm ps`, `vllm stop`, `vllm logs`, and `vllm rm`.
- Add more backend compatibility notes as additional hardware plugins mature.
