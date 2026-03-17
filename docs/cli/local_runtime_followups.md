# Local Runtime Follow-ups

These items are intentionally tracked so the new local vLLM UX can ship in stages without losing scope.

## Remaining feature goals

- Expand built-in model aliases beyond the initial curated set.
- Add richer `vllm ls` output for alias catalog browsing, not just pulled models.
- Add shell UX niceties for `vllm run`, such as slash commands, history persistence, and multiline editing.
- Add better cache eviction for `vllm rm --purge-cache` so it can clean Hugging Face cache metadata and shared blobs safely.
- Add deeper service management commands such as restart, rename, and structured status output.
- Add optional Ollama-compatible local API endpoints if parity beyond CLI ergonomics becomes a priority.
- Add model creation and packaging flows comparable to Ollama's `create` / modelfile workflow.
- Add model copy and publish flows comparable to Ollama's `cp` / `push`.

## Platform follow-ups

- Validate and document the local installer flow on macOS.
- Validate and document the local installer flow on Windows.
- Add platform-specific launcher and process-management behavior where Linux assumptions are currently baked in.

## Documentation follow-ups

- Add dedicated docs pages for `vllm pull`, `vllm run`, `vllm ls`, `vllm ps`, `vllm stop`, `vllm logs`, and `vllm rm`.
- Add a local-runtime quickstart separate from the broader developer/source install docs.
