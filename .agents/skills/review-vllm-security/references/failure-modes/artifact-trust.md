# Artifact trust and unsafe interpretation

- **Executable loading.** Carry `trust_remote_code` to the actual dynamic
  import or instantiation in model, tokenizer, processor, and config paths.
  Treat artifact-provided class names and plugin declarations as data until
  the trusted loader authorizes the action.
- **Revision pins.** Track `revision` and `code_revision` separately to each
  relevant file lookup and import. Include secondary config reads, weights,
  tokenizer and processor assets, dynamic code dependencies, and fallback
  resolution. Check failure behavior for missing pinned assets and
  cross-repository references.
- **Fallback trust.** Check fast and fallback loaders, remote and local
  snapshots, dependency helpers, and optional formats. Preserve the operator's
  decision through retry, error recovery, and model-specific wrappers.
- **Deserialization.** Use data-only formats and safe APIs for untrusted
  checkpoints, serialized embeddings, and IPC messages. Check indirect
  deserialization through libraries, shared-memory helpers, and alternative
  readers. Validate metadata before materializing tensors.
- **Data-driven dispatch.** Use declarative parsing and validated registries
  for model configuration and tool arguments. Keep authority to select
  executable loaders, callbacks, modules, or plugins with the trusted code.
- **Packed and local artifacts.** Resolve artifact-relative paths under the
  intended root. Review quantization metadata that governs native dimensions,
  offsets, or allocation. Use the project's private-cache trust assumptions
  when a new path consumes cached files.
- **Dependency guarantees.** Verify loader safeguards across the supported
  versions and paths of the dependency, including legacy and fallback readers.
