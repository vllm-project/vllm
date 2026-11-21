# Sensitive output and error responses

- **Safe client errors.** Use structured error codes and safe fields. Keep
  internal paths, usernames, native object representations, stack traces, raw
  request bodies, and credentials out of client-visible errors. Compare HTTP,
  gRPC, streaming, and model-specific wrappers.
- **Error amplification.** Limit how many validation errors are collected,
  how much each field echoes, and the total response size. Account for
  validation, redaction, and serialization costs on rejected requests.
- **Logs and metrics.** Keep secret and high-volume input fields out of
  routine diagnostics. Bound log records and use a fixed vocabulary for
  metric labels derived from methods, paths, or error classes.
- **Model-visible context.** Render only intended tool and prompt fields.
  Keep service credentials, connection headers, and private execution state
  in the corresponding transport or server context.
- **Output provenance.** Verify that returned tokens, logprobs, media, and
  intermediate representations come from this request and initialized storage.
  Derive usage and cached-token counts from the trusted accounting source.
- **Signals from shared resources.** Check timing, cache-hit indicators, and
  operational metadata when changing what callers can learn about shared
  state. Tie guarantees to the documented cache and deployment assumptions.

## Published advisory examples

Check diagnostics for both
disclosure and unbounded output.

- [Validation error disclosure](https://github.com/vllm-project/vllm/security/advisories/GHSA-hwrm-c4cx-rf4j)
- [Error representation across frontends](https://github.com/vllm-project/vllm/security/advisories/GHSA-hgg8-fqqc-vfmw)
- [Metric label cardinality](https://github.com/vllm-project/vllm/security/advisories/GHSA-5fj9-pfhr-6j48)
