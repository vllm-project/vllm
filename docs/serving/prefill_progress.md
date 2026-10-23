# Request-scoped prefill progress

The experimental Rust gRPC frontend can report remaining prefill work to a
router. A client first checks
`Control.GetServerInfo.supports_prefill_progress`, then opts in with
`GenerateRequest.include_prefill_progress` on `GenerateStream`. Unary
`Generate` rejects this option.

Progress responses contain `request_id`, a monotonically increasing `sequence`,
and token-work accounting. They are snapshots: a newer response replaces an
older one. They do not measure FLOPs, GPU utilization, or total worker load.
Consumers must ignore out-of-order snapshots and release request reservations
when generation completes, errors, or is cancelled, even if no final progress
response reaches them.

## Reporting cost and batching

Ordinary updates require both at least 100ms since the previous report and at
least 1% more completed uncached prompt work, rounded up to one token. The
scheduler skips intermediate states instead of queuing them. A continuous
prefill with stable cache accounting therefore generates no more than 100
ordinary updates, regardless of its chunk count. First reports, prefill
completion, phase/cache-accounting changes, and increases in remaining work
bypass the thresholds. In particular, recomputation must not leave a router
underestimating the remaining work.

All requests processed in one scheduler step share the same sampling timestamp.
Their outputs use the existing `EngineCoreOutputs` batch on the engine-to-frontend
transport. The gRPC API still uses one stream per request; it does not combine
unrelated requests into a shared response. Progress bypasses the generated-token
`stream_interval`, since a request can spend many steps in prefill without
generating a token. No per-request timer or per-snapshot informational log is
created.

These limits apply per request. Aggregate traffic still grows with the number
of active prefills. A worker-wide shared gRPC progress stream would require a
separate protocol with request lifecycle and reconnection handling; the current
per-request API does not provide that transport-level batching.

The thresholds trade fresher load estimates for fewer messages. Validate their
effect under the deployment's prompt lengths, chunk sizes, and concurrency;
functional correctness alone does not establish negligible serving overhead.

See the [protocol documentation](https://github.com/vllm-project/vllm/blob/main/rust/proto/README.md) for schema and
client-generation details.
