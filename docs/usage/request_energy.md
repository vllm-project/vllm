# Request energy tracing

This research fork can export request lifecycle, scheduled demand and GPU
execution intervals for the standalone request-energy processor. The default is
off. vLLM does not poll NVML/RAPL or calculate joules in its scheduler.

Install the matching llm-request-energy **0.2.0** SDK wheel in the engine
environment, for example using `uv pip install /path/to/the/matching.whl`.
Build/obtain this development wheel from the independent attribution project;
it is not assumed to be published on PyPI. Tracing disabled needs no SDK.
Set REQUEST_ENERGY_TRACE_DIR to an existing writable trace location and
REQUEST_ENERGY_INSTANCE_ID to a unique server-launch ID shared by TP workers.
The SDK creates a separate append-only file per scheduler/worker producer.
Normal single-host TP needs TP size + 1 required engine producers.

Clients can provide vllm_xargs.request_energy_client_id in a request body.
The trace retains that explicit ID alongside the server request ID. Without
the field, the client ID is null; no timestamp-based match is inferred.

The scheduler snapshots demands before advancing optimistic token counters.
Queued submissions hold lifecycle progress until update_from_output acknowledges
the result. Worker CUDA events bracket forward and sampling execution separately.
The shared worker wrapper covers the V1 and V2 runner dispatches. A resolver
queries event completion rather than synchronizing the GPU per step.
Forward and sampling observations have distinct execution IDs and the same
explicit `logical_step_id`. Producer metadata identifies the canonical format
and SDK version. Bounded writing and integrity failures remain visible to the
central processor.

Computed token positions include executed repeated work. Returned token counts
and GPU execution durations are different quantities. FLOP and byte fields are
null until a supported estimator supplies them; no hardware traffic counters
are inferred from KV occupancy.

Speculative, multimodal/SSM, offloaded, distributed and pipeline-parallel
configurations are outside the trusted initial profile. Their ownership is
marked unsupported. Dense text, chunked prefill, prefix reuse, cancellation,
preemption, ordinary scheduling overlap and TP hooks require the native model
and output-equivalence checks before this development patch is released.

The [validation receipt](request_energy_validation.md) records the executed native
smoke cases and their limits. Full serving-profile coverage, physical accuracy and overhead
remain separate acceptance gates; a successful replay is not a physical
per-request energy reference.
