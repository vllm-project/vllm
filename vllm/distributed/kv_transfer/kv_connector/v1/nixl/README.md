# NIXL Connector

Source location: `vllm/distributed/kv_transfer/kv_connector/v1/nixl`.

This directory implements `NixlConnector`, the main high-performance
Prefill/Decode disaggregation connector. It handles the v1 scheduler/worker
split, NIXL agent metadata, heterogeneous TP mapping, HMA/hybrid SSM support,
transfer failure reporting, lease/heartbeat behavior, and connector metrics.

## Key source files

- `connector.py`: facade class `NixlConnector`; delegates scheduler role to
  `NixlConnectorScheduler` and worker role to `NixlConnectorWorker`.
- `scheduler.py`: request lifecycle on the scheduler side: new requests, remote
  hit accounting, allocation updates, connector metadata, finished requests,
  lease and timeout state.
- `worker.py`: NIXL worker implementation: memory registration, handshake,
  remote-agent tracking, transfer reads/writes, HMA handling, prefix-cache
  post-processing, heartbeats, and completion reporting.
- `metadata.py`: wire and in-process metadata structures, including
  `NixlAgentMetadata`, `NixlHandshakePayload`, `RemoteMeta`, `ReqMeta`, and
  `NixlConnectorMetadata`.
- `tp_mapping.py`: local-to-remote TP mapping for heterogeneous deployments.
  This is the first file to read for `P_TP != D_TP` work.
- `stats.py`: NIXL transfer stats and Prometheus metrics.
- `utils.py`: small utilities local to the NIXL package.

## NIXL contribution reading path

1. `connector.py`: understand which calls route to scheduler vs worker.
2. `scheduler.py`: follow `get_num_new_matched_tokens()`,
   `update_state_after_alloc()`, `build_connector_meta()`, and
   `request_finished()`.
3. `metadata.py`: map scheduler metadata to worker transfer inputs.
4. `worker.py`: read `add_remote_agent()`, registration helpers, load paths,
   prefix-cache post-processing, and completion accounting.
5. `tp_mapping.py`: read before touching heterogeneous TP, replicated KV
   heads, MLA, or hybrid SSM/GDN/KDA.
6. `stats.py`: extend when adding transfer modes, new failure classes, or
   latency counters.

## Common change areas

- Heterogeneous TP: `tp_mapping.py`, `worker.py`, and shared topology helpers in
  `kv_connector/utils.py`.
- Hybrid SSM/GDN/KDA: `worker.py` plus
  `v1/ssm_conv_transfer_utils.py`.
- Prefix caching in P/D mode: `scheduler.py`, `worker.py`, and HMA coordinator
  code under `vllm/v1/core/`.
- Connector observability: `stats.py`, `metrics.py`, and failure logging in
  `worker.py`.

## Useful tests and docs

- Unit tests: `tests/v1/kv_connector/unit/test_nixl_connector.py`.
- HMA/hybrid tests: `tests/v1/kv_connector/unit/test_nixl_connector_hma.py`.
- TP mapping tests: `tests/v1/kv_connector/unit/test_tp_mapping.py`.
- Integration examples: `tests/v1/kv_connector/nixl_integration/`.
- User docs: `docs/features/nixl_connector_usage.md` and
  `docs/features/nixl_connector_compatibility.md`.
