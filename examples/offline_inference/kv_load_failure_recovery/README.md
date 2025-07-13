# KV Load Failure Recovery Test

This example builds upon the `disaggregated-prefill-v1` example in `examples/offline_inference`.

It demonstrates vLLM's ability to recover from KV load failures.
The goal is to verify that vLLM correctly identifies invalid KV blocks and reschedules the affected requests to ensure successful completion.

## Files

- `prefill_example.py` – performs the prefill stage and saves KV data (same as in `disaggregated-prefill-v1`).
- `decode_example.py` – performs the decode stage. It accepts a `--simulate-failure` flag to optionally simulate KV load failures using a custom connector.
- `rogue_shared_storage_connector.py` – defines `RogueSharedStorageConnector`, a subclass of `SharedStorageConnector`, that simulates missing or corrupted external KV blocks by intentionally failing to load blocks for the first decode request.
- `run.sh` – helper script that runs the prefill stage, then performs two decode stages: one normal run and one with simulated KV load failure.

## Notes

- This example reuses the structure and logic of `disaggregated-prefill-v1` with minimal changes.
- The test dynamically loads `RogueSharedStorageConnector` via `KVTransferConfig.kv_connector_module_path`, enabling controlled simulation of failure scenarios without modifying the original storage connector or using monkey-patching.
