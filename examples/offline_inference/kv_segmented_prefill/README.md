# Segmented Prefill test

This example builds upon the `disaggregated-prefill-v1` example in `examples/offline_inference`.

It demonstrates vLLM's ability to perform segmented prefill, in case the KV Connector reports "gaps" in the external cache.
The goal is to verify that vLLM correctly recalculates the tokens missing in cache (the gaps). Correctness is tested by comparing the generation output using the full cache and the output using a cache with gaps.

## Files

- `segmented_prefill_example_connector.py` – defines `SegmentedPrefillExampleConnector`, a subclass of `ExampleConnector`, that simulates missing external KV blocks by creating gaps in the cache - intentionally failing to load blocks of tokens in the middle of each  prompt.
- `run.sh` – orchestrates the test: runs a prefill stage which generates the external KV-Cache, then two decode stages:
    1. Normal decode (baseline).
    2. Decode with simulated gaps in the KV cache.

    It then compares the two outputs to verify correctness.

<!-- TODO: add info about prefill_example.py / decode_example.py or their replacements -->

## How It Works

- The test dynamically loads `SegmentedPrefillExampleConnector` via `KVTransferConfig.kv_connector_module_path`, enabling controlled simulation of cache gaps without modifying the original connector.
- The decode stage that simulates gaps is expected to trigger Segmented Prefill in vLLM, resulting in the same output as the baseline decode.
- In case the outputs differ, the script prints a unified diff of mismatch and exits with error.

## Usage

```bash
./run.sh
```
