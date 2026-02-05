<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the vLLM project -->

# Receipts harness (benchmark runner + telemetry)

## Goal

Performance discussions die when results cannot be forwarded and re-checked.
This harness turns an arbitrary benchmark command into a **receipt**:

- command + args
- wall time
- optional GPU telemetry (power/util/temp/memory) via NVML (preferred) or `nvidia-smi`
- stdout/stderr logs saved alongside the receipt
- SHA256 helper for forwarding integrity

This does **not** claim a specific vLLM bottleneck. It provides the measurement object.

## Run

From repo root:

```bash
python -m benchmarks.receipts.run_receipt --out receipts.json -- \
  python -m vllm.entrypoints.cli.main bench throughput --help
```

This writes:

- `receipts.json` (the receipt)
- `receipts.json.stdout.log`
- `receipts.json.stderr.log`

## Summarize

```bash
python -m benchmarks.receipts.summarize_receipt receipts.json
```

## Compare two receipts (before/after)

```bash
python -m benchmarks.receipts.compare_receipts before.json after.json
```

## Notes

- If GPU telemetry is unavailable, telemetry fields will be `null` and the receipt still records command + duration.
- This is intentionally lightweight: stdlib-only, NVML optional.
