# vllm bench multimodal-processor

Benchmark the latency of the multimodal processor module across different configurations.

This benchmark measures per-stage timing for multimodal processing including:
- HuggingFace processor calls
- Multimodal item hashing
- Cache lookups and merges
- Prompt update application

## Prerequisites

The benchmark requires that the vLLM server is started with multimodal processor stats enabled:

```bash
VLLM_ENABLE_MM_PROCESSOR_STATS=1 vllm serve <model> [options]
```

The benchmark queries the `/metrics` endpoint (Prometheus format) to retrieve timing statistics. The multimodal processor timing is exposed as histograms with the metric name `vllm:mm_processor_time_seconds`, labeled by stage:
- `stage="hf_processor"` - HuggingFace processor calls
- `stage="hashing"` - Multimodal item hashing
- `stage="cache_lookup"` - Cache lookups and merges
- `stage="prompt_update"` - Prompt updates and placeholder finding

The benchmark parses these histograms to calculate percentiles and aggregate statistics.

## Usage

```bash
vllm bench multimodal-processor \
    --model <model> \
    --dataset-name hf \
    --dataset-path lmarena-ai/VisionArena-Chat \
    --data-parallel-size 1 \
    --num-instances 1 \
    --max-concurrency 100 \
    --request-rate 10.0 \
    --output-json results.json
```

## Configuration Options

### Data Parallel and Scaling

- `--data-parallel-size`: Number of data parallel workers (default: 1)
- `--num-instances`: Number of vLLM instances to run (default: 1)
- `--max-concurrency`: Maximum number of concurrent requests (default: unlimited)

### Request Generation

- `--request-rate`: Request rate in requests per second (default: inf)
- `--burstiness`: Burstiness factor for request generation (default: 1.0)
- `--num-warmups`: Number of warmup requests (default: 10)

### Dataset

The benchmark supports multimodal datasets via the `--dataset-name hf` option with appropriate `--dataset-path`:

- VisionArena: `lmarena-ai/VisionArena-Chat`
- MMVU: `yale-nlp/MMVU`
- Other HuggingFace multimodal datasets

See `vllm bench serve --help` for full dataset options.

## Output

The benchmark reports:

1. **End-to-end latency metrics**: Mean, median, std, and percentiles
2. **MM Processor timing by stage**:
   - `hf_processor_time`: Time in HuggingFace processor calls
   - `hashing_time`: Time computing multimodal item hashes
   - `cache_lookup_time`: Time in cache operations
   - `prompt_update_time`: Time applying prompt updates
   - `total_time`: Total multimodal processing time

Results can be saved to JSON using `--output-json`.

## JSON CLI Arguments

--8<-- "docs/cli/json_tip.inc.md"

## Arguments

--8<-- "docs/argparse/bench_multimodal_processor.inc.md"

## Limitations

The initial version supports:
- Data parallel configurations
- Multiple instances
- High concurrent request loads

Future versions will add:
- Custom DP + TP combinations
- Multi-node usage
- Mixed input types (text, text+image)

