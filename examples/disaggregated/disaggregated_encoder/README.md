# Disaggregated Encoder

These example scripts that demonstrate the disaggregated encoder (EPD) features of vLLM.

For a detailed explanation of the EPD features, please refer to the [Disaggregated Encoder Feature Documentation](../../../docs/features/disagg_encoder.md).

## Files

- `disagg_epd_proxy.py` - Proxy script that demonstrates the XeYpZd setup (X encode instances, Y prefill instances, Z decode instances). Currently stable for the 1e1p1d configuration.

- `disagg_1e1p1d_example.sh` - Sets up the 1e1p1d configuration, runs the VisionArena benchmark, and processes a single request with a local image.

- `disagg_1e1pd_example.sh` - Sets up the 1e1pd configuration, runs the VisionArena benchmark, and processes a single request with a local image.

- `size_epd.py` - Recommends a split from engine measurements or a calibrated workload model.

- `epd_capacity/measure.py` - Measures an E or PD worker's saturation rate with a representative replay.

## Size an E+PD deployment

The sizing example assumes one GPU per worker, a fixed serving configuration,
and replicas with comparable service rates. For each split it estimates
saturation capacity as the smaller of the two pools' aggregate rates.

### Measure rates from vLLM workers

Run two benchmark passes through the E+PD proxy on an isolated deployment.
For the E pass, use one encoder and enough PD workers to keep up. For the PD
pass, use one PD worker and enough encoders to keep it supplied. Pass all worker
metrics URLs for the active topology. Use the same model, workload distribution,
batching, connector, and cache policy in both passes. For a cold-cache replay,
disable prefix caching and use fresh image inputs and encoder caches.

Start the proxy with `--decode-retries 0` for both passes. The measurement tool
uses one image per user request. The default `random-mm` replay supports mixed
image sizes through `--random-mm-bucket-config`. To replay real requests, pass
`--dataset-name custom_image --dataset-path workload.jsonl` after `--`. The
tool checks each JSONL record for exactly one image before sending requests.
Before accepting a probe, it also checks that the completion count from each
pool equals the benchmark's completed user requests. Run any warmup separately
before measuring.

From `examples/disaggregated/disaggregated_encoder/`, run the E pass. Adjust
the port numbers to match your deployment and pass representative work to
`vllm bench serve` after `--`. This example uses synthetic images.

```bash
python -m epd_capacity.measure --stage encoder --output encoder-rates.json \
    --target-metrics http://localhost:8101/metrics \
    --peer-metrics http://localhost:8102/metrics \
    --peer-metrics http://localhost:8103/metrics -- \
    --model Qwen/Qwen2.5-VL-3B-Instruct --port 8000 --num-prompts 1000
```

Run the PD pass with `--stage pd`, its single PD worker as `--target-metrics`,
the encoder endpoints as `--peer-metrics`, and `--output pd-rates.json`.
The tool controls offered traffic by increasing concurrent requests. It infers
the plateau from successful completion deltas across two consecutive levels
where the target is backlogged and peer queues stay short. It reports an error
if the request count produces too short a sampling window or the tested topology
does not expose an isolated plateau. Use a larger representative replay or add
peer headroom as appropriate.

```bash
python size_epd.py --num-gpus 4 --measured-rates encoder-rates.json pd-rates.json
```

### Predict rates for a new request distribution

Provide a frozen calibration for the target deployment and a representative
sample of completed requests with one image each. The workload JSON contains
an `encoder` list of objects with `visual_tokens` and a `pd` object with
`input_lens`, `output_lens`, and `failed`. Keep PD input/output lengths paired
per request. For a replay using one fixed image shape, pass the detailed
`vllm bench serve` result with
`--benchmark-result PATH MERGED_TOKENS` in place of `--workload`.

```bash
python size_epd.py --num-gpus 4 --frozen-model frozen.json --workload workload.json
```

The predictor evaluates each encoder resource curve at the sampled merged-token
counts and uses the largest mean normalized occupancy to estimate the encoder
rate upper bound. The PD model uses paired lengths, the KV capacity, prefill chunk
size, and a calibrated step time. It caps the KV estimates by the scheduler rate
for an uncached prompt and ordinary decoding. The scheduled token demand is the
prompt length plus the output length minus one, since prefill produces the first
output token. The lower capped estimate drives the sizing calculation.

Set `pd.max_num_scheduled_tokens` to the deployment's effective token budget per
step. When this field is absent, the predictor uses `pd.chunk_tokens` as the
budget, as in the archived calibration. The encoder fit and PD parameters are
reused across request mixes within the calibrated encoder range on the same
deployment. The estimates refer to saturation and depend on replica balance and
shared resource contention.

To fit the encoder from collected timing summaries, run
`python -m epd_capacity.model --profiles profiles.json --output frozen.json`.
The input includes `measurements` with per-component serial and saturated timing
anchors and their effective parallelism, as well as calibrated `pd` settings.
Use at least three distinct token-count anchors. With exactly three, the tool
warns that leave-one-anchor-out validation of the quadratic components is
unavailable.
Before writing the model, the command checks that each fitted duration is finite
and nonnegative throughout the calibrated visual-token range.
For a worked model and held-out workload, use `frozen_predictor/frozen.json`
from the [reproducer archive][epd-reproducer] and
`results-783695/shift-workload.json` from the [results archive][epd-results].

You can also supply `--encoder-rps` and `--pd-rps` from an external estimator.
For equal predicted capacities, the example favors fewer encoder workers.

[epd-reproducer]: https://raw.githubusercontent.com/Kelvin-Al/vllm/271a17af0967c5bd5270a6fb5e893675b74478d4/epd-sizing-67204-reproducer.tar.gz
[epd-results]: https://raw.githubusercontent.com/Kelvin-Al/vllm/271a17af0967c5bd5270a6fb5e893675b74478d4/epd-sizing-783695-results.tar.gz

### Custom Configuration

```bash
# Use specific GPUs
GPU_E=0 GPU_PD=1 GPU_P=1 GPU_D=2 bash disagg_1e1p1d_example.sh

# Use specific ports
ENDPOINT_PORT=10001 bash disagg_1e1p1d_example.sh

# Use specific model
MODEL="Qwen/Qwen2.5-VL-3B-Instruct" bash disagg_1e1p1d_example.sh

# Use specific storage path
EC_SHARED_STORAGE_PATH="/tmp/my_ec_cache" bash disagg_1e1p1d_example.sh

# Run on XPU; scripts switch from CUDA_VISIBLE_DEVICES to ZE_AFFINITY_MASK
DEVICE_PLATFORM=xpu GPU_E=0 GPU_PD=1 bash disagg_1e1pd_example.sh
```

`DEVICE_PLATFORM` defaults to `cuda`. Set `DEVICE_PLATFORM=xpu` when running these examples on Intel GPUs so the scripts use `ZE_AFFINITY_MASK` instead of `CUDA_VISIBLE_DEVICES` for device selection.

## Encoder Instances

Encoder engines should be launched with the following flags:

- `--enforce-eager` **(required)** – The current EPD implementation is only compatible with encoder instances running in this mode.

- `--disable-prefix-caching` **(required)** – Encoder instances do not consume KV cache; prefix caching is disabled to avoid conflicts with other features.

- `--max-num-batched-tokens=<large value>` **(default: 2048)** – This flag controls the token scheduling budget per decoding step and is irrelevant to encoder-only instances. **Set it to a very high value (effectively unlimited) to bypass scheduler limitations.** The actual token budget is managed by the encoder cache manager.

- `--mm-encoder-only` **(Optional)** - If possible, skips the language model during initialization to reduce device memory usage.

## Local media inputs

To support local image inputs (from your ```MEDIA_PATH``` directory), add the following flag to the encoder instance:

```bash
--allowed-local-media-path $MEDIA_PATH
```

The vllm instances and `disagg_encoder_proxy` supports local URIs with ```{"url": "file://'"$MEDIA_PATH_FILENAME"'}``` as multimodal inputs. Each URI is passed unchanged from the `disagg_encoder_proxy` to the encoder instance so that the encoder can load the media locally.

## EC connector and KV transfer

`ECExampleConnector` stores encoder cache on local disk. For RDMA/TCP transfer
without a shared filesystem, use `ECMooncakeConnector` (1E + 1PD only through
this proxy). See
[ECMooncakeConnector Usage Guide](../../../docs/features/mooncake_ec_connector_usage.md).

To enable filesystem encoder disaggregation, add:

```bash
# Add to encoder instance: 
--ec-transfer-config '{
    "ec_connector": "ECExampleConnector",
    "ec_role": "ec_producer",
    "ec_connector_extra_config": {
        "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
    }
}' 

# Add to prefill/prefill+decode instance: 
--ec-transfer-config '{
    "ec_connector": "ECExampleConnector",
    "ec_role": "ec_consumer",
    "ec_connector_extra_config": {
        "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
    }
}' 
```

`$EC_SHARED_STORAGE_PATH` is the path where the EC connector temporarily stores the cache.

If you enable prefill instance (`--prefill-servers-urls` not disabled), you will need --kv-transfer-config to facilitate the PD disaggregation. Currently, we use the `NixlConnector` for this purpose. Refer to `tests/v1/kv_connector/nixl_integration` for more example codes on PD disaggregation with Nixl.

```bash
# Add to prefill instance:    
--kv-transfer-config '{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_producer"
}' 

# Add to decode instance:
--kv-transfer-config '{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_consumer"
}' 
```

## Proxy Instance Flags (`disagg_epd_proxy.py`)

| Flag | Description |
| ---- | ----------- |
| `--encode-servers-urls` | Comma-separated list of encoder endpoints. Every multimodal item extracted from the request is fanned out to one of these URLs in a round-robin fashion. |
| `--prefill-servers-urls` | Comma-separated list of prefill endpoints. Set to `disable`, `none`, or `""` to skip the dedicated prefill phase and run E+PD (encoder + combined prefill/decode). |
| `--decode-servers-urls` | Comma-separated list of decode endpoints. Non-stream and stream paths both round-robin over this list. |
| `--ec-consumer-zmq-addrs` | Mooncake EC consumer control addresses, aligned with `--decode-servers-urls`. Required for `ECMooncakeConnector`. Supported only in E+PD mode (`--prefill-servers-urls disable`). |
| `--ec-consumer-dp-size` | Data-parallel replicas per EC consumer (default 1). |
| `--host`, `--port` | Bind address for the proxy itself (defaults: `0.0.0.0:8000`). |

When using `--enable-per-video-kwargs-slicing` with audio extracted from video,
put per-video `fps` or `num_frames` lists inside
`mm_processor_kwargs.videos_kwargs` so the audio cache keys match across stages.

### Dynamic registration

Alternatively, let the external launcher register ready instances over HTTP.
No vLLM configuration changes or worker registration threads are required.
Set `ADMIN_API_KEY` on the proxy and supply it as `X-API-Key` for registration
and removal. These are trusted control-plane APIs; do not expose them publicly.

```bash
export ADMIN_API_KEY="your-admin-key"
python disagg_epd_proxy.py --port 8000 --dynamic-registration
```

Omit the static server URL flags and register every stage. The roles determine
the topology: `encode` + `prefill_decode` gives E+PD; `encode` + `prefill` +
`decode` gives E+P+D. Standalone `decode` always requires an available `prefill`,
even when D registers first or all P instances go offline; otherwise requests
return `503`. The proxy rejects mixing combined PD with standalone P/D,
including unhealthy instances still in the registry. Explicitly remove the
old topology's P/D or PD registrations before switching topologies.

After each instance is ready, the launcher registers its reachable HTTP URL:

```bash
curl --fail-with-body http://proxy-host:8000/instances \
    -H "X-API-Key: $ADMIN_API_KEY" -H 'Content-Type: application/json' \
    -d '{"role":"encode","url":"http://e-host:8001"}'
curl --fail-with-body http://proxy-host:8000/instances \
    -H "X-API-Key: $ADMIN_API_KEY" -H 'Content-Type: application/json' \
    -d '{"role":"prefill_decode","url":"http://pd-host:8002"}'
```

For E+P+D, register P with `role: "prefill"` and D with `role: "decode"`.
Example and NIXL EC connectors need no additional registration fields.
For Mooncake, register `ec_zmq_addrs` on the **EC consumer** (`prefill_decode`
or `prefill`), using its configured `ec_ip` and `ec_port`. Keep the fixed-port
layout: supply the TP-rank-0 address for each DP replica, in DP-rank order;
replica `r` uses `ec_port + r * tensor_parallel_size`. The connector discovers
the remaining TP ranks itself. For example, DP=2, TP=2, `ec_port=19019`:

```json
{
  "role": "prefill_decode",
  "url": "http://pd-host:8002",
  "dp_size": 2,
  "ec_zmq_addrs": ["tcp://pd-host:19019", "tcp://pd-host:19021"]
}
```

The proxy selects one consumer replica and uses it for both the encoder push
and the consumer HTTP request. Standalone D does not need EC control addresses.
Port allocation and avoiding collisions remain the launcher's responsibility.

Inspect or remove instances without restarting the proxy:

```bash
curl http://proxy-host:8000/instances
curl --fail-with-body -X DELETE \
    'http://proxy-host:8000/instances?url=http://e-host:8001' \
    -H "X-API-Key: $ADMIN_API_KEY"
```

Registration is idempotent. The proxy probes registered instances every
`--probe-interval` seconds (default 5), with `--probe-timeout` seconds per probe
(default 2). After `--fail-threshold` consecutive failures (default 3), it stops
sending new requests to that instance. Healthy instances rejoin automatically;
unreachable ones are forgotten after `--evicted-ttl` seconds (default 900;
`0` retains them indefinitely). Removal stops new routing; already routed
requests retain their selected endpoints, so drain requests before stopping
the instance. The launcher must re-register instances after a proxy restart.

The example launch scripts also support this flow: export `ADMIN_API_KEY` and
set `DYNAMIC_REGISTRATION=1` when running `disagg_1e1pd_example.sh` or
`disagg_1e1p1d_example.sh`. Their default remains static routing.

### Static configuration

The proxy batches images from the same user request assigned to the same encoder.
Set the proxy environment variable `ENCODER_MAX_BATCH_SIZE` to limit the number
of images per encoder subrequest. It defaults to `0` (unlimited); `1` sends each
image separately. Audio and video remain separate subrequests.

For encoders with a smaller `--limit-mm-per-prompt` image limit than P/PD, set
`ENCODER_MAX_BATCH_SIZE` no higher than the smallest encoder image limit.
For example, for encoders configured with `--limit-mm-per-prompt '{"image": 2}'`:

```bash
ENCODER_MAX_BATCH_SIZE=2 python disagg_epd_proxy.py \
    --encode-servers-urls "http://e1:8001,http://e2:8002" \
    --prefill-servers-urls disable \
    --decode-servers-urls "http://pd1:8003"
```

Example usage:
For E + PD setup:

```bash
$ python disagg_encoder_proxy.py \
      --encode-servers-urls "http://e1:8001,http://e2:8002" \
      --prefill-servers-urls "disable" \
      --decode-servers-urls "http://pd1:8003,http://pd2:8004"
```

For E + P + D setup:

```bash
$ python disagg_encoder_proxy.py \
      --encode-servers-urls "http://e1:8001,http://e2:8001" \
      --prefill-servers-urls "http://p1:8003,http://p2:8004" \
      --decode-servers-urls "http://d1:8005,http://d2:8006"
```
