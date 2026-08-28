# Disaggregated Encoder

These example scripts that demonstrate the disaggregated encoder (EPD) features of vLLM.

For a detailed explanation of the EPD features, please refer to the [Disaggregated Encoder Feature Documentation](../../../docs/features/disagg_encoder.md).

## Files

- `disagg_1e1p1d_example.sh` - Sets up the 1e1p1d configuration, runs the VisionArena benchmark, and processes a single request with a local image.

- `disagg_1e1pd_example.sh` - Sets up the 1e1pd configuration, runs the VisionArena benchmark, and processes a single request with a local image.

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

- `--no-enable-prefix-caching` **(required)** – Encoder instances do not consume KV cache; prefix caching is disabled to avoid conflicts with other features.

- `--max-num-batched-tokens=<large value>` **(default: 2048)** – This flag controls the token scheduling budget per decoding step and is irrelevant to encoder-only instances. **Set it to a very high value (effectively unlimited) to bypass scheduler limitations.** The actual token budget is managed by the encoder cache manager.

- `--mm-encoder-only` **(Optional)** - If possible, skips the language model during initialization to reduce device memory usage.

## Local media inputs

To support local image inputs (from your ```MEDIA_PATH``` directory), add the following flag to the encoder instance:

```bash
--allowed-local-media-path $MEDIA_PATH
```

The vllm instances and the proxy support local URIs with ```{"url": "file://'"$MEDIA_PATH_FILENAME"'}``` as multimodal inputs. Each URI is passed unchanged from the proxy to the encoder instance so that the encoder can load the media locally.

## EC connector and KV transfer

The `ECExampleConnector` is used to store the encoder cache on local disk and facilitate transfer. To enable the encoder disaggregation feature, add the following configuration:

```bash
# Add to encoder instance: 
--ec-transfer-config '{
    "ec_connector": "ECExampleConnector",
    "ec_role": "ec_producer",
    "ec_connector_extra_config": {
        "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'",
        "proxy_url": "http://localhost:'"$PROXY_PORT"'"
    }
}' 

# Add to prefill/prefill+decode instance: 
--ec-transfer-config '{
    "ec_connector": "ECExampleConnector",
    "ec_role": "ec_consumer",
    "ec_connector_extra_config": {
        "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'",
        "proxy_url": "http://localhost:'"$PROXY_PORT"'"
    }
}' 
```

`$EC_SHARED_STORAGE_PATH` is the path where the EC connector temporarily stores the cache.

If you run a separate prefill instance, you will need --kv-transfer-config to facilitate the PD disaggregation. Currently, we use the `NixlConnector` for this purpose. Refer to `tests/v1/kv_connector/nixl_integration` for more example codes on PD disaggregation with Nixl.

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

## Proxy

Start the proxy first, with no topology:

```bash
vllm disagg-proxy --port 8000
```

It comes up with an empty roster and answers `503` until instances register.
Each instance announces itself once it is serving, by naming the proxy in its
EC transfer config:

```bash
"ec_connector_extra_config": {"proxy_url": "http://proxy-host:8000"}
```

A decode instance in an E+P+D deployment moves no embeddings and so has no EC
role, but the proxy still has to know where to forward. Give it an EC config
carrying nothing but the proxy URL:

```bash
--ec-transfer-config '{
    "ec_connector_extra_config": {
        "proxy_url": "http://proxy-host:8000"
    }
}'
```

The role is inferred, not declared: an encoder-only instance registers as
`encode`, a KV producer as `prefill`, and anything else as `decode` -- which
covers a fused prefill+decode instance.

Adding capacity means starting another instance; removing it means stopping
one. Nothing else has to be restarted or reconfigured.

### Liveness

The proxy probes every registered instance and stops routing to one that fails
`--fail-threshold` probes in a row. A single missed probe is not enough: a busy
encoder can miss one under load. An evicted instance keeps being probed and
rejoins on its own once it answers again, so recovering from a blip does not
mean restarting anything. After `--evicted-ttl` seconds without a response it
is forgotten.

Instances also re-announce periodically, so a proxy that restarts refills its
roster by itself.

| Flag | Description |
| ---- | ----------- |
| `--host`, `--port` | Bind address for the proxy (defaults: `0.0.0.0:8000`). |
| `--probe-interval` | Seconds between health probes. `0` disables probing. |
| `--probe-timeout` | Per-probe timeout. |
| `--fail-threshold` | Consecutive failed probes before an instance stops being routed to. |
| `--evicted-ttl` | Seconds to keep probing an unreachable instance before forgetting it. `0` probes forever. |

### Inspecting the roster

```bash
curl http://proxy-host:8000/instances
```

```json
{
  "encode":  {"live": ["http://e1:8001", "http://e2:8001"], "evicted": []},
  "prefill": {"live": [], "evicted": []},
  "decode":  {"live": ["http://pd1:8003"], "evicted": []}
}
```
